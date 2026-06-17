"""Point-in-time membership, survivorship & delisting (stage 7).

Sources (PREREG §5 / config data.universe): Refinitiv TR.IndexConstituentRIC for
>=2016; Datastream monthly lists LS&PCOMP MMYY for 2005-2016; stitched with a 2016
OVERLAP-YEAR cross-validation table (both vendors cover 2016 — their agreement is
the splice's quality certificate). Survivorship bias of ~0.9-1.4%/yr (Elton, Gruber
& Blake 1996; Brown, Goetzmann & Ross 1995) is the reason PIT membership exists.

Delisting: terminal returns are taken from the vendor when present; when missing,
Shumway (1997 JF) / Shumway & Warther (1999 JF) corrections apply (-30% NYSE/AMEX,
-55% Nasdaq; config data.series.delisting_corrections) — EVERY application is
appended to an audit log returned to the caller (logged corrections into derived
layers; raw untouched, R4). Leavers are LIQUIDATED at last traded price net of cost
in the panel stage — never silently dropped.

Leakage posture: `top30_at` uses the latest membership month-end STRICTLY BEFORE the
window start and market caps from the last session STRICTLY BEFORE the window start —
selection at t uses only information < t (R3).
"""
from __future__ import annotations

import pandas as pd

from ..config import get


def normalize_membership(long_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """-> tidy frame {month_end: Timestamp, ric: str, source: str}, deduplicated."""
    df = long_df.rename(columns=str.lower).copy()
    if not {"month_end", "ric"} <= set(df.columns):
        raise ValueError("membership frame needs columns month_end, ric")
    df["month_end"] = pd.to_datetime(df["month_end"]) + pd.offsets.MonthEnd(0)
    df["ric"] = df["ric"].astype(str).str.strip()
    df["source"] = source
    return df[["month_end", "ric", "source"]].drop_duplicates().sort_values(
        ["month_end", "ric"]).reset_index(drop=True)


def stitch_membership(pre2016: pd.DataFrame | None, post2016: pd.DataFrame,
                      splice_year: int = 2016) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splice Datastream (<splice_year) onto Refinitiv (>=splice_year) and build the
    overlap cross-validation table for every month BOTH sources cover.

    Returns (pit_membership, overlap_table). With pre2016 absent (entitlement gap),
    the splice degrades to the post-2016 span alone — the gap is the caller's
    quarantine item, never silently back-filled (R4)."""
    overlap_rows = []
    if pre2016 is not None and not pre2016.empty:
        both = sorted(set(pre2016["month_end"]) & set(post2016["month_end"]))
        for m in both:
            a = set(pre2016.loc[pre2016["month_end"] == m, "ric"])
            b = set(post2016.loc[post2016["month_end"] == m, "ric"])
            union, inter = a | b, a & b
            overlap_rows.append({
                "month_end": m, "n_pre_source": len(a), "n_post_source": len(b),
                "jaccard": round(len(inter) / len(union), 4) if union else 1.0,
                "only_pre": sorted(a - b)[:10], "only_post": sorted(b - a)[:10],
            })
        pit = pd.concat([
            pre2016[pre2016["month_end"] < pd.Timestamp(f"{splice_year}-01-01")],
            post2016[post2016["month_end"] >= pd.Timestamp(f"{splice_year}-01-01")],
        ], ignore_index=True)
    else:
        pit = post2016.copy()
    pit = pit.sort_values(["month_end", "ric"]).reset_index(drop=True)
    return pit, pd.DataFrame(overlap_rows)


def joiners_leavers(pit: pd.DataFrame) -> pd.DataFrame:
    """Month-on-month membership events -> the audit log {month_end, ric, event}."""
    months = sorted(pit["month_end"].unique())
    rows = []
    for prev, cur in zip(months, months[1:]):
        a = set(pit.loc[pit["month_end"] == prev, "ric"])
        b = set(pit.loc[pit["month_end"] == cur, "ric"])
        rows += [{"month_end": cur, "ric": r, "event": "joiner"} for r in sorted(b - a)]
        rows += [{"month_end": cur, "ric": r, "event": "leaver"} for r in sorted(a - b)]
    return pd.DataFrame(rows)


def _last_valid_label(out: pd.DataFrame, name: str, date: pd.Timestamp):
    """The index label at which to BOOK the delisting return for column `name`.

    A dead RIC's nominal delist `date` row is, by construction, POST the last traded
    session for that name — that whole column is NaN there (and `date` may not even be
    a session in the XNYS index). Writing onto it with ``out.loc[date, name] = value``
    therefore (a) KeyErrors when `date` is off-grid, or (b) plants the crash return on a
    phantom post-death row that the env's ``liquidate_to_cash`` would then zero-fill — the
    return never reaches the tail. We instead book it onto the LAST VALID (non-NaN)
    observation that is <= `date`, which is the real terminal session.

    Returns the label, or ``None`` when the name carries NO valid observation in the
    panel window (nothing to compound into — the caller logs a skip, never raises)."""
    col = out[name]
    valid = col.index[col.notna() & (col.index <= date)]
    if len(valid):
        return valid.max()
    # No observation at/before the delist date: fall back to the column's last valid
    # session anywhere in the window (the name died entirely before `date`'s grid slot).
    any_valid = col.index[col.notna()]
    return any_valid.max() if len(any_valid) else None


def classify_delist_reason(reason: str | None) -> str:
    """Bucket a vendor delisting-reason string into ``mna`` | ``performance`` | ``unknown``
    via the config keyword lists (``data.series.delisting_reason_classes``). Substring,
    case-insensitive. ``None``/empty -> ``unknown`` (the conservative default: surcharge,
    never guess M&A). Pure; used by :func:`apply_shumway_corrections` to gate the surcharge
    ONCE the reason is re-pulled — it is a no-op on the
    current vault, which carries no reason (verified)."""
    if reason is None or not str(reason).strip() or str(reason).strip().lower() == "nan":
        return "unknown"
    text = str(reason).lower()
    classes = get("data.series.delisting_reason_classes", {})
    for kw in classes.get("mna_keep_terminal", []):
        if str(kw).lower() in text:
            return "mna"
    for kw in classes.get("performance_surcharge", []):
        if str(kw).lower() in text:
            return "performance"
    return "unknown"


def apply_shumway_corrections(
    returns: pd.DataFrame,
    delisted: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compound the Shumway-STYLE delisting return into the terminal session where the
    vendor's terminal return is missing. `delisted` maps column ->
    {date, exchange ('nyse_amex'|'nasdaq'), vendor_terminal_return (float|None)}.

    Shumway-STYLE TRANSPLANT (R7): Refinitiv carries no CRSP ``DLSTCD``, so the
    performance-delisting return is the fixed −30% NYSE/AMEX / −55% Nasdaq surcharge
    (config data.series.delisting_corrections; Shumway 1997 JF, Shumway & Warther 1999
    JF). The VENDOR terminal return is PREFERRED where present — those names already
    carry their last traded return in the frame, so the fixed surcharge is reserved for
    names with no vendor terminal observation (the identifiable performance
    terminations). The headline reports a {0% (= ``liquidate_to_cash``), −30%, −55%,
    −100%} sensitivity band around this choice (the band is the caller's to assemble).

    Booking rule (the FIX): the correction is compounded MULTIPLICATIVELY,
    ``(1 + r_terminal)(1 + dl) − 1``, onto the LAST VALID session for the name (see
    :func:`_last_valid_label`) — NEVER additively (additive can breach −100%;
    OpenSourceAP #49) and NEVER onto the all-NaN nominal-delist row (the old
    ``out.loc[date, name] = value`` KeyErrored off-grid / planted a phantom row the
    env then zero-filled).

    Returns (corrected_copy, audit_log). The input frame is NEVER mutated; every
    correction (and every skip — vendor-covered, or no in-window observation) is one
    audit row — the dissertation reports this log verbatim (R4: explicit, logged,
    cited)."""
    corr = get("data.series.delisting_corrections")
    out = returns.copy()
    log = []
    for name, info in delisted.items():
        if name not in out.columns:
            continue
        date = pd.Timestamp(info["date"])
        vendor_ret = info.get("vendor_terminal_return")
        if vendor_ret is not None and pd.notna(vendor_ret):
            # Vendor already booked the terminal return in the frame — KEEP it as-is
            # (the preferred source); do not overwrite, just record the decision.
            log.append({"ric": name, "date": date, "action": "vendor_terminal_kept",
                        "value": float(vendor_ret)})
            continue
        # Reason gate (latent until the reason is re-pulled):
        # an M&A / merger / buyout name was cashed out (often at a premium), so the Shumway
        # performance surcharge MUST NOT apply — keep its real last return untouched. Only
        # performance / bankruptcy / compliance / liquidation (or an ABSENT reason, the
        # conservative current default) gets the −30/−55 surcharge. On the current vault
        # every reason is absent, so this branch never fires (univ4 byte-identical).
        reason_class = classify_delist_reason(info.get("reason"))
        if reason_class == "mna":
            log.append({"ric": name, "date": date, "action": "mna_keep_vendor_terminal",
                        "reason": str(info.get("reason")),
                        "citation": "Shumway-style surcharge withheld: M&A, not a performance delisting"})
            continue
        exchange = info["exchange"]
        if exchange not in corr:
            raise KeyError(f"no delisting correction configured for exchange '{exchange}'")
        dl = float(corr[exchange])
        label = _last_valid_label(out, name, date)
        if label is None:
            # No observation to compound into anywhere in the window — record the skip
            # rather than fabricate a row (R4: missing stays missing, never invented).
            log.append({"ric": name, "date": date, "action": "shumway_skipped_no_obs",
                        "exchange": exchange, "value": dl,
                        "citation": "Shumway 1997 JF; Shumway & Warther 1999 JF"})
            continue
        prior = float(out.at[label, name])
        value = (1.0 + prior) * (1.0 + dl) - 1.0   # multiplicative, never additive
        out.at[label, name] = value
        log.append({"ric": name, "date": date, "booked_on": pd.Timestamp(label),
                    "action": "shumway_correction_applied", "exchange": exchange,
                    "reason_class": reason_class,   # 'performance' (gated) or 'unknown' (reason absent)
                    "delisting_return": dl, "prior_return": prior, "value": value,
                    "citation": "Shumway 1997 JF; Shumway & Warther 1999 JF"})
    return out, pd.DataFrame(log)


def members_asof(pit: pd.DataFrame, when: pd.Timestamp) -> list[str]:
    """Members per the latest month-end STRICTLY BEFORE `when` (PIT, R3)."""
    prior = pit[pit["month_end"] < pd.Timestamp(when)]
    if prior.empty:
        raise ValueError(f"no membership month-end before {when} — span starts later")
    last = prior["month_end"].max()
    return sorted(prior.loc[prior["month_end"] == last, "ric"])


def top30_at(window_start: pd.Timestamp, mcap: pd.DataFrame,
             pit: pd.DataFrame, n: int | None = None) -> list[str]:
    """The pre-registered selection rule (PREREG §5): top-n by market cap among PIT
    members as of the window start. Caps are read from the LAST SESSION STRICTLY
    BEFORE window_start; names without a prior cap cannot be selected (no
    information -> no position, rather than any imputation)."""
    n = int(n if n is not None else get("environment.universe.n_assets"))
    when = pd.Timestamp(window_start)
    members = [m for m in members_asof(pit, when) if m in mcap.columns]
    prior_caps = mcap.loc[mcap.index < when]
    if prior_caps.empty:
        raise ValueError("no market-cap observation strictly before the window start")
    snapshot = prior_caps.iloc[-1][members].dropna().sort_values(ascending=False)
    if len(snapshot) < n:
        raise ValueError(f"only {len(snapshot)} capped members before {when.date()} — need {n}")
    return list(snapshot.index[:n])
