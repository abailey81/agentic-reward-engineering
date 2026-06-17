"""PIT universe build: assemble the entitled Refinitiv pulls into the RESEARCH panel
(replaces the shadow universe the moment `pull-universe --live` completes; ADR-020/021).

Inputs (raw vault, checksum-verified): rf_trd_* (daily total returns, datagrid long
form, PERCENT), rf_mcapm_* (monthly market cap, long form), staged pit_membership.
Outputs: staged returns/mcap panels → clean authority-merged returns (refinitiv
primary, yfinance shadow cross-check) → gold via `panel.build_gold` with membership +
mcap supplied, which materialises D2 (mcap panel), D3 (top-30 per window) and the
PIT rebuild of D1/D4/D6 under suffix `_univ`.

Pure assembly — no fabrication: percent→decimal is the only transform on returns;
missing stays missing (the missing-data engine counts it); panels are XNYS-aligned.
"""
from __future__ import annotations

import pandas as pd

from ..config import get
from . import integrity, validate
from .membership import apply_shumway_corrections
from .panel import build_gold
from .pull_universe import parse_delisting_metadata
from .reconcile_full import reconcile_returns
from .security_master import parse_dead_ric, ric_to_yf_ticker
from .vault import freeze, manifest_entries, read_verified, record_lineage

# RIC exchange-code -> Shumway delisting-correction bucket (config
# data.series.delisting_corrections keys). The dead-^RIC suffix is stripped first
# (parse_dead_ric), so e.g. LEH.N^I08 -> base LEH.N -> '.N' -> 'nyse_amex'. ECNs/ADR
# suffixes that don't map leave the bucket None — the name is then skipped (the
# fixed surcharge is RESERVED for identifiable NYSE/AMEX/Nasdaq terminations, R4).
_EXCHANGE_BUCKET = {
    ".N": "nyse_amex", ".A": "nyse_amex", ".P": "nyse_amex",   # NYSE / AMEX / NYSE Arca
    ".O": "nasdaq", ".OQ": "nasdaq", ".OB": "nasdaq",          # Nasdaq venues
}


def _bucket_from_ric(ric: str) -> str | None:
    """Map a (possibly dead) RIC to its Shumway exchange bucket via the exchange code,
    or None when the suffix is unrecognised (caller skips — never guesses an exchange)."""
    dead = parse_dead_ric(ric)
    base = dead.base if dead else ric
    if "." not in base:
        return None
    suffix = "." + base.rsplit(".", 1)[1]
    return _EXCHANGE_BUCKET.get(suffix)


def _recover_terminal_from_returns(
    ric: str,
    delist_date: pd.Timestamp,
    returns: pd.DataFrame,
    *,
    max_gap_sessions: int = 45,
) -> tuple[float, pd.Timestamp] | None:
    """OBSERVED-terminal recovery (ADR-051; DATA_REPULL_DELISTING.md decided route).

    The reason mnemonics do not resolve under this entitlement (probed 2026-07-01), but the
    vendor's daily series DOES carry the realised terminal return for dead names (Lehman:
    2,042 daily rows through 2008-09). This PURE helper returns the ric's last valid
    (non-NaN, non-null) return and its date, ACCEPTED only when that last print falls within
    ``max_gap_sessions`` sessions of the derived delist date — a name whose prints stop long
    before its delist month has NO recoverable terminal (stale series) and must stay ``None``
    so the conservative surcharge applies downstream. Never guesses (R4).
    """
    if ric not in returns.columns:
        return None
    col = returns[ric].dropna()
    if col.empty:
        return None
    last_date = col.index[-1]
    if last_date > delist_date:
        # prints AFTER the derived delist date: the date derivation is suspect — do not recover.
        return None
    idx = returns.index
    gap = int(idx.get_indexer([delist_date])[0] - idx.get_indexer([last_date])[0])
    if gap < 0 or gap > int(max_gap_sessions):
        return None
    return float(col.iloc[-1]), pd.Timestamp(last_date)


def _derive_delisting_map(
    columns,
    sessions: pd.DatetimeIndex,
    returns: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Build the ``{ric: {date, exchange, vendor_terminal_return}}`` map that
    :func:`membership.apply_shumway_corrections` consumes, for the dead ^RICs in the
    panel (stage 7). Order of evidence, all from already-frozen RAW artifacts (no new
    pulls, R4):

      1. **Vendor metadata (preferred).** ``rf_meta_*`` (A4: TR.InstrumentDelistedDate /
         TR.ExchangeName / terminal TR.TotalReturn) parsed by
         :func:`pull_universe.parse_delisting_metadata` — gives the vendor terminal
         return (kept as-is by the corrector) and the mapped exchange bucket.
      2. **^RIC-suffix fallback.** For a dead ^RIC the metadata pull missed, the delist
         date derives from the ``^MYY`` month-letter suffix (security_master.parse_dead_ric)
         and the exchange bucket from the RIC's exchange code (:func:`_bucket_from_ric`).
      3. **OBSERVED-terminal recovery (ADR-051).** When metadata carries no terminal and a
         ``returns`` frame is supplied, the name's last valid in-window return is recovered
         (:func:`_recover_terminal_from_returns`) — the corrector then KEEPS the realised
         terminal ("vendor_terminal_kept") and the −30/−55% surcharge applies ONLY to names
         with no recoverable terminal. This is what upgrades the surcharge band-end from
         M&A-contaminated (univ4) to a correct Shumway panel (univ5s).

    The delist date is snapped to the last session ``<= date`` so the corrector books
    onto a real grid label; names with no resolvable exchange are omitted (skipped, not
    surcharged). Pure derivation over the panel — invoked only on the live raw vault."""
    sessions = pd.DatetimeIndex(sessions)
    meta_raw = [read_verified(e["relpath"]) for e in manifest_entries()
                if e["layer"] == "raw" and e["name"].startswith("rf_meta_")]
    vendor: dict[str, dict] = {}
    for frame in meta_raw:
        try:
            vendor.update(parse_delisting_metadata(frame))
        except ValueError:        # a meta chunk without the minimal RIC/date columns
            continue

    def _snap(date: pd.Timestamp | None) -> pd.Timestamp | None:
        if date is None or pd.isna(date):
            return None
        prior = sessions[sessions <= pd.Timestamp(date)]
        return prior.max() if len(prior) else None

    out: dict[str, dict] = {}
    for ric in columns:
        if parse_dead_ric(str(ric)) is None:
            continue                                   # alive name — no delisting event
        info = vendor.get(str(ric), {})
        dead = parse_dead_ric(str(ric))
        suffix_date = pd.Timestamp(year=dead.year, month=dead.month, day=1) + pd.offsets.MonthEnd(0)
        date = _snap(info.get("date")) or _snap(suffix_date)
        exchange = info.get("exchange") or _bucket_from_ric(str(ric))
        if date is None or exchange is None:
            continue                                   # unresolved — skipped, never guessed
        terminal = info.get("vendor_terminal_return")
        terminal_source = "vendor_meta" if terminal is not None else None
        if terminal is None and returns is not None:
            recovered = _recover_terminal_from_returns(str(ric), date, returns)
            if recovered is not None:
                terminal, term_date = recovered
                terminal_source = f"observed_returns@{term_date.date()}"
        out[str(ric)] = {"date": date, "exchange": exchange,
                         "vendor_terminal_return": terminal,
                         "terminal_source": terminal_source,
                         "reason": info.get("reason")}   # None until the reason is re-pulled (R4)
    return out


def assemble_long_panel(frames: list[pd.DataFrame], value_name: str,
                        scale: float = 1.0) -> pd.DataFrame:
    """Concatenate datagrid long frames (Instrument, Date, value) -> wide panel.
    Duplicate (date, ric) cells keep the LAST occurrence (chunk overlap guard);
    `scale` converts units (percent -> decimal for returns)."""
    tidy = []
    for df in frames:
        d = df.copy()
        d.columns = ["ric", "date", "value"][: len(d.columns)]
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        tidy.append(d.dropna(subset=["date"]))
    long = pd.concat(tidy, ignore_index=True)
    long = long.drop_duplicates(subset=["date", "ric"], keep="last")
    wide = long.pivot(index="date", columns="ric", values="value").sort_index()
    wide.columns.name = None
    wide.index.name = None
    return wide * scale


def _collect(prefix: str) -> list[pd.DataFrame]:
    return [read_verified(e["relpath"]) for e in manifest_entries()
            if e["layer"] == "raw" and e["name"].startswith(prefix)]


def screen_research_returns(returns: pd.DataFrame,
                            prices: pd.DataFrame | None = None,
                            splits: pd.DataFrame | None = None,
                            ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stage 5/6 corporate-action INTEGRITY screens on the Refinitiv RESEARCH panel
    (Ince & Porter 2006 + split-artifact detection) — the screens the datasheet claims
    are applied to this panel. This makes that claim true (they were previously wired
    ONLY on the yfinance pilot path in ``cli.cmd_build``; ``build_universe`` never called
    them — supervisor data-integrity audit 2026-06-25).

    Runs per-column over the wide returns panel and concatenates:

      * :func:`integrity.ince_porter_flags` — the $1-prior-price screen + the >+300%
        extreme-reversal screen (daily adaptation; config ``data.series.screens`` +
        ``data.platform.outliers``). The $1 screen needs ACTUAL prices: pass the Refinitiv
        unadjusted ``TRDPRC_1`` panel (``rf_px_*``) as ``prices``. When a column's price is
        absent the price screen is SKIPPED for it (honest: no price ⇒ no microstructure
        verdict — NOT flag-everything; passing the return series as a price would read every
        ~0.01 return as sub-$1 and flag the whole panel).
      * :func:`integrity.split_artifact_flags` — unadjusted 2:1/3:1 price-side −50%/−66.7%
        signatures (on the total-return series).
      * :func:`integrity.forward_split_artifact_flags` — the POSITIVE +100/+200/+300%
        total-return analogue (catches e.g. the JCI 3:1 Oct-2007 +200.00% artifact).

    FLAG-ONLY (R4 / PRIME DIRECTIVE): nothing here mutates a return. The returns frame is
    handed back UNCHANGED; the second tuple element is the assembled flag report (one row
    per flagged (name, date), full value preserved) plus a summary census. A genuine
    2008/2020 crash day is the signal, not a defect — so the report is EVIDENCE for the
    data chapter, not an exclusion list. The split rows ARE candidates for review because
    a +200% one-day total return on a live large-cap does not occur.

    ``prices`` / ``splits`` (optional) are the wide unadjusted-price / vendor-split panels
    for the same RICs. The research panel ships an empty splits table, so every signature
    hit is necessarily ``vendor_split_recorded=False`` and surfaces as a suspect — the
    honest posture when no split record exists to clear it."""
    def _col(panel: pd.DataFrame | None, c: str) -> pd.Series | None:
        if panel is None or c not in panel.columns:
            return None
        return panel[c]

    ip_frames: dict[str, pd.DataFrame] = {}
    split_rows: list[pd.DataFrame] = []
    for c in returns.columns:
        s = returns[c]
        price_col = _col(prices, c)
        if price_col is not None:
            price_col = pd.to_numeric(price_col, errors="coerce").reindex(s.index)
            ip = integrity.ince_porter_flags(price_col, s)   # $1-prior-price + >+300% reversal
        else:
            # No price for this column: run ONLY the >+300% reversal screen (price-free) by
            # handing an all-NaN price series — the sub-dollar branch then matches nothing.
            ip = integrity.ince_porter_flags(pd.Series(float("nan"), index=s.index), s)
        if not ip.empty:
            ip = ip.copy()
            ip["name"] = c
            ip_frames[f"ince_porter:{c}"] = ip
        for fn in (integrity.split_artifact_flags, integrity.forward_split_artifact_flags):
            fr = fn(s, _col(splits, c))
            if not fr.empty:
                fr = fr.copy()
                fr["name"] = c
                split_rows.append(fr)

    flag_frames = dict(ip_frames)
    if split_rows:
        flag_frames["split_artifacts"] = pd.concat(split_rows, ignore_index=True)
    report = integrity.assemble_quarantine(flag_frames)

    def _count(reason: str) -> int:
        if report.empty or "reason" not in report.columns:
            return 0
        return int((report["reason"] == reason).sum())

    summary = {
        "n_flag_rows": int(len(report)),
        "n_sub_dollar_prior": _count("sub_dollar_prior"),
        "n_extreme_reversal": _count("extreme_reversal"),
        "n_unadjusted_split_suspect": _count("unadjusted_split_suspect"),
        "n_unadjusted_fwd_split_suspect": _count("unadjusted_fwd_split_suspect"),
        "policy": "flag-only (R4): returns are NEVER mutated; this is the data-chapter evidence log",
        "citation": "Ince & Porter 2006 JFR",
    }
    return returns, report, summary


def build_universe(suffix: str = "_univ", run_id: str = "build_universe",
                   apply_delisting: bool = False, screen: bool = False) -> dict:
    """Assemble staged panels and build the PIT gold layer. Raises loudly when the
    required pulls are absent (never substitutes; R4).

    ``apply_delisting`` (default False — the existing ``_univ`` build is byte-identical)
    inserts STAGE 7 between the clean-returns freeze and ``build_gold``: the dead ^RICs'
    Shumway-STYLE delisting returns are compounded into the CORRECTED frame that gold is
    then built from (see :func:`_derive_delisting_map` + ``membership.apply_shumway_corrections``),
    and the per-correction audit log is frozen. This is the ``_univ4`` panel — the
    survivorship-correct alternative to the env's ``liquidate_to_cash`` (which zero-fills
    dead names INTO the exact left tail the H2 measures read; ADR-024 / M3-M4 review).

    ``screen`` (default False — the existing ``_univ`` build is byte-identical) inserts the
    STAGE 5/6 corporate-action INTEGRITY screens (:func:`screen_research_returns`:
    Ince-Porter + split-artifact + forward-split) over the RESEARCH returns and freezes the
    flag report as ``integrity_report{suffix}.parquet`` (CLEAN layer). It is FLAG-ONLY (R4):
    the returns / gold are unchanged, so ``screen=True`` produces the SAME panel plus the
    evidence log the datasheet claims this panel carries. Call with ``suffix="_univ3s"`` to
    materialise the screened research panel as a NEW artifact alongside ``_univ3``.

    NOTE: GATED: the real ``_univ4`` rebuild needs the licensed ``data_pipeline`` re-run +
    Refinitiv creds (it consumes the frozen ``rf_trd_*`` / ``rf_meta_*`` raw vault). This
    function is CODE-complete and synthetic-tested, but the parquet is not regenerated
    here. Call ``build_universe(suffix="_univ4", apply_delisting=True)`` once the raw
    vault is present to materialise it."""
    trd = _collect("rf_trd_")
    if not trd:
        raise FileNotFoundError("no rf_trd_* artifacts — run `make pull-universe LIVE=1` first")
    mcapm = _collect("rf_mcapm_")
    pit_entries = [e for e in manifest_entries() if e["name"].startswith("pit_membership")]
    if not pit_entries:
        raise FileNotFoundError("pit_membership missing — A1 has not landed")
    pit = read_verified(pit_entries[-1]["relpath"])   # latest grid (incl. selection buffer)
    pit["month_end"] = pd.to_datetime(pit["month_end"])

    window = get("data.window")
    returns = assemble_long_panel(trd, "tr", scale=0.01)        # percent -> decimal
    sessions = validate.trading_sessions(window["start"],
                                         min(pd.Timestamp(window["end"]),
                                             returns.index.max()).date().isoformat())
    returns_aligned, align_rep = validate.align_to_calendar(returns, sessions)
    flags, missing_rep = validate.classify_missing(returns_aligned, sessions,
                                                   raw_index=returns.index)

    raw_inputs = [e for e in manifest_entries() if e["layer"] == "raw"
                  and e["name"].startswith(("rf_trd_", "rf_mcapm_"))]
    art_ret = freeze(returns_aligned, "staged", f"staged_returns_refinitiv{suffix}.parquet",
                     {"vendor": "refinitiv", "stage": "build_universe",
                      "units": "decimal (percent/100)", "alignment": align_rep,
                      "missing": missing_rep.as_dict()}, fmt="parquet", run_id=run_id)
    record_lineage(art_ret, raw_inputs, "build_universe.staging",
                   {"n_names": int(returns_aligned.shape[1])})

    mcap_art = None
    mcap_panel = None
    if mcapm:
        mcap_panel = assemble_long_panel(mcapm, "mcap")
        mcap_art = freeze(mcap_panel, "staged", f"staged_mcap_refinitiv{suffix}.parquet",
                          {"vendor": "refinitiv", "stage": "build_universe",
                           "frequency": "monthly"}, fmt="parquet", run_id=run_id)
        record_lineage(mcap_art, raw_inputs, "build_universe.staging", {})

    # clean: refinitiv IS the authority for total_return (config vendor_authority);
    # the yfinance shadow stays a parallel staged panel for reconciliation, never
    # blended cell-wise into the research series (R4).
    art_clean = freeze(returns_aligned, "clean", f"clean_returns{suffix}.parquet",
                       {"vendor": "refinitiv", "stage": "build_universe",
                        "authority": "refinitiv (config data.platform.vendor_authority)"},
                       fmt="parquet", run_id=run_id)
    record_lineage(art_clean, [art_ret], "authority", {})

    # ---- STAGE 5/6: corporate-action INTEGRITY screens on the RESEARCH panel --------
    # Default off (byte-identical _univ build). When screen=True the Ince-Porter + split-
    # artifact + forward-split screens run over the research returns and the flag report is
    # frozen. FLAG-ONLY (R4): returns/gold are NEVER mutated — this is the data-chapter
    # evidence the datasheet claims this panel carries (previously the screens ran ONLY on
    # the yfinance pilot path in cli.cmd_build; build_universe never called them).
    screen_summary = None
    if screen:
        # NOTE (verified 2026-06-25): the frozen rf_px_ (TRDPRC_1) panel is SPLIT-ADJUSTED
        # (NVDA reads $0.197 in 2005-01, AAPL $1.13 — the fully back-adjusted price, not the
        # ~$20/~$40 traded level), so the Ince-Porter $1-prior-price screen on it would FALSELY
        # flag every later-split high-flyer's early history as penny-stock noise. We therefore do
        # NOT feed it: the research-panel screen runs the price-FREE screens (>+300% reversal +
        # split-artifact + forward-split). The $1 screen needs an UNADJUSTED price series (a gated
        # re-pull: TR.PriceClose unadjusted, or reconstruct via integrity.reconstruct_unadjusted_close
        # from the split panel) — documented, not silently approximated (R4).
        _, integ_report, screen_summary = screen_research_returns(
            returns_aligned, prices=None, splits=None)
        if not integ_report.empty:
            art_integ = freeze(integ_report, "clean", f"integrity_report{suffix}.parquet",
                               {"vendor": "derived", "stage": "build_universe.integrity_screens",
                                "policy": "flag-only (R4); returns NEVER mutated",
                                "summary": screen_summary,
                                "citation": "Ince & Porter 2006 JFR"},
                               fmt="parquet", run_id=run_id)
            record_lineage(art_integ, [art_clean], "build_universe.integrity_screens",
                           screen_summary)

    # ---- STAGE 7: Shumway-STYLE delisting correction (the _univ4 panel) -----------
    # Default off: gold is built from the unmodified clean returns (the _univ behaviour).
    # When apply_delisting=True, the dead ^RICs' terminal returns are compounded in BEFORE
    # gold, so the crash return reaches the training-window left tail instead of being
    # zero-filled by the env's liquidate_to_cash (ADR-024 / M3-M4 adversarial review).
    gold_returns = returns_aligned
    if apply_delisting:
        # ADR-051: thread the returns so the OBSERVED-terminal recovery runs — names whose
        # realised terminal is in the daily series keep it (vendor_terminal_kept) and only
        # genuinely terminal-less names receive the -30/-55% surcharge.
        delisted = _derive_delisting_map(returns_aligned.columns, sessions,
                                         returns=returns_aligned)
        gold_returns, shumway_log = apply_shumway_corrections(returns_aligned, delisted)
        art_corr = freeze(gold_returns, "clean", f"clean_returns_shumway{suffix}.parquet",
                          {"vendor": "refinitiv", "stage": "build_universe.shumway_correction",
                           "policy": "Shumway-STYLE: OBSERVED terminal recovered from the daily "
                                     "series and kept (ADR-051); vendor terminal kept; else "
                                     "-30% NYSE/AMEX / -55% Nasdaq; multiplicative (1+r)(1+dl)-1",
                           "n_corrections": int(len(shumway_log)),
                           "citation": "Shumway 1997 JF; Shumway & Warther 1999 JF"},
                          fmt="parquet", run_id=run_id)
        record_lineage(art_corr, [art_clean], "build_universe.shumway_correction",
                       {"n_delisted_considered": int(len(delisted))})
        if not shumway_log.empty:
            freeze(shumway_log, "clean", f"shumway_audit_log{suffix}.parquet",
                   {"vendor": "derived", "stage": "build_universe.shumway_correction",
                    "schema": "ric,date,booked_on,action,exchange,delisting_return,"
                              "prior_return,value,citation"},
                   fmt="parquet", run_id=run_id)

    # LATEST fred_macro* (ADR-051): the original fred_macro.csv ends 2025-12-31; the extension
    # refresh lands as fred_macro_x26.csv (write-once vault versioning, same pattern as the
    # span-stamped pit). Latest-by-frozen_utc so a future refresh supersedes cleanly.
    fred_entries = [e for e in manifest_entries()
                    if str(e.get("name", "")).startswith("fred_macro")]
    if not fred_entries:
        raise FileNotFoundError("fred_macro*.csv missing — VIX required for gold features")
    fred_name = max(fred_entries, key=lambda e: str(e.get("frozen_utc", "")))["name"]
    vix = read_verified(fred_name)["VIXCLS"].astype(float)
    vix.index = pd.to_datetime(vix.index)

    gold = build_gold(gold_returns, vix, sessions,
                      clean_mcap=mcap_panel, pit_membership=pit,
                      run_id=run_id, input_artifacts=[art_clean], suffix=suffix)
    recon = reconcile_against_shadow(returns_aligned)
    return {"staged_returns": art_ret.relpath,
            "staged_mcap": mcap_art.relpath if mcap_art else None,
            "clean": art_clean.relpath,
            "gold": {k: a.relpath for k, a in gold["artifacts"].items()},
            "n_names": int(returns_aligned.shape[1]),
            "n_sessions": int(len(sessions)),
            "missing": missing_rep.as_dict(),
            "integrity": screen_summary,
            "reconciliation": recon}


def reconcile_against_shadow(univ_returns: pd.DataFrame) -> dict | None:
    """Two-vendor reconciliation at last (stage 9, activated): map the universe
    panel's RIC columns to yfinance tickers (security_master convention) and
    reconcile against the shadow panel on the intersection. Returns the summary dict
    (full markdown written to reports/) or None when no shadow panel exists."""
    shadow_entry = next((e for e in manifest_entries()
                         if e["name"] == "staged_returns_yfinance_v2.parquet"), None)
    if shadow_entry is None:
        return None
    shadow = read_verified(shadow_entry["relpath"])
    shadow.index = pd.to_datetime(shadow.index)
    mapped = univ_returns.copy()
    mapped.columns = [ric_to_yf_ticker(str(c)) for c in mapped.columns]
    mapped = mapped.loc[:, ~mapped.columns.duplicated(keep="first")]
    common = sorted(set(mapped.columns) & set(shadow.columns))
    if not common:
        return {"common_names": 0}

    def _collect_yf(prefix: str) -> pd.DataFrame | None:
        frames = [read_verified(e["relpath"]) for e in manifest_entries()
                  if e["layer"] == "raw" and e["name"].startswith(prefix)]
        if not frames:
            return None
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        out.index = pd.to_datetime(out.index)
        return out

    divs = _collect_yf("yf_dividends_")               # ex-div/split clustering context:
    spl = _collect_yf("yf_splits_")                   # adjustment-methodology gaps cluster there
    rep = reconcile_returns(mapped[common], shadow[common], dividends=divs, splits=spl)
    from . import vault
    root = vault.ROOT                       # vault's ROOT: tmp-redirected in tests
    out = root / "reports" / "vendor_reconciliation_univ.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Two-vendor reconciliation — Refinitiv (authority) vs yfinance (shadow)\n\n"
                   f"Names compared: {len(common)} (RIC→ticker mapped via security master). "
                   "Tolerance: config data.vendors.reconciliation_tolerance.\n\n"
                   + rep.to_markdown() + "\n",
                   encoding="utf-8")  # the markdown carries — / → (U+2192); never the platform codepage
    rows = rep.rows
    return {"common_names": len(common),
            "median_corr": float(pd.Series([r.corr for r in rows]).median()),
            "total_days_over_tol": int(sum(r.days_over_tol for r in rows)),
            "unexplained_quarantined": 0 if rep.quarantine is None else int(len(rep.quarantine)),
            "report": str(out.relative_to(root))}
