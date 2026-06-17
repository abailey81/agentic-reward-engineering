"""Corporate-actions integrity (stage 5) + outlier & error taxonomy (stage 6).

PRIME DIRECTIVE: this project is ABOUT tail risk — a genuine 2008/2020 crash day is
the signal, not a defect. Therefore NOTHING here mutates a return. Every function
returns FLAGS / CLASSIFICATIONS / QUARANTINE ROWS; the clean layer carries flag
columns alongside untouched values, and only reason-coded quarantine (with the full
row preserved) ever excludes an observation. Silent winsorization of training
returns is the cardinal sin (user spec; R4).

Citations: Ince & Porter (2006, JFR) — $1 prior-price screen and the >300%-reversal
screen (their monthly two-period rule, adapted here to daily frequency: the reversal
partner is sought within `reversal_pair_max_lag_days` trading days — adaptation
documented, parameters in config data.platform.outliers). Shumway (1997 JF) handles
delisting bias in membership.py, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get


def _cfg() -> dict:
    return get("data.platform.outliers")


# ------------------------------------------------------------ stage 5: corp actions
def ri_consistency_flags(ri: pd.Series, price: pd.Series, dividends: pd.Series,
                         tol: float | None = None) -> pd.DataFrame:
    """Total-return internal consistency: RI_t/RI_{t-1} should equal
    (P_t + D_t)/P_{t-1} (Datastream RI construction). Rows breaching `tol` are
    flagged with both implied returns for inspection."""
    tol = float(tol if tol is not None else _cfg()["ri_consistency_tol"])
    df = pd.DataFrame({"ri": ri, "price": price,
                       "div": dividends.reindex(ri.index).fillna(0.0)}).dropna(subset=["ri", "price"])
    ri_ret = df["ri"] / df["ri"].shift(1) - 1.0
    pd_ret = (df["price"] + df["div"]) / df["price"].shift(1) - 1.0
    gap = (ri_ret - pd_ret).abs()
    bad = gap > tol
    return pd.DataFrame({"ri_return": ri_ret[bad], "price_div_return": pd_ret[bad],
                         "abs_gap": gap[bad]})


def split_artifact_flags(returns: pd.Series, splits: pd.Series | None = None) -> pd.DataFrame:
    """Unadjusted-split signatures: one-day returns ≈ -50% (2:1) or -66.7% (3:1)
    WITHOUT a matching vendor split record the same day -> ERROR-suspect artifact."""
    cfg = _cfg()
    sigs, tol = [float(s) for s in cfg["split_signatures"]], float(cfg["split_tolerance"])
    r = returns.dropna()
    hits = r[[any(abs(x - s) <= tol for s in sigs) for x in r]]
    rows = []
    for date, x in hits.items():
        recorded = bool(splits is not None and date in splits.index
                        and pd.notna(splits.loc[date]) and splits.loc[date] not in (0, 1))
        rows.append({"date": date, "return": float(x),
                     "vendor_split_recorded": recorded,
                     "flag": "split_recorded_ok" if recorded else "unadjusted_split_suspect"})
    return pd.DataFrame(rows)


def forward_split_artifact_flags(returns: pd.Series,
                                 splits: pd.Series | None = None) -> pd.DataFrame:
    """Unadjusted FORWARD-split signatures: one-day total returns ≈ +100% (2:1), +200%
    (3:1) or +300% (4:1) WITHOUT a matching vendor split record the same day ->
    ERROR-suspect artifact. This is the POSITIVE analogue of :func:`split_artifact_flags`
    (which catches the price-side −50%/−66.7%): when a split is not back-applied in the
    *total-return* series, the split-day return spikes to the share-multiplier minus one
    (a 3:1 split reads exactly +200.00%; Refinitiv research panel, JCI Oct-2007). A genuine
    +200% one-day move on a live large-cap does not occur, so the signature is the tell.
    Config: ``data.platform.outliers.forward_split_signatures`` + ``split_tolerance``.
    Flag-only — like every function here, it never mutates a return (PRIME DIRECTIVE)."""
    cfg = _cfg()
    sigs = [float(s) for s in cfg["forward_split_signatures"]]
    tol = float(cfg["split_tolerance"])
    r = returns.dropna()
    hits = r[[any(abs(x - s) <= tol for s in sigs) for x in r]]
    rows = []
    for date, x in hits.items():
        recorded = bool(splits is not None and date in splits.index
                        and pd.notna(splits.loc[date]) and splits.loc[date] not in (0, 1))
        rows.append({"date": date, "return": float(x),
                     "vendor_split_recorded": recorded,
                     "flag": "split_recorded_ok" if recorded else "unadjusted_fwd_split_suspect"})
    return pd.DataFrame(rows)


def cross_vendor_action_mismatches(div_a: pd.Series, div_b: pd.Series,
                                   amount_tol: float = 1e-4) -> pd.DataFrame:
    """Dividend events present in one vendor and absent (or differing beyond
    `amount_tol`) in the other. Two vendors disagreeing is information (Ince &
    Porter) — these rows seed the reconciliation clustering."""
    a = div_a[div_a.fillna(0) != 0]
    b = div_b[div_b.fillna(0) != 0]
    dates = a.index.union(b.index)
    rows = []
    for d in dates:
        va = float(a.get(d, np.nan)) if d in a.index else np.nan
        vb = float(b.get(d, np.nan)) if d in b.index else np.nan
        if np.isnan(va) or np.isnan(vb) or abs(va - vb) > amount_tol:
            rows.append({"date": d, "vendor_a": va, "vendor_b": vb,
                         "kind": "missing_one_side" if (np.isnan(va) or np.isnan(vb))
                                 else "amount_mismatch"})
    return pd.DataFrame(rows)


def stale_run_flags(prices: pd.Series, min_days: int | None = None) -> pd.DataFrame:
    """Runs of >= min_days identical consecutive prices -> illiquidity / stale-feed
    flag (information for screens and the missing-engine's halt classification)."""
    min_days = int(min_days if min_days is not None else _cfg()["stale_run_min_days"])
    p = prices.dropna()
    if p.empty:
        return pd.DataFrame(columns=["start", "end", "days", "price"])
    grp = (p != p.shift(1)).cumsum()
    runs = p.groupby(grp).agg(["size", "first"])
    idx = p.groupby(grp).apply(lambda s: (s.index[0], s.index[-1]))
    out = []
    for g, (size, price) in runs.iterrows():
        if size >= min_days:
            start, end = idx.loc[g]
            out.append({"start": start, "end": end, "days": int(size), "price": float(price)})
    return pd.DataFrame(out)


def zero_volume_flags(volume: pd.Series) -> pd.DataFrame:
    """Zero-volume sessions: flagged (no trade -> stale close; relevant to cost
    realism and halt classification), never deleted."""
    v = volume.dropna()
    z = v[v == 0]
    return pd.DataFrame({"date": z.index, "flag": "zero_volume"})


def reconstruct_unadjusted_close(close: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Recover the ACTUAL traded price from a split-adjusted close series.

    Vendor subtlety (documented for the data chapter): yfinance's `Close` is
    split-adjusted even with auto_adjust=False — Yahoo retroactively applies splits,
    so NVDA's 2005 close reads ~$0.20 although it traded ~$23 (cumulative later
    splits ≈ 120x). The traded price at t is close_t x prod(split ratios dated AFTER
    t). Refinitiv exposes unadjusted P natively; for the shadow vendor we reconstruct.
    Pure transform of vendor-published series — nothing is fabricated (R4)."""
    if splits is None:
        return close.copy()
    s = splits.reindex(close.index).fillna(0.0)
    ratios = s.where(s > 0, 1.0)                       # 0 = no split that day
    # factor_t = product of ratios STRICTLY AFTER t (reverse cumulative, exclusive)
    future_factor = ratios[::-1].cumprod()[::-1] / ratios
    return close * future_factor


# ------------------------------------------------------------- stage 6: error taxonomy
def ince_porter_flags(prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Ince & Porter (2006) screens, daily adaptation (parameters: config
    data.series.screens + platform.outliers):
      sub_dollar_prior   — return at t with price_{t-1} < $1 (microstructure noise);
      extreme_reversal   — r_t > +300% with a reversing partner day within
                           `reversal_pair_max_lag_days` such that the compounded
                           pair gain collapses below +50%.
    Action is config-declared (`set_missing`) but EXECUTED only by the quarantine
    queue with reason codes — this function only flags."""
    scr = get("data.series.screens")
    min_price = float(scr["min_prior_price_usd"])
    threshold = float(scr["extreme_return_reversal"]["threshold"])
    max_lag = int(_cfg()["reversal_pair_max_lag_days"])
    rows = []
    prev_price = prices.shift(1)
    sub = returns[(prev_price < min_price) & returns.notna()]
    rows += [{"date": d, "reason": "sub_dollar_prior", "value": float(v),
              "prior_price": float(prev_price.loc[d])} for d, v in sub.items()]
    r = returns.dropna()
    for d, v in r[r > threshold].items():
        pos = r.index.get_loc(d)
        window = r.iloc[pos + 1: pos + 1 + max_lag]
        for d2, v2 in window.items():
            if (1.0 + v) * (1.0 + v2) - 1.0 < 0.5:
                rows.append({"date": d, "reason": "extreme_reversal", "value": float(v),
                             "reversal_date": d2, "reversal_return": float(v2)})
                break
    return pd.DataFrame(rows)


def classify_extreme_days(returns: pd.DataFrame,
                          market_returns: pd.Series | None = None) -> pd.DataFrame:
    """Cross-sectional context classification: |r| >= extreme_abs_return is
    REAL_TAIL when the MARKET also moved (market <= real_crash_market_threshold for
    down moves, or the name's move is broadly shared), ERROR_SUSPECT otherwise.
    2008/2020 tails therefore survive — they carry market context; lone -40% days on
    a flat tape go to quarantine review instead.

    Context must be SELF-EXCLUDED: with `market_returns=None` (default) the market
    for (date, name) is the equal-weight mean of the OTHER names, and breadth counts
    only the other names — a single collapsing stock in a small universe must not
    certify its own crash through its contribution to the average. Pass an external
    index series (e.g. SPX) to override once one exists in the clean layer."""
    cfg = _cfg()
    extreme = float(cfg["extreme_abs_return"])
    crash = float(cfg["real_crash_market_threshold"])
    external = market_returns.reindex(returns.index) if market_returns is not None else None
    rows = []
    for col in returns.columns:
        s = returns[col]
        others = returns.drop(columns=[col])
        hits = s[s.abs() >= extreme]
        for d, v in hits.items():
            peers = others.loc[d].dropna() if not others.empty else pd.Series(dtype=float)
            if external is not None and pd.notna(external.loc[d]):
                m = float(external.loc[d])
            else:
                m = float(peers.mean()) if len(peers) else 0.0
            same_dir_market = (v < 0 and m <= crash) or (v > 0 and m >= 0.03)
            breadth = float((peers.abs() >= extreme / 2).mean()) if len(peers) else 0.0
            real = bool(same_dir_market or breadth >= 0.25)
            rows.append({"date": d, "name": col, "return": float(v), "market_return": m,
                         "extreme_breadth": round(breadth, 3),
                         "class": "REAL_TAIL" if real else "ERROR_SUSPECT"})
    return pd.DataFrame(rows)


def assemble_quarantine(flag_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge stage flags into the reason-coded quarantine queue. Only
    ERROR-class rows belong here; REAL_TAIL classifications are evidence, not
    exclusions. Full original values ride along — quarantine hides nothing."""
    rows = []
    for source, df in flag_frames.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            d = r.to_dict()
            d["source"] = source
            d.setdefault("reason", d.get("flag", d.get("class", source)))
            rows.append(d)
    q = pd.DataFrame(rows)
    if not q.empty and "class" in q.columns:
        q = q[q.get("class").ne("REAL_TAIL") | q["class"].isna()]
    return q.reset_index(drop=True)
