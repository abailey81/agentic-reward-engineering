"""Portfolio-level REFERENCE series for the evaluation / reporting layer (block B11/B13).

These series live ENTIRELY in inference/reporting — they never enter the frozen, anonymised env
observation or a reward (so H2's contribution surface is untouched; data-enrichment research
2026-06-20). All three are already pulled + frozen on disk by ``data_pipeline``:

* **risk-free rate** — FRED ``DGS3MO`` (3-month T-bill secondary-market yield) from
  ``data/raw/fred_macro.csv``, converted to a per-session decimal. Replaces the ``rf = 0`` hardwire
  in the headline Sharpe/Sortino/Deflated-Sharpe (the "rf cancels in pairwise comparison" defence is
  false for differing-volatility benchmark comparisons). The per-session rate is the GEOMETRIC
  ``(1 + DGS3MO/100)**(1/252) - 1`` (annualised yield in percent -> daily decimal); DGS3MO is preferred
  over the Fama-French daily RF, which is piecewise-constant within each month (Dimitrov & Govindaraj,
  JOIM 2021). Any change to the rf CONVENTION in the headline is pre-registration-relevant (R20).
* **market proxy** — ``market_ew`` (equal-weight return of the FULL survivorship-free PIT universe)
  from ``data/gold/market_proxy_<suffix>.parquet``. A REAL broad-market line for alpha / beta /
  information-ratio reporting — distinct from the 30-asset 1/N strategy (the anonymised panel has no
  index column, so this EW-universe series is the EQUAL-WEIGHT market stand-in).
* **S&P 500 total return** — ``.SPXTR`` from ``data/raw/rf_spxtr.csv`` (+ the ``_x26`` extension),
  pulled from Refinitiv and frozen with provenance on 2026-07-01. **CAP-WEIGHTED**, and therefore the
  benchmark a reader actually means by "the market"; the equal-weight proxy above is a different
  animal (it tilts small and rebalances). Added 2026-07-30 — see :func:`load_spx_total_return`.

  NOTE: **CORRECTION, 2026-07-30.** This docstring previously read *"a cap-weighted SPX-TR remains a
  documented limitation"*. **That was FALSE**: the ``.SPXTR`` series had been pulled, frozen and
  provenance-stamped on 2026-07-01 and was sitting in ``data/raw/`` unloaded — ``grep rf_spxtr src
  scripts`` returned nothing. We documented a limitation we did not have. Found only because someone
  asked "don't we have S&P 500?", not by any check of ours; the lesson is filed with the process
  errors — **a "documented limitation" must be re-verified against the disk before it is written, or
  it becomes a false statement that survives precisely because it sounds humble.**
* **Fama-French factors** — ``Mkt-RF, SMB, HML`` from ``data/raw/french_F-F_Research_Data_Factors_daily.csv``
  (the Momentum file ``french_F-F_Momentum_Factor_daily.csv`` is on disk but NOT loaded here; add it to
  ``load_ff_factors`` if a 4-factor attribution is wanted), for OUT-OF-SAMPLE attribution in the analysis chapter.

Every loader ALIGNS to the panel's own session axis (so a value is knowable at each trading date),
forward-fills the rare publication gaps (never reading the future), and degrades gracefully to a
zero/empty series with ``available=False`` when a file is absent (a synthetic-only install), so no
caller crashes. Any change to the RF/benchmark CONVENTION is pre-registration-relevant (ADR-038, R20).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

__all__ = [
    "RiskFreeResult",
    "MarketProxyResult",
    "FactorResult",
    "load_risk_free_daily",
    "load_market_proxy_returns",
    "load_ff_factors",
    "TRADING_DAYS_PER_YEAR",
]

TRADING_DAYS_PER_YEAR = 252
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"
_FRED_CSV = "fred_macro.csv"
_FF_CSV = "french_F-F_Research_Data_Factors_daily.csv"

#: ADR-051 (Split C to 2026-06-30): the ORIGINAL raw reference files end before the cutoff
#: (fred_macro 2025-12-31; the French dailies 2026-04-30), so extension refreshes land as
#: VERSIONED write-once artifacts. Readers prefer the newest refresh WHEN PRESENT and fall back
#: to the canonical name — same layout/units (verified 2026-07-02: fred keeps observation_date +
#: percent yields; the ff3 refresh keeps Date + Mkt-RF/SMB/HML/RF decimals). Newest-first tuples.
_REFRESHED_RAW: dict[str, tuple[str, ...]] = {
    _FRED_CSV: ("fred_macro_x26.csv",),
    _FF_CSV: ("french_ff3_daily_x26.csv",),
    # 2026-07-05: the Momentum refresh was pulled + frozen (refresh_french_2026.py, to 2026-05-29)
    # but had NO mapping here, so attribution's Carhart-4 Mom (and its RF read) silently used the
    # canonical file ending 2026-04-30 and forward-filled the test window's tail with a constant.
    "french_F-F_Momentum_Factor_daily.csv": ("french_mom_daily_x26.csv",),
}


def _raw_path(raw_dir: Path | str, canonical: str) -> Path:
    """The freshest available raw artifact for ``canonical`` (refresh preferred, else canonical)."""
    for cand in _REFRESHED_RAW.get(canonical, ()):
        p = Path(raw_dir) / cand
        if p.exists():
            return p
    return Path(raw_dir) / canonical


class RiskFreeResult:
    """Per-session risk-free decimal returns aligned to a date axis, + provenance."""

    def __init__(
        self, daily: np.ndarray, *, available: bool, source: str, annual_pct_mean: float,
        last_observation: str | None = None, n_extrapolated: int = 0,
    ) -> None:
        self.daily = daily
        self.available = available
        self.source = source
        self.annual_pct_mean = annual_pct_mean
        #: Provenance of the forward-fill (deep review #53): the last REAL observation and how many
        #: target sessions fall beyond it (those carry a constant, not data). 0 == fully covered.
        self.last_observation = last_observation
        self.n_extrapolated = int(n_extrapolated)


class MarketProxyResult:
    """Per-session market-proxy decimal returns aligned to a date axis, + provenance."""

    def __init__(
        self, returns: np.ndarray, *, available: bool, column: str,
        last_observation: str | None = None, n_extrapolated: int = 0,
    ) -> None:
        self.returns = returns
        self.available = available
        self.column = column
        #: See :class:`RiskFreeResult` (deep review #53). Reachable here: the ACTIVE ``univ5`` proxy
        #: covers the frozen window exactly (0), but legacy suffixes end 2025-12-31 — so a run under
        #: ``LLM_RP_GOLD_SUFFIX=univ3`` (the documented delisting-band 0% end) would otherwise carry
        #: ~123 constant market sessions with no signal.
        self.last_observation = last_observation
        self.n_extrapolated = int(n_extrapolated)


class FactorResult:
    """Per-session Fama-French factor decimals (columns Mkt-RF/SMB/HML), + provenance."""

    def __init__(
        self, factors: dict[str, np.ndarray], *, available: bool,
        last_observation: str | None = None, n_extrapolated: int = 0,
    ) -> None:
        self.factors = factors
        self.available = available
        #: See :class:`RiskFreeResult` — the last real observation across the loaded factor columns
        #: and how many target sessions are forward-filled beyond it (deep review #53).
        self.last_observation = last_observation
        self.n_extrapolated = int(n_extrapolated)


def _aligned_series(
    frame: pd.DataFrame, col: str, dates: np.ndarray
) -> tuple[np.ndarray, pd.Timestamp | None]:
    """Reindex ``frame[col]`` onto ``dates`` (forward-filled, no future read).

    Returns ``(values, last_observation)`` where ``last_observation`` is the timestamp of the last
    REAL (non-NaN) source value. Target dates beyond it are EXTRAPOLATED — a constant carried
    forward — which is what :func:`_extrapolated_after` exists to count and report.

    The second element used to be a hardcoded ``True`` "ok" flag that no caller could ever act on
    (deep review 2026-07-26, #53): a status that is always success is not a status. Every call site
    discarded it, so returning the real provenance instead is free.
    """
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    s = pd.to_numeric(frame[col], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(frame.index if frame.index.name else frame.iloc[:, 0]))
    # Restrict to <= each target date then forward-fill so a gap reads the LAST KNOWN value, never a
    # future publication; leading gaps (before the first observation) fall back to 0.0.
    s = s[~s.index.duplicated(keep="last")].sort_index()
    real = s.dropna()
    last_obs = real.index.max() if not real.empty else None
    aligned = s.reindex(s.index.union(d)).ffill().reindex(d)
    return aligned.to_numpy(dtype=float), last_obs


def _extrapolated_after(dates: np.ndarray, last_obs: pd.Timestamp | None, label: str) -> int:
    """Count target dates BEYOND the last real observation, and warn loudly if there are any.

    WHY THIS EXISTS (deep review 2026-07-26, #53). The forward-fill that correctly bridges a
    publication gap ALSO silently extends the last known value past the end of the source file, for
    as long as the target axis runs. That is not hypothetical: this module's own ``_REFRESHED_RAW``
    note records it happening once already (the Momentum refresh had no mapping, so attribution
    "silently used the canonical file ending 2026-04-30 and forward-filled the test window's tail
    with a constant"). The fix then was a new mapping — but nothing was added to DETECT a recurrence,
    so the same condition returned: MEASURED 2026-07-26, the French dailies end 2026-05-29 while the
    frozen test window runs to 2026-06-30, leaving **21 of 1631 test sessions (1.3%)** of the
    factor ladder regressing against repeated values, with ``available=True`` and no signal anywhere.

    This does NOT change any value — inventing factor data would be worse. It makes the condition
    VISIBLE, exactly as ``rf_source`` already is, so a stale pull is caught by audit rather than
    believed. Callers surface the count on their result object; the log line names the fix.
    """
    if last_obs is None:
        return 0
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    n = int((d > last_obs).sum())
    if n:
        _LOG.warning(
            "market_reference_EXTRAPOLATED %s: %d of %d target sessions fall AFTER the last real "
            "observation (%s) and carry a CONSTANT forward-filled value. Refresh the raw pull and "
            "add it to _REFRESHED_RAW; until then these sessions are not real data.",
            label, n, d.size, str(last_obs)[:10],
        )
    return n


def load_risk_free_daily(
    dates: np.ndarray, *, source: str = "DGS3MO", raw_dir: Path | str = _RAW_DIR
) -> RiskFreeResult:
    """Per-session risk-free DECIMAL return aligned to ``dates`` (FRED ``DGS3MO``/252, default).

    DGS3MO is an ANNUALISED yield in PERCENT; the per-session decimal is ``(1 + y/100)**(1/252) - 1``.
    Falls back to an all-zero series with ``available=False`` if the FRED file is absent.
    """
    path = _raw_path(raw_dir, _FRED_CSV)
    n = int(np.asarray(dates).size)
    if not path.exists():
        return RiskFreeResult(np.zeros(n), available=False, source=source, annual_pct_mean=0.0)
    fm = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in fm.columns else fm.columns[0]
    fm = fm.set_index(date_col)
    if source not in fm.columns:
        return RiskFreeResult(np.zeros(n), available=False, source=source, annual_pct_mean=0.0)
    annual_pct, last_obs = _aligned_series(fm, source, dates)
    n_extra = _extrapolated_after(dates, last_obs, f"risk_free[{source}]")
    annual_pct = np.nan_to_num(annual_pct, nan=0.0)  # leading gap -> 0
    daily = np.power(1.0 + annual_pct / 100.0, 1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return RiskFreeResult(
        daily.astype(float), available=True, source=source,
        annual_pct_mean=float(np.nanmean(annual_pct)),
        last_observation=(str(last_obs)[:10] if last_obs is not None else None),
        n_extrapolated=n_extra,
    )


def load_market_proxy_returns(
    dates: np.ndarray, *, suffix: str | None = None, gold_dir: Path | str = _GOLD_DIR
) -> MarketProxyResult:
    """Per-session market-proxy DECIMAL return (``market_ew``) aligned to ``dates``.

    Reads ``data/gold/market_proxy_<suffix>.parquet`` (a one-column daily-return series). ``suffix=None``
    (default) resolves to ``gold_suffix()`` so the market line tracks the SAME universe as the traded
    panel under ``LLM_RP_GOLD_SUFFIX`` (critical-review 2026-06-20). Falls back to an all-zero series with
    ``available=False`` if the file is absent.
    """
    if suffix is None:
        from src.data.loaders import gold_suffix

        suffix = gold_suffix()
    path = Path(gold_dir) / f"market_proxy_{suffix}.parquet"
    n = int(np.asarray(dates).size)
    if not path.exists():
        return MarketProxyResult(np.zeros(n), available=False, column="market_ew")
    mp = pd.read_parquet(path)
    col = "market_ew" if "market_ew" in mp.columns else str(mp.columns[0])
    s = pd.to_numeric(mp[col], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(mp.index))
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    # Forward-fill interior publication gaps (e.g. bond-market holidays on which equities still
    # traded) so a gap carries the LAST KNOWN market return rather than a spurious flat 0.0% — same
    # convention as ``_aligned_series`` (RF/FF) and the module's forward-fill contract. We can't reuse
    # ``_aligned_series`` directly because this parquet stores dates in an UNNAMED DatetimeIndex, which
    # its column-0 index heuristic would misread as the value column (fix: market-proxy ffill).
    s = s[~s.index.duplicated(keep="last")].sort_index()
    real = s.dropna()
    last_obs = real.index.max() if not real.empty else None
    n_extra = _extrapolated_after(dates, last_obs, f"market_proxy[{suffix}:{col}]")
    aligned = s.reindex(s.index.union(d)).ffill().reindex(d)
    # nan_to_num now only catches a genuine LEADING gap (a target date before the first observation).
    return MarketProxyResult(
        np.nan_to_num(aligned.to_numpy(dtype=float), nan=0.0), available=True, column=col,
        last_observation=(str(last_obs)[:10] if last_obs is not None else None),
        n_extrapolated=n_extra,
    )


#: The ``.SPXTR`` history and its 2026 extension. Unlike :data:`_REFRESHED_RAW`, where a refresh
#: REPLACES the canonical file, these two are CONCATENATED: the base pull ends 2025-12-31 and the
#: ``_x26`` pull carries 2026-01-02 → 2026-06-30, so the sealed window (2020-03-30 → 2026-06-30) is
#: only covered by both together. Using either alone silently truncates the test window.
_SPXTR_CSVS: tuple[str, ...] = ("rf_spxtr.csv", "rf_spxtr_x26.csv")
#: The Refinitiv total-return LEVEL column in those files (Date, TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, …).
_SPXTR_LEVEL_COL = "TRDPRC_1"


def load_spx_total_return(
    dates: np.ndarray, *, raw_dir: Path | str = _RAW_DIR
) -> MarketProxyResult:
    """Per-session **S&P 500 TOTAL-RETURN** decimal returns (``.SPXTR``) aligned to ``dates``.

    The cap-weighted market line — what a reader means by "the market" — as distinct from
    :func:`load_market_proxy_returns`, which is the EQUAL-WEIGHT return of our own universe. Over
    2020-2026 the two differ materially (equal weight tilts small and rebalances), so they are
    reported side by side rather than substituted for one another.

    **The files store a LEVEL, not returns**, which decides the alignment order. The level is
    forward-filled onto the panel's session axis FIRST and the return is differenced from the aligned
    levels SECOND. Doing it the other way round — differencing on the source axis and then
    forward-filling the RETURNS — would repeat a return on any session the index did not publish, i.e.
    it would book the same move twice. With this order a non-publication session correctly yields
    0.0 and every panel-date return is the true change since the previous panel date.

    Never reads the future: the forward-fill only ever carries values from the past, the leading gap
    (a panel date before the first observation) yields 0.0, and ``n_extrapolated`` reports how many
    target sessions fall beyond the last REAL observation, exactly as the RF/FF/market loaders do.

    Degrades gracefully to zeros with ``available=False`` when the raw files are absent (a
    synthetic-only install; the ``.SPXTR`` pull is licensed Refinitiv data and is not in git).
    """
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    n = int(np.asarray(dates).size)

    frames: list[pd.DataFrame] = []
    for name in _SPXTR_CSVS:
        path = Path(raw_dir) / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return MarketProxyResult(np.zeros(n), available=False, column=_SPXTR_LEVEL_COL)

    raw = pd.concat(frames, ignore_index=True)
    if _SPXTR_LEVEL_COL not in raw.columns or "Date" not in raw.columns:
        return MarketProxyResult(np.zeros(n), available=False, column=_SPXTR_LEVEL_COL)

    level = pd.to_numeric(raw[_SPXTR_LEVEL_COL], errors="coerce")
    level.index = pd.DatetimeIndex(pd.to_datetime(raw["Date"]))
    # The base file and the _x26 extension can overlap at a boundary session; keep the LAST reading,
    # matching the convention every other loader here uses.
    level = level[~level.index.duplicated(keep="last")].sort_index()

    real = level.dropna()
    last_obs = real.index.max() if not real.empty else None
    n_extra = _extrapolated_after(dates, last_obs, f"spxtr[{_SPXTR_LEVEL_COL}]")

    aligned_level = level.reindex(level.index.union(d)).ffill().reindex(d)
    rets = aligned_level.pct_change()
    return MarketProxyResult(
        np.nan_to_num(rets.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        available=True, column=_SPXTR_LEVEL_COL,
        last_observation=(str(last_obs)[:10] if last_obs is not None else None),
        n_extrapolated=n_extra,
    )


def load_ff_factors(dates: np.ndarray, *, raw_dir: Path | str = _RAW_DIR) -> FactorResult:
    """Per-session Fama-French factor DECIMALS (Mkt-RF/SMB/HML) aligned to ``dates`` for attribution."""
    path = _raw_path(raw_dir, _FF_CSV)
    if not path.exists():
        return FactorResult({}, available=False)
    ff = pd.read_csv(path)
    date_col = "Date" if "Date" in ff.columns else ff.columns[0]
    ff = ff.set_index(date_col)
    out: dict[str, np.ndarray] = {}
    last_seen: pd.Timestamp | None = None
    for col in ("Mkt-RF", "SMB", "HML"):
        if col in ff.columns:
            vals, last_obs = _aligned_series(ff, col, dates)
            out[col] = np.nan_to_num(vals, nan=0.0)  # already decimals in the French daily file
            # the EARLIEST last-observation across columns governs: a factor set is only as fresh as
            # its stalest member, since the regression uses them jointly
            if last_obs is not None and (last_seen is None or last_obs < last_seen):
                last_seen = last_obs
    if not out:
        return FactorResult({}, available=False)
    n_extra = _extrapolated_after(dates, last_seen, "ff_factors[Mkt-RF/SMB/HML]")
    return FactorResult(
        out, available=True,
        last_observation=(str(last_seen)[:10] if last_seen is not None else None),
        n_extrapolated=n_extra,
    )
