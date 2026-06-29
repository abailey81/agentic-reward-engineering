"""Leakage-safe cash-row state features (Sood et al. 2023 tail: vol20, vol20/vol60, VIX).

Closes the marked gap in `src/portfolio_env.py` (ADR-007).

Leakage posture (R3): every feature at row t is a function of information available
STRICTLY BEFORE the decision at t — rolling statistics are computed on returns through
t-1 (explicit shift(1)), and the VIX column uses the close of t-1. Two invariances are
unit-tested (tests/test_features.py):

  1. truncation invariance — features[:m] computed on the full series equal features
     computed on the series truncated at m;
  2. future-perturbation invariance — changing inputs at rows >= t leaves rows <= t
     unchanged.

Design decisions (ADR-007):
  * market-return proxy = equal-weighted mean across the universe's columns (no extra
    data dependency; deterministic from the env's own return matrix); an explicit
    `market_returns` series (e.g. the index) may be passed instead — same API.
  * VIX is a REQUIRED argument: callers without it must pass an explicit placeholder
    (e.g. zeros) so the absence of the feature is a visible decision, never a silent
    default. Scaled by config `environment.state.vix_scale` (stateless, leakage-free).
  * volatilities are SAMPLE std (ddof=1) of daily returns, unannualised; the ratio is
    1.0 (neutral) when the long-window vol is ~0 (zero-variance guard).
  * warm-up rows (insufficient history) are NaN by construction; PortfolioEnv refuses
    non-finite features on or after its first decision row (fail loudly, R4 spirit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import get


def rolling_vol_shifted(returns_1d: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample std (ddof=1) over `window` past observations, shifted so row t
    uses returns through t-1 only. Rows with fewer than `window` prior obs are NaN."""
    if window < 2:
        raise ValueError("window must be >= 2 for a sample std")
    s = pd.Series(np.asarray(returns_1d, dtype=float))
    return s.rolling(window, min_periods=window).std(ddof=1).shift(1).to_numpy()


def build_cash_features(
    asset_returns: np.ndarray,
    vix: np.ndarray,
    market_returns: np.ndarray | None = None,
    vol_short: int | None = None,
    vol_long: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build the (T, 3) cash-row feature block [vol20, vol20/vol60, vix_scaled].

    Args:
        asset_returns: (T, n_assets) simple daily returns (the env's own matrix).
        vix: (T,) VIX CLOSE levels aligned to the same rows; row t may hold day t's
            close — it is shifted internally so the feature at t is the t-1 close.
            Pass an explicit zeros array if VIX is unavailable (visible decision).
        market_returns: optional (T,) series overriding the equal-weight proxy.
    Returns:
        (T, 3) float array; warm-up rows NaN. Column order follows
        config `environment.state.cash_features`.
    """
    r = np.asarray(asset_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("asset_returns must be (T, n_assets)")
    t_len = r.shape[0]
    vix_arr = np.asarray(vix, dtype=float).ravel()
    if vix_arr.shape[0] != t_len:
        raise ValueError("vix must align row-for-row with asset_returns")

    if market_returns is not None:
        mkt = np.asarray(market_returns, dtype=float).ravel()
        if mkt.shape[0] != t_len:
            raise ValueError("market_returns must align row-for-row with asset_returns")
    else:
        # NaN-SAFE equal-weight market proxy (2026-07-05 fix): on the full research panel virtually
        # every row carries pre-IPO/post-delisting NaNs, so the plain mean() propagated them and the
        # derived vol columns in the gold cash_features came out 100% NaN (verified on univ5; harmless
        # to the campaign — the engine reads only the vix column — but wrong for any future consumer).
        with np.errstate(invalid="ignore"):
            mkt = np.nanmean(r, axis=1)  # equal-weight market proxy over the LIVE names (ADR-007)

    vol_short = int(vol_short if vol_short is not None else get("environment.state.vol_short_window"))
    vol_long = int(vol_long if vol_long is not None else get("environment.state.vol_long_window"))
    if vol_short >= vol_long:
        raise ValueError("vol_short_window must be < vol_long_window")

    v_s = rolling_vol_shifted(mkt, vol_short)
    v_l = rolling_vol_shifted(mkt, vol_long)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(v_l > eps, v_s / np.where(v_l > eps, v_l, 1.0), 1.0)
    ratio = np.where(np.isnan(v_l) | np.isnan(v_s), np.nan, ratio)  # warm-up stays NaN

    vix_scale = float(get("environment.state.vix_scale"))
    vix_shifted = np.concatenate([[np.nan], vix_arr[:-1]]) / vix_scale  # t uses close of t-1

    return np.column_stack([v_s, ratio, vix_shifted])
