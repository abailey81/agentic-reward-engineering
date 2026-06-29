"""Synthetic panel generator with realistic stylised facts.

Produces a :class:`~src.data.panel.Panel` of identical shape to the gold data, exhibiting the
stylised facts the dissertation's tail premise rests on (Cont 2001): **heavy tails**
(Student-t innovations), **volatility clustering** (a GARCH(1,1)-style recursion), a common
**market factor**, and a **VIX-like** index driven by cross-sectional volatility.

It exists so the entire system — environment, pipeline, measurement, the full loop — runs
end-to-end and is testable **without the licensed Refinitiv/LSEG data**, and so the repository can ship a
runnable synthetic panel in place of data it may not redistribute (audit C-3).
"""
from __future__ import annotations

import numpy as np

from src.data.panel import Panel

__all__ = ["make_synthetic_panel"]


def make_synthetic_panel(
    n_assets: int = 30,
    n_days: int = 2_000,
    *,
    seed: int = 0,
    df: float = 4.0,
    market_vol: float = 0.008,
    idio_vol: float = 0.012,
    garch_alpha: float = 0.08,
    garch_beta: float = 0.90,
) -> Panel:
    """Generate a synthetic, heavy-tailed, vol-clustering market panel.

    Parameters
    ----------
    n_assets, n_days : int
        Panel dimensions ``(n_days, n_assets)``.
    seed : int
        RNG seed (reproducible).
    df : float
        Degrees of freedom of the Student-t innovations (``df`` small => heavier tails).
    market_vol, idio_vol : float
        Baseline daily volatilities of the common factor and the idiosyncratic component.
    garch_alpha, garch_beta : float
        GARCH(1,1) parameters for the conditional-variance recursion (``alpha + beta < 1``
        for stationarity). Drive volatility clustering.

    Returns
    -------
    Panel
        ``returns`` (T, N), ``vix`` (T,), ``dates`` (T,), ``asset_ids`` (N,), ``market_caps`` (T, N).
    """
    if garch_alpha + garch_beta >= 1.0:
        raise ValueError("garch_alpha + garch_beta must be < 1 for a stationary variance process")
    # df must exceed 2 so the Student-t has finite variance; otherwise the unit-variance rescale
    # below (sqrt(df / (df - 2))) divides by zero (df==2) or takes sqrt of a negative (df<2),
    # producing inf/NaN that only surfaces much later as a misleading finiteness error in Panel.
    if df <= 2.0:
        raise ValueError(f"df must be > 2 for a finite-variance Student-t (got {df})")
    rng = np.random.default_rng(seed)

    def _garch_sigma(base_vol: float) -> tuple[np.ndarray, np.ndarray]:
        omega = base_vol**2 * (1.0 - garch_alpha - garch_beta)
        var = np.empty(n_days)
        var[0] = base_vol**2
        shock = rng.standard_t(df, size=n_days) / np.sqrt(df / (df - 2.0))  # unit-variance t
        for t in range(1, n_days):
            var[t] = omega + garch_alpha * (var[t - 1] * shock[t - 1] ** 2) + garch_beta * var[t - 1]
        return np.sqrt(var), shock

    # Common market factor (clustered, heavy-tailed).
    mkt_sigma, mkt_shock = _garch_sigma(market_vol)
    market = mkt_sigma * mkt_shock + 0.0003  # small positive drift

    # Idiosyncratic component per asset, each with its own clustered variance, plus a beta to market.
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    returns = np.empty((n_days, n_assets))
    for j in range(n_assets):
        idio_sigma, idio_shock = _garch_sigma(idio_vol)
        returns[:, j] = betas[j] * market + idio_sigma * idio_shock

    # VIX-like index: annualised cross-sectional dispersion, persistent; consumers must lag it.
    cross_vol = returns.std(axis=1)
    vix = 100.0 * np.sqrt(252.0) * _ewma(cross_vol, halflife=10.0)

    # Market caps: random-walk in log, used by point-in-time top-N selection.
    log_caps = np.log(rng.uniform(5e9, 2e12, size=n_assets))[None, :] + np.cumsum(
        rng.normal(0.0, 0.01, size=(n_days, n_assets)), axis=0
    )
    market_caps = np.exp(log_caps)

    dates = np.arange("2005-01-03", n_days, dtype="datetime64[D]")
    asset_ids = np.arange(n_assets, dtype=np.int64)
    return Panel(
        returns=returns.astype(np.float64),
        vix=vix.astype(np.float64),
        dates=dates,
        asset_ids=asset_ids,
        market_caps=market_caps.astype(np.float64),
    )


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially-weighted moving average (causal)."""
    lam = 0.5 ** (1.0 / halflife)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = lam * out[t - 1] + (1.0 - lam) * x[t]
    return out
