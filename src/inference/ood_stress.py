"""OOD tail-stress harness (robustness appendix; deep-research §5.12). NEVER headline.

What this is — and the hard over-claim guard
---------------------------------------------
A standalone generator of **synthetic stressed return paths** for re-evaluating FROZEN
campaign winners. The protocol (deep-research §5.12): freeze the winners, roll the frozen
policy through synthetic *test*-length paths, score with the same per-seed rliable tail
metrics, and **report this only as a robustness appendix, never as the headline**.

The decisive caveat, stated up front because it bounds every claim this module can make:
**the generators are calibrated to the SAME history the winners were selected on.** This
makes the exercise a **falsification / stress-probe, not out-of-sample generalisation** — it
can *break* a winner (show its tail edge was fragile to a plausible re-draw of the dynamics)
but it cannot *certify* generalisation to genuinely unseen futures. No single generator
reproduces all stylised facts at once, so each stressed path set must pass a **validation
battery** (:func:`validate_stylized_facts`) before it is used, and Sharpe + drawdown are
reported alongside every tail metric so a "tail win" cannot be tautological. The Bauer (2025)
low-power caveat applies to the CVaR-1% leg.

Tier-1 stressors (no new deps — ``arch`` 7.x + ``statsmodels`` 0.14.x + ``scipy.genpareto``,
all pinned in ``pyproject.toml``):

* :func:`garch_evt_fhs` — **GARCH-EVT filtered historical simulation** (McNeil & Frey 2000,
  *J. Empirical Finance* 7:271-300). Per asset: fit an AR(1)-GJR/EGARCH-t volatility filter,
  standardise to (approximately) i.i.d. residuals, fit a GPD/POT tail to the standardised
  residuals, then simulate by re-inflating innovations through each asset's GARCH recursion.
  **Mode caveat (the two innovation paths differ on the tail):** in the **default
  ``preserve_cross_section=True``** path the innovations are **empirical residual ROWS resampled
  jointly across assets** (preserving the cross-sectional copula so portfolio tails are real) —
  the **GPD/EVT tail extension is NOT applied on this path**, so a default-mode path **cannot
  exceed the empirical residual tail support**. The semi-parametric GPD tail extension (drawing
  residuals beyond the empirical support from the fitted GPD) runs **only when
  ``preserve_cross_section=False``**, where each asset's marginal is drawn independently from its
  own GPD-extended sampler (diagnostic only — it breaks the copula). So "EVT tail extension" and
  "copula-preserving" are mutually exclusive here; the headline/default mode keeps the copula and
  forgoes the tail extension.
* :func:`block_bootstrap_paths` — **stationary block bootstrap** of the panel rows jointly
  (Politis & Romano 1994 via :func:`src.inference.bootstrap.stationary_bootstrap_indices`;
  block length from ``arch.bootstrap.optimal_block_length``, Politis-White / PPW-2009). The
  most assumption-light stressor: it only re-orders observed rows, so it cannot invent a
  crash larger than history but faithfully preserves the contemporaneous cross-section.
* :func:`markov_crash_paths` — **Markov-switching crash regimes** via
  ``statsmodels.MarkovAutoregression`` on a market aggregate. Uses **FILTERED** (info up to
  ``t``) state inference, NEVER smoothed (smoothed conditions on the future and leaks),
  honouring the OOS no-look-ahead rule. Simulates a regime path from the fitted transition
  matrix and draws returns from each regime's conditional law.
* :func:`vol_spike_paths` — parametric, **mean-preserving** variance scaling (1.5x / 2x) of
  the panel: a cheap deterministic stressor for a coarse sensitivity sweep.

Scope-out (wrong layer / same-path objection): ABIDES / JAX-LOB microstructure sims (daily
rebalancing is the wrong granularity) and diffusion/GFlowNet generators (same fatal
objection: calibrated to the same path). TAIL-GAN (Cont et al. 2025) is a hard-caveated Tier-2
stretch, intentionally NOT implemented here.

Degrades gracefully
-------------------
Every generator validates its inputs and raises a clear ``ValueError`` on malformed panels,
but the public scoring helpers (:func:`tail_metrics`, :func:`score_paths`) return structured
dicts and never fabricate. The harness needs only a numeric ``(T, n_assets)`` return panel —
no campaign artefacts — so it runs on the synthetic panel when real gold is absent. The
env/agent roll-out wiring is specified with the arm roster (this module
deliberately does NOT import the env or the agent, to stay dependency-light and Door-C clean).

References
----------
McNeil & Frey (2000) *J. Empirical Finance* 7:271-300; Politis & Romano (1994); Politis &
White (2004) / Patton-Politis-White (2009) corrected block length; Hamilton (1989) regime
switching; Glosten-Jagannathan-Runkle (1993) GJR-GARCH; Bauer (2025) tail-power caveat.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable

import numpy as np
from scipy import stats

from src.inference.bootstrap import cvar, sharpe_ratio, stationary_bootstrap_indices

__all__ = [
    "garch_evt_fhs",
    "block_bootstrap_paths",
    "markov_crash_paths",
    "vol_spike_paths",
    "tail_metrics",
    "score_paths",
    "validate_stylized_facts",
    "claims",
]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_panel(returns: np.ndarray) -> np.ndarray:
    """Coerce and validate a ``(T, n_assets)`` finite return panel; raise on malformed input."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"returns must be (T, n_assets); got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("returns must have >= 2 time steps")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values")
    return arr


# ---------------------------------------------------------------------------
# (a) GARCH-EVT filtered historical simulation (McNeil-Frey 2000)
# ---------------------------------------------------------------------------
def _fit_garch_filter(
    series: np.ndarray, *, p: int = 1, o: int = 1, q: int = 1
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit an AR(1)-GJR-GARCH(1,1)-t filter to one return series.

    Returns ``(std_resid, cond_vol, fitted_result)``: the standardised residuals
    ``e_t = a_t / sigma_t`` (approximately i.i.d., the EVT input), the conditional volatility
    path, and the fitted ``arch`` result (carrying parameters for simulation). Returns are
    rescaled to ~unit scale internally for the optimiser's benefit and rescaled back by the
    caller via ``cond_vol``.
    """
    from arch.univariate import arch_model

    x = np.asarray(series, dtype=float).ravel()
    scale = float(np.std(x)) or 1.0
    xs = x / scale  # arch's optimiser prefers O(1) data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(
            xs, mean="AR", lags=1, vol="GARCH", p=p, o=o, q=q, dist="studentst"
        )
        res = am.fit(disp="off", show_warning=False)
    std_resid = np.asarray(res.std_resid, dtype=float)
    cond_vol = np.asarray(res.conditional_volatility, dtype=float) * scale
    # arch may drop the first ``lags`` observations from resid arrays; align by trimming NaNs.
    good = np.isfinite(std_resid)
    return std_resid[good], cond_vol[good], res


def _gpd_tail_sampler(
    std_resid: np.ndarray, *, threshold_q: float = 0.10, rng: np.random.Generator
) -> Callable[[int], np.ndarray]:
    """Semi-parametric residual sampler: empirical body + GPD tails (both sides).

    Builds a sampler that, given ``n``, returns ``n`` draws from the standardised-residual law:
    a bootstrap of the empirical residuals, with draws that land in the lower/upper tail mass
    replaced by GPD-extrapolated exceedances (so simulated extremes can exceed the empirical
    support — the EVT contribution to FHS). Falls back to a pure empirical bootstrap if a tail
    GPD fit is degenerate.
    """
    e = np.asarray(std_resid, dtype=float)
    n_e = e.size
    lo_u = float(np.quantile(e, threshold_q))
    hi_u = float(np.quantile(e, 1.0 - threshold_q))

    def _fit_side(exceed: np.ndarray) -> tuple[float, float] | None:
        if exceed.size < 5:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)
            except Exception:  # pragma: no cover - optimiser failure is rare
                return None
        if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
            return None
        return float(xi), float(beta)

    lower_exceed = lo_u - e[e < lo_u]  # positive exceedances below the lower threshold
    upper_exceed = e[e > hi_u] - hi_u
    lower_par = _fit_side(lower_exceed)
    upper_par = _fit_side(upper_exceed)
    p_lower = float(np.mean(e < lo_u))
    p_upper = float(np.mean(e > hi_u))

    def sample(n: int) -> np.ndarray:
        out = e[rng.integers(0, n_e, size=n)].copy()
        if lower_par is not None:
            mask = rng.random(n) < p_lower
            k = int(mask.sum())
            if k:
                xi, beta = lower_par
                exc = stats.genpareto.rvs(xi, loc=0.0, scale=beta, size=k, random_state=rng)
                out[mask] = lo_u - exc
        if upper_par is not None:
            mask = rng.random(n) < p_upper
            k = int(mask.sum())
            if k:
                xi, beta = upper_par
                exc = stats.genpareto.rvs(xi, loc=0.0, scale=beta, size=k, random_state=rng)
                out[mask] = hi_u + exc
        return out

    return sample


def garch_evt_fhs(
    returns: np.ndarray,
    *,
    n_paths: int = 100,
    horizon: int | None = None,
    threshold_q: float = 0.10,
    preserve_cross_section: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GARCH-EVT filtered historical simulation (McNeil-Frey 2000); cross-section preserved.

    Per asset: fit an AR(1)-GJR-GARCH(1,1)-t filter, extract standardised residuals, and build a
    semi-parametric residual sampler (empirical body + GPD tails). Then simulate ``n_paths``
    return paths of length ``horizon`` by drawing standardised innovations and re-inflating
    through each asset's fitted GARCH recursion.

    **The two innovation paths differ on the tail (read before interpreting a tail result):**

    * ``preserve_cross_section=True`` (the **default** and the copula-correct choice) draws the SAME
      empirical residual-row index across all assets per step (a joint row resample), preserving the
      empirical cross-sectional copula so PORTFOLIO tails are real rather than diversified away by
      independent per-asset draws. On this path the innovations are **empirical residual rows only —
      the per-asset GPD/EVT tail samplers are NOT used**, so a default-mode path **cannot exceed the
      empirical residual tail support** (it re-orders/co-draws observed residuals; it does not
      synthesise a residual larger than history).
    * ``preserve_cross_section=False`` draws each asset's innovations **independently from its own
      GPD-extended sampler**, which DOES synthesise residuals beyond the empirical support (the
      semi-parametric EVT tail extension) — but this **breaks the cross-sectional copula** and is a
      marginal-only diagnostic, not the headline mode.

    In short: the GPD/EVT tail extension and the copula preservation are mutually exclusive here;
    the default keeps the copula and forgoes the tail extension.

    Parameters
    ----------
    returns : np.ndarray
        ``(T, n_assets)`` historical return panel (the calibration data).
    n_paths : int
        Number of synthetic paths to generate.
    horizon : int, optional
        Length of each path (default: the input ``T``).
    threshold_q : float
        Lower/upper tail-mass fraction defining the GPD POT thresholds on the residuals.
    preserve_cross_section : bool
        If True, resample residual rows jointly across assets (preserve the copula). If False,
        resample each asset's residuals independently (diagnostic only — understates tail
        co-movement).
    rng : numpy.random.Generator, optional
        Seeded for reproducibility.

    Returns
    -------
    np.ndarray
        ``(n_paths, horizon, n_assets)`` stressed return paths.

    Notes
    -----
    The GARCH recursion is re-run forward with simulated innovations from each asset's fitted
    GJR-GARCH-t parameters (omega, alpha, gamma, beta), seeded at the last in-sample conditional
    variance — the standard McNeil-Frey forward filter. If ``arch`` is unavailable or a fit
    fails, the asset falls back to an i.i.d. draw from its empirical residual sampler at constant
    (unconditional) volatility, logged at WARNING — the path set is still produced (graceful
    degradation) but is flagged less realistic.
    """
    panel = _validate_panel(returns)
    T, n_assets = panel.shape
    H = int(horizon) if horizon is not None else T
    if H < 1:
        raise ValueError("horizon must be >= 1")
    if rng is None:
        rng = np.random.default_rng(0)

    # Fit per-asset filters + residual samplers.
    samplers: list[Callable[[int], np.ndarray]] = []
    params: list[dict[str, float] | None] = []
    resid_mat_cols: list[np.ndarray] = []
    for j in range(n_assets):
        try:
            std_resid, _cond_vol, res = _fit_garch_filter(panel[:, j])
            # arch exposes the conditional VOLATILITY (sigma), in the scaled units the model was
            # fit on (returns / std). Square the last value for the recursion's seed variance.
            cvol = np.asarray(res.conditional_volatility, dtype=float)
            cvol = cvol[np.isfinite(cvol)]
            sigma2_last = float(cvol[-1] ** 2) if cvol.size else 1.0
            pj = {
                "omega": float(res.params.get("omega", 0.0)),
                "alpha": float(res.params.get("alpha[1]", 0.0)),
                "gamma": float(res.params.get("gamma[1]", 0.0)),
                "beta": float(res.params.get("beta[1]", 0.0)),
                "mu": float(res.params.get("Const", res.params.get("mu", 0.0))),
                "scale": float(np.std(panel[:, j])) or 1.0,
                "sigma2_last": sigma2_last,
            }
            params.append(pj)
        except Exception as exc:  # pragma: no cover - depends on arch optimiser
            _LOG.warning("GARCH fit failed for asset %d (%s); falling back to i.i.d. residuals", j, exc)
            mu = float(panel[:, j].mean())
            sd = float(panel[:, j].std()) or 1.0
            std_resid = (panel[:, j] - mu) / sd
            params.append(None)
        resid_mat_cols.append(std_resid)
        samplers.append(_gpd_tail_sampler(std_resid, threshold_q=threshold_q, rng=rng))

    # For the cross-section-preserving path we need an aligned residual matrix (common length).
    # The empirical residual ROWS are kept intact to preserve the copula; the per-asset GPD
    # samplers (used only when ``preserve_cross_section`` is False) extend the marginal tails.
    t_min = min(c.size for c in resid_mat_cols)
    resid_mat = np.column_stack([c[-t_min:] for c in resid_mat_cols])  # (t_min, n_assets)

    out = np.empty((n_paths, H, n_assets), dtype=float)
    for p in range(n_paths):
        if preserve_cross_section:
            rows = rng.integers(0, t_min, size=H)
            innov = resid_mat[rows, :]  # (H, n_assets) joint residual rows -> copula preserved
        else:
            innov = np.column_stack([samplers[j](H) for j in range(n_assets)])
        for j in range(n_assets):
            par = params[j]
            col = innov[:, j]
            if par is None:
                # i.i.d. fallback at unconditional scale.
                out[p, :, j] = panel[:, j].mean() + (panel[:, j].std() or 1.0) * col
                continue
            # Forward GJR-GARCH-t recursion on standardised innovations (scaled units).
            omega, alpha, gamma, beta = par["omega"], par["alpha"], par["gamma"], par["beta"]
            sigma2 = par["sigma2_last"]
            scale, mu = par["scale"], par["mu"]
            a_prev = 0.0
            path = np.empty(H, dtype=float)
            for t in range(H):
                neg = 1.0 if a_prev < 0.0 else 0.0
                sigma2 = omega + (alpha + gamma * neg) * a_prev * a_prev + beta * sigma2
                sigma2 = max(sigma2, 1e-12)
                a_t = math.sqrt(sigma2) * col[t]
                path[t] = mu + a_t
                a_prev = a_t
            out[p, :, j] = path * scale
    return out


# ---------------------------------------------------------------------------
# (b) Stationary block bootstrap of the panel rows (jointly)
# ---------------------------------------------------------------------------
def optimal_block_length(returns: np.ndarray) -> float:
    """Politis-White / PPW-2009 optimal stationary-bootstrap block length for a panel.

    Uses ``arch.bootstrap.optimal_block_length`` on the equally weighted portfolio return (a
    scalar summary of the panel's serial dependence). Returns the stationary-bootstrap mean
    block length; the block-restart probability is ``p = 1 / block_length``. Falls back to a
    heuristic ``T ** (1/3)`` if ``arch`` errors.
    """
    panel = _validate_panel(returns)
    port = panel.mean(axis=1)
    try:
        from arch.bootstrap import optimal_block_length as _obl

        obl = _obl(port)
        # arch returns a DataFrame with 'stationary' / 'circular' columns.
        val = float(np.asarray(obl["stationary"]).ravel()[0])
        if np.isfinite(val) and val >= 1.0:
            return val
    except Exception as exc:  # pragma: no cover
        _LOG.warning("optimal_block_length fell back to heuristic (%s)", exc)
    return max(1.0, float(panel.shape[0]) ** (1.0 / 3.0))


def block_bootstrap_paths(
    returns: np.ndarray,
    *,
    n_paths: int = 100,
    horizon: int | None = None,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stationary block bootstrap of panel ROWS jointly (preserves the cross-section).

    Resamples whole rows of the panel with a stationary-bootstrap (geometric, wrap-around) index
    path, so the contemporaneous cross-asset structure of each kept row is intact and the
    short-range serial dependence is preserved. The most assumption-light stressor — it cannot
    fabricate an unobserved extreme, but it re-orders history without breaking the copula.

    Parameters mirror :func:`garch_evt_fhs`; ``block_length`` defaults to
    :func:`optimal_block_length`. Returns ``(n_paths, horizon, n_assets)``.
    """
    panel = _validate_panel(returns)
    T, n_assets = panel.shape
    H = int(horizon) if horizon is not None else T
    if rng is None:
        rng = np.random.default_rng(0)
    bl = float(block_length) if block_length is not None else optimal_block_length(panel)
    p = min(1.0, max(1.0 / max(bl, 1.0), 1.0 / T))

    out = np.empty((n_paths, H, n_assets), dtype=float)
    for i in range(n_paths):
        idx = stationary_bootstrap_indices(T, p, rng)
        # Tile/trim the index path to the requested horizon.
        if H <= T:
            sel = idx[:H]
        else:
            reps = int(np.ceil(H / T))
            sel = np.concatenate([stationary_bootstrap_indices(T, p, rng) for _ in range(reps)])[:H]
        out[i] = panel[sel, :]
    return out


# ---------------------------------------------------------------------------
# (c) Markov-switching crash regimes (filtered states; no look-ahead)
# ---------------------------------------------------------------------------
def markov_crash_paths(
    returns: np.ndarray,
    *,
    n_paths: int = 100,
    horizon: int | None = None,
    k_regimes: int = 2,
    order: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Markov-switching crash-regime simulation on the market aggregate (FILTERED states).

    Fits ``statsmodels.MarkovAutoregression`` (switching mean + variance) to the equally
    weighted market return, then SIMULATES a regime path from the estimated transition matrix and
    draws market returns from each regime's conditional Gaussian law. Per-asset paths are formed
    by scaling the simulated market shock with each asset's market beta plus an idiosyncratic
    residual draw (a single-factor decomposition), so a "crash regime" hits the whole panel
    coherently.

    CRITICAL: the regime inference used to seed the simulation is the **FILTERED** marginal
    probability (information up to ``t`` only). Smoothed probabilities condition on the *entire*
    sample (future included) and would leak look-ahead into an OOS stress — they are never used.

    Returns a dict ``{"status", "paths" (n_paths, horizon, n_assets), "transition", "regime_means",
    "regime_vars", "n_regimes"}``. On a fit failure returns ``{"status": "fit_failed", "reason":
    ...}`` (graceful degradation) rather than raising — the caller can fall back to the block
    bootstrap.
    """
    panel = _validate_panel(returns)
    T, n_assets = panel.shape
    H = int(horizon) if horizon is not None else T
    if rng is None:
        rng = np.random.default_rng(0)
    market = panel.mean(axis=1)

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MarkovAutoregression(
                market, k_regimes=k_regimes, order=order, switching_variance=True
            )
            res = mod.fit(disp=False)
    except Exception as exc:
        return {"status": "fit_failed", "reason": f"MarkovAutoregression fit failed: {exc}"}

    # Transition matrix: statsmodels stores regime_transition as (k, k, [t]); take the
    # time-invariant slice. Column j -> distribution over next state from state j.
    trans = np.asarray(res.regime_transition)
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    trans = np.atleast_2d(trans).astype(float)
    # Normalise columns to proper transition probabilities (guard numerical drift).
    trans = np.clip(trans, 0.0, None)
    colsum = trans.sum(axis=0, keepdims=True)
    colsum[colsum == 0] = 1.0
    trans = trans / colsum

    # Per-regime conditional mean / variance.
    params = res.params
    means = np.array(
        [float(params.get(f"const[{k}]", market.mean())) for k in range(k_regimes)]
    )
    sig2 = np.array(
        [float(params.get(f"sigma2[{k}]", market.var())) for k in range(k_regimes)]
    )
    sig2 = np.where(np.isfinite(sig2) & (sig2 > 0), sig2, market.var())

    # Seed the simulated regime from the LAST filtered state (info up to T only -> no look-ahead).
    try:
        filt = np.asarray(res.filtered_marginal_probabilities)
        last_filt = filt[-1] if filt.ndim == 2 else filt[:, -1]
        start_state = int(np.argmax(last_filt))
    except Exception:  # pragma: no cover
        start_state = 0

    # Single-factor decomposition for per-asset paths.
    beta = np.empty(n_assets)
    resid_sd = np.empty(n_assets)
    mvar = float(np.var(market)) or 1.0
    for j in range(n_assets):
        cov = float(np.cov(panel[:, j], market)[0, 1])
        beta[j] = cov / mvar
        fitted = beta[j] * (market - market.mean())
        resid_sd[j] = float(np.std(panel[:, j] - panel[:, j].mean() - fitted)) or 1e-8
    asset_mean = panel.mean(axis=0)

    states_grid = np.arange(k_regimes)
    out = np.empty((n_paths, H, n_assets), dtype=float)
    for p in range(n_paths):
        state = start_state
        for t in range(H):
            shock = means[state] + math.sqrt(sig2[state]) * rng.standard_normal()
            mkt_dev = shock - market.mean()
            out[p, t, :] = asset_mean + beta * mkt_dev + resid_sd * rng.standard_normal(n_assets)
            # advance the regime using column `state` of the transition matrix
            state = int(rng.choice(states_grid, p=trans[:, state]))

    return {
        "status": "ok",
        "paths": out,
        "transition": trans,
        "regime_means": means,
        "regime_vars": sig2,
        "n_regimes": int(k_regimes),
        "start_state": int(start_state),
    }


# ---------------------------------------------------------------------------
# (d) Parametric mean-preserving vol-spike injection
# ---------------------------------------------------------------------------
def vol_spike_paths(
    returns: np.ndarray,
    *,
    multiplier: float = 2.0,
    n_paths: int = 1,
    horizon: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mean-preserving variance scaling of the panel (a coarse deterministic stressor).

    Inflates each asset's deviations-from-mean by ``sqrt(multiplier)`` (variance x ``multiplier``)
    while preserving the per-asset mean and the exact cross-sectional correlation. With
    ``n_paths == 1`` and ``horizon == T`` this is the deterministic re-scaling of the historical
    panel; with ``n_paths > 1`` it block-resamples rows first (so paths differ) then inflates.
    Returns ``(n_paths, horizon, n_assets)``.
    """
    panel = _validate_panel(returns)
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    T, n_assets = panel.shape
    H = int(horizon) if horizon is not None else T
    if rng is None:
        rng = np.random.default_rng(0)
    mu = panel.mean(axis=0)
    factor = math.sqrt(multiplier)
    if n_paths == 1 and H == T:
        return (mu + factor * (panel - mu))[None, :, :]
    base = block_bootstrap_paths(panel, n_paths=n_paths, horizon=H, rng=rng)
    return mu + factor * (base - mu)


# ---------------------------------------------------------------------------
# Scoring + validation
# ---------------------------------------------------------------------------
def _portfolio_returns_equal_weight(paths: np.ndarray) -> np.ndarray:
    """Default policy: equal-weight portfolio per path. ``(n_paths, H, n_assets) -> (n_paths, H)``."""
    return paths.mean(axis=2)


def tail_metrics(
    port_returns: np.ndarray, *, cvar_levels: tuple[float, ...] = (0.05, 0.01)
) -> dict[str, Any]:
    """Per-path tail + performance metrics, aggregated across paths (rliable IQM + spread).

    ``port_returns`` is ``(n_paths, H)`` (one realised portfolio return path per simulation).
    Reuses :func:`src.inference.bootstrap.sharpe_ratio` / :func:`cvar`. Reports, per path, the
    Sharpe, the max drawdown, and the CVaR at each level; then aggregates each across paths to an
    IQM point estimate with a [5th, 95th] percentile band. Sharpe AND drawdown are reported
    alongside the tail so a tail "win" cannot be tautological (deep-research §5.12 over-claim
    guard). The CVaR-1% line carries a Bauer (2025) low-power flag.

    Returns ``{"n_paths", "sharpe": {...}, "max_drawdown": {...}, "cvar": {level: {...}},
    "cvar_01_power_caveat"}``.
    """
    from src.inference.bootstrap import iqm

    R = np.atleast_2d(np.asarray(port_returns, dtype=float))
    n_paths = R.shape[0]

    def _max_drawdown(r: np.ndarray) -> float:
        eq = np.cumsum(r)  # log-return cumulant proxy
        peak = np.maximum.accumulate(eq)
        return float(np.max(peak - eq)) if eq.size else 0.0

    def _agg(vals: np.ndarray) -> dict[str, float]:
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return {"iqm": float("nan"), "p05": float("nan"), "p95": float("nan"), "mean": float("nan")}
        return {
            "iqm": float(iqm(v)),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()),
        }

    sharpes = np.array([sharpe_ratio(R[i]) for i in range(n_paths)])
    drawdowns = np.array([_max_drawdown(R[i]) for i in range(n_paths)])
    cvar_out: dict[float, dict[str, float]] = {}
    for lvl in cvar_levels:
        cvar_out[float(lvl)] = _agg(np.array([cvar(R[i], float(lvl)) for i in range(n_paths)]))

    return {
        "n_paths": int(n_paths),
        "sharpe": _agg(sharpes),
        "max_drawdown": _agg(drawdowns),
        "cvar": cvar_out,
        "cvar_01_power_caveat": (
            "CVaR-1% under stress is high-variance and low-power (Bauer 2025); interpret with care."
        ),
    }


def score_paths(
    paths: np.ndarray,
    *,
    policy_returns: Callable[[np.ndarray], np.ndarray] | None = None,
    cvar_levels: tuple[float, ...] = (0.05, 0.01),
) -> dict[str, Any]:
    """Reduce a ``(n_paths, H, n_assets)`` path set to aggregate tail+performance metrics.

    ``policy_returns`` maps a path set to ``(n_paths, H)`` realised portfolio returns; it is the
    seam where a FROZEN winner's rolled-out policy is injected (the campaign supplies a closure
    that runs the env+agent on each synthetic path — see the wiring spec). When ``None``, an
    equal-weight portfolio is used as a placeholder so the harness self-tests without the agent
    (this is NOT a substitute for the real policy — it exists only so the module degrades
    gracefully). Returns the :func:`tail_metrics` dict.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim == 2:
        p = p[:, :, None]
    if p.ndim != 3:
        raise ValueError(f"paths must be (n_paths, H, n_assets); got {p.shape}")
    fn = policy_returns if policy_returns is not None else _portfolio_returns_equal_weight
    port = np.atleast_2d(fn(p))
    return tail_metrics(port, cvar_levels=cvar_levels)


def validate_stylized_facts(
    synthetic: np.ndarray,
    historical: np.ndarray,
    *,
    max_lag: int = 10,
    synthetic_is_paths: bool = True,
) -> dict[str, Any]:
    """Validation battery: do synthetic paths reproduce the key stylised facts of the history?

    A generated path set must clear this before it is used (deep-research §5.12: no generator
    reproduces all stylised facts at once, so check). Compares, on the equally weighted portfolio
    return of ``synthetic`` (``(n_paths, H, n_assets)`` or ``(n_paths, H)``) vs ``historical``
    (``(T, n_assets)`` or ``(T,)``):

    * **excess kurtosis** (fat tails) — synthetic should be heavy-tailed, not Gaussian;
    * **ACF of |returns|** (volatility clustering) — slowly-decaying positive autocorrelation;
    * **leverage effect** — ``corr(r_t, |r_{t+1}|) < 0`` (down-moves precede higher vol).

    Returns ``{"historical": {...}, "synthetic": {...}, "checks": {"fat_tails": bool,
    "vol_clustering": bool, "leverage": bool}, "passed": bool}``. The checks are *directional
    sanity gates*, not formal tests — a failed gate flags an unrealistic generator to disclose,
    not a quantitative claim.
    """
    def _to_port(x: np.ndarray, *, is_paths: bool) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.ndim == 3:  # (n_paths, H, n_assets) -> concatenate per-path EW returns
            return a.mean(axis=2).ravel()
        if a.ndim == 2:
            # A 2-D array is genuinely ambiguous: a HISTORICAL panel is (T, n_assets) -> EW mean over
            # axis 1, but a SYNTHETIC path-set is (n_paths, H) -> FLATTEN (each row is a return series).
            # The old `mean(axis=1) if shape[1] > 1` collapsed a (n_paths, H) path-set over TIME
            # (nonsense: per-path means, and a (1, H) set -> a single number -> all-NaN facts), so it
            # only worked for the 3-D synthetic. Disambiguate by the explicit `is_paths` hint
            # (2026-07-05 fix).
            if is_paths:
                return a.ravel()
            return a.mean(axis=1) if a.shape[1] > 1 else a.ravel()
        return a.ravel()

    def _facts(r: np.ndarray) -> dict[str, float]:
        r = r[np.isfinite(r)]
        if r.size < max_lag + 2:
            return {"excess_kurtosis": float("nan"), "abs_acf1": float("nan"), "leverage": float("nan")}
        dev = r - r.mean()
        absdev = np.abs(dev)
        # ACF(1) of |returns|.
        a0 = absdev[:-1] - absdev[:-1].mean()
        a1 = absdev[1:] - absdev[1:].mean()
        denom = math.sqrt(float(np.sum(a0**2) * np.sum(a1**2)))
        abs_acf1 = float(np.sum(a0 * a1) / denom) if denom > 0 else float("nan")
        # Leverage: corr(r_t, |r_{t+1}|).
        rt = dev[:-1]
        anext = absdev[1:] - absdev[1:].mean()
        ld = math.sqrt(float(np.sum(rt**2) * np.sum(anext**2)))
        leverage = float(np.sum(rt * anext) / ld) if ld > 0 else float("nan")
        return {
            "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
            "abs_acf1": abs_acf1,
            "leverage": leverage,
        }

    hist = _facts(_to_port(historical, is_paths=False))  # historical is a (T, n_assets) panel
    syn = _facts(_to_port(synthetic, is_paths=synthetic_is_paths))
    # The vol-clustering gate uses a MATERIALLY-positive threshold (0.02), not > 0: the ACF(1) of
    # |returns| of an i.i.d. series is ~0 with small-sample positive noise, so a > 0 gate would pass
    # a Gaussian generator spuriously. Real daily series cluster at ACF(1) ~ 0.1-0.4.
    checks = {
        "fat_tails": bool(np.isfinite(syn["excess_kurtosis"]) and syn["excess_kurtosis"] > 0.5),
        "vol_clustering": bool(np.isfinite(syn["abs_acf1"]) and syn["abs_acf1"] > 0.02),
        "leverage": bool(np.isfinite(syn["leverage"]) and syn["leverage"] < 0.0),
    }
    return {
        "historical": hist,
        "synthetic": syn,
        "checks": checks,
        "passed": bool(checks["fat_tails"] and checks["vol_clustering"]),
    }


def claims() -> dict[str, str]:
    """The load-bearing-vs-overclaim statement for this harness (single source of truth).

    Returns a mapping spelling out exactly what an OOD-stress result can and cannot claim, for
    the write-up's robustness appendix.
    """
    return {
        "can_claim": (
            "Falsification / stress-probe: whether a frozen winner's tail edge SURVIVES a "
            "plausible re-draw of the calibrated dynamics (GARCH-EVT FHS, block bootstrap, "
            "Markov crash). A robustness appendix, reported with Sharpe + drawdown beside every "
            "tail metric."
        ),
        "cannot_claim": (
            "Out-of-sample generalisation to genuinely unseen futures. The generators are "
            "calibrated to the SAME history the winners were selected on, so a 'pass' is not "
            "evidence of generalisation. Never headline; never replace the sealed test leg."
        ),
        "power_caveat": "CVaR-1% under stress is low-power (Bauer 2025).",
        "scope_out": "ABIDES/JAX-LOB (wrong granularity); diffusion/GFlowNet (same-path objection); TAIL-GAN is a hard-caveated Tier-2 stretch, not implemented.",
    }
