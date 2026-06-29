"""Benchmark portfolio allocators (non-learned baselines).

Purpose
-------
Classical and rule-based allocators that compete against the learned agents
(FINAL_PLAN F.6). Each function maps market state to a vector of portfolio
weights lying on the probability simplex (non-negative, summing to one — long
only, fully invested). They provide the floor that any discovered reward must
beat to be interesting.

Allocator canon (FINAL_PLAN F.6) — the authoritative list is :data:`STRATEGY_CANON` at the foot
of this module; all TEN members are named here (the list previously stopped at the first five,
omitting the entire block-B8 extension including the tail-optimal ``min_cvar`` — corrected
2026-07-26, deep review)
------------------------------------------------------------------------------------------------
    spy_buy_and_hold : equal-weight market proxy; a DUPLICATE of equal_weight (see its docstring)
                       and therefore excluded from the benchmark gate.
    equal_weight     : 1/N naive diversification; no forecast (DeMiguel et al. 2009 floor).
    mean_variance    : Markowitz with Ledoit-Wolf shrinkage; long-only tangency QP.
    risk_parity      : equal risk contribution, convex Spinu (2013) log-barrier form.
    hrp              : Hierarchical Risk Parity (Lopez de Prado): hierarchical
                       clustering -> quasi-diagonalization -> recursive bisection.
    minimum_variance : long-only global minimum-variance QP (Clarke et al. 2011).
    inverse_volatility       : naive risk parity, w ∝ 1/σ (correlation-blind).
    maximum_diversification  : most-diversified portfolio (Choueifaty & Coignard 2008).
    cross_sectional_momentum : top-tertile past-return winners, long-only (Jegadeesh & Titman 1993).
    min_cvar         : CVaR-optimal / TAIL-optimal via the Rockafellar-Uryasev LP (2000) — the
                       benchmark that optimises the very tail a tail-aware reward targets.

The nine gate names in ``analyze_campaign::_BENCHMARK_NAMES`` are this canon MINUS
``spy_buy_and_hold`` (which would double-count the 1/N floor).

Conventions
-----------
    returns : a window of per-asset returns (rows = time, cols = assets) used
              to estimate moments where required.
    cfg     : configuration object (risk aversion, shrinkage flags, lookback).
    output  : 1-D weight vector on the simplex with one entry per asset.

Tests (tests/test_baselines.py)
-------------------------------
    - test_strategies_return_simplex: every allocator returns simplex weights.
    - test_mean_variance_uses_shrinkage: mean_variance applies Ledoit-Wolf.
    - test_no_forecast_baselines: equal_weight and hrp need no return forecast.
"""

from __future__ import annotations


from typing import Any

import numpy as np


def _as_window(returns: Any) -> np.ndarray:
    """Coerce a returns window to a 2-D (time, assets) float array."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("returns window must be 1-D or 2-D (time, assets)")
    return arr


def _n_assets(returns: Any) -> int:
    return _as_window(returns).shape[1]


def spy_buy_and_hold(returns: Any, cfg: Any = None) -> np.ndarray:
    """Equal-weight market proxy — **identical to** :func:`equal_weight` (a fully-invested 1/N basket).

    HONEST LABELLING (leakage/rigor audit 2026-06-20, R19). The anonymized survivorship-free PIT panel
    carries **no index column and no market caps** (the gold parquet is returns-only; caps live in the
    non-redistributable Refinitiv source and were consumed only for top-N selection). This function is
    therefore NOT the actual S&P 500 / SPY — it returns the same 1/N weights as :func:`equal_weight`,
    so it is a DUPLICATE of the DeMiguel floor, not a distinct market benchmark. It is consequently
    **excluded from the frozen benchmark gate** (``_BENCHMARK_NAMES`` in ``analyze_campaign``) to avoid
    double-counting 1/N.

    NOTE: **UPDATED 2026-07-30 — the last sentence of this paragraph used to read "a genuine market
    benchmark … is a documented gated data addition — it needs a non-anonymized data pull". THAT IS
    NO LONGER TRUE, and had not been true since 2026-07-01**, when the `.SPXTR` total-return series
    was pulled from Refinitiv and frozen with provenance into ``data/raw/rf_spxtr.csv``. It then sat
    unloaded for a month while two separate docstrings described it as a limitation. The cap-weighted
    market line is now available as
    :func:`src.data.market_reference.load_spx_total_return`, and on the agents' own 1,571-session
    sealed axis it returns **+213.3 % cumulative, Sharpe +1.1302** — reported alongside the
    equal-weight proxy (+1.1656) rather than in place of it, because they measure different things.

    Retained as a back-compatible alias only; prefer :func:`equal_weight`. The name is kept because
    it is a registry key several call sites match on, and the honest description above is what a
    reader gets on inspection — but **this function is not, and never was, SPY.**

    FINAL_PLAN F.6 (benchmark allocators).
    """
    n = _n_assets(returns)
    return np.full(n, 1.0 / n, dtype=float)


def equal_weight(returns: Any, cfg: Any = None) -> np.ndarray:
    """Naive 1/N equal-weight allocation.

    Algorithm sketch
    -----------------
    w_i = 1 / N for all N assets. Requires NO return forecast — only the asset
    count is needed.

    FINAL_PLAN F.6 (benchmark allocators).
    """
    n = _n_assets(returns)
    return np.full(n, 1.0 / n, dtype=float)


def mean_variance(returns: Any, cfg: Any = None) -> np.ndarray:
    """Markowitz mean-variance allocation with Ledoit-Wolf shrinkage.

    Algorithm sketch
    -----------------
    1. Estimate expected returns mu from ``returns``.
    2. Estimate covariance Sigma via Ledoit-Wolf shrinkage (shrinks the sample
       covariance toward a structured target; stabilizes inversion).
    3. Solve w* proportional to Sigma^{-1} mu under risk aversion from ``cfg``,
       then project onto the long-only simplex.

    FINAL_PLAN F.6 (benchmark allocators; shrinkage is the audited detail).
    """
    from sklearn.covariance import LedoitWolf

    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)

    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w

    al = arr[:, live]
    mu = al.mean(axis=0)
    # Ledoit-Wolf shrinkage toward a scaled-identity target (symmetric positive-definite; stable).
    cov = LedoitWolf().fit(al).covariance_ + 1e-12 * np.eye(nl)
    # Long-only TANGENCY (max-Sharpe) via the convex QP, NOT a projection of the unconstrained Σ^{-1}μ
    # (which collapses to one asset; critical-review 2026-06-20). GMV fallback when μ has no upside.
    w_live = _long_only_max_sharpe(mu, cov)
    w = np.zeros(n, dtype=float)
    w[live] = w_live
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


# NB: a Euclidean ``_project_simplex`` helper was REMOVED (2026-06-20). Projecting an UNCONSTRAINED
# inverse-covariance vector onto the simplex is NOT the long-only optimum — it collapses to a single asset
# (the exact bug that made minimum_variance / maximum_diversification degenerate). The allocators now solve
# the long-only QP directly (`_long_only_min_variance` / `_long_only_max_sharpe`); the footgun is gone.


def _live_mask(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Boolean mask of LIVE assets — those with non-degenerate return variation over the window.

    Delisted names (the ``liquidate_to_cash`` zero-fill) have ~0 variance; an inverse-vol or
    minimum-variance allocator would otherwise hand them ~100% of the weight (``1/σ`` with ``σ→0``).
    They are excluded here and weighted 0 (critical-review 2026-06-20).
    """
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    return np.asarray(sd > eps, dtype=bool)


def _long_only_min_variance(cov: np.ndarray) -> np.ndarray:
    """Long-only global minimum-variance weights: ``argmin_w wᵀΣw  s.t. w>=0, Σw=1`` (a convex QP).

    Solving the CONSTRAINED QP is essential: Euclidean-projecting the UNCONSTRAINED ``Σ^{-1}1`` onto the
    simplex collapses to a single asset (it is NOT the long-only GMV — critical-review 2026-06-20).
    """
    n = int(cov.shape[0])
    if n == 1:
        return np.ones(1, dtype=float)
    w0 = np.full(n, 1.0 / n)
    try:
        from scipy.optimize import minimize

        res = minimize(
            lambda w: float(w @ cov @ w), w0, jac=lambda w: 2.0 * (cov @ w), method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones(n)},),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        # A NON-CONVERGED solve must not be reported as the GMV (2026-07-26, deep review):
        # ``res.x`` is returned regardless of success, and an unconverged iterate renormalises to a
        # perfectly valid-looking simplex vector that is NOT minimum-variance — measured at up to 60x
        # the optimal portfolio SD on a forced maxiter=1 solve. The sibling ``_long_only_min_cvar``
        # already gates on ``res.success``; these two QPs did not. Not triggerable in 500 adversarial
        # solves (singular and 1e8-condition covariances all converged), so this is HARDENING, not a
        # live-bug fix — it makes the failure detectable-by-fallback instead of silent.
        if not getattr(res, "success", False):
            raise RuntimeError(f"SLSQP min-variance did not converge: {getattr(res, 'message', '')}")
        w = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    except Exception:  # noqa: BLE001 - optimiser unavailable/non-converged -> inverse-variance heuristic (still long-only)
        w = 1.0 / np.maximum(np.diag(cov), 1e-12)
    s = float(w.sum())
    return w / s if s > 1e-300 else np.full(n, 1.0 / n)


def _long_only_max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only tangency (max-Sharpe) weights via the convex trick ``min wᵀΣw s.t. wᵀμ=1, w>=0``.

    Renormalised to the simplex. Falls back to the long-only GMV when ``μ`` carries no positive signal
    (the tangency portfolio is then undefined). Estimation error in ``μ`` makes the unconstrained
    Σ^{-1}μ direction notoriously unstable, so the long-only constraint is the robustifier.
    """
    n = int(len(mu))
    if n == 1:
        return np.ones(1, dtype=float)
    if not np.any(np.asarray(mu) > 1e-12):
        return _long_only_min_variance(cov)
    try:
        from scipy.optimize import minimize

        res = minimize(
            lambda w: float(w @ cov @ w), np.full(n, 1.0 / n), jac=lambda w: 2.0 * (cov @ w),
            method="SLSQP", bounds=[(0.0, None)] * n,
            constraints=({"type": "eq", "fun": lambda w: float(w @ mu - 1.0), "jac": lambda w: np.asarray(mu)},),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        # Same convergence gate as the GMV QP above: an unconverged tangency iterate would be
        # renormalised into a plausible-looking simplex vector and reported as the max-Sharpe
        # benchmark. Fall back to the long-only GMV, exactly as the solver-unavailable path does.
        if not getattr(res, "success", False):
            raise RuntimeError(f"SLSQP tangency did not converge: {getattr(res, 'message', '')}")
        w = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    except Exception:  # noqa: BLE001 - optimiser unavailable/non-converged -> GMV fallback
        return _long_only_min_variance(cov)
    s = float(w.sum())
    return w / s if s > 1e-300 else _long_only_min_variance(cov)


def risk_parity(returns: Any, cfg: Any = None) -> np.ndarray:
    """Equal-risk-contribution (risk parity) allocation — convex log-barrier formulation.

    Finds weights such that each asset contributes equally to total portfolio risk
    (``w_i*(Σw)_i`` constant across ``i``). Rather than the fragile multiplicative fixed-point
    (which on ill-conditioned windows oscillates to a CONCENTRATED, non-risk-parity solution and could
    divide by zero), this solves the **convex** Spinu (2013) / Maillard-Roncalli-Teiletche (2010)
    problem ``min_{y>0} ½ yᵀΣy − (1/n) Σ log y_i`` whose unique minimiser has equal risk contributions;
    the normalised ``y`` is the ERC portfolio. Solved with L-BFGS-B (robust, reliably convergent), with
    a guarded analytic-Newton fallback. Needs no return forecast.

    FINAL_PLAN F.6 (benchmark allocators).
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)

    # Exclude DELISTED / zero-variance names FIRST (like every other cov/vol allocator): the ERC
    # log-barrier objective puts ~0 risk on a zero-variance name, so equal-risk-contribution is satisfied
    # DEGENERATELY by dumping weight onto the dead name — run on the LIVE sub-panel (critical-review 2026-06-20).
    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w

    cov = np.atleast_2d(np.cov(arr[:, live], rowvar=False)) + 1e-8 * np.eye(nl)
    c = 1.0 / nl
    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
    y0 = inv_vol / inv_vol.sum()

    def _obj(y: np.ndarray) -> float:
        return 0.5 * float(y @ cov @ y) - c * float(np.sum(np.log(np.maximum(y, 1e-300))))

    def _grad(y: np.ndarray) -> np.ndarray:
        return cov @ y - c / np.maximum(y, 1e-300)

    try:
        from scipy.optimize import minimize

        res = minimize(
            _obj, y0, jac=_grad, method="L-BFGS-B",
            bounds=[(1e-12, None)] * nl, options={"maxiter": 1000, "ftol": 1e-15},
        )
        y = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    except Exception:  # noqa: BLE001 - optimiser unavailable -> guarded sqrt-damped Newton fallback
        y = y0.copy()
        for _ in range(50_000):
            g = cov @ y - c / np.maximum(y, 1e-300)
            if float(np.max(np.abs(g))) < 1e-12:
                break
            y = np.maximum(y * np.sqrt(np.maximum(c / np.maximum(y * (cov @ y), 1e-300), 1e-300)), 1e-300)

    s = float(y.sum())
    y = y / s if s > 1e-300 else np.full(nl, 1.0 / nl)
    w = np.zeros(n, dtype=float)
    w[live] = y
    return w


def hrp(returns: Any, cfg: Any = None) -> np.ndarray:
    """Hierarchical Risk Parity allocation (Lopez de Prado).

    Algorithm sketch
    -----------------
    1. Compute the correlation-distance matrix from ``returns``.
    2. Hierarchical clustering (linkage) of assets.
    3. Quasi-diagonalization: reorder assets so similar ones are adjacent.
    4. Recursive bisection: split the ordered tree and allocate inversely to
       cluster variance down the dendrogram.
    Requires NO expected-return forecast — only the covariance structure.

    FINAL_PLAN F.6 (benchmark allocators).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)

    # Exclude DELISTED / zero-variance names BEFORE clustering: they make ``corr = cov/outer(std,std)``
    # NaN (linkage then raises "must contain only finite values") AND give the inverse-variance bisection
    # a 1/0 weight. Every 2020-2026 test window holds >=1 such name (liquidate_to_cash zero-fill). Run HRP
    # on the LIVE sub-panel and weight dead names 0 (critical-review 2026-06-20).
    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w

    cov = np.atleast_2d(np.cov(arr[:, live], rowvar=False))
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)  # defense-in-depth (live names are finite)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # 1. Correlation distance d = sqrt(0.5 * (1 - corr)); scrub any residual non-finite to MAX distance 1.
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform

    # 2-4. Cluster, quasi-diagonalise, recursive-bisection over the LIVE names.
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link, nl)
    weights_live = _recursive_bisection(cov, sort_ix)
    w = np.zeros(n, dtype=float)
    w[live] = weights_live
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


def _quasi_diag(link: np.ndarray, n_leaves: int) -> list[int]:
    """Return the leaf ordering implied by a SciPy linkage matrix."""
    link = link.astype(int)
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    # Expand any cluster ids (>= n_leaves) into their constituent leaves.
    while max(sort_ix) >= n_leaves:
        new_order: list[int] = []
        for item in sort_ix:
            if item < n_leaves:
                new_order.append(item)
            else:
                row = item - n_leaves
                new_order.append(int(link[row, 0]))
                new_order.append(int(link[row, 1]))
        sort_ix = new_order
    return sort_ix


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance-weighted variance of a sub-portfolio (HRP)."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.maximum(np.diag(sub), 1e-12)  # floor: a zero-variance leaf must not give 1/0 (defense)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Allocate weights by recursive bisection over the quasi-diagonal order."""
    w = np.ones(cov.shape[0], dtype=float)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            c[start:stop]
            for c in clusters
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]
            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            for idx in left:
                w[idx] *= alpha
            for idx in right:
                w[idx] *= 1.0 - alpha
    return w


# =========================================================================== #
# Extended allocator canon (block B8): more standard benchmarks so the agent     #
# competes against a broad, citation-anchored benchmark suite.                   #
# =========================================================================== #
def minimum_variance(returns: Any, cfg: Any = None) -> np.ndarray:
    """Long-only global minimum-variance portfolio (Markowitz 1952; Clarke, de Silva & Thorley 2011).

    Needs NO return forecast — only the covariance. Solves the CONSTRAINED convex QP
    ``argmin_w wᵀΣw s.t. w>=0, Σw=1`` over the LIVE names via :func:`_long_only_min_variance`.

    NOTE: CORRECTED 2026-07-26 (deep review): this docstring previously said the *unconstrained*
    ``Σ^{-1}1`` solution "is projected onto the long-only simplex" — describing the exact footgun that
    was REMOVED on 2026-06-20 (see the module-level note above ``_live_mask``). Euclidean-projecting the
    unconstrained GMV is NOT the long-only optimum: it collapses to a single asset, which is what made
    this allocator degenerate in the first place. The code has solved the QP since that fix; only the
    prose lagged, and it contradicted the inline comment three lines below it.
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)
    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w
    cov = np.atleast_2d(np.cov(arr[:, live], rowvar=False)) + 1e-8 * np.eye(nl)
    w = np.zeros(n, dtype=float)
    w[live] = _long_only_min_variance(cov)  # CONSTRAINED long-only GMV over the live names
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


def _long_only_min_cvar(scenarios: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Long-only CVaR-optimal weights ``argmin_w CVaR_α(−w·r)  s.t. w>=0, Σw=1`` — the Rockafellar-Uryasev LP.

    CVaR of the portfolio LOSS ``L_t = −r_t·w`` is ``min_ζ  ζ + 1/(αT) Σ_t [L_t − ζ]^+`` (Rockafellar &
    Uryasev 2000, *J. Risk*); jointly minimising over ``(w, ζ, u)`` with epigraph variables ``u_t ≥ L_t − ζ``
    and ``u_t ≥ 0`` is a LINEAR PROGRAM in ``T + n + 1`` variables, solved with HiGHS. Deterministic. On any
    solver failure it falls back to long-only 1/N over the given block (never a degenerate single-name weight).
    """
    scen = np.asarray(scenarios, dtype=float)
    T, n = scen.shape
    if n == 1:
        return np.ones(1, dtype=float)
    if T < 2:
        return np.full(n, 1.0 / n, dtype=float)
    try:
        from scipy.optimize import linprog

        # x = [ w(n) | ζ(1) | u(T) ];  minimise ζ + 1/(αT) Σ u_t.
        c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / (alpha * T))])
        # Epigraph rows:  −r_t·w − ζ − u_t ≤ 0   (⇔  u_t ≥ L_t − ζ,  with L_t = −r_t·w).
        A_ub = np.hstack([-scen, -np.ones((T, 1)), -np.eye(T)])
        b_ub = np.zeros(T)
        # Budget:  Σ w = 1.
        A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T   # w∈[0,1], ζ free, u≥0
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not getattr(res, "success", False):
            return np.full(n, 1.0 / n, dtype=float)
        w = np.maximum(np.asarray(res.x[:n], dtype=float), 0.0)
    except Exception:  # noqa: BLE001 - solver unavailable/failed -> long-only 1/N (never degenerate)
        return np.full(n, 1.0 / n, dtype=float)
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n, dtype=float)


def min_cvar(returns: Any, cfg: Any = None) -> np.ndarray:
    """CVaR-optimal (tail-optimal) portfolio ``argmin_w CVaR_α(−w·r)`` (Rockafellar & Uryasev 2000).

    The classical TAIL-optimal allocator: it minimises the Conditional Value-at-Risk of the portfolio loss
    DIRECTLY, so it is the natural hand-crafted comparator for a tail-risk study — does the LLM's tail-aware
    reward beat even the benchmark that optimises the very tail the reward targets? Long-only over the LIVE
    names (delisted zero-variance names excluded, as for the other risk allocators — else a dead name with
    ~0 loss variation would absorb the whole book). The CVaR level defaults to the headline ``α=0.05`` and is
    overridable via ``cfg['cvar_alpha']``. Report-only benchmark context, DSR-deflated at N=1 like the other
    allocators (`benchmark_floor`). Deterministic (HiGHS LP; no RNG).
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)
    alpha = 0.05
    if isinstance(cfg, dict):
        alpha = float(cfg.get("cvar_alpha", 0.05))
    elif cfg is not None and hasattr(cfg, "cvar_alpha"):
        alpha = float(getattr(cfg, "cvar_alpha"))
    alpha = min(max(alpha, 1e-3), 0.5)   # guard the 1/(αT) objective coefficient
    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w
    w = np.zeros(n, dtype=float)
    w[live] = _long_only_min_cvar(arr[:, live], alpha)
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


def inverse_volatility(returns: Any, cfg: Any = None) -> np.ndarray:
    """Inverse-volatility (naive risk parity): ``w_i ∝ 1/σ_i`` (equal risk under zero correlation).

    The common, robust risk-parity approximation that ignores correlations; needs no forecast.
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(n)
    live = sd > 1e-10  # exclude dead (zero-vol) names: 1/σ would otherwise put ~100% on a delisted name
    if not live.any():
        return np.full(n, 1.0 / n)
    w = np.zeros(n, dtype=float)
    w[live] = 1.0 / sd[live]
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


def maximum_diversification(returns: Any, cfg: Any = None) -> np.ndarray:
    """Maximum-diversification portfolio (Choueifaty & Coignard 2008, *J. Portfolio Mgmt*).

    Maximises the diversification ratio ``(w·σ)/sqrt(wᵀΣw)``. Needs no return forecast.

    Implemented the Choueifaty-Coignard way: the most-diversified long-only portfolio is the long-only
    GMV of the CORRELATION matrix, de-scaled by ``1/σ`` and renormalised (see the inline comment below).

    NOTE: CORRECTED 2026-07-26 (deep review): this docstring previously said the unconstrained
    ``w ∝ Σ^{-1}σ`` solution is "projected onto the long-only simplex" — the very approach the inline
    comment below marks as WRONG (it collapses to one asset, giving diversification ratio 1.0, the
    WORST possible value for this objective). The prose asserted the opposite of the code.
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)
    live = _live_mask(arr)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    if nl == 1:
        w = np.zeros(n, dtype=float)
        w[live] = 1.0
        return w
    cov = np.atleast_2d(np.cov(arr[:, live], rowvar=False))
    sigma = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    # Choueifaty-Coignard: the most-diversified portfolio is the long-only GMV of the CORRELATION matrix,
    # then de-scaled by 1/σ and renormalised (NOT a projection of the unconstrained Σ^{-1}σ, which
    # collapses to one asset -> diversification ratio 1.0, the WORST; critical-review 2026-06-20).
    corr = cov / np.outer(sigma, sigma)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    corr = corr + 1e-8 * np.eye(nl)
    y = _long_only_min_variance(corr)
    w_live = np.maximum(y / sigma, 0.0)
    sl = float(w_live.sum())
    w_live = w_live / sl if sl > 0 else np.full(nl, 1.0 / nl)
    w = np.zeros(n, dtype=float)
    w[live] = w_live
    s = float(w.sum())
    return w / s if s > 0 else np.full(n, 1.0 / n)


def cross_sectional_momentum(returns: Any, cfg: Any = None) -> np.ndarray:
    """Cross-sectional momentum: equal-weight the top-tertile past-return assets (Jegadeesh & Titman 1993).

    Long-only (holds the winners, not shorting the losers) over the observation window's cumulative
    return. A standard active benchmark that the learned reward must beat.
    """
    arr = _as_window(returns)
    n = arr.shape[1]
    if n == 1:
        return np.ones(1, dtype=float)
    cum = np.prod(1.0 + arr, axis=0) - 1.0  # past cumulative return per asset
    # Rank only LIVE names: a delisted (zero-return -> cum==0) name must not outrank a live name that fell,
    # and NaN must never sort to the top (np.argsort places NaN last) — critical-review 2026-06-20.
    live = _live_mask(arr) & np.isfinite(cum)
    nl = int(live.sum())
    if nl == 0:
        return np.full(n, 1.0 / n)
    cum_ranked = np.where(live, cum, -np.inf)  # dead/NaN names sink to the bottom of the ranking
    k = max(1, nl // 3)  # top tertile of the LIVE universe
    top = np.argsort(cum_ranked)[-k:]
    w = np.zeros(n, dtype=float)
    w[top] = 1.0 / k
    return w


#: The full allocator canon (name -> callable) for the benchmark suite.
STRATEGY_CANON: dict[str, Any] = {
    "spy_buy_and_hold": spy_buy_and_hold,
    "equal_weight": equal_weight,
    "mean_variance": mean_variance,
    "risk_parity": risk_parity,
    "hrp": hrp,
    "minimum_variance": minimum_variance,
    "inverse_volatility": inverse_volatility,
    "maximum_diversification": maximum_diversification,
    "cross_sectional_momentum": cross_sectional_momentum,
    "min_cvar": min_cvar,   # CVaR-optimal / TAIL-optimal (Rockafellar & Uryasev 2000)
}
