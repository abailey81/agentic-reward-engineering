"""Bayesian optimization over a fixed parametric reward template (H4b).

Purpose
-------
A non-LLM control arm (FINAL_PLAN F.10, hypothesis H4b). Rather than searching
over arbitrary code, it FIXES the six-term parametric reward family
(``src/baselines/reward_family.py``) and optimizes its continuous weights with a
Gaussian-process Bayesian-optimization loop (GP + Matern-2.5 surrogate, Expected
Improvement acquisition; Snoek, Larochelle & Adams 2012 — NOT Optuna/TPE). Under
the MATCHED candidate budget, H4b compares the LLM's FREE-FORM reward code against
the best member of this fixed parametric family it could tune, i.e. it isolates
the value of an *open-ended reward language* against tuning a fixed parametric one.

Scope note. H4b is NOT a like-for-like
search-procedure comparison against the LLM: BO is denied the LLM's functional
freedom (it can only reweight six fixed primitives), so a positive H4b reflects
the richer language AND procedure, not procedure alone. The procedure-only
comparison is H4a (random search), which now draws from the SAME six-term family
at the SAME fixed discrete dims as this arm — so the H4a-vs-H4b contrast is
purely surrogate (GP-EI) versus uniform sampling at matched budget. The companion
``random_search_over_template`` below is the budget-matched in-family random
reference used to certify the GP surrogate is a fair control at this budget.

Algorithm (FINAL_PLAN F.10, H4b)
--------------------------------
    1. Fix a parametric reward template; its coefficients live in a bounded box.
    2. Seed with ``n_init`` uniform random points in the box (i.i.d. uniform).
    3. Fit a Gaussian-process surrogate (Matern kernel) to the observed
       (coefficient -> held-out fitness) pairs.
    4. Maximize the Expected-Improvement acquisition over a dense random sample
       of the box to choose the next coefficient vector.
    5. Evaluate it via the injected ``template_eval_fn`` (which, in production,
       instantiates the templated reward, trains the fixed headline agent, and
       scores validation fitness; deterministic in tests), update the surrogate.
    6. Stop when the matched candidate budget is exhausted; return the best
       coefficients + score + full history.

The total number of ``template_eval_fn`` calls equals the matched candidate
budget (``n_init`` random seed points + the remaining BO-guided steps), so this
arm consumes exactly the same compute as the others.

Tests (tests/test_search.py)
----------------------------
    - on a known concave objective (``-(x-0.3)^2-(y+0.1)^2``) the optimizer gets
      close to the optimum within the budget;
    - it BEATS random sampling of the same budget on average;
    - it respects the matched budget from ``cfg``.
"""

from __future__ import annotations


from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.utils.config import DotDict

__all__ = ["bayes_opt_over_template", "random_search_over_template"]


def _budget(cfg: Any) -> int:
    """Read the matched candidate budget from ``cfg``."""
    if cfg is None:
        raise ValueError("bayes_opt_over_template requires a cfg with a budget")
    if isinstance(cfg, (DotDict, dict)):
        for key in ("matched_budget", "candidate_budget", "candidate_budget_total"):
            if key in cfg:
                return int(cfg[key])
        raise KeyError(
            "cfg must carry one of matched_budget / candidate_budget / "
            "candidate_budget_total"
        )
    return int(cfg)


def _as_bounds(bounds: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce bounds to a validated ``(d, 2)`` lower/upper array."""
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"bounds must be a (d, 2) array of [low, high] pairs; got shape "
            f"{arr.shape}"
        )
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError("each bound must satisfy low < high")
    return arr


def _sample_uniform(
    bounds: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` points uniformly inside the bounding box."""
    low, high = bounds[:, 0], bounds[:, 1]
    return rng.uniform(low, high, size=(n, bounds.shape[0]))


def _expected_improvement(
    candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition (maximization), vectorized over rows.

    Parameters
    ----------
    candidates : ndarray, shape (m, d)
        Candidate coefficient vectors.
    gp : GaussianProcessRegressor
        Fitted surrogate.
    best_y : float
        Best observed objective value so far (incumbent).
    xi : float
        Exploration margin.

    Returns
    -------
    ndarray, shape (m,)
        EI value at each candidate (>= 0).
    """
    mu, sigma = gp.predict(candidates, return_std=True)
    # Capture the raw posterior std BEFORE clamping so the zero-sigma guard
    # below can actually fire; clamping first made `sigma < 1e-9` identically
    # False, defeating the guard (audit fix: dead/unreachable EI mask).
    zero_sigma = sigma < 1e-9
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[zero_sigma] = 0.0
    return np.maximum(ei, 0.0)


def _winsorize_for_surrogate(y: np.ndarray, n_mad: float = 10.0) -> np.ndarray:
    """Floor catastrophic outliers in the GP's TRAINING TARGETS only (deep review 2026-07-26, #51).

    WHY. Both production evaluators score a FAILED candidate with the sentinel ``-1e9``
    (``cluster/campaign.py`` — which also counts ``state["failed"]`` — and ``orchestration/
    parallel.py``). CMA-ES and TPE are rank-based, so an extreme value is merely "last" and costs
    them nothing. A GP is NOT: with ``normalize_y=True`` the targets are standardised by their own
    mean/std, so ONE sentinel among ~12 observations gives std ~2.8e8 and collapses every genuine
    candidate into a span of ~4e-10 normalised units — numerically identical to the surrogate.
    MEASURED on a smooth 6-D objective: the posterior mean range blew out from [0.018, 0.161] to
    [-1.1e9, 1.1e8], and the EI-selected point fell from a true value of 0.0662 to -0.0813 (best
    achievable 0.1934). One failed training therefore degraded GP-EI to worse-than-random for the
    REST of the arm — silently, and asymmetrically, since the other three H4 optimisers are immune.
    That skews the H4 beat-the-max comparison against ``bayes_opt`` specifically.

    WHAT. A median/MAD fence at ``n_mad`` deviations, applied ONLY to the surrogate's targets:
    ``x_obs``/``y_obs``/``history`` and the returned winner are untouched, so the archive, the
    matched budget, the reported ``best_score`` and the rng consumption are all bit-identical. The
    fence is deliberately permissive — 10 MAD is ~6.7 sigma for Gaussian data — so genuine variation
    is never clipped; only a value no real fitness could take is pulled in, and it still sorts BELOW
    every retained point, preserving its rank as the worst candidate.

    A zero MAD (at least half the observations identical) leaves the array unchanged: there is no
    robust scale to build a fence from, and a GP fitted to a constant has no signal to protect.
    """
    if y.size < 3:
        return y
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    if not (mad > 0.0):
        return y
    fence = med - float(n_mad) * mad
    return np.maximum(y, fence) if np.any(y < fence) else y


def random_search_over_template(
    template_eval_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]],
    cfg: Any,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Pure random search over the same template box (budget-matched baseline).

    Used by the tests to confirm Bayesian optimization beats random sampling of
    the same budget on average. Evaluates exactly ``budget`` uniform draws.

    Parameters
    ----------
    template_eval_fn : Callable[[ndarray], float]
        Maps a coefficient vector to a fitness score (higher is better).
    bounds : sequence of (low, high)
        Per-coefficient box bounds.
    cfg : DotDict or dict or int
        Carries the matched candidate budget.
    rng : numpy.random.Generator, optional
        Source of randomness.

    Returns
    -------
    dict
        ``{"best_coeffs", "best_score", "history", "n_evaluated", "budget"}``.
    """
    box = _as_bounds(bounds)
    budget = _budget(cfg)
    if budget <= 0:
        raise ValueError(f"matched budget must be positive; got {budget}")
    if rng is None:
        # 2026-07-06 (repro audit): the omitted-rng default is now DETERMINISTIC (seed 0), not OS
        # entropy — the exact footgun that once made the parallel BO winner non-reproducible
        # (final-audit #2). Every production caller passes a run-seeded generator explicitly.
        rng = np.random.default_rng(0)

    points = _sample_uniform(box, budget, rng)
    history: list[dict[str, Any]] = []
    best_score = -np.inf
    best_coeffs: Optional[np.ndarray] = None
    for x in points:
        score = float(template_eval_fn(x))
        history.append({"coeffs": x.copy(), "score": score})
        if score > best_score:
            best_score = score
            best_coeffs = x.copy()

    return {
        "best_coeffs": best_coeffs,
        "best_score": best_score,
        "history": history,
        "n_evaluated": len(history),
        "budget": budget,
    }


def bayes_opt_over_template(
    template_eval_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]],
    cfg: Any,
    n_init: int = 5,
    rng: Optional[np.random.Generator] = None,
    cache_lookup: Optional[Callable[[int, np.ndarray], Optional[float]]] = None,
    on_evaluated: Optional[Callable[[int, np.ndarray, float], None]] = None,
    batch_eval_fn: Optional[Callable[[Sequence[np.ndarray]], Sequence[float]]] = None,
) -> dict[str, Any]:
    """Bayesian-optimize reward-template coefficients under the matched budget.

    Parameters
    ----------
    template_eval_fn : Callable[[ndarray], float]
        Maps a coefficient vector to a held-out fitness score (higher is
        better). In production this instantiates the templated reward, trains
        the fixed headline agent, and scores validation fitness; in tests it is
        a deterministic closure (no agent training required).
    bounds : sequence of (low, high)
        Per-coefficient box bounds defining the template's coefficient space.
    cfg : DotDict or dict or int
        Carries the matched candidate budget (total ``template_eval_fn`` calls).
    n_init : int, optional
        Number of random seed evaluations before the GP-guided loop. Clamped to
        the budget. Default 5.
    rng : numpy.random.Generator, optional
        Source of randomness; a fresh default generator is created when ``None``.

    Returns
    -------
    dict
        ``{"best_coeffs", "best_score", "history", "n_evaluated", "budget",
        "n_init"}`` where ``history`` records every evaluation in order.

    Notes
    -----
    The surrogate is a ``GaussianProcessRegressor`` with a Matern(nu=2.5) kernel
    plus a white-noise term, and the acquisition is Expected Improvement
    maximized over a dense uniform sample of the box. The total number of
    ``template_eval_fn`` calls equals ``budget``.
    """
    box = _as_bounds(bounds)
    dim = box.shape[0]
    budget = _budget(cfg)
    if budget <= 0:
        raise ValueError(f"matched budget must be positive; got {budget}")
    if rng is None:
        # 2026-07-06 (repro audit): the omitted-rng default is now DETERMINISTIC (seed 0), not OS
        # entropy — the exact footgun that once made the parallel BO winner non-reproducible
        # (final-audit #2). Every production caller passes a run-seeded generator explicitly.
        rng = np.random.default_rng(0)

    n_init = int(max(1, min(n_init, budget)))

    history: list[dict[str, Any]] = []
    x_obs: list[np.ndarray] = []
    y_obs: list[float] = []

    def _evaluate(x: np.ndarray, source: str) -> float:
        # Resume/checkpoint hooks (2026-07-05 crash-resume hardening). A cached score skips the
        # expensive training while leaving the OBSERVED (x, y) history identical, so the GP
        # posterior — and therefore every subsequent acquisition draw and proposal — reproduces the
        # original trajectory byte-for-byte (the per-iteration rng consumption is unconditional).
        # ``cache_lookup(i, x)`` returns the archived score (the caller hash-verifies the
        # materialized source for these coeffs and fails loud on drift) or ``None``;
        # ``on_evaluated(i, x, score)`` fires only for FRESH evaluations so the caller can
        # checkpoint each candidate the moment it completes instead of after the whole arm.
        idx = len(history)
        cached = cache_lookup(idx, x) if cache_lookup is not None else None
        if cached is not None:
            score = float(cached)
        else:
            score = float(template_eval_fn(x))
            if on_evaluated is not None:
                on_evaluated(idx, x, score)
        x_obs.append(x.copy())
        y_obs.append(score)
        history.append({"coeffs": x.copy(), "score": score, "source": source})
        return score

    # 1. Random seed phase.
    # INIT PHASE — batched when the caller can evaluate concurrently (2026-07-26).
    # These ``n_init`` points are i.i.d. UNIFORM draws from the box: they are fixed by ``rng``
    # before any training runs and depend on NO observed value, so evaluating them together is a
    # pure DISPATCH change with identical results. It matters because the cluster driver turns each
    # ``_evaluate`` into its own array-of-1 job, so the "5 parallel" init was in fact 5 more SERIAL
    # queue-and-train steps — making the real GP chain 30, not 25, on the campaign's critical path
    # (``src/cluster/lanes.py``). Without ``batch_eval_fn`` the behaviour is byte-identical to
    # before, so the laptop path and every existing caller are untouched.
    init_points = _sample_uniform(box, n_init, rng)
    if batch_eval_fn is not None and n_init > 0:
        idx0 = len(history)
        _cached = [cache_lookup(idx0 + i, x) if cache_lookup is not None else None
                   for i, x in enumerate(init_points)]
        _need = [i for i, c in enumerate(_cached) if c is None]
        _got = list(batch_eval_fn([init_points[i] for i in _need])) if _need else []
        if len(_got) != len(_need):
            raise ValueError(
                f"batch_eval_fn returned {len(_got)} scores for {len(_need)} points — the init "
                "batch must be evaluated 1:1 or the observed history would desynchronise")
        _fresh = dict(zip(_need, _got))
        for i, x in enumerate(init_points):
            score = float(_cached[i]) if _cached[i] is not None else float(_fresh[i])
            if _cached[i] is None and on_evaluated is not None:
                on_evaluated(idx0 + i, x, score)
            x_obs.append(x.copy())
            y_obs.append(score)
            history.append({"coeffs": x.copy(), "score": score, "source": "init"})
    else:
        for x in init_points:
            _evaluate(x, "init")

    # 2. BO-guided phase for the remaining budget.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    n_remaining = budget - n_init
    n_acq = 2000  # acquisition-optimization sample size

    for _ in range(n_remaining):
        X = np.asarray(x_obs, dtype=float)
        y = _winsorize_for_surrogate(np.asarray(y_obs, dtype=float))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        gp.fit(X, y)

        candidates = _sample_uniform(box, n_acq, rng)
        ei = _expected_improvement(candidates, gp, best_y=float(np.max(y)))
        next_x = candidates[int(np.argmax(ei))]
        _evaluate(next_x, "bo")

    y_all = np.asarray(y_obs, dtype=float)
    best_idx = int(np.argmax(y_all))

    return {
        "best_coeffs": x_obs[best_idx].copy(),
        "best_score": float(y_all[best_idx]),
        "history": history,
        "n_evaluated": len(history),
        "budget": budget,
        "n_init": n_init,
    }
