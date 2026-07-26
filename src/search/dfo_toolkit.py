"""CMA-ES and TPE search over the reward-template coefficients — the derivative-free-optimization (DFO)
toolkit that completes the H4 comparator beyond random-search + GP-EI (``hansen2001cmaes``; ``bergstra2011tpe``
via ``akiba2019optuna``), spanning the four principal DFO families (model-free / surrogate-model / evolution-strategy /
density-ratio).

These are **drop-in siblings** of ``bayes_opt.random_search_over_template`` / ``bayes_opt_over_template``: the
same signature, the same ``{best_coeffs, best_score, history, n_evaluated, budget}`` return, the same **matched
candidate budget** (total ``template_eval_fn`` calls == budget — the fair H4 comparison), the same
**deterministic-from-``rng``** contract (an omitted ``rng`` defaults to seed 0, never OS entropy — the exact
reproducibility footgun ``bayes_opt.py:242`` documents), and the same ``cache_lookup`` / ``on_evaluated``
resume/checkpoint hooks so a winner round-trips to sealed-leg-executable source via
``reward_family.params_to_source`` unchanged.

Parallelism (the campaign-speed lens): CMA-ES proposes a whole **population** per generation and TPE a startup
**batch** — both of which the parallel search leg can dispatch concurrently, unlike GP-EI's inherently
sequential chain. The matched budget is UNCHANGED; only *how* the candidates are dispatched differs.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.search.bayes_opt import _as_bounds, _budget

__all__ = ["cma_es_over_template", "tpe_over_template", "over_template_optimizer"]


def _make_recorder(
    template_eval_fn: Callable[[np.ndarray], float],
    cache_lookup: Optional[Callable[[int, np.ndarray], Optional[float]]],
    on_evaluated: Optional[Callable[[int, np.ndarray, float], None]],
) -> tuple[Callable[[np.ndarray, str], float], list[dict[str, Any]], list[np.ndarray], list[float]]:
    """The shared evaluation recorder — byte-identical semantics to ``bayes_opt._evaluate``: a cached score
    skips training while leaving the OBSERVED (x, y) history identical, and ``on_evaluated`` fires only for
    FRESH evaluations so each candidate can be checkpointed the moment it completes."""
    history: list[dict[str, Any]] = []
    x_obs: list[np.ndarray] = []
    y_obs: list[float] = []

    def _evaluate(x: np.ndarray, source: str) -> float:
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

    return _evaluate, history, x_obs, y_obs


def _result(history: list[dict[str, Any]], x_obs: list[np.ndarray], y_obs: list[float], budget: int) -> dict[str, Any]:
    y = np.asarray(y_obs, dtype=float)
    best = int(np.argmax(y))
    return {
        "best_coeffs": x_obs[best].copy(),
        "best_score": float(y[best]),
        "history": history,
        "n_evaluated": len(history),
        "budget": int(budget),
    }


def cma_es_over_template(
    template_eval_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]],
    cfg: Any,
    rng: Optional[np.random.Generator] = None,
    cache_lookup: Optional[Callable[[int, np.ndarray], Optional[float]]] = None,
    on_evaluated: Optional[Callable[[int, np.ndarray, float], None]] = None,
    *,
    init_step_frac: float = 0.25,
    popsize: Optional[int] = None,
    batch_eval_fn: Optional[Callable[[Sequence[np.ndarray]], Sequence[float]]] = None,
) -> dict[str, Any]:
    """CMA-ES over the template coefficients (Hansen & Ostermeier 2001) — the low-dimensional continuous DFO
    gold standard — at the matched budget and fully deterministic from ``rng``.

    Per-dimension initial step = ``init_step_frac`` of each coordinate's box range (via ``CMA_stds``), so the
    search is scale-correct without normalising the caller's bounds. FULL generations of ``popsize`` candidates
    are told to CMA (the covariance update only advances on complete generations); the final partial batch that
    would overshoot the budget is evaluated for the archive but NOT told, so the total is EXACTLY the matched
    budget with the CMA state left clean.

    ``batch_eval_fn`` (optional) evaluates a WHOLE GENERATION at once and, when supplied, replaces the
    per-member loop. **D27, 2026-08-02 — and it is a throughput fix, not a science change.** The module
    docstring above has always said a population "can be dispatched concurrently", and
    ``src/cluster/campaign.py`` asserted in a comment that CMA-ES "already dispatches a whole population
    per generation" — but the code below evaluated the population with a SERIAL list comprehension, and on
    the cluster path every one of those calls is a blocking ``run_batch``. Measured on RUN 4: ``cma_es``
    stood at 9 of 30 candidates on a 21-step remaining chain of ~8.6 h each, gating the CORE line's C1
    barrier and therefore the campaign's COMMON rung, while ``lanes._CMA_SERIAL_GENERATIONS`` priced that
    chain at 4 and the sentinel consequently reported the arm COMPLETE at 9/4.

    Dispatching the generation together is a PURE DISPATCH change: ``es.ask()`` produces the whole
    population before any member is evaluated and ``es.tell()`` consumes all of it, so no member's proposal
    depends on another member's fitness. Identity is not argued but MEASURED —
    ``tests/test_dfo_cma_batch.py`` asserts the same points in the same order, the same scores, the same
    winner, the same history and the same ``on_evaluated``/cache behaviour as the serial path, and carries
    a mutation control so the comparison can fail. Opt-in and backward-compatible: without
    ``batch_eval_fn`` the behaviour is byte-identical to before, which the same test also asserts.
    """
    import cma  # pycma (BSD)

    box = _as_bounds(bounds)
    budget = _budget(cfg)
    if budget <= 0:
        raise ValueError(f"matched budget must be positive; got {budget}")
    if rng is None:
        rng = np.random.default_rng(0)
    lo, hi = box[:, 0].astype(float), box[:, 1].astype(float)
    rng_span = np.maximum(hi - lo, 1e-12)
    _evaluate, history, x_obs, y_obs = _make_recorder(template_eval_fn, cache_lookup, on_evaluated)

    def _evaluate_generation(xs: list, source: str) -> list[float]:
        """Score a WHOLE generation while preserving ``_evaluate``'s semantics exactly.

        Mirrors ``tpe_over_template``'s already-shipped startup-batch block rather than inventing a
        second pattern: same index convention (``idx0 = len(history)``), same per-member cache lookup,
        same 1:1 length check, same rule that ``on_evaluated`` fires ONLY for fresh work, same append
        order. Without ``batch_eval_fn`` this IS the previous list comprehension.
        """
        if batch_eval_fn is None:
            return [_evaluate(x, source) for x in xs]
        idx0 = len(history)
        cached = [cache_lookup(idx0 + i, x) if cache_lookup is not None else None
                  for i, x in enumerate(xs)]
        need = [i for i, c in enumerate(cached) if c is None]
        got = list(batch_eval_fn([xs[i] for i in need])) if need else []
        if len(got) != len(need):
            raise ValueError(
                f"batch_eval_fn returned {len(got)} scores for {len(need)} points — a CMA generation "
                "must be evaluated 1:1 or the covariance update and the history would desynchronise")
        fresh = dict(zip(need, got))
        out: list[float] = []
        for i, x in enumerate(xs):
            score = float(cached[i]) if cached[i] is not None else float(fresh[i])
            if cached[i] is None and on_evaluated is not None:
                on_evaluated(idx0 + i, x, score)
            x_obs.append(x.copy())
            y_obs.append(score)
            history.append({"coeffs": x.copy(), "score": score, "source": source})
            out.append(score)
        return out

    x0 = (lo + hi) / 2.0
    opts: dict[str, Any] = {
        "bounds": [lo.tolist(), hi.tolist()],
        "CMA_stds": (float(init_step_frac) * rng_span).tolist(),
        "seed": int(rng.integers(1, 2**31 - 1)),
        "verbose": -9,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(x0.tolist(), 1.0, opts)

    # The MATCHED BUDGET is the sole stop — CMA's own convergence/flat-fitness stops are ignored so that,
    # like random_search / GP-EI / TPE, exactly ``budget`` candidates are evaluated (a fair H4 comparison);
    # after convergence CMA simply keeps sampling its distribution, spending the remaining budget on refinement.
    evaluated = 0
    while evaluated < budget:
        xs = [np.clip(np.asarray(x, dtype=float), lo, hi) for x in es.ask()]
        remaining = budget - evaluated
        if len(xs) <= remaining:
            scores = _evaluate_generation(xs, "cma")
            es.tell(xs, [-s for s in scores])  # CMA minimises; we maximise fitness
            evaluated += len(xs)
        else:  # last partial batch: evaluate for the archive, do NOT tell (keep the CMA state clean)
            _evaluate_generation(xs[:remaining], "cma_tail")
            evaluated += remaining
            break
    return _result(history, x_obs, y_obs, budget)


def tpe_over_template(
    template_eval_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]],
    cfg: Any,
    rng: Optional[np.random.Generator] = None,
    cache_lookup: Optional[Callable[[int, np.ndarray], Optional[float]]] = None,
    on_evaluated: Optional[Callable[[int, np.ndarray, float], None]] = None,
    *,
    n_startup: Optional[int] = None,
    batch_eval_fn: Optional[Callable[[Sequence[np.ndarray]], Sequence[float]]] = None,
) -> dict[str, Any]:
    """Tree-structured Parzen Estimator over the template coefficients (Bergstra et al. 2011) via Optuna
    (Akiba et al. 2019) — the density-ratio, low-budget-friendly model-based complement to GP-EI — at the
    matched budget and deterministic from ``rng``. ``n_startup`` random trials seed the density models before
    TPE guides (default ``min(10, budget)``, TPE's designed low-budget regime).

    ``batch_eval_fn`` (optional) evaluates a LIST of points at once and, when supplied, is used for the
    ``n_startup`` phase — cutting the SEQUENTIAL chain from ``budget`` (30) to ``budget - n_startup``
    (~21). Those startup points come from Optuna's RANDOM sampler and depend on no observed value, so
    concurrent evaluation is a pure DISPATCH change that returns identical results; the matched budget,
    the seed, and the guided phase are untouched. Omit it and behaviour is byte-identical to before.
    This matters because ``study.optimize`` evaluates one trial at a time, which made TPE a 30-step
    serial chain — LONGER than GP-EI's 25 — i.e. the campaign's binding critical path
    (``src/cluster/lanes.py``)."""
    import optuna  # MIT

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    box = _as_bounds(bounds)
    dim = int(box.shape[0])
    budget = _budget(cfg)
    if budget <= 0:
        raise ValueError(f"matched budget must be positive; got {budget}")
    if rng is None:
        rng = np.random.default_rng(0)
    _evaluate, history, x_obs, y_obs = _make_recorder(template_eval_fn, cache_lookup, on_evaluated)

    n_startup_trials = int(n_startup) if n_startup is not None else min(10, int(budget))
    sampler = optuna.samplers.TPESampler(seed=int(rng.integers(1, 2**31 - 1)), n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _ask_x(trial: "optuna.Trial") -> "np.ndarray":
        return np.array(
            [trial.suggest_float(f"c{j}", float(box[j, 0]), float(box[j, 1])) for j in range(dim)],
            dtype=float,
        )

    def _objective(trial: "optuna.Trial") -> float:
        return _evaluate(_ask_x(trial), "tpe")

    # STARTUP BATCH (2026-07-26, the campaign-speed lens — CAPACITY lane finding T5-a).
    # `study.optimize` evaluates ONE trial at a time, so the whole budget was a SEQUENTIAL chain of
    # 30 — LONGER than GP-EI's 25 serial steps, which would have made TPE the campaign's binding
    # critical path (src/cluster/lanes.py). The first `n_startup_trials` are drawn by Optuna's
    # RANDOM sampler and do NOT depend on any observed value, so evaluating them CONCURRENTLY
    # yields IDENTICAL results — a pure DISPATCH change, exactly like the bayes_opt in-job chain.
    # Cuts the serial chain 30 -> ~21 with no change to the optimiser, the matched budget, or the
    # seed. Opt-in and backward-compatible: without `batch_eval_fn` the behaviour is byte-identical
    # to before (the single-point `template_eval_fn` cannot parallelise anything by itself).
    n_batched = min(n_startup_trials, int(budget)) if batch_eval_fn is not None else 0
    if n_batched > 0:
        trials = [study.ask() for _ in range(n_batched)]
        xs = [_ask_x(t) for t in trials]
        # Cache-aware, with the SAME index convention as `_evaluate` (idx = position in history),
        # so search-replay RESUME stays free and `on_evaluated` still fires only for FRESH work.
        idx0 = len(history)
        cached = [cache_lookup(idx0 + i, x) if cache_lookup is not None else None
                  for i, x in enumerate(xs)]
        need = [i for i, c in enumerate(cached) if c is None]
        got = list(batch_eval_fn([xs[i] for i in need])) if need else []
        if len(got) != len(need):
            raise ValueError(
                f"batch_eval_fn returned {len(got)} scores for {len(need)} points — the startup "
                "batch must be evaluated 1:1 or the study and history would desynchronise")
        fresh = dict(zip(need, got))
        for i, (trial, x) in enumerate(zip(trials, xs)):
            score = float(cached[i]) if cached[i] is not None else float(fresh[i])
            if cached[i] is None and on_evaluated is not None:
                on_evaluated(idx0 + i, x, score)
            x_obs.append(x.copy())
            y_obs.append(score)
            history.append({"coeffs": x.copy(), "score": score, "source": "tpe"})
            study.tell(trial, score)

    remaining = int(budget) - n_batched
    if remaining > 0:
        study.optimize(_objective, n_trials=remaining)
    return _result(history, x_obs, y_obs, budget)


def over_template_optimizer(arm: str) -> Callable[..., dict[str, Any]]:
    """Resolve an H4 family-search arm name to its ADAPTIVE over-template optimizer.

    GP-EI (``bayes_opt``), CMA-ES (``cma_es``) and TPE (``tpe``) share ONE driver body in BOTH
    dispatchers — the cluster ``campaign.run_family_search_arm`` and the laptop
    ``parallel._drive_search_arm``: each searches the SAME reward template, at the SAME matched candidate
    budget, through the SAME scalar ``template_eval(coeffs) -> fitness`` closure; only the proposal
    strategy differs. This is the SINGLE SOURCE OF TRUTH for the arm->function map, so the two dispatchers
    cannot drift, and it fails LOUD on an unregistered arm rather than silently defaulting to GP-EI (the
    old ``else``-branch behaviour, which would mislabel a typo'd arm as bayes_opt).

    ``random_search`` is deliberately NOT here: it draws all candidates UP FRONT (an
    embarrassingly-parallel batch, not an adaptive ask-tell loop), so each dispatcher keeps its own
    distinct up-front-draw path for it. Only the adaptive optimizers share this driver.
    """
    if arm == "bayes_opt":
        from src.search.bayes_opt import bayes_opt_over_template
        return bayes_opt_over_template
    if arm == "cma_es":
        return cma_es_over_template
    if arm == "tpe":
        return tpe_over_template
    raise ValueError(
        f"unknown over-template search arm {arm!r} — expected 'bayes_opt' / 'cma_es' / 'tpe' "
        f"(random_search uses its own up-front-draw path, not this adaptive driver)"
    )
