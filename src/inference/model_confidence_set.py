"""Model Confidence Set over the arms — report-only multiplicity-honest ranking (R69).

The Hansen-Lunde-Nason (2011) Model Confidence Set returns the *set* of arms that are statistically
indistinguishable from the best at level ``size`` — "a confidence interval for the best model". It is a
genuinely new, multiplicity-aware companion to the confirmatory H2 IUTs: rather than testing a fixed
pairwise contrast, it asks "which arms could be the best, accounting for all the comparisons at once?".
For the *predicted null* this is exactly the right shape of statement — if no arm dominates, the MCS
contains (nearly) all arms, which is the honest multiplicity-corrected way to say "indistinguishable",
complementing the IUT non-rejection and the Bayesian equivalence evidence (R67).

It operates on the **per-seed scores** (the same ~30 per-seed Sharpe / CVaR values the rliable IQM and the
headline IUT consume), NOT per-day returns — so the inferential unit matches the rest of the stack
(per-seed = i.i.d. training-RNG draws). The per-period "loss" is the negated metric (higher score = lower
loss), so the MCS-best set is the high-score set.

Scope: report-only, DISJOINT from the frozen m=6 family; never gates a hypothesis. Built on ``arch`` (an
EXISTING dependency, already trusted as the StationaryBootstrap/optimal_block_length oracle in
``tests/test_inference_crosscheck.py``); DETERMINISTIC via an explicit ``seed`` (``arch`` routes it through
``numpy.random.default_rng``), with ``block_size=1`` because seeds are i.i.d. (no model fit, so none of the
McNeil-Frey optimiser non-determinism the team rejected). Pre-registered as a secondary descriptive ranking
with its loss (per-seed Sharpe for the risk-adjusted dimension, per-seed CVaR for the tail dimension), its
``size``, ``method='R'``, and seed pinned.

Reference: Hansen, Lunde & Nason (2011), *Econometrica* 79:453-497.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

__all__ = ["model_confidence_set"]


def model_confidence_set(
    arm_scores: dict[str, np.ndarray],
    *,
    size: float = 0.10,
    reps: int = 1000,
    seed: int = 0,
    higher_is_better: bool = True,
    bootstrap: str = "stationary",
    block_size: int = 1,
    method: str = "R",
) -> dict[str, Any]:
    """The Model Confidence Set over the arms on their per-seed scores (report-only).

    Parameters
    ----------
    arm_scores : dict[str, np.ndarray]
        ``{arm: per-seed score array}``. Every array must be the SAME length (paired by the shared training
        seed) and have at least 2 entries; at least 2 arms are required.
    size : float
        MCS level; the set contains every arm whose elimination p-value exceeds ``size`` (default 0.10).
    reps, seed, bootstrap, block_size, method :
        Forwarded to :class:`arch.bootstrap.MCS`. ``seed`` pins determinism; ``block_size=1`` = i.i.d.
        resampling of the per-seed scores; ``method='R'`` is the authors' recommended range statistic.
    higher_is_better : bool
        If True (Sharpe, CVaR-as-return where less-negative = safer), the loss is the negated score so the
        confidence set is the high-score set.

    Returns
    -------
    dict
        ``{included, excluded, pvalues, best_arm, best_in_set, n_seeds, size, method}``. ``pvalues`` maps
        each arm to its MCS elimination p-value (the best arm's is 1.0). For a true null the set typically
        contains (almost) all arms — the multiplicity-honest "indistinguishable" statement.
    """
    from arch.bootstrap import MCS

    arms = list(arm_scores)
    if len(arms) < 2:
        raise ValueError("model_confidence_set needs >= 2 arms")
    cols = [np.asarray(arm_scores[a], dtype=float).ravel() for a in arms]
    lengths = {c.size for c in cols}
    if len(lengths) != 1:
        raise ValueError(f"all arms must have the same number of paired per-seed scores; got {lengths}")
    n_seeds = cols[0].size
    if n_seeds < 2:
        raise ValueError("need >= 2 paired per-seed scores per arm")
    # The FOURTH precondition, added 2026-07-26 (deep review, #74). The three above are validated with a
    # clear message; a non-finite score was not, and it does not degrade gracefully — it reaches
    # ``arch.bootstrap.MCS`` and surfaces as ``IndexError: index 0 is out of bounds for axis 0 with size
    # 0`` from inside the dependency (MEASURED, one NaN in one arm). The analysis caller wraps this block
    # in a broad ``except`` that records ``str(exc)`` as the block's reason, so that opaque message is
    # what an operator would be left with for a whole registered report-only analysis. LATENT, not live:
    # ``analyze_campaign._test_returns`` rejects any record whose test series is non-finite, so no
    # production path can currently deliver a NaN score here — this closes the contract gap, it does not
    # fix an observed failure.
    unusable = [a for a, c in zip(arms, cols) if not np.isfinite(c).all()]
    if unusable:
        raise ValueError(
            f"non-finite per-seed score(s) in arm(s) {unusable}; MCS needs finite scores for every arm"
        )

    scores = np.column_stack(cols)  # (n_seeds, n_arms)
    losses = -scores if higher_is_better else scores

    mcs = MCS(
        losses, size=size, reps=reps, block_size=block_size,
        method=cast(Any, method), bootstrap=cast(Any, bootstrap), seed=seed,
    )
    mcs.compute()

    pvalues = {arms[cast(int, i)]: float(p) for i, p in mcs.pvalues["Pvalue"].items()}
    included = [arms[cast(int, i)] for i in mcs.included]
    excluded = [arms[cast(int, i)] for i in mcs.excluded]
    means = {a: float(np.asarray(arm_scores[a], dtype=float).mean()) for a in arms}
    best_arm = max(arms, key=lambda a: means[a] if higher_is_better else -means[a])

    return {
        "status": "ok",
        "included": included,
        "excluded": excluded,
        "pvalues": pvalues,
        "best_arm": best_arm,
        "best_in_set": best_arm in included,
        "n_seeds": int(n_seeds),
        "size": float(size),
        "method": f"MCS-{method}",
    }
