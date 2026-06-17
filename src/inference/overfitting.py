"""Probability of Backtest Overfitting via CSCV (FINAL_PLAN F.11, B.9).

Purpose
-------
Estimate the Probability of Backtest Overfitting (PBO) for a family of
candidate strategies using Combinatorially Symmetric Cross-Validation (CSCV).
This is the PRIMARY overfitting guard for the project: under a guided LLM
search the number of trials is ill-defined, so a rank-based, trial-count-free
diagnostic (PBO) is preferred to the Deflated Sharpe Ratio for the headline
overfitting claim.

Algorithm (PBO / CSCV; Bailey et al. 2017)
-------------------------------------------
Given a performance matrix of shape ``(T, N)`` (``T`` time observations of a
per-period performance statistic for each of ``N`` candidates):

1. Partition the ``T`` rows into ``S`` disjoint, contiguous, equal-size blocks.
2. For every way of choosing ``S/2`` blocks as in-sample (IS) -- the
   complementary ``S/2`` blocks are out-of-sample (OOS) -- form the IS/OOS
   split.
3. On each split: pick the candidate that is best in-sample (highest mean IS
   performance); find its relative rank ``omega`` in the OOS ordering, mapped to
   ``(0, 1)``.
4. Map to the logit ``lambda = ln(omega / (1 - omega))``.
5. PBO is the fraction of splits with ``lambda < 0`` -- i.e. the IS-best
   candidate lands *strictly* below the OOS median. (FINAL_PLAN B.9 and Bailey
   et al. 2017 use the strict inequality, so an exact OOS-median tie, ``lambda
   == 0``, does NOT count as overfit.)

PBO lies in ``[0, 1]``; values near ``0.5`` or above indicate that in-sample
selection carries no OOS information (severe overfitting).

If ``C(S, S/2)`` is too large to enumerate, a capped random sample of
combinations is drawn with ``rng`` (documented; ``max_combinations``).
"""

from __future__ import annotations


import itertools
import math

import numpy as np

__all__ = ["pbo"]

_MAX_COMBINATIONS = 4000


def pbo(
    perf_matrix: np.ndarray,
    n_blocks: int = 16,
    *,
    rng: np.random.Generator | None = None,
    max_combinations: int = _MAX_COMBINATIONS,
) -> float:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    perf_matrix:
        Array of shape ``(T_obs, N_configs)``: ``T_obs`` time observations of a
        per-period performance statistic for each of ``N_configs`` candidates.
    n_blocks:
        Number of CSCV blocks ``S`` (must be even, ``>= 2``); rows are
        partitioned into ``S`` contiguous equal-size blocks and combined
        ``S/2`` at a time.
    rng:
        Optional NumPy generator; used only when the number of IS/OOS
        combinations exceeds ``max_combinations`` and a random subset is drawn.
    max_combinations:
        Cap on the number of IS/OOS splits actually evaluated. When
        ``C(S, S/2)`` exceeds this, that many splits are sampled at random.

    Returns
    -------
    float
        PBO estimate in ``[0, 1]``: the fraction of IS/OOS splits whose
        in-sample-best candidate has a *negative* OOS rank logit (FINAL_PLAN B.9).
    """
    perf = np.asarray(perf_matrix, dtype=float)
    if perf.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T_obs, N_configs)")
    t_obs, n_cfg = perf.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 2")
    if t_obs < n_blocks:
        raise ValueError("need at least n_blocks rows")

    s = n_blocks
    # Contiguous, equal-size blocks (drop the remainder rows so blocks are equal).
    block_size = t_obs // s
    # Equal-size contiguous blocks; the trailing remainder rows are dropped by slicing.
    blocks = [
        perf[k * block_size : (k + 1) * block_size, :] for k in range(s)
    ]

    half = s // 2
    all_block_ids = range(s)
    n_total = math.comb(s, half)

    if n_total <= max_combinations:
        combos = list(itertools.combinations(all_block_ids, half))
    else:
        if rng is None:
            rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)
        seen: set[tuple[int, ...]] = set()
        block_arr = np.arange(s)
        while len(seen) < max_combinations:
            pick = tuple(sorted(rng.choice(block_arr, size=half, replace=False).tolist()))
            seen.add(pick)
        combos = list(seen)

    logits_negative = 0
    n_splits = 0
    for is_ids in combos:
        is_set = set(is_ids)
        oos_ids = [k for k in all_block_ids if k not in is_set]

        is_data = np.concatenate([blocks[k] for k in is_ids], axis=0)
        oos_data = np.concatenate([blocks[k] for k in oos_ids], axis=0)

        is_perf = is_data.mean(axis=0)
        oos_perf = oos_data.mean(axis=0)

        best = int(np.argmax(is_perf))

        # OOS rank of the IS-best config, in (0, 1). Use average rank to handle
        # ties robustly, then map to the open unit interval.
        order = np.argsort(oos_perf, kind="mergesort")
        ranks = np.empty(n_cfg, dtype=float)
        ranks[order] = np.arange(1, n_cfg + 1)
        # Average ranks for ties.
        # (Build via the standard "rankdata average" trick.)
        sorted_vals = oos_perf[order]
        i = 0
        while i < n_cfg:
            j = i
            while j + 1 < n_cfg and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg = (i + 1 + j + 1) / 2.0
                ranks[order[i : j + 1]] = avg
            i = j + 1

        omega = ranks[best] / (n_cfg + 1.0)  # in (0, 1)
        lam = math.log(omega / (1.0 - omega))
        # FINAL_PLAN B.9 + Bailey et al. 2017: PBO counts splits whose logit is
        # STRICTLY < 0 (the IS-best lands strictly below the OOS median); an exact
        # OOS-median tie (lam == 0) does NOT count as overfit.
        if lam < 0.0:
            logits_negative += 1
        n_splits += 1

    return float(logits_negative / n_splits)
