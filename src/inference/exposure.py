"""Exposure / allocation summaries of a rolled-out policy's per-step portfolio weights.

Pure, deterministic numpy reductions of a ``(T, N)`` per-step weight matrix (the sealed test leg's
frozen winner, collected by :func:`src.env.runner.rollout_port_diagnostics`) into SMALL, archivable
summaries — so the frozen, replay-only campaign records the data the allocation / exposure / concentration
figures need without persisting the full ~1.5M-float matrix per record (M1/M1b,
the run-time diagnostics capture).

Two summaries, both cash-index-agnostic (they read only the weight vector, never assuming which column is
cash), so they are robust to the env's simplex convention:

* :func:`exposure_series` — per-step concentration scalars (Herfindahl, effective #positions, max weight,
  top-5 concentration). ~4xT floats -> archived for EVERY sealed-leg record.
* :func:`alloc_snapshots` — the top-K assets by mean weight, sampled at ~monthly snapshots, for the
  allocation heatmap (Cartea/Coache/RAMAC archetype). Small; archived per record.

Nothing here touches training or the RNG; it post-processes an already-rolled trajectory.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["exposure_series", "alloc_snapshots"]


def exposure_series(weights: Any) -> dict[str, list[float]]:
    """Per-step concentration scalars from a ``(T, N)`` weight matrix.

    Returns
    -------
    dict[str, list[float]]
        ``{"hhi", "eff_n", "max_weight", "top5"}``, each a length-``T`` list:
        ``hhi`` = Herfindahl :math:`\\sum_i w_i^2`; ``eff_n`` = :math:`1/\\text{hhi}` (effective number of
        positions); ``max_weight`` = the single largest holding; ``top5`` = sum of the 5 largest.
        An empty / degenerate input yields empty lists (never raises).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.size == 0:
        return {"hhi": [], "eff_n": [], "max_weight": [], "top5": []}
    sq = w * w
    hhi = sq.sum(axis=1)
    eff_n = np.where(hhi > 0.0, 1.0 / hhi, 0.0)
    max_w = w.max(axis=1)
    k = min(5, w.shape[1])
    # top-5 concentration: sum of the k largest per row (partition is O(N), order within top-k irrelevant)
    top5 = np.sort(w, axis=1)[:, -k:].sum(axis=1)
    return {
        "hhi": hhi.tolist(),
        "eff_n": eff_n.tolist(),
        "max_weight": max_w.tolist(),
        "top5": top5.tolist(),
    }


def alloc_snapshots(weights: Any, *, top_k: int = 20, n_snapshots: int = 48) -> dict[str, Any]:
    """Top-K-by-mean allocation, sampled at ~``n_snapshots`` evenly-spaced steps — the heatmap source.

    Picks the ``top_k`` assets with the largest MEAN weight over the whole path (a coherent, fixed asset
    set so the heatmap rows are stable over time), then samples their weights at ``n_snapshots`` evenly
    spaced time indices. ``other`` carries the residual mass (``1 - sum(top_k)``) so the column still sums
    to the invested fraction.

    Returns
    -------
    dict[str, Any]
        ``{"asset_idx": [K ints], "steps": [S ints], "weights": [[S x K]], "other": [S floats]}`` where
        ``weights[s][j]`` is the weight of asset ``asset_idx[j]`` at step ``steps[s]``. Empty dict-of-lists
        on a degenerate input (never raises).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.size == 0:
        return {"asset_idx": [], "steps": [], "weights": [], "other": []}
    t, n = w.shape
    k = int(min(max(1, top_k), n))
    mean_w = w.mean(axis=0)
    top_idx = np.argsort(mean_w)[::-1][:k]
    top_idx = np.sort(top_idx)  # stable, ascending asset order for a readable heatmap
    s = int(min(max(1, n_snapshots), t))
    step_idx = np.unique(np.linspace(0, t - 1, s).round().astype(int))
    sub = w[np.ix_(step_idx, top_idx)]           # (S, K)
    other = 1.0 - sub.sum(axis=1)                 # residual mass outside the top-K
    return {
        "asset_idx": top_idx.astype(int).tolist(),
        "steps": step_idx.astype(int).tolist(),
        "weights": sub.tolist(),
        "other": other.tolist(),
    }
