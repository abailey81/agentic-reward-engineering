"""k-seed search-candidate aggregation (B-A2 — the σ_seed selection-noise denoiser).

The σ_D pilot MEASURED that seed-to-seed noise (σ_seed≈0.244) dominates the SESOI, and that noise
enters the study TWICE: once at selection (which candidate wins) and once at the test leg. The test
leg is already denoised by the uniform-n seeds; this closes the SELECTION channel — each search
candidate is trained at ``k`` seeds ``{run_seed, run_seed+1, …, run_seed+k-1}`` and evaluated on the
**IQM** of its per-seed scores, so the winner (and the LLM's reflection target) is a k-seed aggregate,
not a single lucky/unlucky draw. Strictly exceeds Eureka's 1-policy-per-reward selection rigor.

DECIDED semantics (PLAN §12 B-A2), implemented here as the SINGLE source both the cluster and laptop
search paths call (so they cannot drift):

* **valid iff ALL k seeds ok** — a candidate with any failed seed is FAILED into the matched-budget
  ledger (no partial-aggregate that would bias selection toward lucky-subset candidates);
* **fitness = IQM of the k per-seed ``val_fitness``** (``src.inference.bootstrap.iqm``) — the SELECTION
  metric AND the LLM reflection metric;
* **PBO vector = per-period MEAN of the k seeds' ``val_returns``** (same T; the per-candidate series
  the DSR/PBO machinery consumes);
* **fed tail block = the 6 frozen stats recomputed on the CONCATENATION of the k seeds' TRAIN-window
  returns** (``metrics.train_returns``, emitted by the worker under the k>1 specs' ``emit_train_returns``
  flag and archived alongside ``val_returns``).

FED-TAIL WINDOW — DECIDED 2026-07-08.
The k-seed fed tail is fit on the **TRAIN** window, matching the single-seed construct exactly
(``parallel.train_candidate``: ``ReturnDistribution().fit(train)``; the estimator's own signature is
``fit(train_realized_returns)``). Rationale: the ratified design is **fed in-sample / scored
out-of-sample** — a named selection-overfitting defense (CH4 threats table; the endogeneity
honesty disclosure). Feeding VAL-derived tails would hand the author richer information about the SELECTION
set (more overfitting pressure) and make the k=1 and k>1 constructs inconsistent. The plan §12 B-A2
draft said "val" — a drafting slip, corrected there with a dated note. Net k=3 claim for the paper:
the fed tail is estimated on 3× the in-sample data (better EVT reliability), construct unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["aggregate_k_seeds", "candidate_seed_ids"]


def candidate_seed_ids(cid: str, k: int, base_seed: int) -> list[tuple[str, int]]:
    """The ``k`` ``(run_id, seed)`` pairs for candidate ``cid``: ``{cid}-s{j}`` at ``base_seed+j``.

    k=1 keeps the bare ``cid`` / ``base_seed`` (byte-identical to the single-seed archive layout);
    k>1 fans out to ``{cid}-s0..s{k-1}`` at consecutive seeds (mirrors the test-leg ``{arm}-s{seed}``).
    """
    if k <= 1:
        return [(cid, int(base_seed))]
    return [(f"{cid}-s{j}", int(base_seed) + j) for j in range(k)]


def aggregate_k_seeds(cid: str, arm: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the ``k`` per-seed training results for ONE candidate into its candidate result.

    All-or-nothing: if any seed is missing/failed, the candidate is FAILED. Otherwise returns the
    IQM-fitness / mean-PBO-vector / concatenated-fed-tail result the selection + reflection consume.
    """
    from src.feedback.measurement import ReturnDistribution
    from src.inference.bootstrap import iqm

    if not per_seed or any(not r or not r.get("ok") for r in per_seed):
        return {
            "ok": False, "candidate_id": cid, "arm": arm, "failed_validation": False,
            "error": f"k-seed candidate {cid}: not all {len(per_seed)} seeds ok",
        }
    fitnesses = np.asarray([float(r["fitness"]) for r in per_seed], dtype=float)
    val_seqs = [np.asarray(r.get("val_returns") or [], dtype=float) for r in per_seed]
    if any(v.size == 0 for v in val_seqs):
        return {"ok": False, "candidate_id": cid, "arm": arm,
                "error": f"k-seed candidate {cid}: a seed archived no val_returns"}
    train_seqs = [np.asarray(r.get("train_returns") or [], dtype=float) for r in per_seed]
    if any(t.size == 0 for t in train_seqs):
        # Fail LOUD: the k>1 specs set emit_train_returns, so a missing train vector means a spec/
        # worker drift — silently falling back to val would change the manipulated variable's window.
        return {"ok": False, "candidate_id": cid, "arm": arm,
                "error": f"k-seed candidate {cid}: a seed archived no train_returns (the k-seed "
                         f"fed tail is TRAIN-window by design; emit_train_returns must be set)"}

    agg_fitness = float(iqm(fitnesses))
    # PBO per-candidate series = per-period MEAN across seeds (truncate to the common length so a
    # ragged seed — should not happen with a fixed window, but guard — cannot misalign the average).
    t_common = min(v.size for v in val_seqs)
    mean_vr = np.mean(np.stack([v[:t_common] for v in val_seqs], axis=0), axis=0)
    # FED tail = the 6 frozen stats on the CONCATENATION of the seeds' TRAIN-window returns —
    # fed in-sample / scored out-of-sample, the single-seed construct at 3x the tail data.
    concat = np.concatenate(train_seqs)
    tail_stats = ReturnDistribution().fit(concat).tail_stats()

    out: dict[str, Any] = {
        "ok": True, "candidate_id": cid, "arm": arm,
        "fitness": agg_fitness,
        "val_returns": [float(x) for x in mean_vr],
        "tail_stats": tail_stats,
        "reward_source": per_seed[0].get("reward_source", ""),
        "reward_hash": per_seed[0].get("reward_hash", ""),
        "k_seeds": len(per_seed),
        "per_seed_fitness": [float(f) for f in fitnesses],
    }
    return out
