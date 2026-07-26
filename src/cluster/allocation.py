"""The allocation ADVISOR — the BRAIN of the adaptive speed system (2026-07-24).

Pure decision logic over a telemetry :class:`~src.cluster.telemetry.Snapshot` + our own measured
throughput, emitting THE ALLOCATION PLAN: pool assignments, the exact ``--seed-pool-blocks``
stripe string, the chunking regime, pack recommendations, and recomputed milestone ETAs.

INVARIANTS ENCODED (so every recommendation is legal by construction):
- NEVER emits any priority action (the standing priority rule; ``-p`` is not in this module's alphabet).
- CRN device homogeneity: stripes are contiguous pool-homogeneous blocks WITHIN each rung
  segment (the ratified block structure); pool/pack changes apply to NEW batches only.
- ADVISORY ONLY: the plan is printed/archived; applying it is an explicit human/launcher act.

Ground truth: the Myriad scheduler dossier (the priority formula, the two-regime
chunking doctrine §3b, pool speeds §2) + the ratified runbook levers (§10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.cluster.telemetry import CONTENDED_QW, POOL_SPEED, Snapshot

#: Conservative pack depth by GPU VRAM GB (validated pack-5 baseline on the pilot pool; deeper
#: packs UNLOCK ONLY once the canary's archived nvidia-smi confirms per-training VRAM headroom).
PACK_BY_VRAM_GB = {16: 5, 32: 6, 40: 8, 80: 10}

#: Default per-pool VRAM (GB) until the canary measures the E-node cards (16 vs 32 unresolved).
POOL_VRAM_DEFAULT = {"EF": 16, "L": 40, "U": 80, "V": 80}


@dataclass
class Plan:
    """The advisor's output — human-readable via :meth:`render`, machine-usable as fields."""

    regime: str                          # "CONTENDED" | "QUIET"
    chunk_tasks: int                     # recommended --chunk-tasks for NEW submissions
    search_pool: str                     # the latency-critical lane's pool pin
    stripe_pools: list[str]              # pools included in the test stripe, best-first
    seed_pool_blocks: str                # the exact --seed-pool-blocks string
    pack: dict[str, int]                 # pool -> recommended pack (canary-gated above 5)
    pack_note: str
    eta_days: dict[str, float] | None    # milestone -> days-from-now (None until measured)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"REGIME: {self.regime}  ->  --chunk-tasks {self.chunk_tasks}",
            f"SEARCH LANE: --pool {self.search_pool}",
            f"TEST STRIPE POOLS (best-first): {','.join(self.stripe_pools)}",
            f"--seed-pool-blocks \"{self.seed_pool_blocks}\"",
            f"PACK: {self.pack}  ({self.pack_note})",
        ]
        if self.eta_days:
            lines.append("ETA (days from now, EACH-IF-RUN-ALONE at the measured rate): "
                         + ", ".join(f"{k}={v:.1f}" for k, v in self.eta_days.items()))
        lines += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


def plan_delta(old: dict | None, new: dict) -> list[str]:
    """Actionable CHANGES between two plans' key fields — the watch loop's alert source.

    Pure + testable. Compares only decision-relevant fields; returns loud human lines
    (empty list = nothing actionable happened).
    """
    if not old:
        return [f"INITIAL PLAN: regime={new.get('regime')} search_pool={new.get('search_pool')}"]
    out: list[str] = []
    if old.get("regime") != new.get("regime"):
        out.append(f"*** REGIME FLIP: {old.get('regime')} -> {new.get('regime')} "
                   f"(re-chunk NEW submissions: --chunk-tasks {new.get('chunk_tasks')})")
    if old.get("search_pool") != new.get("search_pool"):
        out.append(f"*** SEARCH-LANE POOL: {old.get('search_pool')} -> {new.get('search_pool')}")
    if set(old.get("stripe_pools", [])) != set(new.get("stripe_pools", [])):
        out.append(f"*** STRIPE POOLS CHANGED: {old.get('stripe_pools')} -> "
                   f"{new.get('stripe_pools')} (U/V unlock or a pool emptied)")
    if old.get("pack") != new.get("pack"):
        out.append(f"*** PACK CHANGED: {old.get('pack')} -> {new.get('pack')} "
                   "(apply to NEW batches only)")
    return out


def usable_pools(snap: Snapshot) -> dict[str, int]:
    """Pools we may schedule on, with free-GPU counts. U/V join ONLY on a RUNNING/ran probe
    verdict (the runbook §10 branch); EF/L are always ours (documented allow=EF|L)."""
    from src.cluster.telemetry import pool_is_usable

    pools = {p: snap.pool_free.get(p, 0) for p in ("EF", "L")}
    for pool in ("U", "V"):
        # POSITIVE admission test, not a word scan (deep review 2026-07-26, #60). This was
        # ``"USABLE" in verdict`` — a case-sensitive substring of prose written in telemetry.py, so a
        # negative reworded to "NOT USABLE"/"UNUSABLE" would have ADMITTED the pool (reproduced).
        # The rule now lives beside the vocabulary it tests, and refuses anything it does not
        # positively recognise.
        if pool_is_usable(snap.probe_states.get(pool)):
            pools[pool] = snap.pool_free.get(pool, 0)
    return pools


def chunking(snap: Snapshot, *, prev_regime: str | None = None) -> tuple[str, int]:
    """The two-regime doctrine (dossier §3b) WITH HYSTERESIS: enter CONTENDED above 1.2x the
    threshold, exit below 0.8x — a noisy boundary can never flap the regime between reads."""
    enter, exit_ = CONTENDED_QW * 1.2, CONTENDED_QW * 0.8
    if prev_regime == "CONTENDED":
        contended = snap.cluster_qw > exit_
    elif prev_regime == "QUIET":
        contended = snap.cluster_qw >= enter
    else:
        contended = snap.cluster_qw >= CONTENDED_QW
    # NOTE: RATIONALE CORRECTED 2026-07-26 (the recommendation is unchanged; its REASON was wrong).
    # This returned chunk-25 because "priority-per-job dominates" under contention — the ticket-
    # concentration doctrine. That doctrine is REFUTED by a controlled test (dossier §0-PRE M5):
    # holding 228 of 309 pending jobs moved our top eligible priority only 2.0165 -> 2.0413
    # (waiting-time accrual, which happens anyway) and placed ZERO 32-core jobs, while 8-core jobs
    # place 67-at-once. Our jobs sit at the cluster priority FLOOR whatever we do. Chunking buys
    # NO priority.
    # The recommendation SURVIVES on a different, real constraint: `max_u_jobs = 1000`. Chunk-1
    # with pipelined rungs can enqueue ~1,200 arrays from the core line alone (runbook §10's C5
    # check), and a cap hit mid-run classes as a transport error. So chunk BIG under contention to
    # stay under the job cap — never to buy priority.
    if contended:
        return "CONTENDED", 25       # few, heavy arrays: stay under max_u_jobs (NOT for priority)
    return "QUIET", 1                # the mode-D flood: 1-task chunks ramp 1/job/cycle in parallel


def advise_cpu_lane(snap: Snapshot, *, rung: int = 568,
                    retreat_cap_cores: int | None = None) -> dict[str, Any]:
    """The CPU-lane recommendation: how many cores to hold, in what shape, and the makespan.

    Closes the gap that left the 2026-07-26 capacity findings applied BY HAND at GO. Composes the
    two modules that own the halves — ``killswitch.plan_footprint`` (how much to take, given live
    pressure and the courtesy reserve) and ``lanes.plan_lanes`` (what the makespan then is, and
    which lever is binding) — and adds the measured job shape.

    Reads ``snap.cpu_free`` for FREE CORES in the confirmatory pools only (``d`` + ``b``; ``t`` is
    excluded on CRN grounds and GPU-node cores are never harvested — ``lanes.EXCLUDED_CPU_POOLS``).
    An empty ``cpu_free`` means "unknown", NOT "zero": the advisory then reports ``target_cores:
    None`` rather than silently recommending a stall.
    """
    from src.cluster.killswitch import plan_footprint
    from src.cluster.lanes import (CONFIRMATORY_CPU_POOLS, CPU_CHAIN_THREADS, plan_lanes)

    known = {p: int(snap.cpu_free.get(p, 0)) for p in CONFIRMATORY_CPU_POOLS if p in snap.cpu_free}
    if not known:
        return {"target_cores": None, "why": "cpu_free unknown (telemetry section empty/failed)",
                "free_cores": None, "pools": list(CONFIRMATORY_CPU_POOLS)}

    free = sum(known.values())
    target, why = plan_footprint(free_cores=free, pending_jobs=int(snap.cluster_qw),
                                 retreat_cap_cores=retreat_cap_cores)
    plan = plan_lanes(rung=rung, cpu_cores=max(1, target), chain_threads=CPU_CHAIN_THREADS)
    return {
        "pools": list(CONFIRMATORY_CPU_POOLS),
        "free_by_pool": known,
        "free_cores": free,
        "target_cores": target,
        "why": why,
        # the measured job shape - see the dossier §0-PRE M2/M4c and runbook §11.1
        "job_shape": "-pe smp 8 -l mem=2G (NEVER smp 36: it requests an EXCLUSIVE whole node)",
        "flood_threads": 1,
        "chain_threads": CPU_CHAIN_THREADS,
        "makespan_days": round(plan.makespan_days, 2),
        "binding": plan.binding,
        "saturation_cores": round(plan.saturation_cores),
    }


def pick_search_pool(pools: dict[str, int]) -> str:
    """The latency-critical chain lane goes to the FASTEST pool with real headroom (>=2 free
    GPUs so a chain never waits on its own lane); falls back to EF (always-eligible)."""
    candidates = sorted(pools.items(), key=lambda kv: -POOL_SPEED.get(kv[0], 0.0))
    for pool, free in candidates:
        if free >= 2:
            return pool
    return "EF"


def stripe(seed_segments: list[tuple[int, int]], pools: dict[str, int],
           pack: dict[str, int] | None = None) -> tuple[list[str], str]:
    """Generate the exact ``--seed-pool-blocks`` string, throughput-weighted.

    Each rung segment ``[a, b]`` (inclusive) is split into contiguous pool-homogeneous blocks
    sized by each pool's effective throughput share (free_gpus x speed x pack/5) — the CRN
    blocked-design invariant of the ratified stripes, generalized to N pools. Pools with zero
    weight are dropped. Returns (pools-used-best-first, the blocks string).
    """
    pack = pack or {}
    weights = {p: f * POOL_SPEED.get(p, 1.0) * (pack.get(p, 5) / 5.0)
               for p, f in pools.items() if f > 0}
    if not weights:                       # nothing free right now -> single-pool EF fallback
        weights = {"EF": 1.0}
    order = sorted(weights, key=lambda p: -weights[p])
    total = sum(weights.values())
    parts: list[str] = []
    for a, b in seed_segments:
        n = b - a + 1
        cursor = a
        for i, p in enumerate(order):
            share = n * weights[p] / total
            take = n - (cursor - a) if i == len(order) - 1 else max(1, round(share))
            take = min(take, b - cursor + 1)
            if take <= 0:
                continue
            parts.append(f"{p}:{cursor}-{cursor + take - 1}")
            cursor += take
            if cursor > b:
                break
    return order, ",".join(parts)


def recommend_pack(pools: dict[str, int], *, measured_vram_per_training_gb: float | None,
                   pool_vram: dict[str, int] | None = None) -> tuple[dict[str, int], str]:
    """Pack depth per pool. WITHOUT a canary measurement everything stays at the validated 5;
    WITH one, depth = floor(VRAM * 0.9 / per-training) capped by PACK_BY_VRAM_GB."""
    vram = pool_vram or POOL_VRAM_DEFAULT
    if measured_vram_per_training_gb is None or measured_vram_per_training_gb <= 0:
        return {p: 5 for p in pools}, "pack-5 everywhere (canary VRAM measurement pending)"
    out: dict[str, int] = {}
    for p in pools:
        fit = int((vram.get(p, 16) * 0.9) // measured_vram_per_training_gb)
        out[p] = max(1, min(fit, PACK_BY_VRAM_GB.get(vram.get(p, 16), 5)))
    return out, f"from measured {measured_vram_per_training_gb:.1f}G/training (canary)"


def eta(remaining_trainings: dict[str, int],
        measured_trainings_per_day: float | None) -> dict[str, float] | None:
    """Milestone ETAs from OUR measured throughput (records/day). None until measured — the
    advisor never invents a rate (the accuracy rule)."""
    if not measured_trainings_per_day or measured_trainings_per_day <= 0:
        return None
    return {k: v / measured_trainings_per_day for k, v in remaining_trainings.items()}


def advise(snap: Snapshot, *, seed_segments: list[tuple[int, int]] | None = None,
           measured_vram_per_training_gb: float | None = None,
           remaining_trainings: dict[str, int] | None = None,
           measured_trainings_per_day: float | None = None,
           prev_regime: str | None = None,
           pool_vram: dict[str, int] | None = None) -> Plan:
    """The full plan from one snapshot (+ optional measured facts)."""
    segments = seed_segments or [(0, 29), (30, 99), (100, 188), (189, 278),
                                 (279, 339), (340, 402), (403, 567)]
    pools = usable_pools(snap)
    if snap.errors and prev_regime:
        # Audit M3: degraded telemetry must never flip decisions — freeze on the last known
        # regime and say so, rather than advising QUIET off a phantom-empty queue.
        regime, chunk = chunking(snap, prev_regime=prev_regime)
        if regime != prev_regime:
            regime = prev_regime
            chunk = 25 if regime == "CONTENDED" else 1
    elif snap.errors:
        # FIRST READ with degraded telemetry (deep review 2026-07-26, #61). M3 above only fires when
        # there IS a last-known regime; with none, a failed PENDING section parses as qw=0 and the
        # doctrine reads that phantom-empty queue as QUIET — the AGGRESSIVE chunk-1 flood. That is
        # the wrong direction under unknown pressure: the constraint chunking actually protects is
        # ``max_u_jobs = 1000``, and chunk-1 with pipelined rungs can enqueue ~1,200 arrays, where a
        # cap hit classes as a transport error. CONTENDED (few, heavy arrays) cannot breach it.
        # This path is not hypothetical: the runbook has the advisor run AT LAUNCH, which is both a
        # first read and the moment ssh/qstat is most likely to hiccup.
        regime, chunk = "CONTENDED", 25
    else:
        regime, chunk = chunking(snap, prev_regime=prev_regime)
    pack, pack_note = recommend_pack(pools, measured_vram_per_training_gb=measured_vram_per_training_gb,
                                     pool_vram=pool_vram)
    order, blocks = stripe(segments, pools, pack)
    plan = Plan(
        regime=regime,
        chunk_tasks=chunk,
        search_pool=pick_search_pool(pools),
        stripe_pools=order,
        seed_pool_blocks=blocks,
        pack=pack,
        pack_note=pack_note,
        eta_days=eta(remaining_trainings or {}, measured_trainings_per_day),
    )
    for pool in ("U", "V"):
        v = snap.probe_states.get(pool, "")
        if v and "USABLE" not in v:
            plan.notes.append(f"pool {pool}: {v}")
    if snap.errors:
        plan.notes.append(f"telemetry degraded: {snap.errors}")
    if plan.regime == "QUIET":
        # Audit M4: the flood is the fast ramp, but the doctrine's own cap still binds.
        plan.notes.append("QUIET cap: keep TOTAL pending jobs < 1000 (max_u_jobs) — the "
                          "pipelined-rung flood at chunk-1 can exceed it; chunk the rung "
                          "blocks up if the pending count approaches the cap")
    if snap.errors and prev_regime:
        plan.notes.append(f"DEGRADED TELEMETRY — regime frozen at {prev_regime}; "
                          "act only on a clean snapshot")
    elif snap.errors:
        plan.notes.append("DEGRADED TELEMETRY on a FIRST read (no prior regime) — defaulted to "
                          "CONTENDED/chunk-25, the setting that cannot breach max_u_jobs. Do NOT "
                          "read this as a measurement of cluster pressure; re-run for a clean one")
    plan.notes.append("advisory only — apply at batch boundaries; priorities untouched")
    return plan
