"""Run the ENTIRE bayes_opt GP-EI chain inside ONE cluster job (the campaign's critical path).

WHY. ``bayes_opt`` is H4b's control arm and the campaign's LONGEST serial path: ``n_init=5``
parallel points plus **25 strictly sequential** GP-EI iterations, each a full 400k-step training
(``src/search/bayes_opt.py``). Once CPU capacity passes ~1,640 cores this chain — not throughput —
is what the whole campaign waits on (``src/cluster/lanes.py``).

The driver-side implementation (``campaign.run_family_search_arm``) trains each proposal as its own
**array-of-1 cluster job**, so the chain pays a QUEUE WAIT 30 times over. On CPU that is tolerable
(~10 min/cycle, ~2%). On GPU it is fatal: our GPU jobs have queued for hours-to-days, so 25
dispatches at even 2 h each bolt ~50 h of pure waiting onto ~27 h of compute — silently destroying
the single biggest speed-up available to the campaign. Running the loop ON THE NODE collapses that
to ONE queue wait.

WHAT THIS DOES **NOT** CHANGE — the science is untouched, deliberately and completely:

* the SAME ``bayes_opt_over_template`` GP (Matern-2.5 EI, ``n_init=5``, Snoek et al. 2012),
* the SAME seed, so the SAME proposal sequence,
* the SAME matched candidate budget and the SAME ``B*`` per candidate,
* the SAME certified training path — ``run_one._run_single`` after ``_worker_init()``, i.e. exactly
  what a normal task executes (the ``pack==1`` inline path already trains specs sequentially
  in-process, and the P18 audit made it share the pooled path's environment contract, so this
  introduces NO new numerical surface),
* the SAME candidate ids (``<arm>-c<i>``) and archive layout, so records written here are
  byte-interchangeable with driver-produced ones and either path can resume the other's work.

Speed here is bought purely by REMOVING QUEUE LATENCY. Nothing about the optimiser, its budget, or
its comparability to the LLM arms is weakened — which matters because ``bayes_opt`` is the control
the LLM must BEAT, so every algorithmic "speed-up" (raising ``n_init``, batch/q-EI, a reduced search
budget) would make our own hypothesis look better and is refused on integrity grounds.

DEADLINE AWARENESS (what makes this work on CPU as well as GPU). SGE caps walltime at 48 h for
2-36 cores and 72 h for 1 core, but the chain needs ~27 h on a GPU and ~214 h on a CPU core. So the
runner takes a wall-clock budget and stops cleanly before it would be killed, having archived every
completed candidate. Because the loop resumes by ARCHIVE REPLAY — completed candidates return their
recorded fitness without retraining — a restart re-derives the identical GP state and continues.
One GPU job finishes the chain; ~4 chained 72 h CPU jobs do the same with 4 queue waits instead
of 25.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["run_bayes_chain", "ChainStopped", "main"]

#: Fitness returned for a candidate whose training failed — matches the driver-side sentinel in
#: ``campaign.run_family_search_arm`` exactly, so the GP sees an identical objective either way.
_FAILED_FITNESS = -1e9

#: Fraction of the wall budget after which we refuse to START another iteration. The reserve
#: absorbs the final archive write plus SGE's notify->SIGKILL grace.
_DEADLINE_SAFETY = 0.92


class ChainStopped(Exception):
    """Internal: unwinds ``bayes_opt_over_template`` when the wall budget is exhausted.

    Raised from inside the objective (the only place that knows an iteration is about to start),
    because the GP loop itself has no pause point. Catching it outside is safe: every completed
    candidate is already archived, so the next job replays them and continues from here.
    """


def run_bayes_chain(
    opts: dict[str, Any],
    *,
    arm: str = "bayes_opt",
    archive_root: str | Path,
    spec_archive_root: str | Path,
    k_seeds: int = 1,
    device: str | None = None,
    deadline_secs: float | None = None,
    est_iter_secs: float | None = None,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Drive the whole GP chain on THIS node; return a status summary.

    Parameters
    ----------
    opts : dict
        The campaign options for the arm (``candidates``, ``seed``, ``proto_cfg``, agent config,
        …) — exactly the dict the driver-side path passes to ``_family_spec``.
    archive_root : path
        The SEARCH archive root (``<root>/search``); candidates land under ``<root>/<arm>/``.
    spec_archive_root : path
        Where per-candidate specs are recorded (mirrors ``run.search_spec_root()``).
    k_seeds : int
        ``search_seeds_per_candidate`` (B-A2). 1 = the byte-identical single-seed default.
    device : str, optional
        ``"cpu"`` / ``"cuda"`` — the SUBSTRATE stamped onto every spec this chain builds. Pass the
        SAME value the job requested: it is part of the determinism envelope (CPU is not
        bit-identical to CUDA), so a mislabelled spec both mis-archives the fingerprint and can
        split one arm's search across two arithmetics. ``None`` leaves the spec bare and lets
        ``run_one`` apply its ``"cuda"`` default — correct ONLY on a GPU job. See ``_chain_specs``.
    deadline_secs : float, optional
        Wall budget. ``None`` = run to completion (use only when the walltime genuinely fits).
    est_iter_secs : float, optional
        Expected seconds per iteration, used for the deadline check BEFORE any iteration has
        completed. Without it the first iteration of each job starts "blind" and may be SIGKILLed
        at the walltime — recoverable (archive replay loses only that one candidate) but wasteful.
        The campaign knows this number from the measured rate: ``train_steps / steps_per_s``, i.e.
        ~8.55 h on a CPU core (13.0 steps/s) or ~1.09 h on a GPU at pack-1 (102 steps/s). PASS IT.

    Returns
    -------
    dict
        ``{"status": "complete"|"partial", "arm", "completed", "failed", "budget",
        "best_score", "elapsed_secs"}``. ``"partial"`` is a NORMAL, successful exit — resubmit the
        same job and it resumes by archive replay.
    """
    from src.baselines.reward_family import family_bounds
    from src.cluster.campaign import (_family_spec, _read_candidate, candidate_failed_before,
                                      record_failed_candidate)
    from src.cluster.lanes import CPU_CHAIN_THREADS
    from src.orchestration.parallel import _worker_init
    from src.search.bayes_opt import bayes_opt_over_template

    import numpy as np

    # The SAME environment contract every normal task gets (pyarrow preload, thread pinning, alloc
    # conf). Without it this path would train under a different thread/alloc regime than the rest of
    # the campaign — a determinism-envelope difference, which is exactly what must not happen.
    #
    # THREADS: ``CPU_CHAIN_THREADS`` (R107 wiring, 2026-07-27). This IS the leg R107 was ratified
    # for — 25 STRICTLY serial GP-EI trainings, the campaign's longest sequential path, where cost
    # is pure LATENCY and aggregate throughput is worthless. At the measured 2.72x the chain falls
    # from 8.9 d to ~3.3 d, which is what moves the core crossover from 1,685 to 4,584. The TEST leg
    # is untouched at 1 thread (``_task_threads`` defaults to 1), so every SCORED contrast keeps one
    # arithmetic. NOTE: The job must request ``CPU_CHAIN_THREADS`` CORES to match, or this is
    # oversubscription and SLOWER than 1 thread — see ``_worker_init``.
    _worker_init(CPU_CHAIN_THREADS)

    arm_root = Path(archive_root) / arm
    budget = max(1, int(opts["candidates"]))
    base_seed = int(opts["seed"])
    t0 = clock()
    state: dict[str, Any] = {"i": 0, "completed": 0, "failed": 0, "durations": []}

    def _out_of_time() -> bool:
        if deadline_secs is None:
            return False
        elapsed = clock() - t0
        # Estimate the next iteration from OBSERVED ones; fall back to the caller's measured
        # estimate so even the first iteration of a job is checked (see est_iter_secs).
        est = ((sum(state["durations"]) / len(state["durations"])) if state["durations"]
               else est_iter_secs)
        remaining = deadline_secs * _DEADLINE_SAFETY - elapsed
        return remaining <= 0 or (est is not None and remaining < est)

    def template_eval(coeffs: Any) -> float:
        i = state["i"]
        state["i"] = i + 1
        cid = f"{arm}-c{i}"
        # ARCHIVE REPLAY FIRST — this is both the resume mechanism and what makes a partial chain
        # free to restart: an already-trained candidate costs nothing and yields the SAME fitness,
        # so the GP re-derives an identical trajectory.
        rec = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
        if rec is not None:
            state["completed"] += 1
            return float(rec["fitness"])
        # A candidate that was ATTEMPTED and FAILED archives no record, so without this marker a
        # resume cannot tell it apart from "never attempted" and RE-TRAINS it. If the retry then
        # succeeds, the GP sees a different fitness at this index than the original job did and every
        # later proposal diverges — MEASURED (deep review #62): an interrupted chain and an
        # uninterrupted one selected entirely different winners. Replaying the sentinel keeps the
        # trajectory identical, which is what the docstring above promises and what
        # "analysis = deterministic archive replay" requires.
        if candidate_failed_before(arm_root, cid):
            state["failed"] += 1
            return _FAILED_FITNESS
        if _out_of_time():
            raise ChainStopped(cid)

        from src.cluster import run_one

        it0 = clock()
        for spec in _chain_specs(arm, cid, list(coeffs), opts, spec_archive_root,
                                 k_seeds, base_seed, _family_spec, device):
            run_one._run_single(spec)          # the certified worker: trains AND archives
        state["durations"].append(clock() - it0)

        rec = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
        if rec is None:
            state["failed"] += 1
            record_failed_candidate(arm_root, cid, reason="no record after _run_single")
            return _FAILED_FITNESS
        state["completed"] += 1
        return float(rec["fitness"])

    stopped = False
    result: dict[str, Any] = {}
    try:
        # EXACTLY the call both reference implementations make (campaign.py:847 and
        # parallel.py:1240): the cfg argument is {"matched_budget": n}, NOT the full opts dict
        # (bayes_opt._budget reads matched_budget/candidate_budget/candidate_budget_total and
        # raises on anything else), and the rng is seeded from opts["seed"] so the proposal
        # sequence is identical to the driver-side chain.
        result = bayes_opt_over_template(
            template_eval, family_bounds(opts.get("proto_cfg")), {"matched_budget": budget},
            rng=np.random.default_rng(base_seed),
        )
    except ChainStopped:
        stopped = True

    return {
        "status": "partial" if stopped else "complete",
        "arm": arm,
        "budget": budget,
        "completed": int(state["completed"]),
        "failed": int(state["failed"]),
        "best_score": (None if stopped else result.get("best_score")),
        "elapsed_secs": round(clock() - t0, 1),
    }


def _chain_specs(arm: str, cid: str, coeffs: list, opts: dict, spec_archive_root: str | Path,
                 k_seeds: int, base_seed: int, family_spec: Any,
                 device: str | None = None) -> list[dict[str, Any]]:
    """Per-candidate specs — identical in shape to ``campaign._family_specs`` (k-seed aware).

    ``device`` STAMPS THE SUBSTRATE (deep review, 2026-07-27). Without it these specs carry no
    ``device`` key at all — ``opts`` has none (its keys are train_steps…window) and
    ``parallel._spec`` adds none — so they reach ``run_one._run_single`` bare and are defaulted
    there to ``"cuda"``. The driver-side path never has this problem because ``campaign`` injects
    the device onto the specs IT builds; the chain builds its own ON-NODE and bypasses that
    entirely. On the CPU lane the request would find no GPU and fall back, so the training ran on
    CPU while the archived fingerprint said ``dev=cuda``; on a node that DID expose a GPU it would
    have taken one the job never requested, splitting this arm's search across two arithmetics.
    LATENT today — the chain-in-one-job entry is built and unit-tested but NOT WIRED (nothing
    constructs a chain spec or selects ``entry_module="src.cluster.bayes_chain"``, so ``bayes_opt``
    currently runs as array-of-1 jobs through the driver, which does inject the device). That is
    precisely why it is fixed now: the defect would go live SILENTLY the moment the latency
    optimisation is switched on, and switching it on is what ``lanes.py``'s Lane 1 plan calls for.
    """
    from src.search.multiseed import candidate_seed_ids

    out = []
    for run_id, seed in candidate_seed_ids(cid, k_seeds, base_seed):
        s = family_spec(arm, "coeffs", coeffs, run_id, opts, str(spec_archive_root))
        s["seed"] = seed
        if device:
            s["device"] = str(device)
        if k_seeds > 1:
            s["emit_train_returns"] = True     # the k-seed FED tail is TRAIN-window (B-A2)
        out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m src.cluster.bayes_chain --spec <chain.json>`` — the on-node entry point.

    The chain spec carries ``{opts, arm, archive_root, spec_archive_root, k_seeds,
    deadline_secs}``. Exits 0 on BOTH ``complete`` and ``partial`` (a partial chain is a successful
    bounded run, not a failure — the driver resubmits until it reports complete).
    """
    import argparse

    p = argparse.ArgumentParser(description="Run the whole bayes_opt GP chain in ONE job.")
    p.add_argument("--spec", required=True, help="Path to the chain spec JSON.")
    p.add_argument("--pack", type=int, default=1, help="Ignored (the chain is serial by nature); "
                                                       "accepted so the shared jobscript template "
                                                       "needs only its entry module swapped.")
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = run_bayes_chain(
        cfg["opts"],
        arm=cfg.get("arm", "bayes_opt"),
        archive_root=cfg["archive_root"],
        spec_archive_root=cfg["spec_archive_root"],
        k_seeds=int(cfg.get("k_seeds", 1)),
        device=cfg.get("device"),   # the substrate, threaded from the chain spec (2026-07-27)
        deadline_secs=cfg.get("deadline_secs"),
        est_iter_secs=cfg.get("est_iter_secs"),
    )
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
