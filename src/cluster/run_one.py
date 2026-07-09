"""On-node entrypoint: run ONE task file (a single spec, or a §15 GPU pack of N specs).

Routes every spec through the EXISTING certified machinery — ``parallel.train_candidate`` for
search/test trainings and ``parallel._archive`` for the atomic record commit — so a cluster run
produces byte-compatible archives with the local pipeline (the resume/replay guarantees carry
over untouched). Pack mode exploits the researched cgroup fact (§15): a ``-l gpu=1`` job owns
its GPU exclusively, so N concurrent trainings inside the job are safe — the exact pattern the
laptop DevicePool certified (3-way == serial, byte-identical).

Exit code: 0 iff EVERY spec in the task succeeded (a pack with any failure exits 1 so SGE/qacct
and the epilogue ledger flag it; the driver's compacted-resume re-emits only the missing
run_ids — completed pack-mates are skipped by run_id idempotency).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["run_task", "main"]

_LOG = logging.getLogger("cluster.run_one")


def _worker_for(spec: dict[str, Any]) -> Any:
    """The certified worker for this spec's LEG: search ``train_candidate`` vs sealed-leg
    ``_test_seed_worker``. The cluster reuses the EXACT laptop workers — a cluster record is
    byte-compatible with the local one, so downstream inference reads them identically."""
    if spec.get("leg") == "test":
        from src.orchestration.test_leg import _test_seed_worker

        return _test_seed_worker
    from src.orchestration.parallel import train_candidate

    return train_candidate


def _write_reject_marker(result: dict[str, Any], spec: dict[str, Any]) -> None:
    """P9 (2026-07-13 pre-spend audit): a FAILED row previously left NOTHING durable on the archive —
    the driver saw only a missing record and burned MAX_RETRIES requeue rounds (queue wait + h_rt
    each) even on a DETERMINISTIC sandbox/contract reject, and every resume re-tried it again.
    Write an atomic marker under ``<archive_root>/_rejects/<run_id>.json`` (a sibling of the arm
    dirs — never named record.json, so completion truth / analysis / integrity ignore it).
    ``permanent`` is True ONLY for the validation-reject class (``failed_validation`` — same-source
    retry is provably futile); anything else (OOM, CUDA hiccup) stays retryable."""
    import os
    import time

    rid = str(result.get("run_id") or result.get("candidate_id")
              or spec.get("run_id") or spec.get("candidate_id") or "")
    if not rid or not spec.get("archive_root"):
        return
    d = Path(spec["archive_root"]) / "_rejects"
    d.mkdir(parents=True, exist_ok=True)
    marker = {
        "run_id": rid, "candidate_id": result.get("candidate_id"), "arm": spec.get("arm"),
        "leg": spec.get("leg", "search"), "error": str(result.get("error"))[:500],
        "permanent": bool(result.get("failed_validation")), "ts": time.time(),
    }
    tmp = d / f".{rid}.tmp"
    tmp.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    os.replace(tmp, d / f"{rid}.json")


def _resolved_device(spec: dict[str, Any]) -> str:
    """The device this training ACTUALLY used, not the one the spec asked for (deep review, 2026-07-27).

    The S6 sealed-leg audit works by comparing env-fingerprint labels: a CPU/GPU mix shows two
    distinct labels and is flagged. That only works if the label records what RAN. It previously
    recorded ``spec.get("device", "cuda")`` — the REQUEST — and the two differ in a case that is
    live on the CPU lane:

      * ``opts`` carries NO ``device`` key at all (its keys are train_steps…window), and
        ``parallel._spec`` does not add one, so every spec the **bayes_opt GP chain** builds on-node
        (``bayes_chain._chain_specs`` → ``_family_spec`` → ``_spec``) reaches ``_run_single``
        WITHOUT a device and is defaulted to ``"cuda"`` there. ``campaign``'s explicit injection
        (``specs = [{**s, "device": device} …]``) fires only for specs IT builds, and only when
        ``device != "cuda"`` — the chain runs as its own job and never passes through it.
      * A CPU-lane job requests no GPU at all (``jobscript``: ``gpu_line`` is empty for
        ``device == "cpu"``), so that ``"cuda"`` request finds no device and torch falls back to CPU.

    The training therefore ran on CPU while the label said ``dev=cuda`` — the audit would have been
    reading a fiction, and on a node where a GPU *was* visible it would have missed a genuine mix.
    Resolving against ``torch.cuda.is_available()`` makes the label true in both cases: an honest
    ``cpu`` when the request fell back, and a real ``cuda`` when it did not — so a mix is flagged
    because it happened, not because a default was guessed.

    Falls back to the requested string if torch cannot be imported (never break archival over a
    label).
    """
    requested = str(spec.get("device", "cuda"))
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch

        return requested if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a label must never sink the archival of a finished training
        return requested


def _archive_result(result: dict[str, Any], spec: dict[str, Any]) -> None:
    """Archive an OK result via the LEG-appropriate path — the SAME atomic ``write_run`` the local
    paths use (search: ``parallel._archive``; test: ``test_leg`` record → ``write_run`` under the
    winner's arm dir). Both workers return their record WITHOUT archiving, so this is the single,
    first archival — no double-write; the poll layer's completion truth reads exactly these dirs.
    FAILED rows leave a durable ``_rejects`` marker instead (P9) so the driver can distinguish a
    deterministic reject from a transient death."""
    if not result.get("ok"):
        _write_reject_marker(result, spec)
        return
    if spec.get("leg") == "test":
        rec = result.get("record")
        if rec is not None:
            from src.io.results import write_run
            from src.orchestration.parallel import _run_env_fp

            arm_root = str(Path(spec["archive_root"]) / str(rec["arm"]))
            # Node-side env fingerprint (the GPU that ACTUALLY trained this seed), overriding the
            # driver-computed spec["env_fp"] the worker carried — parity with the search leg
            # (``_archive`` -> ``_run_env_fp``) and CORRECT for the S6 sealed-leg homogeneity audit,
            # which must attribute each test seed to the node it ran on, not the laptop that authored
            # the spec. Best-effort (falls back to the driver label if capture fails), writes the
            # replayable <run_dir>/env.json exactly like the search records.
            # 2026-07-13 audit fix (2 independent auditors): rec["env_fingerprint"] is already the
            # driver's {label, env_json_sha256} DICT — passing it whole as env_fp nested a dict
            # into the node fingerprint's label, and the C3 gate's label census then crashed on
            # `unhashable type: dict` AFTER the full floor had trained. Unwrap to the string label.
            _fp = rec.get("env_fingerprint")
            _lbl = _fp.get("label", "") if isinstance(_fp, dict) else _fp
            # 2026-07-26 (the CPU lane, REPRODUCIBILITY): capture_env records the NODE's cuda
            # availability, NOT the device this training actually used — so a `device=cpu` spec
            # running on a GPU node captured `cuda_available: true` and the S6 homogeneity audit
            # was BLIND to a CPU/GPU mix. CPU and CUDA are not bit-identical (the reason
            # run_campaign.py refuses `--cpu` on the sealed leg), so an undetected mix would break
            # the CRN pairing that every paired contrast rests on. Stamp the RESOLVED device into
            # the label the C3 label-census compares: a mixed run now shows two distinct labels and
            # is flagged, instead of passing silently.
            _dev = _resolved_device(spec)
            _lbl = f"{_lbl}|dev={_dev}" if _lbl else f"dev={_dev}"
            node_fp = _run_env_fp(
                arm_root, str(rec["run_id"]),
                {"seed": rec.get("seed"), "env_fp": _lbl},
            )
            write_run({**rec, "env_fingerprint": node_fp}, arm_root)
    else:
        from src.orchestration.parallel import _archive

        # Provenance parity: the record must carry the EXACT authored prompt,
        # but authoring is laptop-side so the job's result lacks it — the orchestrator threads it onto
        # the spec, and we copy it onto the result so ``_archive`` persists it exactly as the local
        # ``_drive_llm_arm`` does (which sets ``r["prompt"]`` before archiving).
        if spec.get("prompt") and not result.get("prompt"):
            result = {**result, "prompt": spec["prompt"]}
        # DEVICE STAMP FOR THE SEARCH LEG TOO (2026-07-27). The test branch above stamps the
        # RESOLVED device into the env-fingerprint label; this branch never did. `_archive` passes
        # the spec to `_run_env_fp`, which reads `opts["env_fp"]` — a key search specs do not carry
        # — so EVERY search record archived a BLANK label. Observed live on the first CPU-lane run:
        # `env_fingerprint: {'env_json_sha256': '…', 'label': ''}` for a training that definitely
        # ran on CPU. The asymmetry mattered less while search was GPU-only, but the CPU lane makes
        # a genuine cross-substrate mix possible in the very leg that decides WHICH candidate wins,
        # and an unlabelled record cannot be audited for it at all. Same one-line convention as the
        # test leg so the C3 label census sees one vocabulary across both legs.
        _dev = _resolved_device(spec)
        _lbl = str(spec.get("env_fp") or "")
        spec = {**spec, "env_fp": (f"{_lbl}|dev={_dev}" if _lbl else f"dev={_dev}")}
        _archive(result, spec["arm"], spec, spec["archive_root"], int(spec.get("generation", 0)))


def _already_archived(spec: dict[str, Any]) -> bool:
    """2026-07-13 audit fix (idempotency the module docstring CLAIMED but never implemented):
    ``-r y`` makes SGE re-execute the WHOLE task after a node failure — without this check every
    completed pack-mate would be RETRAINED (up to pack−1 × ~1.5 h wasted) and its remote record
    overwritten with a different node fingerprint (remote ≠ pulled-mirror divergence)."""
    rid = str(spec.get("run_id") or spec.get("candidate_id") or "")
    arm = str(spec.get("arm") or "")
    root = spec.get("archive_root")
    if not (rid and arm and root):
        return False
    return (Path(str(root)) / arm / rid / "record.json").is_file()


def _run_single(spec: dict[str, Any]) -> dict[str, Any]:
    """One training through the certified LEG worker; archives on success; returns the result row."""
    spec = dict(spec)
    # NOTE: 2026-07-27 (deep review): this default is REACHABLE and it is SILENT. `opts` carries no
    # `device` key (its keys are train_steps…window) and `parallel._spec` adds none, so every spec
    # the bayes_opt GP chain builds on-node arrives here WITHOUT one and is defaulted to "cuda" —
    # campaign's explicit injection fires only for specs IT builds, and only when device != "cuda".
    # The default is LEFT AS IS (changing it would alter the GPU path, and resolving it to "cuda if
    # available" would let a CPU-lane job seize a GPU it never requested — the one impairment
    # `lanes.py` refuses by construction). Instead it is made VISIBLE: the archived fingerprint now
    # records the RESOLVED device (`_resolved_device`), so a fallback is labelled `cpu` honestly and
    # a genuine mix is FLAGGED by the S6 audit rather than passing silently, and this logs the
    # default so it is greppable in the job log.
    if "device" not in spec:
        _LOG.info(
            "spec %r carries no 'device'; defaulting to 'cuda' (the archived fingerprint records the "
            "RESOLVED device, so a CPU fallback is labelled honestly)",
            spec.get("run_id") or spec.get("candidate_id"),
        )
    spec.setdefault("device", "cuda")  # the job's cgroup-exclusive GPU
    if _already_archived(spec):
        return {"ok": True, "candidate_id": spec.get("candidate_id"),
                "run_id": spec.get("run_id"), "skipped": "already_archived"}
    result = _worker_for(spec)(spec)
    try:
        _archive_result(result, spec)
    except Exception as exc:  # noqa: BLE001 — 2026-07-13 audit: an archival hiccup must not
        # abort the surviving pack-mates' collection loop (their finished GPU-hours would be
        # dropped unarchived). Stamp the error; the record is absent so the driver requeues.
        result = {**result, "ok": False, "archive_error": f"{type(exc).__name__}: {exc}"}
    return result


def _task_threads(specs: list[dict[str, Any]]) -> int:
    """The intra-op thread count for this task, read from the specs (R107 wiring, 2026-07-27).

    Absent key = 1, which is the TEST leg and every pre-R107 caller — so the default path is
    byte-unchanged. A task's specs all come from one batch and therefore one dispatch decision, so
    a MIXED thread count means a spec-construction bug upstream. Fail LOUD rather than silently
    pinning one of them: threads are part of the determinism envelope, and a job that trained half
    its specs under a different float reduction order would archive records the S6 homogeneity
    audit could not distinguish from clean ones (the same blindness the ``device`` stamp fixed).
    """
    seen = {int(s.get("threads", 1) or 1) for s in specs}
    if len(seen) > 1:
        raise ValueError(
            f"MIXED thread counts in one task: {sorted(seen)}. Threads are part of the determinism "
            "envelope (multi-threaded BLAS reorders float reductions), so one job must train every "
            "spec under ONE regime. Split these into separate batches."
        )
    return max(1, seen.pop() if seen else 1)


def _task_device(specs: list[dict[str, Any]]) -> str:
    """The SUBSTRATE for this task, read from the specs (deep review, 2026-07-27).

    Mirrors :func:`_task_threads`, and exists for the same reason: ``DevicePool`` hands each spec a
    device TOKEN and ``submit_with`` does ``{**spec, "device": token}`` — it OVERWRITES whatever the
    spec said. The pool was constructed as ``n_gpu=pack, n_cpu=0``, and every ``n_gpu`` token is the
    literal string ``"cuda"``, so on the CPU lane a packed job silently replaced ``device="cpu"``
    with ``"cuda"`` on every one of its specs. The campaign's device injection, this morning's
    ``bayes_chain`` stamp and the ladder's stamp were all defeated at the last step by the pool.

    Only the pack>1 path is affected (pack=1 runs inline and never builds a pool), which is exactly
    why it survived: the CPU lane has only ever been exercised at pack=1. It matters now because
    PACKING is the main lever for holding a high core count — one placement doing ``pack``
    trainings — so the campaign's throughput plan depends on this path being correct.

    Homogeneous by construction (one array = one dispatch decision); FAIL LOUD on a mix rather than
    let half a job train on a different substrate.
    """
    seen = {str(s.get("device") or "cuda") for s in specs}
    if len(seen) > 1:
        raise ValueError(
            f"MIXED devices in one task: {sorted(seen)}. The device is part of the determinism "
            "envelope (CPU is not bit-identical to CUDA) and every CRN comparison unit must be "
            "device-HOMOGENEOUS. Split these into separate batches."
        )
    return seen.pop() if seen else "cuda"


def run_task(payload: Any, pack: int = 1) -> list[dict[str, Any]]:
    """Execute a task payload (dict = one spec; list = a pack). Returns all result rows.

    ``pack`` > 1 with a list payload runs the specs CONCURRENTLY on this job's exclusive GPU via
    the certified DevicePool (n_gpu=pack tokens on the same physical card). A dict payload or
    pack=1 runs inline (no spawn overhead).
    """
    specs = payload if isinstance(payload, list) else [payload]
    if len(specs) <= 1 or pack <= 1:
        # P18 (2026-07-13 audit): the pack path gets _worker_init via the DevicePool spawn
        # initializer, but the inline path skipped it — no pyarrow-before-torch preload (the
        # SIGSEGV class verified 2026-06-20) and no thread pinning. Run the SAME init so the
        # two node paths share one environment contract (BLAS threads=1, alloc conf, preload).
        from src.orchestration.parallel import _worker_init

        _worker_init(_task_threads(specs))
        # 2026-07-26 review: the inline path ALSO needs the `-r y` idempotency skip the pack branch
        # below applies. `#$ -r y` (jobscript.py:38) makes SGE re-execute the WHOLE task after a node
        # failure, and `pack=1` — which routes HERE — is the DEFAULT (jobscript.py:94,
        # campaign.py:82/154; the renderer even sizes h_rt specifically for it). Without this check a
        # re-executed task RETRAINS an already-completed spec (~1.5 h burnt against the Aug-27
        # exogenous stop) and OVERWRITES its remote record with a different node fingerprint — the
        # remote-vs-pulled-mirror divergence `_already_archived` was added to prevent, and which
        # `cluster/integrity.py`'s env-fingerprint census would then flag on the SEALED leg.
        # (Same class as the P18 note above: that audit fixed the inline path's MISSING `_worker_init`
        # and did not notice the inline path also skipped this guard.)
        out: list[dict[str, Any]] = []
        for s in specs:
            if _already_archived(s):
                out.append({"ok": True, "candidate_id": s.get("candidate_id"),
                            "run_id": s.get("run_id"), "skipped": "already_archived"})
            else:
                out.append(_run_single(s))
        return out

    from concurrent.futures import as_completed

    from src.orchestration.parallel import DevicePool

    if len(specs) > pack:
        # More specs than concurrent slots means MULTIPLE WAVES inside one job — legal (the pool
        # queues), but the jobscript's h_rt must have been sized for it (render_jobscript sizes
        # for ONE wave by default). Surface it loudly rather than time out silently at 1h30.
        print(
            f"[run_one] WARNING: {len(specs)} specs > pack={pack} -> "
            f"{-(-len(specs) // pack)} waves; ensure h_rt was sized accordingly",
            flush=True,
        )
    rows: list[dict[str, Any]] = []
    # 2026-07-13 audit (-r y idempotency): SGE re-executes the WHOLE task after a node failure —
    # skip pack-mates whose record already exists instead of retraining them (and overwriting the
    # remote record with a divergent node fingerprint).
    to_run: list[dict[str, Any]] = []
    for s in specs:
        if _already_archived(s):
            rows.append({"ok": True, "candidate_id": s.get("candidate_id"),
                         "run_id": s.get("run_id"), "skipped": "already_archived"})
        else:
            to_run.append(s)
    if not to_run:
        return rows
    # submit_with routes each spec to its LEG worker (the pool injects the device token); a pack is
    # homogeneous in practice (one array = one leg), but per-spec routing keeps mixed packs correct.
    # Size the pool's tokens to the SPECS' substrate, not to a hardcoded n_gpu (2026-07-27). Every
    # `n_gpu` token is the literal string "cuda" and `submit_with` stamps it OVER the spec's own
    # device, so `n_gpu=pack, n_cpu=0` silently turned every packed CPU-lane training into a CUDA
    # one. See `_task_device`.
    _dev = _task_device(to_run)
    _slots = min(pack, len(to_run))
    # PROVENANCE PARITY FOR THE PACKED PATH (2026-07-27, found by the pack-4 rehearsal).
    # At pack>=2 the WORKERS train but the PARENT archives, so `_archive` -> `_run_env_fp` ->
    # `capture_env` samples the PARENT's environment. The parent never runs `_worker_init`, and SGE
    # sets `OMP_NUM_THREADS` from the `-pe smp N` slot count — so a run launched at 1 thread archived
    # `OMP_NUM_THREADS: 4, torch num_threads: 4`. The trainings were CORRECT (heartbeat ~12.2 steps/s
    # each, consistent with 1 thread); the RECORD was not. That field is exactly what the S6
    # homogeneity audit reads to rule out a heterogeneous determinism envelope, so a knowingly-false
    # value must not ship. Proven by contrast: the pack-1 rehearsal recorded OMP=1 correctly, because
    # inline archiving happens in the same process `_worker_init` initialised.
    #
    # Deliberately NOT `_worker_init(...)` here: that also calls `_apply_cpu_placement()`, and a
    # spawned child INHERITS CPU affinity — pinning the parent could silently constrain every packed
    # worker to a subset of cores. Only the fields `capture_env` reads are synced, with no affinity,
    # no preload and no allocator change. The value is well-defined because `_task_threads` FAILS
    # LOUD on a mixed task, so every spec in this job shares one thread count.
    _thr = _task_threads(to_run)
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(_thr)
    try:
        import torch as _torch

        _torch.set_num_threads(_thr)
    except Exception:  # noqa: BLE001 - provenance parity must never sink a finished training
        pass
    with DevicePool(n_gpu=(0 if _dev == "cpu" else _slots),
                    n_cpu=(_slots if _dev == "cpu" else 0),
                    init_threads=_task_threads(to_run)) as pool:
        futs = {pool.submit_with(_worker_for(s), dict(s)): s for s in to_run}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                row = fut.result()  # the worker captures its own failures -> a result dict
            except Exception as exc:  # noqa: BLE001 — V5 audit fix: a pool-level crash (worker
                # killed by the OS, unpicklable result) must not lose the SIBLING pack-mates'
                # results or leave this spec unattributed for the compacted resume.
                row = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate_id": s.get("candidate_id"),
                    "run_id": s.get("run_id"),
                }
            try:
                _archive_result(row, s)
            except Exception as exc:  # noqa: BLE001 — 2026-07-13 audit: an archival hiccup on ONE
                # row must not abort the as_completed loop and drop the surviving pack-mates'
                # finished GPU-hours unarchived. Stamp it; absent record -> the driver requeues.
                row = {**row, "ok": False, "archive_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one Myriad task file (spec or pack).")
    p.add_argument("--spec", required=True, help="Path to task_<i>.json (sha-verified).")
    p.add_argument("--pack", type=int, default=1, help="Concurrent trainings on this GPU (§15).")
    args = p.parse_args(argv)

    from src.cluster.spec_io import read_spec

    payload = read_spec(args.spec)
    rows = run_task(payload, pack=args.pack)
    n_ok = sum(1 for r in rows if r.get("ok"))
    # 2026-07-12 (live-rehearsal fix): failed rows carried their error string but the log printed
    # only counts — a node-side failure was UNDIAGNOSABLE from the .o file (the scalar_g0 sandbox
    # reject surfaced as a bare rc=1). One line per failure, greppable, before the summary.
    for r in rows:
        if not r.get("ok"):
            print(json.dumps({"failed": r.get("candidate_id") or r.get("run_id"),
                              "error": str(r.get("error"))[:500]}), flush=True)
    print(json.dumps({"task": args.spec, "n": len(rows), "ok": n_ok}))
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
