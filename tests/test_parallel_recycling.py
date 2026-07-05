"""Unit test for the manual pool-recycling primitive (``src.orchestration.parallel.run_recycling``).

In-pool worker recycling (``ProcessPoolExecutor(max_tasks_per_child=...)``) DEADLOCKS on Windows +
spawn across CPython 3.11-3.14 (measured 2026-06-24, all four versions). The campaign therefore reclaims
each worker's fragmented replay-buffer heap by tearing the ``DevicePool`` down and recreating it every
``recycle_every`` tasks. This test locks in the contract: every spec is processed, order is preserved,
device tokens are assigned, and fresh pools spawn fresh workers (the reclaim mechanism).

Uses ``initializer=None`` so the spawned workers are bare (no torch / no pyarrow) -> the test is fast and
has no heavy-dependency requirement.
"""
from __future__ import annotations

import os
import time

from src.orchestration.parallel import run_recycling


def _ping(spec: dict) -> dict:
    """Bare worker (no torch): echo the pid + index + the assigned device token."""
    return {"pid": os.getpid(), "i": spec["i"], "device": spec.get("device")}


def _sleepy(spec: dict) -> dict:
    """Bare worker that sleeps ``spec['sleep']`` seconds (drives the as-completed/stall tests)."""
    time.sleep(float(spec.get("sleep", 0)))
    return {"ok": True, "i": spec["i"], "cid": spec.get("cid")}


def _raises_on_odd(spec: dict) -> dict:
    """Worker that RAISES on odd indices (vs returning an error dict) — exercises 1A-4."""
    if spec["i"] % 2 == 1:
        raise RuntimeError(f"boom on {spec['i']}")
    return {"ok": True, "i": spec["i"]}


def test_run_recycling_processes_all_in_order_and_recycles_workers() -> None:
    specs = [{"i": i} for i in range(10)]
    res = run_recycling(specs, worker=_ping, n_gpu=0, n_cpu=2, recycle_every=4, initializer=None)

    # every spec processed, in submission order
    assert [r["i"] for r in res] == list(range(10))
    # device tokens assigned from the cpu pool (n_gpu=0)
    assert all(r["device"] == "cpu" for r in res)
    # 10 tasks / recycle_every=4 -> 3 fresh pools -> strictly more than n_cpu (2) distinct worker pids,
    # i.e. the pool was actually torn down + respawned between batches (the RAM-reclaim mechanism).
    assert len(set(r["pid"] for r in res)) > 2


def test_run_recycling_single_batch_when_recycle_every_exceeds_count() -> None:
    """recycle_every >= len(specs) -> one pool, still correct (no spurious teardown)."""
    specs = [{"i": i} for i in range(3)]
    res = run_recycling(specs, worker=_ping, n_gpu=0, n_cpu=2, recycle_every=16, initializer=None)
    assert [r["i"] for r in res] == [0, 1, 2]


def test_run_recycling_captures_raising_worker_without_aborting() -> None:
    """1A-4: a worker that RAISES must not abort the run — every spec is still attempted, in order."""
    specs = [{"i": i} for i in range(6)]
    res = run_recycling(specs, worker=_raises_on_odd, n_gpu=0, n_cpu=2, recycle_every=4, initializer=None)
    assert len(res) == 6  # all 6 attempted despite the 3 odd ones raising
    # order preserved: even indices returned ok with their i; odd indices captured as error dicts
    assert res[0].get("i") == 0 and res[2].get("i") == 2 and res[4].get("i") == 4
    assert all(not res[j].get("ok") and "boom" in res[j]["error"] for j in (1, 3, 5))


def test_run_recycling_empty_specs_returns_empty() -> None:
    """Edge case: no specs -> no pool spawned, returns []."""
    assert run_recycling([], worker=_ping, n_gpu=0, n_cpu=2, recycle_every=4, initializer=None) == []


def test_run_recycling_streams_results_as_completed_not_in_submission_order() -> None:
    """2026-07-06 as-completed collection: ``on_result`` fires at COMPLETION time, so a slow
    early-submitted training cannot delay the archival of later-submitted completed ones (the old
    submission-order collection could orphan up to recycle_every-1 COMPLETED trainings on a driver
    crash) — while the RETURNED list keeps submission order (run_sigma_pilot_train re-attributes
    results by position)."""
    specs = [{"i": 0, "sleep": 1.5}] + [{"i": i, "sleep": 0.05} for i in (1, 2, 3)]
    streamed: list[dict] = []
    res = run_recycling(
        specs, worker=_sleepy, n_gpu=0, n_cpu=2, recycle_every=8,
        initializer=None, on_result=streamed.append,
    )
    assert [r["i"] for r in res] == [0, 1, 2, 3]  # returned order = submission order (the contract)
    assert {r["i"] for r in streamed} == {0, 1, 2, 3}  # every row streamed exactly once
    assert streamed[-1]["i"] == 0  # ... but the SLOW task streamed LAST (as-completed, not as-submitted)


def test_run_recycling_stall_detection_fires_and_run_completes() -> None:
    """2026-07-06 wedge detection: when NO future completes within ``stall_after_s``, ``on_stall``
    fires with the pending spec identities — detection only; the run still completes."""
    specs = [{"i": 0, "cid": "c0", "sleep": 1.2}, {"i": 1, "cid": "c1", "sleep": 0.05}]
    stalls: list[dict] = []
    res = run_recycling(
        specs, worker=_sleepy, n_gpu=0, n_cpu=2, recycle_every=8,
        initializer=None, stall_after_s=0.25, on_stall=stalls.append,
    )
    assert [r["i"] for r in res] == [0, 1]
    assert stalls, "no stall callback fired for a 1.2s-quiet batch with a 0.25s window"
    # once only the slow task remains pending, the payload names exactly it
    assert any(s["pending"] == [{"cid": "c0"}] for s in stalls)
    assert all(s["since_last_completion_s"] >= 0 for s in stalls)


def _sleepy_stamped(spec: dict) -> dict:
    """Sleep then return — the driver stamps arrival times via on_result (S15 latency test)."""
    time.sleep(float(spec.get("sleep", 0)))
    return {"ok": True, "i": spec["i"]}


def test_run_recycling_streams_during_submission_not_only_after_it() -> None:
    """2026-07-06 S15: submission must NOT block collection. With capacity 1 and four 0.4s tasks,
    a sliding-window scheduler streams the first result ~3 task-durations before the last; the old
    blocking whole-batch submission could not start collecting until 3 tasks had already finished,
    compressing the first-to-last streaming gap to ~one task duration. The spawn overhead cancels
    in the difference, so the assertion is timing-robust."""
    specs = [{"i": i, "sleep": 0.4} for i in range(4)]
    stamps: list[float] = []
    run_recycling(
        specs, worker=_sleepy_stamped, n_gpu=0, n_cpu=1, recycle_every=8,
        initializer=None, on_result=lambda _r: stamps.append(time.monotonic()),
    )
    assert len(stamps) == 4
    spread = stamps[-1] - stamps[0]
    assert spread > 0.9, (
        f"first-to-last streaming spread {spread:.2f}s — results were collected in a burst, "
        "i.e. submission blocked collection (the S15 regression)"
    )


def test_parse_affinity_specs() -> None:
    """2026-07-06 P-core placement: the affinity spec parser (ranges, csv, mixed, junk-total)."""
    from src.orchestration.parallel import _parse_affinity

    assert _parse_affinity("0-11") == list(range(12))
    assert _parse_affinity("0,2,4") == [0, 2, 4]
    assert _parse_affinity("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert _parse_affinity("junk,,x-y") == []  # total on garbage — placement is never a blocker
    assert _parse_affinity("3-3") == [3]


def test_apply_cpu_placement_noop_without_env(monkeypatch) -> None:
    """No env var -> a pure no-op (placement must never surprise a non-configured host)."""
    from src.orchestration import parallel

    monkeypatch.delenv("LLMRP_WORKER_CPU_AFFINITY", raising=False)
    parallel._apply_cpu_placement()  # must not raise or change anything


def test_apply_cpu_placement_pins_this_process(monkeypatch) -> None:
    """With the env var set, the process affinity is applied (verified via psutil round-trip)."""
    import psutil

    from src.orchestration import parallel

    p = psutil.Process()
    original = p.cpu_affinity()
    try:
        want = original[: max(1, len(original) - 1)]  # a strict subset that exists on any host
        monkeypatch.setenv("LLMRP_WORKER_CPU_AFFINITY", ",".join(str(c) for c in want))
        parallel._apply_cpu_placement()
        assert sorted(p.cpu_affinity()) == sorted(want)
    finally:
        p.cpu_affinity(original)  # restore — never leak a pinned test runner
