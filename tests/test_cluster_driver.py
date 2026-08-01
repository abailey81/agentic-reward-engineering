"""Driver-kernel tests: the FULL submit → poll → requeue → done loop against a scripted fake
cluster (zero network, zero GPU). Every path is exercised: resume no-op, happy path, compacted
requeue, retry exhaustion → permanent ledger, the double-submit/adoption guard, transient pull
and qsub failures (bounded), the max-wall guard, and the stale-pending re-filter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cluster.driver import batch_jobs_in_queue, run_batch, submit_batch


class FakeCluster:
    """A scripted Myriad: a queue of jobnames, qsub registration, and a pull whose per-call
    actions are scripted as dicts ``{"complete": [run_ids], "drain": bool, "raise": bool}``."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.archive = tmp / "archive"
        self.batches = tmp / "batches"
        self.queue: set[str] = set()
        self.qsubs: list[str] = []
        self.pushes: list[Path] = []
        self.pull_script: list[dict] = []
        self.qsub_fail_next = 0
        self.sleeps: list[float] = []
        self._clock = 0.0
        # P13: scripted qacct output. The default (rows but NO taskid) is unattributable
        # evidence -> the drain degrades to the legacy bump-all, keeping older tests intact.
        self.qacct_text = "==============\njobnumber 1\nexit_status 1\nfailed 0\n"

    # --- injectable seams -------------------------------------------------
    def runner(self, cmd: list[str]) -> str:
        if cmd[:2] == ["qstat", "-r"]:
            return "\n".join(f"Full jobname: {n}" for n in sorted(self.queue))
        if cmd[0] == "qstat":
            return ""
        if cmd[0] == "mkdir":
            return ""
        if cmd[0] == "qacct":
            return self.qacct_text
        if cmd[0] == "qsub":
            if self.qsub_fail_next > 0:
                self.qsub_fail_next -= 1
                raise RuntimeError("transient qsub failure")
            name = Path(cmd[1]).stem
            self.queue.add(name)
            self.qsubs.append(name)
            return f"Your job {100 + len(self.qsubs)} ok"
        raise AssertionError(f"unexpected cmd {cmd}")

    def push(self, batch_dir, dest) -> None:
        self.pushes.append(Path(batch_dir))

    def pull(self) -> int:
        act = self.pull_script.pop(0) if self.pull_script else {}
        if act.get("raise"):
            raise ConnectionError("vpn blip")
        if act.get("raise_bug"):  # P14: a LOCAL bug, not transport
            raise TypeError("local bug: not a transport blip")
        for rid in act.get("complete", []):
            d = self.archive / "search" / rid
            d.mkdir(parents=True, exist_ok=True)
            (d / "record.json").write_text("{}")
        for rid, perm in act.get("reject", []):  # P9 node-side reject markers ride the pull
            d = self.archive / "search" / "_rejects"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{rid}.json").write_text(json.dumps(
                {"run_id": rid, "permanent": perm, "error": "sandbox: bad name"}))
        if act.get("drain"):
            self.queue.clear()
        return len(act.get("complete", []))

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self._clock += secs

    def clock(self) -> float:
        return self._clock


def _specs(n: int) -> list[dict]:
    return [{"candidate_id": f"c{i}", "arm": "scalar", "seed": i} for i in range(n)]


def _run(fc: FakeCluster, specs: list[dict], **kw):
    return run_batch(
        specs, "b1",
        local_batch_root=fc.batches, local_archive_root=fc.archive,
        remote_root="/r", remote_outputs_root="/r/outputs", gold_dir="/inputs",
        runner=fc.runner, push=fc.push, pull=fc.pull,
        poll_secs=1.0, sleep=fc.sleep, clock=fc.clock, **kw,
    )


def test_all_already_complete_is_a_pure_resume_noop(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.pull_script = [{"complete": ["c0", "c1", "c2"]}]
    out = _run(fc, specs)
    assert out["ok"] and out["completed"] == 3 and out["rounds"] == 0
    assert out["job_ids"] == [] and fc.qsubs == [] and fc.pushes == []


def test_happy_path_submits_once_and_finishes(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.pull_script = [{}, {"complete": ["c0", "c1", "c2"]}]
    out = _run(fc, specs)
    assert out["ok"] and out["rounds"] == 1 and fc.qsubs == ["b1"]
    batch = fc.batches / "b1"
    assert (batch / "task_3.json").is_file() and (batch / "index.json").is_file()
    raw = (batch / "b1.sh").read_bytes()
    assert raw.startswith(b"#!/bin/bash -l\n") and b"\r" not in raw  # V11 via write_jobscript
    assert fc.pushes == [batch]


def test_requeue_is_compacted_to_exactly_the_missing_specs(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.pull_script = [{}, {"complete": ["c0", "c1"], "drain": True}, {"complete": ["c2"]}]
    out = _run(fc, specs)
    assert out["ok"] and out["rounds"] == 2 and out["exhausted"] == []
    assert fc.qsubs == ["b1", "b1_r1"]
    index = json.loads((fc.batches / "b1_r1" / "index.json").read_text())
    assert len(index) == 1  # ONLY c2 was re-emitted (compaction)
    assert json.loads((fc.batches / "b1_r1" / "task_1.json").read_text())["candidate_id"] == "c2"
    assert (fc.batches / "b1.qacct.txt").is_file()  # forensics harvested on the drain


def test_retries_exhaust_into_the_permanent_ledger(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(1)
    fc.pull_script = [{}, {"drain": True}, {"drain": True}, {"drain": True}]
    out = _run(fc, specs)
    assert not out["ok"] and out["exhausted"] == ["c0"] and out["completed"] == 0
    assert out["rounds"] == 3 and fc.qsubs == ["b1", "b1_r1", "b1_r2"]
    rows = [json.loads(x) for x in (fc.batches / "b1.permanent.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reason"] == "retries_exhausted"
    assert rows[0]["spec"]["candidate_id"] == "c0"


def test_permanently_ledgered_specs_are_skipped_on_restart(tmp_path):
    """Resume-hardening (long run): a spec already permanently ledgered on a PRIOR run must not be
    re-submitted/re-tried on restart — else a deterministically-failing seed loops every restart."""
    fc = FakeCluster(tmp_path)
    specs = _specs(2)  # c0, c1
    fc.batches.mkdir(parents=True, exist_ok=True)
    (fc.batches / "b1.permanent.jsonl").write_text(
        json.dumps({"task": 0, "spec": {"candidate_id": "c0"}, "reason": "retries_exhausted"}) + "\n")
    fc.pull_script = [{}, {"complete": ["c1"]}]
    out = _run(fc, specs)
    assert out["exhausted"] == ["c0"] and out["ok"] is False and out["completed"] == 1
    # only c1 was ever submitted — c0 was skipped straight from the permanent ledger
    index = json.loads((fc.batches / "b1" / "index.json").read_text())
    assert len(index) == 1
    assert json.loads((fc.batches / "b1" / "task_1.json").read_text())["candidate_id"] == "c1"


def test_adoption_guard_never_double_submits(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(2)
    fc.queue = {"b1"}  # a previous driver invocation's job is still queued
    fc.pull_script = [{}, {"complete": ["c0", "c1"], "drain": True}]
    out = _run(fc, specs)
    assert out["ok"] and out["rounds"] == 0 and fc.qsubs == []  # adopted, never re-submitted


def test_transient_pull_failures_are_tolerated_then_bounded(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(2)
    fc.pull_script = [{"raise": True}, {"raise": True}, {"complete": ["c0", "c1"]}]
    out = _run(fc, specs)
    # 2026-07-13 audit fix: the driver must NEVER act on a stale mirror — the old behavior
    # ("submitted on cycle 1 despite the failed pull") could requeue already-completed work
    # (double-training + retry-budget burn + remote-record overwrite). Failed-pull cycles now
    # beat-and-wait; the first SUCCESSFUL pull (cycle 3) reveals the work complete -> no submit.
    assert out["ok"] and out["rounds"] == 0 and fc.qsubs == []

    fc2 = FakeCluster(tmp_path / "b")
    fc2.pull_script = [{"raise": True}] * 10
    with pytest.raises(RuntimeError, match="consecutive pull failures"):
        _run(fc2, _specs(1), max_consecutive_errors=3)


def test_pull_outage_is_fatal_on_the_wall_time_bound_not_just_the_count(tmp_path):
    """Long-outage tolerance (resume from ANY stoppage): with the count cap set high, a persistent
    transport outage is fatal on the WALL-TIME bound (decoupled from poll cadence) — the driver rides
    out a multi-hour VPN blip instead of dying every N cycles, but a genuinely dead link is surfaced."""
    fc = FakeCluster(tmp_path)
    fc.pull_script = [{"raise": True}] * 50
    # count cap huge; the 5 s outage bound is what trips (clock advances via the poll sleeps)
    with pytest.raises(RuntimeError, match="VPN/ssh down too long"):
        _run(fc, _specs(1), max_consecutive_errors=1000, max_transport_outage_secs=5.0)


def test_heartbeat_is_emitted_each_cycle_and_marks_done(tmp_path):
    """The driver beats a read-only status snapshot every cycle (the sentinel's lease + queue panel),
    and a FINAL phase='done' beat on completion so a finished batch never reads as a hung one."""
    fc = FakeCluster(tmp_path)
    beats: list[dict] = []
    fc.pull_script = [{}, {"complete": ["c0", "c1"]}]
    out = _run(fc, _specs(2), heartbeat=beats.append)
    assert out["ok"]
    assert beats and beats[-1]["phase"] == "done" and beats[-1]["pending"] == 0
    assert any(b["phase"] == "running" for b in beats)
    running = next(b for b in beats if b["phase"] == "running")
    assert set(running) >= {"base_name", "done", "pending", "queue_names", "pull_failures"}


def test_max_wall_guard_raises_loud(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(1)  # never completes; job stays alive
    with pytest.raises(RuntimeError, match="max_wall_secs"):
        _run(fc, specs, max_wall_secs=2.5)


def test_failed_qsub_retries_without_a_retry_bump(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.qsub_fail_next = 1  # the FIRST submission attempt blips
    fc.pull_script = [{}, {}, {"complete": ["c0", "c1", "c2"]}]
    out = _run(fc, specs)
    assert out["ok"] and out["rounds"] == 1 and fc.qsubs == ["b1"]
    assert not (fc.batches / "b1.permanent.jsonl").exists()  # nothing was retry-accounted
    assert "_cluster_retries" not in specs[0]  # caller's dicts are never mutated


def test_stale_pending_submit_is_refiltered_not_retrained(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.pull_script = [
        {},                                            # cycle 1: submit b1
        {"complete": ["c0", "c1"], "drain": True},     # cycle 2: drain -> requeue c2 owed
        {"complete": ["c2"]},                          # cycle 3: c2 completed elsewhere
    ]
    fc_qsub_after_first = fc.qsub_fail_next  # noqa: F841 — clarity: failure armed below

    # arm the blip AFTER the first successful submit: fail the requeue round's qsub once
    orig_runner = fc.runner

    def runner(cmd):
        if cmd[0] == "qsub" and fc.qsubs == ["b1"] and not fc.queue:
            fc.qsubs_blipped = True
            raise RuntimeError("transient qsub failure")
        return orig_runner(cmd)

    out = run_batch(
        specs, "b1",
        local_batch_root=fc.batches, local_archive_root=fc.archive,
        remote_root="/r", remote_outputs_root="/r/outputs", gold_dir="/inputs",
        runner=runner, push=fc.push, pull=fc.pull,
        poll_secs=1.0, sleep=fc.sleep, clock=fc.clock,
    )
    assert out["ok"] and out["exhausted"] == []
    assert fc.qsubs == ["b1"]  # the owed requeue round evaporated once c2 arrived


def test_submit_batch_packs_and_batch_jobs_in_queue(tmp_path):
    fc = FakeCluster(tmp_path)
    submitted = submit_batch(
        _specs(5), "s1_search",
        local_batch_root=fc.batches, remote_root="/r", gold_dir="/inputs",
        runner=fc.runner, push=fc.push, pack=2,
    )
    assert submitted == [("101", "s1_search")]  # unchunked round = ONE array, round-named
    index = json.loads((fc.batches / "s1_search" / "index.json").read_text())
    assert len(index) == 3  # 5 specs at pack=2 -> tasks of 2+2+1
    js = (fc.batches / "s1_search" / "s1_search.sh").read_text()
    assert "--pack 2" in js and "-t 1-3" in js
    assert batch_jobs_in_queue("s1_search", fc.runner)[0] == {"s1_search"}
    assert batch_jobs_in_queue("other", fc.runner) == (set(), {})


def test_batch_jobs_in_queue_captures_states_for_eqw_detection():
    """P1 (2026-07-13 audit): an Eqw array never dispatches, so without state capture it waits
    forever with green heartbeats. The parser pairs each queue row's state with the following
    'Full jobname:' detail line (real qstat -r block shape)."""
    text = (
        " 771972 2.29514 p6ladder   ucestes      Eqw   07/11/2026 15:16:29    2 1\n"
        "       Full jobname:     s1_search\n"
        " 771973 2.10000 other_job  ucestes      r     07/11/2026 15:16:29    2 1\n"
        "       Full jobname:     s1_search_r1\n"
    )
    names, states = batch_jobs_in_queue("s1_search", lambda cmd: text)
    assert names == {"s1_search", "s1_search_r1"}
    assert states == {"s1_search": "Eqw", "s1_search_r1": "r"}


# --------------------------------------------------------------------------- #
# P9 (2026-07-13 pre-spend audit): permanent node-side rejects are abandoned
# IMMEDIATELY (no requeue rounds); transient markers keep the bounded retry.
# --------------------------------------------------------------------------- #
def test_permanent_node_reject_abandons_without_requeue(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(2)
    fc.pull_script = [{}, {"complete": ["c0"], "reject": [("c1", True)], "drain": True}]
    out = _run(fc, specs)
    assert not out["ok"] and out["completed"] == 1 and out["exhausted"] == ["c1"]
    assert fc.qsubs == ["b1"]  # NO requeue round was burned on the deterministic reject
    rows = [json.loads(x) for x in (fc.batches / "b1.permanent.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reason"] == "permanent_node_reject"
    assert rows[0]["spec"]["candidate_id"] == "c1"


def test_transient_node_reject_still_gets_the_bounded_requeue(tmp_path):
    fc = FakeCluster(tmp_path)
    specs = _specs(1)
    fc.pull_script = [{}, {"reject": [("c0", False)], "drain": True}, {"complete": ["c0"]}]
    out = _run(fc, specs)
    assert out["ok"] and out["completed"] == 1 and out["exhausted"] == []
    assert fc.qsubs == ["b1", "b1_r1"]  # transient -> retried normally, then succeeded


def test_local_bug_in_pull_propagates_immediately_not_as_transport(tmp_path):
    """P14: only whitelisted transport exceptions ride the outage budget — a local bug
    (TypeError here) must crash LOUD on the first cycle, not burn 12 h mislabeled as
    'VPN/ssh down'."""
    fc = FakeCluster(tmp_path)
    fc.pull_script = [{"raise_bug": True}]
    with pytest.raises(TypeError, match="local bug"):
        _run(fc, _specs(1))
    assert fc.sleeps == []  # no tolerate-and-retry cycle happened


def test_purged_array_requeues_without_a_retry_bump(tmp_path):
    """P13: a drain with NO qacct trace (deleted-pending: the array was purged before dispatch)
    must not bump retry counts — under the old bump-all drain, 2 purge events permanently
    abandoned specs that never ran once. Three trace-less purges here; the spec survives all of
    them and completes on the 4th round."""
    fc = FakeCluster(tmp_path)
    fc.qacct_text = ""  # deleted-pending leaves no accounting rows
    fc.pull_script = [{}, {"drain": True}, {"drain": True}, {"drain": True}, {"complete": ["c0"]}]
    out = _run(fc, _specs(1))
    assert out["ok"] and out["completed"] == 1 and out["exhausted"] == []
    assert fc.qsubs == ["b1", "b1_r1", "b1_r2", "b1_r3"]  # requeued, never abandoned
    assert not (fc.batches / "b1.permanent.jsonl").exists()


def test_per_task_qacct_evidence_bumps_only_the_dispatched_specs(tmp_path):
    """P13: with per-task qacct rows, ONLY the dispatched spec's retry count bumps — the
    never-attempted pack-mate rides every requeue unbumped and completes normally."""
    fc = FakeCluster(tmp_path)
    # task 1 (c0) dispatched + failed each round; task 2 (c1) never ran
    fc.qacct_text = "==============\ntaskid 1\nexit_status 1\nfailed 100\n"
    fc.pull_script = [{}, {"drain": True}, {"drain": True}, {"drain": True},
                      {"complete": ["c1"]}]
    out = _run(fc, _specs(2))
    assert not out["ok"] and out["completed"] == 1 and out["exhausted"] == ["c0"]
    rows = [json.loads(x) for x in (fc.batches / "b1.permanent.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["spec"]["candidate_id"] == "c0"
    assert rows[0]["reason"] == "retries_exhausted"


def test_driver_lock_refuses_a_live_second_driver_and_breaks_stale(tmp_path):
    """P12: a second driver of the SAME batch is refused while the owner pid is alive; a lock
    left by a DEAD pid (crash) is broken automatically; the lock is released on completion."""
    import subprocess
    import sys

    fc = FakeCluster(tmp_path)
    lock = fc.batches / "b1.driver.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # a live FOREIGN pid owns the lock (a sleeper subprocess stands in for a concurrent driver)
    #
    # NOTE: 2026-08-01 (RUN 10, D20): the stand-in must now actually LOOK like a driver. The lock no
    # longer asks "does that pid exist" — that check cannot survive pid reuse and stranded two live
    # lines (`h3` on 07-31, `leg4` for ~14 h on 08-01) — it asks "is that pid the owner that wrote
    # the lock". For a LEGACY lock carrying no `create_time`, as written here, the fallback is
    # identity-by-command-line, and a bare `time.sleep` process is provably NOT a driver, so the
    # lock is correctly BROKEN and no RuntimeError is raised.
    #
    # The assertion below is NOT weakened: it still demands that a live owner be refused. Only the
    # impersonation is made faithful, by putting the marker the predicate reads into the stand-in's
    # own argv. The complementary case — a legacy lock whose pid now belongs to a NON-driver must
    # be broken — is asserted in tests/test_driver_lock_identity_d20.py.
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)  # run_campaign_cluster"])
    try:
        lock.write_text(json.dumps({"pid": sleeper.pid, "ts": 0.0}))
        fc.pull_script = [{"complete": ["c0"]}]
        with pytest.raises(RuntimeError, match="another driver"):
            _run(fc, _specs(1))
    finally:
        sleeper.kill()
        sleeper.wait()
    # dead owner -> stale lock broken automatically, run proceeds, lock released afterwards
    lock.write_text(json.dumps({"pid": sleeper.pid, "ts": 0.0}))
    fc.pull_script = [{"complete": ["c0"]}]
    out = _run(fc, _specs(1))
    assert out["ok"] and not lock.exists()


def test_run_one_failed_rows_leave_durable_reject_markers(tmp_path):
    from src.cluster.poll import permanent_reject_ids
    from src.cluster.run_one import _archive_result

    base = {"archive_root": str(tmp_path), "arm": "scalar", "leg": "search"}
    _archive_result({"ok": False, "error": "sandbox: bad name", "failed_validation": True,
                     "candidate_id": "c7"}, {**base, "candidate_id": "c7"})
    _archive_result({"ok": False, "error": "MemoryError: oom", "candidate_id": "c8"},
                    {**base, "candidate_id": "c8"})
    # both durable; only the validation reject is PERMANENT
    assert (tmp_path / "_rejects" / "c7.json").is_file()
    assert (tmp_path / "_rejects" / "c8.json").is_file()
    assert permanent_reject_ids(tmp_path) == {"c7"}
    # torn marker never sinks the reader
    (tmp_path / "_rejects" / "torn.json").write_text("{not json")
    assert permanent_reject_ids(tmp_path) == {"c7"}



def test_chunked_submission_defeats_the_serialization_policy(tmp_path):
    """Max-throughput lever (2026-07-13): chunk_tasks=1 splits every round into single-task
    arrays — no hqw tail for the snx=1 policy to hold or purge; requeue rounds chunk too when
    they have >1 task; the adoption matcher recognises part names but still rejects foreign
    batches."""
    fc = FakeCluster(tmp_path)
    specs = _specs(3)
    fc.pull_script = [{}, {"complete": ["c0"], "drain": True}, {"complete": ["c1", "c2"]}]
    out = _run(fc, specs, chunk_tasks=1)
    assert out["ok"] and out["completed"] == 3
    # round 0: three single-task arrays; requeue round (c1+c2 missing): two more parts
    assert fc.qsubs[:3] == ["b1_p01", "b1_p02", "b1_p03"]
    assert fc.qsubs[3:] == ["b1_r1_p01", "b1_r1_p02"]
    # each part carries exactly ONE task file
    assert (fc.batches / "b1_p02" / "task_1.json").is_file()
    assert not (fc.batches / "b1_p02" / "task_2.json").exists()

    # adoption matcher: parts + requeue parts adopted; foreign lookalikes rejected
    lines = [
        " 1 2.0 x u r 07/13/2026 10:00:00 q 2 1",
        "       Full jobname:     b1_p02",
        " 2 2.0 x u r 07/13/2026 10:00:00 q 2 1",
        "       Full jobname:     b1_r1_p01",
        " 3 2.0 x u r 07/13/2026 10:00:00 q 2 1",
        "       Full jobname:     b1_rehearsal",
        " 4 2.0 x u r 07/13/2026 10:00:00 q 2 1",
        "       Full jobname:     b1_p02x",
    ]
    text = chr(10).join(lines)
    names, _states = batch_jobs_in_queue("b1", lambda cmd: text)
    assert names == {"b1_p02", "b1_r1_p01"}
