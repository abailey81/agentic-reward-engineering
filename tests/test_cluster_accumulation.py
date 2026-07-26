"""Slot parsing + the GO-day ACCUMULATION-CURVE report (2026-07-26).

The campaign plan's one un-measured projection is that ~8.5 h campaign tasks ACCUMULATE to
~2,000-3,000 cores, rather than churning at the ~636 measured with 20-min probe jobs (the ~75-job
plateau is a flow equilibrium, `concurrent = dispatch_rate x duration`). The archived telemetry
could not check that, because it recorded job COUNT but not CORES. These lock the fix.
"""
from __future__ import annotations

import json
import time

from src.cluster.telemetry import accumulation_report, parse_our_jobs, running_slots

# Real `qstat` shapes. NOTE the column offset: a RUNNING row carries a queue name that a PENDING
# row does not, so slots sit at index 8 running / 7 pending. Reading a fixed index silently returns
# the date or time field for half the rows.
_QSTAT = """job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
-----------------------------------------------------------------------------------------------
  16816 2.01453 wa1        ucestes      r     07/26/2026 03:41:01 Bran@node-d00a-144.myriad.ucl.     8
  16817 2.01453 wa2        ucestes      r     07/26/2026 03:41:01 Bran@node-b00a-011.myriad.ucl.    16
  16818 2.01453 wa3        ucestes      qw    07/26/2026 03:41:01                                    8
  16819 0.00000 mr_leg2    ucestes      hqw   07/23/2026 17:48:05                                    2 10-15:1
"""


def test_slots_are_parsed_for_BOTH_running_and_pending_layouts():
    jobs = parse_our_jobs(_QSTAT)
    assert len(jobs) == 4
    by_id = {j["id"]: j for j in jobs}
    assert by_id["16816"]["slots"] == 8 and by_id["16816"]["state"] == "r"
    assert by_id["16817"]["slots"] == 16                      # running, index 8
    assert by_id["16818"]["slots"] == 8 and by_id["16818"]["state"] == "qw"   # pending, index 7
    assert by_id["16819"]["slots"] == 2                       # pending array row


def test_running_slots_counts_only_running():
    jobs = parse_our_jobs(_QSTAT)
    n_jobs, cores = running_slots(jobs)
    assert n_jobs == 2 and cores == 24        # 8 + 16; the qw/hqw rows excluded


def test_parse_survives_malformed_rows():
    """Telemetry must degrade, never raise — a truncated transport must not crash the advisor."""
    jobs = parse_our_jobs("h1\nh2\n123 1.0 n u r\nnotajob\n\n")
    assert jobs and jobs[0]["slots"] == 0     # missing column -> 0, not an exception


# --- the accumulation report ----------------------------------------------------------------

def _write_log(tmp_path, series):
    p = tmp_path / "telemetry.jsonl"
    now = time.time()
    with p.open("w", encoding="utf-8") as fh:
        for i, cores in enumerate(series):
            # gmtime, NOT localtime: `telemetry.collect` stamps frames with ``time.gmtime()`` and
            # the trailing ``Z`` says UTC. This fixture previously wrote LOCAL time under a ``Z``
            # label, which happened to CANCEL the reader's ``time.mktime`` (local) bug — so the
            # tests passed while production was an offset out. Matching the real writer is what
            # lets these tests see that class of defect at all (deep review 2026-07-26, #59).
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(now - (len(series) - i) * 300))
            fh.write(json.dumps({"ts": ts,
                                 "our_jobs": [{"id": "1", "prior": 2.0, "state": "r",
                                               "slots": cores}]}) + "\n")
    return p


def test_report_detects_a_still_CLIMBING_curve(tmp_path):
    """THE case that must not be misread: if concurrency is still rising, re-forecasting the rung
    from the current number would understate what the campaign can reach."""
    p = _write_log(tmp_path, [100, 200, 400, 800, 1200, 1800, 2400, 2800, 3000])
    out = accumulation_report(p, hours=24)
    assert out["status"] == "climbing"
    assert "do NOT re-forecast" in out["advice"]
    assert out["peak_cores"] == 3000


def test_report_detects_a_PLATEAU_and_says_what_to_do_with_it(tmp_path):
    p = _write_log(tmp_path, [630, 640, 636, 628, 640, 636, 632, 638, 636])
    out = accumulation_report(p, hours=24)
    assert out["status"] == "plateaued"
    assert "re-forecast from" in out["advice"]
    assert 600 < out["late_mean_cores"] < 700


def test_report_detects_DECLINE(tmp_path):
    """A falling curve is the kill/retreat signature or a draining queue — never a plateau."""
    out = accumulation_report(_write_log(tmp_path, [3000, 2500, 2000, 1200, 800, 400, 200, 92, 60]),
                              hours=24)
    assert out["status"] == "declining"


def test_report_refuses_to_conclude_from_too_little_data(tmp_path):
    out = accumulation_report(_write_log(tmp_path, [600, 620]), hours=24)
    assert out["status"] == "insufficient"


def test_report_ignores_pre_slots_frames_and_missing_logs(tmp_path):
    """Frames archived BEFORE slots were recorded carry no core count — they must be skipped, not
    counted as zero (which would fake a decline)."""
    p = tmp_path / "t.jsonl"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text("\n".join(
        json.dumps({"ts": ts, "our_jobs": [{"id": "1", "prior": 2.0, "state": "r"}]})
        for _ in range(10)) + "\n", encoding="utf-8")
    assert accumulation_report(p, hours=24)["status"] == "insufficient"
    assert accumulation_report(tmp_path / "nope.jsonl")["status"] == "no-data"


def test_report_survives_a_truncated_tail_line(tmp_path):
    p = _write_log(tmp_path, [600] * 9)
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "2026-07-26T0')      # a crash mid-write
    assert accumulation_report(p, hours=24)["status"] == "plateaued"


def test_frames_are_read_as_UTC_so_the_GO_day_window_is_not_truncated(tmp_path):
    """The `Z` on a frame means UTC — reading it as LOCAL silently truncates the window (#59).

    `telemetry.collect` stamps `ts` with ``time.gmtime()`` and marks it ``Z``; the reader used
    ``time.mktime``, which interprets its struct as LOCAL time. Under BST — which covers the whole
    campaign window — every frame therefore read back 3600 s OLDER than it was.

    MEASURED before the fix, on 18 frames written 10 min apart across exactly 3.0 h: the report
    returned ``n=12`` (a third of the window silently discarded) and ``early_mean_cores=250``
    against a true 150, so the climb ratio read 1.64 instead of 2.73. Near the 1.15/0.85 thresholds
    that can FLIP the verdict, and ``early_mean_cores`` is the number the operator re-forecasts the
    seed rung from on GO day. This pins the UTC contract end-to-end."""
    now = time.time()
    p = tmp_path / "utc.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(18):                       # 18 frames x 10 min = exactly 3.0 h
            t = now - (17 - i) * 600
            fh.write(json.dumps({
                # stamped the way `collect` really stamps it: gmtime + a Z marker
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
                "our_jobs": [{"id": "1", "prior": 2.0, "state": "r", "slots": 100 + i * 20}],
            }) + "\n")

    out = accumulation_report(p, hours=3.0)
    assert out["n"] == 18, f"frames were dropped from the window: n={out['n']} of 18"
    # the early third is frames 0-5 -> 100,120,140,160,180,200 -> mean 150
    assert out["early_mean_cores"] == 150, out["early_mean_cores"]
    assert out["status"] == "climbing"
    # and the ratio the verdict turns on is computed over the FULL window
    assert out["late_mean_cores"] / out["early_mean_cores"] > 2.0

    # a frame one second inside the window is kept; one comfortably outside is not
    edge = tmp_path / "edge.jsonl"
    with edge.open("w", encoding="utf-8") as fh:
        for offset in (3.0 * 3600 - 60, 60):      # just inside, and very recent
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - offset)),
                "our_jobs": [{"id": "1", "prior": 2.0, "state": "r", "slots": 500}],
            }) + "\n")
        fh.write(json.dumps({                     # 5 h old -> must be excluded
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 5 * 3600)),
            "our_jobs": [{"id": "1", "prior": 2.0, "state": "r", "slots": 999}],
        }) + "\n")
    res = accumulation_report(edge, hours=3.0)
    assert res["status"] == "insufficient" and res["n"] == 2, res


def test_a_FAILED_probe_is_not_logged_as_a_zero_capacity_frame(tmp_path, monkeypatch, capsys):
    """`collect` never raises, so a timed-out ssh yields EMPTY sections.

    Appending that produces a "0 jobs, 0 slots" frame which reads on the capacity curve as the
    campaign LOSING the whole cluster. Observed live 2026-07-28T10:25Z between frames of ~1,400
    slots. Absence is honest; a false zero is not.
    """
    from src.cluster.telemetry import Snapshot, append_log

    log = tmp_path / "telem.jsonl"

    failed = Snapshot(ts="2026-07-28T10:25:22Z", pool_free={}, cluster_qw=0, cluster_users=0,
                      our_jobs=[], probe_states={}, errors=["ssh rc=255: timed out"])
    healthy = Snapshot(ts="2026-07-28T10:41:07Z", pool_free={}, cluster_qw=7, cluster_users=1,
                       our_jobs=[{"id": "1", "prior": "1.8", "state": "r", "slots": 4}],
                       probe_states={})

    # The guard the advisor applies before logging.
    for snap in (failed, healthy):
        if snap.errors and not snap.our_jobs:
            continue
        append_log(snap, log)

    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1, "the failed probe must not appear as a frame at all"
    assert rows[0]["ts"] == "2026-07-28T10:41:07Z"

    # A frame that legitimately observes zero RUNNING jobs (no transport error) is still recorded:
    # that is real data, not a probe failure.
    idle = Snapshot(ts="2026-07-28T11:00:00Z", pool_free={}, cluster_qw=0, cluster_users=0,
                    our_jobs=[{"id": "2", "prior": "1.8", "state": "qw", "slots": 1}],
                    probe_states={})
    if not (idle.errors and not idle.our_jobs):
        append_log(idle, log)
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 2, "a genuine all-queued observation is real data and must be kept"
