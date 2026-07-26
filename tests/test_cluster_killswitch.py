"""Behaviour tests for the administrative-kill detector + graduated retreat (2026-07-26).

The asymmetry under test: a FALSE POSITIVE costs a few hours of a run with ~7 days of slack; a
FALSE NEGATIVE (hammering the scheduler right after an admin qdel) costs the Myriad account. So
every ambiguous case must resolve toward retreat, and the concurrency cap must never rise.
"""
from __future__ import annotations

import json

import pytest

from src.cluster.killswitch import (
    INCIDENT_FILENAME,
    classify_task_deaths,
    clear_incident,
    incident_blocks_submission,
    retreat_cap,
    write_incident,
)

NOW = 1_000_000.0


def _deaths(n: int, *, hosts: int, secs: float = 60.0, rc: int = 137, ts: float = NOW):
    """n dead tasks spread round-robin over `hosts` distinct nodes."""
    return [{"host": f"node-d00a-{100 + (i % hosts)}", "rc": rc, "secs": secs, "ts": ts}
            for i in range(n)]


# --- classification ------------------------------------------------------------------------

def test_no_deaths_is_ok_and_continues():
    v = classify_task_deaths([{"host": "node-d00a-1", "rc": 0, "secs": 900, "ts": NOW}], now=NOW)
    assert v.classification == "ok" and v.action == "continue"


def test_single_host_burst_is_a_node_failure_and_requeues():
    """A node falling over kills everything ON THAT NODE. Requeue is correct and harms nobody."""
    v = classify_task_deaths(_deaths(20, hosts=1), now=NOW)
    assert v.classification == "node_failure"
    assert v.action == "requeue"
    assert v.n_hosts == 1


def test_walltime_proximate_deaths_are_not_an_admin_kill():
    """THE KEY FALSE-POSITIVE GUARD: a badly-sized h_rt kills many tasks on many hosts at once and
    would otherwise look exactly like an administrative qdel."""
    v = classify_task_deaths(_deaths(30, hosts=10, secs=2950), now=NOW, h_rt_secs=3000)
    assert v.classification == "walltime"
    assert v.action == "requeue"
    assert v.new_core_cap is None


def test_multi_host_burst_is_an_admin_kill_and_retreats():
    """30 tasks dying across 10 nodes in 5 minutes, none near their walltime, has no benign
    explanation — a human ran qdel over our job list."""
    v = classify_task_deaths(_deaths(30, hosts=10, secs=60), now=NOW,
                             h_rt_secs=3000, current_core_cap=640)
    assert v.classification == "admin_kill"
    assert v.action == "retreat"
    assert v.n_hosts == 10
    assert v.new_core_cap == 320  # halved
    assert "do NOT requeue" in v.reason.lower() or "not requeue" in v.reason.lower()


def test_small_scatter_stays_below_the_admin_threshold():
    """Ordinary flaky failures must not trip the retreat."""
    v = classify_task_deaths(_deaths(3, hosts=2), now=NOW, h_rt_secs=3000)
    assert v.classification == "node_failure" and v.action == "requeue"


def test_stale_events_outside_the_window_are_ignored():
    """A kill is a BURST. Deaths spread over hours are just attrition."""
    old = _deaths(30, hosts=10, ts=NOW - 10_000)
    v = classify_task_deaths(old, now=NOW, h_rt_secs=3000)
    assert v.classification == "ok"


def test_mixed_burst_excludes_walltime_deaths_from_the_evidence():
    """Walltime deaths are stripped first; the remaining genuine burst still trips the detector."""
    events = _deaths(10, hosts=8, secs=2950) + _deaths(12, hosts=6, secs=45)
    v = classify_task_deaths(events, now=NOW, h_rt_secs=3000, current_core_cap=600)
    assert v.classification == "admin_kill"
    assert v.n_deaths == 12  # only the non-walltime ones count as evidence


# --- the monotone cap ----------------------------------------------------------------------

def test_retreat_cap_halves_and_never_rises():
    assert retreat_cap(640) == 320
    assert retreat_cap(320) == 160
    assert retreat_cap(160) == 80
    # ...and parks at the floor rather than collapsing to zero
    assert retreat_cap(80) == 64
    assert retreat_cap(64) == 64
    assert retreat_cap(None) is None
    assert retreat_cap(0) is None


def test_repeated_retreats_are_monotone_non_increasing():
    cap = 640
    seen = [cap]
    for _ in range(6):
        cap = retreat_cap(cap)
        seen.append(cap)
    assert seen == sorted(seen, reverse=True), "the cap must never rise on its own"
    assert min(seen) >= 64


# --- the incident gate ---------------------------------------------------------------------

def test_incident_file_blocks_submission_until_a_human_clears_it(tmp_path):
    v = classify_task_deaths(_deaths(30, hosts=10, secs=60), now=NOW,
                             h_rt_secs=3000, current_core_cap=640)
    assert incident_blocks_submission(tmp_path) == (False, "")  # nothing yet

    p = write_incident(tmp_path, v)
    assert p.name == INCIDENT_FILENAME
    blocked, why = incident_blocks_submission(tmp_path)
    assert blocked and "admin_kill" in why

    clear_incident(tmp_path, who="tamer", note="confirmed not UCL; relaunching at 320")
    blocked, _ = incident_blocks_submission(tmp_path)
    assert not blocked
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["cleared"] is True and data["cleared_by"] == "tamer"


def test_unreadable_incident_file_fails_SAFE_and_blocks(tmp_path):
    """Ambiguity must resolve to 'do not submit' — the expensive error is submitting."""
    (tmp_path / INCIDENT_FILENAME).write_text("{not json", encoding="utf-8")
    blocked, why = incident_blocks_submission(tmp_path)
    assert blocked and "refusing to submit" in why


def test_incident_records_the_operator_runbook(tmp_path):
    """The file must tell a human what to do — it is read under stress, possibly by the operator alone."""
    v = classify_task_deaths(_deaths(30, hosts=10, secs=60), now=NOW, h_rt_secs=3000,
                             current_core_cap=640)
    p = write_incident(tmp_path, v)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "rc-support" in data["what_to_do"]
    assert data["new_core_cap"] == 320


def test_killswitch_is_PURE_and_cannot_touch_the_cluster():
    """Standing order: NEVER lower our SGE priority — and more broadly, this module
    DECIDES, it never ACTS. Locked structurally: no subprocess/os.system/ssh import or call
    anywhere in the module's executable code, so no future edit can smuggle in a `qalter -p`
    (forbidden), a `qdel`, or any other scheduler mutation. Docstrings are exempt — they must be
    free to NAME the forbidden commands in order to explain why they are forbidden.
    """
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "src" / "cluster" / "killswitch.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("subprocess", "os", "paramiko", "fabric", "shutil"):
        assert banned not in imported, f"killswitch must not import {banned} — it must stay pure"

    # ...and no dynamic-execution escape hatch that could re-introduce one at runtime. (Scanning
    # string CONSTANTS is deliberately NOT done: the verdict prose must be free to say
    # "administrative qdel" in order to explain the diagnosis to a human.)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec", "__import__", "compile"), (
                f"dynamic execution ({node.func.id}) would defeat the purity guarantee")


@pytest.mark.parametrize("rc", [137, 143, 127, 255, 126])
def test_any_nonzero_INFRASTRUCTURE_rc_counts_as_a_death(rc):
    """Signal deaths, failed starts and unknown codes all still retreat.

    2026-07-28: ``rc=1`` was REMOVED from this list, deliberately, and the case it used to assert
    now has its own test below asserting the OPPOSITE. That is a corrected assumption, not a
    weakened test — see ``test_a_burst_of_authoring_rejects_is_not_an_admin_kill``.
    """
    v = classify_task_deaths(_deaths(30, hosts=10, secs=60, rc=rc), now=NOW, h_rt_secs=3000)
    assert v.classification == "admin_kill"


def test_a_burst_of_authoring_rejects_is_not_an_admin_kill():
    """rc=1 is OUR OWN verdict on the LLM's reward code — the phenomenon under study.

    ``run_one.main`` returns 1 when any row fails, so a sandbox/contract reject surfaces as a bare
    rc=1 after a few seconds. On the weaker legs those are frequent BY DESIGN (``qwen3.5-9b``'s
    measured gate-pass rate is ~17 %), and a generation's worth lands fast, across many hosts — the
    exact shape this detector calls an administrative qdel.

    Without this exclusion the campaign writes ``MYRIAD_KILL_INCIDENT.json`` and hard-blocks
    submission on ALL twelve lines until a human clears it. MEASURED on the RUN 1 archive: the worst
    300 s rc=1 burst was 7 deaths across 6 hosts — one under ``MIN_DEATHS`` — and only 61 % of RUN 1's
    candidate slots ever reached a node (680 of ~1,119); the cross-line reject collision suppressed
    439 (39 %) UNSUBMITTED. Fixing that makes the flow ~1.65x denser, so the threshold WOULD be
    crossed.
    """
    v = classify_task_deaths(_deaths(30, hosts=10, secs=5, rc=1), now=NOW, h_rt_secs=3000)
    assert v.classification == "ok" and v.action == "continue", (
        f"a burst of authoring rejects was classified {v.classification!r}/{v.action!r} — this "
        "hard-blocks submission on every line until a human intervenes"
    )
    assert "application exit" in v.reason


def test_application_exits_do_not_mask_a_real_admin_kill_in_the_same_window():
    """The exclusion must be surgical: real infrastructure deaths alongside rejects still retreat."""
    events = _deaths(30, hosts=10, secs=5, rc=1) + _deaths(10, hosts=5, secs=60, rc=137)
    v = classify_task_deaths(events, now=NOW, h_rt_secs=3000)
    assert v.classification == "admin_kill" and v.action == "retreat"
    assert v.n_deaths == 10, "the rejects must not be counted as evidence of an administrative kill"


def test_application_exits_are_logged_even_though_they_are_excluded(caplog):
    """Excluded must never mean invisible — a mass self-inflicted failure has to stay greppable."""
    with caplog.at_level("WARNING"):
        classify_task_deaths(_deaths(12, hosts=5, secs=5, rc=1), now=NOW, h_rt_secs=3000)
    # getMessage() applies the lazy %-args the logger was called with; formatting them by hand
    # here raised TypeError on a record whose args are a tuple.
    assert any("APPLICATION_EXITS_EXCLUDED" in r.getMessage() for r in caplog.records)


# --- the proactive footprint governor -------------------------------------------------------
# The 2026-07-26 probe found ~5,000 cores FREE while ~2,000 jobs pended: the queue wants GPUs and
# memory, not plain cores. So taking idle CPU cores is both the fast move and the courteous one.

from src.cluster.killswitch import ABSOLUTE_CORE_CEILING, plan_footprint  # noqa: E402


def test_footprint_takes_what_is_granted_when_the_cluster_is_quiet():
    """90% of 5000 free = 4500, cut to 4000 by the reserve. We take it: declining capacity the
    scheduler has already arbitrated in our favour costs campaign speed and buys nothing."""
    cores, why = plan_footprint(free_cores=5000, pending_jobs=100)
    assert cores == 4000
    assert "quiet" in why and "reserve" in why


def test_footprint_still_scales_down_when_the_cluster_is_busy():
    """Aggressive is not indiscriminate: the share still falls with contention."""
    busy, _ = plan_footprint(free_cores=4000, pending_jobs=3000)
    normal, _ = plan_footprint(free_cores=4000, pending_jobs=1000)
    quiet, _ = plan_footprint(free_cores=4000, pending_jobs=100)
    assert busy < normal <= quiet
    assert busy == 2000 and normal == 2800 and quiet == 3000  # quiet cut by the reserve


def test_the_free_core_RESERVE_is_the_courtesy_guarantee(tmp_path=None):
    """THE rule that makes a 2560 ceiling defensible: we never consume the last cores, so our
    campaign can never be the reason another user waits for a plain CPU slot."""
    from src.cluster.killswitch import FREE_CORE_RESERVE

    # exactly at the reserve -> take nothing at all
    cores, why = plan_footprint(free_cores=FREE_CORE_RESERVE, pending_jobs=0)
    assert cores == 0
    # below the reserve -> still nothing (never negative)
    assert plan_footprint(free_cores=200, pending_jobs=0)[0] == 0
    # just above -> only the headroom, not the 60% share
    cores, why = plan_footprint(free_cores=FREE_CORE_RESERVE + 300, pending_jobs=0)
    assert cores == 300 and "reserve" in why


def test_footprint_never_exceeds_the_self_imposed_ceiling():
    """No live condition may push us past the ceiling, however idle the cluster looks."""
    cores, _ = plan_footprint(free_cores=12_000, pending_jobs=0)
    assert cores == ABSOLUTE_CORE_CEILING


def test_a_standing_retreat_cap_overrides_everything():
    """An uncleared incident encodes a decision no live signal may reverse."""
    cores, why = plan_footprint(free_cores=12_000, pending_jobs=0, retreat_cap_cores=160)
    assert cores == 160
    assert "retreat cap" in why


def test_footprint_is_zero_when_nothing_is_free():
    cores, _ = plan_footprint(free_cores=0, pending_jobs=5000)
    assert cores == 0


def test_an_open_incident_BLOCKS_the_real_submission_choke_point(tmp_path, monkeypatch):
    """The gate must be WIRED, not merely available: build_cluster_run's run_batch is the single
    seam every batch funnels through, so an open incident there stops ALL resubmission."""
    from src.cluster.campaign import build_cluster_run

    v = classify_task_deaths(_deaths(30, hosts=10, secs=60), now=NOW,
                             h_rt_secs=3000, current_core_cap=640)
    write_incident(tmp_path, v)

    run = build_cluster_run(
        remote_root="/home/u/Scratch/llmrp", remote_outputs_root="/home/u/Scratch/llmrp/outputs",
        local_batch_root=tmp_path / "batches", local_archive_root=tmp_path,
        gold_dir="/inputs", host="myriad",
    )
    with pytest.raises(RuntimeError, match="submission BLOCKED"):
        run.run_batch([{"run_id": "r1"}], "any_batch")

    # NB: we deliberately do NOT call run_batch again after clearing. Past the gate it performs a
    # REAL ssh push/submit and the driver rides out transport failure for `max_transport_outage_secs`
    # (12 h by default) — calling it in a unit test hangs the suite rather than failing it. That the
    # gate re-opens on clearance is covered directly by
    # test_incident_file_blocks_submission_until_a_human_clears_it.
    clear_incident(tmp_path, who="tamer")
    assert incident_blocks_submission(tmp_path) == (False, "")


def test_governor_and_retreat_compose_monotonically():
    """After successive incidents the governor must keep shrinking, never rebound on a quiet
    cluster — the cap is a ratchet."""
    cap = ABSOLUTE_CORE_CEILING
    prev = None
    for _ in range(4):
        cap = retreat_cap(cap)
        cores, _ = plan_footprint(free_cores=12_000, pending_jobs=0, retreat_cap_cores=cap)
        assert prev is None or cores < prev
        prev = cores


def test_UNDATED_rows_cannot_masquerade_as_a_burst(caplog):
    """A row with no ``ts`` must never be counted as "within the window" (deep review #56).

    The jobscript's epilogue trap emitted ``task/host/gpu/rc/secs`` and NO timestamp, while
    ``_recent`` admitted any row whose ``ts`` was missing. The 300s window was therefore INERT on
    every real ledger: a whole campaign's scattered, unrelated failures read as one burst.
    MEASURED before the fix — 12 ordinary deaths spread over 20 DAYS across 6 hosts classified as
    ``admin_kill`` -> ``retreat``, which halves the core cap AND writes the incident file that blocks
    all submission until a human clears it. Over a 23-day campaign that false positive was close to
    certain, and it would have fired unattended.

    Note the asymmetry this module is built around (see its docstring): a false negative costs the
    Myriad account. So undated rows are excluded from the burst evidence but the exclusion is
    COUNTED and WARNED — never silent."""
    import logging

    now = 1_700_000_000.0
    day = 86_400.0

    # exactly the shape the jobscript used to emit: no "ts"
    undated = [{"task": i, "host": f"node-{i % 6:02d}", "rc": 1, "secs": 900.0} for i in range(12)]
    with caplog.at_level(logging.WARNING):
        v = classify_task_deaths(undated, now=now, current_core_cap=636)
    assert v.classification == "ok" and v.action == "continue", (
        f"12 undated deaths were treated as a burst: {v.classification}/{v.action}"
    )
    assert v.n_undated == 12
    assert v.new_core_cap is None, "an undated ledger must not trigger a cap retreat"
    assert "killswitch_UNDATED_EPILOGUE_ROWS" in caplog.text, "the degradation must be LOUD"

    # the SAME events, correctly timestamped days apart, are also benign
    spread = [dict(e, ts=now - (20.0 - i * 1.5) * day) for i, e in enumerate(undated)]
    v_spread = classify_task_deaths(spread, now=now, current_core_cap=636)
    assert v_spread.classification == "ok" and v_spread.n_undated == 0

    # and a GENUINE burst is still caught — the fix must not blind the detector
    burst = [{"task": i, "host": f"node-{i % 5:02d}", "rc": 137, "secs": 120.0, "ts": now - i * 20.0}
             for i in range(10)]
    v_burst = classify_task_deaths(burst, now=now, current_core_cap=636)
    assert v_burst.classification == "admin_kill" and v_burst.action == "retreat"
    assert v_burst.new_core_cap == 318 and v_burst.n_undated == 0

    # a non-numeric ts is undated too, not a crash
    junk = [{"task": 1, "host": "n1", "rc": 1, "secs": 5.0, "ts": "not-a-number"}]
    assert classify_task_deaths(junk, now=now).n_undated == 1


def test_jobscript_epilogue_stamps_a_timestamp():
    """The emitter must write ``ts`` — the burst window is meaningless without it (#56)."""
    import json as _json
    import re

    from src.cluster.jobscript import render_jobscript

    txt = render_jobscript(name="arr", n_tasks=2, remote_root="/scratch/x/run", gold_dir="/scratch/x/gold")
    line = next(ln for ln in txt.splitlines() if "epilogue.jsonl" in ln)
    # Asserted as an INVARIANT (ts is stamped from date) rather than as one exact command string:
    # this previously pinned `$(date +%s)` verbatim, which made the emitter un-hardenable.
    assert '\\"ts\\":$(date +%s' in line, f"epilogue trap does not stamp ts: {line}"

    # the emitted line must still be VALID JSON once the shell expands it
    tpl = re.search(r'echo "(.*)" >>', line).group(1).replace('\\"', '"')
    cmd = re.search(r'"ts":(\$\(date[^)]*\))', tpl).group(1)   # whatever the ts command now is
    base = (tpl.replace("${SGE_TASK_ID}", "7").replace("$(hostname)", "node-a1")
               .replace("${GPUINFO}", "A100").replace("${RC}", "137")
               .replace("${SECONDS}", "120"))
    row = _json.loads(base.replace(cmd, "1769472000"))
    assert set(row) == {"task", "host", "gpu", "rc", "secs", "ts"}
    assert isinstance(row["ts"], int)

    # ...AND it must still be valid JSON when `date` is UNAVAILABLE and the substitution yields
    # nothing. Without a numeric fallback the row reads `"ts":` with no value, `read_epilogue`
    # discards it as a torn line, and the death record `ts` exists to timestamp is silently LOST —
    # reproduced 2026-07-26 under a bash with no MSYS PATH, where `date` is simply absent.
    assert "|| echo 0" in cmd, f"ts command has no numeric fallback: {cmd}"
    degraded = _json.loads(base.replace(cmd, "0"))
    assert degraded["ts"] == 0 and isinstance(degraded["ts"], int)


def test_enforcement_writes_the_incident_on_retreat_but_NEVER_on_undated_rows(tmp_path, caplog):
    """The driver must ENFORCE a retreat — and must not be fooled by an undated ledger (#57/#56).

    RATIFIED 2026-07-26. Detection alone does not stop the driver resubmitting after an
    administrative qdel, which is the escalation the module exists to prevent; an alert nobody reads
    at 03:00 is not a control. But enforcement must trust only evidence it can place in time: an
    undated ledger is exactly the #56 false-positive shape (a whole campaign's scattered failures
    read as one burst), so it must produce NO incident."""
    import logging
    import time

    from src.cluster.campaign import _enforce_kill_switch
    from src.cluster.killswitch import INCIDENT_FILENAME, incident_blocks_submission

    root = tmp_path / "archive"
    ledger = root / "ledger"
    ledger.mkdir(parents=True)
    now = time.time()

    def _write(rows):
        (ledger / "arr.epilogue.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # 1. UNDATED rows that WOULD have looked like a burst -> no incident, loud warning
    _write([{"task": i, "host": f"node-{i % 6:02d}", "rc": 1, "secs": 900.0} for i in range(12)])
    with caplog.at_level(logging.WARNING):
        _enforce_kill_switch(root)
    assert not (root / INCIDENT_FILENAME).exists(), "an UNDATED ledger must never trigger a retreat"
    assert "NOT enforced" in caplog.text
    assert incident_blocks_submission(root)[0] is False

    # 2. benign dated rows -> still no incident
    _write([{"task": i, "host": f"node-{i % 6:02d}", "rc": 1, "secs": 900.0,
             "ts": now - (20.0 - i * 1.5) * 86400.0} for i in range(12)])
    _enforce_kill_switch(root)
    assert not (root / INCIDENT_FILENAME).exists()

    # 3. a GENUINE admin-kill burst -> incident written AND submission blocked
    _write([{"task": i, "host": f"node-{i % 5:02d}", "rc": 137, "secs": 120.0, "ts": now - i * 20.0}
            for i in range(10)])
    _enforce_kill_switch(root)
    inc = root / INCIDENT_FILENAME
    assert inc.exists(), "a real admin kill must write the incident file"
    blocked, why = incident_blocks_submission(root)
    assert blocked is True and "admin_kill" in why
    payload = json.loads(inc.read_text(encoding="utf-8"))
    assert payload["action"] == "retreat" and payload["cleared"] is False
    assert payload["new_core_cap"] is None or payload["new_core_cap"] > 0

    # 4. re-running is idempotent-safe (rewrites the same block, never compounds)
    _enforce_kill_switch(root)
    assert incident_blocks_submission(root)[0] is True

    # 5. a human clears it -> submission is released again
    from src.cluster.killswitch import clear_incident
    clear_incident(root, who="tamer", note="verified with RC support")
    assert incident_blocks_submission(root)[0] is False
