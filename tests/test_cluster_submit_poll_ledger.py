"""submit/poll/ledger layer tests — fake runners and synthetic archives; zero network."""
from __future__ import annotations

import json
import subprocess

import pytest

from src.cluster.ledger import failures, parse_qacct, read_epilogue, requeue_specs
from src.cluster.poll import completed_run_ids, pending_specs, spec_run_id
from src.cluster.submit import parse_job_id, qsub, submit_marker


def test_parse_job_id_array_and_plain_and_garbage():
    assert parse_job_id("Your job-array 4211375.1-630:1 (\"s1\") has been submitted") == "4211375"
    assert parse_job_id("Your job 99 (\"m\") has been submitted") == "99"
    with pytest.raises(RuntimeError, match="could not parse"):
        parse_job_id("qsub: error")


def test_qsub_and_marker_use_the_runner_and_hold_chain():
    sent: list[list[str]] = []

    def fake_runner(cmd):
        sent.append(cmd)
        return "Your job-array 777.1-10:1 submitted" if "qsub" in cmd else "Your job 778 ok"

    assert qsub("/remote/s1.sh", fake_runner) == "777"
    mid = submit_marker("m_s1", "777", "/remote", fake_runner)
    assert mid == "777" or mid == "778"  # id parsed from whatever the runner returned
    joined = " ".join(sent[-1])
    assert "-hold_jid 777" in joined and "h_rt=0:5:0" in joined


def test_poll_completed_and_compacted_resume(tmp_path):
    # synthetic archive: two committed records
    for rid in ("distributional-g0-c0", "scalar-s5"):
        d = tmp_path / "search" / rid
        d.mkdir(parents=True)
        (d / "record.json").write_text("{}")
    specs = [
        {"candidate_id": "distributional-g0-c0"},   # done
        {"run_id": "scalar-s5"},                    # done
        {"candidate_id": "distributional-g0-c1"},   # missing
    ]
    assert completed_run_ids(tmp_path) == {"distributional-g0-c0", "scalar-s5"}
    pend = pending_specs(specs, tmp_path)
    assert [spec_run_id(s) for s in pend] == ["distributional-g0-c1"]
    with pytest.raises(KeyError):
        spec_run_id({"arm": "x"})


QACCT = """==============================================================
jobnumber    777
taskid       3
exit_status  1
failed       0
maxvmem      3.1G
==============================================================
jobnumber    777
taskid       4
exit_status  0
failed       0
"""


def test_qacct_parse_failures_and_bounded_requeue(tmp_path):
    rows = parse_qacct(QACCT)
    assert len(rows) == 2
    bad = failures(rows)
    assert len(bad) == 1 and bad[0]["taskid"] == "3"

    specs = {3: {"candidate_id": "c3"}}
    ledger = tmp_path / "perm.jsonl"
    r1 = requeue_specs([3], specs, ledger)          # retry 1
    r2 = requeue_specs([3], {3: r1[0]}, ledger)     # retry 2
    r3 = requeue_specs([3], {3: r2[0]}, ledger)     # exhausted -> permanent ledger
    assert r1 and r2 and r3 == []
    perm = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert perm[0]["reason"] == "retries_exhausted"


def test_marker_script_is_real_multiline_printf_not_echo():
    """V2 regression: echo '\\n' does NOT expand in POSIX sh — the marker must be printf-built
    so every #$ directive lands on its own line (else hold/h_rt are dead comments)."""
    sent: list[list[str]] = []

    def fake_runner(cmd):
        sent.append(cmd)
        return "Your job 5 ok"

    from src.cluster.submit import submit_marker
    submit_marker("m_ok", "42", "/r", fake_runner)
    script = " ".join(sent[-1])
    assert "printf '%s\\n'" in script and "'#$ -hold_jid 42'" in script and "echo" not in script
    with pytest.raises(ValueError, match="numeric"):
        submit_marker("m_ok", "42; rm -rf /", "/r", fake_runner)
    with pytest.raises(ValueError, match="invalid SGE job name"):
        submit_marker("bad name!", "42", "/r", fake_runner)


def test_prepare_remote_creates_log_dirs_before_qsub():
    """V4 regression: Grid Engine must be able to open the -o path at job START. V9: outputs/
    is pre-created too, so the first pull's remote find cleanly reads '0 completed'."""
    sent: list[list[str]] = []
    from src.cluster.submit import prepare_remote
    prepare_remote("/home/u/Scratch/llmrp", ["s1_search"], lambda c: sent.append(c) or "")
    assert sent[0][0] == "mkdir" and any("logs/s1_search" in a for a in sent[0])
    assert any(a.endswith("/outputs") for a in sent[0])


def test_spec_missing_index_fails_closed(tmp_path):
    """V6 regression: an index-less batch (partial rsync) must refuse to run, not run unverified."""
    import json as _json
    from src.cluster.spec_io import read_spec
    p = tmp_path / "task_1.json"
    p.write_text(_json.dumps({"candidate_id": "c0"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="index missing"):
        read_spec(p)


def test_write_specs_requires_resume_identity(tmp_path):
    """V6 regression: identity-at-write — a spec without run_id/candidate_id cannot be resumed."""
    from src.cluster.spec_io import write_specs
    with pytest.raises(ValueError, match="resume identity"):
        write_specs([{"arm": "scalar"}], tmp_path)
    with pytest.raises(ValueError, match="resume identity"):
        write_specs([[{"candidate_id": "ok"}, {"arm": "no-id"}]], tmp_path / "b2")


def test_jobscript_apptainer_launcher_is_on_the_run_line():
    """V3 regression: containerized jobs must actually LAUNCH through apptainer.
    G1 regression (2026-07-10, caught against the live cluster): the container image is BARE
    python — the run line must call the VENV interpreter through the container, and $TMPDIR
    (NOT auto-bound by apptainer) plus the gold dir must be explicitly --bind mounted, or the
    staged-gold env var points at a path that does not exist inside the container."""
    from src.cluster.jobscript import render_jobscript
    js = render_jobscript("t2", 5, "/r", "/inputs", apptainer_sif="$HOME/llmrp.sif")
    assert (
        'apptainer exec --nv --bind "$TMPDIR,/inputs" $HOME/llmrp.sif '
        "$HOME/venvs/llmrp/bin/python -m src.cluster.run_one" in js
    )
    # the bare container python must never be the interpreter
    assert "llmrp.sif python -m" not in js
    assert "source $HOME/venvs" not in js


def test_epilogue_reader_tolerates_torn_lines(tmp_path):
    p = tmp_path / "a.epilogue.jsonl"
    p.write_text('{"task":1,"rc":0}\n{"task":2,"rc"\n{"task":3,"rc":1}\n')
    rows = read_epilogue(p)
    assert [r["task"] for r in rows] == [1, 3]


# ---------------------------------------------------------------------------
# V9/V10 deep-dive regressions: no rsync on the driver host + remote requoting
# ---------------------------------------------------------------------------


def test_ssh_runner_requotes_for_the_remote_shell(monkeypatch):
    """V10 regression: ssh space-joins its argv and the REMOTE shell re-splits it — the runner
    must POSIX-quote every word so the remote side reconstructs EXACTLY our argv (round-trip)."""
    import shlex as _sh

    captured: dict[str, list[str]] = {}

    class _P:
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

        def poll(self):
            return 0

        def kill(self):
            pass

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return _P()

    monkeypatch.setattr("src.cluster.submit.subprocess.Popen", fake_popen)
    from src.cluster.submit import ssh_runner

    cmd = ["bash", "-c", "printf '%s\\n' 'a b' > /r/m.sh && qsub /r/m.sh"]
    assert ssh_runner("h")(cmd) == "ok"
    argv = captured["argv"]
    # hardened prefix (BatchMode/accept-new) then the host then the single remote payload word
    assert argv[0] == "ssh" and argv[-2] == "h"
    assert "BatchMode=yes" in argv and "StrictHostKeyChecking=accept-new" in argv
    assert _sh.split(argv[-1]) == cmd  # the property that matters: exact argv reconstruction


def test_push_batch_is_tar_over_ssh_not_rsync(tmp_path, monkeypatch):
    """V9 regression: the Windows driver host has NO rsync (only scp/ssh) — push must stream a
    tar pipe with a POSIX-quoted remote side, and validate the batch dir + name first."""
    import io
    import shlex as _sh

    calls: dict[str, object] = {}

    class _FakeTar:
        def __init__(self):
            self.stdout = io.BytesIO(b"TARBYTES")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, stdout=None, cwd=None, stdin=None):
        calls["tar"] = (argv, cwd)
        return _FakeTar()

    def fake_run(argv, stdin=None, check=None, timeout=None):
        calls["ssh"] = argv

    monkeypatch.setattr("src.cluster.submit.subprocess.Popen", fake_popen)
    monkeypatch.setattr("src.cluster.submit.subprocess.run", fake_run)
    from src.cluster.submit import push_batch

    batch = tmp_path / "s1_search"
    batch.mkdir()
    (batch / "task_1.json").write_text("{}")
    root = "/home/u/Scratch/llm rp/specs"  # a space — exercises the quoting
    push_batch(batch, root + "/")

    tar_argv, tar_cwd = calls["tar"]
    assert tar_argv == ["tar", "-cf", "-", "s1_search"] and tar_cwd == str(tmp_path)
    ssh_argv = calls["ssh"]
    assert ssh_argv[0] == "ssh" and "myriad" in ssh_argv and "rsync" not in " ".join(ssh_argv)
    assert "BatchMode=yes" in ssh_argv  # driver hang-guard present
    q = _sh.quote(root)
    assert f"mkdir -p {q}" in ssh_argv[-1] and f"tar -xf - -C {q}" in ssh_argv[-1]

    with pytest.raises(FileNotFoundError):
        push_batch(tmp_path / "nope", root)
    bad = tmp_path / "bad name!"
    bad.mkdir()
    with pytest.raises(ValueError, match="invalid SGE job name"):
        push_batch(bad, root)


def _fake_find_runner(root: str, rels: list[str]):
    """A runner whose only supported call is the pull's find — returns one line per record."""

    def _run(cmd):
        assert cmd[0] == "find" and cmd[1] == root
        return "\n".join(f"{root}/{r}/record.json" for r in rels) + "\n"

    return _run


def _writing_fetch(payload_note: str = "x", *, omit_record: set[str] = frozenset()):
    """A fetch that materializes the requested dirs in staging (record.json + one sibling)."""
    fetched_batches: list[list[str]] = []

    def _fetch(relpaths, staging):
        fetched_batches.append(list(relpaths))
        for rel in relpaths:
            d = staging / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "returns.parquet").write_text(payload_note)
            if rel not in omit_record:
                (d / "record.json").write_text("{}")

    _fetch.batches = fetched_batches  # type: ignore[attr-defined]
    return _fetch


def test_pull_archive_exact_incremental_staged_and_chunked(tmp_path):
    """V9: pull = remote-vs-local run-dir DIFF (immutable records ⇒ exact), staged whole-dirs-
    only arrival, chunked argv, staging swept afterwards."""
    from src.cluster.poll import pull_archive

    rels = ["search/c0", "search/c1", "test/s1"]
    runner = _fake_find_runner("/r/out", rels)
    # the mirror already holds search/c0 — it must NOT be re-fetched
    pre = tmp_path / "search" / "c0"
    pre.mkdir(parents=True)
    (pre / "record.json").write_text("{}")

    fetch = _writing_fetch()
    n = pull_archive("/r/out", tmp_path, runner=runner, fetch=fetch, chunk=1)
    assert n == 2
    assert fetch.batches == [["search/c1"], ["test/s1"]]  # chunk=1 → one dir per pipe
    assert (tmp_path / "search" / "c1" / "record.json").is_file()
    assert (tmp_path / "test" / "s1" / "returns.parquet").is_file()
    assert not (tmp_path / ".pull_tmp").exists()
    # idempotent second pull: nothing missing, no fetch
    n2 = pull_archive("/r/out", tmp_path, runner=runner, fetch=_writing_fetch())
    assert n2 == 0


def test_pull_archive_torn_dir_fails_loud_and_stale_staging_never_counts(tmp_path):
    """V9: (a) a dir arriving without record.json (torn tar) raises and never enters the
    mirror; (b) a PREVIOUS pull's staging leftovers are swept — neither trusted as complete
    nor counted by completed_run_ids."""
    from src.cluster.poll import completed_run_ids, pull_archive

    # stale staging from a killed pull: contains a record.json — must be ignored + swept
    stale = tmp_path / ".pull_tmp" / "search" / "c9"
    stale.mkdir(parents=True)
    (stale / "record.json").write_text("{}")
    assert completed_run_ids(tmp_path) == set()  # staging is not the mirror

    runner = _fake_find_runner("/r/out", ["search/c9"])
    fetch = _writing_fetch()
    assert pull_archive("/r/out", tmp_path, runner=runner, fetch=fetch) == 1
    assert (tmp_path / "search" / "c9" / "record.json").is_file()  # re-fetched fresh

    # torn transfer: record.json missing in staging → loud failure, mirror untouched
    runner2 = _fake_find_runner("/r/out", ["search/c9", "test/s2"])
    torn = _writing_fetch(omit_record={"test/s2"})
    with pytest.raises(RuntimeError, match="without record.json"):
        pull_archive("/r/out", tmp_path, runner=runner2, fetch=torn)
    assert not (tmp_path / "test" / "s2").exists()


def test_remote_completed_dirs_parses_skips_and_fails_loud():
    from src.cluster.poll import remote_completed_dirs

    out = (
        "/r/out/search/c0/record.json\n"
        "/r/out/record.json\n"            # stray root-level file → skipped
        "/r/out/test/s1/other.txt\n"      # not a record → skipped
        "\n"
    )
    got = remote_completed_dirs("/r/out/", lambda cmd: out)
    assert got == {"search/c0"}

    def failing(cmd):
        import subprocess as sp

        raise sp.CalledProcessError(1, cmd)

    with pytest.raises(RuntimeError, match="wrong outputs root"):
        remote_completed_dirs("/r/out", failing)


def test_pull_archive_reports_per_chunk_progress(tmp_path):
    """P14: the pull ticks its progress callback once per committed chunk so the driver can
    heartbeat mid-pull (a big pull is many pipes, each up to an hour)."""
    from src.cluster.poll import pull_archive

    runner = _fake_find_runner("/r/out", ["search/c0", "search/c1"])
    ticks: list[tuple[int, int]] = []
    n = pull_archive("/r/out", tmp_path, runner=runner, fetch=_writing_fetch(), chunk=1,
                     progress=lambda i, t: ticks.append((i, t)))
    assert n == 2 and ticks == [(1, 2), (2, 2)]


def test_pull_archive_mirrors_reject_markers_incrementally(tmp_path):
    """P9: node-side reject markers (flat JSON under _rejects/) ride the same pull as record
    dirs — exact incremental, staged, idempotent — and permanent_reject_ids reads them."""
    import json as _json

    from src.cluster.poll import permanent_reject_ids, pull_archive

    def runner(cmd):
        assert cmd[0] == "find"
        if "-path" in cmd:  # the reject-marker find
            return "/r/out/search/_rejects/c1.json\n"
        return "/r/out/search/c0/record.json\n"

    def fetch(relpaths, staging):
        for rel in relpaths:
            p = staging / rel
            if "_rejects" in rel:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(_json.dumps(
                    {"run_id": "c1", "permanent": True, "error": "sandbox: bad name"}))
            else:
                p.mkdir(parents=True, exist_ok=True)
                (p / "record.json").write_text("{}")

    n = pull_archive("/r/out", tmp_path, runner=runner, fetch=fetch)
    assert n == 2  # one record dir + one marker
    assert (tmp_path / "search" / "_rejects" / "c1.json").is_file()
    assert permanent_reject_ids(tmp_path) == {"c1"}
    # idempotent second pull — both diffs empty, nothing fetched
    assert pull_archive("/r/out", tmp_path, runner=runner, fetch=fetch) == 0


def test_remote_home_refuses_a_noisy_resolution_instead_of_building_a_garbage_root():
    """A login banner must not end up inside the remote root (deep review #63).

    ``remote_home`` resolved ``$HOME`` through a LOGIN shell (``sh -lc``), which sources the profile
    files — and on a shared HPC those routinely echo module-load or notice lines to STDOUT. The
    validation was ``startswith("/")`` alone, which is asymmetric and fails OPEN: ``.strip()`` clears
    the ends, so banner text BEFORE the path was refused, but banner text AFTER it left "/" at
    position 0 and was ACCEPTED. REPRODUCED — ``home='/home/ucestes\nWelcome to Myriad!'`` expanded
    to ``'/home/ucestes\nWelcome to Myriad!/Scratch/run'``.

    That garbage root goes straight into the jobscript's ``#$ -wd``, which is exactly the 2026-07-11
    incident this helper exists to prevent: an invalid ``-wd`` puts the whole array in ``Eqw`` at
    dispatch, where UCL's cleanup deletes it with NO qacct record — a traceless loss. A submission
    that cannot resolve its own root must fail LOUD."""
    import pytest

    from src.cluster.submit import expand_remote, remote_home

    def _runner(out: str):
        return lambda cmd: out

    # the clean case still works, and expands correctly
    home = remote_home(_runner("/home/ucestes\n"))
    assert home == "/home/ucestes"
    assert expand_remote("~/Scratch/run", home) == "/home/ucestes/Scratch/run"

    # every noisy resolution is REFUSED rather than silently becoming a root
    for bad in ("/home/ucestes\nWelcome to Myriad!\n",     # banner AFTER — the fail-open case
                "Loading modules...\n/home/ucestes\n",     # banner BEFORE
                "/home/ucestes\n\n[NOTICE] maintenance\n",  # MOTD with a blank line
                "/home/uce stes\n",                        # embedded whitespace
                "home/ucestes\n",                          # not absolute
                "\n", ""):                                 # empty
        with pytest.raises(RuntimeError, match=r"remote \$HOME"):
            remote_home(_runner(bad))

    # and the noise SOURCE is removed too: a NON-login shell (profiles are not sourced)
    seen: list[list[str]] = []

    def _capture(cmd):
        seen.append(list(cmd))
        return "/home/ucestes\n"

    remote_home(_capture)
    assert seen[0][:2] == ["sh", "-c"], f"remote_home must not use a login shell: {seen[0]}"


# --------------------------------------------------------------------------------------------
# 2026-07-28 TRANSPORT-LEAK REGRESSION (found live at T+11 h of the confirmatory campaign).
#
# Both tar-over-ssh pipes placed `proc.wait()` AFTER their try/finally, so any exception on the
# consuming side skipped it and left the child running forever. Measured on the live driver: 13
# leaked children, 8 of them pulls still alive 1.1-6.7 h past their own 3600 s timeout, each
# holding a session on the SHARED UCL login node -- which is what makes the NEXT pull stall.
# These tests FAIL against the pre-fix code, which is the only reason to trust them.
# --------------------------------------------------------------------------------------------


class _FakeProc:
    """A child that ignores `wait(timeout=...)` until someone actually kills it."""

    def __init__(self, stdout=None):
        self.stdout = stdout
        self.returncode = None
        self.killed = False
        self.waits: list[float | None] = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def poll(self):
        return self.returncode


def test_reap_kills_a_child_that_outlives_its_grace_and_returns_promptly():
    from src.cluster.submit import reap

    stuck = _FakeProc()
    reap(stuck, grace=0.5)
    assert stuck.killed, "reap must KILL a child that ignores its grace period"
    assert stuck.waits[0] == 0.5, "the caller's grace must be honoured before killing"

    # a child that exits cleanly inside the grace is never killed
    clean = _FakeProc()
    clean.returncode = 0
    reap(clean, grace=300.0)
    assert not clean.killed


def test_failed_pull_reaps_its_ssh_child_instead_of_leaking_it(tmp_path, monkeypatch):
    """THE campaign defect: a stalled pull raises TimeoutExpired, and the ssh child was orphaned."""
    import io

    from src.cluster import poll as _poll

    created: list[_FakeProc] = []

    def fake_popen(argv, stdout=None, stdin=None):
        p = _FakeProc(stdout=io.BytesIO(b"TARBYTES"))
        created.append(p)
        return p

    def fake_run(argv, stdin=None, check=None, timeout=None, cwd=None):
        # exactly what a stalled transfer does to the consuming side
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout or 0)

    monkeypatch.setattr(_poll.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(_poll.subprocess, "run", fake_run)

    fetch = _poll._default_fetch("myriad", "/home/u/Scratch/llmrp/outputs")
    with pytest.raises(subprocess.TimeoutExpired):
        fetch(["test/arm/arm-s1"], tmp_path)

    assert len(created) == 1
    assert created[0].killed, "the ssh child MUST be reaped when the pull fails, never leaked"
    # and the failure-path grace is short: waiting is the cost being eliminated
    assert created[0].waits[0] == 10.0


def test_failed_push_reaps_its_local_tar_child_instead_of_leaking_it(tmp_path, monkeypatch):
    """Same defect on the push side (milder: the child is local, but identical in kind)."""
    import io

    from src.cluster import submit as _submit

    created: list[_FakeProc] = []

    def fake_popen(argv, stdout=None, cwd=None, stdin=None):
        p = _FakeProc(stdout=io.BytesIO(b"TARBYTES"))
        created.append(p)
        return p

    def fake_run(argv, stdin=None, check=None, timeout=None):
        raise subprocess.CalledProcessError(255, argv)

    monkeypatch.setattr(_submit.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(_submit.subprocess, "run", fake_run)

    batch = tmp_path / "c1_search"
    batch.mkdir()
    (batch / "task_1.json").write_text("{}")
    with pytest.raises(subprocess.CalledProcessError):
        _submit.push_batch(batch, "/home/u/Scratch/llmrp/specs")

    assert len(created) == 1
    assert created[0].killed, "the local tar MUST be reaped when the push fails, never leaked"


def test_a_pull_whose_ssh_had_to_be_killed_fails_loud_rather_than_mirroring_short(tmp_path, monkeypatch):
    """A reap that had to KILL means a torn stream; it must never be reported as a good pull."""
    import io

    from src.cluster import poll as _poll

    created: list[_FakeProc] = []

    def fake_popen(argv, stdout=None, stdin=None):
        p = _FakeProc(stdout=io.BytesIO(b"TARBYTES"))
        created.append(p)
        return p

    def fake_run(argv, stdin=None, check=None, timeout=None, cwd=None):
        return None  # the local tar "succeeded" but the remote side never exited

    monkeypatch.setattr(_poll.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(_poll.subprocess, "run", fake_run)

    fetch = _poll._default_fetch("myriad", "/home/u/Scratch/llmrp/outputs")
    with pytest.raises(subprocess.CalledProcessError):
        fetch(["test/arm/arm-s1"], tmp_path)
    assert created[0].killed and created[0].waits[0] == 300.0


def test_one_lines_reject_marker_cannot_condemn_another_lines_candidate(tmp_path):
    """2026-07-28 REGRESSION, reproduced from the live confirmatory campaign.

    All twelve supervised lines share ONE `local_archive_root`, and search candidate ids
    (`scalar-g1-c0`) are reused verbatim by every line. The driver resolved permanent rejects
    mirror-wide, so `search_leg_qwen3_5_9b`'s markers -- the weakest model in the suite -- silently
    abandoned the confirmatory `claude-opus-5` line's identically-named candidates WITHOUT
    submitting them: 439 of 498 abandonments were spurious, 36 of 36 on the core line.

    Completion truth was already sub-root scoped by the 2026-07-19 audit; this asserts reject
    truth uses the SAME scoping, in both directions (a foreign marker must not condemn, an own
    marker still must).
    """
    from src.cluster.poll import permanently_rejected_specs

    def marker(root: str, run_id: str, permanent: bool = True) -> None:
        d = tmp_path / root / "_rejects"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{run_id}.json").write_text(
            json.dumps({"run_id": run_id, "permanent": permanent}), encoding="utf-8"
        )

    def spec(remote_sub: str, cid: str) -> dict:
        # cluster shape: archive_root is the REMOTE sub-root, mirrored under local_root by basename
        return {"candidate_id": cid, "archive_root": f"/home/u/Scratch/llmrp/outputs/{remote_sub}"}

    # the weakest leg genuinely rejected these; the core line never did
    marker("search_leg_qwen3_5_9b", "scalar-g1-c0")
    marker("search_leg_qwen3_5_9b", "scalar-g1-c1")
    # and the core line has one genuine reject of its own
    marker("search", "scalar-g1-c4")

    core = [spec("search", f"scalar-g1-c{i}") for i in range(5)]
    dead = permanently_rejected_specs(core, tmp_path)
    dead_ids = {s["candidate_id"] for s in dead}
    assert dead_ids == {"scalar-g1-c4"}, (
        "only the core's OWN marker may condemn a core candidate; a foreign line's marker for the "
        f"same id must be invisible -- got {sorted(dead_ids)}"
    )

    # symmetric: the leg's own markers still condemn the leg's own candidates
    leg = [spec("search_leg_qwen3_5_9b", f"scalar-g1-c{i}") for i in range(5)]
    leg_dead = {s["candidate_id"] for s in permanently_rejected_specs(leg, tmp_path)}
    assert leg_dead == {"scalar-g1-c0", "scalar-g1-c1"}

    # a TRANSIENT marker is a diagnostic, never an abandonment
    marker("search", "scalar-g1-c2", permanent=False)
    assert {s["candidate_id"] for s in permanently_rejected_specs(core, tmp_path)} == {"scalar-g1-c4"}

    # a spec with no archive_root keeps the legacy mirror-wide behaviour
    legacy = [{"candidate_id": "scalar-g1-c0"}]
    assert {s["candidate_id"] for s in permanently_rejected_specs(legacy, tmp_path)} == {"scalar-g1-c0"}


def test_reject_truth_and_completion_truth_resolve_the_same_archive(tmp_path):
    """The two must never disagree about which archive a spec belongs to -- that asymmetry IS the
    2026-07-28 defect. Asserts both go through `spec_local_root`."""
    from src.cluster.poll import spec_local_root

    s = {"candidate_id": "scalar-g1-c0", "archive_root": "/home/u/Scratch/llmrp/outputs/search"}
    assert spec_local_root(s, tmp_path) == tmp_path / "search"

    # local/pack run: archive_root IS the local record dir
    local = tmp_path / "packroot"
    local.mkdir()
    assert spec_local_root({"archive_root": str(local)}, tmp_path) == local

    # no archive_root: mirror-wide fallback
    assert spec_local_root({}, tmp_path) == tmp_path


def test_the_driver_ssh_timeout_is_bounded_well_below_the_old_300s(monkeypatch):
    """2026-07-28: the 300 s bound was 50x the measured worst-case real latency, and MEASURED on
    the live driver the wait was not in the ssh call at all (no child ever aged past 10 s while
    300 s timeouts were being logged). The bound is what caps the damage, so it is pinned here:
    generous against real latency, but not so generous that a parked thread idles for five
    minutes."""
    from src.cluster import submit as _submit

    assert 60.0 <= _submit._RUNNER_TIMEOUT_SECS <= 180.0, _submit._RUNNER_TIMEOUT_SECS

    captured: dict[str, object] = {}

    class _P:
        returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return "ok", ""

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr("src.cluster.submit.subprocess.Popen", lambda argv, **kw: _P())
    assert _submit.ssh_runner("h")(["qstat", "-r"]) == "ok"
    assert captured["timeout"] == _submit._RUNNER_TIMEOUT_SECS, (
        "ssh_runner must use the module bound, not a hardcoded literal"
    )


def test_ssh_runner_runs_unattended_with_no_inherited_stdin(monkeypatch):
    """2026-07-28: `ssh` forwards stdin to the remote command unless told not to, and the old
    `capture_output=True` form left stdin INHERITED from the driver — whose own stdin is a pipe from
    the supervisor's `| Out-File`. An A/B showed no measurable effect on the stall, so this is not a
    cure; it is simply how ssh should be run unattended, and it removes a class of hazard for free."""
    seen: dict[str, object] = {}

    class _P:
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

        def poll(self):
            return 0

        def kill(self):
            pass

    def fake_popen(argv, **kw):
        seen.update(kw)
        seen["argv"] = argv
        return _P()

    monkeypatch.setattr("src.cluster.submit.subprocess.Popen", fake_popen)
    from src.cluster.submit import ssh_runner

    assert ssh_runner("h")(["qstat", "-r"]) == "ok"
    assert seen["stdin"] is subprocess.DEVNULL, "ssh must not inherit the driver's stdin"
    assert seen["stdout"] is subprocess.PIPE and seen["stderr"] is subprocess.PIPE


def test_ssh_runner_records_whether_the_child_had_ALREADY_EXITED_on_timeout(monkeypatch, caplog):
    """The diagnostic that will localise the phantom 300 s stall if it recurs.

    Seven hypotheses have been tested and refuted. Rather than guess an eighth, the runner records
    the one fact that settles it: was the child already dead when the timeout fired? `poll()` is
    consulted BEFORE the kill, so a non-None returncode proves the wall-clock was spent in the
    PARENT waiting on the pipe — which no remote-side or cluster-side investigation could show.
    """
    import logging

    class _P:
        returncode = None

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
            return "", ""

        def poll(self):
            return 0        # the child ALREADY EXITED

        def kill(self):
            pass

    monkeypatch.setattr("src.cluster.submit.subprocess.Popen", lambda argv, **kw: _P())
    from src.cluster.submit import ssh_runner

    with caplog.at_level(logging.WARNING, logger="src.cluster.submit"):
        with pytest.raises(subprocess.TimeoutExpired):
            ssh_runner("h")(["mkdir", "-p", "/x"])
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "ssh_timeout_diagnostic" in joined
    assert "child_already_exited=True" in joined, (
        "the diagnostic must state whether the child was already dead — that is the whole point"
    )


def test_the_spend_ledger_persists_the_providers_stop_reason(tmp_path):
    """A truncated completion reaches the sandbox as 'defines no callable named reward', identical
    to a model that could not write the code. Without a STRUCTURED stop_reason the per-model
    authoring-reliability table cannot separate a MODEL failure from OUR cap."""
    from src.llm.spend_ledger import record_spend

    led = tmp_path / "spend.jsonl"
    record_spend(led, provider="anthropic", model="claude-opus-5", cost_usd=0.01,
                 tokens_in=10, tokens_out=20, note="realized", stop_reason="max_tokens")
    row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    assert row["stop_reason"] == "max_tokens"

    # and a normal completion still records the field (as None) so the schema is uniform
    record_spend(led, provider="anthropic", model="claude-opus-5", cost_usd=0.01)
    row2 = json.loads(led.read_text(encoding="utf-8").splitlines()[1])
    assert "stop_reason" in row2 and row2["stop_reason"] is None
