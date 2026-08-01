"""Cluster adapter unit tests (§12.4 B-A1): spec batches, the §14.6 jobscript, run_one routing.

No network, no GPU, no Myriad: run_one's heavy calls are monkeypatched (the real end-to-end is
the G1 on-cluster dry-run acceptance); everything else is exercised for real.
"""
from __future__ import annotations

import json
import re

import pytest

from src.cluster.jobscript import render_jobscript
from src.cluster.spec_io import payload_sha, read_spec, write_specs

SPEC = {"arm": "distributional", "reward_kind": "source", "reward": "def reward(...): ...",
        "candidate_id": "distributional-g0-c0", "archive_root": "/scratch/out", "seed": 0}


def test_spec_roundtrip_and_index(tmp_path):
    n = write_specs([SPEC, [SPEC, {**SPEC, "seed": 1}]], tmp_path)  # single + a pack
    assert n == 2
    loaded = read_spec(tmp_path / "task_1.json")
    assert loaded == SPEC
    pack = read_spec(tmp_path / "task_2.json")
    assert isinstance(pack, list) and pack[1]["seed"] == 1
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["1"] == payload_sha(SPEC)


def test_write_specs_rejects_non_json_serializable_specs(tmp_path):
    """A non-JSON-native field (would be silently str-coerced by default=str AND pass the sha check,
    since payload_sha coerces too) must FAIL LOUD at write time instead of mis-training on the node."""
    import numpy as np

    bad = {"candidate_id": "c0", "arm": "x", "weird": np.int64(5)}  # numpy scalar smuggled in
    with pytest.raises(TypeError, match="not cleanly JSON-serializable"):
        write_specs([bad], tmp_path / "bad")
    # a clean spec (all native types) still writes + round-trips
    ok = {"candidate_id": "c0", "arm": "x", "seed": 5, "window": [0, 100]}
    write_specs([ok], tmp_path / "ok")
    assert read_spec(tmp_path / "ok" / "task_1.json") == ok


def test_spec_sha_mismatch_fails_loud(tmp_path):
    write_specs([SPEC], tmp_path)
    p = tmp_path / "task_1.json"
    tampered = dict(SPEC, seed=999)
    p.write_text(json.dumps(tampered, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="sha MISMATCH"):
        read_spec(p)


def test_jobscript_encodes_every_researched_rule():
    # NOTE (2026-07-31): the explicit ``priority=-100`` below is DELIBERATE and stays. The campaign
    # no longer ASKS for a negative priority anywhere (record §54 retired the §14.3 ladder), but the
    # RENDERER must still faithfully emit whatever it is handed — `run_campaign_cluster.py` keeps a
    # documented `--priority` + `--allow-deprioritise` escape hatch. The guard belongs at the
    # campaign layer, which is where `test_cluster_campaign.py` asserts no batch is ever negative.
    js = render_jobscript("s1_search", 630, "/home/u/Scratch/llmrp",
                          "/acfs/users/u/llmrp-inputs", pool="EF", tc=38,
                          priority=-100, hold_jid="marker_1", pack=3)
    for needle in ("#!/bin/bash -l", "-l gpu=1", "-pe smp 12", "-l mem=4G", "-l tmpfs=15G",
                   "-l h_rt=1:30:0", "-ac allow=EF", "#$ -r y", "#$ -p -100",
                   "-t 1-630 -tc 38", "#$ -hold_jid marker_1", "umask 077",
                   'if cp "/acfs/users/u/llmrp-inputs"/*.parquet "$TMPDIR/gold/"',
                   'export LLM_RP_GOLD_STAGED_DIR="$TMPDIR/gold"',
                   'export LLM_RP_GOLD_STAGED_DIR="/acfs/users/u/llmrp-inputs"',
                   'export PYTHONPATH="$HOME/llmrp:',  # BUG-4: `src` importable (repo != -wd)
                   "--pack 3", "epilogue.jsonl"):
        assert needle in js, f"missing: {needle}"


def test_jobscript_pack1_defaults_and_priority_guard():
    js = render_jobscript("t", 10, "/r", "/inputs")
    assert "-pe smp 4" in js and "-l h_rt=3:0:0" in js and "#$ -p 0" in js
    with pytest.raises(ValueError, match="<= 0"):
        render_jobscript("t", 1, "/r", "/p", priority=5)


# --- the CPU lane (2026-07-26; dossier MYRIAD_EXPERT_DOSSIER §0-PRE) ------------------------
# Live-measured: 636 CPU cores held vs 0 GPUs granted in 3 days. The d pool alone is 10,584 cores
# against 74 GPUs cluster-wide, so the CPU lane is the substrate that actually gets scheduled.


def test_jobscript_cpu_lane_requests_no_gpu_and_no_pool_allow():
    """A CPU job must not request a GPU or pin a GPU pool - that is what lets it place on any of
    the ~294 d nodes instead of queueing behind the 74 contended GPUs."""
    js = render_jobscript("cpu_batch", 100, "/home/u/Scratch/llmrp", "/inputs", device="cpu")
    assert "-l gpu=1" not in js
    assert "-ac allow=" not in js
    assert "#$ -pe smp 1" in js          # 1 core per packed training, not the GPU lane's 4
    assert "#!/bin/bash -l" in js


def test_jobscript_cpu_lane_sizes_one_core_per_packed_training():
    """CPU trainings are single-threaded (multi-thread BLAS would break CRN determinism), so the
    GPU lane's 4-cores-per-training default would over-request 4x."""
    assert "#$ -pe smp 8" in render_jobscript("b", 1, "/r", "/i", pack=8, device="cpu")
    assert "#$ -pe smp 32" in render_jobscript("b", 1, "/r", "/i", pack=8)  # cuda lane unchanged


def test_jobscript_cpu_memory_is_sized_from_the_measured_peak_not_a_flat_4g():
    """The CPU lane's memory request must come from the MEASURED footprint, and scale with the PACK.

    Regression for record §38/§43. The flat ``mem=4G`` per slot asked 32 GB for an 8-slot search job
    whose measured peak is 1.64 GB (n=55, qacct scoped to our own RUN-4 tasks). On Myriad memory, not
    slots, is the scarce consumable, so that over-request was the binding placement constraint: eight
    canary jobs identical except one field showed a 15 h 8-slot job waiting 43-46 min at ``mem=4G``
    and placing at the FIRST scheduling pass at 1G/2G/3G. It is also what makes the C4 target
    unreachable -- 1,000 jobs (the ``max_u_jobs`` cap, = 4,000 cores at 4 cores each) would reserve
    16 TB against ~12 TB of free pool-d memory at 4G, and 8 TB at 2G.

    A previous draft of the fix used a 4x headroom, which computes 6.8G/slot for the pack-4 lane --
    LARGER than the 4G it replaced. Hence the explicit per-lane expectations below.
    """
    # search lane: one training on eight threads -> 1G/slot = 8 GB/job, 4.9x the 1.64 GB peak
    search = render_jobscript("s", 1, "/r", "/i", device="cpu", pack=1, cores=8)
    assert "#$ -l mem=1G" in search, search.splitlines()[3]

    # packed test lane: four concurrent trainings on four cores -> 2G/slot = 8 GB/job, 1.29x the
    # measured 5.86-6.16 GB peak. NOT below it: a request under the real footprint is dishonest even
    # though Myriad enforces the value as a reservation rather than a limit.
    packed = render_jobscript("t", 1, "/r", "/i", device="cpu", pack=4, cores=4)
    assert "#$ -l mem=2G" in packed, packed.splitlines()[3]

    # the GPU lane is deliberately untouched -- the measurement was made on CPU tasks
    assert "#$ -l mem=4G" in render_jobscript("g", 1, "/r", "/i", pack=3)

    # an explicit value always wins
    assert "#$ -l mem=7G" in render_jobscript(
        "x", 1, "/r", "/i", device="cpu", pack=1, cores=8, mem_per_core="7G")


def test_jobscript_gpu_lane_is_byte_unchanged_by_the_cpu_build():
    """Regression lock: adding the CPU lane must not alter a single GPU-lane directive."""
    js = render_jobscript("s1_search", 630, "/home/u/Scratch/llmrp", "/acfs/users/u/llmrp-inputs",
                          pool="EF", tc=38, priority=-100, hold_jid="marker_1", pack=3)
    for needle in ("-l gpu=1", "-ac allow=EF", "-pe smp 12", "-l h_rt=1:30:0"):
        assert needle in js, f"GPU lane regressed: {needle}"


def test_jobscript_refuses_the_exclusive_whole_node_core_count():
    """THE 2026-07-26 STARVATION LOCK: `-pe smp 36` makes UCL's JSV add exb/exd, so the job needs
    an ENTIRELY EMPTY node. Job cpucurve_d sat queued 2+ days on exactly this."""
    with pytest.raises(ValueError, match="EXCLUSIVE"):
        render_jobscript("t", 1, "/r", "/i", cores=36)
    # 35 is clean - 97% of a node with none of the exclusivity penalty
    assert "#$ -pe smp 35" in render_jobscript("t", 1, "/r", "/i", cores=35)


def test_jobscript_containerized_job_guards_against_a_missing_apptainer():
    """node-d00a-230 had no apptainer (2026-07-26): the venv python lives INSIDE the .sif, so the
    task burned its granted slot on a bare rc=127. Fail with a NAMED error instead."""
    js = render_jobscript("t", 1, "/r", "/i", apptainer_sif="/home/u/python311.sif")
    assert "command -v apptainer" in js and "FATAL apptainer missing" in js
    # ...and a non-containerized job must not carry the guard
    assert "command -v apptainer" not in render_jobscript("t", 1, "/r", "/i")


def test_jobscript_cpu_lane_drops_nv_from_the_apptainer_line():
    """--nv injects the host NVIDIA stack; meaningless on a CPU node."""
    js = render_jobscript("t", 1, "/r", "/i", device="cpu", apptainer_sif="/home/u/python311.sif")
    assert "apptainer exec --bind" in js and "--nv" not in js


def test_jobscript_rejects_an_unknown_device():
    with pytest.raises(ValueError, match="device must be"):
        render_jobscript("t", 1, "/r", "/i", device="tpu")


def test_jobscript_entry_module_defaults_to_run_one_and_can_select_the_bayes_chain():
    """The chain runner must be REACHABLE from a jobscript, else the critical-path optimisation
    exists only in tests. Default behaviour is unchanged."""
    assert "-m src.cluster.run_one --spec" in render_jobscript("t", 1, "/r", "/i")
    chain = render_jobscript("bo_chain", 1, "/r", "/i", device="cuda",
                             entry_module="src.cluster.bayes_chain")
    assert "-m src.cluster.bayes_chain --spec" in chain
    # the RUN LINE must not still call run_one (an explanatory comment in the template does
    # mention the module name, so match the run-line form, not a bare substring)
    assert "-m src.cluster.run_one --spec" not in chain


def test_jobscript_rejects_tilde_and_relative_paths():
    """2026-07-11 rehearsal incident regression: '~' is expanded by NOTHING the template touches
    (SGE '#$' directives, double-quoted bash strings, PYTHONPATH) — the rendered '#$ -wd ~/...'
    sent every array to Eqw at dispatch, where UCL's cleanup deleted them with no qacct record.
    The render choke point must fail LOUD on any tilde, and on a non-absolute remote_root."""
    with pytest.raises(ValueError, match="ABSOLUTE"):
        render_jobscript("t", 2, "~/Scratch/llmrp", "/inputs")
    with pytest.raises(ValueError, match="ABSOLUTE"):
        render_jobscript("t", 2, "Scratch/llmrp", "/inputs")
    for kwargs in (
        {"gold_dir": "~/Scratch/inputs"},
        {"gold_dir": "/inputs", "venv": "~/venvs/llmrp"},
        {"gold_dir": "/inputs", "repo_root": "~/llmrp"},
        {"gold_dir": "/inputs", "apptainer_sif": "~/python311.sif"},
    ):
        gold = kwargs.pop("gold_dir")
        with pytest.raises(ValueError, match="literal '~'"):
            render_jobscript("t", 2, "/r", gold, **kwargs)
    # $HOME IS allowed in the shell-only paths (double-quoted bash expands variables)
    js = render_jobscript("t", 2, "/r", "/inputs")
    assert 'export PYTHONPATH="$HOME/llmrp:' in js and "~" not in js
    # 2026-07-12 regression: MSYS path conversion mangled '/acfs/...' into 'C:/Program Files/Git/
    # acfs/...' on the laptop CLI — every task died at the Apptainer mount. Fail at render.
    with pytest.raises(ValueError, match="drive-letter"):
        render_jobscript("t", 2, "/r", "C:/Program Files/Git/acfs/users/u/gold")


def test_expand_remote_and_remote_home():
    """The user-facing '~' paths are expanded ONCE against the real remote $HOME (resolved via an
    explicit remote shell — the quoted ssh runner keeps a bare $HOME argv word literal)."""
    from src.cluster.submit import expand_remote, remote_home

    assert expand_remote("~/Scratch/llmrp", "/home/ucestes") == "/home/ucestes/Scratch/llmrp"
    assert expand_remote("~", "/home/u") == "/home/u"
    assert expand_remote("/abs/path", "/home/u") == "/abs/path"
    with pytest.raises(ValueError, match="~user"):
        expand_remote("~other/x", "/home/u")

    calls: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> str:
        calls.append(cmd)
        return "/home/ucestes\n"

    assert remote_home(fake_runner) == "/home/ucestes"
    # `-c`, NOT `-lc` (deep review 2026-07-26, #63). A LOGIN shell is required in the JOBSCRIPT
    # (`#!/bin/bash -l` loads the module system — MYRIAD_DEEP_RESEARCH §5), but NOT here: `$HOME`
    # is set by sshd from the passwd entry before any profile runs, and sourcing the profile only
    # adds a stdout noise source — module-load/notice lines that a shared HPC routinely prints.
    # The reason this needs a shell at all is QUOTING (see the docstring above), not login-ness.
    assert calls == [["sh", "-c", 'printf %s "$HOME"']]
    with pytest.raises(RuntimeError, match="remote \\$HOME"):
        remote_home(lambda _cmd: "garbage")
    # a resolution polluted by a trailing banner must FAIL LOUD, not become the remote root: it
    # would land in the jobscript's `#$ -wd`, and an invalid -wd is dispatch-time Eqw with no trace
    with pytest.raises(RuntimeError, match="remote \\$HOME"):
        remote_home(lambda _cmd: "/home/ucestes\nWelcome to Myriad!\n")


def test_write_jobscript_forces_lf_endings(tmp_path):
    """V11 regression: the driver runs on Windows — a platform-translated CRLF shebang
    (``#!/bin/bash -l\\r``) makes qsub/exec fail on the cluster. Bytes must be LF-pure."""
    from src.cluster.jobscript import write_jobscript

    js = render_jobscript("t", 2, "/r", "/inputs")
    p = write_jobscript(js, tmp_path / "sub" / "t.sh")
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.startswith(b"#!/bin/bash -l\n")


def test_epilogue_line_produces_valid_json_under_real_bash(tmp_path):
    """Execute the template's trickiest escaping — the epilogue echo — under a REAL bash and
    parse the result as JSON (what the torn-line-tolerant ledger reader will ingest)."""
    import shutil
    import subprocess

    # Resolve a bash that actually RUNS: on Windows, which("bash") can hit the WSL shim in
    # System32, which fails with a relay error when no distro is installed (seen live after
    # the 2026-07-10 OS in-place repair). Probe each candidate and use the first working one.
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\usr\bin\bash.exe"]
    bash = None
    for cand in candidates:
        if not cand:
            continue
        try:
            probe = subprocess.run(
                [cand, "-c", "echo ok"], capture_output=True, text=True, timeout=15
            )
        except OSError:
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            bash = cand
            break
    if bash is None:
        pytest.skip("no working bash on this host")
    js = render_jobscript("t", 2, "/r", "/inputs")
    # P17/A4-F10: the epilogue echo now rides an EXIT trap (so a SOFT kill still records the
    # task) — extract the trap-installation line and exercise the REAL trap under bash.
    trap_line = next(ln for ln in js.splitlines() if ln.startswith("trap 'echo "))
    ledger = tmp_path / "t.epilogue.jsonl"
    script = (
        "SGE_TASK_ID=7\nRC=0\nGPUINFO='Tesla V100-SXM2-16GB, 525.105'\n"
        + trap_line.replace('"/r/ledger/t.epilogue.jsonl"', f'"{ledger.as_posix()}"')
        + "\nexit 0"
    )
    try:
        r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)
    except OSError:
        pytest.skip("bash present but not runnable")
    assert r.returncode == 0, r.stderr
    row = json.loads(ledger.read_text().strip())
    assert row["task"] == 7 and row["rc"] == 0 and "V100" in row["gpu"]
    # -notify must accompany the trap: SIGUSR2 precedes the h_rt SIGKILL, so the trap actually
    # gets a chance to fire on a walltime kill (a bare SIGKILL is untrappable)
    assert "#$ -notify" in js and "trap 'exit 143' TERM USR2" in js


def test_run_one_routing_and_exit_semantics(monkeypatch, tmp_path):
    import src.cluster.run_one as ro

    calls: list[str] = []
    archived: list[str] = []

    def fake_train(spec):
        calls.append(spec["candidate_id"])
        return {"ok": spec["seed"] != 13, "candidate_id": spec["candidate_id"]}

    import src.orchestration.parallel as par
    monkeypatch.setattr(par, "train_candidate", fake_train)
    monkeypatch.setattr(par, "_archive", lambda r, arm, opts, root, gen=0: archived.append(r["candidate_id"]))

    # single spec, ok -> archived, exit 0
    write_specs([SPEC], tmp_path)
    assert ro.main(["--spec", str(tmp_path / "task_1.json")]) == 0
    assert archived == [SPEC["candidate_id"]]

    # single spec, failing seed -> NOT archived, exit 1
    write_specs([{**SPEC, "seed": 13, "candidate_id": "bad-c0"}], tmp_path / "b2")
    assert ro.main(["--spec", str(tmp_path / "b2" / "task_1.json")]) == 1
    assert "bad-c0" not in archived


def test_run_one_routes_the_test_leg_to_the_sealed_worker_with_node_env_fp(monkeypatch, tmp_path):
    """A leg=='test' spec routes to _test_seed_worker (NOT train_candidate), archives via write_run,
    and OVERRIDES the record's env_fingerprint with a NODE-captured one (S6 homogeneity parity)."""
    import src.cluster.run_one as ro
    import src.orchestration.parallel as par
    import src.orchestration.test_leg as tl

    written: list[dict] = []
    monkeypatch.setattr(tl, "_test_seed_worker",
                        lambda spec: {"ok": True, "run_id": spec["run_id"], "arm": spec["arm"],
                                      "record": {"run_id": spec["run_id"], "arm": spec["arm"],
                                                 "seed": spec["seed"], "env_fingerprint": "DRIVER-fp"}})
    monkeypatch.setattr(par, "_run_env_fp", lambda root, rid, opts: "NODE-fp")
    monkeypatch.setattr("src.io.results.write_run", lambda rec, root: written.append(rec))
    # train_candidate must NOT be called for a test leg
    monkeypatch.setattr(par, "train_candidate",
                        lambda spec: (_ for _ in ()).throw(AssertionError("search worker on a test leg")))

    test_spec = {"leg": "test", "run_id": "distributional-s0", "arm": "distributional", "seed": 0,
                 "archive_root": str(tmp_path / "out")}
    write_specs([test_spec], tmp_path / "t")
    assert ro.main(["--spec", str(tmp_path / "t" / "task_1.json")]) == 0
    assert len(written) == 1 and written[0]["run_id"] == "distributional-s0"
    assert written[0]["env_fingerprint"] == "NODE-fp"  # driver label overridden by the node capture


# --- -tc is LANE-AWARE: a GPU pool width must never govern a CPU array (2026-07-27) -----------

def _tc_of(js: str) -> int:
    line = next(ln for ln in js.splitlines() if " -t " in ln and "-tc " in ln)
    return int(line.split("-tc ")[1].split()[0])


def test_cuda_lane_keeps_the_pool_width_throttle_UNCHANGED():
    """Regression guard: 38 IS the right number on cuda -- it is the EF/V100 pool width, a real
    physical limit. The lane-aware default must not disturb the GPU path at all."""
    from src.cluster.jobscript import render_jobscript

    js = render_jobscript("b", 40_000, "/h/llmrp", "/gold", pool="EF", device="cuda")
    assert _tc_of(js) == 38


def test_cpu_lane_does_NOT_self_throttle_to_a_gpu_count():
    """38 is a GPU COUNT. On the CPU lane the d pool alone is 10,584 cores and ~5,800 sat free at
    the 2026-07-27 probe, so a 38-wide throttle would cap the campaign at ~0.7% of the machine.

    Inert under the documented launch config (``--chunk-tasks 1`` renders one-task arrays, where a
    throttle of 38 on ``-t 1-1`` does nothing) -- but ``--chunk-tasks`` DEFAULTS TO None, and that
    legacy path submits the whole round as ONE array, where 38 would silently become the campaign's
    concurrency ceiling. The real governors are max_u_jobs and killswitch.plan_footprint, both of
    which are designed for the job and reserve capacity for other users; this one was not.
    """
    from src.cluster.jobscript import render_jobscript

    js = render_jobscript("b", 40_000, "/h/llmrp", "/gold", pool="d", device="cpu",
                          pack=1, cores=1)
    assert _tc_of(js) == 40_000, "the CPU lane must not impose a throttle of its own"


def test_an_EXPLICIT_tc_still_wins_on_both_lanes():
    """The lane-aware value is a DEFAULT, not a policy -- an operator who asks for a throttle
    (e.g. to stand down under contention) must still get exactly the one they asked for."""
    from src.cluster.jobscript import render_jobscript

    for device, pool in (("cuda", "EF"), ("cpu", "d")):
        js = render_jobscript("b", 40_000, "/h/llmrp", "/gold", pool=pool, device=device,
                              pack=1, cores=1, tc=12)
        assert _tc_of(js) == 12


def test_bad_node_fence_is_opt_in_and_leaves_the_default_BYTE_IDENTICAL():
    """A node that fails a job in SECONDS is always free, so the scheduler keeps feeding it work and
    it keeps destroying it -- a job vacuum. Found live 2026-07-27: `node-d00a-230` has no apptainer
    and ate 3 jobs inside an hour, each dying at the container guard. Self-healing (no record, so
    resume re-runs) but pure waste across 42,128 trainings."""
    from src.cluster.jobscript import render_jobscript

    base = dict(pool="d", device="cpu", pack=1, cores=1)
    plain = render_jobscript("b", 4, "/h/llmrp", "/gold", **base)
    fenced = render_jobscript("b", 4, "/h/llmrp", "/gold",
                              exclude_hosts=["node-d00a-230", "node-x-1"], **base)

    assert not [ln for ln in plain.splitlines() if ln.startswith("#$ -l h=")],         "the default must render NO host constraint"
    assert "#$ -l h=!node-d00a-230&!node-x-1" in fenced.splitlines()
    # and nothing else moves
    assert plain.splitlines() == [ln for ln in fenced.splitlines()
                                  if not ln.startswith("#$ -l h=")]
    # empty / whitespace entries are ignored rather than emitting a broken directive
    assert not [ln for ln in render_jobscript("b", 4, "/h/llmrp", "/gold",
                                              exclude_hosts=["", "  "], **base).splitlines()
                if ln.startswith("#$ -l h=")]


def test_tmpfs_is_sized_from_the_MEASURED_stage_not_a_round_number():
    """Regression for record §60: `tmpfs` is a CONSUMABLE, so an over-request caps jobs-per-node.

    Measured 2026-07-31: we staged 71 MB of gold and requested 15G — a 216x over-request that made
    only 11 of 348 pool-d hosts eligible and pinned us to 1.18 jobs per node on 36-slot machines
    where four 8-slot jobs fit. The bound below keeps a healthy multiple of the staged bytes while
    staying under the 2G cliff where host eligibility collapses.
    """
    js = render_jobscript("t", 1, "/r", "/g", device="cpu", pack=1, cores=8)
    m = re.search(r"^#\$ -l tmpfs=(\d+)G", js, re.M)
    assert m, "the jobscript must request tmpfs explicitly"
    gb = int(m.group(1))
    assert gb <= 1, (
        f"tmpfs={gb}G: at 2G or above only 11 of 348 pool-d hosts qualify, which caps us at ~1 job "
        f"per node regardless of free slots (record §60)"
    )
    # ...and still comfortably above the ~71 MB actually staged.
    assert gb * 1024 >= 71 * 10, f"tmpfs={gb}G leaves under 10x headroom over the 71 MB staged"
