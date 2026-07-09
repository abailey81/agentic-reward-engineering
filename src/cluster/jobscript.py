"""Render the §14.6 Myriad jobscript (Grid Engine array) — the ONE template every batch uses.

Encodes every researched rule/lever: ``#!/bin/bash -l`` (R13), per-core ``-l mem`` (R13),
``$TMPDIR`` gold staging (§14.2), ``-r y`` native resume (§14.1), the ``-p`` intra-user
priority ladder (§14.3), ``-hold_jid`` pipelining, organized ``-o`` logs, ``umask 077`` (§14.5),
the per-task epilogue ledger line (§14.4), and §15 GPU packing via ``run_one --pack``.

ALWAYS write the rendered text to disk via :func:`write_jobscript` (V11 audit fix): the driver
runs on Windows, where a bare ``write_text`` translates ``\\n`` to ``\\r\\n`` — and a CRLF
shebang (``#!/bin/bash -l\\r``) makes qsub/exec fail on the cluster. The classic landmine,
killed at the API level.

PATH CONTRACT (2026-07-11 incident fix): every path handed to :func:`render_jobscript` must be
**tilde-free**, and ``remote_root`` must be **absolute**. ``~`` is expanded by NOTHING this
template touches: SGE ``#$`` directives (``-wd``/``-o``) expand neither ``~`` nor ``$HOME``
(an invalid ``-wd`` puts the whole array in ``Eqw`` at dispatch, where UCL's cleanup deletes it
with NO qacct record — the observed fate of the 2026-07-11 rehearsal), double-quoted bash
strings keep ``~`` literal (``mkdir -p "~/..."`` creates a directory named ``~``; the Apptainer
``--bind`` list fails on the nonexistent literal path), and Python never expands ``~`` in
``PYTHONPATH``. ``$HOME`` IS allowed in the shell-only paths (``gold_dir``/``venv``/
``repo_root``/``apptainer_sif``: they appear only inside the bash body, where double quotes
expand variables) but NOT in ``remote_root`` (directive sink). Callers expand user-supplied
``~`` against the REAL remote home via ``submit.remote_home`` + ``submit.expand_remote``.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["render_jobscript", "write_jobscript"]

_TEMPLATE = """#!/bin/bash -l
#$ -N {name}
{gpu_line}#$ -pe smp {cores}
#$ -l mem={mem_per_core}
#$ -l tmpfs={tmpfs}
#$ -l h_rt={h_rt}
{pool_line}{excl_line}#$ -notify
#$ -r y
#$ -p {priority}
#$ -t 1-{n_tasks} -tc {tc}
{hold_line}#$ -wd {remote_root}
#$ -o {remote_root}/logs/{name}/$TASK_ID.o -j y
umask 077
mkdir -p "{remote_root}/logs/{name}" "{remote_root}/ledger"
# The Apptainer run below bind-mounts {gold_dir}; a NON-EXISTENT bind source makes Apptainer
# FATAL at container creation BEFORE any python runs (caught live 2026-07-24: the synthetic
# rehearsal crashed rc=255/secs=1 every dispatch on a missing inputs/ dir). mkdir -p makes the
# mount always succeed: a synthetic run needs no gold; a REAL run whose gold is genuinely absent
# then fails LOUD in the loader (a clear "panel not found") instead of a cryptic mount FATAL.
mkdir -p "{gold_dir}"
# Gold artifacts staged to node-local tmpfs (§14.2) — LOAD-BEARING via the loaders' staged-dir
# hook (V7 closed 2026-07-07): LLM_RP_GOLD_STAGED_DIR is honoured per file by its canonical
# <name>_<suffix>.parquet filename (wrong-suffix staging = filename miss -> canonical fallback)
# and staged bytes are still checksum-verified against the frozen manifest. If the tmpfs copy
# fails, the ACFS input dir itself is exported instead — gold reads keep working either way
# (the repo-default data/gold/ does not exist on a node; licensed gold is never in git).
mkdir -p "$TMPDIR/gold"
if cp "{gold_dir}"/*.parquet "$TMPDIR/gold/"; then
  export LLM_RP_GOLD_STAGED_DIR="$TMPDIR/gold"
else
  export LLM_RP_GOLD_STAGED_DIR="{gold_dir}"
fi
export TORCH_HOME="$TMPDIR/torch"
{cpu_lane_guard}{env_line}
# Make `src` importable: the job's -wd is {remote_root} (Scratch, where logs/specs/outputs live),
# but the CODE is at {repo_root} and the venv does NOT pip-install the repo, so without this
# `python -m src.cluster.run_one` dies with ModuleNotFoundError on EVERY task. $HOME auto-binds
# into Apptainer, so this also works inside the container.
export PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"
# P17/A4-F10 (2026-07-13 audit): the epilogue line rides an EXIT trap so a SOFT kill (SGE sends
# SIGUSR2/SIGTERM ahead of the h_rt SIGKILL) still records the task — the old echo-after-the-run
# missed exactly the walltime-killed/node-failed cases the ledger exists for. RC=126 marks
# "trapped before the run returned"; a hard SIGKILL remains unrecordable (forensics via qacct).
RC=126
{apptainer_guard}GPUINFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
# `ts` (epoch seconds at task exit) is REQUIRED, not decorative (deep review 2026-07-26, #56):
# killswitch.classify_task_deaths judges an ADMIN KILL by burstiness — N deaths across M hosts
# inside a 300s window. Without a timestamp its window admits every row ever written, so a whole
# campaign's scattered benign failures read as one burst and the killswitch retreats + blocks
# submission. MEASURED: 12 ordinary deaths spread over 20 days classified as `admin_kill`.
# The `|| echo 0` is NOT defensive noise: a bare `$(date +%s)` that yields nothing emits a
# `"ts":` with NO VALUE — invalid JSON — and `read_epilogue` is torn-line tolerant, so it
# silently DISCARDS the row. The death record is then lost, defeating the very detector `ts` was
# added for, and the loss is invisible. Reproduced 2026-07-26: `bash.exe` invoked directly (no
# MSYS PATH) has no `date`, and the epilogue test failed to parse at exactly that column. A
# numeric fallback keeps the row PARSEABLE and merely un-timed, which the window treats as old.
trap 'echo "{{\\"task\\":${{SGE_TASK_ID}},\\"host\\":\\"$(hostname)\\",\\"gpu\\":\\"${{GPUINFO}}\\",\\"rc\\":${{RC}},\\"secs\\":${{SECONDS}},\\"ts\\":$(date +%s 2>/dev/null || echo 0)}}" >> "{remote_root}/ledger/{name}.epilogue.jsonl"' EXIT
trap 'exit 143' TERM USR2
{launcher} -m {entry_module} --spec "{remote_root}/specs/{name}/task_${{SGE_TASK_ID}}.json" --pack {pack}
RC=$?
exit $RC
"""

#: Peak resident footprint of ONE concurrent 400k-step training, in GB, MEASURED on RUN 4's own tasks
#: (2026-07-30, record §38.3/§43.3): ``maxvmem`` p50 1.57 / max **1.64** over n=55 completed 8-slot
#: search tasks, and 5.86-6.16 GB over the pack-4 ``c1_baselines_pNN`` tasks (= ~1.55 GB each). The
#: job's need therefore scales with the PACK, not with the slot count. Sized from the archive rather
#: than assumed, and the scoping matters: the harvested ``qacct`` files also contain other users'
#: accounting from 2022-23, so an unscoped read gives a number that is not ours.
_MEASURED_PEAK_GB_PER_TRAINING = 1.64


def render_jobscript(
    name: str,
    n_tasks: int,
    remote_root: str,
    gold_dir: str,
    *,
    pool: str = "EF",
    tc: int | None = None,
    priority: int = 0,
    hold_jid: str | None = None,
    pack: int = 1,
    cores: int | None = None,
    mem_per_core: str | None = None,
    #: 2026-07-31 (record §60): 15G -> 1G. **`tmpfs` is a CONSUMABLE** (`qconf -sc`:
    #: `tmpfs  scratch  MEMORY  <=  YES  JOB  10G`), so the request is RESERVED per job and a node
    #: can host only `total_tmpfs / request` of our jobs no matter how many slots are free.
    #: MEASURED on the live estate 2026-07-31:
    #:   * what we actually stage: the ACFS gold dir is **71 MB** (plus a small TORCH_HOME)
    #:   * what we asked: **15G** -- a **216x** over-request
    #:   * pool-d hosts with >=15G tmpfs free: **11 of 348**; with >=1G free: **348 of 348**
    #:   * consequence, measured directly: our 60 running jobs sat on **51 distinct hosts, 1.18 jobs
    #:     per node**, on 36-slot nodes where FOUR 8-slot jobs fit. Slots were never the limit --
    #:     our own scratch reservation was.
    #: This is the §38 memory defect one consumable over: a round number nobody had measured against
    #: the thing it reserves. Sized here from the measurement with ~14x headroom over the staged
    #: bytes, which is also exactly the value that clears the eligibility cliff (2G already drops us
    #: back to 11 hosts).
    #: SAFE BY CONSTRUCTION, not merely by margin: the jobscript stages gold with
    #: `if cp ...; then export LLM_RP_GOLD_STAGED_DIR=...`, so a copy that does not fit simply falls
    #: back to reading the ACFS input dir (see the staging comment above). An undersized tmpfs
    #: therefore degrades I/O; it cannot fail a training. And tmpfs is outside the determinism
    #: envelope -- it changes where bytes are read from, never the arithmetic.
    #: SCOPED TO THE CPU LANE, following §38's precedent: the measurement was made on pool-d CPU
    #: tasks, so the GPU lane keeps its 15G and stays byte-unchanged (there is a regression test that
    #: asserts exactly that). ``None`` => lane default; an explicit value always wins.
    tmpfs: str | None = None,
    h_rt: str | None = None,
    venv: str = "$HOME/venvs/llmrp",
    repo_root: str = "$HOME/llmrp",
    apptainer_sif: str | None = None,
    device: str = "cuda",
    entry_module: str = "src.cluster.run_one",
    exclude_hosts: list[str] | None = None,
) -> str:
    """Return the §14.6 jobscript text for one array batch.

    Defaults implement the researched decisions: EF/V100 pool, ``-tc`` = the full pool width,
    priority 0 (the caller passes the §14.3 ladder value: 0 / -100 / -200 / -500), cores scale
    with the pack (4 per concurrent training), walltime scales with the pack wave (1h30 per
    §15's one-wave shape, 3h for pack=1's conservative margin).

    ``device`` selects the LANE (2026-07-26, the CPU-lane build — dossier §0-PRE):

    * ``"cuda"`` (default, unchanged): requests ``-l gpu=1`` + ``-ac allow=<pool>``.
    * ``"cpu"``: NO ``gpu`` request and NO pool ``allow`` (a CPU job places on any node type, which
      is the whole point — the d pool alone is 10,584 cores vs 74 GPUs cluster-wide), ``--nv`` is
      dropped from the Apptainer line, and ``cores`` defaults to ONE PER PACKED TRAINING (a
      training is single-threaded; the 4-per-training GPU sizing would over-request 4x).
      Pass ``pool`` explicitly to pin a CPU node type (e.g. ``"D"``/``"B"``/``"T"``).

    Two guards encode live-probed Myriad facts (2026-07-26); both fail LOUD at render:

    * **``-pe smp 36`` is an EXCLUSIVE WHOLE-NODE request.** UCL's JSV silently adds ``exb=true``
      + ``exd=true`` when the core count equals a full node, so the job needs an ENTIRELY EMPTY
      node and starves (job ``cpucurve_d`` sat queued 2+ days). 35 is clean; 36 is not.
    * **Apptainer is not on every node** (``node-d00a-230`` had none): the venv python lives INSIDE
      the ``.sif``, so a missing container means ``rc=127`` after the slot was already granted. The
      rendered script probes for it and exits with a named error instead.

    ``entry_module`` selects the on-node entry point. The default trains the task's specs; pass
    ``"src.cluster.bayes_chain"`` to run the WHOLE bayes_opt GP chain inside one job instead of
    dispatching its 30 sequential proposals as 30 separate array-of-1 jobs (each paying a queue
    wait). That packaging change is what makes the GPU lane worth having for the campaign's
    critical path — see ``src/cluster/bayes_chain.py`` and ``src/cluster/lanes.py``.
    """
    if pack < 1:
        raise ValueError(f"pack must be >= 1, got {pack}")
    device = str(device).lower()
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
    # ``-tc`` IS LANE-AWARE (deep review, 2026-07-27). The historical default was a literal 38 —
    # "the full pool width" of EF/V100, i.e. a GPU-COUNT. On the CPU lane that number governs
    # nothing real: the d pool alone is 10,584 cores and ~5,800 sat free at the 2026-07-27 probe,
    # so a 38-wide throttle would be a self-imposed cap of ~0.7 % of the machine. It is INERT under
    # the documented launch config (``--chunk-tasks 1`` renders one-task arrays, where a throttle
    # of 38 on ``-t 1-1`` does nothing) — but ``--chunk-tasks`` DEFAULTS TO None, and that legacy
    # path submits the whole round as ONE array, where 38 silently becomes the concurrency ceiling
    # for the entire campaign. A latent footgun, and exactly the class of artifact-read-as-law that
    # produced the "96-core ceiling" (a 12-job probe) and cost us weeks of wrong planning.
    # So: cuda keeps the pool width; cpu imposes NO throttle of its own, because a renderer has no
    # business inventing a capacity policy out of a GPU count.
    #
    # NOTE: BE PRECISE ABOUT WHAT THEN GOVERNS — an earlier draft of this comment claimed
    # ``killswitch.plan_footprint`` as a governor, which is FALSE and is the R85 failure mode
    # ("a pin nobody can verify is FICTIONAL"). ``plan_footprint`` is ADVISORY ONLY: its sole
    # caller is ``allocation.advise_cpu_lane`` (the GO-day advisor a human reads), and NOTHING in
    # the submission path consults it. What actually governs is ``max_u_jobs`` (1000 array jobs)
    # and SGE fair-share, which arbitrates our entitlement against every other user before it
    # dispatches anything — so removing this throttle cannot starve the cluster, it can only stop
    # us from starving OURSELVES. What is genuinely NOT enforced is the stated 1000-core courtesy
    # RESERVE; that policy lives only in the advisor's printout. Tracked in
    # the enforcement point is the SUBMIT layer (which can
    # query live capacity), never this function, which stays a pure deterministic renderer.
    # An explicit ``tc`` always wins, and is how an operator enforces a stand-down.
    if tc is None:
        tc = 38 if device == "cuda" else max(1, int(n_tasks))
    if priority > 0:
        raise ValueError("SGE -p only accepts <= 0 for users (self-deprioritization)")
    # PATH CONTRACT (2026-07-11 rehearsal incident — see the module docstring). Fail LOUD here,
    # at the single render choke point, instead of silently shipping a script whose array Eqw-dies
    # at dispatch (deleted with no qacct trace) or whose every task dies on a literal-~ path.
    if not remote_root.startswith("/"):
        raise ValueError(
            f"remote_root must be an ABSOLUTE path (got {remote_root!r}): it lands in the SGE "
            "'#$ -wd'/'#$ -o' directives, which expand neither '~' nor '$HOME' — an invalid -wd "
            "sends the whole array to Eqw at dispatch. Expand via submit.remote_home/expand_remote."
        )
    for label, val in (
        ("remote_root", remote_root), ("gold_dir", gold_dir), ("venv", venv),
        ("repo_root", repo_root), ("apptainer_sif", apptainer_sif or ""),
    ):
        if "~" in val:
            raise ValueError(
                f"{label} contains a literal '~' (got {val!r}): nothing in the jobscript expands "
                "it (SGE directives, double-quoted bash strings, PYTHONPATH). Pass an absolute "
                "path, or '$HOME/...' for shell-only paths; expand user input via "
                "submit.remote_home + submit.expand_remote."
            )
        if ":" in val.split("=", 1)[-1][:12] and (":/" in val or ":\\" in val):
            # 2026-07-12 incident: Git Bash (MSYS) path conversion rewrote a laptop CLI arg
            # '/acfs/users/.../gold' into 'C:/Program Files/Git/acfs/...' BEFORE Python saw it —
            # every task of the batch then died at the Apptainer mount (rc=255, 1s). A Windows
            # drive-letter path can never be valid on the cluster: fail at render, not on-node.
            # (Launcher-side fix: prefix the command with MSYS_NO_PATHCONV=1, or keep POSIX paths
            # in argparse DEFAULTS rather than shell argv.)
            raise ValueError(
                f"{label} looks like a Windows drive-letter path (got {val!r}) — Git Bash MSYS "
                "path conversion has mangled a POSIX argument. Launch with MSYS_NO_PATHCONV=1 "
                "or avoid passing absolute POSIX paths through the shell."
            )
    # CPU trainings are single-threaded (the 2026-07-25 profile: ~97% is the SAC gradient update,
    # and multi-thread BLAS would change float reduction order = break CRN determinism), so the
    # GPU lane's 4-cores-per-training sizing would over-request 4x on CPU.
    cores = cores if cores is not None else ((1 if device == "cpu" else 4) * pack)
    if int(cores) == 36 or (device == "cpu" and int(cores) >= 36):
        raise ValueError(
            f"cores={cores}: a full-node core request makes UCL's JSV add the EXCLUSIVE complexes "
            "(exb/exd/ext), so the job needs an ENTIRELY EMPTY node and starves — live-probed "
            "2026-07-26 (job cpucurve_d, '-pe smp 36', queued 2+ days). Use <= 35 (35 still gets "
            "97% of a node); 8 places best."
        )
    # MEMORY IS SIZED FROM THE MEASUREMENT, NOT FROM A ROUND NUMBER (2026-07-30; record §38, §43).
    #
    # The old flat ``mem_per_core = "4G"`` asked 32 GB for an 8-slot search job whose measured peak is
    # 1.64 GB — a 19.5x over-request — and on Myriad MEMORY, not slots, is the scarce consumable. Two
    # consequences, both measured:
    #   * DISPATCH LATENCY. Eight one-off canary jobs, identical except one field: at ``mem=4G`` an
    #     8-slot 15 h job waited 43-46 min; at 1G/2G/3G the same job placed at the FIRST scheduling
    #     pass, four times out of four. Walltime was NOT the discriminator (4/8/12/15 h all placed in
    #     the same window), which independently reproduces the 15/15 result already recorded in
    #     ``autosize_h_rt``'s docstring.
    #   * THE C4 CEILING. ``max_u_jobs = maxujobs = 1000``. At 4 cores/job that is exactly the 4,000
    #     cores at which the registered makespan model saturates — but 1,000 jobs x 16 GB = 16 TB of
    #     reservation against ~12 TB of free pool-d memory, i.e. unreachable. At 2G/slot it is 8 TB.
    #     **This sizing is the precondition for the 4,000-core target, not a queue-time nicety.**
    #
    # MEASURED FOOTPRINTS (our own jobs, qacct scoped by job name inside RUN 4's window — the
    # harvested files also contain OTHER USERS' accounting, so the scoping is load-bearing):
    #   search lane, pack 1 on 8 slots : maxvmem p50 1.57 GB, max 1.64 GB   (n=55)
    #   test lane,   pack 4 on 4 slots : maxvmem 5.86-6.16 GB               (c1_baselines_pNN, exit 0)
    # i.e. ~1.55-1.64 GB per CONCURRENT TRAINING, so the need scales with the PACK, not the slots.
    #
    # NOTE: AN EARLIER DRAFT OF THIS FIX USED A 4x HEADROOM, which computes 6.8G/slot for the pack-4 lane
    # — LARGER than the 4G it replaces, i.e. it would have made placement worse while looking like a
    # fix. Caught by measuring the pack-4 peak instead of inferring it. 1.3x on the measured peak
    # lands on 1G/slot for the search lane (8 GB/job, 4.9x its 1.64 GB) and 2G/slot for the packed
    # lane (8 GB/job, 1.29x its 6.2 GB).
    #
    # ENFORCEMENT, probed on-node: with ``mem=2G`` a job sees ``ulimit -v unlimited``, ``Max address
    # space unlimited``, no cgroup memory limit and only an informational ``SGE_UCL_MEM``; a canary
    # then held 3 GiB — 1.5x the per-slot value — for 90 s and exited rc=0. The request is a
    # SCHEDULING RESERVATION, not a kill limit. An explicit ``mem_per_core`` always wins.
    #
    # SCOPED TO THE CPU LANE ON PURPOSE. The footprints above were measured on CPU tasks, and the
    # campaign is CPU-only; the ``cuda`` branch keeps its historical ``4G`` so this change cannot move
    # a lane it was not measured on (and the GPU-lane render test keeps asserting exactly that).
    if mem_per_core is None:
        if device == "cpu":
            _need_gb = _MEASURED_PEAK_GB_PER_TRAINING * max(1, int(pack)) * 1.3
            mem_per_core = f"{max(1, int(round(_need_gb / max(1, int(cores))))):d}G"
        else:
            mem_per_core = "4G"
    # tmpfs, same shape and the same reasoning one consumable over (record §60). See the parameter's
    # docstring for the measurement: 71 MB staged against a 15G request, which left only 11 of 348
    # pool-d hosts eligible and pinned us to 1.18 jobs per node where four would fit.
    if tmpfs is None:
        tmpfs = "1G" if device == "cpu" else "15G"
    h_rt = h_rt if h_rt is not None else ("3:0:0" if pack == 1 else "1:30:0")
    from src.cluster.submit import sanitize_name

    name = sanitize_name(name)
    hold_line = f"#$ -hold_jid {hold_jid}\n" if hold_jid else ""
    gold_dir = gold_dir.rstrip("/")
    # V3 audit fix: the launcher is PART OF THE RUN LINE — the old apptainer branch set a shell
    # variable the run line never used (containerized jobs would have crashed on bare python).
    # A CPU lane requests no GPU and pins no GPU pool; pass `pool` explicitly to pin a CPU node
    # type (D/B/T) if a contrast ever needs CPU-model homogeneity beyond the seed-block scheme.
    gpu_line = "" if device == "cpu" else "#$ -l gpu=1\n"
    # CPU LANE: make the GPU UNREACHABLE, don't just decline to request it (deep review 2026-07-27).
    # Two live paths would otherwise put CUDA into a CPU campaign. (1) `opts` carries no `device`
    # key and `parallel._spec` adds none, so every spec the bayes_opt GP chain builds on-node
    # arrives at `run_one._run_single` without one and is defaulted to "cuda" there — campaign's
    # explicit injection fires only for specs IT builds. (2) A CPU job lands on a node that may
    # still expose GPUs, so `torch.cuda.is_available()` can be True even though we requested none,
    # and taking that card is exactly the impairment of other users `lanes.py` refuses by
    # construction. Emptying CUDA_VISIBLE_DEVICES makes the substrate a property of the JOB rather
    # than of every spec remembering to say so: torch sees no device, the chain's "cuda" default
    # falls back to CPU, and `run_one._resolved_device` then labels it honestly as cpu so the S6
    # homogeneity audit compares what RAN. Illegal state made unrepresentable, rather than guarded.
    cpu_lane_guard = (
        'export CUDA_VISIBLE_DEVICES=""   # CPU lane: no GPU requested, so none may be used\n'
        if device == "cpu" else ""
    )
    pool_line = f"#$ -ac allow={pool}\n" if (pool and (device == "cuda" or pool not in ("", "EF"))) else ""
    # BAD-NODE FENCE (2026-07-27, found live). A node that fails a job in SECONDS is always FREE, so
    # the scheduler keeps feeding it work and it keeps destroying it — a job vacuum. Observed:
    # `node-d00a-230` has no apptainer and ate 3 of our jobs inside an hour, each dying at the
    # container guard (exit 127). It is self-healing (no record is written, so resume re-runs the
    # spec) but pure waste, and across 42,128 trainings one such node can eat a great many slots.
    # SGE's `hostname` is a requestable HOST complex, so a negated conjunction fences them off —
    # verified accepted by the scheduler before wiring this. None/empty renders NOTHING, so the
    # default jobscript is byte-identical to before.
    _ex = [h.strip() for h in (exclude_hosts or []) if h and h.strip()]
    excl_line = ("#$ -l h=" + "&".join(f"!{h}" for h in _ex) + "\n") if _ex else ""
    if apptainer_sif is None:
        env_line = f"source {venv}/bin/activate"
        launcher = "python"
    else:
        # G1 audit fix (2026-07-10): the container image is BARE python — every dep lives in
        # the venv on $HOME (auto-bound), so the run line must call the VENV interpreter
        # through the container, not the container's own `python`. And $TMPDIR is NOT
        # auto-bound: without the explicit --bind, the staged-gold dir and TORCH_HOME would
        # not exist inside the container and gold reads would fall back to a data/gold that
        # is absent on nodes. gold_dir is bound too, for the cp-failed ACFS fallback path.
        env_line = f"# containerized: {apptainer_sif} (venv python called through the container)"
        # --nv injects the host NVIDIA stack; it is meaningless (and noise) on a CPU node.
        nv = "" if device == "cpu" else "--nv "
        launcher = (
            f'apptainer exec {nv}--bind "$TMPDIR,{gold_dir}" {apptainer_sif} {venv}/bin/python'
        )
    # Apptainer is NOT installed on every node (node-d00a-230, 2026-07-26): the venv python lives
    # INSIDE the .sif, so a missing container burns the granted slot with a bare rc=127 and no
    # diagnosis. Fail with a NAMED error the ledger can count instead.
    apptainer_guard = "" if apptainer_sif is None else (
        'command -v apptainer >/dev/null 2>&1 || { '
        'echo "FATAL apptainer missing on $(hostname) - cannot start the containerized venv" >&2; '
        'exit 127; }\n'
    )
    return _TEMPLATE.format(
        name=name, n_tasks=n_tasks, remote_root=remote_root.rstrip("/"),
        gold_dir=gold_dir, pool=pool, tc=tc, priority=priority,
        hold_line=hold_line, pack=pack, cores=cores, mem_per_core=mem_per_core,
        tmpfs=tmpfs, h_rt=h_rt, env_line=env_line, launcher=launcher,
        repo_root=repo_root.rstrip("/"), gpu_line=gpu_line, pool_line=pool_line,
        excl_line=excl_line,
        apptainer_guard=apptainer_guard, entry_module=entry_module,
        cpu_lane_guard=cpu_lane_guard,
    )


def write_jobscript(text: str, path: str | Path) -> Path:
    """Write a rendered jobscript with FORCED LF endings (V11: Windows driver, Linux cluster).

    ``newline=""`` disables the platform translation that would otherwise smuggle ``\\r`` into
    the shebang and every ``#$`` directive. The bytes on disk are exactly what qsub executes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p
