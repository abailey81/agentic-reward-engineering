"""Batch-driver kernel (§12.4 driver loop): submit → poll → requeue → until done.

Design law: THE ARCHIVE IS THE ONLY TRUTH. Every cycle re-derives the remaining work from the
compacted-resume diff (``pending_specs`` over pulled records); scheduler output decides only
WHEN to act (no live batch job + work remaining → a submission round), and ``qacct`` text is
harvested as FORENSICS, never as completion truth. SGE's own walltime is the stall detector: a
wedged task is killed at ``h_rt``, the array drains, the job leaves the queue, and the next
cycle's requeue round re-emits exactly the missing run_ids (bounded by
:data:`src.cluster.ledger.MAX_RETRIES`, then permanently ledgered). A driver crash/restart is
harmless for the same reason — rerunning :func:`run_batch` resumes from the diff (retry counts
reset on restart: a human intervened, so a fresh retry budget is the intended semantics).

Everything is injectable (runner/push/pull/sleep/clock) so the FULL loop is unit-tested with
zero network; production wires :func:`src.cluster.submit.ssh_runner` and the tar-over-ssh
transports. The generation-level composition (LLM authoring → specs per arm/generation, marker
hold-chains between search and test arrays) lives with the campaign scripts ON GO; this kernel
is the one they all call.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from src.cluster.jobscript import render_jobscript, write_jobscript
from src.cluster.ledger import requeue_specs
from src.cluster.poll import (
    pending_specs,
    permanently_rejected_specs,
    pull_archive,
    spec_run_id,
)
from src.cluster.spec_io import write_specs
from src.cluster.submit import prepare_remote, push_batch, qsub, sanitize_name, ssh_runner

__all__ = ["submit_batch", "run_batch", "batch_jobs_in_queue"]

_LOG = logging.getLogger(__name__)

#: P14 (2026-07-13 pre-spend audit): the ONLY exception classes the pull loop may ride out as a
#: transport blip. ConnectionError/TimeoutError/OSError = network + ssh; SubprocessError covers
#: CalledProcessError/TimeoutExpired from the tar|ssh pipes; RuntimeError = pull_archive's own
#: loud-but-retryable failures (torn transfer, remote find). Everything else is a local bug.
_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, OSError,
                     subprocess.SubprocessError, RuntimeError)


def _ledgered_run_ids(ledger_path: Path) -> set[str]:
    """Run_ids PERMANENTLY abandoned in a prior run (from the permanent JSONL ledger; torn-line
    tolerant). Consulted on resume so a deterministically-failing spec is not re-tried every restart."""
    ids: set[str] = set()
    if not ledger_path.is_file():
        return ids
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            spec = json.loads(line).get("spec", {}) or {}
            rid = spec.get("run_id") or spec.get("candidate_id")
        except Exception:  # noqa: BLE001 — a torn ledger line just lets that one spec re-try
            continue
        if rid:
            ids.add(str(rid))
    return ids


def batch_jobs_in_queue(
    base_name: str, runner: Callable[[list[str]], str]
) -> tuple[set[str], dict[str, str]]:
    """(full jobnames, {jobname: state}) of this batch's live jobs (``base``/``base_r<k>``).

    Uses ``qstat -r`` because plain ``qstat`` TRUNCATES the name column — an exact-name guard
    on truncated output would silently never match, and the double-submit guard would be dead.
    P1 (2026-07-13 audit): the STATE is now captured too — an ``Eqw`` array previously waited
    forever with green heartbeats (never dispatched ⇒ h_rt never starts ⇒ no bound applied).
    Parsing: each ``qstat -r`` job block starts with the normal queue row (state in column 5),
    followed by indented detail lines including ``Full jobname:`` — pair the last row's state
    with the next full name."""
    names: set[str] = set()
    states: dict[str, str] = {}
    last_state = ""
    for line in runner(["qstat", "-r"]).splitlines():
        stripped = line.strip()
        parts = stripped.split()
        # a queue row: "<job-ID> <prior> <name> <user> <state> ..." — job-ID is all digits
        if parts and parts[0].isdigit() and len(parts) >= 5:
            last_state = parts[4]
        elif stripped.startswith("Full jobname:"):
            nm = stripped.split(":", 1)[1].strip()
            # P17/A2 (2026-07-13 audit): ANCHORED match — bare startswith(base + "_r") also
            # adopted foreign batches like "b1_rehearsal"; only our own requeue rounds
            # (base_r<digits>) and chunk parts (base[_r<digits>]_p<digits>) belong here.
            import re as _re

            if nm == base_name or _re.fullmatch(
                _re.escape(base_name) + r"(_r\d+)?(_p\d+)?", nm
            ):
                names.add(nm)
                states[nm] = last_state
    return names, states


def _chunk_packs(flat: list[dict[str, Any]], pack: int,
                 specs_per_task: int | None = None) -> list[Any]:
    """Flat specs → task payloads: dicts for size 1, lists of ≤N specs for §15 packing.

    ``pack`` is CONCURRENCY — how many trainings run at once inside a task, and therefore the
    job's core request. ``specs_per_task`` is BATCH SIZE — how many trainings that one task
    performs in total. They were the same number until 2026-08-06, and separating them is what
    lets a job hold its cores for several WAVES instead of one.

    WHY THAT IS THE ONLY LEVER LEFT, measured 2026-08-06 rather than inferred: the steady state is
    ``cores = dispatch_rate × duration × 8``, and dispatch_rate is set by fair share and is not
    ours to move — 78 D-pool hosts held ≥8 free slots while we won ZERO dispatches in two hours,
    because dispatch order is decided ENTIRELY by ``ntckts`` (``weight_urgency=0``, so waiting time
    contributes nothing) and ours is the lowest of any major user but one. DURATION is the only
    free variable, and ``ucbtjji`` is the existence proof: ``h_rt`` 48 h against our 15 h, holding
    768 cores from 98 jobs on 1.65× our tickets.

    NOTE: ``specs_per_task > pack`` means MULTIPLE WAVES inside one task. ``run_one`` supports that and
    warns, but sizes nothing: it is the CALLER's job to pass a matching ``--h-rt``. This function
    cannot see the walltime and deliberately does not guess at it.

    NOTE: UNSET IS BYTE-IDENTICAL TO THE OLD BEHAVIOUR. That is the safety property that lets this land
    on a live campaign — nothing changes until a line is restarted with the flag.
    """
    n = int(specs_per_task) if specs_per_task else int(pack)
    if n <= 1:
        return list(flat)
    return [flat[i : i + n] for i in range(0, len(flat), n)]


def submit_batch(
    flat_specs: list[dict[str, Any]],
    name: str,
    *,
    local_batch_root: str | Path,
    remote_root: str,
    gold_dir: str,
    runner: Callable[[list[str]], str],
    push: Callable[..., None] = push_batch,
    pack: int = 1,
    chunk_tasks: int | None = None,
    specs_per_task: int | None = None,
    **jobscript_kwargs: Any,
) -> list[tuple[str, str]]:
    """Write specs + jobscripts locally, prepare the remote tree, push, qsub — possibly as
    MANY SMALL ARRAYS (``chunk_tasks``; see the serialization-policy note below).

    Returns ``[(job_id, batch_dir_name), …]`` (one pair per submitted array). The jobscript
    rides INSIDE each batch dir; the array's task count is the number of task FILES (a §15
    pack of N specs is one task).
    """
    name = sanitize_name(name)
    tasks = _chunk_packs(flat_specs, pack, specs_per_task)
    # CHUNKED SUBMISSION (2026-07-13 max-throughput lever): the scheduler's serialization
    # policy (snx=1, OBSERVED ACTIVE) holds a multi-task array's tail in hqw and self-releases
    # ~one task per ~2 h — a 180-task tier array would crawl for days regardless of free GPUs.
    # ``chunk_tasks`` splits the round into MANY SMALL ARRAYS (``name_p01…``): no pending tail
    # to hold or purge, every part immediately eligible (the 2026-07-13 singles recovery is the
    # live existence proof). Returns ``[(job_id, batch_dir_name), …]`` so the drain's P13
    # attempt-evidence attribution can follow each part's own task files.
    parts: list[tuple[str, list[Any]]] = []
    if chunk_tasks is not None and int(chunk_tasks) >= 1 and len(tasks) > int(chunk_tasks):
        k = int(chunk_tasks)
        for i in range(0, len(tasks), k):
            parts.append((f"{name}_p{i // k + 1:02d}", tasks[i:i + k]))
    else:
        parts.append((name, tasks))
    for pname, ptasks in parts:
        batch_dir = Path(local_batch_root) / pname
        write_specs(ptasks, batch_dir)
        js = render_jobscript(pname, len(ptasks), remote_root, gold_dir, pack=pack,
                              **jobscript_kwargs)
        write_jobscript(js, batch_dir / f"{pname}.sh")
    prepare_remote(remote_root, [pn for pn, _ in parts], runner)
    out: list[tuple[str, str]] = []
    for pname, _ in parts:
        push(Path(local_batch_root) / pname, f"{remote_root.rstrip('/')}/specs")
        jid = qsub(f"{remote_root.rstrip('/')}/specs/{pname}/{pname}.sh", runner)
        out.append((jid, pname))
    return out


def _harvest_qacct(
    job_id: str | None, runner: Callable[[list[str]], str], diag_path: Path
) -> str:
    """Append ``qacct -j`` output for a drained job to the diagnostics file (best-effort);
    return the raw text ("" when unavailable) so the drain can weigh the ATTEMPT EVIDENCE (P13)."""
    if job_id is None:
        return ""
    try:
        text = runner(["qacct", "-j", job_id])
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== job {job_id} =====\n{text}\n")
        return text
    except Exception as exc:  # noqa: BLE001 — forensics must never kill the loop
        _LOG.warning("qacct harvest failed for job %s: %s", job_id, exc)
        return ""


def _attempted_run_ids(
    qacct_text: str, batch_root: Path, round_name: str
) -> set[str] | None:
    """P13 (2026-07-13 pre-spend audit): which run_ids did the scheduler ACTUALLY dispatch?

    * ``None`` — no accounting trace at all (the deleted-pending class: an admin purge / Eqw
      qdel before dispatch leaves NO qacct rows). Nothing was attempted, so retry counts must
      NOT be bumped — under the old bump-everything drain, 2 purge events permanently abandoned
      specs that never ran once.
    * empty set — rows exist but cannot be attributed per-task (no ``taskid``, or the round's
      task files are unreadable): the caller degrades to the legacy bump-all.
    * non-empty set — exactly the run_ids in the tasks qacct says were dispatched.
    """
    from src.cluster.ledger import parse_qacct

    rows = [r for r in parse_qacct(qacct_text) if r]
    # P17/A2: Myriad REUSES job numbers, so `qacct -j <id>` can return a FOREIGN job's blocks —
    # whose taskids would then be attributed to OUR round. Keep only rows whose jobname matches
    # this round (rows without a jobname field are kept: a truncated qacct block is still ours
    # far more often than not, and mis-keeping only degrades toward the legacy bump-all).
    rows = [r for r in rows if str(r.get("jobname", round_name)).strip() == round_name]
    if not rows:
        return None
    ids: set[str] = set()
    for r in rows:
        t = str(r.get("taskid", "")).split()
        if not t or not t[0].isdigit():
            return set()
        try:
            payload = json.loads(
                (batch_root / round_name / f"task_{t[0]}.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unattributable evidence -> legacy bump-all
            return set()
        for s in (payload if isinstance(payload, list) else [payload]):
            rid = s.get("run_id") or s.get("candidate_id") if isinstance(s, dict) else None
            if not rid:
                return set()
            ids.add(str(rid))
    return ids


def _lock_owner_is_live_driver(pid: int, recorded_create_time: float | None) -> bool:
    """Is `pid` still the SAME process that wrote this lock? (D20)

    `psutil.pid_exists` answers a WEAKER question — *does anything hold that pid* — and the OS
    recycles pids. That is not hypothetical: on 2026-07-31 Windows recycled a dead driver's pid
    onto `OpenConsole.exe` and stranded the `h3` line, and on 2026-08-01 it recycled pid 34216
    onto `backgroundTaskHost.exe` and stranded `leg4` — a line ALREADY AT C4 — for ~14 h, dying
    12 s into every relaunch while every guard stayed green.

    The identity test is the process CREATE-TIME, recorded beside the pid at write. A recycled
    pid necessarily has a LATER create-time than the one recorded, so the mismatch is decisive.

    **Asymmetric on purpose.** Two errors are possible and they are not equally bad:
      * calling a live owner STALE breaks its lock -> two drivers on one batch -> double requeue
        rounds and corrupted retry accounting. Unrecoverable damage to the run.
      * calling a dead owner LIVE stalls the line until someone intervenes. Recoverable, and the
        monitoring cycle now auto-reaps exactly this case.
    So every ambiguity resolves to "owned". Only a definite verdict breaks a lock.
    """
    import psutil  # never os.kill(pid, 0): on Windows that TERMINATES the process

    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True                      # cannot inspect -> assume owned
    if recorded_create_time is not None:
        try:
            return abs(proc.create_time() - float(recorded_create_time)) < 1.0
        except psutil.NoSuchProcess:
            return False
        except Exception:                # noqa: BLE001 — cannot decide -> assume owned
            return True
    # LEGACY lock written before this fix (no create_time). Fall back to identity-by-command-line,
    # which is still strictly stronger than pid existence. An unreadable or EMPTY cmdline must be
    # read as "owned", never as "not a driver" — that is the direction that destroys a run.
    try:
        cmd = " ".join(proc.cmdline())
    except Exception:                    # noqa: BLE001
        return True
    return True if not cmd else ("run_campaign_cluster" in cmd)


def _acquire_driver_lock(lock_path: Path) -> None:
    """P12 (2026-07-13 pre-spend audit): ONE driver per batch name. The queue-adoption guard
    dedupes by job NAME, so a second concurrent driver of the same batch would silently ADOPT
    the first's arrays and then race it on every drain (double requeue rounds, duplicated
    permanent-ledger rows, interleaved pulls on the shared mirror). O_EXCL lockfile carrying
    the owner's pid AND process create-time (D20 — pid alone is not an identity); a lock whose
    owner is dead OR whose pid has been RECYCLED is broken automatically, so crash-resume stays
    one-command — no manual lock cleanup after a kill/BSOD."""
    import os

    import psutil

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _my_create_time: float | None = psutil.Process(os.getpid()).create_time()
    except Exception:  # noqa: BLE001 — degrade to the legacy lock shape rather than fail to lock
        _my_create_time = None
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "create_time": _my_create_time,
                           "ts": time.time()}, fh)
            return
        except FileExistsError:
            create_time: float | None = None
            try:
                _payload = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(_payload.get("pid", -1))
                _ct = _payload.get("create_time")
                create_time = float(_ct) if _ct is not None else None
            except Exception:  # noqa: BLE001 — torn lock = stale lock
                pid = -1

            if pid > 0 and pid != os.getpid() and _lock_owner_is_live_driver(pid, create_time):
                raise RuntimeError(
                    f"another driver (pid {pid}) is already running batch "
                    f"{lock_path.name!r} — refusing to double-drive (double requeues would "
                    f"corrupt the retry accounting). If that pid is NOT a driver, delete "
                    f"{lock_path} and relaunch."
                )
            lock_path.unlink(missing_ok=True)  # stale: dead owner, OR the pid was RECYCLED
    raise RuntimeError(f"could not acquire the driver lock {lock_path} after breaking a stale one")


def run_batch(
    flat_specs: list[dict[str, Any]],
    base_name: str,
    *,
    local_batch_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Public entry: :func:`_run_batch_unlocked` under the per-batch driver lock (P12)."""
    lock_path = Path(local_batch_root) / f"{sanitize_name(base_name)}.driver.lock"
    _acquire_driver_lock(lock_path)
    try:
        return _run_batch_unlocked(flat_specs, base_name,
                                   local_batch_root=local_batch_root, **kwargs)
    finally:
        lock_path.unlink(missing_ok=True)


def _run_batch_unlocked(
    flat_specs: list[dict[str, Any]],
    base_name: str,
    *,
    local_batch_root: str | Path,
    local_archive_root: str | Path,
    remote_root: str,
    remote_outputs_root: str,
    gold_dir: str,
    host: str = "myriad",
    runner: Callable[[list[str]], str] | None = None,
    push: Callable[..., None] = push_batch,
    pull: Callable[[], int] | None = None,
    pack: int = 1,
    chunk_tasks: int | None = None,
    specs_per_task: int | None = None,
    poll_secs: float = 600.0,
    max_wall_secs: float | None = None,
    max_consecutive_errors: int = 72,
    max_transport_outage_secs: float = 43200.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
    **jobscript_kwargs: Any,
) -> dict[str, Any]:
    """Drive one batch of specs to completion; return the summary dict.

    Resume-safe + crash-safe by construction: the FIRST action of every cycle is a pull + the
    compacted diff, so calling ``run_batch`` again after any interruption continues exactly
    where the archive says — including adopting a still-queued job from a previous driver
    invocation (the double-submit guard matches this batch's jobnames in the queue). Requeue
    naming is ``base``, ``base_r1``, ``base_r2``, …; name reuse across driver restarts is
    harmless because specs are content-addressed and completed run_ids are never re-emitted.

    Summary: ``{"ok", "completed", "total", "rounds", "exhausted", "job_ids"}`` where ``ok`` is
    True iff every spec's run_id is archived; ``exhausted`` lists run_ids abandoned after
    :data:`~src.cluster.ledger.MAX_RETRIES` requeues (each also a permanent-ledger row).

    LONG-OUTAGE TOLERANCE (resume from ANY stoppage). Transport (VPN/ssh) is allowed to blip: a
    consecutive-failure streak is fatal only once it persists past BOTH bounds' first trip —
    ``max_consecutive_errors`` cycles OR ``max_transport_outage_secs`` wall-seconds (default 12 h).
    The time bound is the real guard (decoupled from ``poll_secs``): the driver PATIENTLY rides out a
    multi-hour VPN/network/Myriad-login outage instead of dying every 2 h, because the archive is the
    truth and nothing is lost by waiting — the queued Myriad arrays keep training regardless of the
    laptop driver's connectivity. Past the bound it raises cleanly (the supervisor relaunches with
    ``--resume`` when connectivity returns); a genuinely permanent break is thus surfaced, not hidden.

    HEARTBEAT (observability). When ``heartbeat`` is given it is called once per cycle with a
    read-only status snapshot ``{ts, done, pending, exhausted, rounds, queue_names, pull_failures,
    ops_failures, phase}`` — the driver-liveness + queue lease the sentinel consumes (it never opens
    an ssh session of its own). A heartbeat failure is swallowed: observability must never break the run.
    """
    runner = runner or ssh_runner(host)
    if pull is None:
        def pull() -> int:
            # P14: heartbeat per pulled CHUNK — a big pull is many pipes (each up to an hour),
            # during which a silent driver looks hung to the sentinel.
            def _pull_beat(i: int, n: int) -> None:
                if heartbeat is not None:
                    try:
                        heartbeat({"base_name": base_name, "done": None, "pending": None,
                                   "exhausted": None, "rounds": None, "queue_names": [],
                                   "pull_failures": 0, "ops_failures": 0,
                                   "phase": f"pulling {i}/{n}"})
                    except Exception:  # noqa: BLE001 — observability never breaks the pull
                        pass

            return pull_archive(
                remote_outputs_root, local_archive_root, host=host, runner=runner,
                progress=_pull_beat,
            )
    base_name = sanitize_name(base_name)
    ledger_path = Path(local_batch_root) / f"{base_name}.permanent.jsonl"
    diag_path = Path(local_batch_root) / f"{base_name}.qacct.txt"

    # Resume-hardening (long-run): consult THIS batch's permanent ledger on start. Without it, a
    # RESTART re-derives pending purely from the archive, so a DETERMINISTICALLY-failing spec (already
    # requeue-exhausted + permanently ledgered on a prior run, hence no record) would be re-emitted and
    # re-tried 2× on EVERY driver restart — an unbounded waste over a days-to-weeks run. Seeding
    # ``exhausted`` from the ledger makes the abandonment durable across restarts.
    # P17/A2 (2026-07-13 audit): intersect with THIS batch's run_ids — the ledger may carry rows
    # from a differently-shaped prior invocation (roster change), and foreign ids would deflate
    # the done count and fail `ok` for specs this batch never asked for.
    batch_ids = {spec_run_id(s) for s in flat_specs}
    prior_exhausted = _ledgered_run_ids(ledger_path) & batch_ids
    live: list[dict[str, Any]] = [
        dict(s) for s in flat_specs if spec_run_id(s) not in prior_exhausted
    ]
    exhausted: list[str] = list(prior_exhausted)
    job_ids: list[str] = []
    last_parts: list[tuple[str, str]] = []  # (job_id, batch_dir_name) of the LAST submitted round
    rounds = 0
    alive_seen = False  # a job of ours has been observed live (or was submitted) this run
    pending_submit: list[dict[str, Any]] | None = None  # set = a submission is owed
    pull_failures = 0
    ops_failures = 0
    first_pull_fail_at: float | None = None  # monotonic start of the current pull-failure streak
    first_ops_fail_at: float | None = None   # …and the queue-op streak (both for the time-outage guard)
    eqw_since: float | None = None           # P1: start of the current Eqw dwell (None = no Eqw seen)
    evidenceless_drains = 0                  # P13: consecutive drains with NO qacct trace (purges)
    last_queue_names: list[str] = []         # last-observed queue members of this batch (for the heartbeat)
    t0 = clock()

    def _outage_is_fatal(n_failures: int, streak_start: float | None) -> bool:
        """Transport is fatal only once a streak trips EITHER bound: the count cap OR the wall-time
        outage bound (the real guard — decoupled from ``poll_secs`` so a long VPN/ssh outage is ridden
        out, not fatal every ``max_consecutive_errors`` cycles)."""
        if n_failures >= max_consecutive_errors:
            return True
        return streak_start is not None and (clock() - streak_start) >= max_transport_outage_secs

    while True:
        # 1) refresh the local mirror (tolerate transient VPN/ssh failures, bounded)
        try:
            pull()
            pull_failures = 0
            first_pull_fail_at = None
        except _TRANSPORT_ERRORS as exc:  # transport is allowed to blip, not to die (P14:
            # anything OUTSIDE this whitelist — KeyError/TypeError/ValueError/MemoryError — is a
            # LOCAL bug or corruption; retrying it for the 12 h outage budget only delays the
            # crash and mislabels it "VPN/ssh down". Those now propagate immediately.)
            pull_failures += 1
            if first_pull_fail_at is None:
                first_pull_fail_at = clock()
            _LOG.warning("[%s] pull failed (%d consecutive, %.0f min down): %s", base_name,
                         pull_failures, (clock() - first_pull_fail_at) / 60.0, exc)
            if _outage_is_fatal(pull_failures, first_pull_fail_at):
                raise RuntimeError(
                    f"{pull_failures} consecutive pull failures over "
                    f"{(clock() - first_pull_fail_at) / 3600.0:.1f} h — VPN/ssh down too long; "
                    f"the batch is safe to resume with run_batch() once connectivity returns"
                ) from exc
            # 2026-07-13 audit fix (HIGH): the old code FELL THROUGH to the archive diff and the
            # submit/requeue step on a STALE mirror — a drained array whose records had not been
            # pulled yet would be REQUEUED (double-training completed work, burning retry budget,
            # overwriting remote records). "Nothing is lost by waiting" must mean exactly that:
            # never act on the queue without a fresh successful pull. Beat, sleep, retry.
            if heartbeat is not None:
                try:
                    heartbeat({"base_name": base_name, "done": None, "pending": None,
                               "exhausted": len(exhausted), "rounds": rounds,
                               "queue_names": list(last_queue_names),
                               "pull_failures": pull_failures, "ops_failures": ops_failures,
                               "phase": "pull_outage"})
                except Exception:  # noqa: BLE001
                    _LOG.debug("[%s] heartbeat emit failed (ignored)", base_name)
            sleep(poll_secs)
            continue

        # 2) the archive-truth diff (over the non-exhausted specs)
        pending = pending_specs(live, local_archive_root)
        # P9 (2026-07-13 audit): a PERMANENT node-side reject marker (deterministic sandbox/
        # contract reject — run_one._write_reject_marker) is abandoned IMMEDIATELY: same-source
        # retry is provably futile, so burning MAX_RETRIES requeue rounds (queue wait + h_rt
        # each) on it wasted 2 rounds per reject. Ledgered in the same permanent-JSONL shape as
        # retries_exhausted, so resume + bank-gate accounting see it identically.
        # 2026-07-28: this MUST be the sub-root-scoped form. `local_archive_root` is the ONE tree
        # all twelve supervised lines share, and reject markers are keyed on the bare candidate id
        # (`scalar-g1-c0`), which every line reuses — so the old mirror-wide
        # `permanent_reject_ids(local_archive_root)` let ONE line's marker permanently abandon
        # every OTHER line's identically-named candidate without ever submitting it. Measured
        # live: 439 of 498 abandonments spurious, including 36 of 36 on the confirmatory core.
        # See `poll.permanently_rejected_specs`.
        dead_now = permanently_rejected_specs(pending, local_archive_root)
        if dead_now:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as fh:
                for s in dead_now:
                    fh.write(json.dumps({"task": None, "spec": s,
                                         "reason": "permanent_node_reject"},
                                        sort_keys=True, default=str) + "\n")
            dead_ids = {spec_run_id(s) for s in dead_now}
            _LOG.error("[%s] %d permanent node reject(s) abandoned without requeue: %s",
                       base_name, len(dead_ids), sorted(dead_ids))
            exhausted.extend(sorted(dead_ids))
            live = [s for s in live if spec_run_id(s) not in dead_ids]
            pending = [s for s in pending if spec_run_id(s) not in dead_ids]
        done = len(flat_specs) - len(pending) - len(exhausted)
        if not pending:
            summary = {
                "ok": not exhausted, "completed": done, "total": len(flat_specs),
                "rounds": rounds, "exhausted": list(exhausted), "job_ids": list(job_ids),
            }
            if heartbeat is not None:
                try:  # a FINAL phase="done" beat so the sentinel reads this batch as finished, not hung
                    heartbeat({"base_name": base_name, "done": int(done), "pending": 0,
                               "exhausted": len(exhausted), "rounds": rounds, "queue_names": [],
                               "pull_failures": 0, "ops_failures": 0, "phase": "done"})
                except Exception:  # noqa: BLE001 — observability must never break completion
                    pass
            _LOG.info("[%s] batch complete: %s", base_name, summary)
            return summary

        # 3) act on queue state (transient scheduler/ssh failures are bounded, not fatal)
        try:
            alive_names, queue_states = batch_jobs_in_queue(base_name, runner)
            last_queue_names = sorted(alive_names)
            # P1 (2026-07-13 audit): an Eqw array never dispatches (h_rt never starts) — without
            # this it waits FOREVER with green heartbeats. Bounded dwell, then harvest + qdel +
            # treat as a drain (the bounded requeue takes over; a deterministic Eqw cause exhausts
            # after MAX_RETRIES and is loudly ledgered).
            eqw_names = sorted(n for n, st in queue_states.items() if "E" in st)
            if eqw_names:
                if eqw_since is None:
                    eqw_since = clock()
                    _LOG.warning("[%s] Eqw detected: %s — dwell timer started", base_name, eqw_names)
                elif clock() - eqw_since > 900.0:  # 15 min: Eqw never self-heals
                    _LOG.error("[%s] Eqw persisted >15 min on %s — harvesting + deleting",
                               base_name, eqw_names)
                    _harvest_qacct(job_ids[-1] if job_ids else None, runner, diag_path)
                    for nm in eqw_names:
                        try:
                            runner(["qdel", nm])
                        except Exception as exc:  # noqa: BLE001 — best-effort; drain handles it
                            _LOG.warning("[%s] qdel %s failed: %s", base_name, nm, exc)
                    eqw_since = None
                    alive_names = alive_names - set(eqw_names)
            else:
                eqw_since = None
            if alive_names:
                alive_seen = True
            else:
                if alive_seen and pending_submit is None:
                    # DRAIN TRANSITION: a round finished with work remaining → bounded requeue.
                    # Chunked rounds: harvest EACH part's job and attribute via that part's own
                    # task files; tri-state union — all-parts-no-rows = the purge class (None);
                    # any rows but nothing attributable = legacy bump-all (empty set); else the
                    # union of attributed run_ids (unmatched specs ride unbumped this round).
                    last_round = base_name if rounds <= 1 else f"{base_name}_r{rounds - 1}"
                    parts = last_parts or ([(job_ids[-1], last_round)] if job_ids else [])
                    attempted: set[str] | None = None
                    any_rows = False
                    for _jid, _dname in parts:
                        qtext = _harvest_qacct(_jid, runner, diag_path)
                        part_att = _attempted_run_ids(qtext, Path(local_batch_root), _dname)
                        if part_att is not None:
                            any_rows = True
                            if part_att:
                                attempted = (attempted or set()) | part_att
                    if any_rows and not attempted:
                        attempted = set()  # rows exist but unattributable -> legacy bump-all
                    # P13: retry bumps require ATTEMPT EVIDENCE. A drain with NO qacct trace at
                    # all is the deleted-pending class (admin purge / qdel before dispatch) —
                    # nothing ran, so bumping would let 2 purge events permanently abandon
                    # never-attempted specs. Requeue UNBUMPED, bounded to 3 consecutive
                    # evidence-less drains (a systemic no-trace crash loop must still exhaust
                    # loudly rather than spin forever).
                    if attempted is None and evidenceless_drains < 3:
                        evidenceless_drains += 1
                        _LOG.warning(
                            "[%s] drain with NO qacct trace (%d/3) — the array was purged "
                            "before dispatch; requeueing %d spec(s) WITHOUT a retry bump",
                            base_name, evidenceless_drains, len(pending))
                        alive_seen = False
                        pending_submit = pending
                    else:
                        evidenceless_drains = 0
                        if attempted:  # per-task evidence: bump ONLY the dispatched specs
                            by_no = {i: s for i, s in enumerate(pending)
                                     if spec_run_id(s) in attempted}
                            unbumped = [s for s in pending
                                        if spec_run_id(s) not in attempted]
                        else:  # unattributable evidence (or the 4th no-trace drain): legacy bump-all
                            by_no = dict(enumerate(pending))
                            unbumped = []
                        retryable = requeue_specs(list(by_no), by_no, ledger_path) + unbumped
                        retry_ids = {spec_run_id(s) for s in retryable}
                        newly_dead = [
                            spec_run_id(s) for s in pending if spec_run_id(s) not in retry_ids
                        ]
                        exhausted.extend(newly_dead)
                        dead = set(exhausted)
                        bumped = {spec_run_id(s): s for s in retryable}
                        live = [
                            bumped.get(spec_run_id(s), s)
                            for s in live
                            if spec_run_id(s) not in dead
                        ]
                        alive_seen = False
                        if not retryable:
                            continue  # only exhausted specs remained → next cycle returns
                        pending_submit = retryable
                elif pending_submit is None:
                    # FIRST submission (or a restart after a drain) — no retry bump.
                    pending_submit = pending
                # Re-filter at the submit moment: anything that completed between the drain
                # and a (retried) submit must not be re-trained (run_id idempotency is about
                # never re-emitting DONE work, not trusting a stale snapshot).
                pending_submit = pending_specs(pending_submit, local_archive_root)
                if not pending_submit:
                    pending_submit = None
                    continue
                round_name = base_name if rounds == 0 else f"{base_name}_r{rounds}"
                submitted = submit_batch(
                    pending_submit, round_name,
                    local_batch_root=local_batch_root, remote_root=remote_root,
                    gold_dir=gold_dir, runner=runner, push=push, pack=pack,
                    chunk_tasks=chunk_tasks, specs_per_task=specs_per_task, **jobscript_kwargs,
                )
                job_ids.extend(jid for jid, _ in submitted)
                last_parts = list(submitted)
                rounds += 1
                pending_submit = None  # consumed ONLY on success → a failed submit re-tries
                alive_seen = True
                _LOG.info("[%s] submitted %s as %d array(s): %s", base_name, round_name,
                          len(submitted), [jid for jid, _ in submitted])
            ops_failures = 0
            first_ops_fail_at = None
        except _TRANSPORT_ERRORS as exc:  # qstat/qsub/push blip; retry next cycle (P14: local
            # bugs — KeyError/TypeError/ValueError — propagate immediately, see the pull clause)
            ops_failures += 1
            if first_ops_fail_at is None:
                first_ops_fail_at = clock()
            _LOG.warning("[%s] queue op failed (%d consecutive, %.0f min): %s", base_name,
                         ops_failures, (clock() - first_ops_fail_at) / 60.0, exc)
            if _outage_is_fatal(ops_failures, first_ops_fail_at):
                raise RuntimeError(
                    f"{ops_failures} consecutive queue-operation failures on {base_name} over "
                    f"{(clock() - first_ops_fail_at) / 3600.0:.1f} h"
                ) from exc

        # 4) wall guard + heartbeat + sleep
        if max_wall_secs is not None and clock() - t0 > max_wall_secs:
            raise RuntimeError(
                f"[{base_name}] max_wall_secs={max_wall_secs} exceeded with "
                f"{len(pending)} specs pending — resume later with run_batch()"
            )
        if heartbeat is not None:
            try:
                heartbeat({
                    "base_name": base_name, "done": int(done), "pending": len(pending),
                    "exhausted": len(exhausted), "rounds": rounds,
                    "queue_names": list(last_queue_names),
                    "pull_failures": pull_failures, "ops_failures": ops_failures,
                    "phase": "running",
                })
            except Exception:  # noqa: BLE001 — observability must never break the run
                _LOG.debug("[%s] heartbeat emit failed (ignored)", base_name)
        _LOG.info(
            "[%s] %d/%d done, %d pending, round %d", base_name, done, len(flat_specs),
            len(pending), rounds,
        )
        sleep(poll_secs)
