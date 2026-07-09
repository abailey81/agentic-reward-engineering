"""Submission layer: push batches up (tar-over-ssh), qsub arrays, marker jobs for hold-chains (§12.4/§14.3).

Every function takes a ``runner`` callable (``runner(cmd: list[str]) -> str`` returning stdout)
so tests inject a fake and NOTHING here touches the network at import or test time. Production
passes :func:`ssh_runner` (the ``myriad`` Host alias — key auth, keepalives from ~/.ssh/config;
credentials never appear here, R10).

Researched rules encoded: one ARRAY = one queue entry (R3: never loop qsub); submit-rate safety
via a single qsub per batch; ``-hold_jid`` released by 5-minute MARKER jobs so dependent arrays
are pre-submitted with ZERO driver latency (§14.3); job ids parsed from Grid Engine's canonical
"Your job[-array] <id>" line.
"""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable

_LOG = logging.getLogger(__name__)

__all__ = [
    "ssh_runner", "ssh_base", "push_batch", "qsub", "submit_marker", "parse_job_id",
    "sanitize_name", "prepare_remote", "remote_home", "expand_remote", "reap",
]

Runner = Callable[[list[str]], str]

_JOB_ID_RE = re.compile(r"Your job(?:-array)? (\d+)")

# Driver-scoped ssh hardening (2026-07-08). Scoped to the driver's OWN calls — interactive
# ``ssh myriad`` (incl. the first-login key install that may need his UCL password) is UNTOUCHED.
#  * BatchMode=yes            — never block on a password/passphrase PROMPT: the unattended driver
#                               uses KEY auth, so a missing/broken key must fail FAST, not hang a
#                               multi-day run forever waiting on stdin no one will type into.
#  * StrictHostKeyChecking=accept-new — auto-trust the host key on FIRST connect (no yes/no prompt
#                               to wedge the driver) while still REFUSING a CHANGED key (MITM guard).
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]

#: Wall-clock bound on ONE driver ssh call. Lowered 300 -> 120 on 2026-07-28 after measuring where
#: the 300 s "timeouts" actually go.
#:
#: MEASURED, on the live RUN 2 driver: over 55 samples in three minutes, only 8 distinct ssh
#: children existed and **not one ever aged past 10 s** — while the driver logged a 300 s timeout
#: roughly every five minutes, on ops as trivial as `mkdir -p`. The remote side is not the cause
#: either (`qstat -r` 1.2 s, `find` over the whole outputs tree 0.046 s, login-node load 3.4), nor
#: is the client (sustained 8-concurrent A/B of Windows OpenSSH 9.5p2 vs Git's 10.2p1: 80/80 ok on
#: both, worst case 6.0 s). So the wait is happening in the PARENT — `subprocess.run`'s pipe
#: reader never observing EOF — and the log message misattributes it to the remote command.
#:
#: The exact parent-side mechanism is NOT yet identified, so this is a BOUND, not a cure: it caps
#: the cost of each event at 120 s instead of 300 s and returns the batch thread to work 2.5x
#: sooner. 120 s is ~20x the measured worst-case real latency, so it cannot truncate a legitimate
#: call. The pull path is unaffected (it uses its own Popen with a 3600 s budget for a bulk tar).
_RUNNER_TIMEOUT_SECS = 120.0


def reap(proc: subprocess.Popen, *, grace: float) -> None:
    """Wait ``grace`` seconds for a clean exit, then KILL — the child cannot outlive this call.

    2026-07-28 LEAK FIX, found live at T+11 h of the confirmatory run. Both tar-over-ssh pipes in
    this package (:func:`push_batch` here, ``poll._default_fetch`` on the pull side) placed their
    ``proc.wait()`` AFTER the ``try/finally``. Any exception on the consuming side therefore
    skipped the wait entirely and left the child running forever — and a stalled pull raises
    ``TimeoutExpired`` by construction, so the leak fired on exactly the path that mattered.

    This is not tidiness. Each leaked ``ssh`` holds an established session on the SHARED UCL login
    node plus a remote ``tar``, and login-node session pressure is what makes the NEXT pull stall:
    a positive feedback loop. Measured on the live campaign at T+11 h — 13 leaked children, 8 of
    them pulls still running 1.1-6.7 h past their own 3600 s timeout, against a driver transport
    failure rate climbing monotonically 5.2 % -> 55.3 % over ten hours while successful poll cycles
    fell from 1,446/h to 224/h.

    ``grace`` is deliberately asymmetric at the call sites: generous on the success path, where the
    peer has already finished writing and its true exit status is still wanted, and short on the
    failure path, where waiting is the very cost being eliminated.
    """
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover — kill() not honoured is unreachable
            pass


def ssh_base(host: str = "myriad") -> list[str]:
    """The hardened ``ssh`` argv prefix for a DRIVER call to ``host`` (see :data:`_SSH_OPTS`)."""
    return ["ssh", *_SSH_OPTS, host]


def ssh_runner(host: str = "myriad") -> Runner:
    """The production runner: ``cmd`` executed via ssh on ``host`` (checked, stdout returned).

    V10 audit fix: ssh joins its argument vector with spaces and the REMOTE shell re-splits the
    result — so an unquoted ``["bash", "-c", script]`` would deliver only ``printf`` to ``-c``
    and scatter the rest. Every word is therefore ``shlex.quote``d (POSIX quoting — correct for
    the remote Linux side regardless of the local OS) so the remote shell sees exactly the argv
    we intended. Hardened with :func:`ssh_base` so an unattended driver never hangs on a prompt.
    """

    def _run(cmd: list[str]) -> str:
        remote = " ".join(shlex.quote(c) for c in cmd)
        argv = [*ssh_base(host), remote]
        # encoding is PINNED to utf-8 (not the OS locale): the cluster emits utf-8, but a non-utf-8
        # Windows console (e.g. cp1251 on a Russian-locale laptop) would otherwise crash the reader
        # thread on any non-ASCII byte. errors="replace" keeps a stray byte from ever killing a pull.
        #
        # stdin=DEVNULL (2026-07-28): `ssh` READS stdin to forward it to the remote command unless
        # told otherwise, and `capture_output=True` leaves stdin INHERITED from the driver — whose
        # own stdin is a pipe from the supervisor's `| Out-File`. An A/B at fan-out 40 showed no
        # measurable difference, so this is not the cure for the stall below; it is simply the
        # correct way to run ssh unattended, and it removes a whole class of hazard for free.
        t0 = time.monotonic()
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            out, err = proc.communicate(timeout=_RUNNER_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            # ── THE DIAGNOSTIC THAT LOCALISES THE STALL ────────────────────────────────────────
            # Seven hypotheses for the phantom timeout have been tested and refuted (archive size,
            # qmaster, MaxStartups, the ssh client, GIL starvation, inherited stdin, pipe-handle
            # inheritance). Rather than guess an eighth, record the ONE fact that settles it: had
            # the child ALREADY EXITED when the timeout fired? `poll()` is checked BEFORE the kill,
            # so a non-None returncode proves the wait was on the PIPE, not on the command — which
            # no amount of remote-side or cluster-side investigation could ever show.
            rc = proc.poll()
            elapsed = time.monotonic() - t0
            _LOG.warning(
                "ssh_timeout_diagnostic cmd=%r elapsed=%.1fs child_already_exited=%s "
                "child_returncode=%r — if child_already_exited is True the wall-clock was spent in "
                "the PARENT waiting on the pipe, not on the remote command",
                cmd[:2], elapsed, rc is not None, rc,
            )
            proc.kill()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover — kill() not honoured
                pass
            raise
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, argv, output=out, stderr=err)
        return out
    return _run


def remote_home(runner: Runner) -> str:
    """Resolve ``$HOME`` on the cluster (ONE call; for expanding user-supplied ``~`` paths).

    2026-07-11 incident fix: every layer between the driver and the node keeps ``~`` LITERAL —
    :func:`ssh_runner` shlex-quotes each argv word, SGE ``#$`` directives never expand it, and
    double-quoted bash strings keep it verbatim — so the rehearsal's ``~/Scratch/...`` root sent
    its arrays to Eqw (admin-purged, no qacct trace) and its spec push into a literal ``~``
    directory under ``$HOME``. All user-facing ``~`` paths are therefore expanded ONCE, up front,
    against the real remote home via this helper + :func:`expand_remote`.
    """
    # The runner quotes each word, so $HOME must be evaluated by an explicit remote shell.
    # `-c`, NOT `-lc` (deep review 2026-07-26, #63): a LOGIN shell sources the profile files, and on
    # a shared HPC those routinely echo module-load / notice lines to STDOUT. `$HOME` is set by
    # sshd/login before any profile runs, so a non-login shell resolves it just as well and removes
    # the noise source entirely.
    home = runner(["sh", "-c", 'printf %s "$HOME"']).strip()
    # Validate it is ONE plausible absolute path. The previous check was `startswith("/")` alone,
    # which is asymmetric and fails OPEN: `.strip()` clears the ends, so banner text BEFORE the path
    # was correctly refused, but banner text AFTER it left "/" at position 0 and was ACCEPTED —
    # REPRODUCED, yielding home='/home/ucestes\nWelcome to Myriad!' and an expanded root of
    # '/home/ucestes\nWelcome to Myriad!/Scratch/run'. That garbage root goes straight into the
    # jobscript's `#$ -wd` directive, which is precisely the 2026-07-11 incident this helper exists
    # to prevent: an invalid -wd puts the whole array in Eqw at dispatch, where UCL's cleanup
    # deletes it with NO qacct record. Fail LOUD instead — a submission that cannot resolve its own
    # root must never proceed.
    if not home.startswith("/") or len(home.splitlines()) != 1 or any(c.isspace() for c in home):
        raise RuntimeError(
            f"could not resolve a single clean remote $HOME (got {home!r}). If the login profile "
            "prints to stdout, silence it or fix the account's shell startup — the resolved home "
            "becomes the jobscript's -wd, and an invalid -wd is dispatch-time Eqw with no trace."
        )
    return home


def expand_remote(path: str, home: str) -> str:
    """Expand a leading ``~``/``~/`` in a REMOTE path against the resolved remote ``home``."""
    if path == "~":
        return home
    if path.startswith("~/"):
        return home + path[1:]
    if path.startswith("~"):
        raise ValueError(f"'~user' remote paths are unsupported: {path!r}")
    return path


def parse_job_id(qsub_stdout: str) -> str:
    """Extract the numeric job id from qsub output; fail loud on anything unexpected."""
    m = _JOB_ID_RE.search(qsub_stdout)
    if not m:
        raise RuntimeError(f"could not parse a job id from qsub output: {qsub_stdout!r}")
    return m.group(1)


def push_batch(batch_dir: str | Path, remote_specs_root: str, *, host: str = "myriad") -> None:
    """Push one local batch directory to ``<host>:<remote_specs_root>/<batch_name>/`` (tar-over-ssh).

    V9 audit fix: the Windows driver host has NO rsync (verified 2026-07-06 — only scp/ssh live
    in PATH), so the batch streams as a tar pipe: local ``tar -cf -`` | ssh | remote ``tar -xf -``.
    Idempotent — re-pushing overwrites byte-identical content-addressed files (spec_io writes
    sorted-keys JSON). A torn push cannot start a wrong training: ``read_spec`` on the node
    fail-CLOSES on a missing index / sha mismatch (V6), so partial arrival is caught at the
    moment of use, not silently trained.
    """
    src = Path(batch_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"batch dir {src} does not exist")
    name = sanitize_name(src.name)
    root = remote_specs_root.rstrip("/")
    # The local tar runs with cwd=parent (no -C: Windows bsdtar/GNU-tar path-flavor quirks); the
    # REMOTE side is POSIX-quoted for the cluster's shell. check=True on ssh catches a dropped
    # connection; tar.wait() catches a local read failure.
    remote_cmd = "mkdir -p " + shlex.quote(root) + " && tar -xf - -C " + shlex.quote(root)
    tar = subprocess.Popen(["tar", "-cf", "-", name], stdout=subprocess.PIPE, cwd=str(src.parent))
    drained = False
    try:
        subprocess.run([*ssh_base(host), remote_cmd], stdin=tar.stdout, check=True, timeout=1800)
        drained = True
    finally:
        # Same 2026-07-28 leak as the pull side: `tar.wait()` used to sit AFTER the try/finally,
        # so a failed ssh skipped it and left the local tar unreaped. Milder here (the child is
        # local and usually dies of SIGPIPE) but identical in kind — see `poll.reap`.
        if tar.stdout is not None:
            tar.stdout.close()
        reap(tar, grace=300.0 if drained else 10.0)
    if tar.returncode != 0:
        raise subprocess.CalledProcessError(tar.returncode, f"tar -cf - {name} (local)")


def qsub(jobscript_remote_path: str, runner: Runner) -> str:
    """Submit ONE array jobscript (already on the cluster); return the job id."""
    return parse_job_id(runner(["qsub", jobscript_remote_path]))


def submit_marker(name: str, after_job_id: str, remote_root: str, runner: Runner) -> str:
    """A 5-minute 1-core job holding on ``after_job_id`` — the anchor dependent arrays hold on.

    Chain shape: search_array(id=A) -> marker(hold A, id=M) -> test_array(hold M). The test
    array is thereby PRE-SUBMITTED at launch and released by Grid Engine itself the second the
    arm's search completes (zero driver latency, §14.3).
    """
    name = sanitize_name(name)
    if not after_job_id.isdigit():
        raise ValueError(f"after_job_id must be a numeric SGE job id, got {after_job_id!r}")
    # V2 audit fix: build the script with ``printf '%s\n'`` (one argument per line). Plain
    # ``echo`` does NOT interpret ``\n`` in POSIX sh, so the old version collapsed every ``#$``
    # directive onto one dead comment line — the hold/h_rt would silently never apply.
    lines = [
        "#!/bin/bash -l",
        f"#$ -N {name}",
        "#$ -l h_rt=0:5:0",
        f"#$ -hold_jid {after_job_id}",
        f"#$ -wd {remote_root}",
        "true",
    ]
    # shlex.quote (not hand-rolled '...') so embedded quotes/spaces in remote_root can never
    # break the script; the redirect target is quoted for the same reason.
    quoted = " ".join(shlex.quote(ln) for ln in lines)
    target = shlex.quote(f"{remote_root}/{name}.sh")
    script = f"printf '%s\\n' {quoted} > {target} && qsub {target}"
    return parse_job_id(runner(["bash", "-c", script]))


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def sanitize_name(name: str) -> str:
    """Validate a job/batch name for SGE ``-N`` and shell interpolation (fail loud, not filter)."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid SGE job name {name!r} (must match {_NAME_RE.pattern})")
    return name


def prepare_remote(remote_root: str, batch_names: list[str], runner: Runner) -> None:
    """Create the remote dirs a batch needs BEFORE qsub (V4 audit fix).

    Grid Engine must be able to open the ``#$ -o`` log path at job START, so ``logs/<name>``
    cannot be left to the job body alone (the in-script ``mkdir`` is only a belt). Call once per
    batch, before :func:`qsub`.
    """
    root = remote_root.rstrip("/")
    # outputs/ is pre-created too (V9): the first pull's remote ``find`` must never race the
    # first job's mkdir — an existing-but-empty outputs root cleanly means "0 completed".
    dirs = [f"{root}/specs", f"{root}/ledger", f"{root}/outputs"]
    dirs += [f"{root}/logs/{sanitize_name(n)}" for n in batch_names]
    runner(["mkdir", "-p", *dirs])
