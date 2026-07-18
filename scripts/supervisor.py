#!/usr/bin/env python3
"""Auto-restart supervisor — keeps the ~2-week campaign alive unattended.

Runs the campaign as a child process — ``--resume`` is appended on EVERY launch, the first included
(M7: a supervisor restarted by the ONSTART task or the operator must never issue a fresh, re-billing
search) — and on a non-zero exit (crash / OOM / transient outage / power-loss recovery) relaunches it
(bounded retries + exponential backoff), logging each restart. A child that survived a healthy stretch
(> 30 min) resets the retry counter (M7: the budget stops crash LOOPS, not weeks of independent
transients). It stops on a clean exit (0), on the user's deliberate graceful shutdown, or once the
restart budget is spent. Because the engine's resume is idempotent and the search-replay cache replays
archived work, a restart never re-bills Opus or re-trains completed candidates — the supervisor only
pays for the lost in-flight unit. Pre-flight is run once up front (NO-GO aborts before any compute is
committed) at the MAX worker width found in the passthrough args (F13 — validate what will actually run).

Hard rails (review 2026-07-02): ``--no-resume`` in the passthrough args is REFUSED loudly (F4 —
20 auto-restarted fresh runs = full re-search re-billing + frozen-winner overwrite + sealed-leg
re-touch); child exit 2 (argparse/usage) is NON-restartable (F9 — the identical bad command line can
never succeed on relaunch); and ``--max-total-restarts`` is an ABSOLUTE lifetime cap that the healthy
reset never touches (F3 — a recurring fault inside one irreducible > 30-min unit would otherwise
restart forever, each cycle able to re-bill one LLM call).

The restart *decision* is a pure function (unit-tested in ``tests/test_supervisor.py``); the process loop is thin.

Run::

    python scripts/supervisor.py --gpu 2 -- --campaign-args here
    python scripts/supervisor.py --gpu 2 --max-restarts 20 --no-preflight
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# A clean, deliberate graceful-shutdown exit code the campaign uses when the operator stops it on purpose:
# the supervisor must NOT fight that (do not relaunch). 0 = success; this sentinel = "stopped on request".
GRACEFUL_SHUTDOWN_CODE = 0
_REPO = Path(__file__).resolve().parents[1]

# M7 (ops audit 2026-07-02): a child that ran HEALTHY for this long before dying is a fresh incident,
# not the next step of a crash loop. 30 min >> the longest startup path (preload + panel load + first
# candidate), so anything that survives it was genuinely making progress.
HEALTHY_RUNTIME_SECS = 1800.0


# ── Pure decision logic (unit-tested) ─────────────────────────────────────────────────────────────────
def should_restart(exit_code: int, attempt: int, max_restarts: int, *, graceful: bool = False) -> bool:
    """Relaunch iff the child crashed (non-zero), the operator did not stop it on purpose, and the restart
    budget is not spent. ``attempt`` is the number of restarts ALREADY performed (0 on the first crash)."""
    if graceful:
        return False
    if exit_code == 0:
        return False
    return attempt < max_restarts


def backoff_seconds(attempt: int, *, base: float = 30.0, cap: float = 600.0) -> float:
    """Exponential backoff (30 s, 60, 120, ... capped at 10 min) so a sustained outage is not hammered."""
    return float(min(cap, base * (2 ** max(0, attempt))))


def reset_attempt_after_healthy_run(
    attempt: int, runtime_secs: float, *, healthy_secs: float = HEALTHY_RUNTIME_SECS
) -> int:
    """M7(i): reset the restart counter after a HEALTHY stretch (child survived > ``healthy_secs``).

    Without the reset, a ~2-3-week campaign that hits one transient crash per day exhausts the
    20-restart budget mid-run and the supervisor gives up on a perfectly recoverable job — the
    budget is meant to stop CRASH LOOPS (die-on-startup), not to meter independent incidents spread
    over weeks. A run that survived the healthy window is progress, so its crash starts a fresh
    incident (attempt -> 0, and thus a fresh 30 s backoff); a fast death keeps counting up.
    """
    return 0 if runtime_secs > float(healthy_secs) else int(attempt)


def _passthrough_gpu_width(rest: list[str]) -> int | None:
    """Max ``--gpu`` / ``--search-gpu`` width in the passthrough campaign args (``None`` when absent).

    F13: pre-flight must validate the worker width the campaign ACTUALLY launches with — those widths
    ride in the passthrough ``rest``, not in the supervisor's own ``--gpu`` (which only feeds
    pre-flight). Handles both ``--gpu 3`` and ``--gpu=3`` forms; a malformed value is ignored here
    (the campaign's own argparse rejects it loudly at launch).
    """
    widths: list[int] = []
    toks = [a for a in rest if a != "--"]
    for i, tok in enumerate(toks):
        for flag in ("--gpu", "--search-gpu"):
            val: str | None = None
            if tok == flag and i + 1 < len(toks):
                val = toks[i + 1]
            elif tok.startswith(flag + "="):
                val = tok.split("=", 1)[1]
            if val is None:
                continue
            try:
                widths.append(int(val))
            except ValueError:
                pass
    return max(widths) if widths else None


def resume_command(base_cmd: list[str], attempt: int = 0) -> list[str]:
    """The command to launch: ``--resume`` is ALWAYS present (appended once, idempotent).

    M7(ii): resume from attempt 0 — including the very FIRST launch. The supervisor itself gets
    restarted (power loss -> the ONSTART task, an operator re-running it), and a first launch
    without ``--resume`` re-searches every arm that has no frozen winner yet: it RE-BILLS the paid
    author and re-trains finished candidates. Resume on a fresh output dir is a no-op (empty replay
    cache), so unconditionally resuming is strictly safer; a deliberately fresh run goes through
    ``run_campaign.py --no-resume`` directly, not through the supervisor. ``attempt`` is retained
    for call-site compatibility and documentation (the command no longer varies by attempt).
    """
    del attempt  # no longer gates anything — every launch resumes (see docstring)
    return base_cmd if "--resume" in base_cmd else [*base_cmd, "--resume"]


# ── Thin process loop ─────────────────────────────────────────────────────────────────────────────────
def _run_once(cmd: list[str]) -> int:
    print(f"[supervisor] launching: {' '.join(cmd)}", flush=True)
    try:
        return subprocess.call(cmd, cwd=str(_REPO))
    except KeyboardInterrupt:
        print("[supervisor] KeyboardInterrupt — leaving the child to drain; not relaunching.", flush=True)
        return GRACEFUL_SHUTDOWN_CODE


def _preflight(n_gpu: int) -> int:
    py = sys.executable
    return subprocess.call([py, str(_REPO / "scripts" / "preflight.py"), "--gpu", str(n_gpu)], cwd=str(_REPO))


def _abbreviates_no_resume(tok: str) -> bool:
    """True if a campaign passthrough token is argparse's unambiguous abbreviation of ``--no-resume``.

    ``run_campaign``'s parser leaves ``allow_abbrev`` at its default (True), so it accepts ANY unambiguous
    prefix of a long option. Its only two ``--no-*`` options — ``--no-resume`` and ``--no-shutdown`` —
    share the prefix ``--no-`` and diverge at ``r`` vs ``s``, so the SHORTEST prefix that unambiguously
    selects ``--no-resume`` is ``--no-r`` (6 chars). Any token from ``--no-r`` through the full
    ``--no-resume`` (optionally ``=value``) therefore disables resume and MUST be refused exactly like the
    literal token — else an abbreviated ``--no-resum`` slips the old exact-match guard yet still turns
    resume off, and every auto-restart re-searches + RE-BILLS the paid author (m6).
    """
    head = tok.split("=", 1)[0]
    # startswith("--no-r") excludes --no-shutdown and the ambiguous "--no-"/"--no" (argparse rejects those
    # itself); "--no-resume".startswith(head) confirms head is a genuine prefix of the option, not e.g.
    # a hypothetical "--no-really".
    return head.startswith("--no-r") and "--no-resume".startswith(head)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-restart supervisor for the campaign run.")
    ap.add_argument("--gpu", type=int, default=2, help="n_gpu (passed to pre-flight; not to the campaign here)")
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument(
        "--max-total-restarts", type=int, default=60,
        help="ABSOLUTE lifetime restart cap, NEVER reset by a healthy stretch (F3). The healthy reset "
             "is deliberately forgiving (weeks of independent transients must not exhaust the budget), "
             "but it lets a recurring fault inside one irreducible > 30-min unit restart FOREVER — each "
             "cycle able to re-bill one LLM call. Exceeding this cap is diagnosed as a SYSTEMIC fault.",
    )
    ap.add_argument("--no-preflight", action="store_true", help="skip the pre-flight gauntlet (not advised)")
    ap.add_argument("--campaign", default=str(_REPO / "scripts" / "run_campaign.py"))
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="args after `--` are passed to the campaign verbatim")
    args = ap.parse_args(argv)

    extra = [a for a in args.rest if a != "--"]
    # F4 (+ m6): REFUSE --no-resume LOUDLY (never silently strip — the operator's command and the executed
    # command must stay identical). A deliberately fresh run belongs directly on run_campaign.py,
    # NEVER under the auto-restart supervisor: 20 auto-restarted fresh runs = a full re-search
    # re-billing per cycle + frozen-winner overwrite + sealed-test-leg re-touch. The check covers argparse
    # prefix ABBREVIATIONS (--no-resum, --no-r), not just the literal token — see _abbreviates_no_resume.
    if any(_abbreviates_no_resume(a) for a in extra):
        print(
            "[supervisor] REFUSING to launch: --no-resume (or an unambiguous argparse abbreviation of it, "
            "e.g. --no-resum / --no-r) found in the campaign args. A fresh run must never run under the "
            "supervisor (every auto-restart would re-search and RE-BILL the paid author, overwrite frozen "
            "winners, and re-touch the sealed test leg). Launch it directly instead: "
            "python scripts/run_campaign.py --no-resume ...",
            flush=True,
        )
        return 2

    if not args.no_preflight:
        # F13: pre-flight at the width the campaign will ACTUALLY use — the max of the passthrough
        # --gpu/--search-gpu and the supervisor's own --gpu — so `--gpu 2 -- --gpu 3` cannot pass
        # pre-flight at 2 and then OOM at 3.
        pass_width = _passthrough_gpu_width(extra)
        n_gpu = max(int(args.gpu), pass_width) if pass_width is not None else int(args.gpu)
        code = _preflight(n_gpu)
        if code != 0:
            print("[supervisor] PRE-FLIGHT NO-GO — aborting before any compute is committed.", flush=True)
            return code

    base_cmd = [sys.executable, args.campaign, *extra]

    attempt = 0
    total_restarts = 0  # F3: lifetime counter — unlike ``attempt`` it is NEVER reset
    while True:
        cmd = resume_command(base_cmd, attempt)
        t_start = time.monotonic()
        exit_code = _run_once(cmd)
        runtime = time.monotonic() - t_start
        if exit_code == 0:
            print("[supervisor] campaign exited cleanly (0). Done.", flush=True)
            return 0
        # F9: exit 2 = argparse/usage — the IDENTICAL bad command line would be relaunched verbatim,
        # so no restart can ever succeed. Non-restartable: exit so the operator fixes the invocation.
        # (1 = generic crash and 3 = resumable-incomplete stay restartable as designed.)
        if exit_code == 2:
            print("[supervisor] campaign exited 2 (argparse/usage error) — NOT restartable; "
                  "fix the command line and relaunch.", flush=True)
            return exit_code
        # M7(i): a crash AFTER a healthy stretch (> 30 min of real progress) opens a FRESH incident —
        # reset the counter so weeks of occasional transients can never exhaust the restart budget;
        # only fast die-on-startup loops keep counting toward max_restarts.
        attempt = reset_attempt_after_healthy_run(attempt, runtime)
        if not should_restart(exit_code, attempt, args.max_restarts):
            print(f"[supervisor] not restarting (exit={exit_code}, attempt={attempt}/{args.max_restarts}).",
                  flush=True)
            return exit_code
        # F3: the ABSOLUTE cap. A fault that recurs after every healthy > 30-min stretch (e.g. an arm
        # that always crashes at hour 1) passes the reset above on every cycle and would restart
        # forever; total_restarts never resets, so exceeding it is a SYSTEMIC fault -> exit non-zero.
        if total_restarts >= int(args.max_total_restarts):
            print(
                f"[supervisor] SYSTEMIC FAULT: {total_restarts} lifetime restarts reached "
                f"(--max-total-restarts {args.max_total_restarts}) — the same failure keeps recurring "
                f"past healthy stretches, so the run is burning restarts (and possibly LLM spend) "
                f"without net progress. Investigate the root cause (last exit={exit_code}); "
                f"not relaunching.",
                flush=True,
            )
            return exit_code
        wait = backoff_seconds(attempt)
        attempt += 1
        total_restarts += 1
        print(f"[supervisor] campaign exited {exit_code} after {runtime:.0f}s; "
              f"restart {attempt}/{args.max_restarts} "
              f"(lifetime {total_restarts}/{args.max_total_restarts}) in {wait:.0f}s with --resume.",
              flush=True)
        time.sleep(wait)


if __name__ == "__main__":
    raise SystemExit(main())
