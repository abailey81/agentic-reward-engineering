"""Unit tests for the auto-restart supervisor's pure decision logic (scripts/supervisor.py)."""
from __future__ import annotations

from scripts import supervisor as sup


def test_no_restart_on_clean_exit() -> None:
    assert sup.should_restart(0, attempt=0, max_restarts=20) is False


def test_no_restart_on_graceful_shutdown() -> None:
    # A non-zero code that the operator caused on purpose must NOT be fought.
    assert sup.should_restart(137, attempt=0, max_restarts=20, graceful=True) is False


def test_restart_on_crash_within_budget() -> None:
    assert sup.should_restart(1, attempt=0, max_restarts=3) is True
    assert sup.should_restart(1, attempt=2, max_restarts=3) is True


def test_no_restart_when_budget_spent() -> None:
    assert sup.should_restart(1, attempt=3, max_restarts=3) is False
    assert sup.should_restart(1, attempt=99, max_restarts=3) is False


def test_backoff_is_monotonic_and_capped() -> None:
    waits = [sup.backoff_seconds(i, base=30.0, cap=600.0) for i in range(8)]
    assert waits[0] == 30.0
    assert all(b <= a or a == 600.0 for a, b in zip(waits[1:], waits[:-1]))  # non-decreasing
    assert max(waits) == 600.0  # capped at 10 min
    assert all(w <= 600.0 for w in waits)


def test_resume_command_always_resumes_including_first_launch() -> None:
    """M7(ii): --resume on EVERY launch, attempt 0 included — a supervisor restarted by the ONSTART
    task (or the operator) must never issue a fresh, re-billing search. (This test previously locked
    in the opposite: a fresh first launch — that WAS the audited bug.)"""
    base = ["py", "run_campaign.py", "--gpu", "2"]
    assert "--resume" in sup.resume_command(base, attempt=0)      # first launch resumes too
    assert "--resume" in sup.resume_command(base, attempt=1)
    # idempotent — never doubles the flag, and the base command is not mutated
    withr = [*base, "--resume"]
    assert sup.resume_command(withr, attempt=2).count("--resume") == 1
    assert base == ["py", "run_campaign.py", "--gpu", "2"]


def test_reset_attempt_after_healthy_run() -> None:
    """M7(i): a child that survived > the healthy window opens a FRESH incident (attempt -> 0);
    a fast die-on-startup keeps counting toward max_restarts (crash loops still bounded)."""
    assert sup.reset_attempt_after_healthy_run(7, runtime_secs=3600.0) == 0        # healthy stretch
    assert sup.reset_attempt_after_healthy_run(7, runtime_secs=10.0) == 7          # startup crash loop
    assert sup.reset_attempt_after_healthy_run(7, runtime_secs=sup.HEALTHY_RUNTIME_SECS) == 7  # boundary: not strictly greater
    assert sup.reset_attempt_after_healthy_run(0, runtime_secs=9999.0) == 0
    # custom window
    assert sup.reset_attempt_after_healthy_run(3, runtime_secs=61.0, healthy_secs=60.0) == 0


def test_healthy_reset_keeps_a_weeks_long_run_restartable() -> None:
    """The composed contract: daily transient crashes over weeks never exhaust the budget, because
    each healthy stretch resets the counter before should_restart consults it."""
    attempt = 0
    for _day in range(40):  # 40 healthy-then-crash cycles >> max_restarts=20
        attempt = sup.reset_attempt_after_healthy_run(attempt, runtime_secs=86_000.0)
        assert sup.should_restart(1, attempt, max_restarts=20) is True
        attempt += 1
