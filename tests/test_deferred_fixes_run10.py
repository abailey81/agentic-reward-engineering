"""The smaller RUN 10 deferred fixes, each with the falsifiable test its register entry demanded.

  * item 9  (§39) — `CPU_THREAD_SPEEDUP[8]` was an ISOLATED BENCH number; production says 1.92x.
  * item 14 (§79.5) — `transport_guard`'s `timeout_events` counted a string nothing ever emits,
    so it was STRUCTURALLY ZERO: a reassuring number that could not move.

Both are REPORTING/MODEL correctness, not behaviour: nothing the campaign computes changes, only
what we predict about it and what we tell ourselves about it. That is precisely why they are easy
to leave broken — and a monitoring figure that cannot move is more dangerous than a missing one,
because it reads as evidence of health.
"""
from __future__ import annotations

from pathlib import Path

from scripts.campaign_guards import transport_guard
from src.cluster import lanes


# --------------------------------------------------------------------------- #
# item 9 / §39 — the ladder model must use the FIELD number, not the bench     #
# --------------------------------------------------------------------------- #
def test_chain_thread_speedup_is_the_field_value_not_the_bench_value() -> None:
    """Measured across 740 timed trainings on shared nodes: 1.92x median / 1.75x mean at 8 threads.

    The 2.72x came from one idle 8-core box. At 2.72x the critical-chain floor reports 3.27 d; the
    measured value puts it at ~4.6 d, so quoting the bench understated the front of the ladder by
    ~1.4 days. Co-tenants take memory bandwidth an idle bench never loses.
    """
    assert 1.8 <= lanes.CPU_THREAD_SPEEDUP[8] <= 2.0, "8-thread speedup looks like the bench number"
    p = lanes.plan_lanes(rung=30, cpu_cores=100_000, chain_threads=8)
    floor = float(p.makespan_days)
    assert 4.3 <= floor <= 5.0, f"critical-chain floor {floor:.2f} d looks like the bench number"


def test_the_speedup_curve_stays_monotone_to_its_optimum_and_falls_after() -> None:
    """The SHAPE must survive the re-measurement: 16 threads is measurably slower than 8.

    Without this, a future edit could 'fix' the numbers into a monotone curve and quietly justify
    `CPU_CHAIN_THREADS = 16`, which the measurement says is SLOWER (44.0 vs 55.1 steps/s) — and
    thread count is inside the determinism envelope, so that change is not even permissible.
    """
    s = lanes.CPU_THREAD_SPEEDUP
    assert s[1] < s[2] < s[4] < s[8], "the curve must rise to its optimum"
    assert s[16] < s[8], "16 threads is measurably SLOWER than 8 — the curve must still say so"
    assert lanes.CPU_CHAIN_THREADS == 8


# --------------------------------------------------------------------------- #
# item 14 / §79.5 — a counter that cannot move is worse than no counter        #
# --------------------------------------------------------------------------- #
def _guard_timeouts(root: Path) -> int:
    _rc, lines = transport_guard(root)
    for ln in lines:
        if ln.startswith("timeout_events="):
            return int(ln.split()[0].split("=")[1])
    raise AssertionError(f"transport_guard printed no timeout_events line: {lines}")


def test_timeout_events_counts_what_a_timeout_actually_emits(tmp_path: Path) -> None:
    """Pre-fix this searched for `"timed out after"`, which NOTHING in the codebase emits.

    `grep -rn "timed out" src/` returns exactly one hit and it is a retry-classification KEYWORD
    LIST, not a log message — so the counter was structurally pinned at 0.
    """
    log = tmp_path / "driver_test.log"
    log.write_text("2026-08-01 00:00:00 | INFO | src.cluster.driver | nothing to see\n",
                   encoding="utf-8")
    assert _guard_timeouts(tmp_path) == 0            # the clean baseline

    log.write_text(log.read_text(encoding="utf-8")
                   + "2026-08-01 00:01:00 | WARNING | src.cluster.poll | ssh_timeout_diagnostic "
                     "child_already_exited=True\n", encoding="utf-8")
    assert _guard_timeouts(tmp_path) == 1, "the counter did not move on a REAL timeout line"

    log.write_text(log.read_text(encoding="utf-8")
                   + "2026-08-01 00:02:00 | ERROR | src.cluster.poll | TimeoutExpired on pull\n",
                   encoding="utf-8")
    assert _guard_timeouts(tmp_path) == 2, "the second real timeout vocabulary is not counted"


def test_the_dead_string_alone_no_longer_inflates_the_counter(tmp_path: Path) -> None:
    """Guard against the opposite error: keeping the dead string as an OR would let unrelated
    prose containing it count as a timeout. Nothing emits it, so nothing should count it."""
    log = tmp_path / "driver_test.log"
    log.write_text("2026-08-01 00:00:00 | INFO | x | the job timed out after a while, allegedly\n",
                   encoding="utf-8")
    assert _guard_timeouts(tmp_path) == 0


def test_the_guards_verdict_still_rests_on_consecutive_depth(tmp_path: Path) -> None:
    """NO REGRESSION: the guard's return code is driven by `worst_consecutive`, not by this
    counter. Changing what is COUNTED must not change what is DECIDED."""
    log = tmp_path / "driver_test.log"
    log.write_text("2026-08-01 | INFO | x | pull failed (2 consecutive)\n", encoding="utf-8")
    assert transport_guard(tmp_path)[0] == 0
    log.write_text("2026-08-01 | INFO | x | pull failed (12 consecutive)\n", encoding="utf-8")
    assert transport_guard(tmp_path)[0] == 2, "a deep consecutive run must still be CRITICAL"
