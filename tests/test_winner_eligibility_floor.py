"""R115 — the WINNER-ELIGIBILITY EXECUTION FLOOR on candidate selection.

Selection was `max(val_fitness)` with no execution-quality condition, so a candidate whose authored
reward RAISED on much of its training (the R66 neutral fallback standing in, counted by
`train_safe_default_count`) could be frozen — and the sealed leg would then RE-TRAIN that reward and
inherit the contamination, confounding an H2 arm contrast with EXECUTION QUALITY when identification
requires the arms to differ ONLY in the authored reward.

Measured over the full RUN 1 archive (613 counter-carrying records): 594 clean, 16 trace (<1 %),
3 SEVERE at 53.66 % / 50.02 % / 39.40 %.

These tests FAIL against the pre-R115 selector, which is the only reason to trust them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _write_candidate(root: Path, cid: str, *, val_fitness: float,
                     fallback: int | None, calls: int | None = 400_000) -> None:
    d = root / cid
    d.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {"val_fitness": val_fitness}
    if calls is not None:
        metrics["train_safe_call_count"] = calls
    if fallback is not None:
        metrics["train_safe_default_count"] = fallback
    # the canonical loader fail-closes on any missing REQUIRED_FIELDS, so the fixture carries the
    # full schema rather than a convenient subset
    record = {
        "run_id": cid, "arm": root.name, "seed": 0, "fold": 0, "candidate_id": cid,
        "generation": 1, "reward_source_hash": "0" * 64, "feedback_block": "",
        "metrics": metrics, "wall_clock": 1.0, "env_fingerprint": "test|dev=cpu",
    }
    (d / "record.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_registered_ceiling_is_read_from_the_prereg_not_hardcoded():
    from run_campaign import _winner_fallback_ceiling

    _winner_fallback_ceiling.cache_clear()
    ceiling = _winner_fallback_ceiling()
    assert 0.0 < ceiling < 1.0, ceiling
    # it must be THE registered value, not a literal in the selector
    import yaml
    from src.utils.config import repo_root

    yml = yaml.safe_load(
        (repo_root() / "config" / "preregistration.yaml").read_text(encoding="utf-8")
    )
    assert ceiling == float(yml["fitness"]["winner_max_fallback_frac"])


def test_a_contaminated_candidate_cannot_win_even_with_the_best_val_fitness(tmp_path):
    """THE defect: the highest val_fitness wins regardless of whether its reward actually ran."""
    from run_campaign import select_winner

    arm = tmp_path / "scalar"
    # the contaminated one is DELIBERATELY the best on val_fitness
    _write_candidate(arm, "scalar-g1-c4", val_fitness=9.99, fallback=214_649)   # 53.66 %
    _write_candidate(arm, "scalar-g1-c0", val_fitness=1.00, fallback=0)
    _write_candidate(arm, "scalar-g1-c1", val_fitness=0.50, fallback=3)         # 0.0008 % trace

    w = select_winner(arm)
    assert w is not None
    assert w["candidate_id"] == "scalar-g1-c0", (
        "a candidate whose reward fell back on 53.66 % of steps must be INELIGIBLE, however good "
        f"its val_fitness — got {w['candidate_id']}"
    )


def test_trace_contamination_is_still_eligible(tmp_path):
    """The floor must not discard good candidates over a handful of steps in 400,000.

    The observed distribution is bimodal with a 96x empty gap (worst trace 0.41 %, mildest severe
    39.40 %), so anything in the trace band is numerically irrelevant and stays eligible.
    """
    from run_campaign import select_winner

    arm = tmp_path / "distributional"
    _write_candidate(arm, "d-g1-c0", val_fitness=2.0, fallback=1650)   # 0.41 % — worst trace seen
    _write_candidate(arm, "d-g1-c1", val_fitness=1.0, fallback=0)
    assert select_winner(arm)["candidate_id"] == "d-g1-c0"


def test_records_without_counters_are_eligible_never_guessed_at(tmp_path):
    """A record carrying no counters says nothing about execution; it must not be excluded."""
    from run_campaign import select_winner

    arm = tmp_path / "placebo"
    _write_candidate(arm, "p-c0", val_fitness=3.0, fallback=None, calls=None)
    _write_candidate(arm, "p-c1", val_fitness=1.0, fallback=0)
    assert select_winner(arm)["candidate_id"] == "p-c0"

    # a zero denominator is equally uninformative and must not divide-by-zero
    arm2 = tmp_path / "placebo2"
    _write_candidate(arm2, "q-c0", val_fitness=3.0, fallback=0, calls=0)
    assert select_winner(arm2)["candidate_id"] == "q-c0"


def test_an_arm_with_no_eligible_candidate_fails_loud(tmp_path):
    """Silently promoting the least-bad contaminated candidate would hide the failure in the
    confirmatory result; the run must stop instead."""
    from run_campaign import select_winner

    arm = tmp_path / "scalar_cvar5"
    _write_candidate(arm, "s-c0", val_fitness=5.0, fallback=200_094)  # 50.02 %
    _write_candidate(arm, "s-c1", val_fitness=4.0, fallback=157_608)  # 39.40 %
    with pytest.raises(RuntimeError, match="R115 winner-eligibility floor"):
        select_winner(arm)


def test_an_empty_arm_still_returns_none_not_an_error(tmp_path):
    """Unchanged pre-existing contract: no candidates is not the same as all contaminated."""
    from run_campaign import select_winner

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert select_winner(empty) is None


def test_the_floor_is_effect_blind_it_never_reads_a_performance_field():
    """The rule's defensibility rests on being structurally unable to see an outcome."""
    import inspect

    import run_campaign

    src = inspect.getsource(run_campaign._winner_eligible)
    src += inspect.getsource(run_campaign._fallback_frac)
    for forbidden in ("val_fitness", "test_sharpe", "test_cvar", "fitness"):
        assert forbidden not in src, (
            f"the eligibility rule must never read {forbidden!r} — that would make it steerable "
            "toward an outcome and destroy the pre-registration argument"
        )


def test_the_floor_is_actually_WIRED_on_the_production_cluster_path(tmp_path):
    """A rule with no call site is not a rule (the 2026-07-26 finding #57 lesson: the killswitch
    had a wired GATE but no production caller, so it could never fire).

    The cluster path resolves SELECT/FREEZE through `_resolve_select_freeze`, which uses whatever
    `run.select_winner` holds and otherwise falls back to `run_campaign`'s. `run_campaign_cluster.py`
    injects NO selector, so production MUST resolve to the R115-bearing implementation. Asserted by
    resolving it exactly as production does, not by reading the source.
    """
    import inspect

    from src.cluster.campaign import ClusterRun, _resolve_select_freeze

    run = ClusterRun(run_batch=lambda *a, **k: None, spec_archive_root="/x", read_root=tmp_path)
    selector, _freeze = _resolve_select_freeze(run)
    assert selector.__module__ == "run_campaign", selector.__module__
    assert "_winner_eligible" in inspect.getsource(selector), (
        "the production selector does not apply the R115 eligibility floor — the amendment would be "
        "registered but inert"
    )


def test_all_contaminated_degrades_the_ARM_it_does_not_crash_the_LINE(tmp_path):
    """R115 must not turn an arm-level problem into a campaign-level crash.

    `run_arm_pipeline` documents that it "never raises for a 'no winner' arm", and NOTHING wrapped
    the three selection sites. An uncaught raise would kill the arm, the supervisor would relaunch
    the line into the same error, and an unattended multi-day run would sit in a 600 s hot loop.

    Degrading is not hiding: the reason must be DISTINCT from `no_winner` (candidates existed but
    none was eligible — a different diagnosis), and the incompleteness is caught downstream because
    a missing arm makes present != expected in the integrity census, so the C3 gate stops the line.
    """
    from src.cluster.campaign import _select_eligible_winner
    from src.selection.fitness import NoEligibleWinnerError

    def _all_contaminated(_root):
        raise NoEligibleWinnerError("R115 winner-eligibility floor: all 30 candidate(s) ...")

    winner, reason = _select_eligible_winner(_all_contaminated, tmp_path, "scalar")
    assert winner is None
    assert reason == "no_eligible_winner", reason
    assert reason != "no_winner", "the two causes must stay distinguishable"

    # an EMPTY arm still reports the original reason, unchanged
    winner2, reason2 = _select_eligible_winner(lambda _r: None, tmp_path, "scalar")
    assert winner2 is None and reason2 is None

    # and a normal selection passes straight through
    winner3, reason3 = _select_eligible_winner(lambda _r: {"candidate_id": "c0"}, tmp_path, "scalar")
    assert winner3 == {"candidate_id": "c0"} and reason3 is None


def test_the_r115_error_is_catchable_by_type_not_by_message():
    """Matching on an error message is brittle; the orchestrator catches the TYPE."""
    from src.selection.fitness import NoEligibleWinnerError

    assert issubclass(NoEligibleWinnerError, RuntimeError), (
        "must subclass RuntimeError so existing broad handlers keep working"
    )
