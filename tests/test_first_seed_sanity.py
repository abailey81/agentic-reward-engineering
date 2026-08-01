"""The first-seed garbage detector — and the property that makes it safe to run mid-campaign.

The pre-registration commits to a SINGLE confirmatory look at a pre-declared date. Anything that
lets a human see WHICH ARM IS WINNING earlier is optional stopping, and would invalidate every
interval in the dissertation. So the load-bearing test here is not "does it spot bad records" but
"is it blind to the effect" — proven by running it over two archives that differ ONLY in which arm
performs better and requiring byte-identical output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.first_seed_sanity import (
    GARBAGE,
    OK,
    SUSPECT,
    assess_record,
    assess_seed,
    check_returns_finite,
    check_returns_nondegenerate,
    check_returns_plausible,
    check_reward_actually_ran,
    check_training_happened,
    verdict,
)
from src.io.results import write_run


def _rec(arm: str, seed: int, rets, *, sd: int = 0, calls: int = 400_000, wall: float = 3100.0):
    return {"run_id": f"{arm}-s{seed}", "arm": arm, "seed": seed, "fold": 0,
            "candidate_id": f"{arm}-winner", "generation": 0, "reward_source_hash": "h" * 64,
            "feedback_block": "", "wall_clock": wall, "env_fingerprint": "env|dev=cpu",
            "metrics": {"test_returns": [float(x) for x in rets],
                        "train_safe_default_count": sd, "train_safe_call_count": calls}}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE LOAD-BEARING PROPERTY: it cannot preview the result
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_the_gate_is_BLIND_to_which_arm_wins(tmp_path: Path):
    """Two archives identical except for WHICH arm performs better must produce IDENTICAL output.

    If the gate leaked any performance signal, these two would differ — and running it mid-campaign
    would be peeking, which breaks the single-look inference the whole design rests on.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(500) * 0.01

    def build(root: Path, dist_better: bool) -> dict:
        edge = 0.004 if dist_better else -0.004
        write_run(_rec("distributional", 0, base + edge), root / "test" / "distributional")
        write_run(_rec("scalar", 0, base - edge), root / "test" / "scalar")
        return assess_seed(root)

    a = build(tmp_path / "dist_wins", True)
    b = build(tmp_path / "scalar_wins", False)
    # normalise only the root-dependent identity, never any measured content
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert a["verdict"] == OK


def test_no_performance_value_appears_anywhere_in_the_output(tmp_path: Path):
    """Even a single fitness/Sharpe number in the report would let a human peek."""
    rng = np.random.default_rng(1)
    write_run(_rec("distributional", 0, rng.standard_normal(300) * 0.01),
              tmp_path / "test" / "distributional")
    blob = json.dumps(assess_seed(tmp_path), default=str).lower()
    for forbidden in ("val_fitness", "test_sharpe", "test_cvar", "sharpe", "cvar", "fitness"):
        assert forbidden not in blob, f"the gate leaked {forbidden!r} — that is a peek"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE GARBAGE SIGNALS — each must actually fire
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_NaN_returns_are_GARBAGE():
    s = check_returns_finite(np.array([0.01, np.nan, 0.02]))
    assert s.status == GARBAGE and "meaningless" in s.detail


def test_a_FLAT_return_series_is_GARBAGE_not_a_result():
    """A policy parked in cash produces a flat line. It completes, it looks healthy, it is useless."""
    s = check_returns_nondegenerate(np.full(300, 0.0004))
    assert s.status == GARBAGE and "degenerate" in s.detail


def test_ALL_ZERO_returns_are_GARBAGE():
    assert check_returns_nondegenerate(np.zeros(300)).status == GARBAGE


def test_an_ABSURD_daily_return_is_GARBAGE():
    """+300% in a day is a broken environment, not a lucky strategy."""
    s = check_returns_plausible(np.array([0.01, 3.0, -0.02]))
    assert s.status == GARBAGE and "broken" in s.detail


def test_a_reward_that_MOSTLY_FELL_BACK_is_GARBAGE():
    """The most valuable signal: the agent trained on the neutral fallback, not on its own reward.
    The run completes and looks healthy, and the record is worthless."""
    s = check_reward_actually_ran(safe_default_count=300_000, safe_call_count=400_000)
    assert s.status == GARBAGE and "not a test of that reward" in s.detail


def test_PARTIAL_fallback_contamination_is_SUSPECT():
    s = check_reward_actually_ran(safe_default_count=4_000, safe_call_count=400_000)
    assert s.status == SUSPECT


def test_a_clean_reward_run_is_OK():
    assert check_reward_actually_ran(0, 400_000).status == OK


def test_ABSENT_counters_are_SUSPECT_never_silently_OK():
    """Absence of evidence must not read as evidence of health."""
    assert check_reward_actually_ran(None, None).status == SUSPECT


def test_zero_wall_clock_with_NO_reward_calls_means_nothing_trained():
    """The genuine catch, preserved: no time AND no evidence of training is GARBAGE."""
    assert check_training_happened(0).status == GARBAGE
    assert check_training_happened(0, 0).status == GARBAGE
    assert check_training_happened(0, None).status == GARBAGE


def test_an_UNTIMED_but_TRAINED_record_is_OK_not_garbage():
    """`test_leg.py` hardcodes `wall_clock: 0.0` on EVERY test-leg record.

    Judging on the timer alone condemned the whole SCORED leg — the 2026-07-28 06:53Z sentinel
    CRITICAL on `baseline_return_minus_cvar-s24`, a record with 400,000 reward calls, a full
    train_curve and real test returns. The clock is provenance; the reward calls are proof.
    """
    s = check_training_happened(0.0, 400_000)
    assert s.status == OK and "prove training ran" in s.detail

    rec = _rec("baseline_return_minus_cvar", 24,
               np.random.default_rng(7).standard_normal(300) * 0.01)
    rec["wall_clock"] = 0.0
    assert verdict(assess_record(rec)) == OK, "an untimed but trained test record must not be GARBAGE"


def test_a_record_with_no_provenance_is_GARBAGE(tmp_path: Path):
    bad = _rec("scalar", 0, np.random.default_rng(2).standard_normal(200) * 0.01)
    bad["reward_source_hash"] = ""
    sigs = assess_record(bad)
    assert verdict(sigs) == GARBAGE
    assert any(s.name == "provenance" and s.status == GARBAGE for s in sigs)


def test_a_HEALTHY_record_is_OK():
    good = _rec("distributional", 0, np.random.default_rng(3).standard_normal(400) * 0.01)
    assert verdict(assess_record(good)) == OK


# ══════════════════════════════════════════════════════════════════════════════════════════════
# END-TO-END
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_an_EMPTY_archive_says_nothing_rather_than_passing(tmp_path: Path):
    out = assess_seed(tmp_path)
    assert out["status"] == "no_records" and "says nothing about quality" in out["note"]
    assert "verdict" not in out


def test_a_broken_first_seed_is_caught_end_to_end(tmp_path: Path):
    """The whole point: garbage visible on seed 1 instead of on day 3."""
    rng = np.random.default_rng(4)
    write_run(_rec("distributional", 0, rng.standard_normal(300) * 0.01),
              tmp_path / "test" / "distributional")
    write_run(_rec("scalar", 0, np.full(300, 0.0), sd=390_000),   # crashed reward + flat line
              tmp_path / "test" / "scalar")
    out = assess_seed(tmp_path)
    assert out["verdict"] == GARBAGE
    bad = [r for r in out["records"] if r["arm"] == "scalar"][0]
    names = {s["name"] for s in bad["signals"]}
    assert "reward_actually_ran" in names and "returns_nondegenerate" in names


def test_search_leg_records_are_ASSESSED_not_silently_skipped(tmp_path: Path):
    """A campaign spends most of its life with `search_leg_*` as the ONLY leg records that exist.

    The gate used to gather `test`, `search` and `test_leg_*` only, so every replication leg was
    invisible to it -- 23 of the 29 records live at 2026-07-28 06:10Z, and precisely where the
    authoring failures concentrate. A broken search-leg record must FAIL the gate, not be skipped:
    skipping reads as a clean bill of health, which is the dangerous direction.
    """
    rng = np.random.default_rng(11)
    write_run(_rec("distributional", 0, rng.standard_normal(300) * 0.01),
              tmp_path / "search_leg_deepseek_v4_pro" / "distributional")
    write_run(_rec("scalar", 0, np.full(300, 0.0), sd=390_000),   # crashed reward + flat line
              tmp_path / "search_leg_deepseek_v4_pro" / "scalar")

    out = assess_seed(tmp_path)
    assert out["status"] != "no_records", "search-leg records were not gathered at all"
    assert len(out["records"]) == 2
    assert out["verdict"] == GARBAGE, "a broken search-leg record must fail the gate"


def test_it_reports_the_FIRST_seed_by_default(tmp_path: Path):
    rng = np.random.default_rng(5)
    for seed in (3, 7):
        write_run(_rec("distributional", seed, rng.standard_normal(200) * 0.01),
                  tmp_path / "test" / "distributional")
    assert assess_seed(tmp_path)["seed"] == 3
    assert assess_seed(tmp_path, seed=7)["seed"] == 7
