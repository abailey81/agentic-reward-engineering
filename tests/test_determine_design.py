"""Tests for the Design Determination Pipeline (scripts/determine_design.py).

Covers the search-saturation engine (the candidate-budget analogue of recommend_budget), the record->curve
reduction, and the determination-status overlay + freeze-readiness logic. No torch, no GPU, no test data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import determine_design as dd  # noqa: E402


# --------------------------------------------------------------------------- #
# Search-saturation engine (recommend_candidates)                              #
# --------------------------------------------------------------------------- #
def test_recommend_candidates_all_saturated_picks_slowest_arm() -> None:
    curves = {
        "distributional": [0.10, 0.20, 0.20, 0.20],   # plateaus at gen 1
        "scalar": [0.05, 0.05, 0.30, 0.30],            # plateaus at gen 2 (the slowest)
    }
    rec = dd.recommend_candidates(curves, candidates_per_gen=5, patience=1)
    assert rec["saturated"] is True
    assert rec["recommended_candidates"] == (2 + 1) * 5  # slowest arm saturated at gen 2 -> 15 candidates


def test_recommend_candidates_flags_still_rising() -> None:
    curves = {
        "distributional": [0.10, 0.20, 0.20],
        "scalar": [0.05, 0.15, 0.40],   # still climbing at the last generation
    }
    rec = dd.recommend_candidates(curves, candidates_per_gen=5, patience=1)
    assert rec["saturated"] is False
    assert rec["recommended_candidates"] is None
    assert "scalar" in rec["reason"]


def test_recommend_candidates_insufficient_generations() -> None:
    rec = dd.recommend_candidates({"a": [0.1]}, candidates_per_gen=5, patience=2)
    assert rec["saturated"] is None and rec["recommended_candidates"] is None


def test_best_so_far_curves_is_monotone_and_per_arm() -> None:
    records = [
        {"arm": "a", "generation": 0, "metrics": {"val_fitness": 0.1}},
        {"arm": "a", "generation": 1, "metrics": {"val_fitness": 0.05}},  # worse -> best-so-far stays 0.1
        {"arm": "a", "generation": 2, "metrics": {"val_fitness": 0.3}},
        {"arm": "b", "generation": 0, "metrics": {"val_fitness": 0.2}},
        {"arm": "b", "generation": 0, "metrics": {"val_fitness": 0.25}},  # same gen -> max within gen
    ]
    curves = dd.best_so_far_curves(records)
    assert curves["a"] == [0.1, 0.1, 0.3]      # monotone non-decreasing
    assert curves["b"] == [0.25]


# --------------------------------------------------------------------------- #
# Determination-status overlay + freeze readiness                              #
# --------------------------------------------------------------------------- #
def test_determine_blocks_freeze_when_measure_params_pending() -> None:
    res = dd.determine({})  # no evidence: B*, candidates, seeds unresolved (lambda is FIX-class, never blocks)
    assert res["freeze_ready"] is False
    for blocker in ("train_steps_per_candidate", "n_seeds", "candidates_per_arm"):
        assert blocker in res["blockers"]
    # lambda_cvar is 0 BY DESIGN (tail-blind selector; prereg fitness.lambda_cvar, freeze gate #7) — it was
    # reclassified CALIBRATE -> FIX (2026-07-02) and must never appear as a freeze blocker.
    assert "lambda_cvar" not in res["blockers"] and "lambda_frozen" not in res["blockers"]
    rows = {r["name"]: r for r in res["rows"]}
    assert rows["lambda_cvar"]["class"] == dd.ParamClass.FIX.value
    assert rows["lambda_cvar"]["status"] == dd.Status.FIXED.value


def test_n_seeds_pilot_ran_but_sigma_trigger_fired_blocks_at_placeholder() -> None:
    """A sigma_D pilot that fires the pre-registered ">0.10 -> raise seeds" trigger must NOT report
    n_seeds DETERMINED while the config still carries the pre-pilot 30-seed placeholder — that false
    READY could greenlight a premature under-powered freeze (2026-07-05 M06)."""
    ev = {"sigma_seed_pilot": True, "sigma_pilot_sigma_d": 0.369, "config_n_seeds": 30}
    rows = {r["name"]: r["status"] for r in dd.determine(ev)["rows"]}
    assert rows["n_seeds"] == dd.Status.PENDING.value
    # Amending the seed count past the placeholder clears it.
    ev_amended = {**ev, "config_n_seeds": 350}
    rows2 = {r["name"]: r["status"] for r in dd.determine(ev_amended)["rows"]}
    assert rows2["n_seeds"] == dd.Status.DETERMINED.value
    # A pilot that did NOT fire the trigger (sigma_D <= 0.10) leaves 30 seeds decided.
    ev_calm = {"sigma_seed_pilot": True, "sigma_pilot_sigma_d": 0.05, "config_n_seeds": 30}
    rows3 = {r["name"]: r["status"] for r in dd.determine(ev_calm)["rows"]}
    assert rows3["n_seeds"] == dd.Status.DETERMINED.value


def test_determine_cash_rate_zero_is_fix_needed() -> None:
    rows = {r["name"]: r["status"] for r in dd.determine({"cash_daily_rate": 0.0})["rows"]}
    assert rows["cash_daily_rate"] == dd.Status.FIX_NEEDED.value
    # A non-zero (risk-free) rate clears it.
    rows2 = {r["name"]: r["status"] for r in dd.determine({"cash_daily_rate": 0.00012})["rows"]}
    assert rows2["cash_daily_rate"] == dd.Status.DETERMINED.value


def test_determine_cash_zero_decided_when_numeraire_ratified() -> None:
    """cash=0 MATCHING the §10-ratified numeraire (2026-07-01) is a DECISION, not a defect:
    FIX_NEEDED encoded pre-ratification semantics. A mismatching anchor must NOT clear it."""
    rows = {r["name"]: r["status"] for r in dd.determine(
        {"cash_daily_rate": 0.0, "numeraire_ratified_cash": 0.0})["rows"]}
    assert rows["cash_daily_rate"] == dd.Status.DECIDED.value
    # anchor present but MISMATCHED -> still FIX_NEEDED (the env drifted from the ratified value)
    rows2 = {r["name"]: r["status"] for r in dd.determine(
        {"cash_daily_rate": 0.0, "numeraire_ratified_cash": 0.0001})["rows"]}
    assert rows2["cash_daily_rate"] == dd.Status.FIX_NEEDED.value


def test_determine_freeze_ready_when_all_blockers_resolved() -> None:
    ev = {
        "recommended_budget": 150_000,
        "candidates_saturated": True,
        "sigma_seed_pilot": True,
        "cash_daily_rate": 0.00012,
    }
    res = dd.determine(ev)
    assert res["freeze_ready"] is True and res["blockers"] == []


def test_determine_candidates_decided_by_ratified_cap_when_not_saturated() -> None:
    """Saturation is diagnostic-only: a False verdict + the ratified 30-cap (2026-07-01, multiplicity)
    must yield DECIDED and NOT block the freeze — requiring saturated=True made freeze-readiness
    unsatisfiable pre-campaign (the real prototype archive judges NOT-saturated: scalar's gen-7 jump)."""
    ev = {
        "recommended_budget": 150_000,
        "candidates_saturated": False,   # the honest prototype verdict
        "candidates_ratified": 30,       # the ratified decision anchor (campaign.yaml)
        "sigma_seed_pilot": True,
        "cash_daily_rate": 0.00012,
    }
    res = dd.determine(ev)
    rows = {r["name"]: r["status"] for r in res["rows"]}
    assert rows["candidates_per_arm"] == dd.Status.DECIDED.value
    assert "candidates_per_arm" not in res["blockers"]
    assert res["freeze_ready"] is True


def test_determine_candidates_pending_without_ratified_anchor() -> None:
    """No saturation AND no ratified value -> the parameter genuinely blocks (no silent unblocking)."""
    res = dd.determine({"candidates_saturated": False})
    rows = {r["name"]: r["status"] for r in res["rows"]}
    assert rows["candidates_per_arm"] == dd.Status.PENDING.value
    assert "candidates_per_arm" in res["blockers"]


def test_registry_blockers_are_only_measure_or_calibrate() -> None:
    for spec in dd.REGISTRY:
        if spec.blocks_freeze:
            assert spec.klass in (dd.ParamClass.MEASURE, dd.ParamClass.CALIBRATE)
