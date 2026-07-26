"""Tests for the experimental-arm factory (FINAL_PLAN F.9).

Covers: all seven arms build; every arm shares the same matched candidate budget;
the five LLM arms are identical except their feedback kind; the two search arms
are non-LLM with the correct search kind; and each LLM feedback kind maps onto a
valid feedback-schema arm.
"""

from __future__ import annotations


from dataclasses import replace

import pytest

from src.arms.factory import (
    FEEDBACK_KIND_TO_SCHEMA_ARM,
    Arm,
    all_arms,
    assert_fixed_agent_across_arms,
    build_arm,
    schema_arm_for,
)
from src.feedback import schema

EXPECTED_NAMES = {
    "distributional",
    "scalar",
    "placebo",
    "scalar_cvar5",
    "placebo_shuffled",
    "random_search",
    "bayes_opt",
    "cma_es",
    "tpe",
}
LLM_NAMES = {"distributional", "scalar", "placebo", "scalar_cvar5", "placebo_shuffled"}


def test_all_nine_arms_build() -> None:
    """Every declared arm builds into a frozen Arm with the right name set (9: 5 LLM + 4 H4 search)."""
    arms = all_arms()
    assert len(arms) == 9
    assert {a.name for a in arms} == EXPECTED_NAMES
    for a in arms:
        assert isinstance(a, Arm)
        # Frozen dataclass: assignment must fail.
        with pytest.raises(Exception):
            a.name = "mutated"  # type: ignore[misc]


def test_llm_arms_routing_tuples_match_config() -> None:
    """The hardcoded ``_LLM_ARMS`` routing tuples (parallel scheduler + run_prototype serial path) must
    list EXACTLY the config's LLM arms. Else a new LLM arm (e.g. ``placebo_shuffled``, R32) is silently
    routed to the SEARCH driver and fails — the drift the 5th arm exposed. Drift-guard for both copies."""
    import importlib

    from src.orchestration.parallel import _LLM_ARMS as PAR_LLM_ARMS

    llm_arms = {a.name for a in all_arms() if a.is_llm}
    assert set(PAR_LLM_ARMS) == llm_arms, (
        f"parallel._LLM_ARMS {set(PAR_LLM_ARMS)} != config LLM arms {llm_arms}"
    )
    rp = importlib.import_module("scripts.run_prototype")
    assert set(rp._LLM_ARMS) == llm_arms, (
        f"run_prototype._LLM_ARMS {set(rp._LLM_ARMS)} != config LLM arms {llm_arms}"
    )


def test_config_arm_rosters_match_factory() -> None:
    """The FROZEN preregistration.yaml roster, the operational campaign.yaml arms, and the arm factory
    must list EXACTLY the same arms — else the freeze can bind a different design than the campaign runs
    (the placebo_shuffled 6-vs-7 drift, V1 2026-06-26). Drift-guard across all three."""
    from src.utils.config import load_config

    factory = {a.name for a in all_arms()}
    prereg_arms = set(load_config("preregistration")["arms"])
    campaign_arms = set(load_config("campaign")["arms"])
    assert prereg_arms == factory, f"preregistration.yaml arms {prereg_arms} != factory {factory}"
    assert campaign_arms == factory, f"campaign.yaml arms {campaign_arms} != factory {factory}"


def test_equal_budget_across_arms() -> None:
    """Every arm receives the same matched candidate budget (matched compute)."""
    arms = all_arms()
    budgets = {a.candidate_budget for a in arms}
    assert len(budgets) == 1, f"budgets differ across arms: {budgets}"
    (budget,) = budgets
    assert budget > 0


def test_llm_arms_are_llm_with_search_none() -> None:
    """The five feedback arms are LLM arms carrying no search kind."""
    for name in LLM_NAMES:
        arm = build_arm(name)
        assert arm.is_llm is True
        assert arm.search_kind is None
        assert arm.feedback_kind is not None


def test_llm_arms_identical_except_feedback_kind() -> None:
    """LLM arms differ ONLY in feedback_kind; all other fields are identical."""
    arms = [build_arm(n) for n in sorted(LLM_NAMES)]

    # All feedback kinds are distinct.
    kinds = [a.feedback_kind for a in arms]
    assert len(set(kinds)) == len(kinds)

    # Normalizing away name + feedback_kind makes every LLM arm identical.
    normalized = {
        replace(a, name="x", feedback_kind="x") for a in arms
    }
    assert len(normalized) == 1, f"LLM arms differ beyond feedback_kind: {normalized}"


def test_search_arms_are_non_llm_with_correct_kind() -> None:
    """random_search/bayes_opt are non-LLM with the right search kind."""
    rs = build_arm("random_search")
    bo = build_arm("bayes_opt")

    assert rs.is_llm is False
    assert rs.search_kind == "code"
    assert rs.feedback_kind is None

    assert bo.is_llm is False
    assert bo.search_kind == "template"
    assert bo.feedback_kind is None


def test_feedback_kinds_map_to_valid_schema_arms() -> None:
    """Each LLM feedback_kind maps onto a real feedback-schema arm.

    Cross-checked against ``src.feedback.schema.block_fields``: the mapped schema
    arm must produce a non-empty field list (i.e. it is a recognized arm).
    """
    for name in LLM_NAMES:
        arm = build_arm(name)
        schema_arm = schema_arm_for(arm)
        assert schema_arm in FEEDBACK_KIND_TO_SCHEMA_ARM.values()
        # Must be a valid arm for the schema module (raises ValueError otherwise).
        fields = schema.block_fields(schema_arm)
        assert isinstance(fields, list) and len(fields) >= 1
        # And the block renders without error.
        tail = {
            "cvar_05": -0.04,
            "cvar_10": -0.03,
            "cvar_25": -0.02,
            "cvar_01": -0.07,
            "left_tail_mass": 0.06,
            "robust_skew": -0.4,
        }
        block = schema.build_block(
            schema_arm, 0.83, tail,
            shuffle_seed=(
                schema.shuffle_seed_from_id(name) if schema_arm == "placebo_shuffled" else None
            ),
        )
        assert isinstance(block, str) and len(block) > 0


def test_distributional_maps_to_full_tail_set() -> None:
    """The headline contribution arm carries the full tail set."""
    arm = build_arm("distributional")
    assert arm.feedback_kind == "full_tail_set"
    assert schema_arm_for(arm) == "distributional"


def test_unknown_arm_raises() -> None:
    """Building an undeclared arm raises KeyError."""
    with pytest.raises(KeyError):
        build_arm("does_not_exist")


# --------------------------------------------------------------------------- #
# runtime equivalence test — matched compute + a fixed agent (audit A-1)       #
# --------------------------------------------------------------------------- #
def test_fixed_agent_equivalence_holds_for_all_arms() -> None:
    """Every arm trains the IDENTICAL fixed SB3-SAC agent at one matched budget (only feedback differs)."""
    arms = all_arms()
    res = assert_fixed_agent_across_arms(arms, {"train_steps": 1000, "batch_size": 256})
    assert res["candidate_budget"] == arms[0].candidate_budget
    assert res["train_steps"] == 1000
    assert "seed" not in res["agent_kwargs"]  # the SHARED config excludes the (legitimately varying) seed
    assert res["agent_kwargs"]["policy"] == "MlpPolicy"  # the documented fixed headline agent


def test_matched_compute_violation_is_caught() -> None:
    """A single arm with a divergent candidate budget breaks the matched-compute invariant."""
    arms = all_arms()
    tampered = [replace(arms[0], candidate_budget=arms[0].candidate_budget + 1), *arms[1:]]
    with pytest.raises(AssertionError, match="matched compute"):
        assert_fixed_agent_across_arms(tampered, {"train_steps": 100})


def test_equivalence_guards_inputs() -> None:
    """No arms -> ValueError; non-distinct seeds (cannot prove seed-only dependence) -> ValueError."""
    with pytest.raises(ValueError):
        assert_fixed_agent_across_arms([], {"train_steps": 100})
    with pytest.raises(ValueError):
        assert_fixed_agent_across_arms(all_arms(), {"train_steps": 100}, seeds=(0, 0))
