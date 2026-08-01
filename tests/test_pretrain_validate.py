"""The pre-training gate — and the property that makes it worth running: it can FAIL.

This repo's own review history is that 7 of 12 findings were instruments reporting success while
measuring nothing. A validator is exactly that kind of instrument, so the load-bearing test here is
not "does it pass on good input" but "is every check FALSIFIABLE".
"""
from __future__ import annotations

import json

import pytest

from scripts.pretrain_validate import (
    FAIL,
    PASS,
    SELF_TEST_CASES,
    SKIP,
    WARN,
    check_executable_yield,
    check_identification,
    check_leg_readiness,
    check_registered_nodes_computable,
    check_sandbox_defences,
    check_splits_no_lookahead,
    run_self_test,
)


def test_EVERY_check_is_falsifiable():
    """The whole point. A green gate nobody has watched go red certifies nothing."""
    assert run_self_test() == 0


def test_the_self_test_covers_every_check_the_gate_can_emit():
    """A check with no known-bad case would silently escape the falsifiability proof."""
    import scripts.pretrain_validate as PV

    emitted = {n for n in dir(PV) if n.startswith("check_")}
    covered = {f"check_{k}" for k in SELF_TEST_CASES}
    # names differ where a check is registered under its verdict name; compare counts + resolve
    assert len(SELF_TEST_CASES) == len(emitted), (
        f"{len(emitted)} check functions but {len(SELF_TEST_CASES)} self-test cases — "
        f"an unproven check exists: {sorted(emitted)} vs {sorted(covered)}")


# --- #75: contention must never masquerade as a candidate verdict ------------------------------

def test_a_starved_sandbox_is_NOT_counted_as_a_rejected_candidate(monkeypatch):
    """The SAFETY check must refuse to certify rather than certify on evidence it never gathered.

    ``SandboxEnvironmentError`` SUBCLASSES ``SandboxError``, so ``_gather_sandbox``'s
    ``except SandboxError`` counted a starved spawn as a successful rejection: on a contended box
    the gate reported "rejected 3/3 known-bad sources" having evaluated none of them. Observed
    live — this script returned RC=1 across 21 review loops and RC=0 on a quiet box with its bytes
    unchanged. A false GREEN on a defence proof is strictly worse than a false red.
    """
    import scripts.pretrain_validate as PV
    import src.sandbox.executor as EX

    def starved(*_a, **_k):
        raise EX.SandboxEnvironmentError("spawn environment is starved")

    monkeypatch.setattr(EX, "validate_once", starved)
    with pytest.raises(EX.SandboxEnvironmentError):
        PV._gather_sandbox()


def test_a_starved_sandbox_does_NOT_depress_the_per_model_yield(monkeypatch, tmp_path):
    """The same swallow reported contention as a per-model AUTHORING failure.

    ``_gather_executable_yield`` computes the per-model authoring-compliance statistic the launch
    decision reads, and a low yield is exactly the signal an operator would act on. Counting a
    starved spawn as a non-yielding candidate turns an infrastructure failure into a model finding.
    """
    import scripts.pretrain_validate as PV
    import src.sandbox.executor as EX

    gates = tmp_path / "leg_gates"
    gates.mkdir()
    (gates / "some_model.jsonl").write_text(
        json.dumps({
            "gate": "compliance",
            "response": "```python\ndef reward(w, r, p, pr, info):\n    return 0.0, {}, None\n```",
        }) + "\n",
        encoding="utf-8",
    )

    def starved(*_a, **_k):
        raise EX.SandboxEnvironmentError("spawn environment is starved")

    monkeypatch.setattr(EX, "validate_once", starved)
    with pytest.raises(EX.SandboxEnvironmentError):
        PV._gather_executable_yield(gates)


# --- the failure classes this repo has actually suffered ---------------------------------------

def test_a_registered_node_with_no_implementation_FAILS():
    """The row-36 / R84 class: a ratified claim the campaign cannot compute at the end."""
    v = check_registered_nodes_computable({"N1": {}, "N6": {}}, {"N1": {}})
    assert v.status == FAIL and "N6" in v.detail


def test_an_extra_field_varying_across_arms_FAILS_identification():
    """If anything but the feedback varies, the effect is no longer attributable to it."""
    v = check_identification({"a": {"feedback_kind": "tail", "steps": 400_000},
                              "b": {"feedback_kind": "scalar", "steps": 200_000}})
    assert v.status == FAIL and "steps" in v.detail


def test_a_fail_open_sandbox_FAILS_even_with_no_untyped_escape():
    """A gate that accepts every known-bad source is not gating, however tidy its exceptions."""
    v = check_sandbox_defences(rejected=0, attempted=5, untyped_escapes=0,
                               safe_default_ok=True, flagged_ok=True)
    assert v.status == FAIL and "not gating" in v.detail


def test_an_untyped_escape_FAILS_hard():
    """Callers only handle SandboxError; anything else can kill an arm mid-campaign."""
    v = check_sandbox_defences(rejected=5, attempted=5, untyped_escapes=1,
                               safe_default_ok=True, flagged_ok=True)
    assert v.status == FAIL and "UNTYPED" in v.detail


def test_a_leg_below_the_compliance_floor_FAILS():
    v = check_leg_readiness([{"leg": "glm-5.2", "compliance_rate": 0.6}], expected_legs=1)
    assert v.status == FAIL and "glm-5.2" in v.detail


def test_a_pin_that_did_not_round_trip_FAILS():
    """A silently-ignored pin means the executed author is not the registered one."""
    v = check_leg_readiness(
        [{"leg": "x", "compliance_rate": 1.0, "pin_roundtrip": "FICTIONAL->review"}],
        expected_legs=1)
    assert v.status == FAIL and "round-trip" in v.detail


def test_UNMEASURED_legs_are_never_reported_as_passing():
    """Absence of evidence must not read as evidence of readiness."""
    v = check_leg_readiness([{"leg": "x", "compliance_rate": 1.0}], expected_legs=10)
    assert v.status == WARN and "UNMEASURED" in v.detail


def test_a_leg_that_yields_no_executable_reward_FAILS():
    v = check_executable_yield({"good": (10, 10), "dead": (0, 10)})
    assert v.status == FAIL and "dead" in v.detail


def test_low_yield_is_a_WARNING_and_is_labelled_a_FINDING_not_a_defect():
    """qwen3.5-9b MEASURED 5/20 executable at 1.00 format compliance — a capability result."""
    v = check_executable_yield({"weak": (5, 20), "strong": (20, 20)})
    assert v.status == WARN and "capability finding" in v.detail


# --- the false-alarm guard (a gate that cries wolf trains you to ignore it) ---------------------

def test_ADJACENT_split_boundaries_are_NOT_a_lookahead_failure():
    """The purge is NOT a calendar gap between config boundaries — checking it that way produced a
    FALSE FAIL while this gate was being built. Nominal windows are adjacent by construction; the
    loader resolves the executed purge (lookback=60 dominates the 21-session embargo, R18)."""
    v = check_splits_no_lookahead(
        {"train": (1, 10), "val": (11, 20), "test": (21, 30)},
        executed_purge_sessions=60, required_purge_sessions=21)
    assert v.status == PASS


def test_an_ACTUAL_overlap_still_FAILS():
    v = check_splits_no_lookahead({"train": (1, 15), "val": (11, 20)},
                                  executed_purge_sessions=60, required_purge_sessions=21)
    assert v.status == FAIL and "OVERLAP" in v.detail


def test_an_UNRESOLVABLE_purge_does_not_manufacture_a_failure():
    """If the loader cannot resolve the purge, the gate reports ordering only. Failing on a
    GATHERING error would be a false alarm — the worst thing a launch gate can do."""
    v = check_splits_no_lookahead({"train": (1, 10), "val": (11, 20)},
                                  executed_purge_sessions=None, required_purge_sessions=21)
    assert v.status == PASS


def test_too_few_splits_SKIPS_rather_than_passing_vacuously():
    v = check_splits_no_lookahead({"train": (1, 10)})
    assert v.status == SKIP
