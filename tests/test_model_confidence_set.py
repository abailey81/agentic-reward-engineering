"""Behaviour tests for the report-only Model Confidence Set over the arms (R69).

Pins: the best arm is in the set with the maximal elimination p-value; a clearly-dominated arm is excluded
under strong separation; the result is byte-deterministic given the seed; the loss sign is honoured; and
malformed input fails loudly. Uses ``arch`` (an existing dep); import-light (no torch).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.model_confidence_set import model_confidence_set


def _arm_scores(means: dict[str, float], sd: float, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {a: rng.normal(m, sd, n) for a, m in means.items()}


def test_best_arm_in_set_with_max_pvalue() -> None:
    sc = _arm_scores({"a": 0.4, "b": 0.0, "c": 0.0, "d": 0.0}, sd=1.0, n=30)
    res = model_confidence_set(sc, size=0.10, reps=500, seed=7)
    assert res["status"] == "ok"
    assert set(res["included"]) | set(res["excluded"]) == {"a", "b", "c", "d"}
    assert res["best_arm"] == "a" and res["best_in_set"] is True
    # the best arm is the least eliminable -> its MCS p-value is the maximum (1.0 by construction)
    assert res["pvalues"]["a"] == max(res["pvalues"].values())


def test_strong_separation_excludes_dominated_arms() -> None:
    """With an overwhelming separation, the clearly-worse arms leave the confidence set."""
    sc = _arm_scores({"good": 10.0, "bad1": 0.0, "bad2": 0.0}, sd=0.3, n=50)
    res = model_confidence_set(sc, size=0.10, reps=800, seed=1)
    assert res["best_arm"] == "good" and "good" in res["included"]
    assert len(res["excluded"]) >= 1  # at least one dominated arm eliminated


def test_deterministic_under_fixed_seed() -> None:
    sc = _arm_scores({"a": 0.2, "b": 0.0, "c": -0.1}, sd=1.0, n=30)
    a = model_confidence_set(sc, size=0.10, reps=400, seed=3)
    b = model_confidence_set(sc, size=0.10, reps=400, seed=3)
    assert a == b


def test_lower_is_better_flips_best() -> None:
    """When lower scores are better (a loss/error metric), the best arm is the minimum-mean one."""
    sc = _arm_scores({"lo": -5.0, "hi": 5.0, "mid": 0.0}, sd=0.5, n=40)
    res = model_confidence_set(sc, size=0.10, reps=500, seed=2, higher_is_better=False)
    assert res["best_arm"] == "lo" and "lo" in res["included"]


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        model_confidence_set({"only": np.zeros(30)})  # < 2 arms
    with pytest.raises(ValueError):
        model_confidence_set({"a": np.zeros(30), "b": np.zeros(20)})  # mismatched lengths
    with pytest.raises(ValueError):
        model_confidence_set({"a": np.zeros(1), "b": np.zeros(1)})  # < 2 seeds


def test_non_finite_scores_raise_a_clear_error_naming_the_arm() -> None:
    """#74: the FOURTH precondition. The three above are validated with a clear message; a non-finite
    per-seed score was not, and it does not degrade gracefully — it reached ``arch.bootstrap.MCS`` and
    surfaced as ``IndexError: index 0 is out of bounds for axis 0 with size 0`` from inside the
    dependency (MEASURED). ``analyze_campaign`` wraps this block in a broad ``except`` that stores
    ``str(exc)`` as the block's reason, so that opaque message would be all an operator got for a whole
    registered analysis.
    """
    rng = np.random.default_rng(0)
    good = {a: 0.5 + rng.normal(0, 0.1, 20) for a in ("distributional", "scalar", "placebo")}

    for bad_value in (np.nan, np.inf, -np.inf):
        scores = {a: v.copy() for a, v in good.items()}
        scores["scalar"][7] = bad_value
        with pytest.raises(ValueError, match=r"non-finite.*scalar"):
            model_confidence_set(scores, size=0.10, reps=200, seed=1)

    # The guard must not over-trigger: all-finite input still computes (verify the LEGITIMATE case).
    res = model_confidence_set(good, size=0.10, reps=200, seed=1)
    assert res["status"] == "ok"
    assert set(res["included"]) | set(res["excluded"]) == set(good)
