"""Slow integration test for the prototype orchestrator (P5; scripts/run_prototype.py).

Runs ONE arm end-to-end on a synthetic panel at a tiny budget (real SAC training), covering
both the LLM path (loop + stub designer) and the search path (C1 evaluator + reward family).
This locks the orchestration into the suite; the full multi-arm parallel run is the (gated) dry-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_OPTS = dict(
    synthetic=True,
    candidates=2,
    generations=1,
    train_steps=150,
    n_trials=3,
    seed=0,
    pass_mode="A",
    provider="stub",
)


def test_run_arm_distributional_llm(tmp_path) -> None:
    import run_prototype

    s = run_prototype.run_arm("distributional", archive_root=str(tmp_path), **_OPTS)
    assert s["arm"] == "distributional"
    assert s["n_candidates"] == 2
    assert s["matched_budget_ok"] is True
    assert (Path(tmp_path) / "distributional" / "COMPLETE").exists()


def test_run_arm_random_search(tmp_path) -> None:
    import run_prototype

    s = run_prototype.run_arm("random_search", archive_root=str(tmp_path), **_OPTS)
    assert s["arm"] == "random_search"
    assert s["n_candidates"] == 2
    assert s["matched_budget_ok"] is True


def test_run_arm_bayes_opt(tmp_path) -> None:
    import run_prototype

    s = run_prototype.run_arm("bayes_opt", archive_root=str(tmp_path), **_OPTS)
    assert s["arm"] == "bayes_opt"
    assert s["n_candidates"] == 2
    assert s["matched_budget_ok"] is True
