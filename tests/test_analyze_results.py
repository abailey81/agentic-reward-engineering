"""Tests for the prototype analysis (P6; scripts/analyze_results.py).

FAST: builds fake archives with known fitnesses / return series / reward sources (no training),
then checks the directional verdict logic, the interpretability gate, and the difference tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _write_arm(root, arm, fitnesses, winner_returns, source) -> None:
    from src.io.results import write_run

    arm_root = Path(root) / arm
    best = max(fitnesses)
    for i, f in enumerate(fitnesses):
        rec = {
            "run_id": f"{arm}-c{i}",
            "arm": arm,
            "seed": 0,
            "fold": 0,
            "candidate_id": f"{arm}-c{i}",
            "generation": 0,
            "reward_source": source,
            "reward_source_hash": f"h-{arm}-{i}",
            "feedback_block": "",
            "wall_clock": 0.0,
            "env_fingerprint": "test",
            "metrics": {
                "val_fitness": float(f),
                "val_returns": list(map(float, winner_returns)) if f == best else None,
            },
        }
        write_run(rec, arm_root)


def test_interpretability_detects_tail_terms() -> None:
    import analyze_results as ar

    assert ar.interpretability("x = np.sort(returns); cvar = x[:5].mean()")["uses_tail"] is True
    assert ar.interpretability("def reward(...):\n    return port_ret, {}, None")["uses_tail"] is False


def test_directional_verdict_logic() -> None:
    import analyze_results as ar

    assert ar.directional_verdict({"distributional": 0.8, "scalar": 0.4, "placebo": 0.3}, True) == "GREEN"
    assert ar.directional_verdict({"distributional": 0.3, "scalar": 0.5, "placebo": 0.4}, False) == "RED"
    assert ar.directional_verdict({"distributional": 0.8, "scalar": 0.4, "placebo": 0.3}, False) == "AMBER"
    assert ar.directional_verdict({"distributional": 0.5}, True) == "INCOMPLETE"


def test_analyze_green(tmp_path) -> None:
    import analyze_results as ar

    rng = np.random.default_rng(0)
    _write_arm(
        tmp_path, "distributional", [0.5, 0.8, 0.7], 0.002 + 0.01 * rng.standard_normal(64),
        "def reward(w, r, pw, pr, info):\n    cvar = np.sort(r)[:3].mean()\n    return pr - cvar, {}, None",
    )
    _write_arm(tmp_path, "scalar", [0.3, 0.4, 0.35], 0.001 + 0.01 * rng.standard_normal(64),
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")
    _write_arm(tmp_path, "placebo", [0.2, 0.25, 0.22], 0.0 + 0.01 * rng.standard_normal(64),
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")

    result = ar.analyze(tmp_path)
    assert result["verdict"] == "GREEN"
    assert result["interpretability"]["uses_tail"] is True
    assert "distributional_vs_scalar" in result["difference_tests"]
    assert "pvalue" in result["difference_tests"]["distributional_vs_scalar"]["sharpe"]
    report = ar.write_report(result, tmp_path)
    assert report.exists()
    assert "Verdict: GREEN" in report.read_text(encoding="utf-8")


def test_analyze_red(tmp_path) -> None:
    import analyze_results as ar

    base = list(0.01 * np.random.default_rng(1).standard_normal(64))
    _write_arm(tmp_path, "distributional", [0.30, 0.31], base,
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")
    _write_arm(tmp_path, "scalar", [0.32, 0.33], base,
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")
    _write_arm(tmp_path, "placebo", [0.34, 0.35], base,
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")
    result = ar.analyze(tmp_path)
    assert result["verdict"] == "RED"


def test_analyze_emits_performance_profiles(tmp_path) -> None:
    """rliable performance profiles are computed per-arm over a shared tau grid, monotone non-increasing,
    in [0,1], and bracketed by their bootstrap band."""
    import analyze_results as ar

    rng = np.random.default_rng(2)
    ret = 0.001 + 0.01 * rng.standard_normal(64)
    _write_arm(tmp_path, "distributional", [0.5, 0.8, 0.7, 0.6], ret,
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")
    _write_arm(tmp_path, "scalar", [0.2, 0.3, 0.25, 0.28], ret,
               "def reward(w, r, pw, pr, info):\n    return pr, {}, None")

    result = ar.analyze(tmp_path)
    prof = result["performance_profiles"]
    assert "taus" in prof and len(prof["taus"]) == 50
    for arm in ("distributional", "scalar"):
        p = np.asarray(prof[arm]["profile"])
        lo = np.asarray(prof[arm]["ci_low"])
        hi = np.asarray(prof[arm]["ci_high"])
        assert p.shape == (50,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(np.diff(p) <= 1e-12)  # survival function: non-increasing in tau
        assert np.all((lo <= p + 1e-9) & (hi >= p - 1e-9))
