"""Behaviour tests for regime-stratified metrics + the T3' table (src/inference/regime_analysis.py).

Deterministic. Checks: correct per-regime stratification + episode counting (the power bound), the
underpowered flag + NaN metrics on thin regimes, the Markdown table render, and degrade paths.
"""

from __future__ import annotations

import numpy as np

from src.inference.regime_analysis import (
    cvar,
    regime_stratified_metrics,
    render_regime_table,
)

SEED = 20260701


def test_cvar_is_mean_of_worst_tail() -> None:
    r = np.array([-0.10, -0.05, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    # worst 20% of 10 obs = 2 obs: -0.10, -0.05 -> mean -0.075
    assert cvar(r, 0.20) == np.float64(-0.075).item() or abs(cvar(r, 0.20) - (-0.075)) < 1e-12
    assert cvar(np.array([]), 0.05) != cvar(np.array([]), 0.05)  # NaN for empty


def test_stratifies_and_counts_episodes() -> None:
    # labels: calm(0) block, stress(2) block, calm(0) block again -> calm has 2 episodes, stress 1.
    labels = np.array([0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0])
    rng = np.random.default_rng(SEED)
    returns_by_arm = {
        "distributional": rng.normal(0.001, 0.01, labels.size),
        "placebo": rng.normal(0.001, 0.01, labels.size),
    }
    res = regime_stratified_metrics(returns_by_arm, labels, min_obs=2)
    assert res["status"] == "ok"
    by_name = {r["name"]: r for r in res["regimes"]}
    assert by_name["calm"]["n_episodes"] == 2     # two contiguous calm blocks
    assert by_name["stress"]["n_episodes"] == 1
    assert by_name["calm"]["n_dates"] == 10
    assert by_name["stress"]["n_dates"] == 2
    assert res["total_episodes"] == 3             # calm, stress, calm
    # metrics present for both arms in each regime
    assert set(by_name["calm"]["per_arm"]) == {"distributional", "placebo"}
    assert "cvar_05" in by_name["calm"]["per_arm"]["distributional"]


def test_thin_regime_is_flagged_underpowered_with_nan_metrics() -> None:
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2])  # stress has only 2 dates
    returns = {"distributional": np.linspace(-0.02, 0.02, labels.size)}
    res = regime_stratified_metrics(returns, labels, min_obs=5)
    stress = next(r for r in res["regimes"] if r["name"] == "stress")
    assert stress["underpowered"] is True
    assert stress["per_arm"]["distributional"]["cvar_05"] != stress["per_arm"]["distributional"]["cvar_05"]  # NaN


def test_render_regime_table_is_markdown_with_power_note() -> None:
    labels = np.array([0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    returns = {"distributional": np.linspace(-0.03, 0.03, 10), "scalar": np.linspace(-0.02, 0.02, 10)}
    res = regime_stratified_metrics(returns, labels, min_obs=3)
    table = render_regime_table(res)
    assert table.startswith("| regime | n | episodes | arm |")
    assert "calm" in table and "stress" in table
    assert "distributional" in table and "scalar" in table
    assert "episodes" in table and "power" in table.lower()


def test_degrade_paths() -> None:
    assert regime_stratified_metrics({}, np.array([0, 1]))["status"] == "no_data"
    # mismatched lengths
    bad = regime_stratified_metrics({"a": np.zeros(5)}, np.array([0, 1, 2]))
    assert bad["status"] == "no_data"
    assert render_regime_table({"status": "no_data", "reason": "x"}).startswith("_regime table unavailable")
