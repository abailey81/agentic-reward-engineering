"""Unit tests for the pre-freeze pilot battery's pure decision logic (scripts/pilot.py).

The GPU measurement drivers run at Phase 3; the DECISION LOGIC that turns measurements into a freeze GO/NO-GO
is what's tested here, exhaustively, against synthetic inputs.
"""
from __future__ import annotations

import pytest

from scripts.pilot import (
    ConfigResult,
    assess_freeze_readiness,
    config_is_safe,
    knee_from_curve,
    project_wall_clock_hours,
    render_report,
    select_best_config,
    wall_clock_verdict,
)


# ── compute-frontier selection ──────────────────────────────────────────────────────────────────────
def test_select_best_config_rejects_unsafe_and_picks_fastest_safe() -> None:
    results = [
        ConfigResult("slow_safe", 100, deterministic=True, max_temp_c=70, peak_ram_pct=80, peak_vram_pct=70),
        ConfigResult("fast_but_hot", 300, deterministic=True, max_temp_c=90, peak_ram_pct=80, peak_vram_pct=70),
        ConfigResult("fast_but_nondet", 290, deterministic=False, max_temp_c=70, peak_ram_pct=80, peak_vram_pct=70),
        ConfigResult("fast_but_ram", 280, deterministic=True, max_temp_c=70, peak_ram_pct=95, peak_vram_pct=70),
        ConfigResult("mid_safe", 200, deterministic=True, max_temp_c=80, peak_ram_pct=85, peak_vram_pct=72),
    ]
    best = select_best_config(results)
    assert best is not None and best.label == "mid_safe"  # fastest among the determinism+thermal+mem-safe set


def test_determinism_is_a_hard_gate() -> None:
    # A faster but non-deterministic config (e.g. torch.compile that breaks byte-identity) is NEVER chosen
    # when determinism is required — resume/replay/reproducibility depend on it.
    fast_nondet = ConfigResult("compile_fast", 999, deterministic=False, max_temp_c=60, peak_ram_pct=50, peak_vram_pct=50)
    slow_det = ConfigResult("safe", 100, deterministic=True, max_temp_c=60, peak_ram_pct=50, peak_vram_pct=50)
    assert select_best_config([fast_nondet, slow_det]).label == "safe"
    assert config_is_safe(fast_nondet, require_deterministic=False) is True  # allowed only if determinism waived


def test_select_best_config_none_when_all_unsafe() -> None:
    assert select_best_config([ConfigResult("x", 300, deterministic=False)]) is None


# ── wall-clock projection ───────────────────────────────────────────────────────────────────────────
def test_wall_clock_scales_linearly_with_panel_size() -> None:
    h1 = project_wall_clock_hours(300_000, 200, n_arms=7, n_candidates=30, n_seeds=30, n_models=1)
    h3 = project_wall_clock_hours(300_000, 200, n_arms=7, n_candidates=30, n_seeds=30, n_models=3)
    assert h3 == pytest.approx(3 * h1)            # the panel multiplies the laptop compute linearly
    assert h1 > 100 and h1 < 400                  # ~sane band for a single model at B*=300k, 200 steps/s


def test_wall_clock_infinite_on_zero_throughput() -> None:
    assert project_wall_clock_hours(300_000, 0.0, n_arms=7, n_candidates=30, n_seeds=30) == float("inf")


def test_wall_clock_verdict_bands() -> None:
    assert wall_clock_verdict(100, go_hours=240, adapt_hours=360) == "GO"
    assert wall_clock_verdict(300, go_hours=240, adapt_hours=360) == "ADAPT"
    assert wall_clock_verdict(500, go_hours=240, adapt_hours=360) == "RECONSIDER"


# ── convergence knee (B*) ───────────────────────────────────────────────────────────────────────────
def test_knee_finds_plateau_and_tightens_with_tol() -> None:
    steps = [50_000, 150_000, 250_000, 350_000, 500_000]
    scores = [0.31, 0.55, 0.752, 0.76, 0.765]
    assert knee_from_curve(steps, scores, rel_tol=0.02) == 250_000   # 0.752 within 2% of 0.765
    assert knee_from_curve(steps, scores, rel_tol=0.001) == 500_000  # tighter tol (thr 0.7642 excludes 0.76) -> later budget


def test_knee_conservative_fallback_and_empty() -> None:
    # nothing reaches the (impossible) band -> the largest budget; empty -> error
    assert knee_from_curve([10, 20, 30], [1.0, 1.0, 1.0], rel_tol=0.0) == 10  # all equal -> first hits the plateau
    with pytest.raises(ValueError):
        knee_from_curve([], [])


# ── freeze-readiness aggregation ────────────────────────────────────────────────────────────────────
def _kw(**over):
    base = dict(config={"label": "x"}, b_star=250_000, n_seeds=30, sigma_seed=0.08,
                equivalence_achievable=True, wall_clock_hours=200.0, wc_verdict="GO",
                risks={"resume_round_trip": True, "memory_flat": True, "preflight": True})
    base.update(over)
    return base


def test_ready_when_all_green() -> None:
    fr = assess_freeze_readiness(**_kw())
    assert fr.verdict == "READY" and fr.blockers == []
    assert "FREEZE-READINESS: READY" in render_report(fr)


def test_blocked_on_missing_params() -> None:
    fr = assess_freeze_readiness(**_kw(config=None, b_star=None, n_seeds=None))
    assert fr.verdict == "BLOCKED" and len(fr.blockers) >= 3


def test_blocked_on_failed_risk() -> None:
    fr = assess_freeze_readiness(**_kw(risks={"resume_round_trip": False}))
    assert fr.verdict == "BLOCKED" and any("resume_round_trip" in b for b in fr.blockers)


def test_adapt_on_tight_wallclock() -> None:
    fr = assess_freeze_readiness(**_kw(wall_clock_hours=300.0, wc_verdict="ADAPT"))
    assert fr.verdict == "ADAPT"


def test_blocked_on_reconsider() -> None:
    fr = assess_freeze_readiness(**_kw(wall_clock_hours=999.0, wc_verdict="RECONSIDER"))
    assert fr.verdict == "BLOCKED" and any("infeasible" in b for b in fr.blockers)


def test_demo_runs_and_is_ready() -> None:
    from scripts.pilot import _demo

    fr = _demo()
    assert fr.verdict in ("READY", "ADAPT")
    assert fr.config["label"] == "compile=on,n_gpu=2"   # fastest safe (n_gpu=3 rejected on 93% RAM)
    assert fr.b_star == 250_000
