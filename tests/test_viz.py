"""Behaviour tests for the report-only figure engine (src/viz + scripts/make_figures).

Headless (Agg): every figure function must return a non-empty matplotlib Figure on synthetic data, the
style helpers must be deterministic + cover all 9 arms, and the demo script must render all fourteen figures
(PNG + PDF) to disk. No display, no network; fast (no torch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.viz import figures as F  # noqa: E402
from src.viz import style as S  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_figures as MF  # noqa: E402


# ---- style helpers ---------------------------------------------------------------------------------- #
def test_arm_style_covers_all_arms_with_distinct_colors() -> None:
    colors = [S.arm_style(a)["color"] for a in S.ARM_ORDER]
    assert len(S.ARM_ORDER) == 9
    assert len(set(colors)) == 9  # every arm a distinct colour (palette has exactly 9; tpe takes black)
    assert S.CONTROL_ARMS <= set(S.ARM_ORDER)
    for ctrl in S.CONTROL_ARMS:
        assert S.arm_style(ctrl)["hatch"]  # controls carry a non-colour (hatch) channel
    # unknown arm -> neutral default, never raises
    assert S.arm_style("mystery")["color"] == S.OKABE_ITO["black"]


def test_iqm_and_bootstrap_ci_deterministic_and_ordered() -> None:
    x = np.linspace(0, 1, 40)
    assert abs(S.iqm(x) - 0.5) < 1e-9  # symmetric -> IQM = mean = 0.5
    a = S.iqm_bootstrap_ci(x, seed=3)
    b = S.iqm_bootstrap_ci(x, seed=3)
    assert a == b  # deterministic under a fixed seed
    point, lo, hi = a
    assert lo <= point <= hi
    # singleton -> degenerate CI, no crash
    assert S.iqm_bootstrap_ci(np.array([0.7])) == (0.7, 0.7, 0.7)


def test_iqm_bootstrap_ci_counts_finite_points_not_raw_slots() -> None:
    """#69: the too-few-points guard must count USABLE points, or it fails on the very case
    ``iqm``'s non-finite filtering exists for.

    Before the fix, ``[1.0, nan]`` returned ``(1.0, nan, nan)``: raw size 2 cleared the guard, then every
    resample of a 1-finite array could draw only NaNs and ``np.quantile`` propagated it. Matplotlib draws
    NO error bar for a NaN interval, so in the headline ``F_rliable_intervals`` panel an arm whose seeds
    were mostly lost rendered as a bare point -- visually the most PRECISE arm on the chart.
    """
    for contaminated in (
        np.array([1.0, np.nan]),
        np.array([1.0, np.nan, np.nan]),
        np.array([1.0, np.inf]),          # inf is non-finite too, not just NaN
    ):
        assert S.iqm_bootstrap_ci(contaminated) == (1.0, 1.0, 1.0)

    # All non-finite -> NaN throughout (nothing to report), and no crash.
    assert all(np.isnan(v) for v in S.iqm_bootstrap_ci(np.array([np.nan, np.nan])))

    # Enough finite points -> a REAL interval, never the degenerate collapse (the fix must not
    # over-trigger: verify the legitimate case, not only the misfire).
    point, lo, hi = S.iqm_bootstrap_ci(np.concatenate([np.linspace(0, 1, 12), [np.nan, np.nan]]), seed=1)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    assert lo <= point <= hi


def test_arm_style_returns_a_copy_so_a_caller_cannot_repaint_the_suite() -> None:
    """#70: the known-arm branch handed back the module-global dict while the unknown-arm branch
    built a fresh one, so a caller could not tell whether mutating the result was safe."""
    original = S.ARM_STYLE["scalar"]["color"]
    got = S.arm_style("scalar")
    assert got is not S.ARM_STYLE["scalar"]
    got["color"] = "#FF00FF"
    assert S.ARM_STYLE["scalar"]["color"] == original, "mutating the result repainted the global style"
    assert S.arm_style("scalar")["color"] == original
    # the values themselves are unchanged by the copy
    assert S.arm_style("scalar") == {"color": original, "marker": "s", "hatch": None}


def test_equivalence_band_draws_span_and_zero_line() -> None:
    fig, ax = plt.subplots()
    S.equivalence_band(ax, 0.05, orient="v")
    assert len(ax.patches) >= 1  # the shaded corridor
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())  # the zero line
    plt.close(fig)


# ---- figure functions ------------------------------------------------------------------------------- #
@pytest.fixture
def demo() -> dict:
    return MF.synthesize_null(seed=1, n_seeds=20)


def test_equivalence_forest_returns_two_panel_figure(demo: dict) -> None:
    fig = F.equivalence_forest(demo["contrasts"])
    assert len(fig.axes) == 2  # Sharpe + CVaR legs
    plt.close(fig)


def test_equivalence_forest_band_is_per_leg(demo: dict) -> None:
    """The co-primary legs are in DIFFERENT units, so the band must be settable PER LEG (2026-07-26).

    ``analyze_campaign.h2_tost`` is "RA-only by construction; the CVaR equivalence stays in h2_tost
    (its own units)", and its P6 note warns the raw ±0.05 DSR margin is LARGE against a daily CVaR
    O(0.01-0.06) so it "can near-trivially contain the posterior and OVER-CLAIM null evidence" — hence
    the relative tail band (25% x |baseline CVaR|). A single scalar drew the CVaR rows FILLED against a
    far-too-wide corridor, visually over-claiming the null on the bankable-null headline leg.
    """
    # A mapping is accepted, and a scalar still works (back-compat for single-leg callers).
    fig = F.equivalence_forest(demo["contrasts"], sesoi={"sharpe": 0.05, "cvar": 0.01})
    assert len(fig.axes) == 2
    plt.close(fig)
    fig = F.equivalence_forest(demo["contrasts"], sesoi=0.05)
    assert len(fig.axes) == 2
    plt.close(fig)
    # THE POINT: the same tail interval flips verdict between the wrong band and the P6 band.
    assert F._is_equivalent(-0.02, 0.012, 0.05) is True       # raw DSR margin: too wide -> over-claims
    assert F._is_equivalent(-0.02, 0.012, 0.25 * 0.04) is False  # P6 relative band: honest verdict


def test_rliable_intervals_one_axis_per_leg(demo: dict) -> None:
    fig = F.rliable_intervals(demo["scores_by_leg"])
    assert len(fig.axes) == 2
    plt.close(fig)


def test_risk_return_clouds_has_all_arms_in_legend(demo: dict) -> None:
    fig = F.risk_return_clouds(demo["sharpe"], demo["cvar"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(labels) == set(S.ARM_ORDER)
    plt.close(fig)


def test_evidence_for_null_with_and_without_mcs(demo: dict) -> None:
    fig = F.evidence_for_null(demo["bf01_by_leg"], demo["mcs"])
    assert len(fig.axes) >= 2  # BF gauge + MCS strip (+ any colorbar)
    plt.close(fig)
    fig2 = F.evidence_for_null(demo["bf01_by_leg"], None)  # no MCS -> single panel, no crash
    assert len(fig2.axes) >= 1
    plt.close(fig2)


def test_reward_code_similarity_renders(demo: dict) -> None:
    fig = F.reward_code_similarity(demo["ast_distance"], demo["cand_arms"])
    assert len(fig.axes) >= 3  # dendrogram + sidebar + heatmap (+ colorbar)
    plt.close(fig)


def test_reward_code_similarity_handles_tiny_matrix() -> None:
    d = np.array([[0.0, 0.4], [0.4, 0.0]])
    fig = F.reward_code_similarity(d, ["distributional", "scalar"])
    assert fig is not None
    plt.close(fig)


def test_controls_overlay_one_row_per_arm(demo: dict) -> None:
    fig = F.controls_overlay(demo["controls_cvar"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == set(demo["controls_cvar"])  # one labelled row per arm
    plt.close(fig)


def test_controls_overlay_is_deterministic(demo: dict) -> None:
    # the strip jitter is seeded internally -> identical artists across calls
    fig_a = F.controls_overlay(demo["controls_cvar"])
    fig_b = F.controls_overlay(demo["controls_cvar"])
    xa = np.concatenate([c.get_offsets()[:, 0] for c in fig_a.axes[0].collections])
    xb = np.concatenate([c.get_offsets()[:, 0] for c in fig_b.axes[0].collections])
    assert np.allclose(xa, xb)
    plt.close(fig_a)
    plt.close(fig_b)


def test_responsiveness_scatter_renders_with_rho(demo: dict) -> None:
    fig = F.responsiveness_scatter(demo["fed_delta"], demo["reward_delta"], rho=demo["rho"])
    ax = fig.axes[0]
    assert any("ρ" in t.get_text() for t in ax.texts)  # the reported Spearman annotation
    plt.close(fig)
    # degenerate input (all-equal x) must not raise (no fit line drawn)
    fig2 = F.responsiveness_scatter(np.zeros(5), np.arange(5.0))
    assert fig2 is not None
    plt.close(fig2)


def test_delisting_robustness_renders_band_and_series(demo: dict) -> None:
    fig = F.delisting_robustness(demo["delisting"], sesoi=0.05)
    ax = fig.axes[0]
    assert len(ax.patches) >= 1  # the ±SESOI corridor
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0%", "-30%", "-55%", "-100%"]
    leg = {t.get_text() for t in ax.get_legend().get_texts()}
    assert any("dist - placebo" in t for t in leg)
    plt.close(fig)


def test_delisting_robustness_flags_an_excursion() -> None:
    out_of_band = {"dist - x": {"0%": {"estimate": 0.0, "ci_lo": -0.02, "ci_hi": 0.02},
                                "-100%": {"estimate": 0.09, "ci_lo": 0.06, "ci_hi": 0.12}}}
    fig = F.delisting_robustness(out_of_band, treatments=["0%", "-100%"], sesoi=0.05)
    leg = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert any("excursion" in t for t in leg)  # the out-of-band series is marked
    plt.close(fig)


def test_learning_curves_two_panels_all_arms(demo: dict) -> None:
    fig = F.learning_curves(demo["curves"])
    assert len(fig.axes) == 2  # critic-loss + return panels
    leg = fig.axes[0].get_legend()
    assert set(t.get_text() for t in leg.get_texts()) == set(demo["curves"])
    assert fig.axes[0].get_yscale() == "log"  # critic loss on a log axis
    plt.close(fig)


# ---- the demo script end to end --------------------------------------------------------------------- #
def test_make_figures_demo_writes_png_and_pdf(tmp_path: Path) -> None:
    data = MF.synthesize_null(seed=2, n_seeds=15)
    saved = MF.render_all(data, tmp_path)
    assert len(saved) == 15  # 9 original headline + G1-G5 (2026-07-26) + F_seed_trajectory (D2, row 38)
    assert any(p.name == "F_seed_trajectory.png" for p in saved), (
        "the D2 seed-trajectory exhibit (registry row 38) must be RENDERED by the suite, not merely "
        "implemented — a function no renderer calls is half a deliverable"
    )
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
        assert p.with_suffix(".pdf").exists()  # vector sibling


def test_synthesize_null_is_deterministic() -> None:
    a = MF.synthesize_null(seed=5, n_seeds=12)
    b = MF.synthesize_null(seed=5, n_seeds=12)
    assert np.allclose(a["sharpe"]["distributional"], b["sharpe"]["distributional"])
    assert np.allclose(a["ast_distance"], b["ast_distance"])


# ---- advanced 3D + animation suite ------------------------------------------------------------------ #
from src.viz import advanced as ADV  # noqa: E402


def test_classical_mds_recovers_euclidean_geometry() -> None:
    # points on a known 3D line -> pairwise distances -> MDS should recover collinear coords (1 real axis).
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    coords = ADV.classical_mds(d, 3)
    assert coords.shape == (4, 3)
    # the spread should live on the first axis (the others ~ 0)
    assert coords[:, 0].std() > 1e-6
    assert coords[:, 1].std() < 1e-6 and coords[:, 2].std() < 1e-6


def test_reward_embedding_3d_has_three_axes_and_all_arms(demo: dict) -> None:
    fig = ADV.reward_embedding_3d(demo["ast_distance"], demo["cand_arms"])
    ax = fig.axes[0]
    assert hasattr(ax, "get_zlim")  # a genuine 3D axis
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == set(S.ARM_ORDER)
    plt.close(fig)


def test_risk_return_generation_3d_renders(demo: dict) -> None:
    fig = ADV.risk_return_generation_3d(demo["trajectory"])
    assert hasattr(fig.axes[0], "get_zlim")
    plt.close(fig)


def test_search_evolution_keyframes_panel_count(demo: dict) -> None:
    fig = ADV.search_evolution_keyframes(demo["frames"], max_panels=3)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_render_advanced_writes_png_and_pdf(tmp_path: Path) -> None:
    data = MF.synthesize_null(seed=3, n_seeds=12)
    saved = MF.render_advanced(data, tmp_path)
    assert len(saved) == 3
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
        assert p.with_suffix(".pdf").exists()


def test_schematics_render() -> None:
    from src.viz import schematics as SC

    for fn in (SC.system_diagram, SC.prediction_branch, SC.splits_timeline):
        fig = fn()
        assert fig.axes and len(fig.axes[0].texts) >= 3  # labelled boxes/annotations present
        plt.close(fig)


def test_render_schematics_writes_png_and_pdf(tmp_path: Path) -> None:
    saved = MF.render_schematics(tmp_path)
    assert len(saved) == 3
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
        assert p.with_suffix(".pdf").exists()


def test_render_animations_writes_gifs(tmp_path: Path) -> None:
    pytest.importorskip("PIL")  # GIF writer needs Pillow
    data = MF.synthesize_null(seed=3, n_seeds=10)
    gifs = MF.render_animations(data, tmp_path, n_frames=6)  # few frames -> fast
    assert len(gifs) == 2
    for p in gifs:
        assert p.exists() and p.suffix == ".gif" and p.stat().st_size > 0


# --------------------------------------------------------------------------- #
# D2 / registry row 38 — the seed-trajectory exhibit                           #
# --------------------------------------------------------------------------- #
def test_seed_trajectory_renders_and_endpoint_matches_headline_estimator() -> None:
    """The curve must END at exactly the headline IQM — same estimator, not a lookalike."""
    from src.viz.style import iqm_bootstrap_ci

    rng = np.random.default_rng(0)
    vals = rng.normal(0.2, 0.5, size=40)
    fig = F.seed_trajectory({"distributional": vals}, rungs=(30,), terminal_rung=30)
    assert fig.axes, "no axes drawn"
    line = fig.axes[0].lines[0]
    terminal_y = float(line.get_ydata()[-1])
    expected, _, _ = iqm_bootstrap_ci(vals, reps=2000, alpha=0.05, seed=0)
    assert terminal_y == pytest.approx(expected), (
        "trajectory endpoint must equal the headline IQM by construction"
    )
    plt.close(fig)


def test_seed_trajectory_refuses_permuted_seed_order() -> None:
    """D2 property (1): a series re-ordered by outcome must FAIL LOUD, not render."""
    vals = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="REGISTERED order"):
        F.seed_trajectory({"scalar": vals}, seeds=[0, 3, 1, 2])
    fig = F.seed_trajectory({"scalar": vals}, seeds=[0, 1, 2, 3])  # registered order is accepted
    plt.close(fig)


def test_seed_trajectory_refuses_empty_input() -> None:
    """An empty exhibit would satisfy the registry row while showing the reader nothing."""
    with pytest.raises(ValueError):
        F.seed_trajectory({})
    with pytest.raises(ValueError):
        F.seed_trajectory({"a": []})


def test_seed_trajectory_always_states_no_inference_at_any_prefix() -> None:
    """D2 property (3): the disclaimer is not a parameter and cannot be switched off."""
    fig = F.seed_trajectory({"placebo": [0.1, 0.2, 0.3]})
    blob = " ".join(t.get_text() for t in fig.texts)
    assert "NO INFERENCE WAS DRAWN AT ANY PREFIX" in blob
    assert "Stopping rule" in blob
    plt.close(fig)


def test_seed_trajectory_grid_keeps_small_n_dense_rungs_and_terminal() -> None:
    """Okhrati asked for 1,2,3,…; the grid must not stride away the small-n region or a rung."""
    g = F._trajectory_grid(568, rungs=(30, 100, 189, 279, 340, 403, 568))
    assert g[:9] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for r in (30, 100, 189, 279, 340, 403, 568):
        assert r in g, f"rung {r} missing from the evaluation grid"
    assert g[-1] == 568 and len(g) <= 200
    assert F._trajectory_grid(0) == []


def test_seed_trajectory_survives_n_equals_one_and_nans() -> None:
    """n=1 has no dispersion (degenerate triple) and NaN seeds must not blank the band."""
    fig = F.seed_trajectory({"haiku": [0.5, float("nan"), 0.7, 0.9]})
    ax = fig.axes[0]
    assert np.isfinite(np.asarray(ax.lines[0].get_ydata(), dtype=float)).all()
    plt.close(fig)
