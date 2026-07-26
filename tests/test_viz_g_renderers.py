"""Tests for the corpus-standard figure additions (2026-07-26): G1-G5.

G1 performance_profile · G2 probability_of_improvement (the other two rliable-quartet members) ·
G3 return_tail_distribution · G4 equity_drawdown · G5 allocation_heatmap. Headless (Agg); each asserts the
renderer returns a structurally valid, non-empty figure on already-computed inputs.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.inference.exposure import alloc_snapshots
from src.viz.figures import (
    allocation_heatmap,
    equity_drawdown,
    performance_profile,
    probability_of_improvement,
    return_tail_distribution,
)

_ARMS = ("distributional", "scalar", "placebo")


def _scores():
    rng = np.random.default_rng(0)
    return {a: rng.normal(0.0, 1.0, size=30) for a in _ARMS}


def test_performance_profile_structure():
    fig = performance_profile(_scores())
    ax = fig.axes[0]
    # one line per arm; all fractions within [0, 1]
    assert len(ax.lines) == len(_ARMS)
    for ln in ax.lines:
        y = ln.get_ydata()
        assert y.min() >= -1e-9 and y.max() <= 1.0 + 1e-9
    lo, hi = ax.get_ylim()
    assert lo <= 0.0 and hi >= 1.0


def test_performance_profile_empty_scores_no_raise():
    fig = performance_profile({"scalar": np.array([])})
    assert fig.axes  # renders (flat-zero profile), never raises


def test_probability_of_improvement_accepts_scalar_and_ci():
    fig = probability_of_improvement({"distributional": (0.55, 0.40, 0.70), "scalar": 0.50})
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 1.0)
    # the 0.5 no-effect reference line is present
    assert any(abs(ln.get_xdata()[0] - 0.5) < 1e-9 for ln in ax.lines if len(ln.get_xdata()) == 2
               and ln.get_xdata()[0] == ln.get_xdata()[1])


def test_return_tail_distribution_marks_alpha():
    rng = np.random.default_rng(1)
    r = {a: rng.normal(0.0004, 0.01, size=500) for a in _ARMS}
    fig = return_tail_distribution(r, alpha=0.05)
    ax = fig.axes[0]
    assert "α=0.05" in ax.get_title(loc="left")
    assert len(ax.lines) >= len(_ARMS)  # an ECDF per arm (+ the alpha reference line)


def test_equity_drawdown_two_panels_and_benchmark():
    rng = np.random.default_rng(2)
    r = {a: rng.normal(0.0003, 0.01, size=250) for a in _ARMS}
    fig = equity_drawdown(r, benchmark=rng.normal(0.0002, 0.011, size=250))
    assert len(fig.axes) == 2
    assert fig.axes[0].get_yscale() == "log"
    # market + 3 arms = 4 equity lines on the top panel
    assert len(fig.axes[0].lines) == len(_ARMS) + 1


def test_equity_drawdown_handles_nan_returns():
    r = {"scalar": np.array([0.01, np.nan, -0.02, 0.03])}
    fig = equity_drawdown(r)                       # NaNs coerced to 0.0, never raises
    assert len(fig.axes) == 2


def test_allocation_heatmap_from_snapshots():
    rng = np.random.default_rng(3)
    w = rng.dirichlet(np.ones(8), size=200)        # (T=200, N=8) simplex weights
    alloc = alloc_snapshots(w, top_k=5, n_snapshots=24)
    fig = allocation_heatmap(alloc, asset_labels={i: f"A{i}" for i in range(8)})
    ax = fig.axes[0]
    assert ax.images                               # an imshow heatmap was drawn
    # rows = top-K assets + the 'other' residual row
    assert ax.get_yticklabels()[-1].get_text() == "other"


def test_allocation_heatmap_empty_is_graceful():
    fig = allocation_heatmap({"asset_idx": [], "steps": [], "weights": [], "other": []})
    assert fig.axes  # a placeholder panel, never raises


# --------------------------------------------------------------------------- #
# Previously-untested engine members (flagged by the 2026-07-26 audit): F10/F11 #
# --------------------------------------------------------------------------- #
def test_mechanism_chain_renders_all_cut_variants():
    """F10: the 3-link spine renders with each severed-link glyph and with none (labelled texts present)."""
    from src.viz.schematics import mechanism_chain

    for cut in (None, 1, 2, 3):
        fig = mechanism_chain(cut_link=cut)
        assert fig.axes
        assert any(t.get_text() for t in fig.axes[0].texts)


def test_budget_curve_exhibit_marks_bstar():
    """F11: the per-seed budget curve renders and marks B*."""
    from src.viz.figures import budget_curve_exhibit

    grid = {"distributional": {100_000: {0: 0.10, 1: 0.12},
                               400_000: {0: 0.20, 1: 0.22},
                               800_000: {0: 0.21, 1: 0.20}}}
    fig = budget_curve_exhibit(grid, b_star=400_000)
    assert fig.axes  # renders a panel per winner with the B* marker


_VERDICT_FIXTURE = {
    "winners": {
        "p6dist": {
            "400000": {"paired_diffs_s0_s1_s2": [0.079, 0.114, 0.242], "mean": 0.1449,
                       "se": 0.0494, "ratio_mean_over_se": 2.93, "fires": True},
            "800000": {"paired_diffs_s0_s1_s2": [0.104, 0.131, 0.249], "mean": 0.1610,
                       "se": 0.0444, "ratio_mean_over_se": 3.62, "fires": True},
        }
    }
}


def test_budget_ascent_exhibit_plots_the_VERDICT_values_not_absolute_levels():
    """F11-fallback: the archive behind the absolute curve was destroyed 2026-07-27, so the exhibit
    must be reconstructible from the git-tracked verdict alone — and must plot exactly its numbers."""
    from src.viz.figures import budget_ascent_exhibit

    fig = budget_ascent_exhibit(_VERDICT_FIXTURE, b_star=400_000)
    ax = fig.axes[0]
    thick = [ln for ln in ax.get_lines() if ln.get_linewidth() > 2]
    assert thick, "the paired-mean line must be drawn"
    # baseline pinned at 0 (a seed differenced against itself), then the verdict means in order
    assert [round(float(y), 4) for y in thick[0].get_ydata()] == [0.0, 0.1449, 0.1610]


def test_budget_ascent_exhibit_draws_one_thin_line_per_CRN_SEED():
    """The per-seed fan-out is the honest part of R77's disclosure — it must not be averaged away."""
    from src.viz.figures import budget_ascent_exhibit

    ax = budget_ascent_exhibit(_VERDICT_FIXTURE).axes[0]
    thin = [ln for ln in ax.get_lines() if abs(ln.get_linewidth() - 0.9) < 1e-9]
    assert len(thin) == 3  # == len(paired_diffs_s0_s1_s2)


def test_budget_ascent_exhibit_anchors_the_baseline_at_the_declared_budget():
    """The 200k baseline is 0 BY CONSTRUCTION, so it must appear on the axis as the leftmost rung —
    captioning it as a measurement would overclaim data the destroyed archive no longer supports."""
    from src.viz.figures import budget_ascent_exhibit

    ax = budget_ascent_exhibit(_VERDICT_FIXTURE, baseline=200_000).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()][0] == "200k"
