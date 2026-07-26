"""Behaviour tests for the v2 replication-suite renderers (F12–F15; src/viz/figures.py).

Headless (Agg). Contracts under test: excluded legs are drawn (greyed), never hidden; the pooled
row carries the permutation p; family-pair segments render only for labels present; the
reliability heatmap rejects shape mismatches and renders NaN as "—" not 0; the winners exhibit
highlights by the fixed registered regex set and truncates with an explicit marker.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.viz import figures as F  # noqa: E402


@pytest.fixture(autouse=True)
def _close_all():
    yield
    plt.close("all")


def _legs(n: int = 9, excluded: set[str] | None = None) -> list[dict]:
    excluded = excluded or set()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = f"leg-{i}"
        est = float(rng.normal(0, 0.001))
        rows.append({"label": label, "estimate": est, "ci_lo": est - 0.002,
                     "ci_hi": est + 0.002, "included": label not in excluded})
    return rows


# ---- F12 cross_leg_forest --------------------------------------------------------------------------- #
def test_cross_leg_forest_draws_all_legs_and_pooled_row() -> None:
    fig = F.cross_leg_forest(_legs(9, excluded={"leg-3"}),
                             pooled={"estimate": 0.0004, "p_value": 0.21})
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "POOLED" and len(labels) == 10           # 9 legs + pooled, none hidden
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "excluded (T0 floor)" in texts                          # the exclusion is visible
    assert "p = 0.210" in texts                                    # the ONE inferential number


def test_cross_leg_forest_without_pooled_row() -> None:
    fig = F.cross_leg_forest(_legs(3))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["leg-0", "leg-1", "leg-2"]


# ---- F13 capability_gradient ------------------------------------------------------------------------ #
def test_capability_gradient_pairs_and_rho_annotation() -> None:
    pts = [{"label": lbl, "x": x, "y": y} for lbl, x, y in
           [("qwen9", 0.2, 0.01), ("qwen27", 0.5, 0.02), ("haiku", 0.4, 0.00), ("opus", 1.0, 0.03)]]
    fig = F.capability_gradient(pts, pairs=[("qwen9", "qwen27"), ("haiku", "opus"),
                                            ("ghost", "opus")], rho=0.4)
    ax = fig.axes[0]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "Spearman" in texts and "n = 4" in texts
    # 2 real pair segments (the "ghost" pair silently absent) + 4 point markers + hline handled
    seg_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_linestyle() == "-"
                 and list(ln.get_xdata()) != [0.0, 0.0]]
    assert sum(1 for ln in seg_lines if ln.get_color() == F.OKABE_ITO["orange"]) == 2


# ---- F14 reliability_heatmap ------------------------------------------------------------------------ #
def test_reliability_heatmap_annotates_and_masks_nan() -> None:
    models, metrics = ["m1", "m2"], ["compliance", "sandbox", "diversity"]
    rates = np.array([[1.0, 0.5, np.nan], [0.9, 0.2, 0.7]])
    fig = F.reliability_heatmap(models, metrics, rates)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "1.00" in texts and "0.20" in texts
    assert "—" in texts                                            # NaN is explicit, never a fake 0


def test_reliability_heatmap_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        F.reliability_heatmap(["m1"], ["a", "b"], np.zeros((2, 2)))


# ---- F15 ten_winners_exhibit ------------------------------------------------------------------------ #
def test_ten_winners_exhibit_highlights_and_truncates() -> None:
    code_tail = "def reward(r):\n    c = cvar_05(r)\n" + "\n".join(f"    x{i} = {i}" for i in range(40))
    code_plain = "def reward(r):\n    return r.mean()"
    fig = F.ten_winners_exhibit({"model-a": code_tail, "model-b": code_plain}, max_lines=10)
    all_texts = [t for ax in fig.axes for t in ax.texts]
    highlighted = [t for t in all_texts if t.get_bbox_patch() is not None]
    assert any("cvar" in t.get_text() for t in highlighted)        # the mechanical rule fired
    assert not any("return r.mean()" in t.get_text() for t in highlighted)  # and only where it should
    assert any("more lines" in t.get_text() for t in all_texts)    # truncation is explicit


def test_ten_winners_exhibit_empty_rejects() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        F.ten_winners_exhibit({})


def test_v2_renderers_deterministic_bytes() -> None:
    # Same inputs -> identical PNG bytes (the house determinism contract).
    import io

    def render() -> bytes:
        fig = F.cross_leg_forest(_legs(4), pooled={"estimate": 0.0, "p_value": 0.5})
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72)
        plt.close(fig)
        return buf.getvalue()

    assert render() == render()
