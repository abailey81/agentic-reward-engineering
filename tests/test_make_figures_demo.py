"""The demo figure path must PREVIEW the engine, never ASSERT a headline verdict (deep review #66/#67).

``scripts/make_figures.py --demo`` is the DEFAULT mode and it writes under the very filenames
``paper/FIGURE_TABLE_MANIFEST.md`` points the PDF at (``F_equivalence_forest.png``,
``F_evidence_for_null.png``, …), into the same ``outputs/figures/`` directory that holds genuine
data-driven figures (``F3_stylised_facts.png`` from the real train window). Two properties therefore have
to hold, and both were BROKEN when these tests were written:

1. **Every demo artifact is self-identifying** (#66). Nothing on the rendered image distinguished a
   fabricated figure from a real one — the demo status lived only in the script docstring and one stdout
   line, neither of which survives into a PNG pasted in a draft.
2. **No demo number asserts the verdict the campaign exists to test** (#67). The contrast CIs were a
   hardcoded ``±0.035`` and the estimate was clipped to ``±0.03``; on the co-primary TAIL leg that
   rendered all three H2-Tail rows GREEN "equivalent", because ±0.035 sits inside the 0.05 default band —
   which is in validation-DSR (Sharpe) units. Against the analysis' own RELATIVE tail band
   (``tail_margin_fraction`` × |baseline CVaR| ≈ 0.0147) not one of those rows is equivalent. The Bayes
   panel was the same class: a hardcoded ``BF01 = 4.8`` renders as a Jeffreys "moderate evidence for H0".
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.inference.bayes_null import bayesian_null_report  # noqa: E402
from src.inference.bootstrap import iqm  # noqa: E402
from src.viz import figures as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_figures as MF  # noqa: E402


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Close every figure this module opens, even when an assertion fails part-way through.

    ``tests/test_viz_eda.py`` asserts ``plt.get_fignums() == []`` as a leak guard, and pytest-randomly
    shuffles file order — so a figure leaked here surfaces as a failure in an unrelated module. Doing this
    in teardown (not at the end of each test body) makes the leak impossible rather than merely unlikely.
    """
    yield
    plt.close("all")


# ---- #66: the demo must be self-identifying on the artifact ----------------------------------------- #
def test_stamp_demo_adds_a_banner_naming_the_data_as_synthetic() -> None:
    fig = F.equivalence_forest(MF.synthesize_null(seed=3, n_seeds=12)["contrasts"])
    before = {t.get_text() for t in fig.texts}
    MF.stamp_demo(fig)
    added = [t for t in fig.texts if t.get_text() not in before]
    assert len(added) == 1, "stamp_demo must add exactly one figure-level banner"
    banner = added[0].get_text()
    # The banner has to say BOTH that the data is synthetic AND that it is not a result — "demo" alone
    # is not enough for a reader who meets the PNG outside this repo.
    assert "SYNTHETIC" in banner.upper()
    assert "NOT RESULTS" in banner.upper()


def test_render_all_stamps_in_demo_mode_and_leaves_real_renders_unstamped(tmp_path: Path) -> None:
    """The banner is the real-vs-demo discriminator, so it must key off ``demo`` and nothing else."""
    data = MF.synthesize_null(seed=3, n_seeds=10)
    stamped = MF.render_all(data, tmp_path / "demo", demo=True)
    clean = MF.render_all(data, tmp_path / "real", demo=False)
    assert stamped and clean, "both modes must still render the suite"
    assert {p.name for p in stamped} == {p.name for p in clean}
    # Same figures, different bytes: identical inputs, so ONLY the banner can differ.
    # (``render_all`` returns ABSOLUTE paths — re-join by ``.name`` or both reads hit the same file.)
    for p in stamped:
        a = (tmp_path / "demo" / p.name).read_bytes()
        b = (tmp_path / "real" / p.name).read_bytes()
        assert a != b, f"{p.name} rendered identically with and without the demo stamp"


# ---- #67: no demo number may assert a headline verdict ---------------------------------------------- #
def test_tail_margin_fraction_mirrors_the_analysis() -> None:
    """The mirrored constant must not drift from the analysis default it stands in for."""
    from scripts.analyze_campaign import h2_tost

    real = inspect.signature(h2_tost).parameters["tail_margin_fraction"].default
    assert MF.TAIL_MARGIN_FRACTION == real, (
        f"make_figures.TAIL_MARGIN_FRACTION={MF.TAIL_MARGIN_FRACTION} has drifted from "
        f"analyze_campaign.h2_tost's tail_margin_fraction={real}; the demo would re-widen the tail band."
    )


def test_every_cvar_contrast_carries_the_relative_tail_band() -> None:
    """A tail row WITHOUT its own band is judged against the Sharpe-unit default — the #67 bug itself."""
    data = MF.synthesize_null(seed=7, n_seeds=30)
    cvar_rows = [c for c in data["contrasts"] if c["leg"] == "cvar"]
    assert cvar_rows, "the demo must still produce tail-leg contrasts"
    for row in cvar_rows:
        assert "margin" in row, f"tail row {row['label']!r} carries no relative band"
        band = row["margin"]
        # Recompute the band the way the analysis does: fraction x |comparator IQM CVaR|.
        comparator = row["label"].split()[-1]
        expected = MF.TAIL_MARGIN_FRACTION * abs(
            float(iqm(np.asarray(data["scores_by_leg"]["CVaR-5%"][comparator], dtype=float)))
        )
        assert band == expected
        # And it must be materially TIGHTER than the Sharpe-unit default, or the bug is back in spirit.
        assert band < 0.5 * 0.05, f"tail band {band:.4f} is not in CVaR units"


def test_the_forest_does_not_warn_that_the_demo_over_claims_equivalence(caplog) -> None:
    """The engine's own over-claim guard is the oracle: rendering the demo must not trip it."""
    data = MF.synthesize_null(seed=7, n_seeds=30)
    with caplog.at_level("WARNING"):
        F.equivalence_forest(data["contrasts"])
    offending = [r.getMessage() for r in caplog.records if "OVER-CLAIM" in r.getMessage().upper()]
    assert not offending, f"the demo still over-claims equivalence: {offending}"


def test_contrast_intervals_are_bootstrapped_not_a_fixed_half_width() -> None:
    """A hardcoded half-width makes every interval exactly symmetric and identically wide — derived
    percentile-bootstrap intervals are neither."""
    contrasts = MF.synthesize_null(seed=7, n_seeds=30)["contrasts"]
    halves = {round(0.5 * (c["tost_hi"] - c["tost_lo"]), 9) for c in contrasts}
    assert len(halves) == len(contrasts), "every contrast shares a width -> the half-width is hardcoded"
    asym = [abs((c["tost_hi"] - c["estimate"]) - (c["estimate"] - c["tost_lo"])) for c in contrasts]
    assert max(asym) > 1e-9, "all intervals are exactly symmetric about the estimate -> not bootstrapped"
    # The 90% TOST interval must sit INSIDE the 95% reported CI, per leg and per row.
    for c in contrasts:
        assert c["ci_lo"] <= c["tost_lo"] and c["tost_hi"] <= c["ci_hi"], c["label"]


def test_bayes_factors_are_derived_from_the_same_draws() -> None:
    """BF01 must be recomputable from the synthesized per-seed scores, not read off a constant."""
    data = MF.synthesize_null(seed=7, n_seeds=30)
    for leg, key in (("sharpe", "Sharpe"), ("cvar", "CVaR-5%")):
        scores = data["scores_by_leg"][key]
        base = np.asarray(scores["scalar"], dtype=float)
        diffs = np.asarray(scores["distributional"], dtype=float) - base
        rope = 0.05 if leg == "sharpe" else MF.TAIL_MARGIN_FRACTION * abs(float(iqm(base)))
        assert data["bf01_by_leg"][leg] == float(bayesian_null_report(diffs, rope)["bf01"])
