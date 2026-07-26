"""Behaviour tests for the F3 stylised-facts EDA figure (``src/viz/eda.py``).

Headless (Agg): the pure helpers are checked against closed forms / handcrafted cases, the figure
must build 4 annotated panels from SYNTHETIC arrays (no licensed data needed), be deterministic
(no RNG inside the figure code), and close cleanly. The real-gold ``build_f3`` end-to-end test is
gated on the licensed panel being present (same pattern as tests/test_loaders.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.viz import eda as E  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_figures as MF  # noqa: E402

# Suffix-aware skip guard (ACTIVE panel = univ5 post-Split-C), matching tests/test_loaders.py.
from src.data.loaders import gold_suffix  # noqa: E402

_GOLD = Path(__file__).resolve().parents[1] / "data" / "gold" / f"returns_panel_{gold_suffix()}.parquet"


def _make_panel(t: int = 700, n: int = 10, seed: int = 0) -> np.ndarray:
    """Deterministic heavy-tailed, vol-clustered synthetic panel with one zero-filled delisting."""
    rng = np.random.default_rng(seed)
    common = 0.008 * rng.standard_t(3, size=t)  # heavy-tailed market factor -> co-crashes
    regime = np.ones(t)
    regime[300:360] = 3.0  # one stress cluster
    idio = 0.006 * rng.standard_t(4, size=(t, n))
    panel = 0.9 * (common * regime)[:, None] + idio * regime[:, None]
    panel[520:, 3] = 0.0  # name 3 "delists" at day 520 (liquidate_to_cash zero fill)
    return panel


# ---- pure helpers ------------------------------------------------------------------------------ #
def test_alive_mask_marks_dead_tail_leading_gap_and_all_zero_column() -> None:
    r = np.array([
        [0.0, 0.01, 0.0],
        [0.02, -0.01, 0.0],
        [0.01, 0.00, 0.0],  # interior zero stays alive (between first/last nonzero)
        [0.00, 0.02, 0.0],  # name 0 dead from here (trailing zeros)
        [0.00, 0.01, 0.0],
    ])
    mask = E.alive_mask_from_returns(r)
    assert mask[:, 0].tolist() == [False, True, True, False, False]  # lead + dead tail excluded
    assert mask[:, 1].tolist() == [True, True, True, True, True]  # interior zero still alive
    assert not mask[:, 2].any()  # all-zero column is never alive
    with pytest.raises(ValueError):
        E.alive_mask_from_returns(np.zeros(5))  # 1-D input rejected


def test_normal_cvar_closed_form_and_monotonicity() -> None:
    # standard normal 5% expected shortfall = -phi(z_.05)/.05 = -2.062713...
    assert abs(E.normal_cvar(0.0, 1.0, 0.05) - (-2.0627128)) < 1e-4
    # location/scale equivariance: CVaR(mu + sigma Z) = mu + sigma CVaR(Z)
    assert abs(E.normal_cvar(0.001, 0.02, 0.05) - (0.001 + 0.02 * E.normal_cvar(0, 1, 0.05))) < 1e-12
    # deeper tail level -> more negative
    assert E.normal_cvar(0.0, 1.0, 0.01) < E.normal_cvar(0.0, 1.0, 0.05)
    with pytest.raises(ValueError):
        E.normal_cvar(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        E.normal_cvar(0.0, 1.0, 1.0)


def test_normal_cvar_agrees_with_repo_empirical_estimator_on_gaussian_sample() -> None:
    # the conventions cross-check: src.inference.bootstrap.cvar on a large N(0,1) sample must land
    # on the closed form eda.normal_cvar uses (both signed lower-tail means).
    from src.inference.bootstrap import cvar as empirical_cvar

    x = np.random.default_rng(7).standard_normal(200_000)
    assert abs(empirical_cvar(x, 0.05) - E.normal_cvar(0.0, 1.0, 0.05)) < 0.03


def test_rolling_realized_vol_shape_and_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    v = E.rolling_realized_vol(x, window=2)
    assert v.shape == (3,)
    assert np.allclose(v, np.std([1, 2], ddof=1))  # every adjacent pair has the same spread
    assert np.allclose(E.rolling_realized_vol(np.ones(50), window=21), 0.0)  # constant -> 0 vol
    with pytest.raises(ValueError):
        E.rolling_realized_vol(np.ones(5), window=21)  # too short


def test_stress_episodes_counts_contiguous_runs() -> None:
    vol = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    threshold, mask, episodes = E.stress_episodes(vol, stress_quantile=0.80)
    assert 0.0 < threshold <= 1.0
    assert int(mask.sum()) == 3  # the three hot days...
    assert episodes == [(2, 3), (5, 5)]  # ...collapse into two contiguous episodes
    # a run reaching the final index is closed properly
    _, _, eps_tail = E.stress_episodes(np.array([0.0, 0.0, 1.0, 1.0]), stress_quantile=0.5)
    assert eps_tail[-1][1] == 3
    with pytest.raises(ValueError):
        E.stress_episodes(np.array([]))


def test_co_crash_fractions_bounds_alive_denominator_and_dead_name() -> None:
    panel = _make_panel()
    alive = E.alive_mask_from_returns(panel)
    frac = E.co_crash_fractions(panel, alive)
    finite = frac[np.isfinite(frac)]
    assert finite.size == panel.shape[0]  # every day has alive names here
    assert float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0
    # the dead name can never crash after delisting, and the denominator must shrink with it:
    # force day 600 to be a universal crash; the fraction must be 1.0 over the 9 ALIVE names.
    crash_day = panel.copy()
    crash_day[600, :] = -0.5
    crash_day[600, 3] = 0.0  # name 3 is dead (zero-filled), cannot move
    frac2 = E.co_crash_fractions(crash_day, E.alive_mask_from_returns(crash_day))
    assert frac2[600] == pytest.approx(1.0)
    # shape guards
    with pytest.raises(ValueError):
        E.co_crash_fractions(panel, alive[:, :2])


def test_stylised_fact_stats_keys_and_sanity() -> None:
    panel = _make_panel()
    stats = E.stylised_fact_stats(panel, alive_mask=E.alive_mask_from_returns(panel))
    from scipy.stats import norm

    assert stats["n_days"] == 700 and stats["n_assets"] == 10
    assert stats["n_dead_by_end"] == 1  # the zero-filled name
    assert stats["excess_kurtosis"] > 1.0  # Student-t mixture is heavy-tailed
    assert stats["tail_3sigma"]["normal_prob"] == pytest.approx(float(norm.cdf(-3)))
    assert set(stats["cvar_by_level"]) == set(E.FED_CVAR_LEVELS)
    deep = stats["cvar_by_level"][0.01]
    assert deep["empirical"] < deep["normal"] < 0.0  # heavy tail dives below the Gaussian tail
    assert stats["n_stress_episodes"] >= 1
    assert stats["longest_episode_days"] >= 1
    assert stats["n_stress_days"] >= stats["n_stress_episodes"]
    # the engineered stress cluster + common factor must show co-crash amplification
    assert stats["co_crash_stress_mean"] > stats["co_crash_calm_mean"] > 0.0
    assert stats["co_crash_ratio"] > 1.0
    assert 0.0 <= stats["worst_day_co_crash"] <= 1.0
    with pytest.raises(ValueError):
        E.stylised_fact_stats(np.zeros((300, 4)))  # degenerate (constant) portfolio fails loud


# ---- the figure -------------------------------------------------------------------------------- #
def test_fig_stylised_facts_builds_four_annotated_panels() -> None:
    panel = _make_panel()
    fig = E.fig_stylised_facts(panel, alive_mask=E.alive_mask_from_returns(panel))
    assert len(fig.axes) == 4
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    for tag in ("(a)", "(b)", "(c)", "(d)"):
        assert any(t.startswith(tag) for t in titles)
    for t in titles:  # every panel carries its one-line takeaway subtitle
        assert "\n" in t
    all_text = " ".join(txt.get_text() for ax in fig.axes for txt in ax.texts)
    assert "kurtosis" in all_text  # panel (a) annotation
    assert "cvar_05" in all_text  # panel (b) fed-level markers
    assert "episodes" in all_text  # panel (c) clustering annotation
    assert "worst day" in all_text  # panel (d) tail-dependence annotation
    assert fig.axes[0].get_yscale() == "log"  # (a) log density axis
    assert fig.axes[1].get_xscale() == "log"  # (b) log alpha axis
    x0, x1 = fig.axes[1].get_xlim()
    assert x0 > x1  # (b) axis inverted: deeper tail rightward
    plt.close(fig)


def test_fig_stylised_facts_is_deterministic() -> None:
    panel = _make_panel()
    fig_a = E.fig_stylised_facts(panel)
    fig_b = E.fig_stylised_facts(panel)
    # CVaR curves (panel b) and bar heights (panel d) identical across calls — no hidden RNG
    ya = [ln.get_ydata() for ln in fig_a.axes[1].get_lines()]
    yb = [ln.get_ydata() for ln in fig_b.axes[1].get_lines()]
    assert all(np.allclose(a, b) for a, b in zip(ya, yb))
    ha = [p.get_height() for p in fig_a.axes[3].patches]
    hb = [p.get_height() for p in fig_b.axes[3].patches]
    assert np.allclose(ha, hb)
    plt.close(fig_a)
    plt.close(fig_b)


def test_fig_stylised_facts_accepts_dates_and_defaults() -> None:
    panel = _make_panel(t=300, n=6, seed=2)
    dates = np.datetime64("2005-01-03") + np.arange(300).astype("timedelta64[D]")
    pre = set(plt.get_fignums())  # whatever earlier tests left open is NOT this test's business
    fig = E.fig_stylised_facts(panel, dates=dates, footnote="train window only — snoop-clean")
    assert any("snoop-clean" in t.get_text() for t in fig.texts)  # the caption footnote is baked in
    plt.close(fig)
    fig2 = E.fig_stylised_facts(panel)  # no dates, no alive mask, no footnote -> integer axis
    assert len(fig2.axes) == 4
    plt.close(fig2)
    # NOTE: 2026-07-26 deep review (#68). This asserted the GLOBAL `plt.get_fignums() == []`, which is not a
    # property of the code under test but of every test that happened to run first — and pytest-randomly is
    # a HARD dependency here precisely to reshuffle that order each run. MEASURED: identical code, identical
    # tree, `--randomly-seed=11` PASSED while `22` and `33` FAILED with 13 figures left open by earlier viz
    # tests. A green suite was therefore partly a property of the shuffle seed. The honest assertion is the
    # DELTA — it still catches the real leak (a stray figure beyond the one returned) and nothing else.
    assert set(plt.get_fignums()) - pre == set(), "fig_stylised_facts leaked a figure beyond the one returned"


def test_render_eda_skips_gracefully_without_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _no_gold(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("gold artifact not found (licensed data)")

    monkeypatch.setattr("src.viz.eda.build_f3", _no_gold)
    assert MF.render_eda(tmp_path) == []
    assert "SKIPPED" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # nothing fabricated


# ---- the real gold panel end to end (licensed-data gated, like tests/test_loaders.py) ----------- #
@pytest.mark.skipif(not _GOLD.exists(), reason="frozen gold panel not present (licensed data)")
def test_build_f3_on_real_train_window(tmp_path: Path) -> None:
    path, stats = E.build_f3(tmp_path / "F3_stylised_facts.png")
    assert path.exists() and path.stat().st_size > 0
    assert path.with_suffix(".pdf").exists()  # vector sibling
    # train-window discipline: the loader's development window ends 2016-12-30, the last session
    # <= the Split-C train end 2016-12-31 (sealed years unread)
    assert stats["window"].endswith("2016-12-30")
    assert stats["n_assets"] == 30 and stats["n_days"] > 2_000
    assert stats["excess_kurtosis"] > 0.5  # the motivating fact itself
    assert stats["co_crash_stress_mean"] > stats["co_crash_calm_mean"]


def test_stress_band_label_FOLLOWS_the_quantile_parameter() -> None:
    """Panel (c)'s stress-band label must be derived from ``stress_quantile``, not hardcoded.

    Regression for a real config-vs-label defect (deep review, finding #55): the annotation read
    "top-decile days" as a fixed word while ``stress_quantile`` is a PUBLIC argument of
    ``fig_stylised_facts`` (which ``scripts/build_notebook_results.py`` imports directly). A caller
    passing 0.95 therefore had a top-5% set labelled a decile. The sibling panel (d) already rendered
    its own ``crash_quantile`` dynamically, so the figure disagreed with itself.

    The paper-facing render was never wrong — ``build_f3`` uses the 0.90 default, a true decile — so
    this pins a latent inconsistency, not a published error."""
    panel = _make_panel()

    def _panel_c_text(q: float) -> str:
        fig = E.fig_stylised_facts(panel, stress_quantile=q)
        try:
            return " ".join(t.get_text() for t in fig.axes[2].texts)
        finally:
            plt.close(fig)

    decile = _panel_c_text(0.90)
    assert "top-10%" in decile, f"expected the decile band to render as top-10%; got: {decile!r}"

    ventile = _panel_c_text(0.95)
    assert "top-5%" in ventile, f"expected the 0.95 band to render as top-5%; got: {ventile!r}"
    assert "decile" not in ventile.lower(), (
        "a top-5% stress set is still being called a 'decile' — the label is hardcoded again"
    )
    # and panel (d) keeps deriving its own quantile (the sibling this now matches)
    fig = E.fig_stylised_facts(panel, crash_quantile=0.10)
    try:
        d_text = " ".join(t.get_text() for t in fig.axes[3].texts)
        assert "10%" in d_text
    finally:
        plt.close(fig)
