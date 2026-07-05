"""The headline figure functions (report-only). Each returns a ``matplotlib.figure.Figure``; deterministic.

Designed for a corroborated NULL — every figure visualises equivalence/overlap HONESTLY:

* :func:`equivalence_forest`     — TOST intervals vs the ±SESOI band, both co-primary legs (F6).
* :func:`rliable_intervals`      — per-arm IQM + stratified-bootstrap CIs, the canonical RL-eval figure.
* :func:`risk_return_clouds`     — the collapsed (CVaR, Sharpe) frontier: the 9 arms pile into one blob.
* :func:`evidence_for_null`      — Bayes-factor gauge + Model-Confidence-Set strip (evidence FOR H0).
* :func:`reward_code_similarity` — AST-distance clustered heatmap: the placebo writes the same code.
* :func:`controls_overlay`       — treatment-vs-controls per-seed rainclouds piling onto one band (F7).
* :func:`responsiveness_scatter` — fed-tail-Δ vs authored-reward-Δ with a Spearman fit: responsiveness≈0 (F8b).
* :func:`learning_curves`        — per-arm critic-loss + return trajectories: training adequacy (F9).
* :func:`delisting_robustness`   — headline contrasts vs ±SESOI across the {0,-30,-55,-100}% delisting band.

v2 replication-suite renderers (R80/R82 report-only; F12–F15):

* :func:`cross_leg_forest`       — per-leg CVaR-contrast CIs + the pooled-mean permutation row (F12).
* :func:`capability_gradient`    — responsiveness vs capability anchor, family pairs as segments (F13).
* :func:`reliability_heatmap`    — models × reliability-metric rates, annotated, fixed [0,1] scale (F14).
* :func:`ten_winners_exhibit`    — verbatim winner code panels, mechanical tail-construct highlights (F15).

The functions take ALREADY-COMPUTED arrays/dicts (per-seed scores, TOST bounds, BF01, MCS result, an
AST-distance matrix) so they are decoupled from the inference layer and trivially testable. Captions and
the SESOI value are the caller's responsibility; defaults match the pre-registration (SESOI = 0.05 DSR).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from src.viz.style import (
    OKABE_ITO,
    arm_style,
    equivalence_band,
    iqm_bootstrap_ci,
)

_LOG = logging.getLogger(__name__)

__all__ = [
    "budget_curve_exhibit",
    "budget_ascent_exhibit",
    "capability_gradient",
    "cross_leg_forest",
    "reliability_heatmap",
    "ten_winners_exhibit",
    "equivalence_forest",
    "rliable_intervals",
    "risk_return_clouds",
    "evidence_for_null",
    "reward_code_similarity",
    "controls_overlay",
    "responsiveness_scatter",
    "learning_curves",
    "seed_trajectory",
    "delisting_robustness",
    "performance_profile",
    "probability_of_improvement",
    "return_tail_distribution",
    "equity_drawdown",
    "allocation_heatmap",
]

_LEG_LABEL = {"sharpe": "H2-RA: Sharpe", "cvar": "H2-Tail: CVaR-5%"}


def _is_equivalent(tost_lo: float, tost_hi: float, sesoi: float) -> bool:
    # STRICT ``>``/``<`` to match the inferential decision rule (``_iqm_tost``/``tost_equivalence``/
    # ``paired_tost`` all require the CI STRICTLY inside (-sesoi, +sesoi)) — a boundary CI at exactly
    # ±sesoi is NOT equivalent, so the figure must not colour it green when the test calls it not-equivalent (P14-F4).
    return bool(tost_lo > -sesoi and tost_hi < sesoi)


def equivalence_forest(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    sesoi: float | Mapping[str, float] = 0.05,
    legs: Sequence[str] = ("sharpe", "cvar"),
    title: str = "Co-primary equivalence: contrasts vs the ±SESOI band",
) -> Any:
    """Forest plot of contrast effect sizes with 90% TOST intervals against the equivalence corridor.

    ``contrasts``: an iterable of mappings with keys ``label`` (e.g. "dist − scalar"), ``leg``
    (``"sharpe"``/``"cvar"``), ``estimate``, ``tost_lo``, ``tost_hi``, ``ci_lo``, ``ci_hi``. A row is drawn
    FILLED when its 90% TOST interval lies inside the band (EQUIVALENT) and OPEN otherwise
    (INCONCLUSIVE) — the figure never reads a null off a p-value.

    NOTE: THE BAND IS PER-LEG (2026-07-26 review). ``sesoi`` may be a single float OR a
    ``{leg: margin}`` mapping, and for the co-primary pair it SHOULD be a mapping. The two legs are
    NOT in the same units — ``analyze_campaign.h2_tost`` is "RA-only by construction; the CVaR
    equivalence stays in ``h2_tost`` (its own units)". Applying the raw ±0.05 DSR margin to the CVaR
    leg is the exact error the analysis layer corrects: per ``analyze_campaign`` (P6 band), *"the raw
    ±SESOI (0.05) ROPE is in RAW CVaR units and is LARGE relative to a daily CVaR magnitude
    O(0.01–0.06), so on the TAIL legs it can near-trivially contain the posterior and OVER-CLAIM null
    evidence"* — so the tail leg uses a RELATIVE band (``tail_margin_fraction`` × |baseline CVaR|,
    default 25 %). A single scalar here would draw the CVaR rows FILLED against a far-too-wide
    corridor, i.e. visually over-claim the null on the bankable-null headline leg. Pass e.g.
    ``sesoi={"sharpe": 0.05, "cvar": 0.25 * abs(baseline_cvar)}``. A scalar is still accepted (it
    applies to every leg) so existing single-leg callers are unchanged.
    """
    import matplotlib.pyplot as plt

    legs = [leg for leg in legs if any(c["leg"] == leg for c in contrasts)] or list(legs)
    fig, axes = plt.subplots(len(legs), 1, figsize=(6.2, 1.2 + 1.6 * len(legs)), sharex=True, squeeze=False)
    for ax, leg in zip(axes[:, 0], legs):
        rows = [c for c in contrasts if c["leg"] == leg]
        # PER-LEG band: the co-primary legs are in different units (see the docstring warning).
        # Precedence: a per-ROW ``sesoi``/``margin`` (the analysis-computed band, e.g. h2_tost's
        # relative tail margin, which depends on THAT contrast's baseline CVaR) > a {leg: margin}
        # mapping > the scalar. Row-level wins because the tail band is per-contrast, not per-leg.
        leg_sesoi = float(sesoi[leg]) if isinstance(sesoi, Mapping) else float(sesoi)
        row_margins = [float(c.get("sesoi", c.get("margin", leg_sesoi))) for c in rows]
        if leg == "cvar" and not isinstance(sesoi, Mapping) and not any(
            ("sesoi" in c or "margin" in c) for c in rows
        ):
            # The over-claim case (2026-07-26 review): a raw DSR-unit margin on the TAIL leg is LARGE
            # against a daily CVaR O(0.01-0.06) and near-trivially contains the interval. Never silent.
            _LOG.warning(
                "equivalence_forest: the %r leg is being drawn against the SCALAR band %.4g, which is "
                "in validation-DSR units — on the tail leg that OVER-CLAIMS equivalence. Pass the "
                "analysis band (h2_tost 'margin', = tail_margin_fraction x |baseline CVaR|) either "
                "per-row as 'sesoi'/'margin' or as sesoi={'cvar': <margin>}.", leg, leg_sesoi,
            )
        equivalence_band(ax, float(np.median(row_margins)) if row_margins else leg_sesoi, orient="v")
        for y, c in enumerate(rows):
            equiv = _is_equivalent(float(c["tost_lo"]), float(c["tost_hi"]), row_margins[y])
            col = OKABE_ITO["blue"] if equiv else OKABE_ITO["vermillion"]
            # 95% CI = thin whisker; 90% TOST interval = thick bar; estimate = marker.
            ax.plot([c["ci_lo"], c["ci_hi"]], [y, y], color=col, lw=1.0, zorder=3)
            ax.plot([c["tost_lo"], c["tost_hi"]], [y, y], color=col, lw=3.2, solid_capstyle="round", zorder=4)
            ax.plot([c["estimate"]], [y], marker="o", ms=6, color=col,
                    markerfacecolor=col if equiv else "white", markeredgecolor=col, zorder=5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([c["label"] for c in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.set_title(_LEG_LABEL.get(leg, leg), loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("contrast effect size (per-leg units: Sharpe / CVaR-5%; vs ±SESOI)")
    # one shared legend entry for the band + a filled/open key
    handles = [
        plt.Line2D([], [], marker="o", color=OKABE_ITO["blue"], ls="-", lw=3.2, label="equivalent (TOST ⊂ band)"),
        plt.Line2D([], [], marker="o", color=OKABE_ITO["vermillion"], markerfacecolor="white", ls="-", lw=3.2, label="inconclusive"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=10)
    return fig


def rliable_intervals(
    scores_by_leg: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int = 0,
    title: str = "Aggregate performance (IQM, 95% stratified-bootstrap CI)",
) -> Any:
    """Per-arm IQM point + bootstrap CI for each leg (Agarwal 2021). Overlapping CIs ⇒ consistent with H0.

    ``scores_by_leg``: ``{leg_label: {arm: per-seed score array}}``. One column per leg; one row per arm
    (controls drawn hatched). The visual payload of a null is overlapping intervals across all arms.
    """
    import matplotlib.pyplot as plt

    legs = list(scores_by_leg)
    arms = list(next(iter(scores_by_leg.values())))
    fig, axes = plt.subplots(1, len(legs), figsize=(3.2 * len(legs), 0.5 + 0.42 * len(arms)), squeeze=False)
    for ax, leg in zip(axes[0], legs):
        for y, arm in enumerate(arms):
            st = arm_style(arm)
            point, lo, hi = iqm_bootstrap_ci(np.asarray(scores_by_leg[leg][arm]), seed=seed + y)
            ax.plot([lo, hi], [y, y], color=st["color"], lw=2.4, solid_capstyle="round", zorder=3)
            ax.plot([point], [y], marker=st["marker"], ms=7, color=st["color"],
                    markerfacecolor=st["color"], markeredgecolor="black", markeredgewidth=0.4, zorder=4)
            if st["hatch"]:
                ax.plot([point], [y], marker="o", ms=13, markerfacecolor="none",
                        markeredgecolor=st["color"], markeredgewidth=0.8, alpha=0.6, zorder=2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms if ax is axes[0, 0] else [""] * len(arms))
        ax.set_ylim(-0.6, len(arms) - 0.4)
        ax.invert_yaxis()
        ax.set_title(leg, fontsize=9, loc="left")
        ax.set_xlabel("IQM")
    fig.suptitle(title, fontsize=10)
    return fig


def risk_return_clouds(
    sharpe_by_arm: Mapping[str, np.ndarray],
    cvar_by_arm: Mapping[str, np.ndarray],
    *,
    title: str = "Risk–return: per-seed clouds collapse onto one neighbourhood",
) -> Any:
    """Scatter each arm's per-seed (CVaR-5% x, Sharpe y) points as a translucent cloud + an IQM centroid.

    The whole-story null figure: if the channel mattered the clouds would separate; here they pile up.
    """
    import matplotlib.pyplot as plt

    from src.viz.style import iqm

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for arm in sharpe_by_arm:
        st = arm_style(arm)
        x = np.asarray(cvar_by_arm[arm], dtype=float).ravel()
        y = np.asarray(sharpe_by_arm[arm], dtype=float).ravel()
        ax.scatter(x, y, s=14, color=st["color"], alpha=0.22, marker=st["marker"],
                   edgecolors="none", zorder=2)
        ax.scatter([iqm(x)], [iqm(y)], s=120, color=st["color"], marker=st["marker"],
                   edgecolors="black", linewidths=0.7, zorder=4, label=arm)
    ax.set_xlabel("CVaR-5% (less negative = safer →)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="best", ncol=2, fontsize=7)
    return fig


def evidence_for_null(
    bf01_by_leg: Mapping[str, float],
    mcs: Mapping[str, Any] | None = None,
    *,
    title: str = "Evidence FOR the null: Bayes factor + Model Confidence Set",
) -> Any:
    """Bayes-factor gauge per leg (log strip with Jeffreys bands) + the MCS membership strip.

    ``bf01_by_leg``: ``{leg: BF01}`` (BF01 > 1 favours H0). ``mcs``: the dict from
    :func:`src.inference.model_confidence_set.model_confidence_set` (``included``/``excluded``/``pvalues``);
    a populated set = "the data cannot single out a best channel". Converts absence-of-evidence into
    bounded evidence-of-absence.
    """
    import matplotlib.pyplot as plt

    has_mcs = bool(mcs and mcs.get("pvalues"))
    fig, axes = plt.subplots(2 if has_mcs else 1, 1, figsize=(6.0, 3.4 if has_mcs else 1.9),
                             squeeze=False, gridspec_kw={"height_ratios": [1, 1.1] if has_mcs else [1]})

    ax = axes[0, 0]
    # Jeffreys evidence bands on a log axis (BF01): 1–3 anecdotal, 3–10 moderate, 10–30 strong, >30 v.strong.
    bands = [(1, 3, "0.92", "anecdotal"), (3, 10, "0.82", "moderate"), (10, 30, "0.70", "strong"),
             (30, 100, "0.58", "very strong")]
    for lo, hi, shade, lab in bands:
        ax.axvspan(lo, hi, color=shade, zorder=0)
        ax.text(np.sqrt(lo * hi), 1.0, lab, ha="center", va="center", fontsize=6.5, color="0.3")
    ax.axvline(1.0, color="0.4", lw=0.8, ls="--", zorder=1)  # BF01=1: no evidence either way
    for i, (leg, bf) in enumerate(bf01_by_leg.items()):
        ax.plot([max(float(bf), 0.3)], [i], marker="D", ms=9, color=OKABE_ITO["blue"],
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.text(max(float(bf), 0.3), i + 0.18, f"BF₀₁={bf:.2g}", ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_yticks(range(len(bf01_by_leg)))
    ax.set_yticklabels([_LEG_LABEL.get(k, k) for k in bf01_by_leg])
    ax.set_ylim(-0.6, len(bf01_by_leg) - 0.2)
    ax.set_xlabel("Bayes factor BF₀₁  (→ favours H0)")
    ax.set_title("evidence for equivalence (JZS BF₀₁)", fontsize=9, loc="left")

    if has_mcs:
        axm = axes[1, 0]
        assert mcs is not None
        pv = mcs["pvalues"]
        arms = list(pv)
        included = set(mcs.get("included", []))
        for x, arm in enumerate(arms):
            st = arm_style(arm)
            inset = arm in included
            axm.scatter([x], [0], s=240, marker=st["marker"],
                        color=st["color"] if inset else "white",
                        edgecolors=st["color"], linewidths=1.4, zorder=3)
            axm.text(x, 0.32, arm, ha="center", va="bottom", rotation=30, fontsize=7)
            axm.text(x, -0.3, f"p={pv[arm]:.2f}", ha="center", va="top", fontsize=6.5, color="0.3")
        axm.set_xlim(-0.7, len(arms) - 0.3)
        axm.set_ylim(-0.8, 0.9)
        axm.set_yticks([])
        axm.set_xticks([])
        for s in axm.spines.values():
            s.set_visible(False)
        n_in = len(included)
        axm.set_title(f"Model Confidence Set (size {mcs.get('size', '?')}): {n_in}/{len(arms)} arms retained "
                      f"— filled = in set", fontsize=9, loc="left")
    fig.suptitle(title, fontsize=10)
    return fig


def reward_code_similarity(
    distance: np.ndarray,
    arm_labels: Sequence[str],
    *,
    title: str = "Authored reward-code structure clusters across arms, not by arm",
) -> Any:
    """AST-distance clustered heatmap + dendrogram + an arm colour sidebar (the mechanism figure).

    ``distance``: a symmetric (N×N) pairwise AST-distance matrix between authored reward programs (e.g.
    ``1 - structural_similarity``). ``arm_labels``: the arm of each of the N candidates. Clusters that cut
    ACROSS arms are the signature of the null: the feedback channel did not change what the LLM wrote.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    d = np.asarray(distance, dtype=float)
    n = d.shape[0]
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    order = list(range(n))
    if n >= 2:
        z = linkage(squareform(d, checks=False), method="average")
        order = dendrogram(z, no_plot=True)["leaves"]

    fig = plt.figure(figsize=(6.6, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 5.0], height_ratios=[1.2, 5.0],
                          wspace=0.02, hspace=0.02)
    ax_dend = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    if n >= 2:
        dendrogram(z, ax=ax_dend, color_threshold=0, above_threshold_color="0.4", no_labels=True)
    ax_dend.set_xticks([])
    ax_dend.set_yticks([])
    for s in ax_dend.spines.values():
        s.set_visible(False)

    dd = d[np.ix_(order, order)]
    im = ax_heat.imshow(dd, cmap="viridis", aspect="auto", origin="upper")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("authored reward programs (clustered)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02, label="AST distance (1 − structural similarity)")

    arms_ordered = [arm_labels[i] for i in order]
    colors = np.array([_to_rgb(arm_style(a)["color"]) for a in arms_ordered]).reshape(n, 1, 3)
    ax_bar.imshow(colors, aspect="auto", origin="upper")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("arm")
    # legend for the arm colour sidebar
    seen: dict[str, Any] = {}
    for a in arm_labels:
        if a not in seen:
            seen[a] = plt.Line2D([], [], marker="s", ls="", color=arm_style(a)["color"], label=a)
    ax_heat.legend(handles=list(seen.values()), loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=7)
    fig.suptitle(title, fontsize=10)
    return fig


def controls_overlay(
    scores_by_arm: Mapping[str, np.ndarray],
    *,
    leg_label: str = "H2-Tail: CVaR-5%",
    title: str = "Treatment vs controls overlap: per-seed rainclouds pile onto one band",
) -> Any:
    """Raincloud overlay of the treatment arm against its placebo / structure-shuffled controls (F7).

    ``scores_by_arm``: ``{arm: per-seed score array}`` — intended to be the distributional arm beside its
    direct controls (``placebo``, ``placebo_shuffled``, ``scalar``). Each arm is drawn as a deterministically
    jittered per-seed strip + an IQR bar + median tick + IQM marker on a shared axis, so the null is read as
    *overlapping distributions*, never off a p-value. Controls are ringed (hatch style) to set them apart.
    """
    import matplotlib.pyplot as plt

    from src.viz.style import iqm

    arms = list(scores_by_arm)
    fig, ax = plt.subplots(figsize=(6.2, 0.8 + 0.6 * len(arms)))
    jitter_rng = np.random.default_rng(0)  # fixed seed → deterministic strip jitter
    for y, arm in enumerate(arms):
        st = arm_style(arm)
        v = np.asarray(scores_by_arm[arm], dtype=float).ravel()
        jit = (jitter_rng.random(v.size) - 0.5) * 0.30
        ax.scatter(v, np.full(v.size, y) + jit, s=16, color=st["color"], alpha=0.32,
                   marker=st["marker"], edgecolors="none", zorder=2)
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [y, y], color=st["color"], lw=3.0, solid_capstyle="round", zorder=3)
            ax.plot([med], [y], marker="|", ms=15, color="black", markeredgewidth=1.4, zorder=4)
            ax.plot([iqm(v)], [y], marker=st["marker"], ms=9, color=st["color"],
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            if st["hatch"]:
                ax.plot([iqm(v)], [y], marker="o", ms=15, markerfacecolor="none",
                        markeredgecolor=st["color"], markeredgewidth=0.9, alpha=0.6, zorder=2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(leg_label)
    ax.set_title(title, fontsize=10, loc="left")
    return fig


def responsiveness_scatter(
    fed_delta: np.ndarray,
    reward_delta: np.ndarray,
    *,
    rho: float | None = None,
    arm: str = "distributional",
    title: str = "Reward edits do not track the fed tail signal (responsiveness ≈ 0)",
) -> Any:
    """Per-generation scatter of (Δ fed tail-statistic, Δ authored-reward) with a least-squares guide (F8b).

    ``fed_delta`` / ``reward_delta``: paired 1-D arrays, one point per gen-N→N+1 transition. A flat / near-zero
    slope is the responsiveness signature of the null — the LLM does not condition its code edits on the
    magnitude of the tail feedback it was shown. ``rho`` (Spearman) is the reported statistic; the dashed line
    is a visual OLS aid only.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(fed_delta, dtype=float).ravel()
    y = np.asarray(reward_delta, dtype=float).ravel()
    st = arm_style(arm)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.scatter(x, y, s=42, color=st["color"], alpha=0.75, marker=st["marker"],
               edgecolors="black", linewidths=0.4, zorder=3)
    if x.size >= 2 and float(np.ptp(x)) > 0.0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, intercept + slope * xs, color=st["color"], lw=1.6, ls="--", zorder=4)
    if rho is not None:
        ax.text(0.04, 0.96, f"Spearman ρ = {float(rho):+.2f}", transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_xlabel("Δ fed tail statistic  (generation N → N+1)")
    ax.set_ylabel("Δ authored-reward  (code-edit magnitude)")
    ax.set_title(title, fontsize=10, loc="left")
    return fig


def learning_curves(
    curves_by_arm: Mapping[str, Mapping[str, np.ndarray]],
    *,
    title: str = "Training adequacy: critic loss settles and returns plateau across arms",
) -> Any:
    """Per-arm training trajectories — critic loss (log) + eval return vs environment steps (F9).

    ``curves_by_arm``: ``{arm: {"steps": arr, "critic_loss": arr, "return": arr}}``. Two stacked panels with all
    arms overlaid: the convergence / training-adequacy diagnostic that answers the standing under-training
    concern (is the critic still diverging at the final step?) and shows the arms reaching the same plateau —
    consistent with a fair, saturated comparison rather than a budget artefact.
    """
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_ret) = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)
    for arm, c in curves_by_arm.items():
        st = arm_style(arm)
        steps = np.asarray(c["steps"], dtype=float).ravel()
        ax_loss.plot(steps, np.asarray(c["critic_loss"], dtype=float).ravel(), color=st["color"],
                     marker=st["marker"], ms=4, lw=1.5, label=arm, zorder=3)
        ax_ret.plot(steps, np.asarray(c["return"], dtype=float).ravel(), color=st["color"],
                    marker=st["marker"], ms=4, lw=1.5, zorder=3)
    ax_loss.set_ylabel("critic loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title(title, fontsize=10, loc="left")
    ax_loss.legend(loc="best", ncol=2, fontsize=7)
    ax_ret.set_ylabel("eval return")
    ax_ret.set_xlabel("environment steps")
    return fig


def _trajectory_grid(n_total: int, rungs: Sequence[int] = (), *, dense_to: int = 30,
                     max_points: int = 200) -> list[int]:
    """Prefix sizes at which to evaluate a running estimate: DENSE at small n, strided after.

    Okhrati's ask is literally *"1,2,3,4,5,6,7,8,9 and etc up to the final seed"*, and the small-n
    region is where the instability lives (the canon's rank order is not stable below ~n=30), so
    every prefix up to ``dense_to`` is kept. Beyond that a full sweep would cost N bootstraps per
    unit for no visual gain, so the grid strides — but it ALWAYS keeps every registered rung and the
    terminal n, because those are the points a reader will actually read values off.
    """
    if n_total <= 0:
        return []
    keep = set(range(1, min(dense_to, n_total) + 1))
    keep.update(r for r in rungs if 1 <= int(r) <= n_total)
    keep.add(n_total)
    remaining = max_points - len(keep)
    if remaining > 0 and n_total > dense_to:
        # CEILING division: floor under-counts the stride, so `range` then emits MORE points than
        # `remaining` budgeted and the grid overruns `max_points` (measured: 213 for N=568). Caught
        # by tests/test_viz.py::test_seed_trajectory_grid_keeps_small_n_dense_rungs_and_terminal.
        stride = max(1, -(-(n_total - dense_to) // remaining))
        keep.update(range(dense_to, n_total + 1, stride))
    return sorted(keep)


def seed_trajectory(
    series_by_unit: Mapping[str, Sequence[float]],
    *,
    seeds: Sequence[int] | None = None,
    rungs: Sequence[int] = (),
    terminal_rung: int | None = None,
    stopping_rule: str = ("EXOGENOUS: the pre-registered assurance ladder, stopped by the Aug-27 "
                          "calendar gate and measured throughput — never by an observed effect."),
    title: str = "Running estimate vs seed count n (registered order)",
    ylabel: str = "IQM over the first n seeds",
    alpha: float = 0.05,
    bootstrap_reps: int = 2000,
) -> Any:
    """Running IQM (+ bootstrap CI) against seed count n = 1…N — Okhrati's D2 duty, registry row 38.

    ``series_by_unit``: ``{unit: per-seed values IN REGISTERED SEED ORDER}``. Because the assurance
    ladder is CUMULATIVE and CRN-paired, every prefix ``[0..n)`` is itself a valid complete study —
    a property most papers cannot show — so this curve answers, with a picture, the question a
    probabilist actually asks: *has your estimate converged, do you understand your own noise, and
    is your conclusion an artefact of where you stopped?* The estimator is
    :func:`~src.viz.style.iqm_bootstrap_ci`, i.e. THE SAME one the headline ``rliable_intervals``
    panel uses, so the trajectory's endpoint equals the headline point estimate by construction
    rather than by coincidence.

    NOTE: **THE FIGURE IS INVALID WITHOUT ALL THREE OF D2's PROPERTIES, so they are enforced here rather
    than left to the caller:**

    1. **REGISTERED ORDER, NEVER SORTED.** A running curve invites a reader to eyeball "where it
       settled", which is precisely the optional-stopping logic the pre-registration forbids; sorting
       the seeds by OUTCOME would turn the curve into a selection artifact. This function never
       sorts, and when ``seeds`` is supplied it VERIFIES the order is the registered one (contiguous
       ascending, as ``resolve_seeds`` emits) and raises otherwise — so a permuted series fails loud
       instead of rendering a beautiful lie.
    2. **THE STOPPING RULE IS STATED ON THE FIGURE**, with the terminal rung marked.
    3. **"NO INFERENCE WAS DRAWN AT ANY PREFIX" IS ALWAYS RENDERED.** It is not a parameter and
       cannot be switched off, because a caption that can be omitted eventually is.

    Done right this is the strongest rigour exhibit in the document; done wrong it hands a referee a
    weapon. Units may have different depths (they sit at different rungs) — each is drawn to its own
    N, which is also why the COMMON rung is a minimum over units and not an average.
    """
    import matplotlib.pyplot as plt

    if not series_by_unit:
        raise ValueError(
            "seed_trajectory: series_by_unit is empty. Refusing to draw an axis with no data — an "
            "empty D2 exhibit would satisfy the registry row while showing the reader nothing."
        )
    if seeds is not None:
        s = [int(v) for v in seeds]
        if not s or s != list(range(s[0], s[0] + len(s))):
            raise ValueError(
                f"seed_trajectory: `seeds` is not in REGISTERED order (expected contiguous ascending "
                f"from {s[0] if s else 'n/a'}); got {s[:8]}… . Sorting or re-ordering seeds turns the "
                "running curve into a selection artifact — see D2."
            )

    # COLOUR COLLISION GUARD (found by rendering real records, 2026-08-01). `arm_style` returns the
    # SAME neutral default for every unrecognised label, so a panel of units that are not arms — the
    # H1 baselines, for instance — drew every series in identical black and the reader could not tell
    # them apart. Where a colour would repeat, fall back to the Okabe-Ito qualitative palette by
    # index. Recognised arms keep their canonical colour, so cross-figure consistency is preserved.
    # NOTE OKABE_ITO is a NAME->HEX MAPPING, not a list; indexing it with an int raises KeyError.
    # Found by rendering four real baseline units (the guard only fires once a colour repeats).
    _fallback = [h for n, h in OKABE_ITO.items() if n != "black"]
    _units = list(series_by_unit)
    _colors: dict[str, str] = {}
    _seen: set[str] = set()
    for _i, _u in enumerate(_units):
        _c = arm_style(_u)["color"]
        if _c in _seen:
            _c = _fallback[_i % len(_fallback)]
        _seen.add(_c)
        _colors[_u] = _c

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    drawn = 0
    _n_max = 0
    for unit, vals in series_by_unit.items():
        v = np.asarray(list(vals), dtype=float).ravel()
        if v.size == 0:
            _LOG.warning("[seed_trajectory] unit %r has no values — skipped", unit)
            continue
        grid = _trajectory_grid(v.size, rungs)
        pt = np.empty(len(grid), dtype=float)
        lo = np.empty(len(grid), dtype=float)
        hi = np.empty(len(grid), dtype=float)
        for i, n in enumerate(grid):
            # n<2 returns the DEGENERATE triple (point, point, point) by contract — a zero-width
            # band at n=1 is honest (one seed carries no dispersion), not a missing one.
            pt[i], lo[i], hi[i] = iqm_bootstrap_ci(
                v[:n], reps=bootstrap_reps, alpha=alpha, seed=0
            )
        st = arm_style(unit)
        col = _colors[unit]
        ax.plot(grid, pt, color=col, lw=1.6, marker=st["marker"], ms=3,
                markevery=max(1, len(grid) // 12), label=unit, zorder=3)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        drawn += 1
        _n_max = max(_n_max, int(v.size))
    if drawn == 0:
        raise ValueError("seed_trajectory: every unit was empty — nothing was drawn.")

    # A caller may legitimately hand the FULL registered tier list (30…568) while the archive only
    # holds the floor rung — mid-campaign that is the normal case. Rungs past the data would stretch
    # the x-axis to 568 and squash every curve into the first 5% of the panel, so they are dropped
    # rather than drawn: the figure shows the ladder REACHED, never the ladder hoped for.
    for r in rungs:
        if int(r) > _n_max:
            continue
        if terminal_rung is not None and int(r) == int(terminal_rung):
            continue
        ax.axvline(float(r), color="0.55", lw=0.8, ls=":", zorder=1)
    if terminal_rung is not None and int(terminal_rung) <= _n_max:
        ax.axvline(float(terminal_rung), color="0.15", lw=1.4, ls="-", zorder=1)
        ax.annotate(f"terminal rung n={int(terminal_rung)}",
                    xy=(float(terminal_rung), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-4, -10), textcoords="offset points",
                    ha="right", va="top", fontsize=7, color="0.15", rotation=90)

    ax.set_xlabel("seed count n (registered order, cumulative)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="best", ncol=2, fontsize=7)
    fig.text(
        0.01, -0.02,
        f"Stopping rule — {stopping_rule}  "
        "NO INFERENCE WAS DRAWN AT ANY PREFIX: the confirmatory look is taken once, at the terminal "
        f"rung. Bands are {int(round((1 - alpha) * 100))}% percentile-bootstrap CIs on the IQM.",
        fontsize=6.5, va="top", wrap=True,
    )
    return fig


def delisting_robustness(
    contrasts_by_treatment: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    treatments: Sequence[str] | None = None,
    sesoi: float = 0.05,
    title: str = "Headline equivalence is robust across the delisting-treatment band",
) -> Any:
    """The delisting-band sensitivity surface: each headline contrast vs the ±SESOI corridor across treatments.

    ``contrasts_by_treatment``: ``{contrast_label: {treatment_label: {"estimate", "ci_lo", "ci_hi"}}}`` — e.g.
    ``{"dist − placebo": {"0%": {...}, "-30%": {...}, "-55%": {...}, "-100%": {...}}}``. The delisting return
    applied to delisted names is a modelling CHOICE, not a fact; pre-registering the whole band
    ``{0, -30, -55, -100}%`` and showing the contrast stays inside ``[-sesoi, +sesoi]`` across ALL of it turns
    "we picked a delisting rule" into "the conclusion does not depend on the delisting rule". A series whose
    every CI sits in the band is drawn solid; any excursion is drawn open + annotated.
    """
    import matplotlib.pyplot as plt

    order = list(treatments) if treatments is not None else list(
        dict.fromkeys(t for series in contrasts_by_treatment.values() for t in series)
    )
    x = np.arange(len(order))
    labels = list(contrasts_by_treatment)
    palette = [OKABE_ITO[c] for c in ("blue", "vermillion", "orange", "purple", "skyblue")]
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(order), 4.2))
    ax.axhspan(-sesoi, sesoi, color=OKABE_ITO["green"], alpha=0.12, zorder=0, label=f"±SESOI {sesoi:g}")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--", zorder=1)
    ax.axhline(sesoi, color=OKABE_ITO["green"], lw=0.8, zorder=1)
    ax.axhline(-sesoi, color=OKABE_ITO["green"], lw=0.8, zorder=1)
    n_series = max(1, len(labels))
    for s, label in enumerate(labels):
        col = palette[s % len(palette)]
        dx = (s - (n_series - 1) / 2) * 0.12  # dodge series so error bars do not overlap
        series = contrasts_by_treatment[label]
        est = np.array([float(series[t]["estimate"]) for t in order])
        lo = np.array([float(series[t]["ci_lo"]) for t in order])
        hi = np.array([float(series[t]["ci_hi"]) for t in order])
        inside = bool(np.all(lo >= -sesoi) and np.all(hi <= sesoi))
        ax.errorbar(x + dx, est, yerr=[est - lo, hi - est], fmt="o-", color=col, lw=1.4, ms=6,
                    markerfacecolor=col if inside else "white", markeredgecolor=col, capsize=3,
                    label=label + ("" if inside else "  (excursion)"), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("delisting return applied to delisted names")
    ax.set_ylabel("contrast effect size (per-seed Sharpe / CVaR-5%)")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="best", fontsize=7)
    return fig


def _to_rgb(hexcolor: str) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    return mcolors.to_rgb(hexcolor)
def budget_curve_exhibit(
    grid: Mapping[str, Mapping[int, Mapping[int, float]]],
    *,
    b_star: int = 400_000,
    title: str = "The measured learning curve: per-seed validation DSR vs training budget (16\u00d7 range)",
) -> Any:
    r"""F11 (R77 MANDATORY exhibit): the extended budget curve, per-seed lines + the paired mean.

    ``grid``: ``{winner_label: {budget: {seed: val_dsr}}}`` (the ``apply_bstar_rule.load_grid``
    shape). One panel per winner, log-x budgets; THIN lines = individual CRN seeds (the honest
    seed fan-out), THICK line = the seed mean; the chosen B\* is marked. Okabe\u2013Ito, grayscale-safe.
    The caption story: the DISTRIBUTIONAL winner's curve rises decisively to the knee at B\* and
    flattens beyond it (the scalar winner clears the rule at B\* too but keeps rising at 1.6M \u2014
    the matched budget is set at the distributional knee); the seed dispersion GROWS with budget
    (the \u03c3_D-recalibration disclosure, R77)."""
    import matplotlib.pyplot as plt

    winners = list(grid.keys())
    fig, axes = plt.subplots(1, len(winners), figsize=(6.4 * len(winners) / 2 + 3.0, 3.6),
                             sharey=False)
    if len(winners) == 1:
        axes = [axes]
    for ax, w in zip(axes, winners):
        budgets = sorted(grid[w].keys())
        seeds = sorted({s for b in budgets for s in grid[w][b].keys()})
        for sd in seeds:
            ys = [grid[w][b].get(sd) for b in budgets]
            ax.plot(budgets, ys, color="0.55", lw=0.9, marker="o", ms=2.5, zorder=2)
        means = [sum(grid[w][b].values()) / len(grid[w][b]) for b in budgets]
        ax.plot(budgets, means, color=OKABE_ITO["blue"], lw=2.4, marker="o", ms=5,
                zorder=4, label="seed mean")
        ax.axvline(b_star, color=OKABE_ITO["vermillion"], lw=1.4, ls="--", zorder=1)
        ax.text(b_star, ax.get_ylim()[1], "  B*", color=OKABE_ITO["vermillion"],
                fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b//1000}k" if b < 1_000_000 else f"{b/1e6:g}M" for b in budgets],
                           fontsize=7)
        ax.set_xlabel("training budget (steps, log scale)")
        ax.set_title(w, fontsize=9, loc="left")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("validation DSR (selection metric)")
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def budget_ascent_exhibit(
    verdict: Mapping[str, Any],
    *,
    b_star: int = 400_000,
    baseline: int = 200_000,
    title: str = ("The pre-committed B* rule as measured: paired ascent over "
                  "200k (±SE), 3 CRN seeds"),
) -> Any:
    r"""F11-fallback (2026-07-27): the R77 ascent curve drawn from the SURVIVING verdict artifact.

    :func:`budget_curve_exhibit` needs ABSOLUTE per-seed validation-DSR levels, which lived only in
    ``outputs/p6ladder/search`` — destroyed 2026-07-27, absent from git and from the backup. This
    renderer reconstructs the load-bearing content from the git-TRACKED
    ``outputs/tables/bstar_rule_verdict.json``, which preserves, per winner per budget, the three
    CRN-paired differences against the 200k baseline plus their mean and SE.

    What it shows FAITHFULLY: the rule's own quantity (paired ascent over 200k), the knee (the
    distributional winner's jump to B\* then near-flat 800k/1.6M), the SE bars the 2× rule is
    defined against, and the per-seed fan-out. What it CANNOT show, and must not be captioned as
    showing: the absolute DSR levels, or the 100k rung — both are gone with the archive. The
    baseline sits at exactly 0 by construction (a seed differenced against itself), which is a
    definition, not a measurement, and is drawn hollow to say so.

    ``verdict``: the parsed ``bstar_rule_verdict.json`` (``{"winners": {label: {budget_str: {...}}}}``).
    """
    import matplotlib.pyplot as plt

    winners = list(verdict["winners"].keys())
    fig, axes = plt.subplots(1, len(winners), figsize=(6.4 * len(winners) / 2 + 3.0, 3.6))
    if len(winners) == 1:
        axes = [axes]
    for ax, w in zip(axes, winners):
        per_b = verdict["winners"][w]
        budgets = [baseline] + sorted(int(b) for b in per_b)
        n_seeds = len(next(iter(per_b.values()))["paired_diffs_s0_s1_s2"])
        for i in range(n_seeds):  # thin lines: the honest per-seed fan-out
            ys = [0.0] + [per_b[str(b)]["paired_diffs_s0_s1_s2"][i] for b in budgets[1:]]
            ax.plot(budgets, ys, color="0.55", lw=0.9, marker="o", ms=2.5, zorder=2)
        means = [0.0] + [per_b[str(b)]["mean"] for b in budgets[1:]]
        errs = [0.0] + [per_b[str(b)]["se"] for b in budgets[1:]]
        ax.errorbar(budgets, means, yerr=errs, color=OKABE_ITO["blue"], lw=2.4, marker="o",
                    ms=5, capsize=3, zorder=4, label="paired mean ± SE")
        ax.plot([baseline], [0.0], marker="o", ms=6, mfc="white",
                mec=OKABE_ITO["blue"], zorder=5)  # hollow: 0 by construction, not measured
        ax.axhline(0.0, color="0.75", lw=0.8, zorder=1)
        ax.axvline(b_star, color=OKABE_ITO["vermillion"], lw=1.4, ls="--", zorder=1)
        ax.text(b_star, ax.get_ylim()[1], "  B*", color=OKABE_ITO["vermillion"],
                fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b//1000}k" if b < 1_000_000 else f"{b/1e6:g}M" for b in budgets],
                           fontsize=7)
        ax.set_xlabel("training budget (steps, log scale)")
        ax.set_title(w, fontsize=9, loc="left")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel(f"paired Δ validation DSR vs {baseline//1000}k")
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------------------------------- #
# v2 replication-suite renderers (F12-F15; R80/R82 report-only — nothing here gates H1-H4)             #
# --------------------------------------------------------------------------------------------------- #

def cross_leg_forest(
    per_leg: Sequence[Mapping[str, Any]],
    *,
    pooled: Mapping[str, Any] | None = None,
    title: str = "Cross-leg replication: (dist − scalar) CVaR-5% contrast at the floor tier",
) -> Any:
    """F12 — per-leg forest of the CVaR-leg contrast with the pooled-mean permutation row.

    ``per_leg``: mappings with ``label``, ``estimate``, ``ci_lo``, ``ci_hi`` (90% seed-bootstrap CI)
    and ``included`` (the registered T0-floor verdict). Excluded legs draw GREYED with an
    "excluded (T0 floor)" annotation — visible, never hidden, never a vote. ``pooled`` (optional):
    ``{"estimate", "p_value"}`` from :func:`src.inference.cross_model.permutation_test` — the ONLY
    inferential number on the figure. Sign convention: positive = distributional safer.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 1.0 + 0.42 * (len(per_leg) + (1 if pooled else 0))))
    ax.axvline(0.0, color=OKABE_ITO["black"], lw=0.8, zorder=2)
    rows = list(per_leg)
    for y, row in enumerate(rows):
        included = bool(row.get("included", True))
        col = OKABE_ITO["blue"] if included else OKABE_ITO["grey"]
        est, lo, hi = float(row["estimate"]), float(row["ci_lo"]), float(row["ci_hi"])
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([est], [y], marker="o", ms=6, color=col,
                markerfacecolor=col if included else "white", markeredgecolor=col, zorder=4)
        if not included:
            ax.annotate("excluded (T0 floor) — authoring/search failure, not a vote",
                        xy=(hi, y), xytext=(4, 0), textcoords="offset points",
                        fontsize=6.5, color=OKABE_ITO["grey"], va="center")
    if pooled is not None:
        y = len(rows)
        est = float(pooled["estimate"])
        ax.plot([est], [y], marker="D", ms=8, color=OKABE_ITO["vermillion"], zorder=5)
        p = pooled.get("p_value")
        ax.annotate(f"pooled mean (joint per-seed flip permutation p = {p:.3f})" if p is not None
                    else "pooled mean (permutation p pending)",
                    xy=(est, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=OKABE_ITO["vermillion"], va="center")
        ax.set_yticks(range(len(rows) + 1))
        ax.set_yticklabels([r["label"] for r in rows] + ["POOLED"])
        ax.set_ylim(-0.6, len(rows) + 0.6)
    else:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["label"] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("(dist − scalar) per-seed mean CVaR-5% diff (signed returns; positive = dist safer)")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def capability_gradient(
    points: Sequence[Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str]] = (),
    anchor_label: str = "capability anchor (pre-declared external composite)",
    rho: float | None = None,
    title: str = "Responsiveness vs author capability (registered: monotone non-decreasing)",
) -> Any:
    """F13 — the capability-gradient scatter with the two family pairs drawn as segments.

    ``points``: mappings with ``label``, ``x`` (capability anchor), ``y`` (SQ1 responsiveness).
    ``pairs``: (bottom_label, top_label) within-family pairs (Qwen 9B→27B open; Haiku→Opus
    closed) — the controlled contrast is made visible as a segment. A flat cloud at y≈0 is the
    honest-null image; ``rho`` (Spearman, from ``capability_regression``) annotates when given.
    """
    import matplotlib.pyplot as plt

    by_label = {str(p["label"]): (float(p["x"]), float(p["y"])) for p in points}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0.0, color=OKABE_ITO["black"], lw=0.8, zorder=1)
    for bottom, top in pairs:
        if bottom in by_label and top in by_label:
            (x0, y0), (x1, y1) = by_label[bottom], by_label[top]
            ax.plot([x0, x1], [y0, y1], color=OKABE_ITO["orange"], lw=1.6, zorder=2,
                    solid_capstyle="round")
    for label, (x, y) in by_label.items():
        ax.plot([x], [y], marker="o", ms=6, color=OKABE_ITO["blue"], zorder=3)
        ax.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    if rho is not None:
        ax.annotate(f"Spearman ρ = {rho:+.2f} (n = {len(by_label)} legs)", xy=(0.02, 0.96),
                    xycoords="axes fraction", fontsize=8, va="top")
    handles = [
        plt.Line2D([], [], marker="o", color=OKABE_ITO["blue"], ls="", label="leg (model)"),
        plt.Line2D([], [], color=OKABE_ITO["orange"], lw=1.6, label="within-family pair"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.set_xlabel(anchor_label)
    ax.set_ylabel("SQ1 responsiveness (fed-signal → authored-code)")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def reliability_heatmap(
    models: Sequence[str],
    metrics: Sequence[str],
    rates: Any,
    *,
    title: str = "Authoring reliability by model (rates in [0, 1])",
) -> Any:
    """F14 — the models × reliability-metrics annotated rate heatmap (the practitioner table).

    ``rates``: array-like (n_models, n_metrics) of rates in [0, 1] (NaN = not measured, rendered
    as an explicit "—" rather than a fake zero). Fixed vmin/vmax [0, 1]: a fully-compliant column
    renders unremarkably — no rhetorical rescaling.
    """
    import matplotlib.pyplot as plt

    mat = np.asarray(rates, dtype=float)
    if mat.shape != (len(models), len(metrics)):
        raise ValueError(
            f"rates shape {mat.shape} != (len(models)={len(models)}, len(metrics)={len(metrics)})"
        )
    fig, ax = plt.subplots(figsize=(1.6 + 0.85 * len(metrics), 1.0 + 0.42 * len(models)))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(models)):
        for j in range(len(metrics)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.55 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=OKABE_ITO["grey"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="rate")
    ax.set_title(title, loc="left", fontsize=10)
    # NO tight_layout here: colorbar layout engines are incompatible with replacing the house
    # constrained-layout engine (the reward_code_similarity precedent).
    return fig


#: The registered tail-construct highlight rule for the winners exhibit (mechanical, not curated):
#: a line is highlighted iff it matches one of these case-insensitive patterns.
TAIL_CONSTRUCT_PATTERNS = (
    r"cvar", r"tail", r"quantile", r"percentile", r"drawdown", r"\bvar\b", r"skew",
    r"downside", r"sortino", r"worst",
)


def ten_winners_exhibit(
    winners: Mapping[str, str],
    *,
    max_lines: int = 26,
    ncols: int = 2,
    title: str = "The winning reward program per model (tail-construct lines highlighted)",
) -> Any:
    """F15 — verbatim side-by-side winner code panels, one per model, mechanically annotated.

    ``winners``: ordered mapping label → reward source. Lines matching
    :data:`TAIL_CONSTRUCT_PATTERNS` get an amber background — the highlight rule is a fixed
    registered regex set, never a curated selection. Panels truncate at ``max_lines`` with an
    explicit "(+k more lines)" marker; code is shown verbatim (no reflow) in monospace.
    """
    import re as _re

    import matplotlib.pyplot as plt

    pat = _re.compile("|".join(TAIL_CONSTRUCT_PATTERNS), _re.IGNORECASE)
    n = len(winners)
    if n == 0:
        raise ValueError("winners must be a non-empty mapping label -> source")
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 0.16 * max_lines * nrows + 1.2),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, (label, source) in enumerate(winners.items()):
        ax = axes[k // ncols][k % ncols]
        lines = source.splitlines()
        shown, extra = lines[:max_lines], max(0, len(lines) - max_lines)
        ax.set_title(label, loc="left", fontsize=9, fontweight="bold")
        for i, line in enumerate(shown):
            y = 1.0 - (i + 1) / (max_lines + 2)
            hit = bool(pat.search(line))
            ax.text(0.01, y, line[:110], transform=ax.transAxes, fontsize=5.6,
                    family="monospace", va="top",
                    bbox=(dict(facecolor="#FFE8B0", edgecolor="none", pad=0.6) if hit else None))
        if extra:
            ax.text(0.01, 1.0 - (len(shown) + 1.5) / (max_lines + 2), f"(+{extra} more lines)",
                    transform=ax.transAxes, fontsize=6, family="monospace",
                    color=OKABE_ITO["grey"], va="top")
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# Corpus-standard additions (2026-07-26): the rliable quartet's other two      #
# members (G1/G2) + the risk-lens/finance staples (G3/G4/G5). All report-only, #
# all take already-computed arrays, all honest-null by construction.           #
# --------------------------------------------------------------------------- #
def performance_profile(
    scores_by_arm: Mapping[str, np.ndarray],
    *,
    n_points: int = 100,
    title: str = "Performance profiles (run-score distributions; Agarwal 2021)",
) -> Any:
    """Run-score distribution per arm: the fraction of seeds scoring above a shared threshold τ (G1).

    ``scores_by_arm``: ``{arm: per-seed score array}``. The second member of the rliable quartet after
    :func:`rliable_intervals`; robust to the heavy tails our data has. Overlapping profiles across arms are
    the visual signature of a null (no arm stochastically dominates)."""
    import matplotlib.pyplot as plt

    pooled = np.concatenate(
        [np.asarray(v, dtype=float).ravel() for v in scores_by_arm.values()] or [np.zeros(1)]
    )
    pooled = pooled[np.isfinite(pooled)]
    lo, hi = (float(pooled.min()), float(pooled.max())) if pooled.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    taus = np.linspace(lo, hi, int(max(2, n_points)))
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for arm, s in scores_by_arm.items():
        st = arm_style(arm)
        s = np.asarray(s, dtype=float).ravel()
        s = s[np.isfinite(s)]
        frac = np.array([float(np.mean(s > t)) for t in taus]) if s.size else np.zeros_like(taus)
        ax.plot(taus, frac, color=st["color"], lw=2.0, label=arm,
                ls=("--" if st["hatch"] else "-"), zorder=3)
    ax.set_xlabel("score threshold τ")
    ax.set_ylabel("fraction of seeds with score > τ")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="best", ncol=2, fontsize=7)
    return fig


def probability_of_improvement(
    prob_by_arm: Mapping[str, Any],
    *,
    baseline_label: str = "scalar",
    title: str = "Probability of improvement over the baseline (0.5 = no effect)",
) -> Any:
    """Per-arm P(arm > baseline) with a CI, against the 0.5 no-effect line (G2; rliable A.28/29).

    ``prob_by_arm``: ``{arm: p}`` or ``{arm: (p, lo, hi)}``. Points clustered on 0.5 with CIs spanning it
    are the null signature; a point whose whole CI sits above 0.5 is a genuine improvement."""
    import matplotlib.pyplot as plt

    arms = list(prob_by_arm)
    fig, ax = plt.subplots(figsize=(5.4, 0.5 + 0.42 * max(1, len(arms))))
    for y, arm in enumerate(arms):
        st = arm_style(arm)
        v = prob_by_arm[arm]
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 3:
            p, lo, hi = (float(v[0]), float(v[1]), float(v[2]))
        else:
            p = lo = hi = float(v)
        ax.plot([lo, hi], [y, y], color=st["color"], lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([p], [y], marker=st["marker"], ms=7, color=st["color"], markeredgecolor="black",
                markeredgewidth=0.4, zorder=4)
    ax.axvline(0.5, color="0.4", lw=0.9, ls="--", zorder=1)  # no improvement
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"P(arm > {baseline_label})")
    ax.set_title(title, fontsize=10, loc="left")
    return fig


def return_tail_distribution(
    returns_by_arm: Mapping[str, np.ndarray],
    *,
    alpha: float = 0.05,
    title: str = "Realized return distributions (left tail annotated)",
) -> Any:
    """Per-arm ECDF of realized returns with the α-quantile (VaR) marked and the left tail shaded (G3).

    ``returns_by_arm``: ``{arm: realized per-step return array}`` (a representative winner or pooled seeds).
    The risk story made visible — where a CVaR/tail claim lives (Tail-Safe / RAMAC archetype). Overlapping
    left tails = the tail-feedback channel did not move the downside."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    var_min = np.inf
    for arm, r in returns_by_arm.items():
        st = arm_style(arm)
        r = np.sort(np.asarray(r, dtype=float).ravel())
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        ecdf = np.arange(1, r.size + 1) / r.size
        ax.plot(r, ecdf, color=st["color"], lw=1.8, label=arm, ls=("--" if st["hatch"] else "-"), zorder=3)
        var = float(np.quantile(r, alpha))
        var_min = min(var_min, var)
        ax.plot([var], [alpha], marker=st["marker"], ms=6, color=st["color"],
                markeredgecolor="black", markeredgewidth=0.4, zorder=4)
    ax.axhline(alpha, color="0.5", ls=":", lw=0.9, zorder=1)
    if np.isfinite(var_min):
        ax.axvspan(ax.get_xlim()[0], var_min, color=OKABE_ITO["vermillion"], alpha=0.06, zorder=0)
    ax.set_xlabel("realized return")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title}  (α={alpha:g} VaR marked)", fontsize=10, loc="left")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    return fig


def equity_drawdown(
    returns_by_arm: Mapping[str, np.ndarray],
    *,
    benchmark: np.ndarray | None = None,
    title: str = "Growth of 1 and drawdown over the sealed test",
) -> Any:
    """Equity curve (log growth of 1) + underwater drawdown per arm on the sealed test (G4).

    ``returns_by_arm``: ``{arm: per-step realized return array}``; ``benchmark`` an optional per-step
    market series (drawn in grey). The finance staple (EIIE / FinRL-DeepSeek / Sood); overlapping curves =
    no arm outperforms out of sample."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    top, bot = axes[0], axes[1]

    def _plot(name: str, r: np.ndarray, color: str, ls: str = "-", lw: float = 1.6, z: int = 3) -> None:
        r = np.asarray(r, dtype=float).ravel()
        r = np.where(np.isfinite(r), r, 0.0)
        eq = np.cumprod(1.0 + r)
        top.plot(eq, color=color, lw=lw, ls=ls, label=name, zorder=z)
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, eq / peak - 1.0, 0.0)
        bot.plot(dd, color=color, lw=max(0.9, lw - 0.4), ls=ls, zorder=z)

    if benchmark is not None:
        _plot("market", benchmark, OKABE_ITO["grey"], ls=":", lw=1.2, z=2)
    for arm, r in returns_by_arm.items():
        st = arm_style(arm)
        _plot(arm, r, st["color"], ls=("--" if st["hatch"] else "-"))
    top.set_yscale("log")
    top.set_ylabel("growth of 1 (log)")
    top.set_title(title, fontsize=10, loc="left")
    top.legend(loc="best", ncol=2, fontsize=7)
    bot.set_ylabel("drawdown")
    bot.set_xlabel("test step")
    bot.axhline(0.0, color="0.5", lw=0.7, zorder=1)
    return fig


def allocation_heatmap(
    alloc: Mapping[str, Any],
    *,
    asset_labels: Mapping[int, str] | None = None,
    title: str = "Allocation over time (top holdings)",
) -> Any:
    """Heatmap of the top-K holdings' weights over time + a residual 'other' row (G5).

    ``alloc``: the dict from :func:`src.inference.exposure.alloc_snapshots`
    (``{asset_idx, steps, weights (S×K), other}``). The learned-policy exhibit (Cartea/Coache/RAMAC);
    a sequential (not categorical) colourmap is correct here — viridis is colourblind-safe."""
    import matplotlib.pyplot as plt

    idx = list(alloc.get("asset_idx", []))
    steps = list(alloc.get("steps", []))
    weights = np.asarray(alloc.get("weights", []), dtype=float)
    other = np.asarray(alloc.get("other", []), dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * (len(idx) + 1) + 1.4))
    if weights.ndim != 2 or weights.size == 0:
        ax.text(0.5, 0.5, "no allocation snapshots", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    mat = np.vstack([weights.T, other[None, :]])  # (K+1, S): rows = top-K assets then 'other'
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0,
                   vmax=float(max(1e-3, np.nanmax(mat))), origin="upper")
    row_labels = [(asset_labels.get(a, str(a)) if asset_labels else str(a)) for a in idx] + ["other"]
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    n_x = len(steps)
    xt = np.linspace(0, n_x - 1, min(6, n_x)).round().astype(int) if n_x else []
    ax.set_xticks(list(xt))
    ax.set_xticklabels([str(steps[i]) for i in xt], fontsize=7)
    ax.set_xlabel("test step")
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="weight")
    return fig
