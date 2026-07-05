"""Static schematic figures (report-only) — the architecture, the split timeline, the prediction branch.

These are *diagrams*, not data plots: they carry no synthetic numbers, so they are correct and final NOW
(no campaign needed) and render to vector PDF for the document. Drawn programmatically in the house style so
they are version-controlled and reproducible rather than a binary draw.io asset.

* :func:`system_diagram`     — F1: the closed Eureka loop (data → SAC → realised returns → tail measurement →
  LLM reward designer → reward code → back), with the FEEDBACK CHANNEL (the contribution) highlighted.
* :func:`splits_timeline`    — F4: train / val / test with the purge+embargo gaps — the leakage-control story.
* :func:`prediction_branch`  — F2: the pre-registered outcome tree (EQUIVALENT / POSITIVE / INCONCLUSIVE) and
  what each branch is read to mean — so the result cannot be reinterpreted after the fact.

The stylised-facts figure (F3) is deliberately NOT here: it is a DATA figure rendered from the real
return panel's TRAIN window by ``src.viz.eda`` (`make_figures` / ``build_f3``), never a fabricated schematic.
"""

from __future__ import annotations

from typing import Any

from src.viz.style import OKABE_ITO

__all__ = ["system_diagram", "splits_timeline", "prediction_branch", "mechanism_chain"]


def _box(ax: Any, x: float, y: float, w: float, h: float, text: str, *,
         fc: str, ec: str = "black", fontsize: float = 8.0, tc: str = "black") -> tuple[float, float]:
    """Draw a rounded box centred-text; return its centre (for arrow anchoring)."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=tc, zorder=3)
    return (x + w / 2, y + h / 2)


def _arrow(ax: Any, p0: tuple[float, float], p1: tuple[float, float], *,
           color: str = "0.25", style: str = "-|>", lw: float = 1.3, rad: float = 0.0,
           label: str | None = None) -> None:
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.8, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=4)


def system_diagram(title: str = "System architecture — the closed reward-design loop (agent fixed)") -> Any:
    """F1: the end-to-end pipeline with the feedback channel — the contribution — highlighted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    blue, green, verm, grey = (OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"], "0.92")

    # top row: the training/measurement path
    _box(ax, 0.2, 4.5, 1.7, 1.0, "Gold panel\n(survivorship-free PIT)", fc=grey)
    _box(ax, 2.5, 4.5, 1.7, 1.0, "Portfolio env\n+ fixed SAC agent", fc=grey)
    _box(ax, 4.8, 4.5, 1.7, 1.0, "Realised\nreturns (OOS)", fc=grey)
    _box(ax, 7.1, 4.5, 2.6, 1.0, "Tail measurement\n(CVaR 1/5/10/25%, mass, skew)", fc=green, tc="black")
    _arrow(ax, (1.9, 5.0), (2.5, 5.0))
    _arrow(ax, (4.2, 5.0), (4.8, 5.0))
    _arrow(ax, (6.5, 5.0), (7.1, 5.0))

    # the feedback CHANNEL (the contribution): tail vector vs scalar, into the designer
    _box(ax, 6.6, 2.6, 3.1, 1.0,
         "FEEDBACK CHANNEL  (the contribution)\nvector tail  vs  scalar", fc=blue, tc="white", fontsize=8)
    _arrow(ax, (8.4, 4.5), (8.2, 3.6), color=green, label="measured")
    _box(ax, 3.6, 2.6, 2.4, 1.0, "LLM reward designer\n(Eureka reflection)", fc=verm, tc="white")
    _arrow(ax, (6.6, 3.1), (6.0, 3.1), color=blue, label="fed each gen")
    _box(ax, 0.4, 2.6, 2.4, 1.0, "Authored\nreward CODE", fc=grey)
    _arrow(ax, (3.6, 3.1), (2.8, 3.1))
    # close the loop: reward code steers the env
    _arrow(ax, (1.6, 3.6), (3.0, 4.5), color="0.25", rad=-0.2, label="steers")

    ax.text(5.0, 1.6, "Arms vary ONLY the feedback channel; the agent (SAC) is held fixed.\n"
                      "The fed tail is the trained policy's OWN realised returns (endogenous; critic-agnostic).",
            ha="center", va="center", fontsize=7.5, color="0.3")
    ax.set_title(title, fontsize=10, loc="left")
    return fig


def splits_timeline(title: str = "Walk-forward splits with purge + embargo (leakage control)") -> Any:
    """F4: train / validation / sealed-test bands with the 60-session purge gaps."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    blue, green, verm = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"]
    bands = [
        ("train\n2005–2016", 2005, 2017, blue, "agent learns + tail feedback measured"),
        ("val\n2017–2019", 2017.17, 2020, green, "winner selection\n(reward-independent DSR)"),
        ("sealed test\n2020–2026", 2020.17, 2026.5, verm, "scored once, never tuned"),
    ]
    for name, x0, x1, col, sub in bands:
        ax.add_patch(plt.Rectangle((x0, 0.55), x1 - x0, 0.6, facecolor=col, edgecolor="black",
                                   linewidth=1.0, alpha=0.85, zorder=2))
        ax.text((x0 + x1) / 2, 0.85, name, ha="center", va="center", color="white", fontsize=8,
                fontweight="bold", zorder=3)
        ax.text((x0 + x1) / 2, 0.30, sub, ha="center", va="center", color="0.3", fontsize=6.6)
    # purge/embargo gaps
    for xg in (2017.0, 2020.0):
        ax.add_patch(plt.Rectangle((xg, 0.55), 0.17, 0.6, facecolor="none", edgecolor="0.2",
                                   hatch="////", linewidth=0.0, zorder=4))
    ax.annotate("purge + embargo\n= max(embargo 21, lookback 60) = 60 sessions",
                xy=(2020.08, 1.15), xytext=(2021.6, 1.7), fontsize=6.6, color="0.2",
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.0))
    # Regime markers on the sealed leg (manifest F4). The COVID marker sits at the EXECUTED test start
    # (2020-03-30): the crash itself (2020-02-19..03-23) falls INSIDE the boundary purge — the CH4 §4.2
    # disclosure — so the sealed leg opens on the post-crash volatility regime, not the drawdown.
    ax.plot([2020.25], [0.50], marker="^", color="0.2", markersize=4, zorder=5)
    ax.text(2020.10, 0.46, "post-crash start (crash in purge)", ha="left", va="top",
            fontsize=5.8, color="0.2")
    ax.plot([2022.3], [1.19], marker="v", color="0.2", markersize=4, zorder=5)
    ax.text(2022.15, 1.26, "2022 bear (rate shock)", ha="left", va="bottom", fontsize=5.8, color="0.2")
    ax.set_xlim(2004.5, 2027)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("year")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    return fig


def prediction_branch(title: str = "Pre-registered outcome tree (read BEFORE the data)") -> Any:
    """F2: the three pre-registered H2 branches and their fixed interpretation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    blue, green, verm, grey = OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"], "0.92"

    _box(ax, 3.7, 4.8, 2.6, 0.9, "H2: tail-vector vs scalar\n(co-primary IUT vs ±SESOI)", fc=grey, fontsize=8)
    branches = [
        (0.4, "TOST ⊂ ±SESOI", green, "EQUIVALENT", "bankable NULL:\ntail specificity adds no\nmarginal value (the prediction)"),
        (3.7, "lower-CI > 0", blue, "POSITIVE", "channel works:\nricher feedback → better\nrisk-adjusted policy"),
        (7.0, "MDE > SESOI", verm, "INCONCLUSIVE", "underpowered:\nreport effect + CI,\nclaim neither"),
    ]
    for x0, cond, col, verdict, meaning in branches:
        cx = x0 + 1.3
        _arrow(ax, (5.0, 4.8), (cx, 3.5), color="0.4", label=cond)
        _box(ax, x0, 2.6, 2.6, 0.9, verdict, fc=col, tc="white", fontsize=9)
        ax.text(cx, 1.7, meaning, ha="center", va="center", fontsize=7.0, color="0.25")
    ax.text(5.0, 0.35, "Every branch has a fixed, pre-registered reading — the result cannot be reinterpreted post hoc.",
            ha="center", va="center", fontsize=7.4, color="0.3", style="italic")
    ax.set_title(title, fontsize=10, loc="left")
    return fig
def mechanism_chain(
    *,
    sq1_label: str = "SQ1 responsiveness\nSpearman ρ = [CAMPAIGN] (CI)",
    sq2_label: str = "SQ2 mediation\nindirect a·b = [CAMPAIGN] (CI)",
    sq3_label: str = "H2-Tail co-primary IUT",
    cut_link: int | None = None,
    title: str = "The 3-link mechanism chain — where does the information stop?",
) -> Any:
    """F10 (D3, integrated 2026-07-13): the paper's spine as ONE image — fed tail signal →
    authored reward code → trained policy → realised tail outcome, with the three sub-questions
    as the arrows. ``cut_link`` (1/2/3 or None) draws the red break glyph on the measured
    severed link — OUTCOME-NEUTRAL scaffolding: built once, the bank gate fills the annotations
    and the cut position; a positive SQ1 simply renders with no cut. Grayscale-safe (Okabe–Ito)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    blue, verm, grey = OKABE_ITO["blue"], OKABE_ITO["vermillion"], "0.92"

    boxes = [
        (0.2, "Fed tail signal", "6 tail statistics\n(e.g. CVaR-5% = [CAMPAIGN])", blue, "white"),
        (3.3, "Authored reward CODE", "the function the LLM writes\n(AST-audited)", grey, "black"),
        (6.4, "Trained policy", "fixed SAC agent\nmatched budget", grey, "black"),
        (9.5, "Realised tail outcome", "out-of-sample CVaR-5%\nacross the seed ladder", grey, "black"),
    ]
    for x0, head, sub, fc, tc in boxes:
        _box(ax, x0, 1.9, 2.3, 1.1, head, fc=fc, tc=tc, fontsize=8.5)
        ax.text(x0 + 1.15, 1.55, sub, ha="center", va="top", fontsize=6.8, color="0.3")

    arrows = [((2.5, 2.45), (3.3, 2.45), sq1_label), ((5.6, 2.45), (6.4, 2.45), sq2_label),
              ((8.7, 2.45), (9.5, 2.45), sq3_label)]
    for i, (p0, p1, lab) in enumerate(arrows, start=1):
        _arrow(ax, p0, p1, color="0.25")
        ax.text((p0[0] + p1[0]) / 2, 3.25, lab, ha="center", va="bottom", fontsize=6.8, color="0.25")
        if cut_link == i:
            mx = (p0[0] + p1[0]) / 2
            ax.plot([mx - 0.12, mx + 0.12], [2.25, 2.65], color=verm, lw=2.4)
            ax.plot([mx - 0.12, mx + 0.12], [2.65, 2.25], color=verm, lw=2.4)
            ax.text(mx, 0.9, "the chain is severed here:\nthe information does not pass this joint",
                    ha="center", va="center", fontsize=7.2, color=verm)
    if cut_link is None:
        ax.text(6.0, 0.9, "[CAMPAIGN]: the cut glyph is placed on the measured severed link — or absent",
                ha="center", va="center", fontsize=7.2, color="0.4", style="italic")
    ax.set_title(title, fontsize=10, loc="left")
    return fig
