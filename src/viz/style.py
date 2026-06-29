"""House style for the figure suite: Okabe-Ito colourblind-safe palette + redundant encoding + helpers.

Every arm has a FIXED (colour, marker, hatch) so the suite is visually consistent across figures and every
distinction survives greyscale print (colour is never the only channel — controls also carry a hatch). The
SESOI equivalence band and the IQM marker are drawn the same way everywhere. Pure matplotlib; deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "OKABE_ITO", "ARM_STYLE", "ARM_ORDER", "CONTROL_ARMS",
    "apply_house_style", "arm_style", "equivalence_band", "iqm", "iqm_bootstrap_ci", "savefig",
]

#: Okabe-Ito 8-colour colourblind-safe qualitative palette (Wong, Nature Methods 2011) + a neutral grey.
OKABE_ITO: dict[str, str] = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}

#: Canonical arm rendering order (the contribution first, controls grouped, search baselines last).
ARM_ORDER: tuple[str, ...] = (
    "distributional", "scalar", "scalar_cvar5", "placebo", "placebo_shuffled", "random_search", "bayes_opt", "cma_es", "tpe",
)

#: Arms that are CONTROLS — drawn with a hatch (a non-colour channel) so they read as controls in greyscale.
CONTROL_ARMS: frozenset[str] = frozenset({"placebo", "placebo_shuffled"})

#: Fixed per-arm style: colour + marker + hatch. Used by every figure for cross-figure consistency.
ARM_STYLE: dict[str, dict[str, Any]] = {
    "distributional":   {"color": OKABE_ITO["blue"],       "marker": "o", "hatch": None},
    "scalar":           {"color": OKABE_ITO["vermillion"], "marker": "s", "hatch": None},
    "scalar_cvar5":     {"color": OKABE_ITO["green"],       "marker": "^", "hatch": None},
    "placebo":          {"color": OKABE_ITO["grey"],        "marker": "D", "hatch": "///"},
    "placebo_shuffled": {"color": OKABE_ITO["purple"],      "marker": "v", "hatch": "xxx"},
    "random_search":    {"color": OKABE_ITO["orange"],      "marker": "P", "hatch": None},
    "bayes_opt":        {"color": OKABE_ITO["skyblue"],     "marker": "X", "hatch": None},
    "cma_es":           {"color": OKABE_ITO["yellow"],      "marker": "*", "hatch": None},
    "tpe":              {"color": OKABE_ITO["black"],       "marker": "d", "hatch": None},
}


def arm_style(arm: str) -> dict[str, Any]:
    """Style dict for an arm; a neutral default for any unrecognised label (never raises).

    Returns a COPY (deep review 2026-07-26, #70). The two branches used to disagree: the unknown-arm
    default was built fresh per call, while a known arm handed back the module-global ``ARM_STYLE``
    entry itself — so ``arm_style("scalar")["color"] = ...`` silently repainted that arm in every
    later figure. A caller could not tell which branch it got, and this module's whole purpose is that
    the per-arm style is FIXED across the suite; handing out a mutable reference to it contradicts that.
    """
    return dict(ARM_STYLE.get(arm, {"color": OKABE_ITO["black"], "marker": "o", "hatch": None}))


def apply_house_style() -> None:
    """Set the publication rcParams (vector-safe fonts, sizing, constrained layout). Call before plotting.

    Idempotent; mutates global rcParams, so ``scripts/make_figures.py`` calls it once. Tests that only check
    a figure's structure need not call it (the figure functions don't depend on it).
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.constrained_layout.use": True,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # embed TrueType (no Type-3) so the PDF is editor/printer safe
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def equivalence_band(ax: Any, sesoi: float, *, orient: str = "v", label: str = "SESOI") -> None:
    """Shade the pre-registered equivalence corridor [-sesoi, +sesoi] and draw the zero line.

    The honesty device for a null: equivalence is read geometrically (the whole interval inside the band),
    never from a p-value. ``orient='v'`` shades a vertical x-corridor (forest/effect plots); ``'h'`` a
    horizontal y-corridor.
    """
    if orient == "v":
        ax.axvspan(-sesoi, sesoi, color=OKABE_ITO["green"], alpha=0.12, zorder=0, label=f"{label} ±{sesoi:g}")
        ax.axvline(0.0, color="0.4", lw=0.8, ls="--", zorder=1)
    else:
        ax.axhspan(-sesoi, sesoi, color=OKABE_ITO["green"], alpha=0.12, zorder=0, label=f"{label} ±{sesoi:g}")
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--", zorder=1)


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (Agarwal 2021): the mean of the middle 50% — robust to seed outliers.

    Non-finite values are dropped first, mirroring the inference-layer oracle
    (``src/inference/reporting.py`` / ``bootstrap.py``): a single NaN per-seed score otherwise
    poisons a FIGURE's IQM point/CI while the reported NUMBER silently ignores it — a
    figure-vs-number divergence in a reproducibility-graded artifact (2026-07-05)."""
    x = np.asarray(x, dtype=float).ravel()
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    mid = x[lo:hi] if hi > lo else x
    return float(np.mean(mid))


def iqm_bootstrap_ci(x: np.ndarray, *, reps: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(IQM, lo, hi) with a percentile bootstrap CI over seeds. Deterministic via ``seed``.

    Non-finite values are dropped BEFORE the too-few-points guard (deep review 2026-07-26, #69). The
    guard used to read the RAW ``x.size`` while :func:`iqm` filters non-finite values — so it failed in
    exactly the case the 2026-07-05 finite-filtering was added for. MEASURED: ``[1.0, nan]`` returned
    ``point=1.0`` with ``lo=hi=nan``, because every bootstrap resample of a 1-finite array can draw only
    NaNs, ``iqm`` then returns NaN, and ``np.quantile`` propagates it. Matplotlib silently draws NO error
    bar for a NaN interval, so an arm whose seeds were mostly lost rendered in the headline
    ``F_rliable_intervals`` figure as a BARE POINT — visually the most precise arm on the panel. The
    declared contract for too-few-points is the degenerate triple (pinned by
    ``tests/test_viz.py``: ``iqm_bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)``), which the NaN path violated.
    Filtering first also makes the resamples NaN-free, so the figure's CI now matches the inference
    layer's, which bootstraps the finite values too.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]  # count USABLE points, not raw slots — the guard below depends on it
    point = iqm(x)
    if x.size < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    boot = np.array([iqm(x[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def savefig(fig: Any, path: Any, *, also_pdf: bool = True) -> None:
    """Save a figure to ``path`` (PNG @600dpi) and, by default, a vector PDF sibling. Creates parent dirs."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight")
    if also_pdf and p.suffix.lower() != ".pdf":
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
