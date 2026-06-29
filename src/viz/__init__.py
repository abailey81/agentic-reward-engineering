"""Publication-grade figure engine (report-only; faultless-presentation lever).

A small, deterministic matplotlib/seaborn layer that renders the dissertation's headline figures from the
per-seed/per-arm results + inference outputs. STRICTLY report-only — it reads results, never gates a
hypothesis and never touches the frozen confirmatory pipeline. Designed around a corroborated NULL: every
figure visualises EQUIVALENCE / overlap honestly (SESOI bands, effect-size CIs, evidence-FOR-the-null),
never manufactures a difference. House style: Okabe-Ito colourblind-safe palette + redundant marker/hatch
encoding so figures survive greyscale print.

See ``src.viz.style`` (palette + house style), ``src.viz.figures`` (the figure functions), and
``src.viz.eda`` (the F3 stylised-facts figure from the REAL train window — the EDA that motivates the
feedback channel); ``scripts/make_figures.py`` wires data -> figures (with a ``--demo`` synthetic mode
buildable pre-campaign).
"""

from __future__ import annotations

from src.viz import eda, figures, style

__all__ = ["eda", "figures", "style"]
