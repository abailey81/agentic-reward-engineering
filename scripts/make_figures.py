"""Render the dissertation's headline figure suite (report-only; faultless-presentation lever).

Produces the nine data-driven headline figures from :mod:`src.viz.figures` — the co-primary equivalence
forest, the rliable IQM intervals, the collapsed risk–return clouds, the evidence-FOR-the-null panel
(Bayes factor + Model Confidence Set), the reward-code AST-similarity heatmap, the treatment-vs-controls
raincloud overlay, the fed-signal→reward responsiveness scatter, the per-arm learning curves, and the
delisting-band robustness surface — into ``outputs/figures/`` as 600-dpi PNG + vector PDF, in a consistent
Okabe-Ito house style. It additionally
renders the **advanced** suite (``src.viz.advanced``): two principled static 3-D figures (a classical-MDS
reward-code embedding + the CVaR×generation×Sharpe search landscape) that go IN the PDF, plus two
**supplementary GIF animations** (generation-by-generation convergence; a rotating embedding) written to
``outputs/figures/anim/`` — pass ``--no-advanced`` to skip them.

It also renders the **F3 stylised-facts EDA figure** (``src.viz.eda``) from the REAL gold panel's TRAIN
window — the one data-driven figure that is final NOW (train-window-only, snoop-clean); it is skipped
with a printed reason when the licensed gold panel is absent (synthetic-only installs).

The confirmatory campaign is unrun, so the headline figures need ``[CAMPAIGN]`` data. This script's
``--demo`` mode (default) synthesises NULL-shaped data (all arms drawn indistinguishable) so the whole
suite is buildable and visually validatable NOW; post-campaign, ``--results-root <dir>`` will load the real
per-seed/inference outputs and re-render the identical figures. Deterministic.

Usage::

    python scripts/make_figures.py                       # demo (synthetic null) -> outputs/figures/
    python scripts/make_figures.py --out outputs/figures
    python scripts/make_figures.py --results-root outputs/campaign   # (post-campaign; not yet wired)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless/remote-safe: render to files, never a GUI

import numpy as np

from src.viz import figures as F
from src.viz.style import ARM_ORDER, apply_house_style, savefig

_LEGS = ("sharpe", "cvar")

#: The analysis' report-only RELATIVE tail band, as a fraction of |baseline CVaR| — MIRRORS the
#: ``tail_margin_fraction`` default of ``scripts.analyze_campaign.h2_tost``. It is duplicated here (rather
#: than imported) to keep the figure path free of that heavy module; the mirror is pinned against the real
#: default by ``tests/test_make_figures_demo.py::test_tail_margin_fraction_mirrors_the_analysis``, so drift
#: fails a test instead of silently re-widening the tail band (#67).
TAIL_MARGIN_FRACTION = 0.25


def synthesize_null(seed: int = 7, n_seeds: int = 30) -> dict[str, Any]:
    """Synthesise NULL-shaped per-seed scores + inference outputs (all arms indistinguishable).

    Pure illustration so the figure ENGINE is validatable pre-campaign; NO number here is a result.
    """
    rng = np.random.default_rng(seed)
    arms = list(ARM_ORDER)
    # All arms share one Sharpe/CVaR neighbourhood; tiny per-arm offsets << the ±0.05 SESOI -> a true null.
    sharpe = {a: 0.55 + rng.normal(0, 0.18, n_seeds) + rng.uniform(-0.02, 0.02) for a in arms}
    cvar = {a: -0.058 + rng.normal(0, 0.012, n_seeds) + rng.uniform(-0.002, 0.002) for a in arms}
    scores_by_leg = {"Sharpe": sharpe, "CVaR-5%": cvar}

    # Co-primary contrasts: distributional vs each control, both legs — every number DERIVED from the
    # draws above by the analysis' OWN estimators, never asserted.
    #
    # NOTE: 2026-07-26 deep review (#67). This block used to hardcode `half = 0.035` and clip the estimate to
    # ±0.03, which IMPOSED the verdict instead of deriving it. MEASURED consequence on the tail leg: all
    # three H2-Tail rows rendered GREEN "equivalent" — because a ±0.035 interval sits inside the 0.05
    # default band, which is in validation-DSR (Sharpe) units. Against the analysis' own RELATIVE tail band
    # (`tail_margin_fraction` × |baseline CVaR| ≈ 0.0147) every one of those rows is NOT equivalent. So the
    # demo asserted the bankable-null verdict on the co-primary tail leg that the campaign exists to TEST,
    # under `F_equivalence_forest.png` — the exact filename `paper/FIGURE_TABLE_MANIFEST.md` points the PDF
    # at. A demo previews the ENGINE; it must never pre-announce the result.
    from src.inference.bootstrap import iqm

    def _paired_iqm_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> tuple[float, np.ndarray]:
        """IQM(a)-IQM(b) + its paired-seed bootstrap draws — `analyze_campaign._iqm_tost`'s estimator."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        boot = np.empty(int(n_boot), dtype=float)
        for i in range(int(n_boot)):
            idx = rng.integers(0, a.size, size=a.size)  # paired: the SAME seed draw on both arms
            boot[i] = iqm(a[idx]) - iqm(b[idx])
        return float(iqm(a) - iqm(b)), boot

    contrasts = []
    for leg, ref in (("sharpe", sharpe), ("cvar", cvar)):
        for other in ("scalar", "placebo", "scalar_cvar5"):
            est, boot = _paired_iqm_boot(ref["distributional"], ref[other])
            row = {
                "label": f"dist − {other}", "leg": leg, "estimate": est,
                # 90% (the TOST interval: two one-sided 5% tests) inside 95% (the reported CI).
                "tost_lo": float(np.quantile(boot, 0.05)), "tost_hi": float(np.quantile(boot, 0.95)),
                "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
            }
            if leg == "cvar":
                # The RELATIVE tail band, PER CONTRAST: fraction × |baseline CVaR|, the baseline being the
                # comparator arm's IQM (`analyze_campaign.h2_tost._legs(..., relative=True)`). Supplying it
                # per-row is what stops the tail leg being judged against the Sharpe-unit DSR margin.
                row["margin"] = TAIL_MARGIN_FRACTION * abs(float(iqm(np.asarray(ref[other], dtype=float))))
            contrasts.append(row)

    # BF01 per leg, DERIVED from the same paired per-seed differences through the real estimator
    # (was a hardcoded {"sharpe": 6.4, "cvar": 4.8} labelled "illustrative" — but it rendered into
    # `F_evidence_for_null.png` as a Jeffreys "moderate evidence for H0" verdict, same #67 class).
    from src.inference.bayes_null import bayesian_null_report

    bf01_by_leg = {}
    for leg, ref in (("sharpe", sharpe), ("cvar", cvar)):
        base = np.asarray(ref["scalar"], dtype=float)
        diffs = np.asarray(ref["distributional"], dtype=float) - base
        rope = 0.05 if leg == "sharpe" else TAIL_MARGIN_FRACTION * abs(float(iqm(base)))
        bf01_by_leg[leg] = float(bayesian_null_report(diffs, rope)["bf01"])

    from src.inference.model_confidence_set import model_confidence_set

    mcs = model_confidence_set(sharpe, size=0.10, reps=500, seed=1)

    # AST-distance matrix: candidates from all arms, with clusters that cut ACROSS arms (the mechanism).
    n_per = 3
    cand_arms = [a for a in arms for _ in range(n_per)]
    n = len(cand_arms)
    # latent structural "template" id NOT aligned to arm -> cross-arm blocks (placebo writes same code).
    template = rng.integers(0, 3, size=n)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base = 0.08 if template[i] == template[j] else 0.62
            dist[i, j] = base + rng.uniform(0, 0.06)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    # F7 controls overlay: the treatment arm beside its direct controls on the tail leg (all overlap).
    controls = [a for a in ("distributional", "scalar", "placebo", "placebo_shuffled") if a in cvar]
    controls_cvar = {a: cvar[a] for a in controls}

    # F8b responsiveness: per-generation (Δ fed tail, Δ authored-reward), drawn INDEPENDENT -> ρ ≈ 0 (the null).
    from scipy.stats import spearmanr

    n_gen = 24
    fed_delta = rng.normal(0.0, 1.0, n_gen)
    reward_delta = rng.normal(0.0, 1.0, n_gen)  # independent of the fed signal -> no responsiveness
    rho = float(spearmanr(fed_delta, reward_delta).correlation)

    # F9 learning curves: critic loss decays to a plateau, return rises to a plateau; arms saturate together.
    steps = np.linspace(2_000, 50_000, 25)
    curves = {}
    for a in arms:
        off = rng.uniform(-0.03, 0.03)
        critic_loss = np.clip(5.0 * np.exp(-steps / 12_000.0) + 0.15 + rng.normal(0, 0.01, steps.size), 1e-3, None)
        ret = (0.6 + off) * (1.0 - np.exp(-steps / 9_000.0)) + rng.normal(0, 0.01, steps.size)
        curves[a] = {"steps": steps, "critic_loss": critic_loss, "return": ret}

    # Advanced 3D / animation data: per-candidate (CVaR, Sharpe, generation) trajectories that START spread
    # and CONVERGE onto the shared null neighbourhood (the optimisation dynamics), + cumulative per-gen frames.
    n_gen_search, per_gen = 5, 6
    trajectory: dict[str, dict[str, np.ndarray]] = {}
    for a in arms:
        cv_a, sh_a, gen_a = [], [], []
        for g in range(n_gen_search):
            spread = 1.0 - g / n_gen_search  # shrinks toward 0 as the search converges
            cv_a.extend(-0.058 + rng.normal(0, 0.004 + 0.012 * spread, per_gen))
            sh_a.extend(0.55 + rng.normal(0, 0.06 + 0.22 * spread, per_gen))
            gen_a.extend([g] * per_gen)
        trajectory[a] = {
            "cvar": np.asarray(cv_a), "sharpe": np.asarray(sh_a), "gen": np.asarray(gen_a),
        }
    frames: list[dict[str, dict[str, np.ndarray]]] = []
    for g in range(n_gen_search):
        frame: dict[str, dict[str, np.ndarray]] = {}
        for a in arms:
            mask = trajectory[a]["gen"] <= g  # cumulative: everything authored by generation g
            frame[a] = {"cvar": trajectory[a]["cvar"][mask], "sharpe": trajectory[a]["sharpe"][mask]}
        frames.append(frame)

    # Delisting-band robustness: the headline contrasts stay inside ±SESOI across {0,-30,-55,-100}% (null robust).
    delisting = {}
    for contrast in ("dist - placebo", "dist - scalar"):
        series = {}
        for treat in ("0%", "-30%", "-55%", "-100%"):
            est = float(np.clip(rng.normal(0.0, 0.012), -0.03, 0.03))
            series[treat] = {"estimate": est, "ci_lo": est - 0.03, "ci_hi": est + 0.03}
        delisting[contrast] = series

    # G1-G5 (2026-07-26): the corpus-standard additions, all NULL-shaped (no number here is a result).
    # G2: all-pairs P(arm > scalar) over seeds -> ~0.5 for every arm (no improvement).
    base_sharpe = sharpe["scalar"]
    prob_improve = {}
    for a in arms:
        pa = float(np.mean(sharpe[a][:, None] > base_sharpe[None, :]))
        prob_improve[a] = (pa, max(0.0, pa - 0.09), min(1.0, pa + 0.09))
    # G3/G4: per-step realized test returns per arm (one representative winner; a shared daily law -> null).
    n_test = 400
    returns_by_arm = {a: 0.0004 + rng.normal(0.0, 0.011, n_test) for a in arms}
    market_bench = 0.0003 + rng.normal(0.0, 0.012, n_test)
    # G5: a representative winner's simplex weights -> allocation snapshots for the heatmap.
    from src.inference.exposure import alloc_snapshots

    weights_demo = rng.dirichlet(np.ones(12), size=n_test)
    alloc_demo = alloc_snapshots(weights_demo, top_k=8, n_snapshots=32)

    return {
        "scores_by_leg": scores_by_leg, "sharpe": sharpe, "cvar": cvar,
        "prob_improve": prob_improve, "returns_by_arm": returns_by_arm,
        "market_bench": market_bench, "alloc": alloc_demo,
        "contrasts": contrasts, "bf01_by_leg": bf01_by_leg, "mcs": mcs,
        "ast_distance": dist, "cand_arms": cand_arms,
        "controls_cvar": controls_cvar, "fed_delta": fed_delta, "reward_delta": reward_delta, "rho": rho,
        "curves": curves, "trajectory": trajectory, "frames": frames, "delisting": delisting,
    }


#: Stamped ONTO every figure rendered from synthetic demo data (deep review 2026-07-26, #66).
_DEMO_STAMP = ("SYNTHETIC DEMO DATA — NOT RESULTS  ·  rendered by make_figures.py --demo  ·  "
               "numbers are illustrative (null-shaped draws), not campaign output")


def stamp_demo(fig: Any) -> None:
    """Mark a figure as synthetic, ON THE ARTIFACT — not merely in a console line.

    WHY (deep review #66). ``--demo`` is the DEFAULT, and it writes fabricated numbers — contrast
    estimates clipped to +/-0.03 with invented CIs, an "illustrative" Bayes factor, ``rng.normal``
    search trajectories — under the EXACT filenames the final PDF wants
    (``F_equivalence_forest.png``, ``F_evidence_for_null.png``, ...), into the SAME
    ``outputs/figures/`` that ``paper/FIGURE_TABLE_MANIFEST.md`` points at, alongside GENUINE figures
    (``F3_stylised_facts`` from the real train window, ``F4_splits_timeline``). The demo status lived
    only in the script docstring and one stdout line, so nothing on the artifact distinguished a
    fabricated forest plot from a real one — and a figure of invented numbers reaching the
    dissertation is the worst failure mode this project has.

    This mirrors the discipline ``eda.build_f3`` already applies (it bakes its provenance footnote
    into the figure, and refuses to render at all from synthetic input rather than produce a
    fabricated one). Post-campaign, ``--results-root`` renders WITHOUT this stamp, so the presence of
    the banner is itself the real-vs-demo test.
    """
    fig.text(0.5, 0.985, _DEMO_STAMP, ha="center", va="top", fontsize=6.5, color="#B3261E",
             bbox=dict(boxstyle="round,pad=0.25", fc="#FDECEA", ec="#B3261E", lw=0.6), zorder=1000)


def render_all(data: dict[str, Any], out: Path, *, demo: bool = True) -> list[Path]:
    """Render the fourteen headline figures from a data bundle; return the saved PNG paths.

    ``demo=True`` (the default, matching the CLI default) stamps every artifact as synthetic (#66).
    """
    apply_house_style()
    saved: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        p = out / name
        if demo:
            stamp_demo(fig)
        savefig(fig, p)
        saved.append(p)
        import matplotlib.pyplot as plt

        plt.close(fig)

    _save(F.equivalence_forest(data["contrasts"]), "F_equivalence_forest.png")
    _save(F.rliable_intervals(data["scores_by_leg"]), "F_rliable_intervals.png")
    _save(F.risk_return_clouds(data["sharpe"], data["cvar"]), "F_risk_return_clouds.png")
    _save(F.evidence_for_null(data["bf01_by_leg"], data["mcs"]), "F_evidence_for_null.png")
    _save(F.reward_code_similarity(data["ast_distance"], data["cand_arms"]), "F_reward_code_similarity.png")
    _save(F.controls_overlay(data["controls_cvar"]), "F_controls_overlay.png")
    _save(F.responsiveness_scatter(data["fed_delta"], data["reward_delta"], rho=data["rho"]),
          "F_responsiveness_scatter.png")
    _save(F.learning_curves(data["curves"]), "F_learning_curves.png")
    # D2 / write-time registry row 38 — Okhrati asked for the estimate AS A FUNCTION OF SEED
    # COUNT, not only its terminal value. Rendered from the Sharpe leg's per-seed scores; the
    # tier list is the REGISTERED ladder and seed_trajectory drops any rung the data has not
    # reached, so the same call is correct mid-campaign and at the full 568.
    _save(F.seed_trajectory(data["scores_by_leg"]["Sharpe"],
                            rungs=(30, 100, 189, 279, 340, 403, 568)),
          "F_seed_trajectory.png")
    _save(F.delisting_robustness(data["delisting"]), "F_delisting_robustness.png")
    # G1-G5 (2026-07-26): the rliable-quartet completion + the risk-lens/finance staples.
    _save(F.performance_profile(data["scores_by_leg"]["Sharpe"]), "F_performance_profile.png")
    _save(F.probability_of_improvement(data["prob_improve"]), "F_probability_of_improvement.png")
    _save(F.return_tail_distribution(data["returns_by_arm"]), "F_return_tail_distribution.png")
    _save(F.equity_drawdown(data["returns_by_arm"], benchmark=data["market_bench"]), "F_equity_drawdown.png")
    _save(F.allocation_heatmap(data["alloc"]), "F_allocation_heatmap.png")
    return saved


def render_advanced(data: dict[str, Any], out: Path, *, demo: bool = True) -> list[Path]:
    """Render the principled static 3-D figures (PDF-safe; they go IN the dissertation); return PNG paths.

    These go IN the PDF, so the ``demo`` stamp matters here most of all (#66).
    """
    from src.viz import advanced as A

    apply_house_style()
    saved: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        if demo:
            stamp_demo(fig)
        savefig(fig, out / name)
        saved.append(out / name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    _save(A.reward_embedding_3d(data["ast_distance"], data["cand_arms"]), "F_reward_embedding_3d.png")
    _save(A.risk_return_generation_3d(data["trajectory"]), "F_risk_return_generation_3d.png")
    _save(A.search_evolution_keyframes(data["frames"]), "F_search_evolution_keyframes.png")
    return saved


def render_schematics(out: Path) -> list[Path]:
    """Render the static schematic figures (data-free; final NOW) — system diagram, splits, prediction tree."""
    from src.viz import schematics as SC

    apply_house_style()
    saved: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        savefig(fig, out / name)
        saved.append(out / name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    _save(SC.system_diagram(), "F1_system_diagram.png")
    _save(SC.prediction_branch(), "F2_prediction_branch.png")
    _save(SC.splits_timeline(), "F4_splits_timeline.png")
    return saved


def render_eda(out: Path) -> list[Path]:
    """Render F3 (stylised facts) from the REAL gold train window; skip loudly when the panel is absent.

    Unlike the headline suite this is NOT synthesisable: F3 is a DATA figure and a fabricated one would
    be worse than none (schematics module note), so a missing licensed gold panel prints the reason and
    returns ``[]`` rather than inventing data. ``build_f3`` also prints the annotated stylised-fact
    numbers (the Data-chapter prose inputs).
    """
    from src.viz.eda import build_f3

    apply_house_style()
    try:
        path, _stats = build_f3(out / "F3_stylised_facts.png")
    except FileNotFoundError as exc:  # synthetic-only install: the licensed gold panel is not shipped
        print(f"[make_figures] F3 stylised facts SKIPPED (gold panel unavailable): {exc}")
        return []
    return [path]


def render_animations(data: dict[str, Any], out: Path, *, n_frames: int = 36) -> list[Path]:
    """Render the supplementary GIF animations to ``out/anim/`` (repo artefacts; PDF references them)."""
    from src.viz import advanced as A

    apply_house_style()
    anim_dir = out / "anim"
    g1 = A.animate_search_evolution(data["frames"], anim_dir / "A_search_evolution.gif", fps=2)
    g2 = A.animate_embedding_rotation(data["ast_distance"], data["cand_arms"],
                                      anim_dir / "A_reward_embedding_rotation.gif", n_frames=n_frames, fps=12)
    return [g1, g2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the headline figure suite (report-only).")
    ap.add_argument("--out", default="outputs/figures", help="output dir (default outputs/figures)")
    ap.add_argument("--results-root", default=None,
                    help="post-campaign results dir to load real data from (not yet wired; uses --demo until then)")
    ap.add_argument("--seed", type=int, default=7, help="synthetic-demo seed")
    ap.add_argument("--no-advanced", action="store_true",
                    help="skip the 3-D figures + GIF animations (headline suite only)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.results_root:
        raise SystemExit(
            "real-data loading is not yet wired (the confirmatory campaign is unrun). Re-run without "
            "--results-root for the synthetic-null demo; the figure functions in src/viz/figures.py are "
            "the post-campaign API (feed per-seed scores + inference outputs)."
        )
    data = synthesize_null(seed=args.seed)
    # demo=True stamps every artifact as synthetic (#66). When --results-root is finally wired,
    # pass demo=False there so the REAL renders carry no banner — the banner is the real-vs-demo test.
    saved = render_all(data, out, demo=True)
    print(f"[make_figures] DEMO (synthetic null) — {len(saved)} headline figures -> {out}/")
    for p in saved:
        print(f"  {p.name}  (+ {p.with_suffix('.pdf').name})")

    sch = render_schematics(out)
    print(f"[make_figures] schematics (data-free) — {len(sch)} -> {out}/")
    for p in sch:
        print(f"  {p.name}  (+ {p.with_suffix('.pdf').name})")

    eda = render_eda(out)
    if eda:
        print(f"[make_figures] EDA (REAL train-window panel) — {len(eda)} -> {out}/")
        for p in eda:
            print(f"  {p.name}  (+ {p.with_suffix('.pdf').name})")

    if not args.no_advanced:
        adv = render_advanced(data, out, demo=True)   # see the note above (#66)
        print(f"[make_figures] advanced static 3-D figures — {len(adv)} -> {out}/")
        for p in adv:
            print(f"  {p.name}  (+ {p.with_suffix('.pdf').name})")
        anims = render_animations(data, out)
        print(f"[make_figures] supplementary animations — {len(anims)} GIF -> {out}/anim/")
        for p in anims:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
