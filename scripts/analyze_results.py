"""Analyze prototype results -> directional GREEN/AMBER/RED verdict (MASTER_EXECUTION_PLAN P6).

Reads the per-arm candidate archives produced by ``scripts/run_prototype.py`` and computes:
  - **H2 directional read**: distributional vs scalar vs placebo vs scalar_cvar5 — each arm's winner
    validation Deflated Sharpe and the rliable IQM of its candidate population (Agarwal et al. 2021);
  - **H4 read**: distributional/LLM vs random_search vs bayes_opt (winner fitness);
  - **Difference tests** between the LLM arms' winners on the archived validation return series:
    studentized stationary-bootstrap Sharpe-difference (Ledoit-Wolf 2008 in spirit) and two-sample
    CVaR-difference (Politis-Romano 1994), via ``src.inference.bootstrap``;
  - **Interpretability** (the §6.1 gate): does the distributional winner's reward *code* actually
    reference the tail statistics it was fed? A metric gap NOT accompanied by tail-responsive code is
    suspect (noise / reward-hacking — Skalse 2022; Hadfield-Menell 2017);
  - a **compute-accounting** table (per-arm candidate count + wall-clock) showing matched compute.

DIRECTIONAL / plumbing only (1 seed; review M3/M4) — **no number here enters the dissertation**; this
is the go/no-go on the mechanism before the multi-seed campaign. PBO/CSCV and the FZ comparative-ES
forecast backtest are campaign tools (they need CPCV folds / a common test set) — noted, not run here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Substrings in a reward's source that indicate it uses tail / risk structure.
_TAIL_TERMS = ("cvar", "tail", "quantile", "percentile", "sort", "drawdown", "var", "std", "min(")
_LLM_ARMS = ("distributional", "scalar", "placebo", "scalar_cvar5")


def load_arms(root: str | Path) -> dict[str, list[dict]]:
    """Load ``{arm: [records]}`` from ``root/<arm>/<candidate>/record.json``."""
    from src.io.results import load_all

    root = Path(root)
    arms: dict[str, list[dict]] = {}
    if not root.is_dir():
        return arms
    for d in sorted(root.iterdir()):
        if d.is_dir():
            recs = load_all(d)
            if recs:
                arms[d.name] = recs
    return arms


def _winner(records: list[dict]) -> dict | None:
    return max(records, key=lambda r: r["metrics"]["val_fitness"]) if records else None


def arm_fitnesses(records: list[dict]) -> np.ndarray:
    return np.asarray([r["metrics"]["val_fitness"] for r in records], dtype=float)


def winner_returns(records: list[dict]) -> np.ndarray | None:
    w = _winner(records)
    vr = w["metrics"].get("val_returns") if w else None
    # Guard on `is None` + `.size`, never the array's truthiness (final-audit #30: `if vr` raises
    # "truth value of an array ... is ambiguous" for an ndarray-valued val_returns). Mirrors
    # analyze_campaign._val_returns.
    if vr is None:
        return None
    arr = np.asarray(vr, dtype=float).ravel()
    return arr if arr.size else None


def interpretability(source: str | None) -> dict[str, Any]:
    """Does the reward source reference tail / risk structure? (§6.1 mechanism gate.)

    ``uses_tail`` is a **COARSE DIRECTIONAL flag, not an arm-discriminating measurement.** Because
    ``_TAIL_TERMS`` includes near-ubiquitous tokens (``"var"``, ``"std"``, ``"min("``) that appear in
    almost any non-trivial reward, the flag **near-saturates across all arms** and therefore **cannot
    discriminate** the distributional arm from scalar/placebo/search. It is a code-smell / sanity probe
    only — **no number derived from it enters the dissertation.** The **authoritative per-arm gate is
    ``scripts/inspect_rewards._was_fed_tail``**, which judges responsiveness from what the LLM *saw* (the
    rendered prompt/feedback block), not from substring-matching the generated source. Use that, not this,
    for any "was the designer fed / did it use the tail" claim.
    """
    s = (source or "").lower()
    terms = sorted({t for t in _TAIL_TERMS if t in s})
    return {"uses_tail": len(terms) > 0, "terms": terms}


def directional_verdict(fits: dict[str, float], uses_tail: bool) -> str:
    """GREEN/AMBER/RED per MASTER_EXECUTION_PLAN §6.1 (directional, Henderson 1-seed caveat)."""
    d, sc, pl = fits.get("distributional"), fits.get("scalar"), fits.get("placebo")
    if d is None or sc is None or pl is None:
        return "INCOMPLETE"
    beats_scalar, beats_placebo = d > sc, d > pl
    if beats_scalar and beats_placebo and uses_tail:
        return "GREEN"
    if (not beats_scalar) and (not beats_placebo) and not uses_tail:
        return "RED"
    return "AMBER"


def analyze(root: str | Path) -> dict[str, Any]:
    """Compute the full directional analysis dict from an archive root."""
    from src.inference.bootstrap import cvar_difference_test, sharpe_difference_test
    from src.inference.reporting import iqm, performance_profile, stratified_bootstrap_ci

    arms = load_arms(root)
    rng = np.random.default_rng(0)
    per_arm: dict[str, Any] = {}
    fits: dict[str, float] = {}
    for arm, recs in arms.items():
        w = _winner(recs)
        f = arm_fitnesses(recs)
        point, lo, hi = (
            stratified_bootstrap_ci(f, n_boot=2000, rng=rng) if f.size else (np.nan, np.nan, np.nan)
        )
        per_arm[arm] = {
            "n_candidates": len(recs),
            "winner_fitness": float(w["metrics"]["val_fitness"]) if w else None,
            "winner_id": w["candidate_id"] if w else None,
            "iqm": float(iqm(f)) if f.size else None,
            "iqm_ci": [float(point), float(lo), float(hi)],
            "has_val_returns": winner_returns(recs) is not None,
        }
        if w:
            fits[arm] = float(w["metrics"]["val_fitness"])

    # rliable performance profiles (score distributions; Agarwal et al. 2021): each arm's candidate-fitness
    # survival fraction over a SHARED tau grid, so arms compare as DISTRIBUTIONS. A profile that dominates
    # another at every tau is a strictly stronger claim than an IQM gap. Pointwise stratified-bootstrap band.
    profiles: dict[str, Any] = {}
    pooled = [a for a in (arm_fitnesses(recs) for recs in arms.values()) if a.size]
    if pooled:
        all_f = np.concatenate(pooled)
        taus = np.linspace(float(all_f.min()), float(all_f.max()), 50)
        profiles["taus"] = taus.tolist()
        for arm, recs in arms.items():
            f = arm_fitnesses(recs)
            if f.size:
                prof, plo, phi = performance_profile(f, taus, n_boot=2000, rng=rng)
                profiles[arm] = {
                    "profile": prof.tolist(),
                    "ci_low": plo.tolist(),
                    "ci_high": phi.tolist(),
                }

    # Interpretability of the distributional winner's CODE (the mechanism gate).
    interp: dict[str, Any] = {"uses_tail": False, "terms": []}
    if "distributional" in arms:
        dw = _winner(arms["distributional"])
        interp = interpretability(dw.get("reward_source") if dw else None)

    # Difference tests between LLM-arm winners (need archived validation return series).
    diffs: dict[str, Any] = {}
    dist_ret = winner_returns(arms["distributional"]) if "distributional" in arms else None
    if dist_ret is not None:
        for other in ("scalar", "placebo", "scalar_cvar5"):
            other_ret = winner_returns(arms[other]) if other in arms else None
            if other_ret is not None and other_ret.shape == dist_ret.shape and dist_ret.size > 8:
                diffs[f"distributional_vs_{other}"] = {
                    "sharpe": sharpe_difference_test(dist_ret, other_ret, n_boot=2000, rng=rng),
                    "cvar": cvar_difference_test(dist_ret, other_ret, n_boot=2000, rng=rng),
                }

    verdict = directional_verdict(fits, bool(interp["uses_tail"]))
    return {
        "verdict": verdict,
        "per_arm": per_arm,
        "winner_fitness": fits,
        "interpretability": interp,
        "difference_tests": diffs,
        "performance_profiles": profiles,
        "n_arms": len(arms),
    }


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:+.5f}" if isinstance(v, float) else str(v))


def write_report(result: dict[str, Any], root: str | Path) -> Path:
    """Write a markdown report + analysis.json next to the archives."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prototype analysis — DIRECTIONAL (plumbing only; no number enters the dissertation)",
        "",
        f"**Verdict: {result['verdict']}** (MASTER_EXECUTION_PLAN §6.1; Henderson 1-seed = directional).",
        "",
        "## Per-arm (winner validation Deflated Sharpe + rliable IQM of the candidate population)",
        "| arm | n | winner fitness | IQM | IQM 95% CI |",
        "|---|---|---|---|---|",
    ]
    for arm, a in result["per_arm"].items():
        ci = a["iqm_ci"]
        lines.append(
            f"| {arm} | {a['n_candidates']} | {_fmt(a['winner_fitness'])} | {_fmt(a['iqm'])} | "
            f"[{ci[1]:+.5f}, {ci[2]:+.5f}] |"
        )
    interp = result["interpretability"]
    lines += [
        "",
        "## Interpretability (the mechanism gate)",
        f"- distributional winner uses tail/risk terms: **{interp['uses_tail']}** "
        f"(found: {', '.join(interp['terms']) or 'none'})",
        "",
        "## Difference tests (LLM-arm winners; stationary-bootstrap)",
    ]
    if result["difference_tests"]:
        for k, v in result["difference_tests"].items():
            lines.append(
                f"- **{k}**: Sharpe Δ p={v['sharpe']['pvalue']:.3f} (stat {v['sharpe']['stat']:+.3f}); "
                f"CVaR Δ p={v['cvar']['pvalue']:.3f} (stat {v['cvar']['stat']:+.3f})"
            )
    else:
        lines.append("- (no archived validation return series — run the prototype first)")
    lines += [
        "",
        "## Notes",
        "- PBO/CSCV and the FZ comparative-ES forecast backtest are campaign tools (CPCV folds / common"
        " test set) — implemented in `src/inference/` and run in the campaign, not on this 1-seed prototype.",
        "- Delisting=`liquidate_to_cash` + the held 2005 cohort bias the measured tails (review M3/M4):"
        " this verdict is a go/no-go on the mechanism, not a result.",
    ]
    report = root / "analysis_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (root / "analysis.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze prototype results -> directional verdict (P6).")
    p.add_argument("--root", default=None, help="Archive root (default: config prototype output_dir).")
    args = p.parse_args()
    if args.root is None:
        from src.utils.config import load_config

        args.root = str(load_config("prototype").get("output_dir", "outputs/prototype"))
    result = analyze(args.root)
    report = write_report(result, args.root)
    print(f"[analyze_results] verdict: {result['verdict']}  ({result['n_arms']} arms)  -> {report}")
    for arm, a in result["per_arm"].items():
        print(f"  {arm:>14}: winner={_fmt(a['winner_fitness'])} iqm={_fmt(a['iqm'])} n={a['n_candidates']}")


if __name__ == "__main__":
    main()
