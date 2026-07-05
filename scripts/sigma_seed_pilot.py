"""σ_D / ρ extractor — the clean seeds-on-winners power pilot (the freeze-blocker for ``n_seeds``).

Purpose
-------
The headline H2 test is the **paired per-seed IQM bootstrap** (``src.inference.bootstrap.
paired_seed_difference_test``), whose minimum detectable effect (MDE) scales ~linearly with
**σ_seed** (the seed-to-seed SD of the per-seed score within an arm) and shrinks with the
**pairing correlation ρ** across the shared training seeds. ``scripts/power_analysis.py`` *consumes*
σ_seed and ρ (``--sigma-seed``, ``--rho``) but nothing *measures* them — until this freeze, the power
analysis runs on the directional proxy ``σ_seed = 0.36`` (which folds in reward-DESIGN variance an
honest, FIXED-reward pilot strips out). ``scripts/determine_design.py`` therefore holds ``n_seeds``
**PENDING** because ``evidence['sigma_seed_pilot']`` is never set.

This module closes that gap. Given the **clean pilot** — *two fixed rewards* re-trained across a
*shared* seed set (Common Random Numbers) — it measures, per headline statistic (annualised Sharpe
for the H2-RA leg, CVaR-5% for the H2-Tail leg):

* **σ_a, σ_b** — per-arm seed-to-seed SD (the ``--sigma-seed`` input to ``power_analysis``);
* **σ_D = SD(a − b)** — the paired-difference SD the paired test's variance actually depends on;
* **ρ = corr(a, b)** — the cross-arm pairing correlation (defined only with ≥ 2 arms; this is *why*
  the pilot needs two fixed rewards, not one).

The CRN identity ``σ_D² = σ_a² + σ_b² − 2·ρ·σ_a·σ_b`` makes explicit that **pairing only helps when
ρ > 0** (Glasserman, *Monte Carlo Methods in Financial Engineering*; L'Ecuyer on CRN). A measured
ρ that is not reliably > 0 means CRN buys nothing and the seed count must be sized off σ_D directly;
the sign of a small pilot ρ is itself unresolved at n≈15 (SE(ρ)≈0.29), so read this as "no reliable
pairing benefit", not a substantive negative correlation.

Seed-count decision (SESOI-derived, NOT an arbitrary σ_D > 0.10 threshold)
-------------------------------------------------------------------------
The equivalence claim is *achievable at n seeds* iff the n-seed 80%-power MDE, mapped into the SESOI's
**validation-DSR** units via :func:`scripts.power_analysis.sharpe_mde_to_dsr`, is **≤ the 0.05 SESOI**.
:func:`estimate_seed_count` returns the **smallest n** on a grid for which that holds (preview via the
doc's *measured* one-sided-IUT scaling ``MDE@80% ≈ k₈₀(ρ)·σ_seed·√(n_ref/n)``; the *authoritative*
MDE is the Monte-Carlo ``power_analysis.py`` re-run this module prints the exact command for). If even
the grid ceiling cannot bring the DSR-MDE ≤ SESOI, equivalence is **not achievable** at a feasible n →
the headline is reported as a rigorous **bounded-effect / INCONCLUSIVE** result (the pre-committed null
framing), never as practical equivalence.

Either outcome is defensible and pre-registered; **this pilot decides which framing the headline takes.**

Deliverable contract
--------------------
* Pure estimators (:func:`paired_seed_stats`, :func:`k80_for_rho`, :func:`estimate_seed_count`) —
  numpy only, no torch, no I/O, unit-tested.
* A driver (:func:`run_pilot`) shaped to the campaign records that **degrades gracefully** —
  ``status="skipped"`` with a reason, NEVER a crash — when < 2 shared seeds exist for a leg.
* Reuses ``variance_decomposition._per_seed_scores_from_records`` (the headline per-seed unit) — no
  duplicate loader.

Sources: Agarwal et al. 2021 (rliable IQM); Colas et al. 2018 (pilot-for-power); Glasserman 2004 /
L'Ecuyer (CRN variance reduction); ``outputs/CAMPAIGN_power.md`` (the measured k₈₀(ρ) scaling table).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "DEFAULT_SESOI",
    "DEFAULT_TRACK_LENGTH",
    "N_REF",
    "paired_seed_stats",
    "k80_for_rho",
    "estimate_seed_count",
    "run_pilot",
    "render_markdown",
    "build_parser",
    "main",
]

_LOG = logging.getLogger(__name__)

#: Frozen SESOI (validation-DSR units; PREREGISTRATION §10 R12) and the EXECUTED validation track
#: length used by the Sharpe→DSR map — Split C (R73): expected_windows.univ5.val = [3081, 3775) =
#: **694** sessions (corrected 2026-07-02 from the stale pre-Split-C 3×252=756; matches
#: power_analysis.VALIDATION_TRACK_LENGTH).
DEFAULT_SESOI: float = 0.05
DEFAULT_TRACK_LENGTH: int = 694

#: The reference seed count at which the doc's k₈₀(ρ) table is measured (the planned floor).
N_REF: int = 30

#: The MEASURED one-sided-IUT k₈₀ (MDE@80% in σ_seed units) at n = N_REF, by pairing ρ, read from
#: ``outputs/CAMPAIGN_power.md`` ("Pairing-correlation (ρ) sensitivity" + σ_seed grid). Used for the
#: at-a-glance preview only; the authoritative MDE is the Monte-Carlo ``power_analysis.py`` re-run.
_RHO_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)
# Synced 2026-07-05 to the REGENERATED outputs/CAMPAIGN_power.md measured table (σ_D-farm inputs, the
# 2026-07-03 build): the previous grid (0.71, 0.65, 0.59, 0.51) was the OLD σ=0.36 build's values, so
# the docstring's "read from CAMPAIGN_power.md" provenance claim had silently gone false. Preview-only
# either way — the authoritative MDE is the Monte-Carlo power_analysis.py re-run.
_K80_GRID: tuple[float, ...] = (0.72, 0.61, 0.53, 0.44)


# --------------------------------------------------------------------------- #
# Pure estimators                                                              #
# --------------------------------------------------------------------------- #
def paired_seed_stats(
    a: dict[int, float] | Sequence[float],
    b: dict[int, float] | Sequence[float],
) -> dict[str, Any]:
    """Per-arm σ, paired-difference σ_D, and pairing ρ from two arms' per-seed scores.

    ``a`` / ``b`` are either ``{seed: score}`` dicts (aligned on the SHARED seeds — the CRN design)
    or already-aligned sequences (same order). With dicts, only seeds present in BOTH arms are used
    (the paired set). Returns ``status="skipped"`` (never raises) when fewer than 2 shared seeds
    exist — σ and ρ are undefined below n = 2.

    Returns
    -------
    dict
        ``{"n_shared", "shared_seeds", "sigma_a", "sigma_b", "sigma_seed", "sigma_d", "rho",
        "mean_diff", "crn_identity_ok", "status", ["reason"]}``. ``sigma_seed`` is the pooled
        per-arm SD ``√((σ_a² + σ_b²)/2)`` (the single ``--sigma-seed`` input). All SDs use ddof=1.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        shared = sorted(set(a) & set(b))
        av = np.array([a[s] for s in shared], dtype=float)
        bv = np.array([b[s] for s in shared], dtype=float)
    else:
        av = np.asarray(list(a), dtype=float).ravel()
        bv = np.asarray(list(b), dtype=float).ravel()
        n = min(av.size, bv.size)
        av, bv = av[:n], bv[:n]
        shared = list(range(n))

    finite = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[finite], bv[finite]
    shared = [s for s, keep in zip(shared, finite.tolist()) if keep]
    n = int(av.size)

    base: dict[str, Any] = {
        "n_shared": n,
        "shared_seeds": shared,
        "sigma_a": None,
        "sigma_b": None,
        "sigma_seed": None,
        "sigma_d": None,
        "rho": None,
        "mean_diff": None,
    }
    if n < 2:
        base.update(status="skipped", reason=f"only {n} shared finite seed(s); need >= 2")
        return base

    sa = float(np.std(av, ddof=1))
    sb = float(np.std(bv, ddof=1))
    diff = av - bv
    sd = float(np.std(diff, ddof=1))
    sigma_seed = float(np.sqrt((sa * sa + sb * sb) / 2.0))

    # Pairing correlation; undefined if either arm has zero variance across seeds (degenerate).
    if sa > 0.0 and sb > 0.0:
        rho = float(np.corrcoef(av, bv)[0, 1])
    else:
        rho = None

    # CRN identity check (descriptive): σ_D² ?= σ_a² + σ_b² − 2ρσ_aσ_b (holds up to ddof rounding).
    crn_ok = None
    if rho is not None:
        implied = sa * sa + sb * sb - 2.0 * rho * sa * sb
        crn_ok = bool(np.isclose(sd * sd, max(implied, 0.0), rtol=1e-6, atol=1e-9))

    base.update(
        sigma_a=sa,
        sigma_b=sb,
        sigma_seed=sigma_seed,
        sigma_d=sd,
        rho=rho,
        mean_diff=float(diff.mean()),
        crn_identity_ok=crn_ok,
        status="ok",
    )
    return base


def k80_for_rho(rho: float | None) -> float:
    """Measured one-sided-IUT k₈₀ (MDE@80% in σ_seed units, n = N_REF) interpolated at ``rho``.

    Linear interpolation over the ``outputs/CAMPAIGN_power.md`` table {0:0.71, 0.3:0.65, 0.5:0.59,
    0.7:0.51}. ``rho`` below 0 (CRN HURTS — pairing provides no benefit) clamps to the ρ=0 value
    0.71 *as a floor*; in that regime the closed form understates the true MDE, so the preview flags
    it and the authoritative figure must come from the Monte-Carlo ``power_analysis.py`` re-run.
    ``rho`` above the grid clamps to 0.51 (``np.interp`` edge behaviour).
    """
    if rho is None:
        return float(_K80_GRID[0])
    r = max(0.0, float(rho))  # negative ρ: no pairing benefit -> floor at the ρ=0 k₈₀
    return float(np.interp(r, _RHO_GRID, _K80_GRID))


def estimate_seed_count(
    sigma_seed: float,
    rho: float | None,
    *,
    sesoi: float = DEFAULT_SESOI,
    track_length: int = DEFAULT_TRACK_LENGTH,
    n_grid: Sequence[int] = (30, 35, 40, 45, 50),
) -> dict[str, Any]:
    """Smallest n whose 80%-power MDE (in validation-DSR units) is ≤ the SESOI — the seed decision.

    Preview MDE scaling (the doc's measured one-sided-IUT relation):
    ``MDE_sharpe(n) ≈ k₈₀(ρ) · σ_seed · √(N_REF / n)`` → ``MDE_dsr = sharpe_mde_to_dsr(MDE_sharpe)``.
    The recommended n is the smallest grid point with ``MDE_dsr ≤ sesoi``. If no grid point reaches
    it, ``achievable=False`` → report bounded-effect / INCONCLUSIVE (not equivalence).

    The MDE here is a fast PREVIEW; the authoritative value is the Monte-Carlo
    ``power_analysis.py --sigma-seed <σ_seed> --rho <ρ>`` (printed by the driver).
    """
    # sharpe_mde_to_dsr lives in the sibling script; import lazily so the pure stats need no deps.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from power_analysis import sharpe_mde_to_dsr  # type: ignore[import-not-found]

    k80 = k80_for_rho(rho)
    ladder: list[dict[str, Any]] = []
    recommended: int | None = None
    for n in n_grid:
        mde_sharpe = float(k80 * sigma_seed * np.sqrt(N_REF / float(n)))
        mde_dsr = float(sharpe_mde_to_dsr(mde_sharpe, track_length=track_length))
        ok = bool(mde_dsr <= sesoi)
        ladder.append(
            {"n": int(n), "mde_sharpe": mde_sharpe, "mde_dsr": mde_dsr, "meets_sesoi": ok}
        )
        if ok and recommended is None:
            recommended = int(n)

    achievable = recommended is not None
    return {
        "k80_rho": float(k80),
        "rho_used": (float(rho) if rho is not None else None),
        "rho_negative_warning": bool(rho is not None and rho < 0.0),
        "sesoi_dsr": float(sesoi),
        "track_length": int(track_length),
        "ladder": ladder,
        "recommended_n": recommended,
        "achievable_equivalence": achievable,
        "headline_framing": (
            "practical_equivalence" if achievable else "bounded_effect_inconclusive"
        ),
    }


# --------------------------------------------------------------------------- #
# Driver over campaign records (graceful — never crashes)                      #
# --------------------------------------------------------------------------- #
def _statistics() -> dict[str, Callable[[np.ndarray], float]]:
    """The two headline per-seed score functionals: Sharpe (H2-RA) and CVaR-5% (H2-Tail)."""
    from src.inference.bootstrap import cvar, sharpe_ratio

    return {"sharpe": sharpe_ratio, "cvar_05": lambda r: cvar(r, 0.05)}


def run_pilot(
    records: list[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
    sesoi: float = DEFAULT_SESOI,
    track_length: int = DEFAULT_TRACK_LENGTH,
    n_grid: Sequence[int] = (30, 35, 40, 45, 50),
) -> dict[str, Any]:
    """Measure σ_a/σ_b/σ_D/ρ + the seed-count decision for both headline statistics, from records.

    ``records`` is one campaign archive (``analyze_campaign.load_campaign_records`` output) holding
    the two FIXED-reward arms' per-seed TEST records over a SHARED seed set. Reuses
    ``variance_decomposition._per_seed_scores_from_records`` (the headline per-seed unit). The
    seed-count decision is driven by the SHARPE leg (the H2-RA equivalence metric); the CVaR leg's
    σ_D/ρ are reported alongside for the tail leg.
    """
    try:
        from variance_decomposition import _per_seed_scores_from_records  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        # Sibling-script import: only resolvable when scripts/ is on sys.path. The full suite gets it
        # from an earlier test file's insert — a solo run (or bare import) did not (latent isolation
        # flaw, found 2026-07-02). Self-provide the path instead of depending on test order.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from variance_decomposition import _per_seed_scores_from_records  # type: ignore[import-not-found]

    stats = _statistics()
    per_stat: dict[str, Any] = {}
    for name, fn in stats.items():
        a = _per_seed_scores_from_records(records, arm_a, fn)
        b = _per_seed_scores_from_records(records, arm_b, fn)
        ps = paired_seed_stats(a, b)
        decision = (
            estimate_seed_count(
                ps["sigma_seed"], ps["rho"], sesoi=sesoi, track_length=track_length, n_grid=n_grid
            )
            if ps["status"] == "ok"
            else {"status": "skipped", "reason": ps.get("reason")}
        )
        per_stat[name] = {"stats": ps, "decision": decision}

    sharpe_dec = per_stat.get("sharpe", {}).get("decision", {})
    return {
        "contrast": f"{arm_a}|{arm_b}",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "per_statistic": per_stat,
        "recommended_n": sharpe_dec.get("recommended_n"),
        "achievable_equivalence": sharpe_dec.get("achievable_equivalence"),
        "headline_framing": sharpe_dec.get("headline_framing"),
        # The evidence flag determine_design reads to flip n_seeds PENDING -> DETERMINED.
        "sigma_seed_pilot": bool(per_stat.get("sharpe", {}).get("stats", {}).get("status") == "ok"),
    }


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(result: dict[str, Any]) -> str:
    """Render the σ_D/ρ measurement + seed-count decision as markdown (regenerates the power doc fields)."""
    a, b = result.get("arm_a", "?"), result.get("arm_b", "?")
    lines = [
        "# σ_D / ρ pilot — clean seeds-on-winners power measurement",
        "",
        f"Two FIXED rewards (`{a}`, `{b}`) re-trained across a SHARED seed set (Common Random "
        "Numbers). Measures the per-arm seed SD (σ_seed, the `power_analysis --sigma-seed` input), "
        "the paired-difference SD (σ_D), and the pairing correlation (ρ, the `--rho` input). The "
        "seed-count decision keeps `n_seeds = 30` iff the 80%-power MDE maps below the 0.05-DSR "
        "SESOI; otherwise it raises n, or reports bounded-effect/INCONCLUSIVE.",
        "",
        "| statistic | n | σ_a | σ_b | σ_seed | σ_D | ρ | CRN helps (ρ>0) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, blk in result.get("per_statistic", {}).items():
        s = blk.get("stats", {})
        rho = s.get("rho")
        crn = "yes" if (isinstance(rho, float) and rho > 0) else ("NO (ρ≤0)" if rho is not None else "n/a")
        lines.append(
            f"| {name} | {s.get('n_shared', '?')} | {_fmt(s.get('sigma_a'))} | "
            f"{_fmt(s.get('sigma_b'))} | {_fmt(s.get('sigma_seed'))} | {_fmt(s.get('sigma_d'))} | "
            f"{_fmt(rho, 3)} | {crn} |"
        )

    sharpe = result.get("per_statistic", {}).get("sharpe", {})
    dec = sharpe.get("decision", {})
    lines += ["", "## Seed-count decision (Sharpe / H2-RA leg)"]
    if dec.get("ladder"):
        lines += [
            f"k₈₀(ρ={_fmt(dec.get('rho_used'), 3)}) = {_fmt(dec.get('k80_rho'), 3)} · "
            f"SESOI = {_fmt(dec.get('sesoi_dsr'), 3)} DSR · T = {dec.get('track_length')}.",
            "",
            "| n_seeds | MDE@80% (Sharpe) | MDE@80% (DSR) | ≤ SESOI? |",
            "|---|---|---|---|",
        ]
        for row in dec["ladder"]:
            lines.append(
                f"| {row['n']} | {_fmt(row['mde_sharpe'])} | {_fmt(row['mde_dsr'])} | "
                f"{'OK' if row['meets_sesoi'] else '—'} |"
            )
        if dec.get("achievable_equivalence"):
            lines += [
                "",
                f"**Decision: n_seeds = {dec['recommended_n']}** → the MDE maps below the SESOI; the "
                "headline can be banked as **practical equivalence (TOST)**.",
            ]
        else:
            lines += [
                "",
                "**Decision: equivalence NOT achievable on the seed grid** → report a rigorous "
                "**bounded-effect / INCONCLUSIVE** result (the pre-committed null framing), not "
                "equivalence. (Raising seeds further has diminishing returns; the binding quantity "
                "is σ_seed/SESOI, not n.)",
            ]
        if dec.get("rho_negative_warning"):
            lines += [
                "",
                "> NOTE: Measured ρ ≤ 0 (not reliably > 0): Common Random Numbers buy no reliable variance "
                "reduction here (σ_D ≥ √(σ_a²+σ_b²)), so the seed count is sized off σ_D directly. Sign "
                "caveat: at n=15, SE(ρ) ≈ 1/√12 ≈ 0.29, so a small point ρ (e.g. −0.14) is NOT statistically "
                "distinguishable from 0 — read ρ ≤ 0 as 'CRN gives no reliable pairing benefit', not as a "
                "substantive negative correlation. The closed-form preview understates the MDE — trust the "
                "Monte-Carlo `power_analysis.py` figure below (its simulator draws the measured ρ, "
                "sign-preserving, since the signed-ρ fix, 2026-07-03).",
            ]
    else:
        lines.append(f"_Skipped — {dec.get('reason', 'insufficient shared seeds')}._")

    # The exact authoritative re-run command.
    s = sharpe.get("stats", {})
    if s.get("status") == "ok":
        lines += [
            "",
            "## Authoritative MDE (Monte-Carlo)",
            "Re-run the real paired-bootstrap power simulator with the measured inputs to finalise "
            "`outputs/CAMPAIGN_power.md`:",
            "```bash",
            f"python -m scripts.power_analysis --sigma-seed {_fmt(s.get('sigma_seed'), 4)} "
            f"--rho {_fmt(s.get('rho'), 4)} --n-seeds {result.get('recommended_n') or 30} "
            "--out outputs/CAMPAIGN_power.md",
            "```",
        ]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="σ_D/ρ extractor — clean seeds-on-winners power pilot (the n_seeds freeze-blocker).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Archive root holding the two fixed-reward arms' per-seed CRN records "
        "(default: outputs/sigma_pilot — where run_sigma_pilot_train.py writes).",
    )
    parser.add_argument("--arm-a", default="differential_sharpe", help="First fixed reward (CRN arm A).")
    parser.add_argument("--arm-b", default="return_minus_cvar", help="Second fixed reward (CRN arm B).")
    parser.add_argument("--sesoi", type=float, default=DEFAULT_SESOI, help="SESOI in validation-DSR units.")
    parser.add_argument("--track-length", type=int, default=DEFAULT_TRACK_LENGTH, help="Val track length T.")
    parser.add_argument(
        "--n-grid",
        default="30,35,40,45,50",
        help="Comma list of candidate seed counts for the decision ladder.",
    )
    parser.add_argument("--out", default=None, help="Output dir (default: the run root).")
    return parser


def _resolve_root(arg_root: str | None) -> str:
    # Default = outputs/sigma_pilot, where run_sigma_pilot_train.py actually writes the CRN cells
    # (batch-5 m2, 2026-07-03): the old default (campaign output_dir = outputs/campaign) pointed at an
    # archive with ZERO pilot cells, so a default-root run wrote a status="skipped" sigma_seed_pilot.json
    # into outputs/campaign — exactly where resume_brief.py used to look for "sigma_D pilot done".
    if arg_root:
        return str(arg_root)
    return "outputs/sigma_pilot"


def main() -> None:
    # Windows consoles default to a legacy codepage (e.g. cp1251) that cannot encode the σ/ρ glyphs
    # this analyzer prints (batch-5 m1, 2026-07-03 — the farm twin already carries this block): force
    # UTF-8 so the decision banner never crashes AFTER the md/json artifacts were written (a confusing
    # non-zero exit in the post-farm chain). MUST precede parse_args(): the parser description itself
    # contains σ_D, so --help would crash first on a legacy codepage.
    import sys

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    root = _resolve_root(args.root)
    n_grid = tuple(int(x) for x in str(args.n_grid).split(","))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_campaign import load_campaign_records  # type: ignore[import-not-found]

    try:
        records = load_campaign_records(root)
    except Exception as exc:  # noqa: BLE001 — a bad root must not crash; report skipped
        _LOG.warning("sigma_seed_pilot: failed to load %r: %s", root, exc)
        records = []

    result = run_pilot(
        records,
        arm_a=args.arm_a,
        arm_b=args.arm_b,
        sesoi=float(args.sesoi),
        track_length=int(args.track_length),
        n_grid=n_grid,
    )

    out_dir = Path(args.out) if args.out else Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sigma_seed_pilot.md").write_text(render_markdown(result), encoding="utf-8")
    (out_dir / "sigma_seed_pilot.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    sharpe = result.get("per_statistic", {}).get("sharpe", {}).get("stats", {})
    print(f"[sigma_seed_pilot] {args.arm_a} | {args.arm_b}  (root={root})")
    print(
        f"  Sharpe leg: n={sharpe.get('n_shared')} σ_seed={_fmt(sharpe.get('sigma_seed'))} "
        f"σ_D={_fmt(sharpe.get('sigma_d'))} ρ={_fmt(sharpe.get('rho'), 3)}"
    )
    if result.get("achievable_equivalence") is True:
        print(f"  DECISION: n_seeds = {result['recommended_n']} -> practical EQUIVALENCE bankable.")
    elif result.get("achievable_equivalence") is False:
        print("  DECISION: equivalence NOT achievable on the grid -> bounded-effect / INCONCLUSIVE.")
    else:
        print("  DECISION: skipped (insufficient shared seeds) -> run the pilot first.")
    print(f"  wrote {out_dir / 'sigma_seed_pilot.md'}")


if __name__ == "__main__":
    main()
