"""Multi-root leg aggregation — leg archives -> the cross-model synthesis input (R80/R82).

Each replication leg archives its test-leg records under its own TEST sub-root (disjoint by
construction: ``run_campaign_cluster`` forces ``--root-suffix leg_<label>``, so the leg root is
``<output_dir>/test_leg_<sanitized label>``), holding one record per arm at
``<arm>/<arm>-s<seed>`` via ``src.io.results``. This module walks a
mapping of ``leg label -> archive root`` and assembles exactly the input contract of
:mod:`src.inference.cross_model`:

* per-seed CVaR-5% difference arrays (distributional − scalar, seed-aligned on the COMMON floor
  seed set), computed from each record's realized ``metrics['test_returns']`` with the empirical
  worst-⌈αT⌉ mean under the signed-return convention (more negative = worse tail);
* the registered T0-floor leg-inclusion flag: a leg votes in the synthesis ONLY if the mean
  per-seed Sharpe of BOTH contrasted arms clears the supplied naive-benchmark floor (legs that
  fail report as authoring/search failures — a finding, never a vote);
* per-arm per-seed Sharpe arrays (for the pair-DiD instrument and the reporting tables).

Missing seeds fail LOUD (a silent subset would quietly change the paired estimator); a wholly
missing arm marks the leg failed with the reason recorded. Deterministic; read-only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.bootstrap import sharpe_ratio
from src.io.results import load_run

__all__ = ["empirical_cvar", "per_seed_series", "t0_floor_sharpe", "leg_results_for_synthesis",
           "discover_leg_roots", "cross_model_synthesis"]

#: The registered T0 floor's source arm and seed set (amendment R84). NOT a free parameter — see
#: :func:`t0_floor_sharpe`.
T0_FLOOR_ARM = "equal_weight"
T0_FLOOR_SEEDS = list(range(30))


def empirical_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Mean of the worst ``ceil(alpha*T)`` signed returns (house convention: more negative = worse).

    Numerically this IS ``bootstrap.cvar`` — verified 2026-07-27 (deep review #114) over 4,000 random
    draws with NaN/-inf injected: max absolute difference 2.8e-17, i.e. summation order alone
    (``np.partition`` vs ``np.sort`` over the same k smallest). It is kept separate for the ONE
    behaviour it deliberately does not share: an all-non-finite input RAISES here, where the canonical
    returns NaN — this module's whole contract is that absence fails loud rather than propagating a
    quiet NaN into a registered cross-model bound.

    ``alpha`` is validated on the same terms as the canonical (#114). It previously was not, so
    ``alpha=0`` returned the single worst return (``k = max(1, 0)``) and ``alpha>1`` silently returned
    the mean of everything — a wrong CVaR rather than an error, in the one module that otherwise fails
    loud on every missing input. Not reachable from production today (``cross_model_synthesis`` does
    not expose ``alpha``; the default 0.05 is what runs), but ``per_seed_series`` and
    ``leg_results_for_synthesis`` both take it as a public parameter.
    """
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"empirical_cvar: alpha must lie in (0, 1]; got {alpha!r}")
    arr = np.asarray(returns, dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])  # row 30c: strip non-finite (mirrors bootstrap.cvar);
    if arr.size == 0:                     # -inf would otherwise poison the tail mean silently
        raise ValueError("empirical_cvar: no finite returns")
    k = max(1, math.ceil(alpha * arr.size))
    return float(arr[:k].mean())


def per_seed_series(
    root: str | Path, arm: str, seeds: list[int], alpha: float = 0.05
) -> dict[str, np.ndarray]:
    """Per-seed CVaR and Sharpe arrays for ``arm`` under ``root`` (records ``<arm>/{arm}-s{seed}``).

    ``root`` is the leg's TEST sub-root (``<output_dir>/test_leg_<label>``); each arm's records sit
    in their OWN ``<arm>/`` directory beneath it.

    LAYOUT FIX 2026-07-26 (deep review, row 34). This previously called ``load_run(f"{arm}-s{seed}",
    root)`` — i.e. it assumed a FLAT ``root/{arm}-s{seed}``. The real archive is TWO-level: the
    campaign hands ``write_run`` an ARM-level root (``src/cluster/run_one.py:108``), producing
    ``test_<sfx>/<arm>/<arm>-s<seed>/`` — verified first-hand on disk
    (``outputs/campaign_dryrun/test/distributional/distributional-s0``). Because
    :func:`leg_results_for_synthesis` passes ONE root for BOTH contrasted arms, the flat assumption
    was not merely wrong but unsatisfiable: no single ``root`` makes ``distributional-s0`` and
    ``scalar-s0`` both resolve when they live in sibling arm directories. Wired as it stood, EVERY
    leg would have raised FileNotFoundError, been caught as a leg failure, and reported
    ``t0_floor_pass: False`` — so the R86 pooled bound would have been computed over ZERO legs and
    read as "every leg failed the floor": a plausible-looking, wholly fabricated scientific outcome.
    The unit fixture wrote the same flat layout, which is why the tests passed; it now mirrors the
    producer.

    Raises FileNotFoundError naming the missing run — a silent seed subset would change the
    paired estimator, so absence is loud, never skipped.
    """
    arm_root = Path(root) / arm
    cvars: list[float] = []
    sharpes: list[float] = []
    for seed in seeds:
        rec = load_run(f"{arm}-s{seed}", arm_root)
        rets = np.asarray(rec["metrics"]["test_returns"], dtype=float)
        cvars.append(empirical_cvar(rets, alpha))
        # UNIT FIX 2026-07-26 (deep review; write-time registry row 34's "latent trap that only bites
        # on wiring"). This computed a PER-PERIOD, ddof=1 Sharpe (`rets.mean()/rets.std(ddof=1)`),
        # while `floor_sharpe` below — and every other Sharpe in the stack — is the ANNUALISED, ddof=0
        # `bootstrap.sharpe_ratio`. Passing the real T0 floor would therefore have compared ~0.04
        # against ~0.6: `floor_ok` False for EVERY leg, so every leg would be excluded from the sign
        # count and the R86 pooled bound, and the registered cross-model statement would come back
        # empty — silently, since an excluded leg is a legitimate reported outcome. Two mismatches
        # (annualisation AND ddof) are removed at once by delegating to the canonical estimator, so
        # there is exactly ONE Sharpe definition in the codebase. The CVaR arrays that feed the pooled
        # bound are untouched.
        sharpes.append(sharpe_ratio(rets))
    return {"cvar": np.asarray(cvars), "sharpe": np.asarray(sharpes)}


def t0_floor_sharpe(core_test_root: str | Path,
                    seeds: list[int] | None = None) -> float:
    """The registered T0 naive-benchmark floor — computed, never chosen.

    This closes the last open input of the R86/R101 pooled cross-model statement. The value was
    briefly treated as an unresolved science decision (the module docstring only said "the supplied
    naive-benchmark floor", and ``analyze_campaign.benchmark_floor`` gates on a DIFFERENT quantity —
    the winner's DSR against the whole benchmark suite). It is not a decision: **amendment R84 pins
    it exactly**, in the prose (``PREREGISTRATION.md``) and in the machine config
    (``config/preregistration.yaml: model_suite.synthesis_exactness.t0_floor_definition`` — path
    corrected 2026-07-26; it was cited as ``inference.t0_floor_definition``, which does not exist, and
    a wrong path in a docstring is how the next reader concludes the value was never registered):

        the EQUAL-WEIGHT benchmark's mean per-seed Sharpe over the common floor seeds 0-29,
        computed from the shared core baseline records

    so this function is a faithful transcription of that sentence and nothing more. ``equal_weight``
    is the canonical naive allocator, and the CORE baseline records are deliberately the source: every
    leg is filtered against the SAME floor, which is what makes the filter arm-symmetric and keeps
    the permutation test's size intact.

    Units matter here and were the trap row 34 flagged: this reuses :func:`per_seed_series`, so the
    floor is the ANNUALISED, ddof=0 ``bootstrap.sharpe_ratio`` — the same statistic the legs are
    compared with. Supplying a per-period Sharpe instead would understate the floor by ~√252 ≈ 15.9×
    and pass every leg; taking a per-period leg Sharpe against an annualised floor would fail every
    leg and silently empty the pooled bound.

    Fails LOUD on missing records: a floor computed from a partial seed set is a different (and
    lower-integrity) number than the registered one.
    """
    seeds = list(T0_FLOOR_SEEDS if seeds is None else seeds)
    series = per_seed_series(core_test_root, T0_FLOOR_ARM, seeds)
    return float(series["sharpe"].mean())


def leg_results_for_synthesis(
    leg_roots: dict[str, str | Path],
    seeds: list[int],
    floor_sharpe: float,
    *,
    arm_a: str = "distributional",
    arm_b: str = "scalar",
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Assemble the :func:`cross_model.sign_count`/``permutation_test`` input across legs.

    ``floor_sharpe`` is the T0 naive-benchmark floor — obtain it from :func:`t0_floor_sharpe`, which
    transcribes the R84 registration (equal-weight, mean per-seed Sharpe, floor seeds 0-29, from the
    shared CORE baseline records) rather than leaving the caller to pick a quantity; the
    registered criterion: BOTH arms' mean per-seed Sharpe must clear it or the leg is marked
    ``t0_floor_pass: False`` (reported, never voting). A leg whose records are missing/corrupt is
    included with ``t0_floor_pass: False`` and the reason in ``failure`` — reliability data,
    never silence.
    """
    out: dict[str, dict[str, Any]] = {}
    for label, root in leg_roots.items():
        try:
            a = per_seed_series(root, arm_a, seeds, alpha)
            b = per_seed_series(root, arm_b, seeds, alpha)
        except Exception as exc:  # noqa: BLE001 — a broken leg is a FINDING, not a crash
            out[label] = {
                "cvar_diff_per_seed": np.zeros(0),
                "t0_floor_pass": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
            continue
        floor_ok = bool(a["sharpe"].mean() > floor_sharpe and b["sharpe"].mean() > floor_sharpe)
        out[label] = {
            "cvar_diff_per_seed": a["cvar"] - b["cvar"],   # dist − scalar; positive = dist safer
            "t0_floor_pass": floor_ok,
            "sharpe_a_per_seed": a["sharpe"],
            "sharpe_b_per_seed": b["sharpe"],
            "cvar_a_per_seed": a["cvar"],
            "cvar_b_per_seed": b["cvar"],
        }
    return out


#: Leg archives are written to ``<output_dir>/test_leg_<sanitized label>`` (``run_campaign_cluster``
#: forces ``--root-suffix leg_<label>``), so discovery is a glob rather than a config lookup.
_LEG_DIR_PREFIX = "test_leg_"


def discover_leg_roots(output_dir: str | Path) -> dict[str, Path]:
    """``{leg label: archive root}`` for every replication leg present under ``output_dir``."""
    base = Path(output_dir)
    if not base.is_dir():
        return {}
    return {p.name[len(_LEG_DIR_PREFIX):]: p
            for p in sorted(base.glob(f"{_LEG_DIR_PREFIX}*")) if p.is_dir()}


def cross_model_synthesis(
    output_dir: str | Path,
    seeds: list[int] | None = None,
    *,
    core_test_root: str | Path | None = None,
    n_reps: int = 10_000,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """THE R86/R101 cross-model statement, assembled end-to-end from the leg archives.

    This is the production caller the pieces were missing. ``cross_model`` and this module were both
    built and unit-tested, but NOTHING invoked them — so the registered "across every model we
    tested, the effect is at most X" claim had no executable route to a number and would have had to
    be withdrawn after the compute was already spent (the R16 failure repeating).

    THE ANTI-FABRICATION CONTRACT. Three states are reported DISTINCTLY, because collapsing them is
    precisely how a plausible-but-empty result appears:

    * ``no_leg_archives``  — no leg ran (missing DATA; says nothing about any effect);
    * ``no_core_baseline`` — the registered T0 floor cannot be computed (missing INPUT);
    * ``no_legs_passed_floor`` — legs ran and every one failed the floor (a FINDING, reportable as
      an authoring/search failure — never as "no effect").

    Only ``ok`` carries the statistics. ``n_legs_found`` and ``n_legs_included`` are always reported
    side by side so a bound can never be read without knowing what it was computed over — the exact
    hole that let a bound over ZERO legs read as "all legs failed the T0 floor".

    Deterministic and read-only: the permutation and bootstrap seeds are the registered constants.
    """
    from src.inference.cross_model import permutation_test, pooled_bound, sign_count

    seeds = list(T0_FLOOR_SEEDS if seeds is None else seeds)
    base = Path(output_dir)
    leg_roots = discover_leg_roots(base)
    core = Path(core_test_root) if core_test_root is not None else base / "test"
    common: dict[str, Any] = {"n_legs_found": len(leg_roots), "legs_found": sorted(leg_roots),
                              "seeds": len(seeds), "core_test_root": str(core)}
    if not leg_roots:
        return {**common, "status": "no_leg_archives", "n_legs_included": 0,
                "note": "no replication-leg archives under the campaign root — MISSING DATA, "
                        "not evidence about the effect"}
    try:
        floor = t0_floor_sharpe(core, seeds)
    except Exception as exc:  # noqa: BLE001 — a missing floor is a missing INPUT, never a result
        return {**common, "status": "no_core_baseline", "n_legs_included": 0,
                "note": f"the registered T0 floor (equal_weight, seeds 0-29) is not computable "
                        f"from {core}: {type(exc).__name__}: {exc}"}

    legs = leg_results_for_synthesis(leg_roots, seeds, floor)
    included = sorted(k for k, v in legs.items() if v.get("t0_floor_pass"))
    failures = {k: v.get("failure") for k, v in legs.items() if v.get("failure")}
    common.update({"floor_sharpe": floor, "n_legs_included": len(included),
                   "legs_included": included, "leg_failures": failures})
    if not included:
        return {**common, "status": "no_legs_passed_floor",
                "note": "every leg failed the T0 floor in at least one contrasted arm — report this "
                        "as an authoring/search failure (a finding), NEVER as a null effect"}
    return {**common, "status": "ok",
            "sign_count": sign_count(legs),
            "permutation_test": permutation_test(legs, n_reps=n_reps),
            "pooled_bound": pooled_bound(legs, n_boot=n_boot)}
