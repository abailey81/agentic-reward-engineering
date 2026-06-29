"""Design Determination Pipeline (DDP) — resolve every campaign parameter to its METHODOLOGICALLY-CORRECT
value, by the right procedure for its class, and report freeze-readiness.

Why this exists (read this before "optimising" anything)
--------------------------------------------------------
This is a CONTROLLED experiment: the agent is a fixed instrument and the *feedback channel* is the only
manipulated variable. A system that searched parameters to MAXIMISE the headline result would be the garden
of forking paths industrialised — it manufactures false positives and is unpublishable (and it cannot even
help a corroborated-NULL headline). So the DDP never optimises for performance. It assigns each parameter to
one of four classes and determines it by that class's CORRECT criterion:

  * MEASURE   — optimise for ADEQUACY (a diagnostic plateau), not performance: B* (convergence), n_seeds
                (power), candidates (search saturation). These are the ONLY "tuned" parameters.
  * CALIBRATE — set ONCE, identically across arms, by a principled procedure on PRE-TEST data only:
                SESOI, embargo, multiplicity, the DSR trial count (λ was reclassified FIX — see its row).
  * FIX       — hold at sensible literature defaults, IDENTICAL across arms; tuning would CONFOUND the
                channel contrast: all SAC/LLM hyperparameters, learning_starts, PopArt.
  * REALISTIC — credible real-world values, not result-maximising: universe, lookback, costs, splits,
                delisting, the cash rate.

INVARIANT: nothing here touches the sealed test split (2020-2026). Determination uses train+val (or
pre-2017 sub-folds) only; the test is opened ONCE, post-freeze, for the confirmatory inference.

What it produces
----------------
``outputs/DESIGN_DETERMINATION.md`` (+ a json) — for every material parameter: its class, the method that fixes
it, the current value, the determination status, and the evidence pointer. A FREEZE-READY verdict is emitted
when every MEASURE/CALIBRATE parameter is resolved (no PENDING / FIX_NEEDED). This table is the methods-section
justification ("every design parameter, the principled procedure that set it, and the evidence").

This module is a reporter/orchestrator: the heavy determinations live in their own engines
(``scripts/learning_curve.recommend_budget``/``project_campaign``, ``scripts/power_analysis``); the one engine
that lived nowhere — search-saturation — is :func:`recommend_candidates` here.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "ParamClass",
    "Status",
    "ParamSpec",
    "REGISTRY",
    "recommend_candidates",
    "best_so_far_curves",
    "determine",
    "main",
]


class ParamClass(str, Enum):
    MEASURE = "MEASURE"        # optimise for adequacy (plateau), not performance
    CALIBRATE = "CALIBRATE"    # set once, identically across arms, on pre-test data
    FIX = "FIX"                # held constant for experimental control (do NOT tune)
    REALISTIC = "REALISTIC"    # credible real-world value, not result-maximising


class Status(str, Enum):
    DETERMINED = "DETERMINED"  # value resolved with evidence
    DECIDED = "DECIDED"        # value fixed by a RATIFIED user decision (diagnostic evidence disclosed)
    PENDING = "PENDING"        # awaits a pilot/calibration before freeze
    FIX_NEEDED = "FIX_NEEDED"  # current value is wrong and must change
    VERIFY = "VERIFY"          # plausibly fine; confirm with an existing test
    FIXED = "FIXED"            # locked by design/convention; no action


@dataclass(frozen=True)
class ParamSpec:
    name: str
    klass: ParamClass
    method: str            # how "best" is decided for this parameter
    determination: str     # where the value comes from / what to run
    blocks_freeze: bool    # True if the campaign must not be frozen while this is unresolved


# The material parameters. Class + method are metadata; current values are read from config at runtime so
# this never duplicates the single source of truth: code reads config and never hardcodes.
REGISTRY: tuple[ParamSpec, ...] = (
    # ---- MEASURE (the only optimised parameters; optimise for adequacy) ----
    # 2026-07-13 (paper-front audit): the criterion label said "eval-return knee (convergence)" —
    # but R74 records that criterion as structurally unsatisfiable on this workload (no confirmed
    # eval plateau ever fired); the ACTUAL decision basis was the totality-of-evidence override
    # (critic knee + range-limited eval flatness + matched compute). The provenance must say so.
    ParamSpec("train_steps_per_candidate", ParamClass.MEASURE,
              "learning-curve pilot (critic knee + range-limited eval flatness; R74 override)",
              "scripts/learning_curve + R74 totality-of-evidence", True),
    ParamSpec("n_seeds", ParamClass.MEASURE,
              "paired-CRN power at the SESOI", "scripts/power_analysis + sigma_seed pilot", True),
    ParamSpec("candidates_per_arm", ParamClass.MEASURE,
              "search saturation (best-fitness plateau)", "determine_design.recommend_candidates", True),
    # ---- CALIBRATE (set once, identically across arms, on pre-test data) ----
    ParamSpec("sesoi", ParamClass.CALIBRATE,
              "smallest practically-relevant edge", "frozen 0.05 DSR (~0.07 ann-Sharpe)", False),
    ParamSpec("embargo_trading_days", ParamClass.CALIBRATE,
              "purge >= feature lookback + horizon", "verify effective purge >= lookback (60)", False),
    ParamSpec("multiplicity", ParamClass.CALIBRATE,
              "family-wise / FDR error control", "IUT within H2 + BH/RW across the union", False),
    ParamSpec("dsr_trial_count_rule", ParamClass.CALIBRATE,
              "selection-aware deflation", "per_arm_candidates (Bailey-Lopez de Prado)", False),
    # ---- FIX (held constant for control; tuning confounds the channel) ----
    # lambda_cvar was reclassified CALIBRATE -> FIX (2026-07-02): the ratified design pins it at 0 BY DESIGN
    # (tail-blind selector; preregistration.yaml fitness.lambda_cvar = 0.0, freeze-gate check #7). A tail-aware
    # selector would confound the H2 feedback channel, so lambda is exactly what this class is for — "tuning
    # confounds the channel" — NOT a pre-2015 calibration target. The legacy calibration apparatus
    # (lambda_grid/lambda_frozen/calibration_fold in config/inference.yaml) was deleted per the prereg §5 note.
    ParamSpec("lambda_cvar", ParamClass.FIX,
              "0 BY DESIGN: tail-blind selector (tuning would confound the H2 channel)",
              "preregistration.yaml fitness.lambda_cvar = 0.0 (RATIFIED 2026-07-01; freeze gate #7)", False),
    ParamSpec("sac_hyperparameters", ParamClass.FIX,
              "SB3 defaults, identical across arms", "config/algos.yaml (null = SB3 default)", False),
    ParamSpec("learning_starts", ParamClass.FIX,
              "Phase-0 validated stability", "1000 (not SB3's 100)", False),
    ParamSpec("popart", ParamClass.FIX,
              "critic value-scale normalisation", "on (explosion fix)", False),
    ParamSpec("llm_decoding", ParamClass.FIX,
              "held identical; only feedback varies", "Opus 5, K=16, max_tokens 8192", False),
    # ---- REALISTIC (credible, not result-maximising) ----
    ParamSpec("n_assets", ParamClass.REALISTIC, "diversified tradable book", "30 from 963 PIT pool (univ5)", False),
    ParamSpec("lookback_days", ParamClass.REALISTIC, "feature window", "60", False),
    ParamSpec("headline_bps", ParamClass.REALISTIC, "realistic large-cap cost", "10 bps proportional", False),
    ParamSpec("data_splits", ParamClass.REALISTIC, "sealed temporal holdout", "12y/3y/6.5y train/val/test (Split C)", False),
    ParamSpec("delisting_returns", ParamClass.REALISTIC, "survivorship-free", "retain", False),
    ParamSpec("cash_daily_rate", ParamClass.REALISTIC, "risk-free accrual on cash",
              "cash=0 numeraire (§10 RATIFIED 2026-07-01; DGS3MO rf-excess robustness leg)", False),
)


def best_so_far_curves(records: list[dict[str, Any]], *, fitness_key: str = "val_fitness") -> dict[str, list[float]]:
    """Reduce raw run records to per-arm best-fitness-so-far curves indexed by generation.

    Each record carries ``arm``, ``generation`` and ``metrics[fitness_key]``. Returns ``{arm: [best after
    gen 0, best after gen 1, ...]}`` (monotone non-decreasing) — the input :func:`recommend_candidates` reads.
    """
    by_arm_gen: dict[str, dict[int, float]] = {}
    for r in records:
        arm = r.get("arm")
        gen = r.get("generation")
        val = (r.get("metrics") or {}).get(fitness_key)
        if arm is None or gen is None or not isinstance(val, (int, float)):
            continue
        g = int(gen)
        cur = by_arm_gen.setdefault(str(arm), {})
        cur[g] = max(cur.get(g, float("-inf")), float(val))
    curves: dict[str, list[float]] = {}
    for arm, gen_best in by_arm_gen.items():
        best = float("-inf")
        curve: list[float] = []
        for g in sorted(gen_best):
            best = max(best, gen_best[g])
            curve.append(best)
        curves[arm] = curve
    return curves


def recommend_candidates(
    best_so_far: dict[str, list[float]],
    *,
    candidates_per_gen: int,
    patience: int = 2,
    rel_tol: float = 0.0,
) -> dict[str, Any]:
    """Objective search-saturation detector — the candidate-budget analogue of ``recommend_budget``.

    Given each arm's best-fitness-SO-FAR curve over generations, an arm is SATURATED when its best has not
    improved over the last ``patience`` generations (by more than ``rel_tol`` of its own range). The
    recommended ``candidates_per_arm`` is set by the SLOWEST arm to saturate (matched compute across arms):
    ``(first generation at which that arm reached its final plateau + 1) * candidates_per_gen``. If ANY
    reflection arm is still improving at the last generation, ``saturated=False`` with a loud reason — the
    search budget is too small and the arms' "best" is budget-limited, biasing the channel contrast.

    Returns ``{recommended_candidates, saturated, reason, per_arm}``. ``saturated`` is ``None`` when no arm
    has enough generations (``< patience + 1``) to judge. Diagnostic only — it INFORMS the (amendment-gated)
    ``candidates_per_arm``; no number enters the dissertation.
    """
    judged: dict[str, dict[str, Any]] = {}
    still_rising: list[str] = []
    sat_gens: list[int] = []
    for arm, curve in best_so_far.items():
        if len(curve) < patience + 1:
            judged[arm] = {"saturated": None, "reason": "too few generations to judge"}
            continue
        rng = max(curve[-1] - curve[0], abs(curve[-1]) * 1e-9, 1e-12)
        tol = float(rel_tol) * rng
        recent_gain = curve[-1] - curve[-1 - patience]
        is_sat = recent_gain <= tol
        # First generation at which this arm reached (within tol) its final plateau value.
        final = curve[-1]
        sat_gen = next((g for g, v in enumerate(curve) if v >= final - tol), len(curve) - 1)
        judged[arm] = {"saturated": bool(is_sat), "saturation_gen": int(sat_gen),
                       "recent_gain": float(recent_gain)}
        if is_sat:
            sat_gens.append(sat_gen)
        else:
            still_rising.append(arm)

    decidable = [a for a, j in judged.items() if j.get("saturated") is not None]
    if not decidable:
        return {"recommended_candidates": None, "saturated": None,
                "reason": "no arm has enough generations to judge saturation", "per_arm": judged}
    if still_rising:
        return {"recommended_candidates": None, "saturated": False,
                "reason": (f"arms still improving at the last generation: {sorted(still_rising)} — the search "
                           "budget is too small (their best is budget-limited); EXTEND candidates_per_arm"),
                "per_arm": judged}
    # All decidable arms saturated: the slowest setter fixes the matched budget (+1 -> count, not index).
    rec = (max(sat_gens) + 1) * int(candidates_per_gen)
    return {"recommended_candidates": int(rec), "saturated": True,
            "reason": (f"all arms saturated by generation {max(sat_gens)} "
                       f"(x{candidates_per_gen} candidates/gen = {rec} candidates)"),
            "per_arm": judged}


def determine(evidence: dict[str, Any]) -> dict[str, Any]:
    """Overlay dynamic determination STATUS onto the registry from available evidence (pure; no IO).

    ``evidence`` keys (all optional; missing -> the parameter reports its default status):
      ``recommended_budget``  -> B* resolved (DETERMINED) else PENDING.
      ``candidates_saturated``-> True/False/None from :func:`recommend_candidates`.
      ``cash_daily_rate``     -> 0.0 -> FIX_NEEDED (silently compounds cash) else DETERMINED/FIXED.
      ``sigma_seed_pilot``    -> True if the seed-variance pilot has run (n_seeds DETERMINED) else PENDING.

    Returns ``{rows, freeze_ready, blockers}``: one row per registry entry with its resolved status, and a
    freeze-readiness verdict (ready iff no freeze-blocking parameter is PENDING/FIX_NEEDED).
    """
    def status_for(spec: ParamSpec) -> Status:
        n = spec.name
        if n == "train_steps_per_candidate":
            if evidence.get("recommended_budget"):
                return Status.DETERMINED
            # R74 (2026-07-02): the R70 knee detector is structurally unsatisfiable on a flat-noise
            # eval curve (its tolerance scales with the curve's own range), so B* was set by the
            # evidence dossier and RATIFIED (prereg R74 + the campaign/algos/prereg-yaml mirror).
            # A converged knee still wins (DETERMINED); the ratified dossier decision reports DECIDED.
            return Status.DECIDED if evidence.get("train_steps_ratified") else Status.PENDING
        if n == "candidates_per_arm":
            sat = evidence.get("candidates_saturated")
            if sat is True:
                return Status.DETERMINED
            # Saturation is DIAGNOSTIC-ONLY for this parameter (see recommend_candidates: "it INFORMS
            # the (amendment-gated) candidates_per_arm"): the budget itself was RATIFIED 2026-07-01 at
            # a hard cap (30 — multiplicity control; "more candidates" explicitly rejected), with the
            # search-width limitation disclosed in CH7. A False/None diagnostic therefore feeds the
            # disclosure, not the freeze gate — requiring saturated=True here made freeze-readiness
            # UNSATISFIABLE pre-campaign (found 2026-07-02 when the engine first ran on the real
            # prototype archive: 3/4 judgeable arms saturated by gen 3-4, scalar jumped at gen 7).
            return Status.DECIDED if evidence.get("candidates_ratified") else Status.PENDING
        if n == "n_seeds":
            if not evidence.get("sigma_seed_pilot"):
                return Status.PENDING
            # The pilot RAN — but if it fired the pre-registered "sigma_D > 0.10 -> raise the seed
            # count" trigger (§6 D2 band) while the frozen config still carries the pre-pilot 30-seed
            # placeholder, the count is NOT yet decided: the seed amendment the pilot mandates is owed
            # (a false DETERMINED here would greenlight a premature freeze at an under-powered n; found
            # 2026-07-05). Determined iff the trigger did not fire OR the config has been amended past 30.
            sd = evidence.get("sigma_pilot_sigma_d")
            cfg_seeds = evidence.get("config_n_seeds")
            trigger_fired = sd is not None and float(sd) > 0.10
            placeholder_unchanged = cfg_seeds is not None and int(cfg_seeds) <= 30
            if trigger_fired and placeholder_unchanged:
                return Status.PENDING
            return Status.DETERMINED
        if n == "cash_daily_rate":
            cdr = evidence.get("cash_daily_rate")
            if cdr is not None and float(cdr) == 0.0:
                # cash=0 was RATIFIED 2026-07-01 (prereg §10 numeraire: rf=0 headline, common-mode in
                # the arm contrast, + DGS3MO rf-excess robustness leg) — when the env value MATCHES the
                # ratified numeraire, report the decision; FIX_NEEDED was pre-ratification semantics
                # ("silently compounds cash"), stale once §10 landed (found 2026-07-02).
                ratified = evidence.get("numeraire_ratified_cash")
                if ratified is not None and float(ratified) == 0.0:
                    return Status.DECIDED
                return Status.FIX_NEEDED
            return Status.DETERMINED
        if n == "embargo_trading_days":
            return Status.VERIFY
        if spec.klass in (ParamClass.FIX, ParamClass.REALISTIC) or spec.klass is ParamClass.CALIBRATE:
            return Status.FIXED
        return Status.FIXED

    rows = []
    blockers: list[str] = []
    for spec in REGISTRY:
        st = status_for(spec)
        rows.append({"name": spec.name, "class": spec.klass.value, "method": spec.method,
                     "determination": spec.determination, "status": st.value})
        if spec.blocks_freeze and st in (Status.PENDING, Status.FIX_NEEDED):
            blockers.append(spec.name)
    return {"rows": rows, "freeze_ready": not blockers, "blockers": blockers}


def _write_markdown(result: dict[str, Any], evidence: dict[str, Any], path: Path) -> None:
    order = {ParamClass.MEASURE.value: 0, ParamClass.CALIBRATE.value: 1,
             ParamClass.FIX.value: 2, ParamClass.REALISTIC.value: 3}
    rows = sorted(result["rows"], key=lambda r: (order[r["class"]], r["name"]))
    verdict = "FREEZE-READY" if result["freeze_ready"] else f"BLOCKED on {result['blockers']}"
    lines = [
        "# Design Determination — every parameter, its method, and its status",
        "",
        "Generated by `scripts/determine_design.py`. \"Best\" means *methodologically-correct and frozen with "
        "evidence*, **not** performance-maximal — optimising parameters to maximise the result would be the "
        "garden of forking paths and is never done here. The sealed test (2020-2026) is touched by nothing in "
        "this determination.",
        "",
        f"**Status: {verdict}**",
        "",
        "| Parameter | Class | Method (how 'best' is decided) | Determination source | Status |",
        "|-----------|-------|--------------------------------|----------------------|--------|",
    ]
    for r in rows:
        lines.append(f"| `{r['name']}` | {r['class']} | {r['method']} | {r['determination']} | {r['status']} |")
    if not result["freeze_ready"]:
        lines += ["", "## Blocking actions before freeze", ""]
        lines += [f"- `{b}`" for b in result["blockers"]]
    if evidence.get("candidates_diagnostic"):
        lines += [
            "", "## Evidence notes", "",
            (f"- `candidates_per_arm` saturation diagnostic (prototype archive; DIRECTIONAL — Sonnet "
             f"author, pre-Split-C window): {evidence['candidates_diagnostic']} The budget itself is "
             f"the ratified 2026-07-01 cap ({evidence.get('candidates_ratified')}; multiplicity "
             "control — 'more candidates' explicitly rejected); the diagnostic feeds the CH7 "
             "search-width disclosure, not the freeze gate."),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gather_evidence() -> dict[str, Any]:
    """Best-effort read of current config values + any pilot outputs already on disk (no GPU, no test data)."""
    ev: dict[str, Any] = {}
    try:
        from src.utils.config import load_config

        env = load_config("environment")
        ev["cash_daily_rate"] = (env.get("state") or {}).get("cash_daily_rate")
    except Exception:  # noqa: BLE001 - the reporter must run even if config import is unavailable
        pass
    lc = Path("outputs/tables/learning_curve.json")
    if lc.exists():
        try:
            data = json.loads(lc.read_text(encoding="utf-8"))
            conv = data.get("convergence") or {}
            # A NOT-converged ladder reports its CEILING as recommended_budget (the extend-the-ladder
            # sentinel) — reading it unconditionally stamped train_steps DETERMINED off an unconverged
            # run (latent bug, found 2026-07-02). Only a converged=True knee counts as evidence.
            ev["recommended_budget"] = (
                conv.get("recommended_budget") if conv.get("converged") is True else None
            )
        except Exception:  # noqa: BLE001
            pass
    # n_seeds: the σ_D pilot evidence must be a SUCCESSFUL analyzer run on ENOUGH shared seeds — NOT
    # the artifact's mere existence (batch-5 M1, 2026-07-03): sigma_seed_pilot.py writes its JSON
    # unconditionally (even status="skipped" on an empty/partial archive), so existence alone could
    # flip n_seeds DETERMINED — and the chain FREEZE-READY — off a failed/husked farm. Gate on the
    # in-JSON success flag (the sharpe-leg status=="ok" bool the analyzer computes exactly for this
    # purpose) AND an n_shared floor: >= 12 of the 15 planned CRN seeds (>= 80%; below that the σ_D/ρ
    # estimate is too noisy to anchor the 30-vs-50 seeds decision). recommended_n is surfaced into the
    # evidence notes so the report shows the actual verdict, not just the flag.
    sp = Path("outputs/sigma_pilot/sigma_seed_pilot.json")
    if sp.exists():
        try:
            spd = json.loads(sp.read_text(encoding="utf-8"))
            sharpe_stats = ((spd.get("per_statistic") or {}).get("sharpe") or {}).get("stats") or {}
            n_shared = sharpe_stats.get("n_shared")
            ev["sigma_pilot_n_shared"] = n_shared
            ev["sigma_pilot_recommended_n"] = spd.get("recommended_n")
            # The measured seed-difference SD gates the PRE-REGISTERED "30 -> raise if sigma_D > 0.10"
            # rule (§6 amendment D2 band): a pilot that fires the trigger means the FROZEN seed count
            # must be amended BEFORE freeze, so the pilot having merely RUN is not "n_seeds decided".
            ev["sigma_pilot_sigma_d"] = sharpe_stats.get("sigma_d")
            if spd.get("sigma_seed_pilot") is True and int(n_shared or 0) >= 12:
                ev["sigma_seed_pilot"] = True
        except Exception:  # noqa: BLE001 — an unreadable evidence artifact is NO evidence
            pass
    # candidates_per_arm: (a) the saturation DIAGNOSTIC from the prototype archive (directional —
    # Sonnet author + pre-Split-C window; the only pre-campaign search archive that exists), and
    # (b) the ratified budget from campaign.yaml (the decision-class anchor; see status_for).
    proto = Path("outputs/prototype")
    if proto.is_dir():
        try:
            from src.io.results import load_all

            records: list[dict[str, Any]] = []
            for d in sorted(p for p in proto.iterdir() if p.is_dir()):
                records.extend(load_all(d))
            curves = best_so_far_curves(records)
            if curves:
                rec = recommend_candidates(curves, candidates_per_gen=5)  # the prototype ran 5/gen
                ev["candidates_saturated"] = rec.get("saturated")
                ev["candidates_diagnostic"] = rec.get("reason")
        except Exception:  # noqa: BLE001
            pass
    try:
        from src.utils.config import load_config

        camp_cfg = load_config("campaign")
        ev["candidates_ratified"] = camp_cfg.get("candidates_per_arm")
        # Seeds may be a bare list OR a schema ({mode: tiered/uniform, …}); resolve to the flat
        # [0..N-1] set so the count is correct for BOTH forms. A naive len() on the tiered dict
        # would read 2 (its key count), spuriously keeping n_seeds PENDING after ratification.
        from src.utils.seeds import resolve_seeds as _resolve_seeds

        _sc = camp_cfg.get("seeds")
        ev["config_n_seeds"] = len(_resolve_seeds(_sc)) if _sc else 0
    except Exception:  # noqa: BLE001
        pass
    # train_steps: the R74 ratified B* — anchored on the PREREG machine mirror AND its equality with
    # the executed campaign value (a mirror mismatch must NOT count as ratified; preflight's
    # budget-mirror guard separately asserts campaign == algos).
    try:
        from src.utils.config import load_config

        prereg_bstar = load_config("preregistration").get("train_steps_per_candidate")
        campaign_bstar = load_config("campaign").get("train_steps_per_candidate")
        if prereg_bstar is not None and prereg_bstar == campaign_bstar:
            ev["train_steps_ratified"] = prereg_bstar
    except Exception:  # noqa: BLE001
        pass
    # cash: the §10 numeraire ratification (2026-07-01) anchors idle cash at 0 — status_for clears
    # FIX_NEEDED only when the env value MATCHES this ratified value.
    try:
        from src.utils.config import load_config

        numeraire = load_config("preregistration").get("numeraire") or {}
        ev["numeraire_ratified_cash"] = numeraire.get("idle_cash_daily_rate")
    except Exception:  # noqa: BLE001
        pass
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description="Design Determination Pipeline — parameter resolution + freeze-readiness.")
    p.add_argument("--out", default="outputs/DESIGN_DETERMINATION.md", help="Determination report path.")
    args = p.parse_args()

    evidence = _gather_evidence()
    result = determine(evidence)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(result, evidence, out)
    (out.with_suffix(".json")).write_text(
        json.dumps({"evidence": evidence, **result}, indent=2), encoding="utf-8")

    print("=" * 72)
    print("[determine_design] Design Determination Pipeline")
    for r in sorted(result["rows"], key=lambda r: r["class"]):
        print(f"  [{r['class']:9s}] {r['name']:28s} -> {r['status']}")
    print("-" * 72)
    print("FREEZE-READY" if result["freeze_ready"] else f"BLOCKED on: {result['blockers']}")
    print(f"wrote {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
