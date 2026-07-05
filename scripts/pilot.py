#!/usr/bin/env python3
"""Unified pre-freeze PILOT battery -> ONE freeze-readiness report.

Calibrates the instrument so ``freeze`` locks a fully-determined, validated design with zero unknowns and
zero unvalidated risks. It resolves / validates:

  * **compute-frontier** : the fastest config that is determinism-safe AND thermally-stable AND memory-safe
                           (a faster config that breaks byte-determinism is REJECTED — it would break the
                           resume/replay/reproducibility the whole study rests on).
  * **convergence (B\\*)** : the per-candidate step budget at the plateau knee (fixes undertraining).
  * **sigma_D (seeds)**   : the seed count + whether the equivalence (TOST) headline is achievable
                           (delegated to ``scripts.sigma_seed_pilot``).
  * **risk-validation**   : resume round-trip + memory-flatness + pre-flight (booleans).
  * **wall-clock**        : GO / ADAPT / RECONSIDER for the (panel-scaled) campaign — the quantitative
                           "won't come out with nothing" projection that also decides panel-vs-SBX.

The DECISION LOGIC here is **pure** (unit-tested in ``tests/test_pilot.py``); the heavy MEASUREMENT drivers
(GPU training, NVML telemetry) are thin and run at Phase 3 on the real gold panel. Run::

    python scripts/pilot.py --demo          # render the report on synthetic numbers (no GPU)
    python scripts/pilot.py --from results/ # aggregate measured sub-pilot JSONs into the report
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

# Default wall-clock bands (hours) for a ~2-week 24/7 laptop budget; override on the CLI.
_GO_HOURS = 240.0      # <= 10 days: comfortable GO
_ADAPT_HOURS = 360.0   # <= 15 days: GO but trim scope (fewer seeds / reduced panel) — the ceiling
_MAX_TEMP_C = 87.0     # sustained GPU temp at/above this = not thermally stable (RTX-4050 throttle band)
_MAX_MEM_PCT = 90.0    # peak RAM/VRAM at/above this = not memory-safe (the measured OOM band)


# ── Compute-frontier selection ────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConfigResult:
    """One benchmarked compute config (e.g. compile=on, n_gpu=3, batch=256)."""
    label: str
    steps_per_sec: float
    deterministic: bool                 # run-twice byte-identity at this config (resume/replay depend on it)
    max_temp_c: float | None = None     # peak GPU temp over a SUSTAINED window
    peak_ram_pct: float | None = None
    peak_vram_pct: float | None = None


def config_is_safe(c: ConfigResult, *, max_temp_c: float = _MAX_TEMP_C, max_mem_pct: float = _MAX_MEM_PCT,
                   require_deterministic: bool = True) -> bool:
    """A config is admissible only if it preserves determinism and stays thermally + memory safe."""
    if require_deterministic and not c.deterministic:
        return False
    if c.max_temp_c is not None and c.max_temp_c >= max_temp_c:
        return False
    if c.peak_ram_pct is not None and c.peak_ram_pct >= max_mem_pct:
        return False
    if c.peak_vram_pct is not None and c.peak_vram_pct >= max_mem_pct:
        return False
    return True


def select_best_config(results: list[ConfigResult], *, max_temp_c: float = _MAX_TEMP_C,
                       max_mem_pct: float = _MAX_MEM_PCT, require_deterministic: bool = True) -> ConfigResult | None:
    """Fastest config among those that are determinism-safe + thermally-stable + memory-safe. None if none qualify."""
    safe = [c for c in results if config_is_safe(c, max_temp_c=max_temp_c, max_mem_pct=max_mem_pct,
                                                 require_deterministic=require_deterministic)]
    return max(safe, key=lambda c: c.steps_per_sec) if safe else None


# ── Wall-clock projection (the quantitative "won't come out with nothing") ───────────────────────────────
def project_wall_clock_hours(b_star: int, steps_per_sec: float, *, n_arms: int, n_candidates: int,
                             n_seeds: int, n_models: int = 1, overhead: float = 1.2) -> float:
    """Total GPU wall-clock for the campaign. Per model the campaign trains ``n_arms x (n_candidates + n_seeds)``
    candidates of ``b_star`` steps (search + winner re-eval); the panel multiplies that by ``n_models``. The
    ``overhead`` factor folds in LLM/IO/recycling time. Returns +inf if throughput is non-positive."""
    if steps_per_sec <= 0 or b_star <= 0:
        return float("inf")
    trainings = max(0, n_models) * max(0, n_arms) * (max(0, n_candidates) + max(0, n_seeds))
    total_steps = trainings * b_star
    return total_steps / steps_per_sec / 3600.0 * max(1.0, overhead)


def wall_clock_verdict(hours: float, *, go_hours: float = _GO_HOURS, adapt_hours: float = _ADAPT_HOURS) -> str:
    """GO (comfortable) / ADAPT (feasible but trim scope) / RECONSIDER (infeasible at this budget)."""
    if hours <= go_hours:
        return "GO"
    if hours <= adapt_hours:
        return "ADAPT"
    return "RECONSIDER"


# ── Convergence knee (B*) ───────────────────────────────────────────────────────────────────────────────
def knee_from_curve(steps: list[int], scores: list[float], *, rel_tol: float = 0.02) -> int:
    """The smallest step budget whose score is within ``rel_tol`` (relative to the plateau magnitude) of the
    best observed score — the convergence knee, i.e. the cheapest budget that still essentially converges.
    Conservative: returns the largest step if nothing reaches the plateau band."""
    if not steps:
        raise ValueError("empty learning curve")
    pairs = sorted(zip(steps, scores))
    best = max(s for _, s in pairs)
    thr = best - rel_tol * (abs(best) if best != 0 else 1.0)
    for st, sc in pairs:
        if sc >= thr:
            return int(st)
    return int(pairs[-1][0])


# ── Freeze-readiness aggregation ────────────────────────────────────────────────────────────────────────
@dataclass
class FreezeReadiness:
    config: dict | None = None
    b_star: int | None = None
    n_seeds: int | None = None
    sigma_seed: float | None = None
    equivalence_achievable: bool | None = None
    wall_clock_hours: float | None = None
    wall_clock_verdict: str | None = None
    risks: dict = field(default_factory=dict)        # name -> bool (validation passed)
    verdict: str = "BLOCKED"                          # READY | ADAPT | BLOCKED
    blockers: list[str] = field(default_factory=list)


def assess_freeze_readiness(*, config: dict | None, b_star: int | None, n_seeds: int | None,
                            sigma_seed: float | None, equivalence_achievable: bool | None,
                            wall_clock_hours: float | None, wc_verdict: str | None,
                            risks: dict | None) -> FreezeReadiness:
    """Aggregate every sub-pilot into a single GO/NO-GO. Any missing parameter or failed risk -> BLOCKED;
    a feasible-but-tight wall-clock -> ADAPT; everything green -> READY."""
    risks = dict(risks or {})
    blockers: list[str] = []
    if config is None:
        blockers.append("no determinism-safe + thermally-stable compute config found")
    if b_star is None:
        blockers.append("B* (convergence knee) not determined")
    if n_seeds is None:
        blockers.append("seed count (sigma_D pilot) not determined")
    for name, ok in risks.items():
        if not ok:
            blockers.append(f"risk-validation FAILED: {name}")
    if wc_verdict == "RECONSIDER":
        blockers.append("wall-clock infeasible at the budget (RECONSIDER) — trim scope or enable SBX")

    if blockers:
        verdict = "BLOCKED"
    elif wc_verdict == "ADAPT":
        verdict = "ADAPT"
    else:
        verdict = "READY"
    return FreezeReadiness(
        config=config, b_star=b_star, n_seeds=n_seeds, sigma_seed=sigma_seed,
        equivalence_achievable=equivalence_achievable, wall_clock_hours=wall_clock_hours,
        wall_clock_verdict=wc_verdict, risks=risks, verdict=verdict, blockers=blockers,
    )


def render_report(fr: FreezeReadiness) -> str:
    """A human-readable freeze-readiness report."""
    L: list[str] = []
    L.append("=" * 70)
    L.append(f"FREEZE-READINESS: {fr.verdict}")
    L.append("=" * 70)
    cfg = fr.config or {}
    L.append(f"compute config   : {cfg.get('label', '—')}  ({cfg.get('steps_per_sec', '—')} steps/s, "
             f"deterministic={cfg.get('deterministic', '—')})")
    L.append(f"B* (steps/cand)  : {fr.b_star if fr.b_star is not None else '—'}")
    L.append(f"seeds            : {fr.n_seeds if fr.n_seeds is not None else '—'}  "
             f"(sigma_seed={fr.sigma_seed}, equivalence_achievable={fr.equivalence_achievable})")
    wch = f"{fr.wall_clock_hours:.0f} h ({fr.wall_clock_hours/24:.1f} d)" if fr.wall_clock_hours not in (None,) else "—"
    L.append(f"wall-clock       : {wch}  -> {fr.wall_clock_verdict}")
    L.append("risk-validation  : " + (", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in fr.risks.items()) or "—"))
    L.append("-" * 70)
    if fr.blockers:
        L.append("BLOCKERS:")
        L.extend(f"  - {b}" for b in fr.blockers)
    else:
        L.append("No blockers. Ratify the reframe+panel into the prereg, then FREEZE.")
    L.append("=" * 70)
    return "\n".join(L)


# ── Driver (thin; heavy measurement is Phase-3) ─────────────────────────────────────────────────────────
def _demo() -> FreezeReadiness:
    """Render the report on synthetic-but-realistic numbers — validates the pipeline with no GPU. NOT a result."""
    bench = [
        ConfigResult("compile=off,n_gpu=2", 178.0, deterministic=True, max_temp_c=78, peak_ram_pct=84, peak_vram_pct=70),
        ConfigResult("compile=off,n_gpu=3", 235.0, deterministic=True, max_temp_c=85, peak_ram_pct=93, peak_vram_pct=78),
        ConfigResult("compile=on,n_gpu=2", 270.0, deterministic=True, max_temp_c=80, peak_ram_pct=85, peak_vram_pct=72),
    ]
    best = select_best_config(bench)  # n_gpu=3 is rejected (RAM 93% >= 90); compile=on,n_gpu=2 wins
    b_star = knee_from_curve([50_000, 150_000, 250_000, 350_000, 500_000],
                             [0.31, 0.55, 0.752, 0.76, 0.765])  # plateau knee ~250k
    hours = project_wall_clock_hours(b_star, best.steps_per_sec, n_arms=7, n_candidates=30, n_seeds=30, n_models=1)
    return assess_freeze_readiness(
        config=asdict(best), b_star=b_star, n_seeds=30, sigma_seed=0.08, equivalence_achievable=True,
        wall_clock_hours=hours, wc_verdict=wall_clock_verdict(hours),
        risks={"resume_round_trip": True, "memory_flat": True, "preflight": True},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unified pre-freeze pilot battery -> freeze-readiness report.")
    ap.add_argument("--demo", action="store_true", help="render on synthetic numbers (no GPU) to validate the pipeline")
    ap.add_argument("--json", action="store_true", help="emit the FreezeReadiness as JSON")
    args = ap.parse_args(argv)

    fr = _demo()  # Phase-3: replace with aggregation of measured sub-pilot result files
    if args.json:
        print(json.dumps(asdict(fr), indent=2, default=str))
    else:
        print(render_report(fr))
    return 0 if fr.verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
