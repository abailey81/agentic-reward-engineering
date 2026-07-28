#!/usr/bin/env python3
"""FIRST-SEED SANITY — "is what we're producing garbage?", answered in hours, not days.

WHY THIS EXISTS
---------------
The campaign banks its first statistical result at the 30-seed floor, ~2 days in. But most ways a
run goes WRONG are visible on the very first completed seed: a reward that crashes every step and
trains on the fallback signal, returns full of NaN, a policy parked in cash producing a flat line,
an absurd reward scale. Waiting two days to discover that wastes two days.

NOTE: THIS IS NOT "BANKING A RESULT AT SEED 1"
------------------------------------------
There is no inference at n=1 — no effect, no interval, nothing to bank. What this does is check
that the MACHINE is producing sane output. That distinction is not pedantry, it is the whole
design: the pre-registration commits to a SINGLE confirmatory look at the pre-declared date, and
peeking at effects earlier is optional stopping, which would invalidate every p-value and interval
in the dissertation. `src/cluster/integrity.py` states the same rule for the C3 review gate:
"conditioning continuation on observed effects is optional stopping"; what a mid-run gate may
inspect is EXECUTION.

So this module is **EFFECT-BLIND BY CONSTRUCTION**:

* it NEVER reads ``val_fitness``, ``test_sharpe``, ``test_cvar`` or any per-arm performance value;
* it NEVER compares one arm against another;
* every check is a per-record execution-quality question whose answer would be the same whichever
  arm turned out to win.

``tests/test_first_seed_sanity.py`` enforces this: it runs the gate over two archives that differ
ONLY in which arm performs better and asserts the output is byte-identical. A gate that leaks the
effect would fail that test.

Usage::

    python scripts/first_seed_sanity.py outputs/campaign_cluster            # first seed present
    python scripts/first_seed_sanity.py outputs/campaign_cluster --seed 0
    python scripts/first_seed_sanity.py outputs/campaign_cluster --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OK, SUSPECT, GARBAGE = "OK", "SUSPECT", "GARBAGE"
_RANK = {OK: 0, SUSPECT: 1, GARBAGE: 2}

#: A daily portfolio return outside this band is not a result, it is a bug (the panel is equities).
_ABSURD_DAILY_RETURN = 1.0
#: Fraction of training steps that may fall back to SAFE_DEFAULT before the candidate is not really
#: "trained on its reward" at all. Any substitution at all is already worth surfacing.
_SAFE_DEFAULT_TOLERANCE = 0.0


@dataclass
class Signal:
    name: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail,
                "evidence": self.evidence}


# ── PURE CHECKS (unit-tested; each is effect-blind) ───────────────────────────────────────────

def check_returns_finite(returns: np.ndarray) -> Signal:
    """NaN or inf anywhere in the realized returns means the number downstream is meaningless."""
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return Signal("returns_present", GARBAGE, "the record carries NO realized returns", {})
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        return Signal("returns_finite", GARBAGE,
                      f"{bad}/{arr.size} realized returns are NaN or inf — anything computed from "
                      "this is meaningless", {"n_nonfinite": bad, "n": int(arr.size)})
    return Signal("returns_finite", OK, f"all {arr.size} realized returns are finite",
                  {"n": int(arr.size)})


def check_returns_nondegenerate(returns: np.ndarray) -> Signal:
    """A flat return series means the policy never really traded — a degenerate run, not a result.

    Effect-blind: this asks "did anything happen at all", never "did it do WELL".
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return Signal("returns_nondegenerate", SUSPECT,
                      f"only {arr.size} finite return(s) — too few to judge", {"n": int(arr.size)})
    # ptp, not std: the std of 300 identical floats is ~5e-20 from summation error, so an `== 0`
    # test silently MISSES the flat series it exists to catch.
    if float(np.ptp(arr)) == 0.0:
        return Signal("returns_nondegenerate", GARBAGE,
                      "every realized return is IDENTICAL — the policy is degenerate (e.g. parked "
                      "in cash); a flat line is not a result", {"peak_to_peak": 0.0})
    if np.count_nonzero(arr) == 0:
        return Signal("returns_nondegenerate", GARBAGE, "every realized return is exactly zero",
                      {"nonzero": 0})
    return Signal("returns_nondegenerate", OK, "the return series varies (the policy traded)",
                  {"std_present": True})


def check_returns_plausible(returns: np.ndarray,
                            absurd: float = _ABSURD_DAILY_RETURN) -> Signal:
    """Daily equity returns beyond +/-100% are a broken environment, not a lucky strategy."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Signal("returns_plausible", SUSPECT, "no finite returns to range-check", {})
    worst = float(np.max(np.abs(arr)))
    if worst > absurd:
        return Signal("returns_plausible", GARBAGE,
                      f"a daily return of {worst:.3g} exceeds +/-{absurd:.0%} — that is a broken "
                      "environment or reward, not performance", {"max_abs_return": worst})
    return Signal("returns_plausible", OK,
                  f"largest daily move {worst:.4f} is within a plausible band",
                  {"max_abs_return": worst})


def check_reward_actually_ran(safe_default_count: Any, safe_call_count: Any,
                              tolerance: float = _SAFE_DEFAULT_TOLERANCE) -> Signal:
    """Did the candidate train on ITS OWN reward, or on the fallback because the reward crashed?

    This is the single most valuable garbage signal. When an authored reward raises at runtime,
    ``safe_call`` substitutes a neutral 0.0 so the rollout survives — correct engineering, but it
    means the agent learned from a CONSTANT signal. The run then looks healthy and completes, and
    the record is worthless. Measured this session: 75% of one leg's rewards raise on step 1.
    """
    if safe_default_count is None or safe_call_count in (None, 0):
        return Signal("reward_actually_ran", SUSPECT,
                      "the record does not carry SAFE_DEFAULT counters — cannot confirm the "
                      "candidate trained on its own reward", {})
    n_sd, n_call = int(safe_default_count), int(safe_call_count)
    frac = n_sd / n_call if n_call else 0.0
    if frac >= 0.5:
        return Signal("reward_actually_ran", GARBAGE,
                      f"the authored reward failed on {n_sd}/{n_call} steps ({frac:.0%}) and the "
                      "agent trained mostly on the neutral fallback — this record is not a test of "
                      "that reward", {"safe_default": n_sd, "calls": n_call, "fraction": frac})
    if frac > tolerance:
        return Signal("reward_actually_ran", SUSPECT,
                      f"the authored reward failed on {n_sd}/{n_call} steps ({frac:.1%}) — partial "
                      "fallback contamination", {"safe_default": n_sd, "calls": n_call})
    return Signal("reward_actually_ran", OK,
                  f"the authored reward executed on every step ({n_sd}/{n_call} fallbacks)",
                  {"safe_default": n_sd, "calls": n_call})


def check_provenance(record: dict[str, Any]) -> Signal:
    """The record must be able to say WHAT produced it, or it cannot be replayed or audited."""
    # `not record.get(f)` would flag seed 0 and generation 0 — both LEGITIMATE values. Absence is
    # None or an empty string, never a falsy number (this exact trap flagged every seed-0 record as
    # garbage the first time this ran).
    missing = [f for f in ("reward_source_hash", "env_fingerprint", "candidate_id", "arm", "seed")
               if record.get(f) is None or record.get(f) == ""]
    if missing:
        return Signal("provenance", GARBAGE,
                      f"record cannot identify what produced it (missing {missing}) — it is not "
                      "replayable", {"missing": missing})
    return Signal("provenance", OK, "reward hash, environment fingerprint and identity all present",
                  {})


def check_training_happened(wall_clock: Any, safe_call_count: Any = None) -> Signal:
    """A record that appeared without spending time did not train — UNLESS it is simply untimed.

    ``src/orchestration/test_leg.py`` builds every TEST-leg record with a hardcoded
    ``"wall_clock": 0.0``. Judging on the timer alone therefore condemns the entire SCORED leg as
    GARBAGE — which is what happened at 2026-07-28 06:53Z, when the sentinel raised CRITICAL on
    ``baseline_return_minus_cvar-s24``, a record carrying ``train_safe_call_count: 400000``, a full
    ``train_curve`` and real test returns. So judge on EVIDENCE OF TRAINING and treat the clock as
    what it is: provenance, not proof. A record with no time AND no reward calls is still GARBAGE —
    that catch is preserved, because absence of evidence must not read as evidence of health.
    """
    try:
        secs = float(wall_clock)
    except (TypeError, ValueError):
        return Signal("training_happened", SUSPECT, "wall_clock is not a number", {})
    try:
        steps = int(safe_call_count) if safe_call_count is not None else 0
    except (TypeError, ValueError):
        steps = 0
    if secs <= 0:
        if steps > 0:
            return Signal("training_happened", OK,
                          f"untimed (wall_clock {secs}; test-leg records hardcode 0.0) but "
                          f"{steps:,} reward calls prove training ran",
                          {"wall_clock": secs, "safe_call_count": steps})
        return Signal("training_happened", GARBAGE,
                      f"wall_clock is {secs} and no reward calls — nothing was trained",
                      {"wall_clock": secs})
    return Signal("training_happened", OK, f"training took {secs:.0f}s", {"wall_clock": secs})


def assess_record(record: dict[str, Any]) -> list[Signal]:
    """Every effect-blind sanity signal for ONE archived record.

    NOTE: SEARCH RECORDS CARRY ``val_returns``, TEST RECORDS CARRY ``test_returns`` (fixed 2026-07-28,
    live, on the campaign's very first archived record). This read only ever looked for
    ``test_returns``, so every SEARCH record — the kind that lands FIRST, hours before any test
    record exists — resolved to an empty array and was reported as
    ``GARBAGE: the record carries NO realized returns``.

    That mattered more than a cosmetic mislabel. The first record of the confirmatory campaign
    (``tpe-c0``) raised a CRITICAL "stop and diagnose before more compute is spent" alert while
    being, on inspection, perfectly healthy: **694 val_returns** (exactly the registered
    ``track_length``), ``train_safe_default_count: 0``, ``dev=cpu``. Hundreds of search records
    follow it, so the alarm would have cried wolf on every one — and an operator who learns to
    ignore a CRITICAL is an operator who will ignore the real one. This repository's own review
    history names "instruments reporting success while measuring nothing" as a recurring class;
    this is its mirror image, an instrument reporting FAILURE while reading the wrong field.

    Prefer ``test_returns`` (the scored leg, which is what this instrument was written for) and fall
    back to ``val_returns`` so a search record is judged on the returns it actually has. Every
    downstream check stays effect-blind: none of them reads a performance value or compares arms.
    """
    m = record.get("metrics", {}) or {}
    _raw = (m.get("test_returns") or record.get("test_returns")
            or m.get("val_returns") or record.get("val_returns") or [])
    rets = np.asarray(_raw, dtype=float)
    return [
        check_provenance(record),
        check_training_happened(record.get("wall_clock"), m.get("train_safe_call_count")),
        check_returns_finite(rets),
        check_returns_nondegenerate(rets),
        check_returns_plausible(rets),
        check_reward_actually_ran(m.get("train_safe_default_count"),
                                  m.get("train_safe_call_count")),
    ]


def verdict(signals: list[Signal]) -> str:
    return max((s.status for s in signals), key=lambda s: _RANK[s], default=OK)


# ── GATHER + CLI ──────────────────────────────────────────────────────────────────────────────

def assess_seed(root: str | Path, seed: int | None = None) -> dict[str, Any]:
    """Assess every record for the FIRST (or a named) seed under a campaign root. Effect-blind."""
    from src.io.results import load_all

    base = Path(root)
    roots = [p for p in (base / "test", base / "search") if p.is_dir()]
    # `search_leg_*` too, NOT just `test_leg_*`: for most of a campaign the ONLY leg records that
    # exist are search-leg ones (the test legs come later), so omitting them left this gate blind to
    # every replication leg -- 23 of the archive's 29 records at 2026-07-28 06:10Z, and precisely
    # where the authoring failures concentrate.
    roots += [p for p in sorted(base.glob("test_leg_*")) if p.is_dir()]
    roots += [p for p in sorted(base.glob("search_leg_*")) if p.is_dir()]
    records: list[dict[str, Any]] = []
    for r in roots:
        for arm_dir in sorted(p for p in r.iterdir() if p.is_dir()):
            try:
                records.extend(load_all(str(arm_dir)))
            except Exception:  # noqa: BLE001 — a torn record must not blind the whole gate
                continue
    if not records:
        return {"status": "no_records", "seed": seed, "n_records": 0,
                "note": "nothing has landed yet — this says nothing about quality"}
    seeds = sorted({int(r["seed"]) for r in records if r.get("seed") is not None})
    target = seeds[0] if seed is None else int(seed)
    chosen = [r for r in records if int(r.get("seed", -1)) == target]
    if not chosen:
        return {"status": "seed_absent", "seed": target, "seeds_present": seeds[:8],
                "n_records": 0}
    per_record = []
    for rec in sorted(chosen, key=lambda r: (str(r.get("arm")), str(r.get("run_id")))):
        sigs = assess_record(rec)
        per_record.append({"run_id": rec.get("run_id"), "arm": rec.get("arm"),
                           "verdict": verdict(sigs),
                           "signals": [s.as_dict() for s in sigs if s.status != OK]})
    worst = max((r["verdict"] for r in per_record), key=lambda s: _RANK[s])
    return {"status": "assessed", "seed": target, "n_records": len(per_record),
            "verdict": worst, "records": per_record,
            "note": "EXECUTION quality only — no performance value is read and no arm is compared, "
                    "so this cannot preview the result or affect the single confirmatory look"}


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — the earliest-warning check must not die printing
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="campaign archive root (e.g. outputs/campaign_cluster)")
    ap.add_argument("--seed", type=int, default=None, help="default: the first seed present")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out = assess_seed(args.root, args.seed)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("verdict", OK) != GARBAGE else 2

    print(f"FIRST-SEED SANITY — seed {out.get('seed')} — {out['status']}")
    if out["status"] != "assessed":
        print("  " + str(out.get("note", "")))
        return 0
    print(f"  {out['n_records']} record(s); overall: {out['verdict']}")
    for r in out["records"]:
        if r["verdict"] == OK:
            print(f"    [OK      ] {r['arm']:<22} {r['run_id']}")
        else:
            print(f"    [{r['verdict']:<8}] {r['arm']:<22} {r['run_id']}")
            for s in r["signals"]:
                print(f"                 - {s['status']}: {s['detail'][:100]}")
    print("  " + out["note"])
    return 2 if out["verdict"] == GARBAGE else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def assess_recent(root: str | Path, limit: int = 300) -> dict[str, Any]:
    """The LIVE hook: assess the most recently archived records, newest first. Effect-blind.

    Designed for the sentinel's ``--watch`` loop, so the interval between a garbage record landing
    and an alert reaching a phone is one poll (~2 min by default) rather than the two days it takes
    to reach the first statistical result. Deliberately STATELESS — it re-reads the newest ``limit``
    records each tick instead of tracking a high-water mark, because a watcher that loses its cursor
    on restart would silently stop checking; re-alerting is harmless (the notifier dedupes), whereas
    a missed record is not.

    Cheap by construction: it stats record mtimes and only PARSES the newest ``limit``.
    """
    base = Path(root)
    paths: list[Path] = []
    for r in [base / "test", base / "search",
              *sorted(base.glob("test_leg_*")), *sorted(base.glob("search_leg_*"))]:
        if r.is_dir():
            paths.extend(p for p in r.rglob("record.json")
                         if not any(x.startswith(".pull_tmp") for x in p.parts))
    if not paths:
        return {"n_assessed": 0, "garbage": [], "suspect": [], "note": "no records yet"}
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    garbage: list[dict[str, Any]] = []
    suspect: list[dict[str, Any]] = []
    assessed = 0
    for p in paths[:limit]:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a torn record is caught by the archive integrity gate
            continue
        if not isinstance(rec, dict):
            continue
        assessed += 1
        sigs = assess_record(rec)
        v = verdict(sigs)
        if v == OK:
            continue
        entry = {"run_id": rec.get("run_id"), "arm": rec.get("arm"), "seed": rec.get("seed"),
                 "reasons": [s.detail[:90] for s in sigs if s.status != OK]}
        (garbage if v == GARBAGE else suspect).append(entry)
    return {"n_assessed": assessed, "garbage": garbage, "suspect": suspect}
