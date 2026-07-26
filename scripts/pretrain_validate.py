#!/usr/bin/env python3
"""EXHAUSTIVE PRE-TRAINING VALIDATION — the last gate before compute is committed.

WHY A NEW GATE, AND WHAT IT IS *NOT*
------------------------------------
``scripts/preflight.py`` already guards the HOST and the CONFIG MIRRORS (disk, RAM, VRAM, keys,
retry layer, freeze state, budget/model/generation mirrors) and ``scripts/freeze.py --check`` guards
the DESIGN-OF-RECORD hashes. Neither is duplicated here. What neither can see is the **science
path**: whether a registered claim has an executable route to a number, whether the sandbox actually
refuses bad code, whether the authors can author, and whether the run will record what the analysis
later needs. Those are the failures that do not announce themselves — they surface as a plausible
result computed from nothing.

THE DESIGN PRINCIPLE: EVERY CHECK MUST PROVE IT CAN FAIL
--------------------------------------------------------
This repository's own review history is unambiguous: **7 of 12 findings in the 2026-07-26 code
review were instruments reporting success while measuring nothing** — a contamination screen that
asked and learnt nothing, a Phase-0 gate that trained nothing, two timing instruments that clocked
warmup as training, an audit that pinned nothing. A validator is exactly that kind of instrument, so
a green validator that has never been observed to go red is worth nothing.

Hence ``--self-test``: every check is handed a KNOWN-BAD input and must return FAIL. A check that
cannot be made to fail is reported as NOT-FALSIFIABLE and the self-test exits non-zero. Run it in CI
and before every launch; a validator you have watched fail is a validator you can believe.

STRUCTURE
---------
Each check is a PURE function over explicitly-gathered inputs, so it is unit-testable with no disk,
no network and no GPU; the gatherers are separate and best-effort. Statuses are PASS / WARN / FAIL /
SKIP, where SKIP means "the input genuinely does not exist yet" and is never silently green.

Usage::

    python scripts/pretrain_validate.py                      # full gate, human-readable
    python scripts/pretrain_validate.py --strict             # WARN counts as failure
    python scripts/pretrain_validate.py --json               # machine-readable
    python scripts/pretrain_validate.py --self-test          # PROVE every check can fail
    python scripts/pretrain_validate.py --gates-dir outputs/leg_gates_20260726_r112
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_RANK = {PASS: 0, SKIP: 1, WARN: 2, FAIL: 3}


@dataclass
class Verdict:
    name: str
    category: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "category": self.category, "status": self.status,
                "detail": self.detail, "evidence": self.evidence}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PURE CHECKS — each one is falsifiable, and `--self-test` proves it
# ══════════════════════════════════════════════════════════════════════════════════════════════

def check_registered_nodes_computable(registered: dict[str, Any],
                                      implemented: dict[str, Any]) -> Verdict:
    """Every registered confirmatory node must have a route to an actual number.

    THE FAILURE THIS CATCHES, verbatim from this repo's history: a node is ratified into the
    validity tier, the write-up promises it, and NOTHING can compute it — "a registered NAME with no
    registered VALUE" (the R84 lesson; row 36 shipped ratified-but-unimplemented and had to be
    caught by hand). A missing entry here means the campaign runs for weeks and the claim cannot be
    made at the end, which is unrecoverable: the compute is spent.
    """
    missing = sorted(set(registered) - set(implemented))
    orphan = sorted(set(implemented) - set(registered))
    if missing:
        return Verdict("registered_nodes_computable", "DESIGN", FAIL,
                       f"{len(missing)} registered node(s) have NO implementation route: {missing} "
                       "— the campaign would finish unable to compute a claim it promises",
                       {"missing": missing, "orphan": orphan})
    if orphan:
        return Verdict("registered_nodes_computable", "DESIGN", WARN,
                       f"implementation defines node(s) the registration does not: {orphan} "
                       "(dead code, or an unregistered claim)", {"orphan": orphan})
    return Verdict("registered_nodes_computable", "DESIGN", PASS,
                   f"all {len(registered)} registered nodes are computable",
                   {"nodes": sorted(registered)})


def check_identification(per_arm_config: dict[str, dict[str, Any]],
                         allowed_varying: tuple[str, ...] = ("feedback_kind",)) -> Verdict:
    """ONLY the reward/feedback may vary across the LLM arms — the study's whole identification.

    If any other field differs between arms, the measured effect is no longer attributable to the
    feedback CONTENT, and the headline claim silently becomes uninterpretable. This is the litmus
    the project applies to every proposal, enforced mechanically here.
    """
    if len(per_arm_config) < 2:
        return Verdict("identification_only_feedback_varies", "DESIGN", SKIP,
                       f"need >=2 arms to compare, got {len(per_arm_config)}", {})
    keys: set[str] = set()
    for cfg in per_arm_config.values():
        keys |= set(cfg)
    varying = []
    for k in sorted(keys):
        vals = {json.dumps(cfg.get(k), sort_keys=True, default=str) for cfg in per_arm_config.values()}
        if len(vals) > 1:
            varying.append(k)
    illegal = [k for k in varying if k not in allowed_varying]
    if illegal:
        return Verdict("identification_only_feedback_varies", "DESIGN", FAIL,
                       f"fields other than {list(allowed_varying)} DIFFER across arms: {illegal} — "
                       "the effect is no longer attributable to the feedback content",
                       {"illegal": illegal, "varying": varying})
    return Verdict("identification_only_feedback_varies", "DESIGN", PASS,
                   f"only {varying or list(allowed_varying)} varies across {len(per_arm_config)} arms",
                   {"varying": varying})


def check_matched_budget(budget_by_arm: dict[str, int]) -> Verdict:
    """Every arm must get the SAME number of candidate attempts (H4a's matched-budget premise).

    Matched budget means matched ATTEMPTS, not matched successes — a weak author spending its budget
    on rejected candidates is the capability being measured, and is fair. An UNEQUAL budget is not:
    it hands one arm more search.
    """
    if not budget_by_arm:
        return Verdict("matched_budget", "DESIGN", SKIP, "no per-arm budget available", {})
    distinct = sorted(set(budget_by_arm.values()))
    if len(distinct) > 1:
        return Verdict("matched_budget", "DESIGN", FAIL,
                       f"candidate budget DIFFERS across arms: {budget_by_arm} — one arm gets more "
                       "search than another, breaking the matched-budget comparison",
                       {"budgets": budget_by_arm})
    return Verdict("matched_budget", "DESIGN", PASS,
                   f"all {len(budget_by_arm)} arms share budget {distinct[0]}",
                   {"budget": distinct[0]})


def check_sandbox_defences(rejected: int, attempted: int, untyped_escapes: int,
                           safe_default_ok: bool, flagged_ok: bool) -> Verdict:
    """The two defences that make untrusted LLM code safe, PROVEN on real known-bad sources.

    Defence 1 — ``validate_once`` must reject bad code BEFORE training is spent, raising the TYPED
    ``SandboxError`` callers handle. Defence 2 — ``safe_call`` must substitute a finite neutral
    reward and FLAG the candidate, so a rollout degrades instead of crashing. An UNTYPED exception
    escaping the sandbox is a hard failure: it would propagate into the driver and can kill an arm.
    """
    if untyped_escapes:
        return Verdict("sandbox_defences", "SAFETY", FAIL,
                       f"{untyped_escapes} UNTYPED exception(s) escaped the sandbox — callers only "
                       "handle SandboxError, so these can kill a run",
                       {"untyped_escapes": untyped_escapes})
    if attempted and rejected == 0:
        return Verdict("sandbox_defences", "SAFETY", FAIL,
                       f"validate_once accepted ALL {attempted} known-bad sources — the gate is "
                       "not gating (a fail-open sandbox admits crashing code into training)",
                       {"attempted": attempted, "rejected": rejected})
    if not (safe_default_ok and flagged_ok):
        return Verdict("sandbox_defences", "SAFETY", FAIL,
                       f"safe_call did not neutralise+flag a crashing reward "
                       f"(neutralised={safe_default_ok}, flagged={flagged_ok}) — a runtime failure "
                       "would either crash the rollout or pass silently",
                       {"safe_default_ok": safe_default_ok, "flagged_ok": flagged_ok})
    return Verdict("sandbox_defences", "SAFETY", PASS,
                   f"defence 1 rejected {rejected}/{attempted} known-bad sources with 0 untyped "
                   "escapes; defence 2 neutralised and flagged a runtime crash",
                   {"rejected": rejected, "attempted": attempted})


def check_leg_readiness(summaries: list[dict[str, Any]], expected_legs: int,
                        floor: float = 1.0) -> Verdict:
    """Every registered leg must be live, pinned, and authoring at the compliance floor.

    Reads the archived ``leg_gates`` verdicts rather than re-billing the providers. A leg BELOW the
    floor, a pin that did not round-trip, or a provider MISROUTE each mean the executed author is
    not the registered one — which is a reproducibility claim failing, not a nuisance.
    """
    if not summaries:
        return Verdict("leg_readiness", "AUTHORING", SKIP,
                       "no leg-gate verdicts found — run scripts/leg_gates.py --all", {})
    low, badpin, badprov = [], [], []
    for s in summaries:
        leg = s.get("leg", "?")
        rate = s.get("compliance_rate")
        if rate is not None and float(rate) < floor:
            low.append(f"{leg}={rate}")
        if "FICTIONAL" in str(s.get("pin_roundtrip", "")) or "IGNORED" in str(s.get("pin_roundtrip", "")):
            badpin.append(leg)
        if "MISROUTE" in str(s.get("provider_roundtrip", "")):
            badprov.append(leg)
    problems = []
    if low:
        problems.append(f"below the {floor:.0%} compliance floor: {low}")
    if badpin:
        problems.append(f"reasoning pin did NOT round-trip: {badpin}")
    if badprov:
        problems.append(f"served provider outside the pin: {badprov}")
    if problems:
        return Verdict("leg_readiness", "AUTHORING", FAIL, "; ".join(problems),
                       {"low": low, "bad_pin": badpin, "bad_provider": badprov})
    if len(summaries) < expected_legs:
        return Verdict("leg_readiness", "AUTHORING", WARN,
                       f"only {len(summaries)}/{expected_legs} legs have gate verdicts — the rest "
                       "are UNMEASURED, which is not the same as passing",
                       {"measured": len(summaries), "expected": expected_legs})
    return Verdict("leg_readiness", "AUTHORING", PASS,
                   f"all {len(summaries)} legs at/above the {floor:.0%} floor, pins and providers "
                   "round-tripped", {"legs": len(summaries)})


def check_executable_yield(yield_by_leg: dict[str, tuple[int, int]]) -> Verdict:
    """Format compliance is not executability — measure what the campaign can actually USE.

    MEASURED 2026-07-26: qwen3.5-9b scored 1.00 on the registered compliance gate while only 5 of 20
    of its rewards survived 12 contract steps. A leg at ZERO is unusable and must fail loudly; a low
    but non-zero yield is a capability FINDING (reported, never silently treated as reliability).
    """
    if not yield_by_leg:
        return Verdict("executable_yield", "AUTHORING", SKIP, "no archived responses to execute", {})
    dead = [leg for leg, (ok, n) in yield_by_leg.items() if n and ok == 0]
    if dead:
        return Verdict("executable_yield", "AUTHORING", FAIL,
                       f"leg(s) produced NO executable reward at all: {dead} — the leg cannot "
                       "contribute a winner", {"dead": dead})
    rates = {leg: round(ok / n, 3) for leg, (ok, n) in yield_by_leg.items() if n}
    weak = {k: v for k, v in rates.items() if v < 0.5}
    if weak:
        return Verdict("executable_yield", "AUTHORING", WARN,
                       f"low executable yield (a capability finding, report it as such — do NOT "
                       f"report format compliance as reliability): {weak}", {"rates": rates})
    return Verdict("executable_yield", "AUTHORING", PASS,
                   f"every leg yields executable rewards (min {min(rates.values()):.0%})",
                   {"rates": rates})


def check_determinism_envelope(recorded_keys: set[str]) -> Verdict:
    """The determinism-relevant facts must be RECORDED, or a violation is undetectable by audit.

    The operative rule: a knob that can vary across records must be visible in the archive in
    the same change that introduces it. Device and BLAS thread counts change floating-point reduction
    order, so an unrecorded mix silently confounds every CRN-paired contrast.
    """
    required = {"device", "threads"}
    have = {r for r in required
            if any(r in k.lower() or (r == "threads" and "num_threads" in k.lower())
                   for k in recorded_keys)}
    missing = sorted(required - have)
    if missing:
        return Verdict("determinism_envelope", "REPRODUCIBILITY", FAIL,
                       f"provenance does NOT record {missing} — a device/thread mix would be "
                       "undetectable by audit, and every paired contrast assumes it cannot happen",
                       {"missing": missing, "recorded": sorted(recorded_keys)[:12]})
    return Verdict("determinism_envelope", "REPRODUCIBILITY", PASS,
                   "device and thread regime are recorded in the provenance",
                   {"recorded_sample": sorted(recorded_keys)[:12]})


def check_splits_no_lookahead(splits: dict[str, tuple[Any, Any]],
                              executed_purge_sessions: int | None = None,
                              required_purge_sessions: int = 0) -> Verdict:
    """Train < val < test strictly, and the EXECUTED purge honours the registered floor.

    A single overlapping boundary invalidates every out-of-sample claim in the dissertation, and it
    is invisible in the results (an overlapping split just looks like good performance).

    NOTE: THE PURGE IS NOT A CALENDAR GAP BETWEEN CONFIG BOUNDARIES, and checking it that way produces a
    FALSE FAILURE (observed while building this gate). ``config/data.yaml`` records the NOMINAL
    windows, which are adjacent by construction; the executed validation start is resolved by
    ``loaders.embargoed_val_start`` as ``max(materialized_floor, first_post_train + max(embargo,
    lookback))``, and at the production ``lookback=60`` the LOOKBACK purge dominates the 21-session
    embargo floor (R18). So the ordering is checked on the config, and the purge is checked on what
    the loader ACTUALLY resolves — never on a date subtraction. A gate that cries wolf is worse than
    no gate: it teaches the operator to ignore the one alarm that matters.
    """
    order = ["train", "val", "test"]
    present = [s for s in order if s in splits]
    if len(present) < 2:
        return Verdict("splits_no_lookahead", "DATA", SKIP,
                       f"need >=2 splits to order, got {present}", {})
    problems = []
    for a, b in zip(present, present[1:]):
        a_end, b_start = splits[a][1], splits[b][0]
        if a_end is None or b_start is None:
            continue
        if b_start <= a_end:
            problems.append(f"{b} starts {b_start} on/before {a} ends {a_end} — OVERLAP")
    if executed_purge_sessions is not None and executed_purge_sessions < required_purge_sessions:
        problems.append(f"executed purge {executed_purge_sessions} sessions < the registered "
                        f"{required_purge_sessions}-session floor")
    if problems:
        return Verdict("splits_no_lookahead", "DATA", FAIL,
                       "look-ahead risk: " + "; ".join(problems),
                       {"splits": {k: [str(v[0]), str(v[1])] for k, v in splits.items()},
                        "executed_purge_sessions": executed_purge_sessions})
    detail = f"{' < '.join(present)} strictly ordered"
    if executed_purge_sessions is not None:
        detail += (f"; loader-resolved purge = {executed_purge_sessions} sessions "
                   f">= the {required_purge_sessions}-session floor")
    return Verdict("splits_no_lookahead", "DATA", PASS, detail,
                   {"order": present, "executed_purge_sessions": executed_purge_sessions})


def check_freeze_gate(returncode: int | None, frozen: bool | None) -> Verdict:
    """Delegate to the design-of-record gate; a RED freeze gate is never a launchable state."""
    if returncode is None:
        return Verdict("freeze_gate", "DESIGN", SKIP, "freeze.py --check was not run", {})
    if returncode != 0:
        return Verdict("freeze_gate", "DESIGN", FAIL,
                       f"freeze.py --check exited {returncode} — the executed config and the "
                       "design of record DISAGREE", {"returncode": returncode})
    return Verdict("freeze_gate", "DESIGN", PASS,
                   f"freeze gate green (frozen={frozen}; pre-freeze null is expected before GO)",
                   {"frozen": frozen})


#: name -> (check callable, GOOD kwargs, BAD kwargs). The BAD row is the falsifiability proof.
SELF_TEST_CASES: dict[str, tuple[Callable[..., Verdict], dict, dict]] = {
    "registered_nodes_computable": (
        check_registered_nodes_computable,
        {"registered": {"N1": {}}, "implemented": {"N1": {}}},
        {"registered": {"N1": {}, "N2": {}}, "implemented": {"N1": {}}}),
    "identification_only_feedback_varies": (
        check_identification,
        {"per_arm_config": {"a": {"feedback_kind": "tail", "steps": 400_000},
                            "b": {"feedback_kind": "scalar", "steps": 400_000}}},
        {"per_arm_config": {"a": {"feedback_kind": "tail", "steps": 400_000},
                            "b": {"feedback_kind": "scalar", "steps": 200_000}}}),
    "matched_budget": (
        check_matched_budget, {"budget_by_arm": {"a": 30, "b": 30}},
        {"budget_by_arm": {"a": 30, "b": 15}}),
    "sandbox_defences": (
        check_sandbox_defences,
        {"rejected": 3, "attempted": 3, "untyped_escapes": 0,
         "safe_default_ok": True, "flagged_ok": True},
        {"rejected": 0, "attempted": 3, "untyped_escapes": 1,
         "safe_default_ok": True, "flagged_ok": True}),
    "leg_readiness": (
        check_leg_readiness,
        {"summaries": [{"leg": "x", "compliance_rate": 1.0}], "expected_legs": 1},
        {"summaries": [{"leg": "x", "compliance_rate": 0.6}], "expected_legs": 1}),
    "executable_yield": (
        check_executable_yield, {"yield_by_leg": {"x": (10, 10)}},
        {"yield_by_leg": {"x": (0, 10)}}),
    "determinism_envelope": (
        check_determinism_envelope, {"recorded_keys": {"device", "torch_num_threads"}},
        {"recorded_keys": {"python_version"}}),
    "splits_no_lookahead": (
        check_splits_no_lookahead,
        {"splits": {"train": (1, 10), "val": (11, 20), "test": (21, 30)},
         "executed_purge_sessions": 60, "required_purge_sessions": 21},
        {"splits": {"train": (1, 15), "val": (11, 20), "test": (21, 30)},
         "executed_purge_sessions": 5, "required_purge_sessions": 21}),
    "freeze_gate": (
        check_freeze_gate, {"returncode": 0, "frozen": False},
        {"returncode": 1, "frozen": False}),
}


def run_self_test() -> int:
    """PROVE every check can fail. A validator never seen red is worth nothing."""
    print("SELF-TEST — each check is handed a KNOWN-BAD input and must return FAIL\n")
    bad = 0
    for name, (fn, good_kw, bad_kw) in SELF_TEST_CASES.items():
        g, b = fn(**good_kw), fn(**bad_kw)
        ok_green = g.status in (PASS, WARN, SKIP)
        ok_red = b.status == FAIL
        mark = "OK " if (ok_green and ok_red) else "BAD"
        if not (ok_green and ok_red):
            bad += 1
        print(f"  [{mark}] {name:<36} good->{g.status:<4} bad->{b.status:<4}"
              + ("" if ok_red else "   <-- NOT FALSIFIABLE"))
    print(f"\n{len(SELF_TEST_CASES) - bad}/{len(SELF_TEST_CASES)} checks proven falsifiable")
    return 1 if bad else 0


# ══════════════════════════════════════════════════════════════════════════════════════════════
# GATHERERS — impure, best-effort; absence yields SKIP, never a silent PASS
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _gather(gates_dir: Path) -> dict[str, Any]:
    import yaml

    g: dict[str, Any] = {}
    prereg = yaml.safe_load((REPO / "config" / "preregistration.yaml").read_text(encoding="utf-8"))

    try:
        from src.inference.validity_tier import NODE_SOURCES
        g["registered_nodes"] = dict(prereg["inference"]["validity_tier"]["nodes"])
        g["implemented_nodes"] = dict(NODE_SOURCES)
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.arms.factory import build_arm_specs  # type: ignore
        specs = build_arm_specs()
        g["per_arm_config"] = {k: (v if isinstance(v, dict) else vars(v)) for k, v in specs.items()}
    except Exception:  # noqa: BLE001 — arms API varies; identification then SKIPs rather than lies
        pass

    try:
        camp = yaml.safe_load((REPO / "config" / "campaign.yaml").read_text(encoding="utf-8"))
        cpa = int(camp.get("candidates_per_arm") or 0)
        arms = list(camp.get("arms") or [])
        if cpa and arms:
            g["budget_by_arm"] = {a: cpa for a in arms}
    except Exception:  # noqa: BLE001
        pass

    # NEWEST verdict per leg, across every gate directory — never just the one that was passed in.
    # Pointing the gate at a stale directory would certify a leg on LAST WEEK's result (observed
    # while building this: the pre-R112 run still showed glm=0.6/kimi=0.8 after both were fixed).
    # A gate that can be aimed at old evidence is a gate that can be talked into a false green.
    latest: dict[str, tuple[float, dict[str, Any]]] = {}
    search_dirs = [gates_dir] + sorted((REPO / "outputs").glob("leg_gates*"))
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.summary.json")):
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            leg, mt = row.get("leg", p.stem), p.stat().st_mtime
            if leg not in latest or mt > latest[leg][0]:
                latest[leg] = (mt, row)
    # R106 (2026-07-27): validate EXACTLY the registered roster, never whatever the archive happens to
    # contain. The substitution gemini-3.5-flash -> gemini-2.5-flash left a verdict for the RETIRED leg
    # in the pre-R106 archive, and this gate reported "all 11 legs" — counting a leg that will never
    # run. Benign that time (it passed), but a retired leg's stale verdict must never be able to fail
    # the gate for a leg the campaign does not execute, nor pad the readiness count. The registration
    # is the authority on which legs exist; the archive is only evidence about them.
    roster = [str(x.get("label")) for x in (prereg.get("model_suite", {}).get("legs") or [])
              if isinstance(x, dict) and x.get("label")]
    if roster:
        retired = sorted(set(latest) - set(roster))
        if retired:
            g["retired_legs_ignored"] = retired
        latest = {leg: v for leg, v in latest.items() if leg in roster}
    sums = [row for _mt, row in latest.values()]
    g["leg_summaries"] = sums
    g["leg_verdict_sources"] = len(search_dirs)
    g["expected_legs"] = len(prereg.get("model_suite", {}).get("legs", []) or [])

    try:
        from scripts.capture_env import _DETERMINISM_ENV_KEYS  # type: ignore
        keys = set(_DETERMINISM_ENV_KEYS) | {"device", "torch_num_threads"}
        g["recorded_keys"] = keys
    except Exception:  # noqa: BLE001
        pass

    try:
        data = yaml.safe_load((REPO / "config" / "data.yaml").read_text(encoding="utf-8"))
        sp = data.get("splits") or {}
        got = {}
        for k, v in sp.items():
            if isinstance(v, dict) and "start" in v and "end" in v:
                got[k] = (v["start"], v["end"])
        if got:
            g["splits"] = got
            g["required_purge"] = int(data.get("embargo_days") or 21)
    except Exception:  # noqa: BLE001
        pass

    # Resolve the EXECUTED purge from the loader itself (never a date subtraction — see the check's
    # docstring: the lookback purge dominates the embargo floor at production lookback=60).
    try:
        import numpy as np

        from src.data.loaders import embargoed_val_start, load_gold_panel

        train_end = g["splits"]["train"][1]
        val_end = g["splits"].get("val", (None, None))[1]
        # The panel must SPAN the validation window or the indices clamp to the array end and the
        # purge reads as 0 — a false alarm, not a finding. `load_gold_panel` returns GoldLoadResult;
        # the dates live on `.panel` (reading them off the result silently yields an empty array).
        res = load_gold_panel(end=str(val_end) if val_end else None)
        dates = np.asarray(getattr(getattr(res, "panel", res), "dates", []))
        lookback = int((yaml.safe_load((REPO / "config" / "campaign.yaml")
                                       .read_text(encoding="utf-8")) or {}).get("lookback") or 60)
        first_post = int(np.searchsorted(dates, np.datetime64(str(train_end)), side="right"))
        idx = int(embargoed_val_start(dates, str(train_end),
                                      embargo_days=g.get("required_purge", 21), lookback=lookback))
        # Only trust the number when both indices sit strictly INSIDE the loaded dates; otherwise the
        # window does not cover the boundary and the difference is an artefact, so report nothing.
        if 0 <= first_post < len(dates) and 0 <= idx < len(dates):
            g["executed_purge"] = idx - first_post
    except Exception:  # noqa: BLE001 — unresolvable => the check reports ordering only, never a guess
        pass
    return g


def _gather_sandbox() -> dict[str, Any]:
    """Run REAL known-bad reward sources through the real defences (no network, no GPU)."""
    import numpy as np

    from src.orchestration.parallel import _FIXTURE
    from src.sandbox.executor import (SandboxEnvironmentError, SandboxError, candidate_failed,
                                      reset_failure_flag, safe_call, validate_once)

    bads = [
        "def reward(w, r, p, pr, info):\n    return info['reward_state']['n'], {}, None\n",
        "def reward(w, r, p, pr, info):\n    return 'not a float'\n",
        "import os\ndef reward(w, r, p, pr, info):\n    os.system('echo hi')\n    return 0.0, {}, None\n",
    ]
    rejected = untyped = 0
    for src in bads:
        try:
            validate_once(src, _FIXTURE)
        except SandboxEnvironmentError:
            # CONTRACT (src/sandbox/executor.py::SandboxEnvironmentError), mirroring
            # src/orchestration/parallel.py:365: a STARVED spawn environment is NOT a candidate
            # verdict. SandboxEnvironmentError SUBCLASSES SandboxError, so the handler below
            # counted it as `rejected` — and this is a SAFETY check, so on a contended box the
            # gate certified "rejected 3/3 known-bad sources" having never actually evaluated
            # them. A false GREEN on a defence proof is strictly worse than a false red. Raise:
            # the caller turns this into a FAIL verdict naming the starvation, so the gate
            # REFUSES to certify rather than mis-certifying (#75, 2026-07-27).
            raise
        except SandboxError:
            rejected += 1
        except Exception:  # noqa: BLE001
            untyped += 1
    ns: dict = {"np": np}
    exec(compile(bads[0], "<bad>", "exec"), ns)  # noqa: S102
    reset_failure_flag()
    total, _c, _s = safe_call(ns["reward"], np.ones(3) / 3, np.zeros(2), np.ones(3) / 3, 0.001,
                              {"reward_state": None})
    return {"rejected": rejected, "attempted": len(bads), "untyped_escapes": untyped,
            "safe_default_ok": total == 0.0, "flagged_ok": bool(candidate_failed())}


def _gather_executable_yield(gates_dir: Path) -> dict[str, tuple[int, int]]:
    from src.sandbox.executor import (SandboxEnvironmentError, SandboxError,
                                      extract_reward_source, validate_once)

    from src.orchestration.parallel import _FIXTURE

    # Same newest-per-leg rule as the summaries: aiming this at one thin directory would let it
    # report "every leg yields" from two legs and hide the weak one — the gate under-reporting
    # itself. The response set must cover the SAME legs the readiness verdict covers.
    newest: dict[str, Path] = {}
    for d in [gates_dir] + sorted((REPO / "outputs").glob("leg_gates*")):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            if p.stem not in newest or p.stat().st_mtime > newest[p.stem].stat().st_mtime:
                newest[p.stem] = p
    out: dict[str, tuple[int, int]] = {}
    for p in newest.values():
        ok = n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if row.get("gate") != "compliance":
                continue
            src = extract_reward_source(row.get("response") or "")
            if not src or "def reward(" not in src:
                continue
            n += 1
            try:
                validate_once(src, _FIXTURE)
                ok += 1
            except SandboxEnvironmentError:
                # Same contract as _gather_sandbox (#75): contention is not a candidate defect.
                # Swallowing it here silently DEPRESSES the per-model authoring-compliance rate
                # this function exists to compute — an infrastructure failure reported as a model
                # finding, inside the script that gates the launch, where a low yield is exactly
                # the signal an operator would act on. Fail loud; re-run on a freed box.
                raise
            except SandboxError:
                pass
            except Exception:  # noqa: BLE001
                pass
        if n:
            out[p.stem] = (ok, n)
    return out


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — a console codepage must never kill a gate
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates-dir", default="outputs/leg_gates_20260726_r112")
    ap.add_argument("--strict", action="store_true", help="treat WARN as failure")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="prove every check can FAIL, then exit")
    ap.add_argument("--skip-sandbox", action="store_true",
                    help="skip the live sandbox proof (it spawns child processes)")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    gates = REPO / args.gates_dir if not Path(args.gates_dir).is_absolute() else Path(args.gates_dir)
    g = _gather(gates)
    verdicts: list[Verdict] = []

    if "registered_nodes" in g:
        verdicts.append(check_registered_nodes_computable(g["registered_nodes"], g["implemented_nodes"]))
    if "per_arm_config" in g:
        verdicts.append(check_identification(g["per_arm_config"]))
    verdicts.append(check_matched_budget(g.get("budget_by_arm", {})))
    verdicts.append(check_leg_readiness(g.get("leg_summaries", []), g.get("expected_legs", 0)))
    if "recorded_keys" in g:
        verdicts.append(check_determinism_envelope(g["recorded_keys"]))
    if "splits" in g:
        verdicts.append(check_splits_no_lookahead(g["splits"], g.get("executed_purge"),
                                                  g.get("required_purge", 0)))

    if not args.skip_sandbox:
        try:
            verdicts.append(check_sandbox_defences(**_gather_sandbox()))
            verdicts.append(check_executable_yield(_gather_executable_yield(gates)))
        except Exception as exc:  # noqa: BLE001
            verdicts.append(Verdict("sandbox_defences", "SAFETY", FAIL,
                                    f"the sandbox proof itself could not run: {exc}", {}))

    try:
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "freeze.py"), "--check"],
                           cwd=REPO, capture_output=True, text=True, timeout=600)
        frozen = "frozen: true" in (r.stdout or "").lower()
        verdicts.append(check_freeze_gate(r.returncode, frozen))
    except Exception:  # noqa: BLE001
        verdicts.append(check_freeze_gate(None, None))

    failed = [v for v in verdicts if v.status == FAIL]
    warned = [v for v in verdicts if v.status == WARN]

    if args.json:
        print(json.dumps({"verdicts": [v.as_dict() for v in verdicts],
                          "failed": len(failed), "warned": len(warned)}, indent=2))
    else:
        print("PRE-TRAINING VALIDATION\n" + "=" * 92)
        for v in sorted(verdicts, key=lambda x: (-_RANK[x.status], x.category)):
            print(f"  [{v.status:<4}] {v.category:<16} {v.name:<36} {v.detail[:110]}")
        print("=" * 92)
        print(f"  {len(verdicts)} checks | FAIL={len(failed)} WARN={len(warned)}")
        print("  (run --self-test to prove each of these checks can actually go red)")

    if failed:
        return 2
    if warned and args.strict:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
