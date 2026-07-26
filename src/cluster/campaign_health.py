"""CAMPAIGN-LANE HEALTH — the five failure modes the existing sentinel cannot see.

``scripts/sentinel.py`` already covers the MACHINE and the RUN well (disk, RAM, silent hang, gate
failure rate, NaN/divergence rates, reward-scale drift, API errors, driver lease, queue health,
completion stall). What it has no visibility into is the thing that actually decides whether the
2026-07-26 CPU-lane campaign succeeds: **are we holding the capacity we assumed, is the critical
path advancing, and is the substrate staying homogeneous?**

Each function is PURE (no disk, no ssh) so it is unit-testable, and each returns the sentinel's own
``HealthCheck`` vocabulary so these compose into the existing report and alerting rather than
forming a second, parallel monitor.

THE FIVE, and why each earns its place — every one is a failure that would otherwise be SILENT:

1. :func:`check_capacity_accumulation` — the campaign plan's one UNMEASURED assumption is that
   ~8.5 h tasks ACCUMULATE to far more than the 636 cores measured with 20-min probes. If instead
   concurrency plateaus low, the reachable seed rung falls and we must know on DAY ONE, not at the
   Aug-27 stop. Compares the observed curve to the forecast and says which.
2. :func:`check_chain_progress` — the makespan is ``max(throughput, longest serial chain)``. The
   `bayes_opt` (25 serial) and TPE (20 serial) chains FLOOR the campaign, so a stalled chain wastes
   the whole run even while thousands of test-leg trainings stream in and every other indicator
   looks green. Nothing else watches the critical path.
3. :func:`check_host_failure_concentration` — a SINGLE bad node silently eats tasks. We measured
   this for real: ``node-d00a-230`` had no apptainer and returned rc=127 on every task routed to
   it. A global failure RATE hides it; concentration by host exposes it.
4. :func:`check_rung_forecast` — re-forecasts the reachable rung from the OBSERVED completion rate
   instead of the model, so the exogenous Aug-27 stop is planned against reality.
5. :func:`check_determinism_homogeneity` — CPU/CUDA and 1-thread/8-thread are not bit-identical,
   and every paired contrast depends on CRN pairing. The env fingerprint now records device and
   thread regime; this makes a mix visible DURING the run rather than at the post-hoc S6 audit,
   when it would be too late to re-run.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

# Reuse the sentinel's vocabulary rather than inventing a second one. Imported as `scripts.sentinel`
# (a PEP-420 namespace package from the repo root, exactly as tests/test_sentinel.py does) and NOT via
# a sys.path hack: `import sentinel` would bind a SECOND module object whose `HealthCheck` is a
# different class, so isinstance would fail and the two report vocabularies would silently diverge.
from scripts.sentinel import CRITICAL, INFO, OK, WARN, HealthCheck

__all__ = [
    "check_capacity_accumulation",
    "check_chain_progress",
    "check_host_failure_concentration",
    "check_rung_forecast",
    "check_determinism_homogeneity",
    "check_substrate_fields",
    "check_winner_execution_quality",
    "check_admin_kill",
    "check_record_sanity",
    "check_authoring_health",
    "check_unreadable_records",
    "check_arm_progress_symmetry",
    "check_seed_replication",
    "check_arm_differentiation",
    "check_duplicate_units",
    "check_test_window_consistency",
    "check_design_drift",
    "check_seed_alignment",
    "MEASURED_AUTHORING_YIELD",
]


def check_design_drift(frozen: bool, recorded_hash: str | None,
                       current_hash: str | None) -> HealthCheck:
    """Is the campaign STILL running the design that was frozen? Verified continuously, not once.

    ``freeze.py --check`` is a pre-launch step. After launch nothing re-verifies it, so an edit to
    any hash-bound file mid-campaign — a stray fix, a concurrent session, a well-meant "small"
    config change — silently splits the run in two: records before the edit were produced under the
    frozen design and records after it were not. The archive stays complete and healthy-looking, and
    the resulting dataset answers no single pre-registered question.

    This is unrecoverable in the worst way: you cannot tell afterwards which records belong to which
    design unless the drift was caught while it happened. Checking every poll costs one hash.

    Pre-freeze (``frozen`` False) this is deliberately silent — the hash MOVES with every legitimate
    design edit before GO, and alarming on that would make the check noise from the start.
    """
    if not frozen:
        return HealthCheck("design_drift", INFO,
                           "pre-freeze: the canonical hash moves with every legitimate design edit "
                           "until GO, so drift is not meaningful yet", {"frozen": False})
    if not recorded_hash or not current_hash:
        return HealthCheck("design_drift", WARN,
                           "frozen, but a hash is unavailable — drift cannot be verified, and an "
                           "unverifiable invariant is not a held one",
                           {"recorded": bool(recorded_hash), "current": bool(current_hash)})
    if recorded_hash != current_hash:
        return HealthCheck("design_drift", CRITICAL,
                           f"THE FROZEN DESIGN CHANGED MID-RUN: recorded {recorded_hash[:12]} vs "
                           f"current {current_hash[:12]} — records produced before and after the "
                           "edit answer DIFFERENT pre-registered questions. Stop, identify the "
                           "edited hash-bound file, and decide explicitly (revert, or register a "
                           "dated amendment and note which records precede it)",
                           {"recorded": recorded_hash, "current": current_hash})
    return HealthCheck("design_drift", OK,
                       f"the running design still matches the freeze ({recorded_hash[:12]})",
                       {"hash": recorded_hash})


def check_seed_alignment(seed_sets: dict[str, set], *, warn_loss: float = 0.05,
                         target_seeds: int | None = None) -> HealthCheck:
    """Are the arms actually PAIRED on the same seeds? Silent power loss, otherwise.

    Every headline contrast is paired on common random numbers, and the analysis takes the seed
    INTERSECTION of the arms it compares (``analyze_campaign._common``). That is the correct thing
    to do, but it is SILENT: if one arm falls behind or loses seeds, the intersection shrinks and the
    effective n drops without any error, any warning, or any visible change. The campaign reports
    "n=340" while the contrast actually ran on 240 — and the confidence intervals widen for a reason
    nobody can see.

    Effect-blind: pure set arithmetic over seed identifiers, never a value.
    """
    usable = {a: set(s) for a, s in seed_sets.items() if s}
    if len(usable) < 2:
        return HealthCheck("seed_alignment", INFO,
                           f"need >=2 arms with seeds to compare, have {len(usable)}", {})
    common = set.intersection(*usable.values())
    widest = max(len(s) for s in usable.values())
    if not widest:
        return HealthCheck("seed_alignment", INFO, "no seeds recorded yet", {})
    loss = 1.0 - (len(common) / widest)
    missing = {a: sorted(set.union(*usable.values()) - s)[:6]
               for a, s in usable.items() if s != set.union(*usable.values())}
    # STILL FILLING? Arms complete seeds at different rates, so mid-fill the intersection is
    # legitimately small or empty and says nothing about final alignment. Live at 2026-07-28 08:16Z
    # the eight baseline arms held 4-18 seeds each with genuinely disjoint sets (one arm on
    # [6,7,8,9], another on [22,23,24,25]) and this check reported CRITICAL "the arms share NO
    # common seed" — factually true, operationally meaningless, and days early. It is a REAL check
    # for the END of a leg, so the finding is downgraded while filling rather than deleted: an arm
    # that never catches up must still be caught, which the WARN below does.
    filling = target_seeds is not None and widest < 0.9 * int(target_seeds)
    if filling:
        return HealthCheck("seed_alignment", INFO,
                           f"leg still FILLING ({widest}/{int(target_seeds)} seeds on the deepest "
                           f"arm): {len(common)} common so far. Arms complete seeds at different "
                           "rates, so alignment is only meaningful near the rung target",
                           {"common": len(common), "widest": widest,
                            "target_seeds": int(target_seeds)})
    if not common:
        return HealthCheck("seed_alignment", CRITICAL,
                           "the arms share NO common seed — every paired contrast is empty, so "
                           "nothing can be compared at all",
                           {"per_arm_counts": {a: len(s) for a, s in usable.items()}})
    if loss > warn_loss:
        return HealthCheck("seed_alignment", WARN if loss < 0.25 else CRITICAL,
                           f"paired contrasts will run on {len(common)} common seeds although the "
                           f"deepest arm has {widest} — {loss:.0%} of the sample is silently lost to "
                           "misalignment, and the analysis intersects without warning",
                           {"common": len(common), "widest": widest, "loss": round(loss, 3),
                            "missing_by_arm": {k: v for k, v in list(missing.items())[:5]}})
    return HealthCheck("seed_alignment", OK,
                       f"arms aligned on {len(common)} common seeds (deepest {widest})",
                       {"common": len(common), "widest": widest})


def check_seed_replication(digest_by_arm: dict[str, dict[int, str]]) -> HealthCheck:
    """FAKE PRECISION — the most dangerous silent failure in the whole design.

    Every confidence interval, every p-value and the entire seed ladder rest on one assumption:
    different seeds produce genuinely DIFFERENT runs. If seeding broke — a seed never threaded into
    SAC, a cached env reused, a resume replaying one record under many ids — then n=568 is really
    n=1, and the analysis reports beautifully tight intervals computed from copies of a single
    trajectory. Nothing else in the system would notice: those records are present, complete,
    finite, non-degenerate and internally consistent. The failure passes every test of looking
    wrong, which is exactly why it needs its own.

    Effect-blind: it digests each seed's realized returns WITHIN one arm and asks only "are these
    the same numbers twice", never whether they are good.
    """
    if not digest_by_arm:
        return HealthCheck("seed_replication", INFO, "no per-seed records yet", {})
    offenders: dict[str, int] = {}
    checked = 0
    for arm, by_seed in digest_by_arm.items():
        if len(by_seed) < 2:
            continue
        checked += 1
        counts = Counter(by_seed.values())
        dupes = sum(c - 1 for c in counts.values() if c > 1)
        if dupes:
            offenders[arm] = dupes
    if not checked:
        return HealthCheck("seed_replication", INFO, "need >=2 seeds in some arm to compare", {})
    if offenders:
        return HealthCheck("seed_replication", CRITICAL,
                           f"IDENTICAL results across DIFFERENT seeds in {sorted(offenders)} — the "
                           "replication is FAKE, so every interval computed from it is fiction (n "
                           "is effectively 1). This is a seeding or resume defect and it cannot be "
                           "repaired after the fact",
                           {"duplicate_seed_pairs": offenders, "arms_checked": checked})
    return HealthCheck("seed_replication", OK,
                       f"every seed produced a distinct run across {checked} arm(s)",
                       {"arms_checked": checked})


def check_arm_differentiation(reward_hash_by_seed: dict[int, dict[str, str]],
                              llm_arms: tuple[str, ...] = (
                                  "distributional", "scalar", "placebo", "scalar_cvar5",
                                  "placebo_shuffled")) -> HealthCheck:
    """Did the ARM MANIPULATION actually reach the model? The study's identification, checked.

    Arms differ in exactly one thing: the feedback block the reward-designer sees. If two arms end
    up with the IDENTICAL authored reward source, then for those arms the experiment was not run —
    the same reward was trained twice under two labels and any contrast between them is
    structurally zero.

    Deliberately reported as something to VERIFY rather than as an effect, because two very
    different causes produce it and only a human can tell them apart: (a) a WIRING defect where the
    fed block never varied, which is a bug to fix before more compute is spent; or (b) the model
    genuinely wrote the same code from different feedback, which is a mechanism OBSERVATION for the
    write-up. This check does not pretend to distinguish them — it says "verify the fed blocks
    differed", which keeps it effect-blind (it inspects the INPUT, never the outcome).
    """
    if not reward_hash_by_seed:
        return HealthCheck("arm_differentiation", INFO, "no reward hashes recorded yet", {})
    collisions: list[str] = []
    seeds_checked = 0
    for seed, by_arm in sorted(reward_hash_by_seed.items()):
        pairs = {a: h for a, h in by_arm.items() if a in llm_arms and h}
        if len(pairs) < 2:
            continue
        seeds_checked += 1
        seen: dict[str, str] = {}
        for arm, h in sorted(pairs.items()):
            if h in seen:
                collisions.append(f"seed {seed}: {seen[h]} and {arm} share one reward source")
            else:
                seen[h] = arm
    if not seeds_checked:
        return HealthCheck("arm_differentiation", INFO,
                           "need >=2 LLM arms at a common seed to compare", {})
    if collisions:
        return HealthCheck("arm_differentiation", CRITICAL,
                           f"{len(collisions)} arm pair(s) trained on an IDENTICAL reward source: "
                           + "; ".join(collisions[:3])
                           + " — VERIFY the fed feedback blocks actually differed. If the config is "
                             "right this is a mechanism observation, not a bug; if it is not, the "
                             "manipulation never reached the model and those contrasts are void",
                           {"collisions": collisions[:10], "seeds_checked": seeds_checked})
    return HealthCheck("arm_differentiation", OK,
                       f"every LLM arm carries a distinct reward source across {seeds_checked} "
                       "seed(s)", {"seeds_checked": seeds_checked})


def check_duplicate_units(unit_counts: dict[str, int]) -> HealthCheck:
    """One (arm, seed) archived twice — silent double-counting in every paired statistic."""
    if not unit_counts:
        return HealthCheck("duplicate_units", INFO, "no units archived yet", {})
    dupes = {k: v for k, v in unit_counts.items() if v > 1}
    if dupes:
        return HealthCheck("duplicate_units", CRITICAL,
                           f"{len(dupes)} (arm, seed) unit(s) archived MORE THAN ONCE "
                           f"{sorted(dupes)[:4]} — the paired estimators double-count them and the "
                           "seed alignment every CRN contrast relies on is broken",
                           {"duplicates": dict(sorted(dupes.items())[:8])})
    return HealthCheck("duplicate_units", OK,
                       f"all {len(unit_counts)} (arm, seed) units are unique", {})


def check_test_window_consistency(lengths_by_arm: dict[str, set]) -> HealthCheck:
    """Every scored record must cover the SAME window, or the numbers are not comparable.

    A record whose return series is a different length was evaluated over a different period, so its
    Sharpe and CVaR sit on a different footing and pooling them silently mixes two experiments.
    """
    if not lengths_by_arm:
        return HealthCheck("test_window_consistency", INFO, "no scored records yet", {})
    seen: set = set()
    for v in lengths_by_arm.values():
        seen |= set(v)
    if not seen:
        return HealthCheck("test_window_consistency", INFO, "no return series recorded", {})
    if len(seen) > 1:
        odd = {a: sorted(v) for a, v in lengths_by_arm.items() if len(set(v)) > 1}
        return HealthCheck("test_window_consistency", CRITICAL,
                           f"scored records span {len(seen)} DIFFERENT return lengths "
                           f"{sorted(seen)[:5]} — those records were evaluated over different "
                           "windows, so their Sharpe and CVaR are not comparable and must not be "
                           "pooled", {"lengths": sorted(seen)[:8],
                                      "arms_with_mixed_lengths": dict(list(odd.items())[:5])})
    return HealthCheck("test_window_consistency", OK,
                       f"every scored record covers the same {sorted(seen)[0]}-step window", {})


#: MEASURED executable yield per author (2026-07-26, live gates: archived responses re-run through
#: the real AST gate and 12 contract steps). These are EVIDENCE, not guesses, and they are what makes
#: the streak detector below CALIBRATED rather than blind: a single global threshold would scream at
#: qwen3.5-9b on every healthy run (it genuinely fails ~3 of 4) and stay silent while haiku quietly
#: failed 40% (it genuinely never fails). Unmeasured authors fall back to the Anthropic-class rate.
MEASURED_AUTHORING_YIELD: dict[str, float] = {
    "qwen3.5-9b": 0.25,       # 5/20 executable at 1.00 format compliance - the capability floor
    "deepseek-v4-pro": 0.88,  # 14/16
    "nemotron-3-super": 0.89,  # 16/18
    "qwen3.6-27b": 0.96,      # 23/24
    "glm-5.2": 0.90,          # 9/10 post-R112
    "kimi-k3": 1.00, "gpt-5.6-luna": 1.00,
    # R106 (2026-07-27): the leg was SUBSTITUTED gemini-3.5-flash -> gemini-2.5-flash (3.5's
    # reasoning is mandatory, so it could not join the uniform reasoning-off suite). Keyed on the
    # OLD label this dict would never match the running leg, `_DEFAULT_YIELD` would silently
    # apply, and the earliest-warning authoring alarm would be mis-calibrated for it.
    "gemini-2.5-flash": 1.00,
    "haiku-4.5": 1.00, "sonnet-5": 1.00,
}
#: Opus-class default for anything unmeasured (the core campaign's confirmatory author).
_DEFAULT_YIELD = 0.95
#: A streak this improbable under the author's OWN measured rate is not bad luck.
_STREAK_IMPROBABILITY = 1e-3


def _streak_alarm_length(success_rate: float) -> int:
    """How many consecutive rejections are too many FOR THIS AUTHOR.

    Solves ``(1 - p)**k < 1e-3`` — the point at which an unbroken run of failures stops being
    consistent with the author's measured competence. It adapts automatically: a 100%-yield author
    trips after 3, while qwen3.5-9b (25%) is allowed ~24 before anyone is woken up.
    """
    p = min(max(float(success_rate), 0.01), 0.999)
    import math
    return max(2, int(math.ceil(math.log(_STREAK_IMPROBABILITY) / math.log(1.0 - p))))


def check_authoring_health(per_arm: dict[str, dict[str, Any]],
                           yields: dict[str, float] | None = None) -> HealthCheck:
    """THE EARLIEST possible alarm — fires in MINUTES, before any training completes.

    Authoring happens first: the model writes reward code and the sandbox accepts or rejects it
    within seconds, and every rejection is flushed to ``<arm>.failures.jsonl`` immediately. A record,
    by contrast, appears only after a FULL training (~3-8 h). So this is the earliest layer at which
    a systematic failure is observable at all, and it is where the most expensive silent failure
    lives: an arm whose authoring is broken burns its entire 30-candidate budget producing nothing,
    and on the current design nobody finds out until the end.

    CALIBRATED PER AUTHOR, which is the whole point. The alarm length is derived from each author's
    OWN measured yield, so the detector is simultaneously sensitive for strong authors and tolerant
    of the weak one whose failures are the expected scientific finding. A single global threshold
    could not be both, and a detector that cries wolf on qwen every run is a detector nobody reads.

    ``per_arm`` maps arm -> ``{"accepted": n, "rejected": n, "consecutive_rejects": k,
    "author": <leg label>}``.
    """
    if not per_arm:
        return HealthCheck("authoring_health", INFO, "no authoring activity yet", {})
    table = {**MEASURED_AUTHORING_YIELD, **(yields or {})}
    alarms, watch, detail_ev = [], [], {}
    for arm, st in sorted(per_arm.items()):
        author = str(st.get("author") or arm)
        p = float(table.get(author, _DEFAULT_YIELD))
        limit = _streak_alarm_length(p)
        streak = int(st.get("consecutive_rejects", 0) or 0)
        acc, rej = int(st.get("accepted", 0) or 0), int(st.get("rejected", 0) or 0)
        detail_ev[arm] = {"author": author, "expected_yield": p, "streak": streak,
                          "alarm_at": limit, "accepted": acc, "rejected": rej}
        if streak >= limit:
            alarms.append(f"{arm} ({author}): {streak} consecutive rejections, and this author "
                          f"normally succeeds {p:.0%} of the time (alarm at {limit})")
        elif acc + rej >= 8 and acc / max(acc + rej, 1) < p / 2:
            watch.append(f"{arm}: accepting {acc}/{acc + rej} vs an expected ~{p:.0%}")
    if alarms:
        return HealthCheck("authoring_health", CRITICAL,
                           "AUTHORING IS FAILING SYSTEMATICALLY — " + "; ".join(alarms[:3])
                           + ". This burns the arm's whole candidate budget for nothing; check the "
                             "prompt, the key and the served model before more spend",
                           {"alarms": alarms, "arms": detail_ev})
    if watch:
        return HealthCheck("authoring_health", WARN,
                           "authoring below the author's measured rate — " + "; ".join(watch[:3]),
                           {"watch": watch, "arms": detail_ev})
    return HealthCheck("authoring_health", OK,
                       f"authoring healthy across {len(per_arm)} arm(s) against each author's "
                       "measured yield", {"arms": detail_ev})


def check_unreadable_records(n_unreadable: int, n_total: int,
                             examples: list[str] | None = None) -> HealthCheck:
    """Torn / unparseable records — the failure EVERY reader currently swallows in silence.

    `integrity.env_label_census`, `first_seed_sanity` and the analysis loaders all skip a record they
    cannot parse, and each skip is individually correct (one bad file must not blind a whole gate).
    But nothing COUNTS them, so a rising tide of corruption — a truncating transport, a full disk, a
    writer killed mid-commit — is invisible while every other indicator stays green. The archive
    integrity seal catches it, but that runs post-hoc, i.e. after the compute is spent.

    Any unreadable record is worth surfacing: the archive is written with an atomic
    write-then-rename precisely so a record is either whole or absent, so a TORN one means that
    invariant was violated and the cause will not fix itself.
    """
    if n_total <= 0:
        return HealthCheck("unreadable_records", INFO, "no records to read yet", {})
    if n_unreadable <= 0:
        return HealthCheck("unreadable_records", OK,
                           f"all {n_total} records parsed cleanly", {"n_total": n_total})
    frac = n_unreadable / n_total
    sev = CRITICAL if (frac > 0.01 or n_unreadable >= 5) else WARN
    return HealthCheck("unreadable_records", sev,
                       f"{n_unreadable}/{n_total} archived records are UNREADABLE ({frac:.1%}) — "
                       "records are written atomically, so a torn one means that invariant broke "
                       "(truncating transport, full disk, or a writer killed mid-commit). Every "
                       "reader silently SKIPS these, so the count is the only symptom",
                       {"n_unreadable": n_unreadable, "n_total": n_total,
                        "examples": (examples or [])[:5]})


def check_arm_progress_symmetry(per_arm: dict[str, dict[str, Any]], *,
                                stale_factor: float = 4.0,
                                min_arms: int = 3) -> HealthCheck:
    """One DEAD arm among many healthy ones — masked by every global rate in the system.

    `sentinel.check_completion_stall` watches the campaign's OVERALL cadence, so with nine arms and
    ten legs running, a single arm whose driver died, whose key expired, or whose chain hung
    contributes ~1/19th of the flow and the global rate still looks fine. That arm can be silent for
    a day before anything notices, and its seeds are simply missing at the end.

    DIFFERENTIAL rather than threshold-based, deliberately: the absolute completion rate swings with
    granted capacity, so any fixed "expected records per hour" would be wrong all the time. Judging
    each arm against its SIBLINGS is robust to that — they share the same cluster, the same hour and
    the same scheduler.

    Alarms only on the conjunction of BEHIND and SILENT: an arm that has produced fewer records than
    its peers AND has gone much staler than them. An arm that is merely FINISHED is behind on
    nothing and is correctly ignored.

    ``per_arm`` maps arm -> ``{"n_records": int, "hours_since_last": float}``.

    SERIAL-CHAIN ARMS ARE EXCLUDED (2026-07-28). ``bayes_opt``/``tpe``/``cma_es`` advance ONE
    training per chain step (~8.5 h at 1 thread), while the LLM legs fan out five candidates per
    generation and emit a record every ~0.2 h. Judged against those siblings a chain arm is
    permanently "behind AND silent" — by construction, not by fault — which is exactly what fired
    live: ``ARM STALLED: cma_es (1 records, silent 4.0h vs peers ~0.2h); tpe (1 records, silent
    4.2h)`` while both were verifiably running with fresh heartbeats and live jobs. The differential
    premise ("siblings share the same hour and scheduler") holds only among arms with the same
    PARALLELISM, so the chains are judged instead by :func:`check_chain_progress`, whose 14 h/28 h
    thresholds are anchored on that measured step cost. Excluding them here removes a duplicate,
    mis-calibrated alarm rather than removing coverage. Names come from ``lanes.SERIAL_CHAIN_STEPS``
    so the two checks cannot disagree about which arms are chains.
    """
    from src.cluster.lanes import SERIAL_CHAIN_STEPS

    per_arm = {a: s for a, s in per_arm.items() if a not in SERIAL_CHAIN_STEPS}
    active = {a: s for a, s in per_arm.items() if int(s.get("n_records", 0) or 0) >= 0}
    if len(active) < min_arms:
        return HealthCheck("arm_progress_symmetry", INFO,
                           f"need >={min_arms} arms to compare, have {len(active)}", {})
    ages = sorted(float(s.get("hours_since_last", 0.0) or 0.0) for s in active.values())
    counts = sorted(int(s.get("n_records", 0) or 0) for s in active.values())
    mid = len(ages) // 2
    median_age = ages[mid] if len(ages) % 2 else 0.5 * (ages[mid - 1] + ages[mid])
    median_n = counts[mid] if len(counts) % 2 else 0.5 * (counts[mid - 1] + counts[mid])
    threshold = max(stale_factor * max(median_age, 0.25), 2.0)

    stalled = []
    for arm, s in sorted(active.items()):
        age = float(s.get("hours_since_last", 0.0) or 0.0)
        n = int(s.get("n_records", 0) or 0)
        if age >= threshold and n < median_n:      # BEHIND *and* SILENT - never merely finished
            stalled.append(f"{arm} ({n} records, silent {age:.1f}h vs peers ~{median_age:.1f}h)")
    if stalled:
        return HealthCheck("arm_progress_symmetry", CRITICAL,
                           "ARM STALLED while its siblings advance: " + "; ".join(stalled[:4])
                           + " — a global completion rate cannot see this, and its seeds will "
                             "simply be missing at the end",
                           {"stalled": stalled, "median_age_h": median_age,
                            "median_records": median_n})
    return HealthCheck("arm_progress_symmetry", OK,
                       f"all {len(active)} arms progressing together "
                       f"(median idle {median_age:.1f}h)",
                       {"n_arms": len(active), "median_age_h": median_age})


def check_record_sanity(summary: dict[str, Any] | None) -> HealthCheck:
    """LIVE per-record garbage detection — the shortest path from a bad record to a phone alert.

    Consumes :func:`scripts.first_seed_sanity.assess_recent`. The campaign's first STATISTICAL
    result arrives at the 30-seed floor about two days in, but the ways a run goes wrong are visible
    on the first completed record: a reward that crashed every step so the agent trained on the
    neutral fallback, NaN returns, a policy parked in cash emitting a flat line, an absurd return
    magnitude. Checking every new record each poll turns "two days" into "one poll interval".

    EFFECT-BLIND, and that is what makes it safe to run mid-campaign: it reads no performance value
    and compares no arms, so it cannot preview the result and cannot contaminate the single
    pre-registered confirmatory look. (``tests/test_first_seed_sanity.py`` proves this by running the
    assessment over two archives that differ ONLY in which arm wins and demanding identical output.)

    CRITICAL on garbage: a record the agent trained on a constant signal is not a slow result, it is
    a void one, and every hour spent producing more of them is wasted.
    """
    if not summary:
        return HealthCheck("record_sanity", INFO, "no records assessed yet", {})
    n = int(summary.get("n_assessed", 0) or 0)
    bad = list(summary.get("garbage") or [])
    sus = list(summary.get("suspect") or [])
    if not n:
        return HealthCheck("record_sanity", INFO,
                           str(summary.get("note", "no records yet")), {})
    def _who(rows: list[dict[str, Any]]) -> str:
        """Name the offending records by their ACTUAL ``run_id``.

        This used to synthesize ``f"{arm}-s{seed}"``, which was wrong twice over (found live
        2026-07-28 07:00Z): every search record of an arm shares one seed, so four DISTINCT
        candidates all rendered as the single string ``scalar-s0`` — indistinguishable — and that
        shape is the TEST-leg naming convention, so a search-leg warning read as a scored-leg one.
        Diagnosing four identical labels cost real time; the run_id is right there in the row.
        """
        return ", ".join(str(r.get("run_id") or f"{r.get('arm')}-s{r.get('seed')}") for r in rows[:4])

    if bad:
        who = _who(bad)
        why = (bad[0].get("reasons") or ["?"])[0]
        return HealthCheck("record_sanity", CRITICAL,
                           f"{len(bad)}/{n} recent record(s) are GARBAGE ({who}) — e.g. {why}. "
                           "These are void, not slow: stop and diagnose before more compute is "
                           "spent producing them", {"garbage": bad[:8], "n_assessed": n})
    if sus:
        who = _who(sus)
        return HealthCheck("record_sanity", WARN,
                           f"{len(sus)}/{n} recent record(s) look SUSPECT ({who}) — partial "
                           "fallback contamination or missing execution counters",
                           {"suspect": sus[:8], "n_assessed": n})
    return HealthCheck("record_sanity", OK,
                       f"all {n} most-recent records pass the execution-sanity checks",
                       {"n_assessed": n})


def check_admin_kill(verdict: dict[str, Any] | None) -> HealthCheck:
    """Surface ``killswitch.classify_task_deaths`` to the OPERATOR — the last link in that chain.

    The classifier had no production call site until the sentinel began computing a verdict from the
    epilogue rows it already reads; but a verdict written into the inputs dict and read by no check
    never reaches the report, the severity aggregation, or the phone alert — the same
    built-but-unwired failure one level up. This is the consumer.

    Severity mirrors what the event MEANS for the standing priority that keeping Myriad access
    outranks throughput: an ``admin_kill`` is CRITICAL because the correct response is to RETREAT
    (stop submitting, do not requeue) rather than to fight the scheduler, and that decision is
    time-critical. ``node_failure``/``walltime`` are ordinary campaign weather — reported, never
    alarming.

    Detection only, by deliberate design: this never writes the incident file. Writing one blocks ALL
    submission until a human clears it, i.e. it can halt a 23-day campaign — an operator decision,
    not something a read-only watcher may take on its own.
    """
    if not verdict:
        return HealthCheck("admin_kill", INFO, "no task-death rows to classify", {})
    kind = str(verdict.get("classification", "ok"))
    action = str(verdict.get("action", "continue"))
    n_deaths = int(verdict.get("n_deaths", 0) or 0)
    n_hosts = int(verdict.get("n_hosts", 0) or 0)
    n_undated = int(verdict.get("n_undated", 0) or 0)
    ev = {**verdict, "enforced": False}
    if kind == "admin_kill":
        return HealthCheck("admin_kill", CRITICAL,
                           f"ADMINISTRATIVE KILL suspected: {n_deaths} deaths across {n_hosts} hosts "
                           f"({verdict.get('reason', '')}) — RETREAT: stop submitting and do NOT "
                           "requeue. Access preservation outranks throughput; no incident file was "
                           "written, so resuming stays a human decision", ev)
    if kind in ("node_failure", "walltime"):
        return HealthCheck("admin_kill", INFO,
                           f"{n_deaths} task death(s) classified {kind} (action: {action}) — "
                           "ordinary campaign weather, not an administrative kill", ev)
    if n_undated:
        return HealthCheck("admin_kill", WARN,
                           f"{n_undated} death row(s) carry no usable timestamp — the burst window "
                           "cannot see them, so an administrative kill could go undetected; check "
                           "that the epilogue trap is stamping `ts`", ev)
    return HealthCheck("admin_kill", OK,
                       f"no administrative-kill signature ({n_deaths} deaths, {n_hosts} hosts)", ev)


def check_capacity_accumulation(report: dict[str, Any], *, expected_cores: int,
                                hours_in: float) -> HealthCheck:
    """Is concurrency climbing toward the forecast, or stuck at the probe-measured floor?

    ``report`` is :func:`src.cluster.telemetry.accumulation_report` output. The grace period matters:
    accumulation is EXPECTED to take ~2-3 h (dispatch ~3.3 jobs/min against ~8.5 h jobs), so a low
    reading in the first hours is normal and must not cry wolf — but a plateau far below forecast
    AFTER that window is a real, plan-changing finding.
    """
    status = report.get("status")
    if status in ("no-data", "insufficient"):
        return HealthCheck("capacity_accumulation", INFO,
                           f"not yet measurable ({status}) — keep the telemetry watcher running",
                           {"report": report})
    late = float(report.get("late_mean_cores") or 0)
    if expected_cores <= 0:
        # No DECLARED forecast to hold ourselves to (GO never wrote one). Report the measurement and
        # stop: judging a plateau against a zero forecast would fire a WARN on every healthy run.
        return HealthCheck("capacity_accumulation", INFO,
                           f"holding ~{late:.0f} cores ({status}); no forecast recorded in "
                           "allocation_state (lane_expected_cores) to compare against",
                           {"report": report})
    frac = late / expected_cores

    if status == "climbing":
        return HealthCheck("capacity_accumulation", OK,
                           f"still ACCUMULATING ({late:.0f} cores and rising) — do NOT re-forecast "
                           "the rung yet", {"report": report, "frac_of_forecast": round(frac, 2)})
    if status == "declining":
        return HealthCheck("capacity_accumulation", WARN,
                           f"concurrency DECLINING to ~{late:.0f} cores — check for a kill event, a "
                           "drained queue, or a cluster-wide load spike",
                           {"report": report, "frac_of_forecast": round(frac, 2)})
    # plateaued
    if hours_in < 3.0:
        return HealthCheck("capacity_accumulation", INFO,
                           f"plateaued at ~{late:.0f} cores but only {hours_in:.1f} h in — "
                           "accumulation needs ~2-3 h; too early to conclude",
                           {"report": report, "frac_of_forecast": round(frac, 2)})
    if frac < 0.5:
        return HealthCheck("capacity_accumulation", WARN,
                           f"plateaued at ~{late:.0f} cores = {frac:.0%} of the {expected_cores} "
                           "forecast — RE-FORECAST the reachable rung from this number and plan "
                           "against it (the forecast was a model, this is the measurement)",
                           {"report": report, "frac_of_forecast": round(frac, 2)})
    return HealthCheck("capacity_accumulation", OK,
                       f"plateaued at ~{late:.0f} cores ({frac:.0%} of forecast) — steady state",
                       {"report": report, "frac_of_forecast": round(frac, 2)})


def check_chain_progress(chains: dict[str, dict[str, Any]], *,
                         stall_warn_h: float = 14.0,
                         stall_crit_h: float = 28.0) -> HealthCheck:
    """The CRITICAL PATH: are the serial search chains advancing?

    ``chains`` maps arm -> ``{"completed": int, "total": int, "hours_since_last": float}``.

    Thresholds are anchored on the measured step cost, not guessed: one chain step is ~8.5 h at
    1 thread and ~3.1 h at the ratified 8 threads, so ~14 h without progress is already several
    missed steps (WARN) and ~28 h is unambiguously stuck (CRITICAL). A stalled chain is the
    campaign's worst silent failure: the test flood keeps streaming, every other indicator stays
    green, and the makespan floor quietly slips a day for every day the chain sits still.
    """
    if not chains:
        return HealthCheck("chain_progress", INFO, "no serial chains reported yet", {})
    stalled, done, worst_h = [], [], 0.0
    for arm, st in chains.items():
        completed, total = int(st.get("completed", 0)), int(st.get("total", 0))
        idle = float(st.get("hours_since_last") or 0.0)
        if total and completed >= total:
            done.append(arm)
            continue
        worst_h = max(worst_h, idle)
        if idle >= stall_warn_h:
            stalled.append(f"{arm} {completed}/{total} idle {idle:.1f}h")
    if not stalled:
        prog = ", ".join(f"{a} {s.get('completed')}/{s.get('total')}" for a, s in chains.items())
        return HealthCheck("chain_progress", OK, f"critical path advancing ({prog})",
                           {"chains": chains, "complete": done})
    sev = CRITICAL if worst_h >= stall_crit_h else WARN
    return HealthCheck("chain_progress", sev,
                       "CRITICAL-PATH CHAIN STALLED: " + "; ".join(stalled) +
                       " — the makespan floor slips a day for every day this sits still; check the "
                       "chain job, not the test flood",
                       {"chains": chains, "stalled": stalled, "max_idle_h": worst_h})


def check_host_failure_concentration(failures_by_host: dict[str, int],
                                     attempts_by_host: dict[str, int], *,
                                     min_attempts: int = 5,
                                     bad_host_rate: float = 0.5) -> HealthCheck:
    """Is ONE node eating tasks? (measured for real: node-d00a-230 had no apptainer -> rc=127)

    A global failure rate hides this completely: 40 dead tasks on one bad host among 4,000 good
    ones is a 1% rate that every aggregate check passes, while that host keeps accepting and
    killing work for the rest of the run. Concentration is the signal, not volume.
    """
    if not attempts_by_host:
        return HealthCheck("host_failure_concentration", INFO, "no per-host attempts recorded", {})
    bad = {}
    for host, att in attempts_by_host.items():
        if att < min_attempts:
            continue
        rate = failures_by_host.get(host, 0) / att
        if rate >= bad_host_rate:
            bad[host] = {"failed": failures_by_host.get(host, 0), "attempts": att,
                         "rate": round(rate, 2)}
    if not bad:
        tot_f, tot_a = sum(failures_by_host.values()), sum(attempts_by_host.values())
        return HealthCheck("host_failure_concentration", OK,
                           f"no bad node ({tot_f}/{tot_a} failures spread across "
                           f"{len(attempts_by_host)} hosts)",
                           {"n_hosts": len(attempts_by_host)})
    return HealthCheck("host_failure_concentration", WARN,
                       f"{len(bad)} node(s) failing >= {bad_host_rate:.0%} of their tasks: "
                       + ", ".join(f"{h} ({v['failed']}/{v['attempts']})" for h, v in bad.items())
                       + " — exclude them (`-l h=!<host>`) or the run keeps feeding them work",
                       {"bad_hosts": bad})


def check_rung_forecast(*, completed_trainings: int, elapsed_hours: float,
                        hours_remaining: float, rung_targets: dict[int, int]) -> HealthCheck:
    """Which seed rung do we actually reach by the exogenous stop, at the OBSERVED rate?

    The stop is exogenous (calendar, never the effect), so this is a planning readout, not a
    decision rule — but planning against a stale model instead of the live rate is how a campaign
    discovers on the last day that it banked a lower rung than the write-up assumed.
    """
    if elapsed_hours <= 0 or completed_trainings <= 0:
        return HealthCheck("rung_forecast", INFO, "no completions yet", {})
    rate = completed_trainings / elapsed_hours
    projected = completed_trainings + rate * hours_remaining
    reachable = [n for n, need in sorted(rung_targets.items()) if need <= projected]
    top = max(reachable) if reachable else 0
    nxt = min((n for n in sorted(rung_targets) if n > top), default=None)
    detail = (f"{completed_trainings:,} done at {rate:.1f}/h -> ~{projected:,.0f} by the stop "
              f"=> rung {top}")
    if nxt is not None:
        short = rung_targets[nxt] - projected
        detail += f" (next rung {nxt} needs {short:,.0f} more)"
    sev = OK if top >= max(rung_targets, default=0) else INFO
    return HealthCheck("rung_forecast", sev, detail,
                       {"rate_per_hour": round(rate, 2), "projected": round(projected),
                        "reachable_rung": top, "next_rung": nxt})


def check_winner_execution_quality(winners: dict[str, dict[str, Any]], *,
                                   severe: float = 0.10, warn: float = 0.001) -> HealthCheck:
    """Did a FALLBACK-CONTAMINATED candidate get frozen as a winner and promoted to the scored leg?

    ``train_safe_default_count`` records the steps on which the authored reward RAISED and the
    neutral 0.0 fallback stood in (R66). It is archived, and summed into the integrity report — but
    it does **not** gate selection: the winner is ``max(val_fitness)``. So a candidate whose reward
    executed on half its steps can win, and the sealed test leg then RE-TRAINS that same reward and
    inherits the same contamination.

    CENSUS, re-derived 2026-07-28 over the FULL RUN 1 archive (613 records carrying the counter,
    superseding an earlier 136-candidate snapshot that reported only two severe cases): **594 clean,
    16 trace (<1 %), 3 SEVERE** — ``qwen3.6-27b/scalar-g1-c4`` 53.66 % (214,649/400,000),
    ``qwen3.5-9b/distributional-g1-c2`` 50.02 %, ``glm-5.2/placebo_shuffled-g0-c0`` 39.40 %. All
    three are open-weight legs, and **zero frozen winners carried any fallback**. Note the RUN 1
    search was invalidated by the cross-line reject collision, but THIS measurement is unaffected:
    it is a per-training execution statistic of a candidate that actually ran, and the collision
    only decided which candidates were allowed to run.

    ``severe=0.10`` is deliberately threshold-INSENSITIVE rather than tuned. The observed
    distribution is strongly bimodal — the worst trace case is 0.41 % and the mildest severe case is
    39.40 %, a **96x empty gap** — so any threshold between ~1 % and ~35 % partitions the data
    identically. The verdict therefore does not depend on where in that gap the line is drawn, which
    is the property that makes it defensible.

    The frozen pre-registration is SILENT on an inclusion threshold, and adding one post-launch
    would move the hash-bound canonical design — a far larger cost than the gap it closes. So this
    check does not gate anything; it makes the event VISIBLE at CRITICAL within one poll, while the
    unit can still be re-run under the dated, effect-blind remediation rule pre-committed in
    ``DECISIONS.md`` (2026-07-28, before any RUN 2 winner existed). That is the difference between a
    disclosed limitation and a result nobody can defend.

    ``winners`` maps a label -> ``{"candidate_id": str, "safe_default_count": int,
    "safe_call_count": int}``.
    """
    rows = []
    for label, w in sorted((winners or {}).items()):
        calls = int(w.get("safe_call_count") or 0)
        if calls <= 0:
            continue                      # no counters -> says nothing; never guessed at
        frac = int(w.get("safe_default_count") or 0) / calls
        rows.append((frac, label, str(w.get("candidate_id") or "?")))
    if not rows:
        return HealthCheck("winner_execution_quality", INFO,
                           "no frozen winner carries execution counters yet", {})
    rows.sort(reverse=True)
    worst_frac, worst_label, worst_cid = rows[0]
    ev = {"n_winners": len(rows),
          "worst": {"leg": worst_label, "candidate_id": worst_cid, "fallback_frac": round(worst_frac, 6)}}
    if worst_frac >= severe:
        return HealthCheck("winner_execution_quality", CRITICAL,
                           f"a CONTAMINATED candidate was frozen as a winner: {worst_label} "
                           f"{worst_cid} fell back on {worst_frac:.1%} of training steps. The scored "
                           "leg will re-train this reward and inherit the contamination — re-run "
                           "that unit or disclose it; do not let it enter the confirmatory result "
                           "unexamined", ev)
    if worst_frac >= warn:
        return HealthCheck("winner_execution_quality", WARN,
                           f"frozen winner {worst_label} {worst_cid} fell back on "
                           f"{worst_frac:.2%} of steps (minor, but it enters the scored leg)", ev)
    return HealthCheck("winner_execution_quality", OK,
                       f"all {len(rows)} frozen winner(s) executed cleanly "
                       f"(worst fallback {worst_frac:.4%})", ev)


def check_substrate_fields(census: dict[str, int], *, scope: str = "test leg") -> HealthCheck:
    """Is the SCORED leg on ONE CPU model and ONE thread regime? The device label cannot answer this.

    Consumes :func:`src.cluster.integrity.substrate_field_census`. ``check_determinism_homogeneity``
    keys on the env LABEL, which carries only ``dev=cpu`` — so an Intel/AMD or Skylake/Cascade-Lake
    mix, or a 1-vs-8 thread mix, passes it silently. Those are precisely the differences that change
    float reduction order and therefore break the CRN pairing every paired contrast rests on.

    Live evidence this is needed (2026-07-28): the SEARCH leg was found holding 116 records on a
    Xeon Gold 6240 and 1 on a Gold 6140, so ``-ac allow=d`` does NOT pin a single CPU model.

    CRITICAL rather than WARN, on the same reasoning as the device check: a substrate mix inside the
    scored leg is a VALIDITY failure, and it is only fixable while there is still time to re-run.
    """
    counts = Counter({str(k): int(v) for k, v in (census or {}).items() if int(v) > 0})
    total = sum(counts.values())
    if not total:
        return HealthCheck("substrate_fields", INFO, "no env.json substrate data yet", {})
    if len(counts) == 1:
        only = next(iter(counts))
        sev = WARN if only == "<no env.json>" else OK
        detail = (f"{total} scored record(s) carry NO env.json — substrate unverifiable"
                  if only == "<no env.json>"
                  else f"{scope} on ONE substrate across {total} records ({only})")
        return HealthCheck("substrate_fields", sev, detail, {"n_records": total})
    top = counts.most_common()
    return HealthCheck("substrate_fields", CRITICAL,
                       f"{scope} spans {len(counts)} SUBSTRATES — CPU model / thread regime differ, "
                       "so float reduction order is not identical and CRN pairing is confounded: "
                       + " || ".join(f"{sig} x{n}" for sig, n in top[:3]),
                       {"distinct": len(counts), "counts": dict(top[:6])})


def check_determinism_homogeneity(env_label_census: dict[str, int], *,
                                  scope: str = "test leg") -> HealthCheck:
    """Are all SCORED trainings on one substrate? CPU/CUDA and 1/8-thread are NOT bit-identical.

    Takes the census produced by :func:`src.cluster.integrity.env_label_census` (``{label: count}``)
    — the SAME normalisation the post-hoc S6 gate uses, so the live verdict and the gate verdict can
    never disagree.

    Every paired contrast rests on CRN pairing, so a device or thread-regime MIX inside the scored
    leg silently confounds the arm effect with a substrate effect. The env fingerprint now carries
    both (``|dev=`` from ``run_one``; ``OMP_NUM_THREADS`` + ``torch.get_num_threads()`` from
    ``capture_env``), which makes this checkable DURING the run — the post-hoc S6 audit finds it far
    too late to re-run anything. CRITICAL, not WARN: this is a validity failure, not a slowdown.
    """
    counts = Counter({str(k): int(v) for k, v in (env_label_census or {}).items() if int(v) > 0})
    total = sum(counts.values())
    if not total:
        return HealthCheck("determinism_homogeneity", INFO, "no env fingerprints yet", {})
    if len(counts) == 1:
        return HealthCheck("determinism_homogeneity", OK,
                           f"{scope} homogeneous across {total} records "
                           f"({next(iter(counts))})", {"n_records": total})
    capture_failed = [c for c in counts if c.startswith("capture-failed")]
    top = counts.most_common()
    return HealthCheck("determinism_homogeneity", CRITICAL,
                       f"{scope} is NOT substrate-homogeneous: {len(counts)} distinct environments "
                       + ", ".join(f"{lbl}x{n}" for lbl, n in top[:4])
                       + " — CPU/CUDA and 1/8-thread are not bit-identical, so a mix confounds "
                       "every paired contrast. Quarantine the minority substrate and re-run it."
                       + (f" ({len(capture_failed)} label(s) are capture FAILURES, not a real mix)"
                          if capture_failed else ""),
                       {"distinct": len(counts), "counts": dict(top[:8])})
