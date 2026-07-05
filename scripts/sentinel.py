#!/usr/bin/env python3
"""Campaign SENTINEL — a continuous, read-only invariant health-monitor for the unattended run.

The dashboard (``scripts/monitor.py``) shows you what is HAPPENING; the sentinel decides whether what
is happening is HEALTHY, and it is designed to catch **anything even slightly off, as early as
possible** — the whole point being that nothing about a wrong result surfaces only at the end
(2026-07-05, world-class-monitoring requirement).

Design.
* **Pure check functions** (``check_*``) take already-parsed inputs and return a :class:`HealthCheck`
  with a severity + human detail + a machine ``evidence`` dict. They are total (never raise on bad
  input) and unit-tested WITHOUT disk or a live run.
* A **gatherer** (:func:`gather_inputs`) reads the on-disk campaign artifacts + system telemetry
  READ-ONLY (records, ``anomalies.jsonl``, ``progress.json``, ``campaign_summary.json``, disk/RAM/GPU),
  each probe guarded so one failure degrades a single check to ``UNKNOWN`` rather than killing the
  sentinel.
* :func:`evaluate_health` runs every check and aggregates the worst severity into a
  :class:`HealthReport`. The CLI (:func:`main`) prints it (``--once``) or watches on an interval
  (``--watch``), emits every status TRANSITION to the structured event log (``events.jsonl``) so the
  precise log carries the full health history, and exits non-zero on a CRITICAL so a supervisor/cron
  can act.

Severity ladder: ``OK`` < ``INFO`` < ``WARN`` < ``CRITICAL`` (and ``UNKNOWN`` for an unreadable probe,
ranked with WARN — an invariant we cannot check is not silently "healthy"). READ-ONLY: the sentinel
never touches the run; it only observes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Severity model                                                              #
# --------------------------------------------------------------------------- #
OK = "OK"
INFO = "INFO"
WARN = "WARN"
CRITICAL = "CRITICAL"
UNKNOWN = "UNKNOWN"

#: Rank for aggregation + exit-code. UNKNOWN ranks with WARN (an uncheckable invariant is NOT healthy).
_RANK: dict[str, int] = {OK: 0, INFO: 1, UNKNOWN: 2, WARN: 2, CRITICAL: 3}


def worst(severities: list[str]) -> str:
    """The highest-rank severity in the list (``OK`` if empty)."""
    return max(severities, key=lambda s: _RANK.get(s, 0)) if severities else OK


@dataclass
class HealthCheck:
    """One invariant's verdict: a name, a severity, a human detail, and machine evidence."""

    name: str
    severity: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "severity": self.severity, "detail": self.detail,
                "evidence": self.evidence}


@dataclass
class HealthReport:
    """The aggregate: every check + the worst severity + the CRITICAL/WARN shortlists."""

    checks: list[HealthCheck]

    @property
    def severity(self) -> str:
        return worst([c.severity for c in self.checks])

    @property
    def healthy(self) -> bool:
        return self.severity in (OK, INFO)

    def by_severity(self, sev: str) -> list[HealthCheck]:
        return [c for c in self.checks if c.severity == sev]

    def as_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "healthy": self.healthy,
                "checks": [c.as_dict() for c in self.checks]}


# --------------------------------------------------------------------------- #
# PURE CHECK FUNCTIONS (unit-tested without disk or a live run)               #
# --------------------------------------------------------------------------- #
def check_disk(free_gb: float | None, *, floor_gb: float = 20.0, warn_gb: float = 30.0) -> HealthCheck:
    """The run drive: the pagefile + Windows Update live on C:, and a full drive kills the run in ways
    the stall detector only reports after the fact."""
    if free_gb is None:
        return HealthCheck("disk", UNKNOWN, "could not read free disk space")
    if free_gb < floor_gb:
        return HealthCheck("disk", CRITICAL, f"free disk {free_gb:.1f} GB < floor {floor_gb} GB",
                           {"free_gb": free_gb, "floor_gb": floor_gb})
    if free_gb < warn_gb:
        return HealthCheck("disk", WARN, f"free disk {free_gb:.1f} GB approaching floor {floor_gb} GB",
                           {"free_gb": free_gb})
    return HealthCheck("disk", OK, f"free disk {free_gb:.1f} GB", {"free_gb": free_gb})


def check_ram(used_pct: float | None, *, warn: float = 88.0, crit: float = 95.0) -> HealthCheck:
    """RAM pressure — the transition-wave OOM (why n_gpu is capped at 3) shows here first."""
    if used_pct is None:
        return HealthCheck("ram", UNKNOWN, "could not read RAM usage")
    if used_pct >= crit:
        return HealthCheck("ram", CRITICAL, f"RAM {used_pct:.0f}% used (OOM risk)", {"used_pct": used_pct})
    if used_pct >= warn:
        return HealthCheck("ram", WARN, f"RAM {used_pct:.0f}% used", {"used_pct": used_pct})
    return HealthCheck("ram", OK, f"RAM {used_pct:.0f}% used", {"used_pct": used_pct})


def check_gpu_temp(temp_c: float | None, *, warn: float = 84.0, crit: float = 90.0) -> HealthCheck:
    """Sustained thermal throttling collapses fps over a multi-day run; the governor pauses at the
    hardware limit, but a rising trend is an early warning."""
    if temp_c is None:
        return HealthCheck("gpu_temp", UNKNOWN, "no GPU temperature telemetry")
    if temp_c >= crit:
        return HealthCheck("gpu_temp", CRITICAL, f"GPU {temp_c:.0f}C (throttling/limit)", {"temp_c": temp_c})
    if temp_c >= warn:
        return HealthCheck("gpu_temp", WARN, f"GPU {temp_c:.0f}C (warm)", {"temp_c": temp_c})
    return HealthCheck("gpu_temp", OK, f"GPU {temp_c:.0f}C", {"temp_c": temp_c})


def check_silent_hang(mtime_age_s: float | None, *, warn_s: float = 1200.0, crit_s: float = 3600.0,
                      terminal: bool = False) -> HealthCheck:
    """No fresh write in a long time = a silent hang (a wedged CUDA candidate, a deadlock). A terminal
    (done/error) run is intentionally quiet, so it is never 'hung'."""
    if terminal:
        return HealthCheck("silent_hang", OK, "run reached a terminal phase (quiet by design)")
    if mtime_age_s is None:
        return HealthCheck("silent_hang", UNKNOWN, "no progress artifact to age-check")
    if mtime_age_s >= crit_s:
        return HealthCheck("silent_hang", CRITICAL,
                           f"no progress write for {mtime_age_s/60:.0f} min (silent hang)",
                           {"age_s": mtime_age_s})
    if mtime_age_s >= warn_s:
        return HealthCheck("silent_hang", WARN, f"no progress write for {mtime_age_s/60:.0f} min",
                           {"age_s": mtime_age_s})
    return HealthCheck("silent_hang", OK, f"last progress write {mtime_age_s/60:.1f} min ago",
                       {"age_s": mtime_age_s})


def check_gate_failure_rate(n_failed: int, n_attempted: int, *, warn: float = 0.10,
                            crit: float = 0.40) -> HealthCheck:
    """LLM gate/validation failures burn a budget slot; the prototype ran ~1/40 (~2.5%). A rising rate
    is an early sign of a systemic authoring fault (a prompt regression, an API degradation)."""
    if n_attempted <= 0:
        return HealthCheck("gate_failures", INFO, "no candidates attempted yet")
    rate = n_failed / n_attempted
    ev = {"n_failed": n_failed, "n_attempted": n_attempted, "rate": rate}
    if rate >= crit:
        return HealthCheck("gate_failures", CRITICAL,
                           f"{rate:.0%} of candidates failed the gate ({n_failed}/{n_attempted})", ev)
    if rate >= warn:
        return HealthCheck("gate_failures", WARN,
                           f"{rate:.0%} gate-failure rate ({n_failed}/{n_attempted})", ev)
    return HealthCheck("gate_failures", OK, f"{rate:.1%} gate-failure rate", ev)


def check_nan_rate(n_nonfinite: int, n_records: int, *, warn: float = 0.02,
                   crit: float = 0.10) -> HealthCheck:
    """Any non-finite fitness/score in the archive is corruption that would only bite at analysis time —
    the exact 'surfaces at the end' failure the sentinel exists to pre-empt."""
    if n_records <= 0:
        return HealthCheck("nan_rate", INFO, "no records yet")
    rate = n_nonfinite / n_records
    ev = {"n_nonfinite": n_nonfinite, "n_records": n_records, "rate": rate}
    if rate >= crit:
        return HealthCheck("nan_rate", CRITICAL, f"{rate:.0%} of records carry a non-finite score", ev)
    if n_nonfinite > 0 or rate >= warn:
        return HealthCheck("nan_rate", WARN, f"{n_nonfinite}/{n_records} records non-finite", ev)
    return HealthCheck("nan_rate", OK, "all records finite", ev)


def check_divergence_rate(n_diverged_runs: int, n_candidates: int, *, warn: float = 0.05,
                          crit: float = 0.20, winner_diverged: bool = False) -> HealthCheck:
    """Critic-explosion clustering. PopArt makes these rare; a rising diverged-RUN rate flags a
    genuinely mis-scaled reward regime, and a DIVERGED WINNER is critical (a diverged candidate should
    lose selection, so a winner that diverged is a selection-integrity alarm)."""
    if winner_diverged:
        return HealthCheck("divergence", CRITICAL, "a FROZEN WINNER's training diverged (selection integrity)",
                           {"winner_diverged": True})
    if n_candidates <= 0:
        return HealthCheck("divergence", INFO, "no candidates yet")
    rate = n_diverged_runs / n_candidates
    ev = {"n_diverged_runs": n_diverged_runs, "n_candidates": n_candidates, "rate": rate}
    if rate >= crit:
        return HealthCheck("divergence", CRITICAL, f"{rate:.0%} of candidates diverged (critic explosions)", ev)
    if rate >= warn:
        return HealthCheck("divergence", WARN, f"{rate:.0%} divergence rate", ev)
    return HealthCheck("divergence", OK, f"{rate:.1%} divergence rate", ev)


def check_reward_scale_drift(raw_rms_by_arm: dict[str, float], *, ratio_warn: float = 100.0,
                             ratio_crit: float = 1e4) -> HealthCheck:
    """The P5 confound made auditable: the reward SCALE (unclamped PopArt raw_rms) differing wildly
    across arms is a latent scale-driven entropy-regularisation difference that entangles with the
    treatment. A large cross-arm max/min ratio is worth surfacing DURING the run (it re-baselines the
    mechanism story), not at analysis time."""
    vals = {a: float(v) for a, v in raw_rms_by_arm.items() if v is not None and float(v) > 0.0}
    if len(vals) < 2:
        return HealthCheck("reward_scale", INFO, "need >= 2 arms' raw_rms for a drift check",
                           {"n_arms": len(vals)})
    # SPLIT hand-designed BASELINES from AUTHORED arms (2026-07-28). The H1 canon contains
    # ratio-form rewards whose scale is a FIXED property of the formula, not drift: measured live,
    # `baseline_differential_downside_ratio` sits at raw_rms 28,774 and `baseline_differential_sharpe`
    # at 16,324 while every other arm is 0.015-2.33. Pooling them made the check report a 5.0e5x
    # ratio CRITICAL on EVERY poll for the whole campaign — a permanent alarm that can never clear,
    # which is the cry-wolf failure that teaches an operator to ignore the one alert that matters.
    #
    # This is a REFINEMENT, not a suppression, and it moves detection power to where the confound
    # actually lives: the P5 concern is that a reward scale differing across the TREATMENT arms
    # entangles with the manipulated variable. Those arms are the AUTHORED ones. A registered
    # baseline's fixed scale cannot drift and cannot entangle with anything — it is reported as
    # standing context instead. If ANY authored arm drifts, this still fires exactly as before.
    authored = {a: v for a, v in vals.items() if not a.startswith("baseline")}
    base = {a: v for a, v in vals.items() if a.startswith("baseline")}

    def _ratio(d: dict[str, float]) -> float:
        return (max(d.values()) / min(d.values())) if len(d) >= 2 and min(d.values()) > 0 else 1.0

    b_ratio = _ratio(base)
    b_note = (f"; baselines span {b_ratio:.0f}x (registered ratio-form rewards — fixed by formula, "
              "reported as context)") if len(base) >= 2 else ""
    if len(authored) < 2:
        return HealthCheck("reward_scale", INFO,
                           f"need >= 2 AUTHORED arms' raw_rms for a drift check "
                           f"(have {len(authored)}){b_note}",
                           {"n_authored": len(authored), "baseline_ratio": b_ratio, "by_arm": vals})
    hi, lo = max(authored.values()), min(authored.values())
    ratio = hi / lo if lo > 0 else float("inf")
    ev = {"max": hi, "min": lo, "ratio": ratio, "authored": authored,
          "baseline_ratio": b_ratio, "baselines": base}
    if ratio >= ratio_crit:
        return HealthCheck("reward_scale", CRITICAL,
                           f"AUTHORED-arm reward-scale ratio {ratio:.0f}x{b_note}", ev)
    if ratio >= ratio_warn:
        return HealthCheck("reward_scale", WARN,
                           f"AUTHORED-arm reward-scale ratio {ratio:.0f}x{b_note}", ev)
    return HealthCheck("reward_scale", OK,
                       f"AUTHORED-arm reward-scale ratio {ratio:.1f}x{b_note}", ev)


def check_api_error_rate(n_api_errors: int, n_api_calls: int, *, warn: float = 0.05,
                         crit: float = 0.25) -> HealthCheck:
    """A rising LLM API error/refusal rate (rate-limit, outage, a systematically-refused prompt) is an
    early sign the authoring loop is degrading — caught in hours, not at the end."""
    if n_api_calls <= 0:
        return HealthCheck("api_errors", INFO, "no API calls logged yet")
    rate = n_api_errors / n_api_calls
    ev = {"n_errors": n_api_errors, "n_calls": n_api_calls, "rate": rate}
    if rate >= crit:
        return HealthCheck("api_errors", CRITICAL, f"{rate:.0%} API error rate ({n_api_errors}/{n_api_calls})", ev)
    if rate >= warn:
        return HealthCheck("api_errors", WARN, f"{rate:.0%} API error rate", ev)
    return HealthCheck("api_errors", OK, f"{rate:.1%} API error rate", ev)


def check_progress(seen_arms: int, expected_arms: int, *, all_arms_tested: bool | None = None) -> HealthCheck:
    """Coverage: are all pre-registered arms progressing? A completed run whose gate says not-all-tested
    is a husk (the exact-M19 failure class), CRITICAL."""
    if all_arms_tested is False:
        return HealthCheck("coverage", CRITICAL,
                           "campaign summary reports NOT all arms tested (husk/failure)",
                           {"all_arms_tested": False})
    ev = {"seen_arms": seen_arms, "expected_arms": expected_arms}
    if expected_arms > 0 and seen_arms < expected_arms:
        return HealthCheck("coverage", INFO, f"{seen_arms}/{expected_arms} arms have produced records", ev)
    return HealthCheck("coverage", OK, f"{seen_arms}/{max(expected_arms, seen_arms)} arms progressing", ev)


def check_exit_code(exit_code: int | None) -> HealthCheck:
    """A recorded FAILED exit (EXIT_INCOMPLETE=3 etc.) from the last pass is a resumable failure worth
    an alert; 0/None = clean or still running."""
    if exit_code is None or int(exit_code) == 0:
        return HealthCheck("exit_code", OK, "no failing exit recorded", {"exit_code": exit_code})
    return HealthCheck("exit_code", CRITICAL, f"last pass exited {exit_code} (resumable failure)",
                       {"exit_code": int(exit_code)})


def cusum(series: list[float], target: float, *, k: float, h: float,
          direction: str = "up") -> tuple[bool, int, float]:
    """One-sided Page (1954) CUSUM change-point detector: does a streaming metric DRIFT off ``target``?

    A hard threshold only fires once a value crosses it; a CUSUM accumulates small persistent
    deviations and alarms on a sustained SHIFT of size ~``k`` (in the series' units) before any single
    value is extreme — the right tool for "catch anything EARLY". For ``direction='up'`` the statistic
    is ``S_i = max(0, S_{i-1} + (x_i - target - k))`` and it alarms when ``S_i > h`` (``down`` mirrors
    it). ``k`` is the allowable slack (half the shift you want to detect); ``h`` is the decision
    interval. Returns ``(alarmed, first_alarm_index, S_final)``. Pure + total (empty series -> no
    alarm)."""
    s = 0.0
    first = -1
    sign = 1.0 if direction == "up" else -1.0
    for i, x in enumerate(series):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        s = max(0.0, s + sign * (xf - target) - k)
        if s > h and first < 0:
            first = i
    return (first >= 0, first, s)


def check_metric_drift(name: str, history: list[float], target: float, *, k: float, h: float,
                       min_points: int = 5, direction: str = "up") -> HealthCheck:
    """Statistical-process-control check: a CUSUM drift alarm on a streaming rate/metric.

    Catches a SUSTAINED creep (e.g. the gate-failure or NaN rate slowly rising, or the training fps
    slowly sinking under thermal creep) that the point-in-time threshold checks would only flag once
    it becomes extreme — so a degrading run is seen in hours, not at the end. WARN (not CRITICAL):
    the drift is an EARLY signal to investigate, and the hard-threshold checks above escalate if it
    actually reaches a breach. Needs ``min_points`` samples before it can alarm (a short history is
    not yet evidence of drift). ``direction='down'`` alarms on a sustained FALL below ``target``."""
    hist = [float(x) for x in history if x is not None]
    if len(hist) < int(min_points):
        return HealthCheck(f"{name}_drift", INFO, f"{len(hist)} samples (need {min_points} for a drift check)",
                           {"n": len(hist)})
    alarmed, idx, s = cusum(hist, target, k=k, h=h, direction=direction)
    word = "upward" if direction == "up" else "downward"
    ev = {"n": len(hist), "cusum": round(s, 4), "target": target, "k": k, "h": h,
          "first_alarm_i": idx, "direction": direction}
    if alarmed:
        return HealthCheck(f"{name}_drift", WARN,
                           f"{name} is DRIFTING {word} (CUSUM {s:.2f} > {h}, since sample {idx})", ev)
    return HealthCheck(f"{name}_drift", OK, f"{name} stable (CUSUM {s:.2f} <= {h})", ev)


def check_mirror_freshness(mirror_age_s: float | None, *, warn_s: float = 43200.0) -> HealthCheck:
    """The irreplaceable archive mirror should refresh (~6-hourly). A stale mirror means a C: failure
    would lose recent runs — worth a nudge, not a stop."""
    if mirror_age_s is None:
        return HealthCheck("mirror", INFO, "no archive mirror found (register the 6-hourly task)")
    if mirror_age_s >= warn_s:
        return HealthCheck("mirror", WARN, f"archive mirror is {mirror_age_s/3600:.0f} h stale",
                           {"age_s": mirror_age_s})
    return HealthCheck("mirror", OK, f"archive mirror {mirror_age_s/3600:.1f} h old", {"age_s": mirror_age_s})


def check_driver_lease(lease_age_s: float | None, *, warn_s: float = 2400.0,
                       crit_s: float = 5400.0, terminal: bool = False) -> HealthCheck:
    """The MYRIAD DRIVER LEASE (deadman) — freshness of the laptop driver's per-cycle heartbeat.

    On the cluster the queued training arrays keep running even if the laptop driver dies, but no
    new results are PULLED and no next generation is SUBMITTED until it returns — so a stale lease
    means orchestration has stopped: a driver crash/hang, a laptop power-loss, or a VPN/ssh outage
    the driver could not ride out (:data:`max_transport_outage_secs`). The supervisor + ONSTART task
    relaunch it; this check makes the gap VISIBLE within minutes rather than at the end. The driver
    beats every ``poll_secs`` (~10 min), so ~4 missed beats (40 min) is a WARN and ~9 (90 min) a
    CRITICAL. Absent lease → INFO: a laptop-only run (no cluster driver) or one not yet started; a
    terminal run is exempt (the driver has exited by design)."""
    if terminal:
        return HealthCheck("driver_lease", OK, "run terminal — driver lease not expected")
    if lease_age_s is None:
        return HealthCheck("driver_lease", INFO,
                           "no live driver heartbeat (laptop-only run, finished, or not started)")
    if lease_age_s >= crit_s:
        return HealthCheck("driver_lease", CRITICAL,
                           f"driver heartbeat {lease_age_s/60:.0f} min stale — ORCHESTRATION DOWN "
                           "(crash/hang/host-death/VPN); supervisor + ONSTART should relaunch",
                           {"age_s": lease_age_s})
    if lease_age_s >= warn_s:
        return HealthCheck("driver_lease", WARN,
                           f"driver heartbeat {lease_age_s/60:.0f} min stale (expected ~10 min)",
                           {"age_s": lease_age_s})
    return HealthCheck("driver_lease", OK, f"driver heartbeat {lease_age_s/60:.1f} min ago",
                       {"age_s": lease_age_s})


def check_queue_health(snapshot: dict[str, Any] | None, *, transport_warn: int = 3,
                       transport_crit: int = 24) -> HealthCheck:
    """The MYRIAD QUEUE + TRANSPORT panel, from the driver heartbeat(s). Effect-blind by construction:
    it reports job COUNTS (queued / pending across active batches) and the driver's OWN transport-
    failure counters — never a result or performance metric.

    A climbing consecutive pull/ops-failure count is the earliest sign of a VPN/ssh/login-node outage
    (the driver rides it out to :data:`max_transport_outage_secs`, but you want to KNOW hours before
    it could ever turn fatal): ``transport_warn`` consecutive failures → WARN, ``transport_crit`` →
    CRITICAL (an outage the driver may soon raise on, triggering a supervisor relaunch). Queue
    occupancy is surfaced as evidence (an empty queue with work pending and a live driver is a
    usually-transient submit gap, not itself an alarm)."""
    if not snapshot:
        return HealthCheck("queue", INFO,
                           "no driver queue snapshot (laptop-only run, finished, or not started)")
    pending = int(snapshot.get("pending", 0) or 0)
    queued = int(snapshot.get("queued", 0) or 0)
    active = int(snapshot.get("active_batches", 0) or 0)
    tfail = max(int(snapshot.get("pull_failures", 0) or 0), int(snapshot.get("ops_failures", 0) or 0))
    ev = {"pending": pending, "queued": queued, "active_batches": active,
          "pull_failures": int(snapshot.get("pull_failures", 0) or 0),
          "ops_failures": int(snapshot.get("ops_failures", 0) or 0)}
    if tfail >= transport_crit:
        return HealthCheck("queue", CRITICAL,
                           f"transport FAILING ({tfail} consecutive pull/ops failures) — VPN/ssh/"
                           "login outage; driver will raise + relaunch if it persists", ev)
    if tfail >= transport_warn:
        return HealthCheck("queue", WARN,
                           f"transport degrading ({tfail} consecutive pull/ops failures)", ev)
    return HealthCheck("queue", OK,
                       f"{queued} queued, {pending} pending across {active} active batch(es)", ev)


def check_completion_stall(
    completion_times: list[float] | None,
    now: float,
    *,
    n_recent_stall_events: int = 0,
    terminal: bool = False,
    factor_warn: float = 3.0,
    factor_crit: float = 8.0,
    floor_s: float = 1800.0,
) -> HealthCheck:
    """Multi-granularity STALL detection (B1, 2026-07-06): is the run still COMPLETING units?

    ``progress.json`` freshness (the silent-hang check) proves the driver process is ALIVE; this
    check proves it is PRODUCTIVE, from the journal's per-unit completion stream (``candidate_done``
    / ``seed_done`` events). The yardstick is the run's OWN measured cadence — the median
    inter-completion gap — so it self-calibrates to any (n_workers, B*) configuration: silence
    longer than ``factor_warn`` x median (min ``floor_s``) is a WARN, ``factor_crit`` x median a
    CRITICAL (a wedged CUDA training with a happily-alive driver — the hang liveness checks cannot
    see). Median (not mean) so one legitimate thermal pause does not inflate the yardstick.

    ``n_recent_stall_events`` counts run_recycling's own ``test_leg_stall`` detections in the recent
    window — the in-driver detector and this outside view corroborate each other (>=1 -> at least
    WARN). Terminal runs are exempt (no units expected); fewer than 3 completions -> INFO (no
    yardstick yet; the mtime hang check covers the early phase)."""
    if terminal:
        return HealthCheck("completion_stall", OK, "run is terminal — no units expected")
    times = sorted(float(t) for t in (completion_times or []))
    if len(times) < 3:
        base = HealthCheck("completion_stall", INFO,
                           f"{len(times)} unit completions journaled (need 3 for a cadence yardstick)",
                           {"n_completions": len(times)})
        if n_recent_stall_events > 0:
            return HealthCheck("completion_stall", WARN,
                               f"run_recycling reported {n_recent_stall_events} stall event(s)",
                               {"n_stall_events": n_recent_stall_events})
        return base
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    n = len(gaps)
    median = gaps[n // 2] if n % 2 == 1 else 0.5 * (gaps[n // 2 - 1] + gaps[n // 2])
    silence = max(0.0, float(now) - times[-1])
    warn_at = max(floor_s, factor_warn * median)
    crit_at = max(2.0 * floor_s, factor_crit * median)
    ev = {"silence_s": round(silence, 1), "median_gap_s": round(median, 1),
          "warn_at_s": round(warn_at, 1), "crit_at_s": round(crit_at, 1),
          "n_completions": len(times), "n_stall_events": n_recent_stall_events}
    if silence >= crit_at:
        return HealthCheck("completion_stall", CRITICAL,
                           f"NO unit completed for {silence/3600:.1f} h (median cadence "
                           f"{median/60:.0f} min) — a training is wedged", ev)
    if silence >= warn_at or n_recent_stall_events > 0:
        return HealthCheck("completion_stall", WARN,
                           f"quiet for {silence/60:.0f} min vs median cadence {median/60:.0f} min"
                           + (f"; {n_recent_stall_events} in-driver stall event(s)"
                              if n_recent_stall_events else ""), ev)
    return HealthCheck("completion_stall", OK,
                       f"last unit {silence/60:.0f} min ago (median cadence {median/60:.0f} min)", ev)


def check_disk_forecast(
    samples: list[tuple[float, float]] | None,
    *,
    floor_gb: float = 20.0,
    warn_h: float = 48.0,
    crit_h: float = 12.0,
    min_points: int = 5,
) -> HealthCheck:
    """PREDICTIVE disk exhaustion (B3, 2026-07-06): "hits the floor in N hours", not just "low now".

    Least-squares slope over the ``--watch`` history of ``(epoch_s, free_gb)`` samples -> hours until
    the free space crosses the same floor the hard ``disk`` check enforces. A multi-week run fills
    the disk SLOWLY (records + logs + mirror), so the lead time is what turns an eventual CRITICAL
    into a calm, planned cleanup. RAM deliberately gets NO forecast twin: worker RSS oscillates by
    design (pool recycling), so a linear leak model would false-alarm every transition wave."""
    pts = [(float(t), float(g)) for t, g in (samples or [])]
    if len(pts) < int(min_points):
        return HealthCheck("disk_forecast", INFO,
                           f"{len(pts)} disk samples (need {min_points} for a fill-rate forecast)",
                           {"n": len(pts)})
    t0 = pts[0][0]
    xs = [t - t0 for t, _g in pts]
    ys = [g for _t, g in pts]
    n = float(len(pts))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return HealthCheck("disk_forecast", INFO, "degenerate sample spacing", {"n": len(pts)})
    slope_gb_s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    slope_gb_h = slope_gb_s * 3600.0
    free_now = ys[-1]
    ev = {"slope_gb_per_h": round(slope_gb_h, 4), "free_gb": round(free_now, 1),
          "floor_gb": floor_gb, "n": len(pts)}
    if slope_gb_h >= -1e-3:
        return HealthCheck("disk_forecast", OK,
                           f"disk not shrinking ({slope_gb_h:+.2f} GB/h)", ev)
    hours_to_floor = max(0.0, (free_now - floor_gb) / (-slope_gb_h))
    ev["hours_to_floor"] = round(hours_to_floor, 1)
    if hours_to_floor <= crit_h:
        return HealthCheck("disk_forecast", CRITICAL,
                           f"disk hits the {floor_gb:.0f} GB floor in ~{hours_to_floor:.0f} h "
                           f"at {slope_gb_h:+.2f} GB/h", ev)
    if hours_to_floor <= warn_h:
        return HealthCheck("disk_forecast", WARN,
                           f"disk hits the {floor_gb:.0f} GB floor in ~{hours_to_floor:.0f} h "
                           f"at {slope_gb_h:+.2f} GB/h", ev)
    return HealthCheck("disk_forecast", OK,
                       f"~{hours_to_floor:.0f} h of headroom at {slope_gb_h:+.2f} GB/h", ev)


def check_unit_coverage(
    done_units: int | None,
    expected_units: int | None,
    stage: str,
    *,
    claimed_complete: bool = False,
    rate_per_h: float | None = None,
    known_failures: int = 0,
) -> HealthCheck:
    """Expected-vs-done UNIT ledger (B4, 2026-07-06) — the anti-husk guarantee at unit granularity.

    The frozen design fixes exactly how many units each stage must produce (arms x candidates for
    SEARCH; (arms + baselines) x seeds for TEST). Three verdicts:

    * ``claimed_complete`` (the summary says the stage is done) with ``done < expected`` -> CRITICAL:
      a silent SHORTFALL — units are missing yet the run claims completeness (the husk class the
      boolean-level M19 check cannot see at this granularity).
    * ``done > expected`` -> WARN: duplicate ids or a config/roster drift — never healthy.
    * otherwise OK/INFO progress, with a data-driven ETA in the evidence when a completion rate is
      available."""
    if done_units is None or expected_units is None or expected_units <= 0:
        return HealthCheck(f"coverage_{stage}", INFO, f"{stage}: expected-unit ledger unavailable")
    ev: dict[str, Any] = {"done": int(done_units), "expected": int(expected_units),
                          "claimed_complete": bool(claimed_complete),
                          "known_failures": int(known_failures)}
    if rate_per_h is not None and rate_per_h > 0 and done_units < expected_units:
        ev["eta_h"] = round((expected_units - done_units) / rate_per_h, 1)
    if claimed_complete and done_units < expected_units:
        # S19 (2026-07-06): a failed unit legitimately writes no record — when the summary/ledger
        # ACCOUNTS for the whole shortfall, this is a disclosed partial completion (WARN, with the
        # reconciliation in evidence), not the silent-shortfall husk class (CRITICAL).
        if done_units + max(0, int(known_failures)) >= expected_units:
            return HealthCheck(f"coverage_{stage}", WARN,
                               f"{stage} complete with {expected_units - done_units} failed unit(s) "
                               f"(accounted: {known_failures} known failures)", ev)
        return HealthCheck(f"coverage_{stage}", CRITICAL,
                           f"{stage} claims complete but {done_units}/{expected_units} units on disk "
                           "(silent shortfall)", ev)
    if done_units > expected_units:
        return HealthCheck(f"coverage_{stage}", WARN,
                           f"{stage} has {done_units} units for {expected_units} expected "
                           "(duplicates or config drift)", ev)
    pct = 100.0 * done_units / expected_units
    eta = f", ETA ~{ev['eta_h']:.0f} h" if "eta_h" in ev else ""
    sev = OK if done_units == expected_units else INFO
    return HealthCheck(f"coverage_{stage}", sev, f"{stage}: {done_units}/{expected_units} ({pct:.0f}%){eta}", ev)


def check_error_taxonomy(
    taxonomy: dict[str, dict[str, Any]] | None,
    *,
    kind_warn: int = 10,
    total_warn: int = 25,
) -> HealthCheck:
    """The clustered error/triage panel (B5, 2026-07-06): errors BY KIND with counts + affected arms.

    The hard checks alarm on specific rates; this one alarms on VOLUME — any single failure kind
    reaching ``kind_warn`` occurrences (a systematic fault, e.g. an OOM cascade or an API refusal
    wave), or ``total_warn`` failures overall. Otherwise it surfaces the taxonomy as evidence so the
    triage protocol (runbook §5b) starts from data, not from grepping logs."""
    tax = taxonomy or {}
    if not tax:
        return HealthCheck("error_taxonomy", OK, "no failures/warnings journaled")
    counts = {k: int(v.get("count", 0)) for k, v in tax.items()}
    total = sum(counts.values())
    worst_kind = max(counts, key=lambda k: counts[k])
    ev: dict[str, Any] = {"counts": counts, "total": total,
                          "arms": {k: v.get("arms", []) for k, v in tax.items()}}
    if counts[worst_kind] >= kind_warn or total >= total_warn:
        return HealthCheck("error_taxonomy", WARN,
                           f"{total} failure events — dominant kind '{worst_kind}' x{counts[worst_kind]}",
                           ev)
    summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return HealthCheck("error_taxonomy", INFO, f"{total} failure event(s): {summary}", ev)


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def evaluate_health(inputs: dict[str, Any]) -> HealthReport:
    """Run every invariant check over a parsed-inputs dict (see :func:`gather_inputs`). Pure."""
    g = inputs.get
    checks = [
        check_exit_code(g("exit_code")),
        check_progress(int(g("seen_arms", 0) or 0), int(g("expected_arms", 0) or 0),
                       all_arms_tested=g("all_arms_tested")),
        check_disk(g("disk_free_gb")),
        check_ram(g("ram_used_pct")),
        check_gpu_temp(g("gpu_temp_c")),
        check_silent_hang(g("progress_age_s"), terminal=bool(g("terminal", False))),
        check_gate_failure_rate(int(g("n_failed", 0) or 0), int(g("n_attempted", 0) or 0)),
        check_nan_rate(int(g("n_nonfinite", 0) or 0), int(g("n_records", 0) or 0)),
        check_divergence_rate(int(g("n_diverged_runs", 0) or 0), int(g("n_candidates", 0) or 0),
                              winner_diverged=bool(g("winner_diverged", False))),
        check_reward_scale_drift(g("raw_rms_by_arm", {}) or {}),
        check_api_error_rate(int(g("n_api_errors", 0) or 0), int(g("n_api_calls", 0) or 0)),
        check_mirror_freshness(g("mirror_age_s")),
        # 2026-07-08 Myriad-native layer: the cluster driver deadman + queue/transport panel. Both
        # effect-blind (freshness + counts + SGE occupancy, never a result) and cluster-only (a
        # laptop run has no driver_status → INFO). The lease is the ONLY monitor that can see a laptop
        # host-death / driver hang during a cluster run (the trainings survive it on Myriad).
        check_driver_lease(g("driver_lease_age_s"), terminal=bool(g("terminal", False))),
        check_queue_health(g("queue_snapshot")),
        # 2026-07-06 deep-monitoring layer (B1/B4/B5): productivity, unit-coverage, triage.
        check_completion_stall(
            g("completion_times"), float(g("now", time.time()) or time.time()),
            n_recent_stall_events=int(g("n_recent_stall_events", 0) or 0),
            terminal=bool(g("terminal", False)),
        ),
        check_unit_coverage(g("done_search_units"), g("expected_search_units"), "search",
                            claimed_complete=bool(g("search_claimed_complete", False)),
                            rate_per_h=g("completion_rate_per_h")),
        check_unit_coverage(g("done_test_units"), g("expected_test_units"), "test",
                            claimed_complete=bool(g("all_arms_tested") is True),
                            rate_per_h=g("completion_rate_per_h"),
                            known_failures=int(g("test_known_failures", 0) or 0)),
        check_error_taxonomy(g("error_taxonomy")),
    ]
    # Statistical-process-control drift checks (opt-in: the --watch loop accumulates a per-tick history
    # and passes it in). Target 0 with slack k = half the shift we care about; h the decision interval.
    if g("gate_failure_history") is not None:
        checks.append(check_metric_drift("gate_failure", g("gate_failure_history"), 0.0, k=0.03, h=0.15))
    if g("nan_rate_history") is not None:
        checks.append(check_metric_drift("nan_rate", g("nan_rate_history"), 0.0, k=0.01, h=0.05))
    # Training-throughput drift (B2): a sustained fps FALL across candidates = thermal creep /
    # swap / fragmentation degrading the whole run. fps IS comparable across candidates (same net,
    # same hardware), unlike critic loss — whose scale is per-candidate, so its cross-candidate
    # CUSUM would mix distributions and false-alarm (the per-candidate explosion detection in the
    # monitor + the divergence/NaN-rate checks above own that axis). Target/k/h are supplied by the
    # --watch loop from the run's OWN early-baseline median (self-calibrating).
    if g("fps_history") is not None and g("fps_target") is not None:
        t = float(g("fps_target"))
        checks.append(check_metric_drift("fps", g("fps_history"), t,
                                         k=0.05 * t, h=0.30 * t, direction="down"))
    # Predictive exhaustion (B3): fed by the --watch loop's accumulated (epoch_s, free_gb) samples.
    if g("disk_history") is not None:
        checks.append(check_disk_forecast(g("disk_history")))
    checks.extend(_campaign_lane_checks(inputs))
    return HealthReport(checks)


def _campaign_lane_checks(inputs: dict[str, Any]) -> list[HealthCheck]:
    """The 2026-07-26 CPU-lane checks (``src/cluster/campaign_health``), each opt-in on its input.

    Every one is a failure the checks above CANNOT see — capacity that never accumulates, a stalled
    serial search chain, one bad node eating tasks, a rung forecast drifting from the model, a
    device/thread MIX inside the scored leg. They live in ``src/cluster`` because they are
    cluster-lane facts, and are imported HERE (function-local) because that module imports this
    one's severity vocabulary: a module-level import either way would be circular.

    A laptop run supplies none of these inputs and gets none of the checks — no false alarms.
    """
    g = inputs.get
    if not any(inputs.get(k) is not None for k in
               ("accumulation_report", "chain_progress", "host_attempts",
                "rung_targets", "env_fp_labels", "substrate_fields", "frozen_winners", "kill_verdict", "record_sanity", "authoring_health",
                "unreadable_records", "arm_progress", "seed_digests",
                "reward_hash_by_seed", "unit_counts", "scored_lengths",
                "freeze_state", "seed_sets")):
        return []
    from src.cluster import campaign_health as ch

    out: list[HealthCheck] = []
    if g("accumulation_report") is not None:
        out.append(ch.check_capacity_accumulation(
            g("accumulation_report"), expected_cores=int(g("expected_cores", 0) or 0),
            hours_in=float(g("lane_hours_in", 0.0) or 0.0)))
    if g("chain_progress") is not None:
        out.append(ch.check_chain_progress(g("chain_progress")))
    if g("host_attempts") is not None:
        out.append(ch.check_host_failure_concentration(
            g("host_failures", {}) or {}, g("host_attempts") or {}))
    if g("rung_targets") is not None:
        out.append(ch.check_rung_forecast(
            completed_trainings=int(g("done_all_trainings", g("done_test_units", 0)) or 0),
            elapsed_hours=float(g("lane_hours_in", 0.0) or 0.0),
            hours_remaining=float(g("lane_hours_remaining", 0.0) or 0.0),
            rung_targets=g("rung_targets")))
    if g("env_fp_labels") is not None:
        out.append(ch.check_determinism_homogeneity(g("env_fp_labels")))
    if g("substrate_fields") is not None:
        out.append(ch.check_substrate_fields(g("substrate_fields")))
    if g("frozen_winners") is not None:
        out.append(ch.check_winner_execution_quality(g("frozen_winners")))
    if g("freeze_state") is not None:
        frozen, rec_h, cur_h = g("freeze_state")
        out.append(ch.check_design_drift(frozen, rec_h, cur_h))
    if g("seed_sets") is not None:
        # Pass the CURRENT rung target so alignment is judged near completion, not mid-fill: arms
        # complete seeds at different rates and a partial intersection is meaningless until then.
        _rt = g("rung_targets") or {}
        _target = max((int(k) for k in _rt), default=None) if isinstance(_rt, dict) else None
        out.append(ch.check_seed_alignment(g("seed_sets"), target_seeds=_target))
    if g("seed_digests") is not None:
        out.append(ch.check_seed_replication(g("seed_digests")))
    if g("reward_hash_by_seed") is not None:
        out.append(ch.check_arm_differentiation(g("reward_hash_by_seed")))
    if g("unit_counts") is not None:
        out.append(ch.check_duplicate_units(g("unit_counts")))
    if g("scored_lengths") is not None:
        out.append(ch.check_test_window_consistency(g("scored_lengths")))
    if g("unreadable_records") is not None:
        nu, nt, ex = g("unreadable_records")
        out.append(ch.check_unreadable_records(nu, nt, ex))
    if g("arm_progress") is not None:
        out.append(ch.check_arm_progress_symmetry(g("arm_progress")))
    if g("authoring_health") is not None:
        # THE EARLIEST layer: authoring happens in the first minutes, long before any training
        # completes, and a systematically broken arm otherwise burns its whole budget unnoticed.
        out.append(ch.check_authoring_health(g("authoring_health")))
    if g("record_sanity") is not None:
        # LIVE garbage detection on every new record — the shortest path from a void record to an
        # alert. Effect-blind, so it cannot preview the result or touch the confirmatory look.
        out.append(ch.check_record_sanity(g("record_sanity")))
    if g("kill_verdict") is not None:
        # The CONSUMER of the admin-kill verdict the gatherer computes. Without this the verdict sits
        # in the inputs dict and never reaches the report, the severity, or the phone alert.
        out.append(ch.check_admin_kill(g("kill_verdict")))
    return out


# --------------------------------------------------------------------------- #
# Disk/telemetry gatherer (best-effort; each probe guarded)                   #
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _cluster_diverged_runs(anomaly_lines: list[dict[str, Any]]) -> int:
    """Count DISTINCT diverged runs from anomalies.jsonl (append-only; the LINE count over-states it).

    Mirrors ``analyze_campaign.divergence_report``: a critic_explosion line belongs to a NEW run when
    its step goes BACKWARDS vs the previous explosion (a step reset = a fresh training). Report-only
    heuristic — good enough for a live rate alarm."""
    explosions = [a for a in anomaly_lines if str(a.get("kind", a.get("event", ""))).startswith("critic")]
    if not explosions:
        return 0
    runs = 1
    prev_step = -1
    for a in explosions:
        step = int(a.get("step", a.get("fields", {}).get("step", 0)) or 0)
        if step < prev_step:
            runs += 1
        prev_step = step
    return runs


#: --watch record-scan cache: path -> (mtime_ns, size, parsed row). Records are WRITE-ONCE, so a
#: (mtime, size) hit skips re-parsing ~540 json files every tick; only NEW records are read.
_RECORD_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}

#: The archive mirror's documented default destination (scripts/mirror_archive.ps1 -MirrorRoot).
_MIRROR_ROOT = Path("D:/llm_rp_archive_mirror")


def _primary_metric(m: dict[str, Any], *, search: bool) -> float | None:
    """The record's PRIMARY score, chosen by stage — the `check_nan_rate` numerator's input.

    Two things were wrong with the previous one-liner ``m.get("val_fitness", m.get("test_sharpe"))``,
    both found live at 2026-07-28 06:53Z when the sentinel raised CRITICAL on a HEALTHY baseline:

    1. ``dict.get(k, default)`` falls back only when the key is ABSENT. Test-leg baseline records
       carry ``val_fitness`` PRESENT-but-``nan``, so the fallback never fired and a record with a
       perfectly good ``test_sharpe: -0.190`` was counted as non-finite.
    2. The primary metric is stage-dependent. On the SEARCH leg ``val_fitness`` IS the score (it
       selects winners). On the TEST leg the scored quantity is ``test_sharpe``, and ``val_fitness``
       is merely inherited from the winner — legitimately absent for BASELINE arms, which have no
       validation-selected winner at all.

    This matters beyond one alert: the H1 canon is 11 baseline comparators across every seed rung, so
    the old expression would have pinned the sentinel to a permanent CRITICAL exactly as the scored
    leg fills up — the cry-wolf failure mode that makes an operator ignore the one alert that counts.

    Detection power is PRESERVED, not suppressed: if no candidate is finite, the first PRESENT value
    is returned so a genuinely NaN-scored record still counts as non-finite.
    """
    import math

    order = ("val_fitness", "test_sharpe") if search else ("test_sharpe", "val_fitness")
    first_present: float | None = None
    for key in order:
        if key not in m:
            continue
        try:
            v = float(m[key])
        except (TypeError, ValueError):
            continue
        if first_present is None:
            first_present = v
        if math.isfinite(v):
            return v
    return first_present


def _scan_records(camp_root: Path) -> dict[str, Any]:
    """One cached pass over the archive's records -> the rate/scale check inputs (S17, 2026-07-06).

    Produces ``n_records`` (metric-bearing records), ``n_nonfinite`` (non-finite primary metric —
    the check_nan_rate numerator), ``n_search_candidates`` (the divergence denominator),
    ``raw_rms_by_arm`` (max realized PopArt ``raw_rms`` per arm — the P5 reward-scale audit), and
    ``arms_seen``. Best-effort per file; an unreadable record is skipped, never fatal."""
    import math

    res: dict[str, Any] = {"n_records": 0, "n_nonfinite": 0, "n_search_candidates": 0,
                           "raw_rms_by_arm": {}, "arms_seen": set()}
    for stage in ("search", "test"):
        d = camp_root / stage
        if not d.is_dir():
            continue
        for rec_path in d.rglob("record.json"):
            try:
                st = rec_path.stat()
                key = str(rec_path)
                hit = _RECORD_CACHE.get(key)
                if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                    row = hit[2]
                else:
                    rec = json.loads(rec_path.read_text(encoding="utf-8", errors="replace"))
                    m = rec.get("metrics") or {}
                    metric = _primary_metric(m, search=stage == "search")
                    scale = m.get("popart_scale")
                    raw_rms = None
                    if isinstance(scale, dict):
                        v = scale.get("raw_rms_max", scale.get("raw_rms_last"))
                        raw_rms = None if v is None else float(v)
                    row = {
                        "arm": str(rec.get("arm") or ""),
                        "metric": None if metric is None else float(metric),
                        "raw_rms": raw_rms,
                        "search": stage == "search",
                    }
                    _RECORD_CACHE[key] = (st.st_mtime_ns, st.st_size, row)
            except Exception:  # noqa: BLE001 — one unreadable record must not kill the probe
                continue
            if row["search"]:
                res["n_search_candidates"] += 1
            if row["arm"]:
                res["arms_seen"].add(row["arm"])
            if row["metric"] is not None:
                res["n_records"] += 1
                if not math.isfinite(row["metric"]):
                    res["n_nonfinite"] += 1
            if row["raw_rms"] is not None and row["arm"]:
                prev = res["raw_rms_by_arm"].get(row["arm"])
                if prev is None or row["raw_rms"] > prev:
                    res["raw_rms_by_arm"][row["arm"]] = row["raw_rms"]
    return res


def _scan_ledgered_failures(camp_root: Path) -> int:
    """Deduped ledgered-failure count across BOTH layouts (S17): the parallel per-arm
    ``search/<arm>/failures.jsonl`` AND the serial ``search/<prefix>-<arm>.failures.jsonl``."""
    sroot = camp_root / "search"
    if not sroot.is_dir():
        return 0
    seen: set[str] = set()
    ledgers = list(sroot.rglob("failures.jsonl")) + list(sroot.glob("*.failures.jsonl"))
    for lp in ledgers:
        try:
            for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except (json.JSONDecodeError, ValueError):
                    continue
                cid = str(row.get("candidate_id") or "")
                seen.add(cid or f"{lp.name}#{len(seen)}")
        except OSError:
            continue
    return len(seen)


def _read_driver_status(camp_root: Path) -> tuple[float | None, dict[str, Any] | None]:
    """Aggregate the Myriad driver heartbeat(s) under ``<camp_root>/driver_status/*.json`` (2026-07-08).

    Only RUNNING beats count: the freshest running beat's age is the driver LEASE (deadman for the
    :func:`check_driver_lease` check), and the running beats' summed pending + summed queued + worst
    transport-failure counters form the queue/transport snapshot. A batch's FINAL beat flips to
    ``phase='done'`` (excluded here), so a cleanly-completed batch never reads as a hung one and a
    finished campaign yields no live lease (→ the checks degrade to INFO, not a false CRITICAL).
    Best-effort: an unreadable beat is skipped, never fatal."""
    sdir = camp_root / "driver_status"
    if not sdir.is_dir():
        return None, None
    now = time.time()
    freshest: float | None = None
    pending = queued = pull_f = ops_f = active = 0
    for f in sdir.glob("*.json"):
        try:
            age = max(0.0, now - f.stat().st_mtime)
            row = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if str(row.get("phase")) != "running":
            continue  # a done/final beat is not a live lease
        freshest = age if freshest is None else min(freshest, age)
        active += 1
        pending += int(row.get("pending", 0) or 0)
        queued += len(row.get("queue_names", []) or [])
        pull_f = max(pull_f, int(row.get("pull_failures", 0) or 0))
        ops_f = max(ops_f, int(row.get("ops_failures", 0) or 0))
    if freshest is None:
        return None, None
    snap = {"pending": pending, "queued": queued, "pull_failures": pull_f,
            "ops_failures": ops_f, "active_batches": active}
    return freshest, snap


def gather_inputs(run_dir: Path) -> dict[str, Any]:
    """Read the on-disk campaign artifacts + system telemetry into the :func:`evaluate_health` dict.

    READ-ONLY and best-effort: every probe is guarded so one failure degrades a single check to
    ``UNKNOWN`` (via a missing key) rather than killing the sentinel. ``run_dir`` is the campaign
    output dir (the sentinel checks both it and ``run_dir/search`` for progress.json / summary)."""
    run_dir = Path(run_dir)
    out: dict[str, Any] = {}

    # System telemetry.
    try:
        import shutil as _sh
        out["disk_free_gb"] = _sh.disk_usage(str(run_dir if run_dir.exists() else Path.cwd())).free / 1e9
    except OSError:
        pass
    try:
        import psutil  # type: ignore
        out["ram_used_pct"] = float(psutil.virtual_memory().percent)
    except Exception:  # noqa: BLE001 — psutil absent or probe failed
        pass
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        out["gpu_temp_c"] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 — no GPU telemetry available
        pass

    # Progress freshness + terminal state.
    prog = None
    for p in (run_dir / "search" / "progress.json", run_dir / "progress.json"):
        if p.is_file():
            prog = p
            break
    if prog is not None:
        try:
            out["progress_age_s"] = max(0.0, time.time() - prog.stat().st_mtime)
        except OSError:
            pass
        try:
            state = json.loads(prog.read_text(encoding="utf-8"))
            out["terminal"] = state.get("phase") in ("done", "error")
            # Live training throughput (B2): the current fps sample for the cross-candidate
            # thermal-creep drift check (the --watch loop accumulates the history).
            fps = (state.get("training") or {}).get("fps")
            if fps is not None:
                out["fps_now"] = float(fps)
        except (OSError, ValueError, TypeError):
            pass

    # Campaign summary (exit code + coverage + the S19 failed-seed reconciliation).
    for p in (run_dir / "campaign_summary.json", run_dir.parent / "campaign_summary.json"):
        if p.is_file():
            try:
                summary = json.loads(p.read_text(encoding="utf-8"))
                out["exit_code"] = summary.get("exit_code", 0)
                out["all_arms_tested"] = summary.get("all_arms_tested")
                arms = summary.get("arms") or summary.get("summaries") or []
                out["seen_arms"] = len({a.get("arm") for a in arms if isinstance(a, dict)})
                out["test_known_failures"] = sum(
                    int(a.get("n_failed", 0) or 0) for a in arms if isinstance(a, dict)
                )
            except (OSError, ValueError):
                pass
            break

    # Anomalies -> divergence clustering.
    anomalies = _read_jsonl(run_dir / "anomalies.jsonl")
    if not anomalies:
        anomalies = _read_jsonl(run_dir / "search" / "anomalies.jsonl")
    out["n_diverged_runs"] = _cluster_diverged_runs(anomalies)

    # Events -> API error + gate-failure rate (best-effort tallies over the structured log).
    # S20 (2026-07-06): the campaign attaches its run logging under <output>/search, so the
    # documented invocation (`sentinel.py outputs/campaign`) must read the UNION of the possible
    # event ledgers, not just run_dir's own.
    _camp_root = run_dir.parent if run_dir.name == "search" else run_dir
    events = []
    _seen_ev_paths = set()
    for _ep in (run_dir / "events.jsonl", _camp_root / "events.jsonl",
                _camp_root / "search" / "events.jsonl",
                _camp_root / "baselines" / "events.jsonl"):
        rp = str(_ep.resolve()) if _ep.exists() else str(_ep)
        if rp in _seen_ev_paths:
            continue
        _seen_ev_paths.add(rp)
        events.extend(_read_jsonl(_ep))
    if events:
        kinds = Counter(str(e.get("event", "")) for e in events)
        out["n_api_calls"] = sum(v for k, v in kinds.items() if "llm_call" in k or "api" in k)
        out["n_api_errors"] = sum(v for k, v in kinds.items() if "error" in k or "refus" in k or "degrad" in k)

    # 2026-07-06 deep-monitoring inputs: the journal's per-unit completion stream (B1), the
    # expected-vs-done unit ledger (B4), and the error taxonomy (B5). All guarded — an absent/
    # unreadable source degrades those checks to INFO, never kills the sentinel. The watcher often
    # points at <campaign>/search (the progress.json home); the archive/journal ledgers live at the
    # campaign ROOT, so resolve it for these probes.
    out["now"] = time.time()
    camp_root = run_dir.parent if run_dir.name == "search" else run_dir
    try:
        import sys as _sys

        _root = Path(__file__).resolve().parents[1]
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from src.utils.journal import completions as _completions
        from src.utils.journal import error_taxonomy as _taxonomy
        from src.utils.journal import parse_ts as _parse_ts
        from src.utils.journal import read_events as _read_events

        # S20: union over every ledger location (the campaign logs under <output>/search).
        evs = []
        _seen_j = set()
        for _jp in (run_dir / "events.jsonl", camp_root / "events.jsonl",
                    camp_root / "search" / "events.jsonl",
                    camp_root / "baselines" / "events.jsonl"):
            rp = str(_jp.resolve()) if _jp.exists() else str(_jp)
            if rp in _seen_j:
                continue
            _seen_j.add(rp)
            evs.extend(_read_events(_jp))
        if evs:
            comp = _completions(evs)
            out["completion_times"] = [t for t, _e in comp]
            out["error_taxonomy"] = _taxonomy(evs)
            _now = float(out["now"])
            n_stall = 0
            for e in evs:
                if e.get("event") != "test_leg_stall":
                    continue
                t = _parse_ts(e)
                if t is not None and _now - t <= 7200.0:
                    n_stall += 1
            out["n_recent_stall_events"] = n_stall
            recent = [t for t in out["completion_times"] if _now - t <= 6 * 3600.0]
            if len(recent) >= 2:
                out["completion_rate_per_h"] = len(recent) / 6.0
    except Exception:  # noqa: BLE001 — journal read is best-effort
        pass

    # S17 (2026-07-06): the rate/scale/divergence/mirror inputs — previously NEVER produced, so six
    # documented checks + both CUSUM drift monitors were permanently inert in live use. All guarded.
    try:
        scan = _scan_records(camp_root)
        n_ledgered = _scan_ledgered_failures(camp_root)
        out["n_records"] = int(scan["n_records"])
        out["n_nonfinite"] = int(scan["n_nonfinite"])
        out["n_candidates"] = int(scan["n_search_candidates"])
        if scan["raw_rms_by_arm"]:
            out["raw_rms_by_arm"] = dict(scan["raw_rms_by_arm"])
        out["n_failed"] = n_ledgered
        # attempted = archived search candidates + ledgered gate failures (the campaign's own
        # budget-slot resolution contract).
        out["n_attempted"] = int(scan["n_search_candidates"]) + n_ledgered
        if not out.get("seen_arms"):
            out["seen_arms"] = len(scan["arms_seen"])
    except Exception:  # noqa: BLE001 — archive scan is best-effort
        n_ledgered = 0
    # Winner divergence (best-effort): a frozen winner whose candidate id appears among the
    # critic-explosion anomaly rows (rows carry identity since the M6 fix).
    try:
        winner_ids: set[str] = set()
        for froot in camp_root.glob("frozen*"):
            for wrec in froot.rglob("record.json"):
                try:
                    winner_ids.add(str(json.loads(
                        wrec.read_text(encoding="utf-8")).get("candidate_id") or ""))
                except (OSError, ValueError):
                    continue
        winner_ids.discard("")
        if winner_ids:
            expl = {str(a.get("cand") or a.get("candidate_id") or "") for a in anomalies
                    if str(a.get("kind", a.get("event", ""))).startswith("critic")}
            out["winner_diverged"] = bool(winner_ids & expl)
    except Exception:  # noqa: BLE001
        pass
    # Mirror freshness: the documented default destination's last-pass marker.
    try:
        marker = _MIRROR_ROOT / "mirror.log"
        target = marker if marker.is_file() else (_MIRROR_ROOT if _MIRROR_ROOT.is_dir() else None)
        if target is not None:
            out["mirror_age_s"] = max(0.0, time.time() - target.stat().st_mtime)
    except OSError:
        pass

    # Myriad driver LEASE + queue/transport snapshot (2026-07-08) — the cluster orchestration deadman
    # (host-death / driver hang / VPN outage). Cluster-only: a laptop run has no driver_status dir.
    try:
        lease_age, queue_snap = _read_driver_status(camp_root)
        if lease_age is not None:
            out["driver_lease_age_s"] = lease_age
        if queue_snap is not None:
            out["queue_snapshot"] = queue_snap
    except Exception:  # noqa: BLE001 — best-effort; absence degrades the two checks to INFO
        pass

    # Expected units come from config (the frozen source of truth); done units from the archive.
    # NB when the per-arm adaptive seed schema lands (seed ratification), the expected-test formula
    # must switch to summing each arm's own seed count.
    try:
        from src.utils.config import load_config

        camp = load_config("campaign")
        arms = list(camp.get("arms") or [])
        cpa = int(camp.get("candidates_per_arm") or 0)
        # NOT `list(...)`: under the tiered schema this is a MAPPING, and coercing it here silently
        # collapses it to its two KEYS before the shape test below can ever see it.
        seeds = camp.get("seeds") or []
        baselines = list(camp.get("h1_baselines") or [])
        if arms and cpa:
            out["expected_search_units"] = len(arms) * cpa
        # TIERED SEEDS (R101). `campaign.seeds` is a MAPPING `{mode: tiered, tiers: [30..568]}`, not
        # a list of seed ids — so `len(seeds)` counted the two dict KEYS and the expectation came out
        # as (9 arms + 11 baselines) x 2 = 40. Live at 2026-07-28 that produced
        # `coverage_test WARN: test has 168 units for 40 expected (duplicates or config drift)`,
        # i.e. a permanent false "config drift" warning from the moment the scored leg passed 40
        # units. The module comment already anticipated this ("must switch to summing each arm's own
        # seed count") — the tiered schema has now landed, so this is that switch. The DESIGN target
        # is the deepest rung; progress against it is honest, whereas the first rung would flip to a
        # false "complete" the moment rung 30 filled.
        n_seeds = 0
        if isinstance(seeds, dict):
            tiers = [int(t) for t in (seeds.get("tiers") or []) if str(t).lstrip("-").isdigit()]
            n_seeds = max(tiers) if tiers else 0
        elif seeds:
            n_seeds = len(seeds)
        if arms and n_seeds:
            out["expected_test_units"] = (len(arms) + len(baselines)) * n_seeds
        out.setdefault("expected_arms", len(arms))
    except Exception:  # noqa: BLE001 — config unavailable -> the coverage checks degrade to INFO
        pass
    # CAMPAIGN-WIDE training count for the RUNG FORECAST (2026-08-01, RUN 12).
    # `done_test_units` below counts ONLY `camp_root/test` -- the CORE line's tier -- which is the
    # right scope for `check_unit_coverage` (its denominator is core-scoped too). It is the WRONG
    # numerator for `check_rung_forecast`, whose denominator is `lanes.total_trainings(rung)` =
    # 1,800 + 71n, i.e. SEARCH + TEST across the WHOLE campaign. Pairing a core-only numerator with
    # a campaign-wide denominator under-counted by ~3x (376 seen vs 1,152 test records, and 2,697
    # trainings in total) and made the monitor announce "rung 0 -- cannot reach rung 30" while the
    # archive was on track for rung ~189. That is the SAME class of defect the comment below already
    # records, and its own verdict applies: a throughput number that wrong "invites exactly the wrong
    # intervention on a frozen design, so it is worse than no number at all".
    # Same quarantine/staging exclusions as the lane-start fallback, for the same reason.
    if camp_root.is_dir():
        try:
            out["done_all_trainings"] = sum(
                1 for q in camp_root.rglob("record.json")
                if not any(part.startswith(("_quarantined", ".pull_tmp")) for part in q.parts)
            )
        except OSError:
            pass
    for key, sub in (("done_search_units", "search"), ("done_test_units", "test")):
        d = camp_root / sub
        if d.is_dir():
            try:
                out[key] = sum(1 for _ in d.rglob("record.json"))
            except OSError:
                pass
        elif camp_root.is_dir():
            # The campaign root exists but this stage hasn't produced its subtree yet: truthfully
            # ZERO done (renders "0/210 (0%)"), vs a mis-pointed run_dir which stays unavailable.
            out[key] = 0
    # S19 (2026-07-06): a budget SLOT is legitimately resolved by an archived record OR a ledgered
    # gate failure (failures write no record.json), so the SEARCH coverage counts RESOLVED units —
    # otherwise any normally-completed campaign with >=1 gate failure would end in a false CRITICAL
    # "silent shortfall". The reconciliation is carried in the evidence via done+ledgered.
    if out.get("done_search_units") is not None:
        out["done_search_units"] = int(out["done_search_units"]) + int(n_ledgered)
    out["search_claimed_complete"] = out.get("all_arms_tested") is True

    try:
        out.update(_gather_campaign_lane(camp_root, out))
    except Exception:  # noqa: BLE001 — best-effort; absence just switches the lane checks off
        pass

    return out


def _prereg_stop_and_rungs(now: float) -> tuple[float | None, dict[int, int]]:
    """``(hours to the pre-registered exogenous stop, {rung: trainings needed})``.

    Both come from ``config/preregistration.yaml`` — ``model_suite.exogenous_stop`` (the calendar
    stop; exogenous BY DESIGN, never a function of the observed effect) and the ``seeds.tiers``
    ladder costed through :func:`src.cluster.lanes.total_trainings`. Returns ``(None, {})`` if either
    is absent so the forecast simply does not run.
    """
    import datetime as _dt

    from src.cluster import lanes
    from src.utils.config import load_config

    prereg = load_config("preregistration")
    stop = ((prereg.get("model_suite") or {}).get("exogenous_stop")) or None
    tiers = ((prereg.get("seeds") or {}).get("tiers")) or []
    if not stop or not tiers:
        return None, {}
    stop_dt = _dt.datetime.fromisoformat(str(stop)).replace(tzinfo=_dt.timezone.utc)
    hours = max(0.0, (stop_dt.timestamp() - now) / 3600.0)
    return hours, {int(t): int(lanes.total_trainings(int(t))) for t in tiers}


def _gather_campaign_lane(camp_root: Path, out: dict[str, Any]) -> dict[str, Any]:
    """Inputs for the CPU-lane checks, each derived from a real artifact or left ABSENT.

    Nothing here is invented: capacity comes from the telemetry log the watcher writes, chain
    progress from the archive, host attribution from the mirrored epilogue ledgers, and the two
    facts that can only be DECLARED (the forecast we are holding ourselves to and the exogenous stop
    date) from ``outputs/allocation_state.json``, written at GO. A missing input means its check
    simply does not run — a monitor that guesses is worse than one that is silent.
    """
    from src.cluster import lanes
    from src.cluster.telemetry import _LOG_PATH, accumulation_report, load_state

    lane: dict[str, Any] = {}
    state = load_state() or {}
    now = float(out.get("now") or time.time())

    if Path(_LOG_PATH).is_file():
        lane["accumulation_report"] = accumulation_report()

    # Elapsed: the DECLARED lane start when GO recorded one, else the earliest archived record. The
    # fallback under-reports by roughly one training (~8.5 h — nothing lands before the first one
    # finishes), which only ever DELAYS a warning; it never manufactures one.
    started = state.get("lane_started_utc")
    if started is None:
        # EXCLUDE quarantined and in-flight staging (2026-07-28). This fallback dates the lane from
        # the EARLIEST archived record, and `rglob` swept `_quarantined_precampaign_*` too — records
        # deliberately set aside precisely BECAUSE they belong to an earlier run, and whose mtimes a
        # move preserves. That dated the lane 2026-07-24 instead of 2026-07-28, reporting ~95 h
        # elapsed against a real 4.8 h and deflating every derived rate ~20x. Live consequence:
        # `rung_forecast` announced "162 done at 1.7/h -> rung 0 (next rung 30 needs 2,538 more)"
        # while the archive was actually completing ~180 records/h and still accelerating — i.e. the
        # monitor said the campaign could not reach its FIRST seed rung when it was on track to pass
        # it comfortably. A throughput number that wrong invites exactly the wrong intervention on a
        # frozen design, so it is worse than no number at all.
        mtimes = [p.stat().st_mtime for p in camp_root.rglob("record.json")
                  if not any(part.startswith(("_quarantined", ".pull_tmp")) for part in p.parts)] \
            if camp_root.is_dir() else []
        started = min(mtimes) if mtimes else None
    if started is not None:
        lane["lane_hours_in"] = max(0.0, (now - float(started)) / 3600.0)
    if state.get("lane_expected_cores"):
        lane["expected_cores"] = int(state["lane_expected_cores"])

    # The stop and the rung ladder are PRE-REGISTERED design facts, so they are read from the
    # pre-registration rather than re-declared at GO — one source of truth, and no launch-day step
    # that can be forgotten (which would leave the rung forecast silently switched off all campaign).
    stop_h, rungs = _prereg_stop_and_rungs(now)
    if stop_h is not None and rungs:
        lane["lane_hours_remaining"] = stop_h
        lane["rung_targets"] = rungs

    # Critical path: completed steps + idle time per serial-chain arm, straight from the archive.
    search_root = camp_root / "search"
    chains: dict[str, dict[str, Any]] = {}
    # NOTE: CORRECTED 2026-08-02 (D27, record §101.2). This iterated `SERIAL_CHAIN_STEPS`, which counts
    # DISPATCH steps, and compared it against a count of RECORDS. The units did not match, so this
    # check reported **"cma_es 9/4"** and `check_chain_progress` classified the campaign's LONGEST
    # serial chain as COMPLETE — leaving the one arm whose stall would silently slip the finish date
    # with no stall detector at all. `SERIAL_CHAIN_BUDGET` is the matched candidate budget, which is
    # the quantity a record count should be compared with.
    for arm, total in lanes.SERIAL_CHAIN_BUDGET.items():
        arm_root = search_root / arm
        if not arm_root.is_dir():
            continue
        recs = [p.stat().st_mtime for p in arm_root.rglob("record.json")]
        chains[arm] = {"completed": len(recs), "total": total,
                       "hours_since_last": (now - max(recs)) / 3600.0 if recs
                       else lane.get("lane_hours_in", 0.0)}
    if chains:
        lane["chain_progress"] = chains

    # Bad-node attribution from the mirrored epilogue ledgers (see poll.sync_epilogue_ledgers).
    ledger_dir = camp_root / "ledger"
    if ledger_dir.is_dir():
        from src.cluster.ledger import host_task_counts, read_epilogue

        rows: list[dict[str, Any]] = []
        for p in sorted(ledger_dir.glob("*.epilogue.jsonl")):
            rows.extend(read_epilogue(p))
        if rows:
            attempts, failed = host_task_counts(rows)
            lane["host_attempts"], lane["host_failures"] = attempts, failed

            # ADMIN-KILL VERDICT (deep review 2026-07-26, #57). `poll.sync_epilogue_ledgers` names
            # TWO consumers of these rows — the bad-node check above and
            # `killswitch.classify_task_deaths`. Only the first was ever wired: the classifier had NO
            # production call site anywhere in the repo, and neither did `killswitch.write_incident`.
            # The submission GATE (`incident_blocks_submission`) IS wired in
            # `cluster/campaign.py`, so the architecture was a gate that nothing could ever trip —
            # the automated Myriad-access guard could not fire.
            # This computes the verdict from rows ALREADY read and reports it. It deliberately does
            # NOT call `write_incident`: that blocks all submission until a human clears it, i.e. it
            # can halt a 23-day campaign, which is an operator decision (flagged for review), not
            # something a read-only watcher should start doing on its own.
            from src.cluster.killswitch import classify_task_deaths

            verdict = classify_task_deaths(rows)
            lane["kill_verdict"] = {
                "classification": verdict.classification,
                "action": verdict.action,
                "reason": verdict.reason,
                "n_deaths": verdict.n_deaths,
                "n_hosts": verdict.n_hosts,
                "n_undated": verdict.n_undated,
                # NOTE: SCOPE, corrected 2026-07-27. This field used to read "no incident file is
                # written from here", which was true of the SENTINEL and false of the SYSTEM: the
                # DRIVER does enforce (campaign._enforce_kill_switch writes the incident and
                # campaign.py's submission gate then blocks every batch). An operator reading
                # "enforced: False" beside a retreat verdict would conclude nothing had happened,
                # while the campaign was already halted. Renamed to say exactly what it means.
                "enforced_by_this_watcher": False,
                "enforced_by_the_driver": True,
                "note": ("this watcher is READ-ONLY; the driver enforces independently via "
                         "_enforce_kill_switch -> MYRIAD_KILL_INCIDENT.json, which blocks ALL "
                         "submission until killswitch.clear_incident() is called by a human"),
                # The watcher classifies WITHOUT h_rt_secs, so its verdict is deliberately the more
                # pessimistic of the two: it cannot tell a walltime kill from an administrative one
                # and will say admin_kill for both. The driver, which knows the walltime it
                # requested, is the authority. Treat a watcher retreat as "go and look".
                "discriminator": "no h_rt_secs available to this watcher",
            }

    # ARCHIVE READABILITY + PER-ARM PROGRESS (2026-07-27), computed in ONE pass over the records.
    # Both close failures that every existing indicator hides: a torn record is silently SKIPPED by
    # every reader, and a single dead arm is masked by the global completion rate its siblings keep
    # healthy. One walk serves both, so the cost is a stat per record.
    try:
        n_total = n_bad = 0
        bad_examples: list[str] = []
        per_arm_prog: dict[str, dict[str, Any]] = {}
        now_e = float(out.get("now") or time.time())
        for rec_path in camp_root.rglob("record.json"):
            # `_quarantined*` as well as `.pull_tmp` (2026-07-28). Quarantined records belong to an
            # EARLIER run and a move preserves their mtimes, so counting them made two arms look
            # permanently stalled: `arm_progress_symmetry` reported CRITICAL "placebo (16 records,
            # silent 66.4h)" on a campaign that was 7.7 h OLD — the 16 were 2026-07-25 leftovers set
            # aside at 00:23. THIRD instrument to trip on this (lane elapsed and the record census
            # were the others), so it is a pattern, not a one-off: anything walking the archive must
            # exclude BOTH the in-flight staging and the deliberately-set-aside past.
            if any(x.startswith((".pull_tmp", "_quarantined")) for x in rec_path.parts):
                continue
            n_total += 1
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                if not isinstance(rec, dict):
                    raise ValueError("not an object")
            except Exception:  # noqa: BLE001 — the COUNT is the symptom; the skip is not
                n_bad += 1
                if len(bad_examples) < 5:
                    bad_examples.append(str(rec_path.relative_to(camp_root)))
                continue
            arm = str(rec.get("arm") or rec_path.parent.parent.name)
            st = per_arm_prog.setdefault(arm, {"n_records": 0, "hours_since_last": 1e9})
            st["n_records"] += 1
            age_h = max(0.0, (now_e - rec_path.stat().st_mtime) / 3600.0)
            st["hours_since_last"] = min(st["hours_since_last"], age_h)
        if n_total:
            lane["unreadable_records"] = (n_bad, n_total, bad_examples)
        if len(per_arm_prog) >= 3:
            lane["arm_progress"] = per_arm_prog
    except Exception:  # noqa: BLE001
        pass

    # DESIGN DRIFT (2026-07-27): re-verify the freeze EVERY poll, not once before launch. An edit to
    # a hash-bound file mid-run splits the campaign into records answering different questions, and
    # nothing else would notice. One hash per poll.
    try:
        import yaml as _yaml

        from freeze import canonical_hash  # scripts/ is on sys.path for the sentinel

        prereg = _yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
        lane["freeze_state"] = (bool(prereg.get("frozen")),
                                prereg.get("freeze_hash"), canonical_hash())
    except Exception:  # noqa: BLE001 — unverifiable is reported by the check itself, never guessed
        pass

    # RESULT MEANINGFULNESS (2026-07-27) over the SCORED records only — the failures that produce a
    # perfectly healthy-looking archive whose numbers mean nothing: replication that is fake (every
    # interval then fiction), arms that never actually differed (the contrast structurally zero),
    # a unit archived twice (silent double-counting), and records scored over different windows.
    try:
        import hashlib

        scored_roots = [p for p in [camp_root / "test", *sorted(camp_root.glob("test_leg_*"))]
                        if p.is_dir()]
        digest_by_arm: dict[str, dict[int, str]] = {}
        hash_by_seed: dict[int, dict[str, str]] = {}
        unit_counts: dict[str, int] = {}
        seed_sets: dict[str, set] = {}
        lengths: dict[str, set] = {}
        for sroot in scored_roots:
            for rec_path in sroot.rglob("record.json"):
                if any(x.startswith(".pull_tmp") for x in rec_path.parts):
                    continue
                try:
                    rec = json.loads(rec_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 — counted by the readability check above
                    continue
                arm, seed = rec.get("arm"), rec.get("seed")
                if arm is None or seed is None:
                    continue
                seed = int(seed)
                key = f"{sroot.name}/{arm}-s{seed}"
                unit_counts[key] = unit_counts.get(key, 0) + 1
                seed_sets.setdefault(f"{sroot.name}/{arm}", set()).add(seed)
                rets = (rec.get("metrics", {}) or {}).get("test_returns")
                if isinstance(rets, list) and rets:
                    lengths.setdefault(str(arm), set()).add(len(rets))
                    digest = hashlib.sha1(
                        ",".join(f"{float(x):.12g}" for x in rets).encode()).hexdigest()
                    digest_by_arm.setdefault(f"{sroot.name}/{arm}", {})[seed] = digest
                rh = rec.get("reward_source_hash")
                if rh:
                    hash_by_seed.setdefault(seed, {})[str(arm)] = str(rh)
        if digest_by_arm:
            lane["seed_digests"] = digest_by_arm
        if hash_by_seed:
            lane["reward_hash_by_seed"] = hash_by_seed
        if unit_counts:
            lane["unit_counts"] = unit_counts
        if lengths:
            lane["scored_lengths"] = lengths
        if len(seed_sets) >= 2:
            lane["seed_sets"] = seed_sets
    except Exception:  # noqa: BLE001
        pass

    # AUTHORING HEALTH (2026-07-27) — the earliest observable signal in the whole campaign. Every
    # sandbox rejection is flushed to `<prefix>-<arm>.failures.jsonl` the instant it happens, and
    # candidate ids carry `-g<gen>-c<idx>`, so successes and failures can be put in TRUE order and a
    # genuine consecutive-rejection streak computed — not merely a rate, which needs far more
    # samples before it becomes conclusive.
    try:
        import re as _re

        cid = _re.compile(r"^(?P<arm>.+)-g(?P<gen>\d+)-c(?P<idx>\d+)$")
        outcomes: dict[str, dict[tuple[int, int], bool]] = {}
        for fp in sorted(camp_root.rglob("*.failures.jsonl")):
            for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    m = cid.match(str(json.loads(line).get("candidate_id", "")))
                except Exception:  # noqa: BLE001
                    continue
                if m:
                    outcomes.setdefault(m["arm"], {})[(int(m["gen"]), int(m["idx"]))] = False
        search_root = camp_root / "search"
        if search_root.is_dir():
            for rec_path in search_root.rglob("record.json"):
                try:
                    rid = str(json.loads(rec_path.read_text(encoding="utf-8"))
                              .get("candidate_id", ""))
                except Exception:  # noqa: BLE001
                    continue
                m = cid.match(rid)
                if m:
                    outcomes.setdefault(m["arm"], {})[(int(m["gen"]), int(m["idx"]))] = True
        if outcomes:
            per_arm = {}
            for arm, by_pos in outcomes.items():
                seq = [by_pos[k] for k in sorted(by_pos)]
                streak = 0
                for ok_ in reversed(seq):        # trailing run of rejections, in candidate order
                    if ok_:
                        break
                    streak += 1
                per_arm[arm] = {"accepted": sum(seq), "rejected": len(seq) - sum(seq),
                                "consecutive_rejects": streak,
                                "author": camp_root.name.replace("test_leg_", "")}
            lane["authoring_health"] = per_arm
    except Exception:  # noqa: BLE001 — monitoring must never break the gatherer
        pass

    # LIVE record sanity (2026-07-27): assess the most recent records EVERY poll, so a garbage
    # record reaches the operator in one interval instead of at the 30-seed floor two days in.
    try:
        from scripts.first_seed_sanity import assess_recent

        lane["record_sanity"] = assess_recent(camp_root)
    except Exception:  # noqa: BLE001 — a monitoring failure must never break the gatherer
        pass

    # Substrate homogeneity over the SCORED leg only — the search leg legitimately explores, but a
    # device/thread mix inside the scored leg breaks the CRN pairing every paired contrast needs.
    test_root = camp_root / "test"
    if test_root.is_dir():
        from src.cluster.integrity import env_label_census, substrate_field_census

        arm_dirs = [p for p in sorted(test_root.iterdir()) if p.is_dir()]
        census = env_label_census(arm_dirs)
        if census:
            lane["env_fp_labels"] = census
        # The label carries only `dev=`; CPU MODEL and thread regime live in env.json and are what
        # actually fix float reduction order. Measured 2026-07-28: the search leg already held a
        # Xeon Gold 6240/6140 mix, so `-ac allow=d` does not pin one microarchitecture.
        sub = substrate_field_census(arm_dirs)
        if sub:
            lane["substrate_fields"] = sub

    # FROZEN WINNERS -> the execution quality of the SEARCH candidate each came from. Selection is
    # `max(val_fitness)` and does NOT consider `train_safe_default_count`, so a reward that executed
    # on half its steps can be promoted into the sealed leg (measured 2026-07-28: two candidates at
    # 50.0% and 53.7%). The frozen winner record is a MANIFEST and carries no counters of its own,
    # so the quality must be joined back from the source candidate by `candidate_id`.
    quality: dict[tuple[str, str], tuple[int, int]] = {}
    for rec_path in camp_root.rglob("record.json"):
        parts = rec_path.relative_to(camp_root).parts
        if not parts or any(x.startswith((".pull_tmp", "_quarantined")) for x in parts):
            continue
        if not (parts[0] == "search" or parts[0].startswith("search_leg")):
            continue
        try:
            r = json.loads(rec_path.read_text(encoding="utf-8"))
            m = r.get("metrics") or {}
        except Exception:  # noqa: BLE001 - a torn record must not blind the check
            continue
        leg = parts[0].replace("search_leg_", "") if parts[0] != "search" else "core"
        quality[(leg, str(r.get("run_id")))] = (int(m.get("train_safe_default_count") or 0),
                                                int(m.get("train_safe_call_count") or 0))
    winners: dict[str, dict[str, Any]] = {}
    for wroot in sorted(camp_root.glob("frozen*")):
        if not wroot.is_dir():
            continue
        leg = wroot.name.replace("frozen_leg_", "") if wroot.name != "frozen" else "core"
        for rec_path in sorted(wroot.rglob("record.json")):
            if any(x.startswith(".pull_tmp") for x in rec_path.parts):
                continue
            try:
                r = json.loads(rec_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            cid = str(r.get("candidate_id") or "")
            dc, cc = quality.get((leg, cid), (0, 0))
            winners[leg] = {"candidate_id": cid, "safe_default_count": dc, "safe_call_count": cc}
    if winners:
        lane["frozen_winners"] = winners
    return lane


# --------------------------------------------------------------------------- #
# Rendering + CLI                                                             #
# --------------------------------------------------------------------------- #
_ICON = {OK: "OK  ", INFO: "info", WARN: "WARN", CRITICAL: "CRIT", UNKNOWN: "??  "}


def render_report(report: HealthReport) -> str:
    lines = [f"CAMPAIGN SENTINEL — {report.severity} ({'HEALTHY' if report.healthy else 'ATTENTION'})"]
    for c in sorted(report.checks, key=lambda x: -_RANK.get(x.severity, 0)):
        lines.append(f"  [{_ICON.get(c.severity, '?')}] {c.name:<14} {c.detail}")
    return "\n".join(lines)


def _emit_transitions(report: HealthReport, last: dict[str, str],
                      run_dir: Path | None = None) -> None:
    """Persist + log every check whose severity CHANGED since the last tick (precise health history).

    S18 (2026-07-06): the sentinel is a SEPARATE process, so the root-logger events.jsonl handler
    (attached only inside the RUN process by ``attach_run_logging``) never sees it — transitions
    went to the console alone and the documented "replayable health history" did not exist. They
    now ALSO append to a sentinel-owned sidecar ``<run_dir>/sentinel_events.jsonl``: a separate
    file (never the run's own events.jsonl) so there are no concurrent-append interleavings with
    the run process and no self-feedback (a WARNING-level sentinel transition must not be counted
    as a run failure by the journal's error taxonomy, which reads events.jsonl)."""
    rows: list[dict[str, Any]] = []
    for c in report.checks:
        if last.get(c.name) != c.severity:
            rows.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "sentinel_check",
                         "check": c.name, "severity": c.severity, "detail": c.detail,
                         **{k: v for k, v in c.evidence.items()}})
            last[c.name] = c.severity
    if not rows:
        return
    if run_dir is not None:
        try:
            with (Path(run_dir) / "sentinel_events.jsonl").open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass  # persistence is best-effort; the console line below still fires
    try:
        from src.utils.logging import get_logger, log_event
        import logging as _lg

        logger = get_logger("sentinel")
        level = {OK: _lg.INFO, INFO: _lg.INFO, WARN: _lg.WARNING, CRITICAL: _lg.ERROR, UNKNOWN: _lg.WARNING}
        for row in rows:
            log_event(logger, "sentinel_check", level=level.get(str(row["severity"]), _lg.INFO),
                      check=row["check"], severity=row["severity"], detail=row["detail"])
    except Exception:  # noqa: BLE001 — logging must never break the sentinel
        pass


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # the monitor runs for the WHOLE campaign (src/utils/console.py)
    ap = argparse.ArgumentParser(description="Campaign health sentinel (read-only invariant monitor).")
    ap.add_argument("run_dir", help="Campaign output dir (e.g. outputs/campaign).")
    ap.add_argument("--watch", action="store_true", help="Poll on an interval instead of a single check.")
    ap.add_argument("--interval", type=float, default=120.0, help="Seconds between polls in --watch.")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON (one object per tick).")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    last: dict[str, str] = {}
    # Streaming histories for the CUSUM drift checks (accumulate across --watch ticks).
    gate_hist: list[float] = []
    nan_hist: list[float] = []
    # (epoch_s, free_gb) samples for the predictive disk-exhaustion forecast (B3).
    disk_hist: list[tuple[float, float]] = []
    # fps samples for the cross-candidate throughput drift (B2; thermal creep / swap). The baseline
    # target FREEZES at the median of the first 10 samples ever seen (held in fps_target_hold) —
    # deriving it from the truncated rolling history would slide the baseline onto already-degraded
    # samples after ~24 h and de-arm the very multi-day creep the check exists to catch.
    fps_hist: list[float] = []
    fps_target_hold: list[float] = []

    def _tick() -> HealthReport:
        inputs = gather_inputs(run_dir)
        na, nc = int(inputs.get("n_failed", 0) or 0), int(inputs.get("n_attempted", 0) or 0)
        if nc > 0:
            gate_hist.append(na / nc)
            inputs["gate_failure_history"] = list(gate_hist)
        nf, nr = int(inputs.get("n_nonfinite", 0) or 0), int(inputs.get("n_records", 0) or 0)
        if nr > 0:
            nan_hist.append(nf / nr)
            inputs["nan_rate_history"] = list(nan_hist)
        if inputs.get("disk_free_gb") is not None:
            disk_hist.append((float(inputs.get("now") or time.time()), float(inputs["disk_free_gb"])))
            del disk_hist[:-720]  # keep ~24 h of samples at the default 2-min tick
            inputs["disk_history"] = list(disk_hist)
        if inputs.get("fps_now") is not None:
            fps_hist.append(float(inputs["fps_now"]))
            if len(fps_target_hold) < 10:
                fps_target_hold.append(float(inputs["fps_now"]))
            del fps_hist[:-720]
            # Self-calibrating baseline: the median of the run's own FIRST 10 samples EVER seen,
            # frozen in fps_target_hold (the rolling fps_hist truncates after ~24 h, so deriving
            # the target from it would slide the baseline onto degraded samples and de-arm the
            # multi-day creep detection). Only armed once the baseline exists.
            if len(fps_target_hold) >= 10:
                base = sorted(fps_target_hold)
                inputs["fps_target"] = 0.5 * (base[4] + base[5])
                inputs["fps_history"] = list(fps_hist)
        report = evaluate_health(inputs)
        _emit_transitions(report, last, run_dir)
        print(json.dumps(report.as_dict()) if args.json else render_report(report), flush=True)
        return report

    if not args.watch:
        report = _tick()
        return 1 if report.severity == CRITICAL else 0
    try:
        while True:
            _tick()
            time.sleep(max(5.0, float(args.interval)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
