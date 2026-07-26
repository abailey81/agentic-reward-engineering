"""Campaign-lane health checks — the five SILENT failure modes the sentinel cannot see.

Every one of these is a failure where the machine looks fine, the test flood keeps streaming, and
every existing check stays green while the campaign quietly loses days or validity. The tests are
written around that: each asserts the check FIRES on the silent case and stays quiet otherwise.
"""
from __future__ import annotations

import pytest

from src.cluster.campaign_health import (
    check_capacity_accumulation,
    check_chain_progress,
    check_determinism_homogeneity,
    check_host_failure_concentration,
    check_rung_forecast,
)


# --- 1. capacity accumulation ----------------------------------------------------------------

def test_climbing_capacity_is_OK_and_forbids_premature_reforecast():
    c = check_capacity_accumulation({"status": "climbing", "late_mean_cores": 1500},
                                    expected_cores=2500, hours_in=1.5)
    assert c.severity == "OK" and "do NOT re-forecast" in c.detail


def test_early_plateau_does_NOT_cry_wolf():
    """Accumulation legitimately takes ~2-3 h; a low reading at 1 h is expected, not a fault."""
    c = check_capacity_accumulation({"status": "plateaued", "late_mean_cores": 400},
                                    expected_cores=2500, hours_in=1.0)
    assert c.severity == "INFO" and "too early" in c.detail


def test_a_LOW_plateau_after_the_grace_window_WARNS_and_says_what_to_do():
    """THE plan-changing case: stuck near the probe floor, so the rung must be re-forecast."""
    c = check_capacity_accumulation({"status": "plateaued", "late_mean_cores": 640},
                                    expected_cores=2500, hours_in=6.0)
    assert c.severity == "WARN"
    assert "RE-FORECAST" in c.detail and "26%" in c.detail


def test_a_healthy_plateau_is_OK():
    c = check_capacity_accumulation({"status": "plateaued", "late_mean_cores": 2300},
                                    expected_cores=2500, hours_in=6.0)
    assert c.severity == "OK"


def test_declining_capacity_warns_and_names_the_likely_causes():
    c = check_capacity_accumulation({"status": "declining", "late_mean_cores": 200},
                                    expected_cores=2500, hours_in=8.0)
    assert c.severity == "WARN" and "kill event" in c.detail


def test_insufficient_data_is_INFO_not_a_false_alarm():
    assert check_capacity_accumulation({"status": "insufficient"}, expected_cores=2500,
                                       hours_in=0.5).severity == "INFO"


def test_NO_declared_forecast_reports_the_measurement_instead_of_warning_on_every_run():
    """Judging a plateau against a zero forecast would fire a WARN on every healthy campaign."""
    c = check_capacity_accumulation({"status": "plateaued", "late_mean_cores": 900},
                                    expected_cores=0, hours_in=9.0)
    assert c.severity == "INFO" and "900" in c.detail and "no forecast recorded" in c.detail


# --- 2. critical-path chain progress ---------------------------------------------------------

def test_advancing_chains_are_OK():
    c = check_chain_progress({"bayes_opt": {"completed": 7, "total": 25, "hours_since_last": 3.0},
                              "tpe": {"completed": 5, "total": 20, "hours_since_last": 2.5}})
    assert c.severity == "OK" and "advancing" in c.detail


def test_a_STALLED_chain_is_caught_even_though_everything_else_looks_green():
    """The worst silent failure: the test flood streams on, the makespan floor slips daily."""
    c = check_chain_progress({"bayes_opt": {"completed": 9, "total": 25, "hours_since_last": 18.0}})
    assert c.severity == "WARN"
    assert "STALLED" in c.detail and "not the test flood" in c.detail


def test_a_long_stall_escalates_to_CRITICAL():
    c = check_chain_progress({"bayes_opt": {"completed": 9, "total": 25, "hours_since_last": 40.0}})
    assert c.severity == "CRITICAL"


def test_a_COMPLETED_chain_is_not_reported_as_stalled():
    """A finished chain has no 'last progress' by definition — it must not alarm forever."""
    c = check_chain_progress({"bayes_opt": {"completed": 25, "total": 25,
                                            "hours_since_last": 99.0}})
    assert c.severity == "OK"


# --- 3. bad-node concentration ---------------------------------------------------------------

def test_a_single_bad_node_is_caught_despite_a_tiny_GLOBAL_failure_rate():
    """node-d00a-230 really did this: no apptainer -> rc=127 on every task routed to it. A 1%
    global rate passes every aggregate check while the host keeps eating work."""
    attempts = {f"node-d00a-{i}": 100 for i in range(40)}
    attempts["node-d00a-230"] = 40
    failures = {"node-d00a-230": 40}
    c = check_host_failure_concentration(failures, attempts)
    assert c.severity == "WARN"
    assert "node-d00a-230" in c.detail and "exclude them" in c.detail
    # the global rate is ~1%, which no aggregate check would flag
    assert sum(failures.values()) / sum(attempts.values()) < 0.02


def test_failures_spread_thinly_do_NOT_flag_a_bad_node():
    attempts = {f"n{i}": 50 for i in range(20)}
    failures = {f"n{i}": 2 for i in range(20)}
    assert check_host_failure_concentration(failures, attempts).severity == "OK"


def test_a_host_with_too_few_attempts_is_not_condemned():
    """One failure on a host's first task is noise, not evidence."""
    c = check_host_failure_concentration({"nX": 1}, {"nX": 1, "nY": 200})
    assert c.severity == "OK"


def test_epilogue_rows_aggregate_into_the_bad_node_detectors_input():
    """End-to-end from the ledger line the jobscript actually writes to the fired check."""
    from src.cluster.ledger import host_task_counts

    rows = ([{"task": i, "host": "node-d00a-230", "rc": 127, "secs": 1} for i in range(10)]
            + [{"task": i, "host": "node-e00a-001", "rc": 0, "secs": 30_000} for i in range(50)])
    attempts, failed = host_task_counts(rows)
    assert attempts == {"node-d00a-230": 10, "node-e00a-001": 50}
    assert failed == {"node-d00a-230": 10}
    assert check_host_failure_concentration(failed, attempts).severity == "WARN"


def test_host_counts_skip_unusable_rows_instead_of_inventing_a_phantom_node():
    """A torn/hostless row must not become a bucket — a fake key would read as a bad node."""
    from src.cluster.ledger import host_task_counts

    attempts, failed = host_task_counts(
        [{"task": 1, "rc": 1}, {"task": 2, "host": "", "rc": 1},
         {"task": 3, "host": "nA", "rc": "?"}, {"task": 4, "host": "nA", "rc": 0}])
    assert attempts == {"nA": 2}
    assert failed == {}          # an UNPARSEABLE rc is unknown, never counted as a failure


# --- 4. rung forecast ------------------------------------------------------------------------

RUNGS = {30: 3_930, 100: 8_900, 189: 15_219, 279: 21_609, 340: 25_940, 403: 30_413, 568: 42_128}


def test_forecast_names_the_reachable_rung_and_the_shortfall_to_the_next():
    """6,000 done in 48 h = 125/h; 96 h left -> ~18,000 => rung 189, and 279 is 3,609 short."""
    c = check_rung_forecast(completed_trainings=6_000, elapsed_hours=48.0,
                            hours_remaining=96.0, rung_targets=RUNGS)
    assert c.evidence["rate_per_hour"] == pytest.approx(125.0)
    assert c.evidence["projected"] == 18_000
    assert c.evidence["reachable_rung"] == 189 and c.evidence["next_rung"] == 279
    assert "next rung 279 needs 3,609 more" in c.detail


def test_forecast_reports_the_TOP_rung_when_on_pace_for_it():
    c = check_rung_forecast(completed_trainings=20_000, elapsed_hours=100.0,
                            hours_remaining=300.0, rung_targets=RUNGS)
    assert c.evidence["reachable_rung"] == 568 and c.severity == "OK"


def test_forecast_stays_quiet_before_any_completion():
    assert check_rung_forecast(completed_trainings=0, elapsed_hours=0.0,
                               hours_remaining=700.0, rung_targets=RUNGS).severity == "INFO"


# --- 5. determinism homogeneity --------------------------------------------------------------

def test_a_homogeneous_scored_leg_is_OK():
    c = check_determinism_homogeneity({"envA|dev=cpu": 500})
    assert c.severity == "OK" and c.evidence["n_records"] == 500


def test_a_DEVICE_MIX_in_the_scored_leg_is_CRITICAL_not_a_warning():
    """This is a VALIDITY failure, not a slowdown: a substrate mix confounds every paired
    contrast, and the post-hoc S6 audit would find it far too late to re-run."""
    c = check_determinism_homogeneity({"envA|dev=cpu": 480, "envA|dev=cuda": 20})
    assert c.severity == "CRITICAL"
    assert "not bit-identical" in c.detail and "Quarantine" in c.detail


def test_a_THREAD_regime_mix_is_caught_too():
    c = check_determinism_homogeneity({"omp1|dev=cpu": 300, "omp8|dev=cpu": 5})
    assert c.severity == "CRITICAL"


def test_capture_FAILURES_are_distinguished_from_a_real_substrate_mix():
    """A failed fingerprint capture is an instrumentation problem, not a science problem — the
    operator must not be sent chasing a phantom device mix."""
    c = check_determinism_homogeneity({"envA|dev=cpu": 100, "capture-failed:envA": 3})
    assert c.severity == "CRITICAL"
    assert "capture FAILURES" in c.detail


# --- 6. admin-kill verdict --------------------------------------------------------------------

def test_an_ADMIN_KILL_is_CRITICAL_and_says_RETREAT_not_fight():
    """The standing priority: keeping Myriad access outranks throughput. The correct response
    is to stop submitting, not to re-submit into an administrative kill."""
    from src.cluster.campaign_health import check_admin_kill

    c = check_admin_kill({"classification": "admin_kill", "action": "retreat",
                          "reason": "18 deaths / 9 hosts in 300s", "n_deaths": 18, "n_hosts": 9,
                          "n_undated": 0})
    assert c.severity == "CRITICAL"
    assert "RETREAT" in c.detail and "do NOT" in c.detail


def test_ordinary_node_failures_do_NOT_alarm():
    from src.cluster.campaign_health import check_admin_kill

    c = check_admin_kill({"classification": "node_failure", "action": "requeue", "n_deaths": 3,
                          "n_hosts": 1, "n_undated": 0})
    assert c.severity == "INFO" and "not an administrative kill" in c.detail


def test_UNDATED_rows_warn_because_the_burst_window_cannot_see_them():
    """A row with no usable ts is invisible to the 300s window, so a real kill could go undetected —
    the failure mode the epilogue `ts` fallback exists to prevent."""
    from src.cluster.campaign_health import check_admin_kill

    c = check_admin_kill({"classification": "ok", "action": "continue", "n_deaths": 5,
                          "n_hosts": 3, "n_undated": 5})
    assert c.severity == "WARN" and "no usable timestamp" in c.detail


def test_the_watcher_NEVER_claims_to_have_enforced_anything():
    """Writing the incident file blocks ALL submission until a human clears it — it can halt a
    23-day campaign, so a read-only watcher must not do it."""
    from src.cluster.campaign_health import check_admin_kill

    for kind in ("admin_kill", "node_failure", "ok"):
        c = check_admin_kill({"classification": kind, "action": "retreat", "n_deaths": 1,
                              "n_hosts": 1, "n_undated": 0})
        assert c.evidence["enforced"] is False


def test_the_kill_verdict_REACHES_the_sentinel_report():
    """The verdict was computed into the inputs dict and read by NO check — so it never reached the
    report, the severity, or the phone alert. This is the wiring lock."""
    from scripts.sentinel import evaluate_health

    report = evaluate_health({
        "exit_code": 0,
        "kill_verdict": {"classification": "admin_kill", "action": "retreat", "n_deaths": 20,
                         "n_hosts": 10, "n_undated": 0},
    })
    names = {c.name for c in report.checks}
    assert "admin_kill" in names
    assert report.severity == "CRITICAL"


# --- integration with the sentinel's report ---------------------------------------------------

def test_a_LAPTOP_run_supplies_none_of_these_inputs_and_gets_none_of_the_checks():
    """No false alarms off-cluster: absent inputs mean the lane checks simply do not appear."""
    from scripts.sentinel import evaluate_health

    names = {c.name for c in evaluate_health({"exit_code": 0}).checks}
    assert not (names & {"capacity_accumulation", "chain_progress", "host_failure_concentration",
                         "rung_forecast", "determinism_homogeneity"})


def test_the_lane_checks_REACH_the_sentinel_report_and_drive_its_severity():
    """The whole point of integrating: a lane failure must escalate the REPORT, not sit in a
    library nobody reads. A device mix has to turn the sentinel CRITICAL."""
    from scripts.sentinel import evaluate_health

    report = evaluate_health({
        "exit_code": 0,
        "accumulation_report": {"status": "plateaued", "late_mean_cores": 600},
        "expected_cores": 2500, "lane_hours_in": 6.0, "lane_hours_remaining": 400.0,
        "chain_progress": {"bayes_opt": {"completed": 3, "total": 25, "hours_since_last": 40.0}},
        "host_attempts": {"nA": 40}, "host_failures": {"nA": 40},
        "rung_targets": RUNGS, "done_test_units": 900,
        "env_fp_labels": {"e|dev=cpu": 800, "e|dev=cuda": 100},
    })
    names = {c.name for c in report.checks}
    assert {"capacity_accumulation", "chain_progress", "host_failure_concentration",
            "rung_forecast", "determinism_homogeneity"} <= names
    assert report.severity == "CRITICAL" and not report.healthy


def test_ONE_lane_input_alone_is_enough_to_activate_just_that_check():
    """The checks are independently opt-in — a partial gather must not suppress the rest."""
    from scripts.sentinel import evaluate_health

    names = {c.name for c in evaluate_health(
        {"exit_code": 0, "chain_progress": {"tpe": {"completed": 1, "total": 20,
                                                    "hours_since_last": 1.0}}}).checks}
    assert "chain_progress" in names and "rung_forecast" not in names


def test_no_fingerprints_yet_is_INFO():
    assert check_determinism_homogeneity({}).severity == "INFO"
    assert check_determinism_homogeneity({"envA": 0}).severity == "INFO"


def test_the_LIVE_check_and_the_S6_GATE_key_on_the_SAME_label():
    """If the live check and the post-hoc gate normalised differently, one would lie about
    homogeneity — and the liar decides whether a paired contrast is valid."""
    from src.cluster.integrity import record_env_label

    cpu = {"env_fingerprint": {"label": "abc|dev=cpu", "env_json_sha256": "0" * 8}}
    gpu = {"env_fingerprint": {"label": "abc|dev=cuda", "env_json_sha256": "0" * 8}}
    census: dict[str, int] = {}
    for rec in [cpu] * 9 + [gpu]:
        lbl = record_env_label(rec)
        census[lbl] = census.get(lbl, 0) + 1
    assert census == {"abc|dev=cpu": 9, "abc|dev=cuda": 1}
    assert check_determinism_homogeneity(census).severity == "CRITICAL"


def test_record_env_label_survives_a_bare_string_and_a_nested_dict():
    """Older records carry a bare string; the 2026-07-13 run_one bug produced a nested dict. Neither
    may crash the gate — the census must still key deterministically."""
    from src.cluster.integrity import record_env_label

    assert record_env_label({"env_fingerprint": "legacy-label"}) == "legacy-label"
    nested = record_env_label({"env_fingerprint": {"label": {"label": "x"}}})
    assert isinstance(nested, str) and record_env_label({"env_fingerprint": {"label": {"label": "x"}}}) == nested


# --- 7. LIVE per-record garbage detection ------------------------------------------------------

def test_a_GARBAGE_record_raises_the_whole_report_to_CRITICAL():
    """The point of live tracking: a void record reaches the operator in ONE poll interval, not at
    the 30-seed floor two days in. It must drive the REPORT severity, which is what pushes."""
    from scripts.sentinel import evaluate_health

    report = evaluate_health({"exit_code": 0, "record_sanity": {
        "n_assessed": 40, "suspect": [],
        "garbage": [{"arm": "scalar", "seed": 0,
                     "reasons": ["the agent trained mostly on the neutral fallback"]}]}})
    names = {c.name for c in report.checks}
    assert "record_sanity" in names
    assert report.severity == "CRITICAL" and not report.healthy


def test_suspect_records_WARN_without_screaming():
    from src.cluster.campaign_health import check_record_sanity

    c = check_record_sanity({"n_assessed": 50, "garbage": [],
                             "suspect": [{"arm": "llm_tail", "seed": 3, "reasons": ["partial"]}]})
    assert c.severity == "WARN" and "SUSPECT" in c.detail


def test_a_clean_batch_is_OK():
    from src.cluster.campaign_health import check_record_sanity

    assert check_record_sanity({"n_assessed": 120, "garbage": [], "suspect": []}).severity == "OK"


def test_NO_records_yet_is_INFO_never_a_false_all_clear():
    from src.cluster.campaign_health import check_record_sanity

    assert check_record_sanity({"n_assessed": 0, "note": "no records yet"}).severity == "INFO"
    assert check_record_sanity(None).severity == "INFO"


# --- 8. AUTHORING HEALTH — the earliest observable layer (minutes, not hours) -------------------

def test_the_alarm_length_is_CALIBRATED_to_each_authors_measured_yield():
    """The core idea: one global threshold cannot serve both a 100%-yield author and a 25% one.
    It would scream at qwen every healthy run, and a detector that cries wolf is not read."""
    from src.cluster.campaign_health import MEASURED_AUTHORING_YIELD, _streak_alarm_length

    strong = _streak_alarm_length(MEASURED_AUTHORING_YIELD["haiku-4.5"])
    weak = _streak_alarm_length(MEASURED_AUTHORING_YIELD["qwen3.5-9b"])
    assert strong <= 3, "an author that never fails must alarm almost immediately"
    assert weak >= 20, "an author that genuinely fails 3 of 4 must not alarm on ordinary failures"
    assert weak > strong * 5


def test_a_STRONG_author_failing_twice_is_CRITICAL_within_minutes():
    """haiku authored 20/20 executable rewards today. Two straight failures is not bad luck."""
    from src.cluster.campaign_health import check_authoring_health

    c = check_authoring_health({"distributional": {"accepted": 0, "rejected": 2,
                                                   "consecutive_rejects": 2,
                                                   "author": "haiku-4.5"}})
    assert c.severity == "CRITICAL"
    assert "burns the arm" in c.detail and "haiku-4.5" in c.detail


def test_the_WEAK_author_failing_repeatedly_stays_quiet():
    """qwen3.5-9b's failures ARE the capability finding — alarming on them would be noise."""
    from src.cluster.campaign_health import check_authoring_health

    c = check_authoring_health({"distributional": {"accepted": 3, "rejected": 10,
                                                   "consecutive_rejects": 10,
                                                   "author": "qwen3.5-9b"}})
    assert c.severity == "OK"


def test_even_the_weak_author_ALARMS_once_the_streak_is_impossible():
    from src.cluster.campaign_health import check_authoring_health

    c = check_authoring_health({"a": {"accepted": 0, "rejected": 30, "consecutive_rejects": 30,
                                      "author": "qwen3.5-9b"}})
    assert c.severity == "CRITICAL"


def test_an_UNMEASURED_author_gets_the_conservative_default():
    """The core campaign's Opus arms have no measured row; they must still be guarded."""
    from src.cluster.campaign_health import check_authoring_health

    c = check_authoring_health({"scalar": {"accepted": 0, "rejected": 4,
                                           "consecutive_rejects": 4, "author": "claude-opus-5"}})
    assert c.severity == "CRITICAL"


def test_a_sustained_low_rate_WARNS_even_without_a_streak():
    """Alternating success/failure never builds a streak, but a rate far under the author's
    measured yield still means something is wrong."""
    from src.cluster.campaign_health import check_authoring_health

    c = check_authoring_health({"a": {"accepted": 2, "rejected": 10, "consecutive_rejects": 1,
                                      "author": "sonnet-5"}})
    assert c.severity == "WARN" and "below the author" in c.detail


def test_no_authoring_activity_is_INFO_not_a_false_all_clear():
    from src.cluster.campaign_health import check_authoring_health

    assert check_authoring_health({}).severity == "INFO"


# --- 9. the two failures EVERY other indicator hides -------------------------------------------

def test_a_torn_record_is_COUNTED_not_silently_skipped():
    """Every reader skips an unparseable record, and each skip is individually correct. Nothing
    counted them, so rising corruption was invisible while all other checks stayed green."""
    from src.cluster.campaign_health import check_unreadable_records

    assert check_unreadable_records(0, 500).severity == "OK"
    assert check_unreadable_records(2, 500).severity == "WARN"
    c = check_unreadable_records(9, 500)
    assert c.severity == "CRITICAL" and "atomically" in c.detail


def test_no_records_yet_is_INFO_not_a_clean_bill():
    from src.cluster.campaign_health import check_unreadable_records

    assert check_unreadable_records(0, 0).severity == "INFO"


def _peers(n=5, records=40, age=0.5):
    return {f"arm{i}": {"n_records": records, "hours_since_last": age} for i in range(n)}


def test_ONE_dead_arm_is_caught_though_the_global_rate_looks_healthy():
    """With 9 arms and 10 legs, a dead arm is ~1/19th of the flow — the campaign-wide cadence stays
    fine and its seeds are simply missing at the end."""
    from src.cluster.campaign_health import check_arm_progress_symmetry

    p = _peers()
    p["dead"] = {"n_records": 3, "hours_since_last": 26.0}
    c = check_arm_progress_symmetry(p)
    assert c.severity == "CRITICAL"
    assert "dead" in c.detail and "siblings" in c.detail


def test_a_FINISHED_arm_does_NOT_alarm():
    """Ahead of its peers and therefore idle. Alarming here would cry wolf every single campaign."""
    from src.cluster.campaign_health import check_arm_progress_symmetry

    p = _peers()
    p["finished"] = {"n_records": 95, "hours_since_last": 30.0}
    assert check_arm_progress_symmetry(p).severity == "OK"


def test_an_arm_that_is_BEHIND_but_still_producing_does_NOT_alarm():
    """Behind alone is not a fault — it may simply have started late or drawn slower nodes."""
    from src.cluster.campaign_health import check_arm_progress_symmetry

    p = _peers()
    p["catching_up"] = {"n_records": 5, "hours_since_last": 0.6}
    assert check_arm_progress_symmetry(p).severity == "OK"


def test_the_detector_is_DIFFERENTIAL_so_it_survives_a_slow_cluster():
    """A fixed 'records per hour' would fire constantly when capacity is low. Judging each arm
    against its siblings — same cluster, same hour — is robust to that."""
    from src.cluster.campaign_health import check_arm_progress_symmetry

    slow = _peers(age=6.0)                       # everything slow together = healthy
    assert check_arm_progress_symmetry(slow).severity == "OK"
    slow["dead"] = {"n_records": 1, "hours_since_last": 40.0}
    assert check_arm_progress_symmetry(slow).severity == "CRITICAL"


def test_too_few_arms_SKIPS_rather_than_guessing():
    from src.cluster.campaign_health import check_arm_progress_symmetry

    assert check_arm_progress_symmetry({"a": {"n_records": 1, "hours_since_last": 99}}).severity \
        == "INFO"


# --- 10. MEANINGLESS results: the archive looks perfect and the numbers are fiction -------------

def test_FAKE_replication_is_caught_though_every_record_looks_perfect():
    """The worst failure in the design: if seeding broke, n=568 is really n=1 and every interval is
    fiction. Those records are present, complete, finite and internally consistent — they fail no
    other check in the system."""
    from src.cluster.campaign_health import check_seed_replication

    c = check_seed_replication({"distributional": {0: "digestA", 1: "digestA", 2: "digestB"}})
    assert c.severity == "CRITICAL"
    assert "FAKE" in c.detail and "effectively 1" in c.detail


def test_genuinely_distinct_seeds_are_OK():
    from src.cluster.campaign_health import check_seed_replication

    assert check_seed_replication({"d": {0: "a", 1: "b", 2: "c"}}).severity == "OK"


def test_one_seed_alone_cannot_be_judged():
    from src.cluster.campaign_health import check_seed_replication

    assert check_seed_replication({"d": {0: "a"}}).severity == "INFO"


def test_arms_sharing_ONE_reward_source_means_the_experiment_did_not_run():
    """If two arms trained on identical reward code, the contrast between them is structurally
    zero — the manipulation never reached the model."""
    from src.cluster.campaign_health import check_arm_differentiation

    c = check_arm_differentiation({0: {"distributional": "h1", "scalar": "h1"}})
    assert c.severity == "CRITICAL"
    assert "VERIFY the fed feedback blocks" in c.detail


def test_the_collision_check_asks_to_VERIFY_rather_than_declaring_a_bug():
    """Two causes produce it — a wiring defect, or the model genuinely writing the same code from
    different feedback (a mechanism observation). The check must not pretend to tell them apart."""
    from src.cluster.campaign_health import check_arm_differentiation

    c = check_arm_differentiation({0: {"distributional": "h", "placebo": "h"}})
    assert "mechanism observation" in c.detail and "if it is not" in c.detail


def test_the_SAME_arm_repeating_its_hash_across_seeds_is_EXPECTED():
    """Every seed of an arm trains the SAME frozen winner, so an identical hash there is correct —
    alarming on it would fire on every healthy campaign."""
    from src.cluster.campaign_health import check_arm_differentiation

    c = check_arm_differentiation({0: {"distributional": "h1"}, 1: {"distributional": "h1"}})
    assert c.severity != "CRITICAL"


def test_a_duplicated_unit_would_double_count_in_every_paired_statistic():
    from src.cluster.campaign_health import check_duplicate_units

    c = check_duplicate_units({"test/scalar-s3": 2, "test/scalar-s4": 1})
    assert c.severity == "CRITICAL" and "double-count" in c.detail
    assert check_duplicate_units({"test/scalar-s4": 1}).severity == "OK"


def test_records_scored_over_DIFFERENT_windows_must_not_be_pooled():
    from src.cluster.campaign_health import check_test_window_consistency

    c = check_test_window_consistency({"distributional": {1571}, "scalar": {900}})
    assert c.severity == "CRITICAL" and "not comparable" in c.detail
    assert check_test_window_consistency({"a": {1571}, "b": {1571}}).severity == "OK"


# --- 11. the two catastrophic invariants nothing re-verified during a run ----------------------

def test_a_mid_run_design_change_is_CRITICAL():
    """freeze --check is a PRE-LAUNCH step. After launch nothing re-verified it, so an edit to a
    hash-bound file splits the campaign into records answering different questions — and you cannot
    tell afterwards which record belongs to which design."""
    from src.cluster.campaign_health import check_design_drift

    c = check_design_drift(True, "abc123def456", "zzz999yyy888")
    assert c.severity == "CRITICAL"
    assert "DIFFERENT pre-registered questions" in c.detail


def test_a_matching_hash_is_OK():
    from src.cluster.campaign_health import check_design_drift

    assert check_design_drift(True, "abc123", "abc123").severity == "OK"


def test_PRE_FREEZE_drift_is_silent_because_the_hash_legitimately_moves():
    """Before GO the canonical hash changes with every legitimate design edit. Alarming there would
    make the check pure noise from the first day."""
    from src.cluster.campaign_health import check_design_drift

    assert check_design_drift(False, None, "anything").severity == "INFO"


def test_an_UNVERIFIABLE_freeze_is_a_WARNING_not_a_pass():
    """An invariant that cannot be checked is not an invariant that holds."""
    from src.cluster.campaign_health import check_design_drift

    assert check_design_drift(True, None, "abc").severity == "WARN"


def test_silent_seed_misalignment_is_caught_before_it_eats_the_power():
    """The analysis intersects seeds without warning, so a lagging arm shrinks the effective n with
    no error and no visible change — the campaign reports 340 while the contrast ran on 240."""
    from src.cluster.campaign_health import check_seed_alignment

    c = check_seed_alignment({"a": set(range(340)), "b": set(range(240))})
    assert c.severity == "CRITICAL"
    assert "silently lost" in c.detail and c.evidence["common"] == 240


def test_a_small_lag_WARNS_rather_than_screaming():
    from src.cluster.campaign_health import check_seed_alignment

    c = check_seed_alignment({"a": set(range(100)), "b": set(range(90))})
    assert c.severity == "WARN"


def test_perfectly_aligned_arms_are_OK():
    from src.cluster.campaign_health import check_seed_alignment

    assert check_seed_alignment({"a": set(range(50)), "b": set(range(50))}).severity == "OK"


def test_arms_sharing_NO_seed_cannot_be_compared_at_all():
    from src.cluster.campaign_health import check_seed_alignment

    c = check_seed_alignment({"a": {0, 1, 2}, "b": {7, 8, 9}})
    assert c.severity == "CRITICAL" and "NO common seed" in c.detail


def test_record_sanity_names_records_by_run_id_not_a_synthesized_label() -> None:
    """Four DISTINCT search candidates all rendered as one string `scalar-s0` (found 2026-07-28).

    Search records of an arm share a seed, so `f"{arm}-s{seed}"` collapsed them into an
    indistinguishable list -- and that shape is the TEST-leg naming convention, so a search-leg
    warning read as a scored-leg one. The operator must be able to tell WHICH record to open.
    """
    from src.cluster.campaign_health import check_record_sanity

    sus = [{"run_id": "scalar-g1-c1", "arm": "scalar", "seed": 0, "reasons": ["partial fallback"]},
           {"run_id": "scalar-g1-c3", "arm": "scalar", "seed": 0, "reasons": ["partial fallback"]}]
    c = check_record_sanity({"n_assessed": 55, "garbage": [], "suspect": sus})
    assert c.severity == "WARN"
    assert "scalar-g1-c1" in c.detail and "scalar-g1-c3" in c.detail
    assert "scalar-s0" not in c.detail

    # A row without a run_id still degrades gracefully rather than raising.
    c2 = check_record_sanity({"n_assessed": 9, "garbage": [],
                              "suspect": [{"arm": "scalar", "seed": 0, "reasons": ["x"]}]})
    assert "scalar-s0" in c2.detail


# ══════════════════════════════════════════════════════════════════════════════════════════════
# SUBSTRATE HOMOGENEITY (the 2026-07-28 false CRITICAL + the real Gold 6240/6140 mix)
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_substrate_key_strips_ARM_IDENTITY_from_the_label() -> None:
    """The env label carries identity AND substrate; keying a leg-wide census on it made every
    ARM look like a distinct environment (8 arms -> "8 distinct environments", CRITICAL, with a
    remedy line telling the operator to quarantine and re-run perfectly good scored records)."""
    from src.cluster.integrity import record_substrate_key

    def k(lbl: str) -> str:
        return record_substrate_key({"env_fingerprint": {"label": lbl}})

    assert k("campaign:baseline_return_minus_cvar:test[3835,5406)|dev=cpu") == "dev=cpu"
    assert k("campaign:baseline_raw_return:test[3835,5406)|dev=cpu") == "dev=cpu"
    assert k("dev=cpu") == "dev=cpu"                      # search-leg bare form
    assert k("campaign:x:test[0,1)|dev=cuda") == "dev=cuda"
    # Missing provenance is a REAL difference and must stay visible, not silently equal a stamped one.
    assert k("sigma_pilot:r:test[3835,5406)") == "dev=<unrecorded>"
    # A capture fault is not a substrate.
    assert k("capture-failed: OSError").startswith("capture-failed")


def test_a_REAL_device_mix_is_still_CRITICAL() -> None:
    from src.cluster.campaign_health import check_determinism_homogeneity

    assert check_determinism_homogeneity({"dev=cpu": 100}).severity == "OK"
    assert check_determinism_homogeneity({"dev=cpu": 100, "dev=cuda": 3}).severity == "CRITICAL"


def test_substrate_fields_catches_a_CPU_MODEL_mix_the_device_label_cannot() -> None:
    """Measured live: the search leg held 116 records on a Xeon Gold 6240 and 1 on a 6140.

    Both are `dev=cpu`, so the label-based check is blind to it by construction.
    """
    from src.cluster.campaign_health import check_substrate_fields

    one = {"cpu=Intel(R) Xeon(R) Gold 6240 | omp=1 | torch_threads=1 | cuda=False": 132}
    assert check_substrate_fields(one).severity == "OK"

    mixed = {"cpu=Intel(R) Xeon(R) Gold 6240 | omp=8 | torch_threads=8 | cuda=False": 108,
             "cpu=Intel(R) Xeon(R) Gold 6140 | omp=8 | torch_threads=8 | cuda=False": 1}
    c = check_substrate_fields(mixed)
    assert c.severity == "CRITICAL" and "6140" in c.detail

    # A thread-regime mix is equally fatal to CRN and equally invisible to the label.
    threads = {"cpu=X | omp=1 | torch_threads=1 | cuda=False": 50,
               "cpu=X | omp=8 | torch_threads=8 | cuda=False": 2}
    assert check_substrate_fields(threads).severity == "CRITICAL"


def test_records_without_env_json_are_WARN_not_silently_OK() -> None:
    from src.cluster.campaign_health import check_substrate_fields

    c = check_substrate_fields({"<no env.json>": 12})
    assert c.severity == "WARN" and "unverifiable" in c.detail
    assert check_substrate_fields({}).severity == "INFO"


def test_sandbox_rejects_and_qdel_kills_are_NOT_blamed_on_the_NODE() -> None:
    """`node-d00a-229` was flagged 3/5 on 2026-07-28 while it had completed three full trainings.

    Its non-zero exits were two `rc=1` sandbox rejects (5 s and 20 s -- bad authored code, which
    fails on every node) and two `rc=126` from our own qdel. The remedy for this WARN is to EXCLUDE
    the host, so a false positive costs a healthy 36-core node in a capacity-bound campaign.
    """
    from src.cluster.ledger import host_task_counts

    rows = [{"host": "n1", "rc": 0}, {"host": "n1", "rc": 0}, {"host": "n1", "rc": 0},
            {"host": "n1", "rc": 1}, {"host": "n1", "rc": 1},
            {"host": "n1", "rc": 126}, {"host": "n1", "rc": 126}]
    attempts, failed = host_task_counts(rows)
    assert attempts["n1"] == 7
    assert failed.get("n1", 0) == 0, "task-level and kill exits are not node faults"


def test_a_REAL_broken_node_is_still_caught() -> None:
    """rc=127 is the measured signature of the case this detector exists for (no apptainer)."""
    from src.cluster.ledger import host_task_counts

    _, failed = host_task_counts([{"host": "bad", "rc": 127} for _ in range(5)])
    assert failed["bad"] == 5


def test_a_CONTAMINATED_frozen_winner_is_CRITICAL() -> None:
    """Selection is `max(val_fitness)` and ignores `train_safe_default_count` entirely.

    So a candidate whose authored reward raised on half its steps can be frozen as the winner, and
    the sealed leg then RE-TRAINS that reward and inherits the contamination. Measured 2026-07-28:
    two search candidates sat at 50.0% and 53.7%. This does not GATE anything -- the pre-registration
    is silent on a threshold and adding one is a dated-amendment decision -- but the event must be
    visible while the unit can still be re-run.
    """
    from src.cluster.campaign_health import check_winner_execution_quality

    c = check_winner_execution_quality({
        "qwen3_6_27b": {"candidate_id": "scalar-g1-c4",
                        "safe_default_count": 214_649, "safe_call_count": 400_000}})
    assert c.severity == "CRITICAL" and "53.7%" in c.detail


def test_clean_frozen_winners_are_OK_and_missing_counters_say_nothing() -> None:
    from src.cluster.campaign_health import check_winner_execution_quality

    clean = {f"leg{i}": {"candidate_id": f"scalar-g3-c{i}",
                         "safe_default_count": 0, "safe_call_count": 400_000} for i in range(5)}
    c = check_winner_execution_quality(clean)
    assert c.severity == "OK" and "5 frozen winner" in c.detail

    # A winner with NO counters must not be guessed at in either direction.
    assert check_winner_execution_quality(
        {"a": {"candidate_id": "x", "safe_default_count": 0, "safe_call_count": 0}}
    ).severity == "INFO"
    assert check_winner_execution_quality({}).severity == "INFO"


def test_a_MINOR_winner_contamination_still_surfaces_as_WARN() -> None:
    from src.cluster.campaign_health import check_winner_execution_quality

    c = check_winner_execution_quality({
        "haiku": {"candidate_id": "scalar-g1-c3",
                  "safe_default_count": 1_650, "safe_call_count": 400_000}})
    assert c.severity == "WARN"
