"""Behaviour tests for the read-only watcher helpers in ``scripts/monitor.py`` (silent-hang detection,
the anomaly error-tracker, LLM token/cost accounting, and the alert payload).

The dashboard is strictly READ-ONLY over on-disk run artefacts, so these tests synthesise a tiny
``progress.json`` / ``events.jsonl`` / ``anomalies.jsonl`` and assert the pure logic — no rich, no torch,
no network (``post_alert`` is never called against a real URL).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import monitor  # noqa: E402


def _write(run_dir: Path, name: str, rows: list[dict]) -> None:
    (run_dir / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _state(phase: str, *, updated: str) -> dict:
    return {
        "title": "campaign", "model": "claude-opus-4-8", "phase": phase, "updated": updated,
        "arms": {"total": 7, "done": 2, "current": "distributional"},
        "candidates": {"run_total": 210, "run_done": 60, "in_arm_total": 30, "in_arm_done": 4},
        "anomalies": {"count": 0, "recent": []},
    }


def test_state_age_and_staleness_threshold() -> None:
    now = time.time()
    fresh = _state("training", updated=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 5)))
    old = _state("training", updated=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 1200)))
    assert monitor.state_age_seconds(fresh, now) is not None and monitor.state_age_seconds(fresh, now) < 60
    assert monitor.is_stale(fresh, now, threshold=300.0) is False
    assert monitor.is_stale(old, now, threshold=300.0) is True  # 20 min silence -> silent-hang flagged


def test_terminal_and_starting_phases_are_never_stale() -> None:
    now = time.time()
    long_ago = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 99999))
    for phase in ("done", "error", "starting"):
        assert monitor.is_stale(_state(phase, updated=long_ago), now, threshold=300.0) is False
    assert monitor.is_stale(None, now, threshold=300.0) is False
    assert monitor.state_age_seconds({"updated": "not-a-timestamp"}, now) is None


def test_anomaly_counts_group_by_kind(tmp_path: Path) -> None:
    _write(tmp_path, "anomalies.jsonl", [
        {"kind": "critic_explosion", "detail": "x"},
        {"kind": "critic_explosion", "detail": "y"},
        {"kind": "ram_pressure_warn", "detail": "z"},
    ])
    counts = monitor.anomaly_counts(tmp_path)
    assert counts == {"critic_explosion": 2, "ram_pressure_warn": 1}
    assert monitor.anomaly_counts(tmp_path / "absent") == {}


def test_token_spend_and_cost_estimate(tmp_path: Path) -> None:
    _write(tmp_path, "events.jsonl", [
        {"event": "run_start"},
        {"event": "llm_call", "in_tok": 1_000_000, "out_tok": 200_000, "model": "claude-opus-4-8"},
        {"event": "llm_call", "in_tok": 500_000, "out_tok": 100_000, "model": "claude-opus-4-8"},
        {"event": "candidate_done", "fitness": 0.1},
    ])
    spend = monitor.token_spend(tmp_path)
    assert spend["calls"] == 2 and spend["costed_calls"] == 2
    assert spend["in_tok"] == 1_500_000 and spend["out_tok"] == 300_000
    # Opus 4.8 list price = $5/Mtok in, $25/Mtok out: 1.5*5 + 0.3*25 = 7.5 + 7.5 = 15.0
    assert abs(spend["usd"] - 15.0) < 1e-6


def test_price_resolution_by_substring_and_unknown_model() -> None:
    assert monitor._resolve_price("claude-sonnet-4-6") == (3.0, 15.0)
    assert monitor._resolve_price("claude-fable-5") == (10.0, 50.0)
    assert monitor._resolve_price("some-other-llm") is None
    assert monitor._resolve_price(None) is None


def test_unknown_model_tokens_counted_but_not_costed(tmp_path: Path) -> None:
    _write(tmp_path, "events.jsonl", [
        {"event": "llm_call", "in_tok": 100, "out_tok": 50, "model": "mystery-model"},
    ])
    spend = monitor.token_spend(tmp_path)
    assert spend["calls"] == 1 and spend["costed_calls"] == 0 and spend["usd"] == 0.0
    assert spend["in_tok"] == 100 and spend["out_tok"] == 50


def test_build_alert_is_status_only(tmp_path: Path) -> None:
    st = _state("done", updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    msg = monitor.build_alert(st, "done", tmp_path)
    assert "DONE" in msg and "arms 2/7" in msg and "cand 60/210" in msg
    # no progress.json -> graceful
    assert "no progress.json" in monitor.build_alert(None, "stall", tmp_path)


def test_jsonl_cache_reparses_on_change(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    _write(tmp_path, "events.jsonl", [{"event": "llm_call", "in_tok": 10, "out_tok": 1, "model": "claude-opus-4-8"}])
    assert monitor.token_spend(tmp_path)["calls"] == 1
    time.sleep(0.01)
    p.write_text(p.read_text() + json.dumps({"event": "llm_call", "in_tok": 10, "out_tok": 1, "model": "claude-opus-4-8"}) + "\n", encoding="utf-8")
    assert monitor.token_spend(tmp_path)["calls"] == 2  # cache invalidated on (mtime,size) change


# ---- m11: clock-skew-tolerant staleness (freshest of parsed stamp vs file mtime) ------------------------ #
def test_state_age_uses_freshest_of_stamp_and_mtime() -> None:
    """m11: a skewed/old writer stamp must not fabricate STALE while the file mtime proves fresh writes,
    and a fresh stamp still wins over a weird old mtime (max of the two evidences)."""
    now = time.time()
    old_stamp = _state("training", updated=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 7200)))
    # File rewritten 5 s ago (mtime) but the stamp is 2 h old (e.g. a DST/clock jump on the writer):
    assert monitor.state_age_seconds(old_stamp, now, mtime_epoch=now - 5) < 60
    assert monitor.is_stale(old_stamp, now, threshold=300.0, mtime_epoch=now - 5) is False
    # Without the mtime evidence the same state IS stale (the pre-m11 behaviour, still correct alone):
    assert monitor.is_stale(old_stamp, now, threshold=300.0) is True
    # A fresh stamp with an ancient mtime stays fresh (max, not min):
    fresh_stamp = _state("training", updated=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 5)))
    assert monitor.state_age_seconds(fresh_stamp, now, mtime_epoch=now - 7200) < 60


def test_state_age_falls_back_to_mtime_when_stamp_unparseable() -> None:
    """m11: a garbled `updated` no longer means 'unknown age' when the file mtime is available."""
    now = time.time()
    st = {"phase": "training", "updated": "not-a-timestamp"}
    assert monitor.state_age_seconds(st, now) is None                       # both evidences missing
    assert monitor.state_age_seconds(st, now, mtime_epoch=now - 10) < 60    # mtime rescues the probe
    assert monitor.is_stale(st, now, threshold=300.0, mtime_epoch=now - 1200) is True


# ---- M5a: alert lifecycle (reason -> dedupe reset on healthy -> add only after a successful post) -------- #
def test_alert_reason_maps_phase_and_stall() -> None:
    now = time.time()
    fresh = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 5))
    old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 1200))
    assert monitor.alert_reason(_state("done", updated=fresh), now, 300.0) == "done"
    assert monitor.alert_reason(_state("error", updated=fresh), now, 300.0) == "error"
    assert monitor.alert_reason(_state("training", updated=old), now, 300.0) == "stall"
    assert monitor.alert_reason(_state("training", updated=fresh), now, 300.0) is None
    assert monitor.alert_reason(None, now, 300.0) is None


def test_process_notification_adds_only_after_successful_post() -> None:
    """M5a(iii): a failed POST must be retried on the next poll — never optimistically deduped away."""
    sent: set[str] = set()
    outcomes = iter([False, True])  # first POST fails (network blip), second succeeds
    posted: list[str] = []

    def _send(reason: str) -> bool:
        posted.append(reason)
        return next(outcomes)

    assert monitor.process_notification("error", sent, _send) is False  # post failed -> NOT recorded
    assert sent == set()
    assert monitor.process_notification("error", sent, _send) is True   # retried next poll -> recorded
    assert sent == {"error"}
    assert monitor.process_notification("error", sent, _send) is False  # now deduped
    assert posted == ["error", "error"]


def test_process_notification_resets_dedupe_when_healthy_again() -> None:
    """M5a(ii): a healthy tick clears the dedupe set, so the NEXT arm's done/stall alerts fire —
    previously the first arm's 'done' suppressed every later alert for the whole campaign."""
    sent: set[str] = set()
    ok = lambda _r: True  # noqa: E731
    assert monitor.process_notification("done", sent, ok) is True   # arm 1 finishes -> alert
    assert monitor.process_notification("done", sent, ok) is False  # deduped while still 'done'
    assert monitor.process_notification(None, sent, ok) is False    # arm 2 starts (healthy) -> reset
    assert sent == set()
    assert monitor.process_notification("done", sent, ok) is True   # arm 2 finishing alerts again


# ---- M5a: --follow-campaign terminal sentinel -------------------------------------------------------- #
def test_campaign_done_requires_sentinel_written_after_watcher_start(tmp_path: Path) -> None:
    """A STALE campaign_summary.json from a previous interrupted run (mtime < watcher start) must not
    terminate the watcher; only a summary (re)written AFTER the watcher started counts."""
    assert monitor.campaign_done(None, started_epoch=100.0) is False          # no sentinel yet
    assert monitor.campaign_done(50.0, started_epoch=100.0) is False          # stale leftover
    assert monitor.campaign_done(150.0, started_epoch=100.0) is True          # written after start

    # _sentinel_mtime resolves run_dir/ then run_dir.parent/ (search dir is the documented usage).
    search = tmp_path / "search"
    search.mkdir()
    assert monitor._sentinel_mtime(search) is None
    (tmp_path / "campaign_summary.json").write_text("{}", encoding="utf-8")
    mt = monitor._sentinel_mtime(search)
    assert mt is not None and mt > 0


def test_campaign_done_keeps_watching_on_resumable_exit3_pass(tmp_path) -> None:
    """2026-07-05: a FRESH sentinel from an exit-3 (resumable) pass must NOT terminate the watcher —
    the supervisor relaunches with --resume and the remaining passes must stay watched. Terminal
    passes (exit_code==0) and legacy summaries (no key) still exit; an unreadable summary defers."""
    from scripts.monitor import _sentinel_exit_code, campaign_done

    # Fresh sentinel + non-zero exit code -> keep watching.
    assert campaign_done(100.0, 50.0, exit_code=3) is False
    # Fresh sentinel + terminal pass -> done (and the legacy default keeps old callers exiting).
    assert campaign_done(100.0, 50.0, exit_code=0) is True
    assert campaign_done(100.0, 50.0) is True
    # Unreadable summary (mid-write) -> defer to the next tick.
    assert campaign_done(100.0, 50.0, exit_code=None) is False
    # Stale sentinel never exits, whatever the code.
    assert campaign_done(10.0, 50.0, exit_code=0) is False

    # _sentinel_exit_code: real file round-trip (run_dir layout: summary one level up).
    run_dir = tmp_path / "search"
    run_dir.mkdir()
    (tmp_path / "campaign_summary.json").write_text('{"exit_code": 3}', encoding="utf-8")
    assert _sentinel_exit_code(run_dir) == 3
    (tmp_path / "campaign_summary.json").write_text('{"all_arms_tested": true}', encoding="utf-8")
    assert _sentinel_exit_code(run_dir) == 0  # legacy summary without the key
    (tmp_path / "campaign_summary.json").write_text("{not json", encoding="utf-8")
    assert _sentinel_exit_code(run_dir) is None  # mid-write -> defer


def test_alert_reason_disk_and_anomaly_rules() -> None:
    """2026-07-05 precision rules: disk-floor and anomaly-surge alert while the run is otherwise
    healthy; terminal phases keep precedence; inactive signals (None) change nothing."""
    st = {"phase": "training"}
    now = 1000.0
    # healthy + inactive extras -> None (back-compat)
    assert monitor.alert_reason(st, now, 600, mtime_epoch=now) is None
    # disk floor fires while alive
    assert monitor.alert_reason(st, now, 600, mtime_epoch=now, disk_free_gb=3.0) == "disk_low"
    assert monitor.alert_reason(st, now, 600, mtime_epoch=now, disk_free_gb=50.0) is None
    # anomaly surge fires on a >=limit per-tick jump, not on slow growth
    assert monitor.alert_reason(st, now, 600, mtime_epoch=now, anomaly_delta=25) == "anomaly_surge"
    assert monitor.alert_reason(st, now, 600, mtime_epoch=now, anomaly_delta=3) is None
    # terminal phases take precedence over everything
    assert monitor.alert_reason({"phase": "error"}, now, 600, disk_free_gb=1.0) == "error"
    assert monitor.alert_reason({"phase": "done"}, now, 600, anomaly_delta=99) == "done"


# --------------------------------------------------------------------------- #
# 2026-07-06: B6 deadman heartbeat + B7 unified sentinel line
# --------------------------------------------------------------------------- #
def test_sentinel_summary_line_healthy_and_none_on_failure(tmp_path: Path) -> None:
    """The one-line health verdict renders for an empty run dir (all probes degrade to INFO/OK)
    and NEVER raises — a broken sentinel import must yield None, not a dashboard crash."""
    line = monitor.sentinel_summary_line(tmp_path)
    # an empty dir yields a valid verdict line (severity word present) or None (sentinel unavailable)
    assert line is None or "SENTINEL" in line


def test_render_includes_sentinel_line(tmp_path: Path) -> None:
    st = {
        "title": "t", "model": "m", "pid": 1, "phase": "training", "elapsed_s": 5.0,
        "arms": {"done": 0, "total": 7, "current": "scalar", "current_idx": 0},
        "candidates": {"run_done": 1, "run_total": 210, "in_arm_done": 1, "in_arm_total": 30, "gen": 0},
        "training": {}, "resources": {}, "anomalies": {"count": 0, "recent": []},
        "best_fitness": {}, "eta_s": None,
    }
    panel = monitor.render(st, tmp_path, sentinel_line="SENTINEL WARN — completion_stall: quiet")
    # rich Panel renders; assert the line made it into the renderable tree
    from rich.console import Console

    buf = Console(record=True, width=120)
    buf.print(panel)
    assert "SENTINEL WARN" in buf.export_text()


def test_snapshot_text_appends_sentinel_line(tmp_path: Path, monkeypatch) -> None:
    st = {
        "title": "t", "model": "m", "pid": 1, "phase": "training", "elapsed_s": 5.0,
        "arms": {"done": 0, "total": 7, "current": None, "current_idx": 0},
        "candidates": {"run_done": 0, "run_total": 210, "in_arm_done": 0, "in_arm_total": 30},
        "training": {}, "resources": {}, "anomalies": {"count": 0, "recent": []},
        "best_fitness": {}, "eta_s": None, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    monkeypatch.setattr(monitor, "sentinel_summary_line", lambda _rd: "SENTINEL OK — all checks OK")
    text = monitor.snapshot_text(st, tmp_path)
    assert "SENTINEL OK" in text


def test_every_anthropic_billed_leg_resolves_to_a_price() -> None:
    """Structural lock: EVERY `provider: anthropic` leg must price in BOTH Anthropic tables.

    Added 2026-07-26 (deep review) after `claude-sonnet-5` — a seated, ANTHROPIC-BILLED leg
    (`config/legs.yaml`: provider anthropic, ANTHROPIC_API_KEY) with a registered per-leg price of
    [2.00, 10.00] — was found MISSING from `src/llm/cost.py::PRICES_PER_MTOK` and
    `scripts/monitor.py::_PRICES_PER_MTOK`. Both match by SUBSTRING, and no key was a substring of
    "claude-sonnet-5", so every sonnet-5 call booked $0.00 — under-reporting spend on the one key with a
    real funded balance in the REPORTED cost summary and on the LIVE monitor dashboard. (The R83 advisory
    ledger is a SEPARATE table -- legs.yaml::planning_prices -- which DOES price sonnet-5 and is already
    locked by test_leg_transport::test_planning_prices_cover_all_legs, so the 80%/100% spend WARNINGS are
    unaffected: this is a reporting/monitoring defect, not a spend-guard failure.) The 2026-07-24 sweep
    fixed this same class for opus-5 and missed sonnet-5; the only existing price test pins
    claude-sonnet-4-6, the leg R92 REMOVED.

    Asserted STRUCTURALLY (derived from legs.yaml) rather than as a fixed list, so seating a new
    Anthropic leg without pricing it fails here instead of silently costing $0.
    """
    import yaml

    from src.llm.cost import _price as cost_price

    legs_path = Path(__file__).resolve().parents[1] / "config" / "legs.yaml"
    legs = (yaml.safe_load(legs_path.read_text(encoding="utf-8")) or {}).get("legs") or []
    anthropic = [leg for leg in legs if str(leg.get("provider", "")).lower() == "anthropic"]
    assert anthropic, "expected at least one anthropic-billed leg in config/legs.yaml"

    unpriced = [
        (leg.get("label"), leg.get("model"))
        for leg in anthropic
        if cost_price(str(leg.get("model"))) is None
        or monitor._resolve_price(str(leg.get("model"))) is None
    ]
    assert not unpriced, (
        "these ANTHROPIC-billed legs resolve to NO price, so their calls book $0.00 and the reported "
        f"spend under-states the funded key: {unpriced}"
    )


def test_alert_reason_latches_a_vanished_state_file_instead_of_reporting_healthy() -> None:
    """A progress.json that DISAPPEARS mid-run must alert, not read as healthy.

    ``st is None`` has two meanings. Before the run writes its first state it means "not started yet"
    (correctly silent). After a state has once been read it means the file VANISHED — deleted,
    truncated, or a write that failed on a full disk — and without the latch the watcher reports the
    run healthy FOREVER, silently swallowing every later stall/error/done alert at exactly the hour
    the operator is asleep and trusting the push."""
    import time as _t

    now = _t.time()
    fresh = _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(now))

    # pre-launch: absent state is silent (unchanged legacy behaviour, default state_seen=False)
    assert monitor.alert_reason(None, now, 300.0) is None
    assert monitor.alert_reason(None, now, 300.0, state_seen=False) is None

    # mid-run disappearance: LOUD
    assert monitor.alert_reason(None, now, 300.0, state_seen=True) == "state_lost"

    # a readable state still routes by its own rules once the latch is set
    assert monitor.alert_reason(_state("training", updated=fresh), now, 300.0, state_seen=True) is None
    assert monitor.alert_reason(_state("error", updated=fresh), now, 300.0, state_seen=True) == "error"

    # the alert text distinguishes "vanished" from "not started", so the operator knows which it is
    msg = monitor.build_alert(None, "state_lost", Path("runs/campaign"))
    assert "VANISHED" in msg and "yet" not in msg
    assert "no progress.json yet" in monitor.build_alert(None, "stall", Path("runs/campaign"))

    # and it drives the notifier exactly once per episode, like every other reason
    sent: set[str] = set()
    posts: list[str] = []
    for _ in range(3):
        monitor.process_notification("state_lost", sent, lambda r: (posts.append(r), True)[1])
    assert posts == ["state_lost"]
