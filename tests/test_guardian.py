"""Unit tests for the auto-guardian's pure decision logic + the cooperative thermal pause (no real sleeps)."""
from __future__ import annotations

import pytest

from src.utils.guardian import ThermalGovernor, should_pause, should_throttle_ram


def test_should_pause_hysteresis() -> None:
    # Not yet paused: only trip at/above hi.
    assert should_pause(70.0, hi=88.0, lo=80.0, paused=False) is False
    assert should_pause(88.0, hi=88.0, lo=80.0, paused=False) is True
    # Already paused: stay paused until BELOW lo (hysteresis — no flapping around one threshold).
    assert should_pause(82.0, hi=88.0, lo=80.0, paused=True) is True
    assert should_pause(79.9, hi=88.0, lo=80.0, paused=True) is False
    # No telemetry never pauses.
    assert should_pause(None, hi=88.0, lo=80.0, paused=False) is False


def test_should_throttle_ram() -> None:
    assert should_throttle_ram(80.0, danger=92.0) is False
    assert should_throttle_ram(92.0, danger=92.0) is True
    assert should_throttle_ram(None, danger=92.0) is False


def test_governor_pauses_until_cool_then_resumes() -> None:
    temps = iter([90.0, 85.0, 79.0])  # entry hot, stays hot one tick, then cools below lo
    slept: list[float] = []
    gov = ThermalGovernor(
        hi=88.0, lo=80.0, poll_secs=5.0,
        read_temp=lambda: next(temps), sleep=slept.append, log=lambda _m: None,
    )
    waited = gov.govern()
    assert gov.pause_events == 1
    assert waited == 10.0            # two 5 s polls before it cooled past lo
    assert gov.total_paused_secs == 10.0
    assert slept == [5.0, 5.0]       # no real time spent (injected sleep)


def test_governor_no_pause_when_cool() -> None:
    gov = ThermalGovernor(read_temp=lambda: 70.0, sleep=lambda _s: None)
    assert gov.govern() == 0.0
    assert gov.pause_events == 0


def test_governor_respects_max_pause_cap() -> None:
    # Stuck hot forever -> bounded by max_pause_secs so it can never wedge the run.
    gov = ThermalGovernor(hi=88.0, lo=80.0, poll_secs=10.0, max_pause_secs=30.0,
                          read_temp=lambda: 95.0, sleep=lambda _s: None)
    waited = gov.govern()
    assert waited == 30.0


def test_governor_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        ThermalGovernor(hi=80.0, lo=88.0)


def test_trainer_make_governor_config_gating() -> None:
    """The trainer builds a thermal governor ONLY when configured; default off -> result-neutral wiring."""
    from src.agents.trainer import _make_governor

    assert _make_governor({}) is None
    assert _make_governor({"thermal_guardian": None}) is None
    gov = _make_governor({"thermal_guardian": {"hi": 90.0, "lo": 82.0, "poll_secs": 3.0}})
    assert gov is not None
    assert gov.hi == 90.0 and gov.lo == 82.0 and gov.poll_secs == 3.0


def test_campaign_config_pins_thermal_guardian_and_governor_builds_from_it() -> None:
    """M6 (ops audit 2026-07-02): config/campaign.yaml's agent block carries a thermal_guardian the
    trainer's _make_governor actually accepts (hi > lo — ThermalGovernor rejects inverted thresholds),
    so the 24/7 laptop run is governed on every path that reads the campaign agent block."""
    from src.agents.trainer import _make_governor
    from src.utils.config import load_config

    agent = load_config("campaign")["agent"]
    tg = agent["thermal_guardian"]
    assert float(tg["lo"]) < float(tg["hi"])
    gov = _make_governor(agent)
    assert gov is not None
    assert gov.hi == float(tg["hi"]) and gov.lo == float(tg["lo"])


def test_thermal_guardian_threads_through_parallel_spec() -> None:
    """M6: parallel._spec forwards opts['thermal_guardian'] into every SEARCH worker spec (and the
    worker hands it to make_agent_trainer's cfg), so the parallel SEARCH leg is governed like the
    serial/TEST paths; absent -> None (governor off, behaviour unchanged)."""
    from src.orchestration.parallel import _spec

    base_opts = {
        "train_steps": 10, "batch_size": 4, "normalize_obs": True, "n_trials": 2,
        "synthetic": True, "data": {}, "cvar_alpha": 0.05, "window": 20, "seed": 0,
    }
    tg = {"hi": 88, "lo": 80}
    spec = _spec("distributional", "source", "src", "cid", {**base_opts, "thermal_guardian": tg})
    assert spec["thermal_guardian"] == tg
    assert _spec("distributional", "source", "src", "cid", base_opts)["thermal_guardian"] is None


def test_thermal_guardian_threads_through_build_parallel_opts() -> None:
    """M6: the shared opts builder lifts agent.thermal_guardian out of the structural config, so the
    campaign's _search_parallel_arm (which passes the RESOLVED campaign agent block) reaches the
    worker with the governor config intact."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_prototype import build_parallel_opts  # type: ignore[import-not-found]

    tg = {"hi": 88, "lo": 80}
    structural = {"agent": {"batch_size": 4, "thermal_guardian": tg}, "reward_family": {}, "data": {}}
    opts = build_parallel_opts(
        structural, {"universe": {"n_assets": 3}}, llm_block={}, train_steps=10, n_trials=2,
        synthetic=True, seed=0, candidates=2, generations=1, pass_mode="A", provider="stub",
    )
    assert opts["thermal_guardian"] == tg
    # absent -> None (off): a config without the key is byte-identical behaviour.
    structural_off = {"agent": {"batch_size": 4}, "reward_family": {}, "data": {}}
    opts_off = build_parallel_opts(
        structural_off, {"universe": {"n_assets": 3}}, llm_block={}, train_steps=10, n_trials=2,
        synthetic=True, seed=0, candidates=2, generations=1, pass_mode="A", provider="stub",
    )
    assert opts_off["thermal_guardian"] is None
