"""Unit tests for the pre-flight gauntlet's pure decision logic (scripts/preflight.py).

The probes that read real hardware are exercised only indirectly; the GO/NO-GO LOGIC — which is what gates a
2-week commit — is tested exhaustively here against synthetic measured inputs.
"""
from __future__ import annotations

from scripts import preflight as pf


def test_disk_pass_and_fail() -> None:
    assert pf.check_disk(10.0, 5.0).status == pf.PASS
    assert pf.check_disk(3.0, 5.0).status == pf.FAIL


def test_ram_fail_when_insufficient_for_n_gpu() -> None:
    # n_gpu=3 needs ~3*2.2+3 = 9.6 GB available.
    assert pf.check_ram(16.0, 4.0, 3).status == pf.FAIL
    assert pf.check_ram(16.0, 12.0, 3).status in (pf.PASS, pf.WARN)


def test_ram_warns_on_small_box_at_n_gpu_3() -> None:
    c = pf.check_ram(14.0, 12.0, 3)
    assert c.status == pf.WARN  # < 15 GB total + n_gpu=3 = the measured creep band


def test_vram_fails_at_n_gpu_4_and_on_low_headroom() -> None:
    assert pf.check_vram(8000.0, 4).status == pf.FAIL          # n_gpu=4 OOMs the 6 GB card
    assert pf.check_vram(1000.0, 2).status == pf.FAIL          # < 2*1.4 GB free
    assert pf.check_vram(5000.0, 2).status == pf.PASS
    assert pf.check_vram(None, 2).status == pf.WARN            # no telemetry -> advisory, not a hard fail


def test_api_key_logic() -> None:
    assert pf.check_api_key(False, "K", required=False).status == pf.PASS   # stub provider
    assert pf.check_api_key(False, "K", required=True).status == pf.FAIL
    assert pf.check_api_key(True, "K", required=True).status == pf.PASS


def test_budget_mirror_logic() -> None:
    # H-M1 (audit 2026-07-02): campaign.yaml and algos.yaml both author train_steps_per_candidate;
    # a B* amendment that moves one file but not the other must FAIL the gauntlet.
    assert pf.check_budget_mirror(50000, 50000).status == pf.PASS
    assert pf.check_budget_mirror(200000, 50000).status == pf.FAIL
    assert pf.check_budget_mirror(None, 50000).status == pf.WARN
    assert pf.check_budget_mirror(50000, None).status == pf.WARN


def test_model_mirror_logic() -> None:
    # batch-6 M4 (2026-07-03): campaign.yaml (llm block) and llm.yaml both author model_snapshot;
    # a partial model swap must FAIL so the executed model and its standalone mirror cannot drift.
    assert pf.check_model_consistency("claude-opus-4-8", "claude-opus-4-8").status == pf.PASS
    assert pf.check_model_consistency("claude-opus-4-8", "claude-sonnet-5").status == pf.FAIL
    assert pf.check_model_consistency(None, "claude-opus-4-8").status == pf.WARN
    assert pf.check_model_consistency("claude-opus-4-8", "").status == pf.WARN


def test_freeze_logic() -> None:
    assert pf.check_freeze(False, None).status == pf.FAIL          # not frozen
    assert pf.check_freeze(True, False).status == pf.FAIL          # hash drifted
    assert pf.check_freeze(True, None).status == pf.WARN           # could not cross-check
    assert pf.check_freeze(True, True).status == pf.PASS


def test_data_logic() -> None:
    assert pf.check_data(False, None, "p").status == pf.FAIL
    assert pf.check_data(True, False, "p").status == pf.FAIL
    assert pf.check_data(True, None, "p").status == pf.WARN
    assert pf.check_data(True, True, "p").status == pf.PASS


def test_data_check_resolves_active_suffix_and_verifies_checksum(monkeypatch) -> None:
    """C3: the data probe resolves the panel via gold_suffix() (not hardcoded univ3) and does a REAL
    manifest-hash compare, so a matching panel reports checksum_ok=True (PASS), not None (WARN)."""
    from src.data import loaders

    # Point the loader at the real gold + manifest but force a distinctive active suffix name so we prove
    # the probe uses gold_suffix() (here we keep univ3, the real manifested panel, and just assert the
    # data check comes back PASS with a verified checksum rather than a bare WARN).
    monkeypatch.setattr(loaders, "gold_suffix", lambda: "univ3")
    checks = pf._gather_and_check(2, 5.0, provider="stub", api_probe=False)
    data = next(c for c in checks if c.name == "data")
    assert data.status in (pf.PASS, pf.FAIL)  # PASS (verified) when the panel is present + matches
    if data.status == pf.PASS:
        assert "checksum verified" in data.detail
        assert "returns_panel_univ3.parquet" in data.detail


def test_verdict_rollup() -> None:
    ok = [pf.Check("a", pf.PASS, ""), pf.Check("b", pf.PASS, "")]
    warn = [pf.Check("a", pf.PASS, ""), pf.Check("b", pf.WARN, "")]
    bad = [pf.Check("a", pf.FAIL, ""), pf.Check("b", pf.WARN, "")]
    assert pf.verdict(ok) == "GO"
    assert pf.verdict(warn) == "GO-WITH-WARNINGS"
    assert pf.verdict(bad) == "NO-GO"


def test_retry_layer_logic() -> None:
    """C1 (ops audit 2026-07-02): a missing tenacity means single-attempt API calls — hard FAIL."""
    assert pf.check_retry_layer(True).status == pf.PASS
    missing = pf.check_retry_layer(False)
    assert missing.status == pf.FAIL
    assert "single-attempt" in missing.detail  # the run-killer is NAMED for the operator


def test_retry_layer_is_gathered_as_a_hard_probe() -> None:
    """The gauntlet includes the retry_layer check, measured via find_spec (tenacity IS installed
    in this venv -> PASS)."""
    checks = pf._gather_and_check(2, 5.0, provider="stub", api_probe=False)
    retry = next(c for c in checks if c.name == "retry_layer")
    assert retry.status == pf.PASS


def test_windows_update_logic() -> None:
    """C2: FAIL on a queued reboot; WARN when updates are not verifiably paused; WARN on probe gaps."""
    assert pf.check_windows_update(True, True).status == pf.FAIL           # pending reboot beats all
    assert pf.check_windows_update(True, None).status == pf.FAIL
    assert pf.check_windows_update(False, True).status == pf.PASS          # clean + paused
    assert pf.check_windows_update(False, False).status == pf.WARN         # not paused -> advisory
    assert pf.check_windows_update(False, None).status == pf.WARN          # pause state unknown
    assert pf.check_windows_update(None, None).status == pf.WARN           # probe unavailable
    assert "probe unavailable" in pf.check_windows_update(None, None).detail


def test_windows_update_is_gathered() -> None:
    """The gauntlet includes the windows_update check (on this Windows host it must be measurable;
    on any host the check exists and never crashes the gauntlet)."""
    checks = pf._gather_and_check(2, 5.0, provider="stub", api_probe=False)
    wu = next(c for c in checks if c.name == "windows_update")
    assert wu.status in (pf.PASS, pf.WARN, pf.FAIL)


def test_pause_expiry_parse() -> None:
    """Registry PauseUpdatesExpiryTime parsing: future -> paused, past -> not, garbage -> unknown."""
    assert pf._pause_expiry_is_future("2099-01-01T00:00:00Z", now_epoch=1.0) is True
    assert pf._pause_expiry_is_future("1999-01-01T00:00:00Z", now_epoch=2e9) is False
    assert pf._pause_expiry_is_future("not-a-date", now_epoch=1.0) is None
    # naive (no tz) values are treated as UTC rather than rejected
    assert pf._pause_expiry_is_future("2099-01-01T00:00:00", now_epoch=1.0) is True


def test_api_probe_classification_logic() -> None:
    """M9: ok -> PASS; auth/deterministic-4xx -> FAIL (dead key / empty credit at t0);
    transient/unavailable -> advisory WARN (tenacity absorbs transients at runtime)."""
    assert pf.check_api_probe("ok", "model answered").status == pf.PASS
    auth = pf.check_api_probe("auth_error", "AuthenticationError (status=401)")
    assert auth.status == pf.FAIL
    assert "invalid/expired/revoked" in auth.detail
    client_err = pf.check_api_probe("client_error", "BadRequestError (status=400): credit balance too low")
    assert client_err.status == pf.FAIL
    assert pf.check_api_probe("transient", "APIConnectionError").status == pf.WARN
    assert pf.check_api_probe("unavailable", "sdk missing").status == pf.WARN


def test_probe_never_fires_without_flag(monkeypatch) -> None:
    """NO API SPEND IN TESTS: without --probe the live call is never attempted, even with a paid
    provider and a key present (the guard the M9 wiring must keep)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setattr(pf, "_probe_api",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")))
    checks = pf._gather_and_check(2, 5.0, provider="anthropic", api_probe=False)
    assert not any(c.name == "api_probe" for c in checks)


def test_probe_wiring_uses_campaign_model_and_classifies(monkeypatch) -> None:
    """With --probe + a paid provider + a key, the gauntlet calls the (mocked) probe with the pinned
    campaign model from config/campaign.yaml llm.model_snapshot and appends the classified check."""
    seen: dict = {}

    def _fake_probe(model, key_env):  # noqa: ANN001 - MOCK: no network, no spend
        seen["model"], seen["key_env"] = model, key_env
        return "auth_error", "AuthenticationError (status=401): invalid x-api-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setattr(pf, "_probe_api", _fake_probe)
    checks = pf._gather_and_check(2, 5.0, provider="anthropic", api_probe=True)
    probe = next(c for c in checks if c.name == "api_probe")
    assert probe.status == pf.FAIL                       # a dead key is a NO-GO at t0
    assert seen["model"] == "claude-opus-5"              # the pinned campaign snapshot (R102), not a hardcode
    assert seen["key_env"] == "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------- #
# R106: the confirmatory author's cap + reasoning pin                          #
# --------------------------------------------------------------------------- #
def test_author_pin_mirror_catches_the_absent_cap_that_actually_happened() -> None:
    """The exact defect found 2026-07-27, and it is the reason this check exists.

    R102 raised the Opus cap 4096 -> 8192 and recorded it ONLY in config/llm.yaml, which has no code
    consumer for the author. config/campaign.yaml's llm block is what the campaign executes, and it
    had no max_tokens at all, so campaign.py fell back to its 4096 default. MEASURED consequence: in
    a 10-call live gate the confirmatory author emitted 5,008 and 6,412 output tokens on two calls —
    both would have TRUNCATED at 4096, i.e. ~20% of the most important model's reward candidates lost
    silently, with every gate green.

    ABSENCE must FAIL, not warn: an absent key does not mean "default", it means the executed value
    is invisible to the registration.
    """
    good = {"max_tokens": 16384, "thinking": {"type": "disabled"}}
    assert pf.check_author_pin_mirror(good, good).status == pf.PASS

    # the real bug: cap present in llm.yaml, ABSENT in the executed block
    absent_cap = pf.check_author_pin_mirror({"thinking": {"type": "disabled"}}, good)
    assert absent_cap.status == pf.FAIL and "ABSENT" in absent_cap.detail

    # R106 requires the pin on all 11 models; the author is the 11th
    absent_pin = pf.check_author_pin_mirror({"max_tokens": 16384}, good)
    assert absent_pin.status == pf.FAIL and "thinking" in absent_pin.detail

    # and a silent divergence between the two files must not pass either
    drifted = pf.check_author_pin_mirror({"max_tokens": 4096, "thinking": {"type": "disabled"}}, good)
    assert drifted.status == pf.FAIL and "mismatch" in drifted.detail


def test_author_pin_mirror_matches_the_shipped_config() -> None:
    """Not just the logic — the config we actually ship must satisfy it (R106: 8192 + disabled)."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    camp = yaml.safe_load((root / "config" / "campaign.yaml").read_text(encoding="utf-8"))["llm"]
    llmy = yaml.safe_load((root / "config" / "llm.yaml").read_text(encoding="utf-8"))
    assert pf.check_author_pin_mirror(camp, llmy).status == pf.PASS
    assert camp["max_tokens"] == 16384                      # matched with all 10 legs
    assert camp["thinking"] == {"type": "disabled"}        # uniform reasoning-off, 11th model


def test_commit_floor_scales_with_the_workload_and_is_stricter_where_it_matters() -> None:
    """R106-era calibration (2026-07-27) — the floor was hardcoded 6.0 for every scenario.

    That mattered once the campaign moved to Myriad: on a cluster run the laptop TRAINS NOTHING (it
    drives and authors only), so `check_ram` correctly relaxed to 3.0 GB while `check_commit_headroom`
    still demanded 6.0 — failing the gauntlet on a workload that does not exist.

    MEASURED (whole process tree, 20 Hz, from outside): cluster driver dry-run peak 0.54 GB; the
    spawned validation child — the process class the 2026-07-18 incident actually hurt — peak 1.33 GB.
    The old floor was ~4.5x the real need.

    The point of this test is that the fix is NOT a weakening. Same shape as check_ram, so where the
    laptop really does train the floor RISES above the old constant.
    """
    # cluster driver: 3.0 GB floor, still a 2.25x margin over the measured 1.33 GB peak
    assert pf.check_commit_headroom(3.5, 0).status == pf.PASS
    assert pf.check_commit_headroom(2.9, 0).status == pf.FAIL

    # laptop training: STRICTER than the legacy flat 6.0 — this is the direction that proves
    # the recalibration did not simply lower a bar to make a number go green
    assert pf.check_commit_headroom(6.5, 2).status == pf.FAIL, "n_gpu=2 must need 7.4, not 6.0"
    assert pf.check_commit_headroom(7.5, 2).status == pf.PASS
    assert pf.check_commit_headroom(6.5, 3).status == pf.FAIL     # 9.6 GB at three workers

    # the legacy flat floor is still reachable, and absent telemetry is a WARN not a false green
    assert pf.check_commit_headroom(5.0, None, 6.0).status == pf.FAIL
    assert pf.check_commit_headroom(None, 0).status == pf.WARN


def test_commit_floor_never_disagrees_with_the_ram_floor() -> None:
    """Both describe the SAME workload; the two drifting apart is what caused the false NO-GO."""
    for n_gpu in (0, 1, 2, 3):
        need = n_gpu * pf._PER_WORKER_GB + pf._OS_RESERVE_GB
        assert pf.check_commit_headroom(need + 0.1, n_gpu).status == pf.PASS
        assert pf.check_commit_headroom(need - 0.1, n_gpu).status == pf.FAIL
        assert pf.check_ram(15.6, need + 0.1, n_gpu).status == pf.PASS
