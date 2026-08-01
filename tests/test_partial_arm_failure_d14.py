"""D14 — a PARTIAL core-arm failure must stop the pass BEFORE the CRN-paired H2 array.

Regression for the campaign execution record §25 and the deferred-fix register item 4.

**The asymmetry that made this dangerous.** `_arm_core`'s `except` is deliberate — one unit must
not sink the ladder. When EVERY arm crashes the line exits, the supervisor logs a non-zero rc and
the watchdog revives it; six lines did exactly that on launch night and all recovered. When only
SOME arms crash, the survivors carry the process onward: nothing exits, nothing revives, and the
dead arms are stranded for the life of the run. `nemotron-3-super` ran 8 h 29 m with 3 of its 5
arms that way. **The louder failure was the safe one.**

**And the damage is scientific, not merely wasted compute.** The statement immediately after the
arm drain builds the H2 pair test from `winners`, as ONE `interleave=True` CRN-paired array. A
crashed arm has no winner, so it would be SILENTLY ABSENT — every seed in that array paired against
a comparator set that is not the registered one. `test_*_pair_test_is_not_submitted_*` is the test
that pins this, and it is the reason the fix stops the pass rather than merely logging.

**The fix must be NARROW, and `test_a_designed_no_winner_does_not_stop_the_line` is the control.**
A `no_winner`, an R115 ineligibility, or a canary-gated arm is a legitimate RESULT of the
experiment, not a fault. Stopping on those would turn an ordinary scientific outcome into an
operational alarm and would loop the line forever. Only an arm that RAISED (carrying an `error`
key) stops anything.
"""
from __future__ import annotations

import json

import src.cluster.campaign as campaign_mod
from src.cluster.campaign import run_campaign_tiered
from tests.test_cluster_campaign import (
    FakeCluster,
    _inject_select_freeze,
    _opts,
    _run,
    _test_leg_kwargs,
)

ARMS = ["distributional", "scalar"]
H2 = ("distributional", "scalar")


def _tiered(run, **kw):
    defaults = dict(
        test_leg_kwargs=_test_leg_kwargs(), h2_arms=H2,
        baseline_names=["differential_sharpe"], canary_baselines=["differential_sharpe"],
        review_gate=False,
    )
    defaults.update(kw)
    return run_campaign_tiered(
        ARMS, lambda a: _opts(generations=1, candidates=2),
        {"mode": "tiered", "tiers": [2, 4]}, run, **defaults)


def _marker(tmp_path, run):
    return tmp_path / f"ARM_CRASH_{run.line_tag()}.json"


# --------------------------------------------------------------------------- #
# THE DEFECT                                                                   #
# --------------------------------------------------------------------------- #
def test_a_crashed_arm_stops_the_pass_and_is_reported(tmp_path, monkeypatch) -> None:
    fake = FakeCluster(tmp_path)
    run = _run(tmp_path, fake)
    _inject_select_freeze(run)

    real = campaign_mod.run_search_arm

    def _boom(arm, opts, r, **kw):
        if arm == "scalar":
            raise RuntimeError("simulated provider fault")
        return real(arm, opts, r, **kw)

    monkeypatch.setattr(campaign_mod, "run_search_arm", _boom)
    out = _tiered(run, frozen_root=tmp_path / "frozen")

    assert out["ok"] is False, "a pass that lost an arm must not report success"
    assert "scalar" in out["arm_crash"]
    assert "RuntimeError" in out["arm_crash"]["scalar"]


def test_the_pair_test_is_not_submitted_with_a_missing_arm(tmp_path, monkeypatch) -> None:
    """THE SCIENCE PROPERTY. The H2 pair array is CRN-paired across arms; running it with one
    arm absent pairs every seed against the wrong comparator set."""
    fake = FakeCluster(tmp_path)
    run = _run(tmp_path, fake)
    _inject_select_freeze(run)
    real = campaign_mod.run_search_arm
    monkeypatch.setattr(campaign_mod, "run_search_arm",
                        lambda arm, opts, r, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
                        if arm == "scalar" else real(arm, opts, r, **kw))
    _tiered(run, frozen_root=tmp_path / "frozen")
    submitted = {c[0] for c in fake.calls}
    assert "h2_pair_test" not in submitted, (
        "the pair array was submitted while an arm was missing — this is the CRN-pairing defect "
        f"D14 exists to prevent; submitted batches were {sorted(submitted)}")
    assert not any(n.startswith("sweep_t") for n in submitted), \
        "the C4 sweep must not run on top of an incomplete C1"


def test_the_crash_writes_a_line_qualified_marker(tmp_path, monkeypatch) -> None:
    """A CRITICAL line in a multi-hundred-megabyte driver log is not an alarm; a file is.

    The name MUST carry the line tag: `read_root` is shared by all twelve supervised lines, so an
    unqualified marker would have twelve writers racing one path and would name the wrong line.
    """
    fake = FakeCluster(tmp_path)
    run = _run(tmp_path, fake)
    _inject_select_freeze(run)
    real = campaign_mod.run_search_arm
    monkeypatch.setattr(campaign_mod, "run_search_arm",
                        lambda arm, opts, r, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
                        if arm == "scalar" else real(arm, opts, r, **kw))
    _tiered(run, frozen_root=tmp_path / "frozen")

    m = _marker(tmp_path, run)
    assert m.exists(), f"no ARM_CRASH marker at {m}; the monitoring cycle would stay blind"
    payload = json.loads(m.read_text(encoding="utf-8"))
    assert "scalar" in payload["arms"] and payload["line"] == run.line_tag()


def test_a_clean_pass_clears_a_stale_marker(tmp_path) -> None:
    """The alarm must not be sticky, or the next clean run reports a defect that no longer exists."""
    fake = FakeCluster(tmp_path)
    run = _run(tmp_path, fake)
    _inject_select_freeze(run)
    m = _marker(tmp_path, run)
    m.write_text('{"stale": true}', encoding="utf-8")
    out = _tiered(run, frozen_root=tmp_path / "frozen")
    assert out["ok"]
    assert not m.exists()


# --------------------------------------------------------------------------- #
# THE NARROWNESS CONTROL — a fix that stops on everything is not a fix       #
# --------------------------------------------------------------------------- #
def test_a_designed_no_winner_does_not_stop_the_line(tmp_path, monkeypatch) -> None:
    """`no_winner` / R115 ineligibility is a RESULT, not a fault.

    Without this control the fix would halt a line on an ordinary scientific outcome and relaunch
    it forever. The discriminator is the `error` key, which only the `except` path sets.
    """
    fake = FakeCluster(tmp_path)
    run = _run(tmp_path, fake)
    _inject_select_freeze(run)
    real_sel = campaign_mod._select_eligible_winner
    monkeypatch.setattr(
        campaign_mod, "_select_eligible_winner",
        lambda sel, path, arm: (None, "r115_default_rate") if arm == "scalar"
        else real_sel(sel, path, arm))

    out = _tiered(run, frozen_root=tmp_path / "frozen")
    assert "arm_crash" not in out, "a designed no-winner must not be reported as a crash"
    assert not _marker(tmp_path, run).exists()
    # and the line KEEPS GOING — the surviving arm's pair test still runs
    assert "h2_pair_test" in {c[0] for c in fake.calls}
