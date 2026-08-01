"""THE 2026-07-27 LAUNCH-GATE REGRESSION LOCKS.

Every test here pins a defect found on the day of the confirmatory launch, in code that the full
suite (2,779 passing) and every pre-launch gate had already certified green. They are collected in
one module because they share a single root cause worth naming:

    **a launch-time VALUE that no gate ever compared against its registered source.**

The freeze gate compares *config to pre-registration*. The suite compares *code to code*. Nothing
compared *the command line, the walltime, or the data path* to anything at all — so the roster
could say seven while the register said nine, the walltime could be sized off a GPU curve on a CPU
lane, and ``--gold-dir`` could point at an empty directory, with every instrument green.

Four of these would each, alone, have destroyed the campaign:

* ``autosize_h_rt`` granting 6 h to a training that needs 8.55 h -> every task SIGKILLed;
* ``_enforce_kill_switch`` reading those walltime deaths as an ADMINISTRATIVE kill -> the whole
  campaign hard-blocked behind an incident file only a human can clear;
* ``--gold-dir`` resolving to an empty directory -> every task dying in the loader;
* a seven-arm roster -> confirmatory node N4 permanently unsatisfiable.

The complementary locks in ``tests/test_run_campaign_cluster.py`` and ``tests/test_mode_d.py``
cover the same incident from the CLI and launcher sides.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import run_campaign_cluster as rcc  # noqa: E402


# ── the walltime ────────────────────────────────────────────────────────────────────────────────

def test_autosize_h_rt_cpu_covers_the_registered_training_rate() -> None:
    """THE ONE THAT WOULD HAVE KILLED EVERY JOB.

    ``autosize_h_rt`` took no ``device`` and priced everything off a GPU aggregate-throughput
    curve, returning ``6:0:0`` for a pack-4 400k task. A 400k CPU training needs 8.55 h at the
    REGISTERED 13.0 steps/s/core and 6.11 h even at the fastest rate ever observed (18.2). The
    grant must cover the REGISTERED rate, never the optimistic one.
    """
    from src.cluster.lanes import CPU_STEPS_PER_S_PER_CORE

    need_h = 400_000 / CPU_STEPS_PER_S_PER_CORE / 3600.0
    granted = rcc.autosize_h_rt(4, 400_000, device="cpu")
    assert int(granted.split(":")[0]) >= need_h, (
        f"CPU walltime {granted} does not cover the {need_h:.2f} h a 400k training needs at the "
        f"registered {CPU_STEPS_PER_S_PER_CORE} steps/s/core")
    assert granted != "6:0:0", "6:0:0 is the value that would have killed every job"


def test_autosize_h_rt_cpu_is_flat_in_pack_because_packing_is_not_time_slicing() -> None:
    """On CPU, ``pack N`` is N INDEPENDENT trainings on N OWN cores, so a task's wall is ONE
    training's wall. The GPU branch multiplies by pack (time-slicing); doing that on CPU is a
    category error which happened to mask part of the rate error rather than compensate for it."""
    vals = {rcc.autosize_h_rt(p, 400_000, device="cpu") for p in (1, 2, 4, 8)}
    assert len(vals) == 1, f"CPU walltime must not vary with pack, got {sorted(vals)}"


def test_autosize_h_rt_cuda_branch_is_unchanged() -> None:
    """The GPU lane keeps its measured curve exactly — the fix adds a branch, it does not re-tune."""
    assert [rcc.autosize_h_rt(p, 400_000) for p in (1, 2, 3, 4, 5, 8)] == [
        "4:0:0", "5:0:0", "5:0:0", "6:0:0", "7:0:0", "10:0:0"]


def test_both_cpu_walltime_estimators_share_one_constant() -> None:
    """The ladder and the campaign must size CPU walltime from the SAME object. They did not, and
    that is exactly why the lane-aware fix reached the ladder and never reached the campaign."""
    import p6_authored_ladder as ladder

    from src.cluster.lanes import CPU_PLANNING_STEPS_PER_SEC

    assert ladder.CPU_PLANNING_STEPS_PER_SEC is CPU_PLANNING_STEPS_PER_SEC


# ── the killswitch discriminator ────────────────────────────────────────────────────────────────

def _burst(secs: float, now: float, n: int = 12) -> list[dict]:
    """``n`` task deaths on ``n`` DISTINCT hosts, all inside the 300 s burst window."""
    return [{"rc": 137, "secs": secs, "host": f"node-d{i:02d}", "ts": now - 250 + i * 5}
            for i in range(n)]


def test_mass_walltime_kills_are_not_classified_as_an_administrative_kill() -> None:
    """THE COMPOUNDING FAILURE. ``_enforce_kill_switch`` called ``classify_task_deaths`` with no
    ``h_rt_secs``, and that function only applies its walltime discriminator ``if h_rt_secs:``. So
    the discriminator was DEAD and every walltime kill counted as admin-kill evidence.

    Combined with the 6 h walltime above, ~142 concurrently-dispatched tasks would all have died at
    their limit, on distinct hosts, within minutes — the exact shape that writes
    MYRIAD_KILL_INCIDENT.json and hard-blocks EVERY subsequent submission until a human clears it
    by hand. A sizing bug would have become a total, silent campaign halt that looked like a
    correctly-working safety system.
    """
    from src.cluster.killswitch import classify_task_deaths

    now = 1_800_000_000.0
    wall = _burst(53_900, now)          # died at ~their 15 h walltime
    assert classify_task_deaths(wall, now=now).action == "retreat", (
        "precondition: without h_rt_secs this burst reads as an admin kill")
    fixed = classify_task_deaths(wall, now=now, h_rt_secs=54_000.0)
    assert fixed.action == "requeue"
    assert dataclasses.astuple(fixed)[0] == "walltime"


def test_a_genuine_administrative_qdel_still_triggers_the_retreat() -> None:
    """The fix must not blunt the guard it repairs: deaths nowhere near the walltime still retreat.
    The killswitch protects continued ACCESS to Myriad, which is worth more than any single run."""
    from src.cluster.killswitch import classify_task_deaths

    now = 1_800_000_000.0
    verdict = classify_task_deaths(_burst(900, now), now=now, h_rt_secs=54_000.0)
    assert verdict.action == "retreat"
    assert dataclasses.astuple(verdict)[0] == "admin_kill"


def test_h_rt_seconds_takes_the_LARGEST_request() -> None:
    """Largest, deliberately: a larger threshold classifies FEWER deaths as walltime kills, leaving
    more on the admin-kill path — the conservative direction under the killswitch's own asymmetry
    (a false positive costs hours a human can undo; a false negative costs the account)."""
    from src.cluster.campaign import _h_rt_seconds

    assert _h_rt_seconds("6:0:0", "15:0:0") == 54_000.0
    assert _h_rt_seconds("15:0:0") == 54_000.0
    assert _h_rt_seconds(None, None) is None
    assert _h_rt_seconds("not-a-walltime") is None


# ── the substrate is now recorded ───────────────────────────────────────────────────────────────

def test_the_archive_records_which_processor_the_training_ran_on() -> None:
    """The determinism envelope: a knob that can vary across records MUST be
    visible in the archive in the same change.

    The confirmatory lane moved to CPU and the archive recorded NOTHING about the CPU:
    ``env_fingerprint``'s only platform field is ``platform.platform()`` (kernel + glibc), and
    ``integrity._record_device`` reads only ``nvidia_smi.gpus[0]`` — which on a CPU node returns
    ``"<absent>"``, a value the report treats as a WILDCARD. So ``crn_pair_device_consistent`` was
    green whatever silicon the pair ran on, while ``lanes.EXCLUDED_CPU_POOLS`` excludes the AMD
    pool precisely because a different microarchitecture selects different oneDNN kernels, changes
    float reduction order, and breaks the CRN bit-exactness every paired contrast rests on.
    """
    from capture_env import _cpu_identity, capture_env

    env = capture_env(seed=0)
    assert "cpu" in env, "the CPU identity must reach the archived env"
    assert env["schema"] == "capture_env/4"
    ident = _cpu_identity()
    assert ident.get("machine") and ident.get("logical_cores")


# ── the freeze provenance anchor ────────────────────────────────────────────────────────────────

def test_the_freeze_tag_is_not_one_that_already_exists() -> None:
    """``FREEZE_TAG`` was ``prereg-v1.0`` — a tag ALREADY IN THIS REPOSITORY from the 2026-07-18
    freeze that R78 later lifted. ``git tag -s`` and its ``-a`` fallback are both best-effort, so
    the v2 freeze would have silently reported "tag SKIPPED" and had NO git provenance anchor; and
    ``_ots_stamp`` writes ``provenance/<FREEZE_TAG>.sha256``, so it would have OVERWRITTEN the committed
    v1.0 digest file with the v2 digest under the v1 name. The one artifact whose whole job is to
    prove which bytes were registered would have been made to attest the wrong thing."""
    import subprocess

    from freeze import FREEZE_TAG

    existing = subprocess.run(["git", "tag"], cwd=_ROOT, capture_output=True, text=True,
                              check=False).stdout.split()
    digest = _ROOT / "provenance" / f"{FREEZE_TAG}.sha256"

    if FREEZE_TAG in existing:
        # POST-FREEZE (from 2026-07-28). The tag now exists BECAUSE this freeze created it, so a
        # bare "must not exist" assertion inverts into failing precisely when the thing it guards
        # has succeeded. The invariant it actually protects is COLLISION: that the tag does not
        # belong to a DIFFERENT, earlier freeze whose digest file would then be overwritten and made
        # to attest the wrong bytes. That is checked directly — the digest must exist and must carry
        # the CURRENT canonical hash, which a stale tag from another freeze could not.
        import sys
        sys.path.insert(0, str(_ROOT / "scripts"))
        import freeze as _f

        assert digest.is_file(), (
            f"tag {FREEZE_TAG!r} exists but provenance/{FREEZE_TAG}.sha256 does not — the freeze left no "
            "digest to attest which bytes were registered")
        body = digest.read_text(encoding="utf-8")
        current = _f.canonical_hash()
        assert current in body, (
            f"tag {FREEZE_TAG!r} exists but provenance/{FREEZE_TAG}.sha256 does NOT carry the current "
            f"canonical hash {current[:12]} — the tag attests a DIFFERENT freeze, which is exactly "
            "the collision this guards against")
    else:
        assert FREEZE_TAG not in existing, (
            f"FREEZE_TAG {FREEZE_TAG!r} already exists as a git tag — the freeze would produce no "
            f"new anchor and would overwrite provenance/{FREEZE_TAG}.sha256")

    assert (_ROOT / "provenance" / "prereg-v1.0.sha256").is_file(), (
        "the v1.0 proof must survive as the historical record of a freeze that was lifted pre-data")


# ── the search lane is placeable ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch", ["{arm}_search", "{arm}_startup"])
def test_search_batches_take_the_search_lane_pack(batch: str) -> None:
    """Job cores are ``max(cores_per_training, threads) * pack``. Two SEARCH batches took the TEST
    flood's pack, so at the registered 8 chain threads they asked for 8 * 4 = 32 cores — under the
    36-core exclusivity refusal, so uncaught, but far past the measured placement cliff (8-core
    ~19 min; 16-core never placed in 28). That is why R107's registered thread count looked
    unusable and was about to be executed as 1 while the register said 8.

    Both are BARRIERS despite looking like bursts (nothing reflects, and no ask/tell step proceeds,
    until every candidate in the wave lands), so their cost is one training's LATENCY — exactly
    where the register says to spend threads.
    """
    src = (_ROOT / "src" / "cluster" / "campaign.py").read_text(encoding="utf-8")
    needle = 'f"' + batch.replace("{arm}", "{arm}")
    idx = src.index(needle)
    window = src[max(0, idx - 400): idx + 200]
    assert "pack=(run.search_pack or run.pack)" in window, (
        f"the {batch} batch must size on the SEARCH lane's pack, not the test flood's")
