"""Behaviour tests for the MODE-D maximum-parallel levers (R88; ops-only — no science changes).

Contracts under test:
* search waves run at ``search_pack`` with the tight ``search_h_rt`` (the latency lane) while
  legacy runs (search_pack=None) stay byte-identical (pack == run.pack, NO h_rt_call kwarg —
  existing fakes must never see a new kwarg);
* pipelined C4 submits ALL assurance blocks concurrently under the descending priority ladder
  (block 1 = tier-100 at PRIORITY_STAGE1 above the legs; blocks 2+ from PRIORITY_RUNG_BASE),
  with results keyed exactly like the sequential path.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cluster.campaign import (  # noqa: E402
    PRIORITY_RUNG_BASE,
    PRIORITY_STAGE1,
    ClusterRun,
    run_search_arm,
)
from src.io.results import write_run  # noqa: E402


def _opts(*, generations=2, candidates=4):
    from src.utils.config import load_config

    return {
        "pass_mode": "A", "provider": "stub", "seed": 0,
        "generations": generations, "candidates": candidates,
        "env_cfg": load_config("environment"), "n_assets": 31, "model": "stub",
        "train_steps": 100, "batch_size": 64, "normalize_obs": True,
        "n_trials": 1, "synthetic": True, "data": {}, "cvar_alpha": 0.05, "window": 20,
        "diversity_prompt_variation": False,
    }


class _Fake:
    """Records (name, pack, h_rt_call) per batch and writes one record per spec."""

    def __init__(self, root):
        self.root = root
        self.calls: list[tuple[str, int, str | None]] = []
        self._lock = threading.Lock()

    def run_batch(self, specs, name, *, pool="EF", pack=1, priority=0, h_rt_call=None):
        with self._lock:
            self.calls.append((name, pack, h_rt_call))
        for s in specs:
            cid = s.get("candidate_id") or s.get("run_id")
            write_run({
                "run_id": cid, "arm": s["arm"], "seed": int(s.get("seed", 0)), "fold": 0,
                "candidate_id": cid, "generation": int(s.get("generation", 0)),
                "reward_source": str(s.get("reward", "")), "reward_source_hash": "h",
                "prompt": s.get("prompt", ""), "feedback_block": "", "wall_clock": 0.0,
                "env_fingerprint": "x",
                "metrics": {"val_fitness": 1.0,
                            "tail_stats": {"cvar_05": -0.05, "cvar_10": -0.04, "cvar_25": -0.03,
                                           "cvar_01": -0.07, "left_tail_mass": 0.05,
                                           "robust_skew": 0.2},
                            "val_returns": [0.01, -0.01, 0.02]},
            }, str(Path(s["archive_root"]) / s["arm"]))
        return {"ok": True, "completed": len(specs)}


def test_search_pack_lane_applies_and_legacy_stays_byte_identical(tmp_path):
    # MODE-D lane: generation batches carry search_pack + the tight h_rt.
    fake = _Fake(tmp_path)
    run = ClusterRun(run_batch=fake.run_batch, spec_archive_root=str(tmp_path),
                     read_root=tmp_path, pack=5, search_pack=2, search_h_rt="5:0:0")
    run_search_arm("distributional", _opts(), run)
    gen_calls = [c for c in fake.calls if "_g" in c[0]]
    assert gen_calls and all(pack == 2 and hrt == "5:0:0" for _, pack, hrt in gen_calls)

    # Legacy: no search_pack -> run.pack, and the h_rt_call kwarg is NEVER passed (old fakes
    # without the parameter must keep working — enforced with a strict fake).
    class _Strict(_Fake):
        def run_batch(self, specs, name, *, pool="EF", pack=1, priority=0):  # no h_rt_call!
            return _Fake.run_batch(self, specs, name, pool=pool, pack=pack, priority=priority)

    strict = _Strict(tmp_path / "b")
    run2 = ClusterRun(run_batch=strict.run_batch, spec_archive_root=str(tmp_path / "b"),
                      read_root=tmp_path / "b", pack=5)
    run_search_arm("distributional", _opts(), run2)
    assert all(pack == 5 for _, pack, _hrt in strict.calls if "_g" in _[0] or True)


def test_pipeline_rungs_concurrent_ladder(monkeypatch, tmp_path):
    """All C4 blocks submit TOGETHER, and every one of them at ``-p 0``.

    The concurrency property (the barrier) is unchanged and is what this test was written for.
    What changed on 2026-07-31 is the priority assertion: the descending ladder
    ``[-100, -300, -310]`` is retired, because ``-p`` is a GLOBAL priority weighted at 4.0 and the
    descent put the HIGHEST rungs at the LOWEST standing against every other user on the cluster.
    See ``PRIORITY_CORE``'s docstring for the measurement that settled it.
    """
    from src.cluster import campaign as C

    seen: list[tuple[str, int, float]] = []
    gate = threading.Barrier(3, timeout=10)  # proves all 3 blocks are in flight SIMULTANEOUSLY

    def fake_test_leg(units, seeds, run, *, name, priority, interleave, resume, **kw):
        seen.append((name, priority, time.monotonic()))
        gate.wait()  # would deadlock (timeout -> BrokenBarrier) under sequential submission
        return {"ok": True, "n": len(seeds)}

    monkeypatch.setattr(C, "run_test_leg", fake_test_leg)
    run = ClusterRun(run_batch=lambda *a, **k: {"ok": True}, spec_archive_root=str(tmp_path),
                     read_root=tmp_path)
    out = C.run_campaign_tiered(
        [], lambda a: {}, {"mode": "tiered", "tiers": [2, 4, 6, 8]}, run,   # 4 tiers -> 3 C4 blocks
        test_leg_kwargs={}, frozen_root=tmp_path / "frozen", review_gate=False,
        pipeline_rungs=True,
    )
    prios = {name: p for name, p, _ in seen}
    assert prios == {"sweep_t1": 0, "sweep_t2": 0, "sweep_t3": 0}, (
        "every C4 rung block must ride at full fair-share standing; a descending ladder sinks the "
        "top rungs below every other cluster user (record §54)")
    assert PRIORITY_STAGE1 == 0 and PRIORITY_RUNG_BASE == 0
    assert out["results"]["sweep_t1"]["ok"] and out["results"]["sweep_t3"]["n"] == 2
    assert out["ok"] is True


def test_search_poll_lane_and_bo_priority_rule(tmp_path):
    """2026-07-21b: chain batches poll at search_poll_secs; bayes_opt is floor-critical-path -> -p 0."""
    from src.cluster.campaign import PRIORITY_CORE, _core_priority

    class _PollFake(_Fake):
        def __init__(self, root):
            super().__init__(root)
            self.polls: list[tuple[str, float | None]] = []

        def run_batch(self, specs, name, *, pool="EF", pack=1, priority=0,
                      h_rt_call=None, poll_call=None):
            self.polls.append((name, poll_call))
            return _Fake.run_batch(self, specs, name, pool=pool, pack=pack)

    fake = _PollFake(tmp_path)
    run = ClusterRun(run_batch=fake.run_batch, spec_archive_root=str(tmp_path),
                     read_root=tmp_path, pack=5, search_pack=2, search_h_rt="5:0:0",
                     search_poll_secs=45.0)
    run_search_arm("distributional", _opts(), run)
    gen_polls = [pc for name, pc in fake.polls if "_g" in name]
    assert gen_polls and all(pc == 45.0 for pc in gen_polls)   # the chain lane polls fast

    # 2026-07-31: EVERY arm rides at PRIORITY_CORE. The old rule hoisted only the H2 pair and
    # bayes_opt and left the three CONTROL arms at -100 — which starved exactly the arms that
    # isolate the effect, and opened a four-generation depth gap against the treatments.
    h2 = ("distributional", "scalar")
    for arm in ("bayes_opt", "distributional", "scalar", "random_search",
                "placebo", "placebo_shuffled", "scalar_cvar5",
                "baseline_differential_sharpe"):
        assert _core_priority(arm, h2) == PRIORITY_CORE == 0, (
            f"{arm} must ride at full fair-share standing (record §54)")


def test_canary_concurrent_with_family_gates_only_authoring(monkeypatch, tmp_path):
    """2026-07-21c: family arms (no Opus spend) start DURING the canary; LLM arms wait for it."""
    from src.cluster import campaign as C

    order: list[str] = []
    canary_may_finish = threading.Event()
    family_started = threading.Event()

    def fake_baselines(names, seeds, run, *, name, **kw):
        if name == "canary":
            order.append("canary_start")
            assert family_started.wait(timeout=10)   # family began while the canary still runs
            canary_may_finish.wait(timeout=10)
            order.append("canary_done")
        return {"ok": True}

    def fake_family(arm, opts, run, *, resume, priority):
        family_started.set()
        order.append(f"family_start:{arm}")
        return {"arm": arm, "accepted": [], "n": 0}

    def fake_search(arm, opts, run, *, resume, priority):
        order.append(f"llm_search:{arm}")            # must appear only after canary_done
        return {"arm": arm}

    monkeypatch.setattr(C, "run_baselines_on_cluster", fake_baselines)
    monkeypatch.setattr(C, "run_family_search_arm", fake_family)
    monkeypatch.setattr(C, "run_search_arm", fake_search)
    import src.io.results as R
    monkeypatch.setattr(R, "load_all", lambda p: [{"seed": 0}, {"seed": 1}])  # smoke coverage
    monkeypatch.setattr(C, "run_test_leg",
                        lambda *a, **k: {"ok": True, "n": 0})
    monkeypatch.setattr(C, "_resolve_select_freeze",
                        lambda run: (lambda p: None, lambda *a, **k: None))
    threading.Timer(0.3, canary_may_finish.set).start()
    run = ClusterRun(run_batch=lambda *a, **k: {"ok": True}, spec_archive_root=str(tmp_path),
                     read_root=tmp_path)
    out = C.run_campaign_tiered(
        ["distributional", "bayes_opt"], lambda a: {"seed": 0}, {"mode": "tiered", "tiers": [2]},
        run, test_leg_kwargs={}, frozen_root=tmp_path / "f", review_gate=False,
        canary_baselines=["raw_return"],
    )
    assert "family_start:bayes_opt" in order and "canary_done" in order
    assert order.index("family_start:bayes_opt") < order.index("canary_done")   # no-spend work early
    assert order.index("llm_search:distributional") > order.index("canary_done")  # authoring gated
    assert out["results"]["canary"] == {"ok": True}


def test_canary_failure_still_aborts_loud_with_zero_authoring(monkeypatch, tmp_path):
    from src.cluster import campaign as C
    import pytest as _pytest

    authored: list[str] = []
    monkeypatch.setattr(C, "run_baselines_on_cluster",
                        lambda *a, **k: {"ok": False, "boom": True})
    monkeypatch.setattr(C, "run_family_search_arm",
                        lambda arm, *a, **k: {"arm": arm, "accepted": [], "n": 0})
    monkeypatch.setattr(C, "run_search_arm",
                        lambda arm, *a, **k: authored.append(arm) or {"arm": arm})
    monkeypatch.setattr(C, "_resolve_select_freeze",
                        lambda run: (lambda p: None, lambda *a, **k: None))
    run = ClusterRun(run_batch=lambda *a, **k: {"ok": True}, spec_archive_root=str(tmp_path),
                     read_root=tmp_path)
    with _pytest.raises(RuntimeError, match="CANARY FAILED"):
        C.run_campaign_tiered(
            ["distributional", "random_search"], lambda a: {"seed": 0},
            {"mode": "tiered", "tiers": [2]}, run, test_leg_kwargs={},
            frozen_root=tmp_path / "f", review_gate=False, canary_baselines=["raw_return"],
        )
    assert authored == []                             # ZERO Opus authoring on a failed canary


# --------------------------------------------------------------------------- #
# R106: the LAUNCHER's roster must not drift from the registration             #
# --------------------------------------------------------------------------- #
def test_every_launcher_and_monitor_roster_matches_the_registered_queue_order():
    """The lock that was missing when R106 renamed a leg — and its absence bit immediately.

    R106 substituted `gemini-3.5-flash` -> `gemini-2.5-flash` (3.5's reasoning is MANDATORY, so it
    could not join the uniform reasoning-off suite). The registration, `config/legs.yaml` and the
    transport layer were all updated and every gate stayed green — because NOTHING checked the three
    places that name legs by LABEL outside those files:

      * `scripts/mode_d_launch.ps1`      — the RATIFIED launch path's queue. A stale label here
                                           launches a leg that no longer exists.
      * `scripts/mode_d_supervisor.ps1`  — the priority ladder and the legN tag map.
      * `MEASURED_AUTHORING_YIELD`       — keyed on the old label the running leg never matches, so
                                           `_DEFAULT_YIELD` silently applies and the earliest-warning
                                           authoring alarm is mis-calibrated for that leg.

    A label is a cross-file contract. Assert it in ONE place so the next substitution cannot be
    half-applied.
    """
    import re

    import yaml

    root = Path(__file__).resolve().parents[1]
    queue = yaml.safe_load(
        (root / "config" / "preregistration.yaml").read_text(encoding="utf-8")
    )["model_suite"]["queue_order"]

    launch = (root / "scripts" / "mode_d_launch.ps1").read_text(encoding="utf-8")
    listed = re.search(r"\$lines = @\((.*?)\)", launch, re.S).group(1)
    launch_legs = [m for m in re.findall(r'"([^"]+)"', listed) if m not in ("core", "h3")]
    assert launch_legs == queue, (
        f"mode_d_launch.ps1 queue drifted from the registration:\n  launcher={launch_legs}\n"
        f"  registered={queue}")

    sup = (root / "scripts" / "mode_d_supervisor.ps1").read_text(encoding="utf-8")
    # 2026-07-27: ``$legPriority`` is GONE, not renamed. R101 (Okhrati's seed-parity directive)
    # retired the -200..-290 ladder — all 11 full-loop models run at EQUAL standing — and finding
    # #96 made a negative --priority a hard SystemExit, so every leg line would have died at argv
    # parsing and this supervisor would have relaunched it forever at 600 s backoff. Only the
    # label->batch-tag map survives, and it is still the cross-file contract.
    body = re.search(r"\$legTag = \[ordered\]@\{(.*?)\}", sup, re.S).group(1)
    keys = re.findall(r'"([^"]+)"\s*=', body)
    assert sorted(keys) == sorted(queue), (
        f"mode_d_supervisor.ps1 $legTag drifted: {sorted(set(keys) ^ set(queue))}")

    from src.cluster.campaign_health import MEASURED_AUTHORING_YIELD

    missing = [leg for leg in queue if leg not in MEASURED_AUTHORING_YIELD]
    assert not missing, (
        f"MEASURED_AUTHORING_YIELD has no entry for {missing} — the authoring-health alarm would "
        "silently fall back to the default rate for those legs")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2026-07-27 LAUNCH GATE — the launcher invariants. Every one of these was VIOLATED by the
# ratified launcher on the day of the confirmatory launch, and none of them was covered by a test.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _strip_ps_comments(src: str) -> str:
    """Drop whole-line PowerShell comments.

    These invariants are about what the launcher EXECUTES, not what it documents — and the headers
    of these files deliberately quote the very flags being banned in order to explain why. Matching
    raw text would make the explanation itself a test failure, which is how a lock stops being
    maintained. Whole-line only: a trailing ``#`` inside a quoted argument is not a comment, and no
    launcher line here carries one.
    """
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _launcher_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: _strip_ps_comments((root / "scripts" / name).read_text(encoding="utf-8"))
            for name in ("mode_d_supervisor.ps1", "campaign_supervisor.ps1",
                         "install_onstart_task.ps1")}


def test_no_launcher_ever_lowers_our_queue_priority():
    """The standing rule is ABSOLUTE: never lower the SGE priority of our jobs, EVER. R101
    independently retired the -200..-290 leg ladder, and #96 made a negative --priority a hard
    refusal — so a launcher that still passed one would simply die at argv parsing, forever, under
    a supervisor that treats a nonzero exit as a crash to retry."""
    for name, src in _launcher_sources().items():
        assert "--priority" not in src, (
            f"{name} passes --priority; the default 0 IS full fair-share standing (R101)")
        assert "--allow-deprioritise" not in src, (
            f"{name} would opt in to deprioritisation — never for the confirmatory campaign")


#: D21 (2026-08-01): a launcher that DELEGATES rather than building its own campaign argument
#: vector. `install_onstart_task.ps1` used to hard-type the whole cluster vector, and that copy had
#: already drifted three ways from the live fleet (`--batch-tag c1`, `--pack 4`, and a fence missing
#: `node-d00b-024`). It now re-enters through `mode_d_launch.ps1` -> `mode_d_supervisor.ps1`, which
#: IS in this set and carries every flag asserted below.
#:
#: NOTE: THIS IS A STRENGTHENING, NOT AN EXEMPTION. Before D21 these invariants passed only because the
#: Myriad block happened to contain the flag strings; the file's LAPTOP branch never carried them
#: and was silently riding on that. A delegating launcher must now PROVE it delegates
#: (`test_a_delegating_launcher_names_the_vector_launcher_it_delegates_to`), so the flags are
#: inherited from a file that is itself checked, rather than merely absent.
_DELEGATING_LAUNCHERS = {"install_onstart_task.ps1"}


def _vector_launchers() -> dict[str, str]:
    """Launchers that BUILD a campaign argument vector, i.e. the ones the flag invariants bind."""
    return {k: v for k, v in _launcher_sources().items() if k not in _DELEGATING_LAUNCHERS}


def test_a_delegating_launcher_names_the_vector_launcher_it_delegates_to():
    """A delegating launcher must hand off to a file that IS flag-checked — never to nothing.

    Without this, dropping a launcher into `_DELEGATING_LAUNCHERS` would be a way to silence the
    flag invariants rather than satisfy them.
    """
    for name in _DELEGATING_LAUNCHERS:
        src = _launcher_sources()[name]
        assert "mode_d_launch.ps1" in src, (
            f"{name} is exempted from the flag invariants but delegates to nothing")
    # and the thing it delegates to must itself be covered
    assert "mode_d_supervisor.ps1" in _vector_launchers(), (
        "the delegation target is not in the flag-checked set — the exemption would be a hole")


def test_no_launcher_mixes_the_cpu_lane_with_gpu_pool_striping():
    """``--device cpu`` + ``--seed-pool-blocks`` is refused by the launcher BY DESIGN: a CPU job
    pins no pool, so the stripe would assert a device stratification the run does not have, and CPU
    and CUDA are not bit-identical. Every line carried the GPU stripe, so the launcher could not
    start on the lane it was meant to launch."""
    for name, src in _vector_launchers().items():
        assert "--seed-pool-blocks" not in src, f"{name} still carries the GPU seed stripe"
        assert "--pool EF" not in src and '"--pool", "EF"' not in src, (
            f"{name} still pins the GPU pool EF")
        assert "--device" in src and "cpu" in src, f"{name} does not select the CPU lane"


def test_no_launcher_hand_types_a_frozen_roster():
    """The roster and the H1 canon are RESOLVED from the frozen config. A hand-typed copy is what
    drifted to 7-of-9 arms (killing node N4) and, a day earlier, to 4-of-11 baselines."""
    for name, src in _launcher_sources().items():
        assert "--arms" not in src, (
            f"{name} hand-types --arms; omit it so resolve_cluster_arms() reads the frozen roster")
        assert "--baselines" not in src, f"{name} hand-types --baselines"


def test_every_launcher_passes_the_acfs_gold_directory():
    """``--gold-dir`` defaults to ~/Scratch/llmrp/inputs, which exists on Myriad and is EMPTY.

    Scoped to VECTOR launchers since D21 — a delegating launcher inherits the flag from the file it
    delegates to, and that file is checked by this same loop.
    """
    for name, src in _vector_launchers().items():
        assert "--gold-dir" in src, (
            f"{name} relies on the --gold-dir default, which points at an empty directory")


def test_the_search_lane_is_configured_so_eight_threads_can_place():
    """Job cores are ``max(cores_per_training, threads) * pack``. The registered chain thread count
    is 8 (R107), so the search lane must run at pack 1 (8 cores, ~19 min to place) rather than
    inheriting the test flood's pack (32 cores, past the placement cliff)."""
    src = _launcher_sources()["mode_d_supervisor.ps1"]
    assert '"--search-pack", "1"' in src and '"--search-threads", "8"' in src


def test_the_monitor_watches_every_mode_d_line_not_just_the_core():
    """The monitor filtered on c1_/h3ss_ only, so under MODE D it watched 2 of 12 lines and
    reported the other ten as nothing — and silence is this script's own signal for HEALTHY."""
    root = Path(__file__).resolve().parents[1]
    mon = (root / "scripts" / "campaign_monitor.sh").read_text(encoding="utf-8")
    assert "leg" in mon.split("qstat")[1].split("\n")[0], (
        "campaign_monitor.sh's qstat filter does not match the leg batch tags")
