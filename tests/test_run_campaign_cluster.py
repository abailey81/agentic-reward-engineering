"""Entry-point tests for scripts/run_campaign_cluster.py — the config assembly + the dry-run wiring
validation, exercised WITHOUT a cluster (synthetic panel, stub author)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ (for the shared FakeCluster)
import run_campaign_cluster as rcc  # noqa: E402


def test_parse_seeds_ranges_lists_and_dedup():
    assert rcc._parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert rcc._parse_seeds("0,1,5") == [0, 1, 5]
    assert rcc._parse_seeds("0-2,1-3") == [0, 1, 2, 3]  # overlapping -> deduped + sorted
    assert rcc._parse_seeds("0-402") == list(range(403))


def test_assemble_cluster_inputs_synthetic_is_arm_independent_and_author_ready():
    inp = rcc.assemble_cluster_inputs(
        arms=["distributional", "scalar"], seeds=[0, 1, 2], output_dir="outputs/_t",
        synthetic=True, train_steps=200, n_trials=1, candidates=4, generations=2,
        search_seed=0, embargo=21, pass_mode="A", provider="stub", llm_cfg=None, resume=False,
    )
    # opts are arm-independent (the arm is a run_search_arm parameter)
    assert inp["opts_for"]("distributional") is inp["opts_for"]("scalar")
    opts = inp["opts"]
    # everything the cluster author + spec builder consume is present
    for key in ("env_cfg", "n_assets", "provider", "pass_mode", "generations", "candidates",
                "seed", "model", "train_steps", "data", "cvar_alpha", "window"):
        assert key in opts, f"opts missing {key}"
    # test_leg_kwargs carries the windows the nodes reuse
    tlk = inp["test_leg_kwargs"]
    assert set(tlk) == {"panel_descriptor", "env_cfg", "agent_cfg", "train_window", "val_window",
                        "test_window", "embargo", "lookback"}
    tw, vw, te = inp["windows"]
    assert tw[0] < tw[1] <= vw[0] < vw[1] <= te[0] < te[1]  # ordered, embargoed
    assert inp["frozen_root"].name == "frozen"


def test_dry_run_validates_wiring_without_a_cluster(capsys):
    rc = rcc.main(["--dry-run", "--synthetic", "--arms", "distributional", "scalar",
                   "--seeds", "0-5", "--candidates", "4", "--generations", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wiring valid" in out and "2 arms" in out


def test_capstone_full_roster_end_to_end(tmp_path):
    """The WHOLE stack in one shot: assemble the REAL (synthetic) config -> run an LLM arm + a
    family arm + an H1 baseline CONCURRENTLY through a fake cluster -> assert the complete archive
    in the analyze_campaign parity layout (search/<arm>, test/<arm>, test/baseline_<name>)."""
    from test_cluster_campaign import FakeCluster, _inject_select_freeze  # shared fake

    from src.cluster.campaign import ClusterRun, run_campaign_on_cluster
    from src.io.results import load_all

    inp = rcc.assemble_cluster_inputs(
        arms=["distributional", "random_search"], seeds=[0, 1], output_dir=str(tmp_path),
        synthetic=True, train_steps=200, n_trials=1, candidates=2, generations=1,
        search_seed=0, embargo=21, pass_mode="A", provider="stub", llm_cfg=None, resume=False,
    )
    fake = FakeCluster(tmp_path)
    run = ClusterRun(run_batch=fake.run_batch, spec_archive_root=str(tmp_path), read_root=tmp_path)
    _inject_select_freeze(run)

    results = run_campaign_on_cluster(
        ["distributional", "random_search"], inp["opts_for"], inp["seeds"], run,
        test_leg_kwargs=inp["test_leg_kwargs"], frozen_root=tmp_path / "frozen",
        baseline_names=["differential_sharpe"],
    )
    assert all(results[k]["ok"] for k in ("distributional", "random_search", "__baselines__"))

    def ids(sub):
        return {r["run_id"] for r in load_all(str(tmp_path / sub))}

    # LLM arm + family arm both produced SEARCH candidates (disjoint sub-root)
    assert ids("search/distributional") == {"distributional-g0-c0", "distributional-g0-c1"}
    assert ids("search/random_search") == {"random_search-c0", "random_search-c1"}
    # every unit produced its sealed TEST records (arms + the H1 baseline), all under test/
    assert ids("test/distributional") == {"distributional-s0", "distributional-s1"}
    assert ids("test/random_search") == {"random_search-s0", "random_search-s1"}
    assert ids("test/baseline_differential_sharpe") == {
        "baseline_differential_sharpe-s0", "baseline_differential_sharpe-s1"}
    # search and test roots are disjoint (BUG-1 invariant) across the whole roster
    assert (tmp_path / "search").is_dir() and (tmp_path / "test").is_dir()


# --------------------------------------------------------------------------- #
# P19 (2026-07-13 pre-spend audit): the cluster entry point mirrors the laptop's
# verify-or-refuse freeze gate. These lock the semantics freeze-state-independently
# (enforce_freeze is monkeypatched on run_campaign, which main imports at call time).
# --------------------------------------------------------------------------- #
def test_freeze_gate_refuses_unfrozen_non_dry_run(tmp_path, monkeypatch):
    import run_campaign as rc

    def _refuse(*, allow_unfrozen=False):
        raise rc.CampaignNotFrozenError("NOT frozen (test)")

    monkeypatch.setattr(rc, "enforce_freeze", _refuse)
    import pytest

    with pytest.raises(SystemExit, match="NOT frozen"):
        rcc.main(["--synthetic", "--arms", "distributional", "--output-dir", str(tmp_path)])


def test_freeze_gate_exempts_dry_run(tmp_path, monkeypatch):
    import run_campaign as rc

    def _refuse(*, allow_unfrozen=False):  # would sink the run if the exemption regressed
        raise rc.CampaignNotFrozenError("NOT frozen (test)")

    monkeypatch.setattr(rc, "enforce_freeze", _refuse)
    rc_code = rcc.main(["--dry-run", "--synthetic", "--arms", "distributional",
                        "--output-dir", str(tmp_path)])
    assert rc_code == 0


def test_freeze_gate_allow_unfrozen_passes_through(tmp_path, monkeypatch):
    """--allow-unfrozen reaches enforce_freeze with the flag set, then main PROCEEDS to the
    next guard (F2 dirty-dir refusal proves we got past the gate without ssh)."""
    import run_campaign as rc

    calls: list[bool] = []

    def _stamp(*, allow_unfrozen=False):
        calls.append(allow_unfrozen)
        return {"enforced": True, "frozen": False, "allow_unfrozen": allow_unfrozen}

    monkeypatch.setattr(rc, "enforce_freeze", _stamp)
    dirty = tmp_path / "search" / "distributional" / "x"
    dirty.mkdir(parents=True)
    (dirty / "record.json").write_text("{}", encoding="utf-8")
    import pytest

    with pytest.raises(SystemExit, match="RE-AUTHORS"):
        rcc.main(["--synthetic", "--arms", "distributional", "--allow-unfrozen",
                  "--output-dir", str(tmp_path)])
    assert calls == [True]


def test_write_campaign_summary_mirrors_run_campaign_keys(tmp_path):
    """P7: the cluster writes an analyze/sentinel-compatible campaign_summary.json at the mirror
    root (test_window is what analyze_campaign's DeMiguel floor reads)."""
    import json

    inp = rcc.assemble_cluster_inputs(
        arms=["distributional"], seeds=[0], output_dir=str(tmp_path), synthetic=True,
        train_steps=200, n_trials=1, candidates=2, generations=1, search_seed=0,
        embargo=21, pass_mode="A", provider="stub", llm_cfg=None, resume=False,
    )
    rcc._write_campaign_summary(str(tmp_path), inp, freeze_stamp={"enforced": True},
                                extra={"tiered": True, "all_arms_tested": True, "exit_code": 0})
    s = json.loads((tmp_path / "campaign_summary.json").read_text(encoding="utf-8"))
    assert s["source"] == "run_campaign_cluster"
    assert s["gold_panel"] == {"synthetic": True}
    assert s["freeze"] == {"enforced": True}
    assert s["all_arms_tested"] is True and s["tiered"] is True
    tw = s["test_window"]  # exactly what analyze_campaign.main consumes for the benchmark floor
    assert isinstance(tw, list) and len(tw) == 2 and all(isinstance(x, int) for x in tw)


def test_h3_singleshot_flag_forces_shape_and_dry_runs(tmp_path, capsys):
    """C5: --h3-singleshot forces arms=[distributional] + generations from campaign.yaml's
    h3_singleshot_generations (1), and the keyless dry-run validates the wiring."""
    rc = rcc.main(["--h3-singleshot", "--dry-run", "--synthetic",
                   "--candidates", "4", "--output-dir", str(tmp_path)])
    assert rc == 0
    outtext = capsys.readouterr().out
    assert "1 arms" in outtext  # distributional only


def test_h3_singleshot_conflicts_with_tiered(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="SEPARATE invocations"):
        rcc.main(["--h3-singleshot", "--tiered", "--dry-run", "--synthetic",
                  "--output-dir", str(tmp_path)])


def test_root_suffix_validation_and_h3_conflict(tmp_path, monkeypatch):
    """C6-class guard: --root-suffix must be [a-z0-9_]+ and cannot combine with --h3-singleshot
    (which manages its own roots). Validation fires before any cluster contact."""
    import pytest
    import run_campaign as rc

    monkeypatch.setattr(rc, "enforce_freeze",
                        lambda allow_unfrozen=False: {"enforced": True, "frozen": True})
    with pytest.raises(SystemExit, match="lowercase"):
        rcc.main(["--synthetic", "--root-suffix", "Curve-C10", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="do not combine"):
        rcc.main(["--synthetic", "--h3-singleshot", "--root-suffix", "curve_c10",
                  "--output-dir", str(tmp_path)])


# ---------------------------------------------------------------------------------------------
# 2026-07-18 DEFAULTS-CLASS SWEEP regression lock (the launch-critical B*-resolution bug + its
# class): argparse defaults must NEVER hardcode mirrors of frozen config values — None resolves
# from the same keys the laptop main reads, asserted against the pre-registration where bound.
# ---------------------------------------------------------------------------------------------

def test_design_argparse_defaults_are_none_not_hardcoded_mirrors():
    """The bug class itself: a hardcoded default (30/6/30/21/25k) is a mirror that drifts."""
    args = rcc.build_parser().parse_args([])
    for name in ("train_steps", "candidates", "generations", "n_trials", "embargo"):
        assert getattr(args, name) is None, f"--{name} regressed to a hardcoded config mirror"


def test_bstar_none_resolves_from_campaign_yaml_and_matches_prereg():
    """THE launch-critical instance: train_steps=None must assemble at campaign.yaml's B*
    (the R77 400k), not prototype.yaml's 25k."""
    from src.utils.config import cfg_get, load_config

    expected = int(cfg_get(load_config("campaign"), "train_steps_per_candidate", 0))
    assert expected > 0
    inp = rcc.assemble_cluster_inputs(
        arms=["distributional"], seeds=[0], output_dir="outputs/_t",
        synthetic=True, train_steps=None, n_trials=1, candidates=4, generations=2,
        search_seed=0, embargo=21, pass_mode="A", provider="stub", llm_cfg=None, resume=False,
    )
    assert int(inp["agent_cfg"]["train_steps_per_candidate"]) == expected
    assert int(inp["opts"]["train_steps"]) == expected
    prereg = int(cfg_get(load_config("preregistration"), "train_steps_per_candidate", 0))
    assert expected == prereg  # the mirror pair the assembly asserts


def test_bstar_mirror_drift_refuses_assembly(monkeypatch):
    """If campaign.yaml and preregistration.yaml disagree on B*, assembly must refuse loudly."""
    import pytest

    import src.utils.config as cfgmod

    real = cfgmod.load_config

    def drifted(name, *a, **k):
        cfg = real(name, *a, **k)
        if name == "preregistration":
            cfg = dict(cfg)
            cfg["train_steps_per_candidate"] = 12345  # deliberate drift
        return cfg

    monkeypatch.setattr(cfgmod, "load_config", drifted)
    with pytest.raises(SystemExit, match="mirror drift"):
        rcc.assemble_cluster_inputs(
            arms=["distributional"], seeds=[0], output_dir="outputs/_t",
            synthetic=True, train_steps=None, n_trials=1, candidates=4, generations=2,
            search_seed=0, embargo=21, pass_mode="A", provider="stub", llm_cfg=None,
            resume=False,
        )


def test_main_resolves_all_design_values_from_configs(monkeypatch):
    """A flag-free launch (the supervisor line's shape) must resolve candidates/generations/
    n_trials/embargo from campaign.yaml + inference.yaml, laptop-parity, and leave train_steps
    None for the asserted in-assembly resolution."""
    captured = {}
    real = rcc.assemble_cluster_inputs

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(rcc, "assemble_cluster_inputs", spy)
    rc = rcc.main(["--dry-run", "--synthetic", "--arms", "distributional"])
    assert rc == 0
    from src.utils.config import cfg_get, load_config

    camp = load_config("campaign")
    assert captured["candidates"] == int(camp["candidates_per_arm"])
    assert captured["generations"] == int(cfg_get(camp.get("llm") or {}, "generations", 1))
    assert captured["n_trials"] == captured["candidates"]  # laptop parity (run_campaign.py:2138)
    inf = load_config("inference")
    assert captured["embargo"] == int(cfg_get(cfg_get(inf, "splits", {}),
                                              "embargo_trading_days", 21))
    assert captured["train_steps"] is None


def test_candidates_mirror_drift_refuses_launch(monkeypatch):
    """campaign.yaml candidates_per_arm vs preregistration.yaml matched_budget must agree."""
    import pytest

    import src.utils.config as cfgmod

    real = cfgmod.load_config

    def drifted(name, *a, **k):
        cfg = real(name, *a, **k)
        if name == "preregistration":
            cfg = dict(cfg)
            cfg["matched_budget"] = 99
        return cfg

    monkeypatch.setattr(cfgmod, "load_config", drifted)
    with pytest.raises(SystemExit, match="mirror drift"):
        rcc.main(["--dry-run", "--synthetic", "--arms", "distributional"])


def test_real_spend_refuses_explicit_design_overrides():
    """Pass-mode B without --allow-unfrozen must refuse ANY explicit design flag (the guard that
    makes the resolved-and-asserted path the only real-spend path)."""
    import pytest

    with pytest.raises(SystemExit, match="explicit design override"):
        rcc.main(["--dry-run", "--pass-mode", "B", "--candidates", "30",
                  "--arms", "distributional"])


def test_autosize_h_rt_sizes_on_the_resolved_bstar():
    """Catch #2: the walltime autosizer read a NONEXISTENT campaign.agent key then a stale
    hardcoded 200k — at B*=400k every pack-5 array task (~6:09 needed) would have been sized
    ~4h and walltime-killed. Lock the formula to the resolved top-level B*."""
    from src.utils.config import cfg_get, load_config

    bstar = int(cfg_get(load_config("campaign"), "train_steps_per_candidate", 0))
    # the campaign agent block must NOT grow a shadowing copy (the key the old code read)
    assert "train_steps_per_candidate" not in (cfg_get(load_config("campaign"), "agent", {}) or {})
    got = rcc.autosize_h_rt(5, bstar)
    hours = int(got.split(":")[0])
    # pack-5 at 400k on the worst-case curve needs ~6.2h -> 7; a 200k-sized 4h is the dead zone
    need_secs = (bstar * 5 / (0.5 * 253.0) + 1200.0) * 1.3
    assert hours * 3600 >= need_secs
    assert got == f"{int(need_secs // 3600) + 1}:0:0"


def test_baselines_guard_rejects_unknown_name_in_dry_run():
    """row 30l: the R97 fail-before-ssh guard must fire IN THE DRY-RUN PATH (it already regressed
    once by sitting below the dry-run exit — this locks the placement)."""
    import pytest

    with pytest.raises(SystemExit, match="unknown REWARD_CANON key"):
        rcc.main(["--dry-run", "--synthetic", "--arms", "distributional",
                  "--baselines", "not_a_real_reward"])


def test_canary_guard_rejects_unknown_name_in_dry_run():
    """row 30l: --canary names route through the same validation (they previously bypassed it
    and would have failed only after ssh/submit)."""
    import pytest

    # --arms is OMITTED (2026-07-27): under --tiered it now resolves the frozen 9-arm roster, and
    # a partial list is refused earlier than this guard. The test's subject is the CANARY name, so
    # let the roster resolve and keep the misspelling as the only thing under test.
    with pytest.raises(SystemExit, match="unknown REWARD_CANON key"):
        rcc.main(["--dry-run", "--synthetic", "--tiered", "--canary", "raw_retrun"])


def test_baselines_guard_accepts_full_canon_in_dry_run():
    """row 30l: every REWARD_CANON name (the ten-name §9 panel) passes the guard."""
    from src.baselines.rewards import REWARD_CANON

    rc = rcc.main(["--dry-run", "--synthetic", "--arms", "distributional",
                   "--baselines", *sorted(REWARD_CANON)])
    assert rc in (0, None)


# --- H1 frozen-family resolution (2026-07-26 launch-defect fix) -----------------------------
# The runbook's headline line hand-mirrored the H1 family and drifted to 4 names after the canon
# expanded to 11, which would have run a SUBSET of the registered family (breaking the N6 IUT,
# whose p = max over the 11 leg p-values) and mis-sized the C0 canary (first 3 of the list).


def _frozen_h1_family() -> list[str]:
    from src.utils.config import cfg_get, load_config

    return [str(b) for b in (cfg_get(load_config("campaign"), "h1_baselines", []) or [])]


def test_resolve_cluster_baselines_tiered_omitted_uses_frozen_family():
    """Headline (--tiered) with the flag OMITTED resolves the frozen config family, so the
    launch command can never carry a stale hand-typed mirror again."""
    frozen = _frozen_h1_family()
    assert len(frozen) >= 11, "the H1 canon expanded to 11 on 2026-07-26"
    assert rcc.resolve_cluster_baselines(None, tiered=True) == frozen


def test_resolve_cluster_baselines_non_tiered_omitted_skips_h1():
    """The h3 single-shot and C6 --root-suffix re-search lines rely on omit == skip."""
    assert rcc.resolve_cluster_baselines(None, tiered=False) is None


def test_resolve_cluster_baselines_refuses_the_drifted_runbook_four():
    """THE REGRESSION LOCK: the exact 4-name list the runbook carried must now be refused,
    naming the 7 missing canon members."""
    import pytest

    drifted = ["raw_return", "return_minus_variance", "return_minus_cvar", "differential_sharpe"]
    with pytest.raises(SystemExit, match="must be the FROZEN config h1_baselines family"):
        rcc.resolve_cluster_baselines(drifted, tiered=True)


def test_resolve_cluster_baselines_accepts_exact_family_any_order():
    """An explicit list is fine when it IS the frozen family (set equality — order-free)."""
    frozen = _frozen_h1_family()
    assert rcc.resolve_cluster_baselines(list(reversed(frozen)), tiered=True) == list(reversed(frozen))


def test_resolve_cluster_baselines_still_rejects_unknown_names_first():
    """The pre-existing R97 unknown-name guard keeps its own error message."""
    import pytest

    with pytest.raises(SystemExit, match="unknown REWARD_CANON key"):
        rcc.resolve_cluster_baselines(["not_a_real_reward"], tiered=True)


# ── 2026-07-27: the SAME drift-proofing, applied to --arms (see resolve_cluster_arms) ───────────

def _frozen_arm_roster() -> list[str]:
    from src.utils.config import cfg_get, load_config

    return [str(a) for a in (cfg_get(load_config("campaign"), "arms", []) or [])]


def test_resolve_cluster_arms_tiered_omitted_uses_the_frozen_nine():
    frozen = _frozen_arm_roster()
    assert len(frozen) == 9, "R108 took the roster 7 -> 9 (the H4 DFO portfolio: +cma_es, +tpe)"
    assert rcc.resolve_cluster_arms(None, tiered=True) == frozen


def test_resolve_cluster_arms_refuses_the_drifted_seven():
    """THE REGRESSION LOCK: the exact 7-arm list that FOUR launch paths hand-typed after R108 took
    the roster to 9 (mode_d_supervisor, campaign_supervisor, install_onstart_task, runbook §2).
    Running it leaves cma_es and tpe untrained, and confirmatory node N4 — whose p is the MAX over
    the four-optimiser portfolio — is then permanently unsatisfiable."""
    import pytest

    drifted = ["distributional", "scalar", "scalar_cvar5", "placebo", "placebo_shuffled",
               "random_search", "bayes_opt"]
    with pytest.raises(SystemExit, match="must be the FROZEN roster"):
        rcc.resolve_cluster_arms(drifted, tiered=True)


def test_resolve_cluster_arms_refuses_the_old_two_arm_argparse_default():
    """The launch-ready doc's command omitted --arms entirely; the old default was these two, so
    the documented launch would have run 2 of 9 arms with every gate green."""
    import pytest

    with pytest.raises(SystemExit, match="must be the FROZEN roster"):
        rcc.resolve_cluster_arms(["distributional", "scalar"], tiered=True)


def test_resolve_cluster_arms_leg_gets_exactly_the_five_llm_arms():
    got = rcc.resolve_cluster_arms(None, tiered=True, leg="deepseek-v4-pro")
    assert len(got) == 5 and set(got) <= set(_frozen_arm_roster())
    for dfo in ("random_search", "bayes_opt", "cma_es", "tpe"):
        assert dfo not in got, f"a leg must not run the DFO arm {dfo}"


def test_resolve_cluster_arms_refuses_a_wrong_arm_set_on_a_leg():
    import pytest

    with pytest.raises(SystemExit, match="runs the five LLM feedback arms"):
        rcc.resolve_cluster_arms(["distributional"], tiered=True, leg="glm-5.2")


def test_llm_arm_roster_is_derived_from_the_authoritative_table():
    """arms.yaml calls itself THE AUTHORITATIVE ROSTER; the LLM/DFO split must come from its own
    ``llm: false`` markers, so seating a sixth feedback arm cannot silently miss the legs."""
    from src.utils.config import cfg_get, load_config

    table = cfg_get(load_config("arms"), "arms", {}) or {}
    expected = [a for a in _frozen_arm_roster()
                if not (isinstance(table.get(a), dict) and table[a].get("llm") is False)]
    assert rcc.llm_arm_roster() == expected


def test_resolve_cluster_arms_non_tiered_omitted_keeps_the_legacy_default():
    """Rehearsals, probes and the D1 curve levels rely on it; the headline cannot reach this branch
    because the headline is --tiered."""
    assert rcc.resolve_cluster_arms(None, tiered=False) == list(rcc._LEGACY_DEFAULT_ARMS)


def test_resolve_cluster_baselines_never_attaches_the_h1_canon_to_a_leg():
    """The eleven H1 rewards are HAND-designed and model-INDEPENDENT, so they belong to the core
    line exactly once. Attaching them per-leg would duplicate the whole H1 leg ten times over and
    let a leg archive masquerade as an H1 replication the design never registered. This branch is
    what makes ``--leg --tiered`` (R101 lockstep) safe to run at all."""
    assert rcc.resolve_cluster_baselines(None, tiered=True, leg="glm-5.2") is None


def test_resolve_cluster_baselines_refuses_explicit_baselines_on_a_leg():
    import pytest

    with pytest.raises(SystemExit, match="does not run the H1 hand-reward canon"):
        rcc.resolve_cluster_baselines(_frozen_h1_family(), tiered=True, leg="glm-5.2")


# ── 2026-07-27/28: the remote preconditions ─────────────────────────────────────────────────────

def _runner(reply: str):
    """A fake ssh runner that ENFORCES the real contract.

    NOTE: WHY THIS EXISTS. The first versions of these preconditions called ``runner("<shell string>")``.
    ``submit.ssh_runner`` takes an **argv LIST** and ``shlex.quote``s each element, so a string makes
    Python iterate its CHARACTERS and quote each one — the cluster received
    ``m k d i r ' ' - p ...`` and returned 127. Every unit test passed, because the fakes were
    ``_runner("...")`` and accepted any object at all. A live rehearsal caught it; a test should
    have. A fake that is laxer than the thing it stands in for tests nothing about the seam.
    """
    def _run(cmd):
        assert isinstance(cmd, list), f"ssh_runner takes an argv LIST, got {type(cmd).__name__}"
        assert all(isinstance(c, str) for c in cmd), f"argv elements must be str: {cmd!r}"
        return reply
    return _run


def test_the_fake_runner_enforces_the_argv_contract():
    """Guard the guard: the fake must reject a bare string, or it cannot catch the real defect."""
    import pytest

    with pytest.raises(AssertionError, match="argv LIST"):
        _runner("x")("sha256sum /a/b")



def test_assert_remote_gold_refuses_an_empty_gold_dir_on_a_real_spend_run():
    """``--gold-dir`` defaults to ``~/Scratch/llmrp/inputs``, which exists on Myriad and is EMPTY;
    the licensed panel lives on ACFS. The jobscript deliberately ``mkdir -p``s the bind source (a
    fix for a different bug), so the container starts happily and every task then dies in the
    loader — uniform, late, per-task failure with no single loud cause."""
    import pytest

    with pytest.raises(SystemExit, match="GOLD PANEL NOT ON THE CLUSTER"):
        rcc.assert_remote_gold(_runner("sha256sum: No such file or directory"),
                               "/nope/gold", real_spend=True)


def test_assert_remote_gold_is_advisory_off_the_real_spend_path():
    assert rcc.assert_remote_gold(_runner(""), "/nope/gold", real_spend=False) == {}


# ── 2026-07-27: no FOREIGN records under the confirmatory roots ─────────────────────────────────

def test_foreign_remote_records_refuse_a_fresh_real_spend_launch(tmp_path):
    """FOUND ON THE CLUSTER, NOT IN THE CODE. ``~/Scratch/llmrp/outputs/search/`` — the core line's
    confirmatory search root — held 8 records from probe runs three days earlier, with run_ids in
    exactly the campaign's namespace: ``distributional-g0-c0..c4`` (the COMPLETE generation-0
    candidate set for that arm, since candidates=30 / generations=6 gives 5 per generation) plus
    ``scalar-g0-c2..c4``.

    The driver's first act is a pull; ``pending_specs`` would then see those run_ids as already
    archived; and ``run_search_arm`` under ``--resume`` REPLAYS an archived candidate rather than
    authoring one. The confirmatory search leg would have silently adopted foreign rewards as its
    own generation 0 and reflected on them. Nothing would have failed — the records are valid, they
    are simply not this experiment's.

    Every existing guard misses it: the F2 guard checks the LOCAL dir and only when ``--resume`` is
    ABSENT, while every confirmatory line correctly passes ``--resume``."""
    import pytest

    empty_local = tmp_path / "campaign_cluster"
    empty_local.mkdir()
    with pytest.raises(SystemExit, match="ALREADY EXIST under the confirmatory archive roots"):
        rcc.assert_no_foreign_remote_records(
            _runner("8"), "/remote/outputs", str(empty_local),
            ["search", "test", "frozen"], real_spend=True)


def test_a_genuine_resume_is_not_blocked_by_its_own_records(tmp_path):
    """THE DISCRIMINATOR, and why it cannot false-positive: a genuine resume has already mirrored
    its own remote records locally. ``local == 0 and remote > 0`` can therefore only mean the remote
    records came from something else."""
    local = tmp_path / "campaign_cluster"
    rec = local / "search" / "distributional" / "distributional-g0-c0"
    rec.mkdir(parents=True)
    (rec / "record.json").write_text("{}", encoding="utf-8")
    assert rcc.assert_no_foreign_remote_records(
        _runner("8"), "/remote/outputs", str(local),
        ["search", "test", "frozen"], real_spend=True) == 0


def test_the_local_check_is_scoped_to_THIS_lines_roots(tmp_path):
    """All twelve MODE-D lines share one --output-dir, and the legs start an hour behind the core
    by design (the canary shield). A whole-directory check would therefore go inert for every line
    that starts after the first one wrote a record — so each line must be judged on ITS OWN roots."""
    import pytest

    local = tmp_path / "campaign_cluster"
    core_rec = local / "search" / "distributional" / "distributional-g0-c0"
    core_rec.mkdir(parents=True)
    (core_rec / "record.json").write_text("{}", encoding="utf-8")
    # the CORE line's records exist; a LEG line starting now is still a fresh run for ITS roots
    with pytest.raises(SystemExit, match="ALREADY EXIST under the confirmatory archive roots"):
        rcc.assert_no_foreign_remote_records(
            _runner("3"), "/remote/outputs", str(local),
            ["search_leg_glm_5_2", "test_leg_glm_5_2", "frozen_leg_glm_5_2"], real_spend=True)


def test_a_clean_remote_root_passes(tmp_path):
    local = tmp_path / "campaign_cluster"
    local.mkdir()
    assert rcc.assert_no_foreign_remote_records(
        _runner("0"), "/remote/outputs", str(local),
        ["search", "test", "frozen"], real_spend=True) == 0


def test_a_line_that_has_already_SUBMITTED_is_never_treated_as_fresh(tmp_path):
    """THE WEDGE THIS PREVENTS, which the local-record test alone would have CAUSED.

    Window: a line submits, its records land REMOTELY, and the driver dies before its next pull
    mirrors them locally. The supervisor relaunches — that is its entire job — the guard sees
    ``local == 0 and remote > 0``, refuses, and the supervisor relaunches again. Forever, at 600 s
    intervals, over records the line produced ITSELF. A safety check that bricks the thing it
    protects is worse than no check.

    ``write_specs`` creates ``batches/<batch_tag>_<name>/`` BEFORE any qsub, so its existence is
    proof that remote records under these roots can be this line's own."""
    local = tmp_path / "campaign_cluster"
    (local / "batches" / "leg1_distributional_g0").mkdir(parents=True)
    assert rcc.assert_no_foreign_remote_records(
        _runner("25"), "/remote/outputs", str(local),
        ["search_leg_glm_5_2"], real_spend=True, batch_tag="leg1") == 0


def test_another_lines_batches_do_not_excuse_a_fresh_line(tmp_path):
    """The batch-dir test must be per-LINE too: twelve lines share one --output-dir, so the core
    line's batches must not vouch for a leg that has never submitted."""
    import pytest

    local = tmp_path / "campaign_cluster"
    (local / "batches" / "c1_distributional_g0").mkdir(parents=True)
    with pytest.raises(SystemExit, match="ALREADY EXIST under the confirmatory archive roots"):
        rcc.assert_no_foreign_remote_records(
            _runner("25"), "/remote/outputs", str(local),
            ["search_leg_glm_5_2"], real_spend=True, batch_tag="leg1")


def test_the_foreign_record_probe_fails_CLOSED_when_it_cannot_see(tmp_path):
    """A check that cannot see is not a check that passed. This repository's own 2026-07-26 review
    named fail-open-on-ABSENT-evidence as one of three recurring bug CLASSES (#28/#29), and the
    failure this guard prevents is the SILENT adoption of foreign rewards — the one class of error
    that yields a plausible result rather than an obvious one."""
    import pytest

    local = tmp_path / "campaign_cluster"
    local.mkdir()
    with pytest.raises(SystemExit, match="REFUSING rather than assuming clean"):
        rcc.assert_no_foreign_remote_records(
            _runner("ssh: connect to host myriad port 22: Connection timed out"),
            "/remote/outputs", str(local), ["search", "test", "frozen"], real_spend=True)


def test_foreign_records_are_advisory_off_the_real_spend_path(tmp_path):
    local = tmp_path / "campaign_cluster"
    local.mkdir()
    assert rcc.assert_no_foreign_remote_records(
        _runner("8"), "/remote/outputs", str(local),
        ["search", "test", "frozen"], real_spend=False) == 8


def test_assert_remote_gold_refuses_bytes_that_differ_from_the_frozen_manifest():
    """A wrong-but-PRESENT panel is worse than an absent one: it produces plausible numbers. The
    laptop loader has always checksum-verified; the remote copy every training actually reads never
    was."""
    import pytest

    from src.data.loaders import gold_suffix

    names = [f"{s}_{gold_suffix()}.parquet" for s in
             ("returns_panel", "cash_features", "splits", "top30_selection")]
    fake = "\n".join(f"{'0' * 64}  /g/{n}" for n in names)
    with pytest.raises(SystemExit, match="DOES NOT MATCH THE FROZEN MANIFEST"):
        rcc.assert_remote_gold(_runner(fake), "/g", real_spend=True)
