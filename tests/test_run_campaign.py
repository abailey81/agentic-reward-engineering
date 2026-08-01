"""FAST tests for the headline campaign post-loop (scripts/run_campaign.py, Rank 2).

These drive the Eureka post-loop — SEARCH -> SELECT -> FREEZE -> TEST — on the synthetic
panel with a FAKE trainer/policy (NO torch / NO real SAC), so they are fast. They lock in
the load-bearing invariants:

  - ``evaluate_winner_on_test`` writes exactly ONE record per (arm, seed) carrying the test
    metrics (``test_sharpe``, ``test_cvar05``, the per-step ``test_returns`` vector,
    ``per_period_pnl``) and the realized test-return vector;
  - the SELECTION path (``select_winner``) never constructs a test-window bundle — the seal
    in ``EnvBundle.test_returns`` holds (a 2-window selection bundle refuses the test leg);
  - ``--resume`` (``done_ids``) skips (arm, seed) records already archived;
  - the test leg is touched EXACTLY ONCE per (arm, seed) — never re-selected;
  - the additive ``OPTIONAL_FIELDS`` carry the per-period vector without breaking the schema.

The real-training path (run_headline_campaign with SAC) is covered by the gated slow test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.panel import Panel
from src.env.runner import EnvBundle, make_env_builder
from src.io.results import OPTIONAL_FIELDS, REQUIRED_FIELDS, load_all, write_run
from src.utils.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_campaign  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes (no torch)                                                            #
# --------------------------------------------------------------------------- #
class _FakePolicy:
    """Predicts a fixed zero action (-> uniform simplex via softmax). No torch."""

    def __init__(self, n_act: int) -> None:
        self.n_act = n_act

    def predict(self, obs: np.ndarray, deterministic: bool = True):  # noqa: ARG002
        return np.zeros(self.n_act, dtype=np.float32), None


def _identity_reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
    pr = float(port_ret)
    return pr, {"port_ret": pr}, None


_WINNER_SOURCE = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    pr = float(port_ret)\n"
    "    return pr, {'port_ret': pr}, None\n"
)


def _fake_trainer_factory(panel: Panel):
    """Return a ``trainer(agent_cfg, seed) -> (train_env -> policy)`` that never trains."""

    def _factory(agent_cfg, seed):  # noqa: ANN001, ARG001
        def _train(train_env):  # noqa: ANN001, ARG001
            return _FakePolicy(panel.N + 1)

        return _train

    return _factory


@pytest.fixture
def env_cfg():
    return load_config("environment")


@pytest.fixture
def windows(synthetic_panel: Panel, env_cfg):
    """Three short, PURGE-respecting windows on the 600-day synthetic panel.

    Each inter-split gap is >= the feature lookback (60) so the builder's R18 purge guard
    (max(embargo, lookback)) is satisfied — train [lb,250) | gap 60 | val [310,360) | gap 60 | test [420,T).
    """
    lookback = int(env_cfg["state"]["lookback_days"])
    return (lookback, 250), (250 + lookback, 360), (360 + lookback, synthetic_panel.T)


# --------------------------------------------------------------------------- #
# resolve_windows — searchsorted on the panel date axis                       #
# --------------------------------------------------------------------------- #
def test_resolve_windows_are_ordered_embargoed_and_nonempty(env_cfg):
    """The derived (train, val, test) windows are ordered, embargoed, and non-empty.

    Uses a synthetic panel whose date axis ACTUALLY SPANS the frozen calendar splits
    (2005-2026, like the real 5,406-session univ5 gold panel) so ``np.searchsorted`` resolves
    the Split-C dev train/val ends AND the 2020-2026 evaluation span — the clamps stay inert
    (the eval END clamps benignly to the panel end) and the full 21-day embargo is honoured.
    (The short 600-day fixture ends in 2006, before the test span, so it cannot manufacture a
    2020-2026 leg — that is a panel limitation, not a campaign-logic one, and is exercised by
    the short-window TEST-stage tests below.)
    """
    from src.data.synthetic import make_synthetic_panel

    panel = make_synthetic_panel(n_assets=4, n_days=7800, seed=1)  # 2005-01-03 .. 2026-05
    lookback = int(env_cfg["state"]["lookback_days"])
    splits = load_config("inference")["splits"]
    train_w, val_w, test_w = run_campaign.resolve_windows(panel, lookback, splits, embargo=21)

    for w in (train_w, val_w, test_w):
        assert w[1] > w[0]  # non-empty
    assert train_w[0] == lookback
    # R18: the inter-split purge must cover the FEATURE LOOKBACK (60), not just the 21-day embargo —
    # otherwise the downstream window's first observations read prior-split returns. Asserting >= lookback
    # (not >= 21) makes a future revert to the embargo-only gap FAIL here instead of silently re-leaking.
    assert val_w[0] >= train_w[1] + lookback     # train->val purge covers the lookback
    assert test_w[0] >= val_w[1] + lookback      # val->test purge covers the lookback
    # the first downstream observation's lookback window does NOT reach back across the split boundary
    assert val_w[0] - lookback >= train_w[1] and test_w[0] - lookback >= val_w[1]
    assert test_w[1] <= panel.T
    # The dev train end lands on the 2016-12-31 boundary (SPLIT C; searchsorted side='right').
    assert train_w[1] == int(np.searchsorted(np.asarray(panel.dates), np.datetime64("2016-12-31"), side="right"))
    # A 3-window builder must ACCEPT these (the embargo guard passes).
    make_env_builder(panel, env_cfg, train_w, val_w, test_window=test_w, embargo=21)


# --------------------------------------------------------------------------- #
# (2) SELECT — winner by validation DSR, and the seal holds                    #
# --------------------------------------------------------------------------- #
def test_select_winner_picks_best_val_fitness(results_dir):
    """select_winner returns the candidate with the greatest metrics['val_fitness']."""
    base = dict(
        arm="distributional", seed=0, fold=0, generation=0,
        reward_source_hash="h", feedback_block="", wall_clock=0.0,
        env_fingerprint="x",
    )
    write_run({**base, "run_id": "c0", "candidate_id": "c0", "metrics": {"val_fitness": 0.1}}, results_dir)
    write_run({**base, "run_id": "c1", "candidate_id": "c1", "metrics": {"val_fitness": 0.9}}, results_dir)
    write_run({**base, "run_id": "c2", "candidate_id": "c2", "metrics": {"val_fitness": 0.5}}, results_dir)
    winner = run_campaign.select_winner(results_dir)
    assert winner is not None
    assert winner["candidate_id"] == "c1"


def test_selection_bundle_never_exposes_the_test_leg(synthetic_panel: Panel, env_cfg, windows):
    """A SELECTION (2-window) bundle refuses the test leg — the seal is structural.

    Mirrors the search/selection path: the builder used during SELECT is 2-window, so
    ``test_returns`` raises. Only the campaign TEST stage builds a 3-window bundle.
    """
    train_w, val_w, _test_w = windows
    selection_bundle = make_env_builder(synthetic_panel, env_cfg, train_w, val_w)(_identity_reward)
    assert isinstance(selection_bundle, EnvBundle)
    assert selection_bundle.test_window is None
    with pytest.raises(RuntimeError, match="sealed"):
        selection_bundle.test_returns(_FakePolicy(synthetic_panel.N + 1))


# --------------------------------------------------------------------------- #
# (3) FREEZE — sealed winner record with the frozen marker                     #
# --------------------------------------------------------------------------- #
def test_freeze_winner_writes_frozen_marker_and_reward(tmp_path):
    """freeze_winner persists reward_source + reward.py with frozen=True."""
    winner = {
        "candidate_id": "distributional-g0-c0",
        "generation": 0,
        "reward_source": _WINNER_SOURCE,
        "reward_source_hash": "deadbeef",
        "feedback_block": "blk",
        "metrics": {"val_fitness": 0.7},
    }
    run_campaign.freeze_winner(
        "distributional", winner, search_seed=0, frozen_root=tmp_path, env_fingerprint="fp"
    )
    rec = load_all(tmp_path)[0]
    assert rec["frozen"] is True
    assert rec["reward_source_hash"] == "deadbeef"
    assert (tmp_path / "distributional-winner" / "reward.py").read_text() == _WINNER_SOURCE


# --------------------------------------------------------------------------- #
# (4) TEST — one record per (arm, seed); the test leg is touched exactly once  #
# --------------------------------------------------------------------------- #
def _winner_record(arm: str = "distributional") -> dict:
    return {
        "arm": arm,
        "candidate_id": f"{arm}-g0-c0",
        "generation": 0,
        "reward_source": _WINNER_SOURCE,
        "reward_source_hash": "deadbeef",
        "feedback_block": "",
        "metrics": {"val_fitness": 0.42},
    }


def test_evaluate_winner_on_test_one_record_per_seed_with_test_metrics(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """evaluate_winner_on_test writes ONE record per (arm, seed) with the test metrics."""
    train_w, val_w, test_w = windows
    seeds = [0, 1, 2]
    written = run_campaign.evaluate_winner_on_test(
        _winner_record(),
        synthetic_panel,
        env_cfg,
        seeds,
        test_w,
        train_window=train_w,
        val_window=val_w,
        embargo=21,
        trainer=_fake_trainer_factory(synthetic_panel),
        env_builder=make_env_builder,
        archive_root=str(tmp_path),
        arm="distributional",
    )
    assert len(written) == len(seeds)

    records = load_all(tmp_path)
    assert {r["run_id"] for r in records} == {"distributional-s0", "distributional-s1", "distributional-s2"}
    expected_len = test_w[1] - test_w[0]
    for r in records:
        m = r["metrics"]
        assert "test_sharpe" in m and "test_cvar05" in m
        assert "val_fitness" in m and m["val_fitness"] == pytest.approx(0.42)
        # The realized per-step test vector is present (Rank 3 PBO consumes it) ...
        assert len(m["test_returns"]) == expected_len
        assert len(m["per_period_pnl"]) == expected_len
        # ... both in metrics AND as the additive optional top-level fields.
        assert r["frozen"] is True
        assert len(r["test_returns"]) == expected_len
        assert np.isfinite(np.asarray(r["test_returns"], dtype=float)).all()
        # Schema stays valid (write_run validated REQUIRED_FIELDS).
        assert all(f in r for f in REQUIRED_FIELDS)


def test_evaluate_winner_touches_test_leg_exactly_once_per_seed(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """The sealed test leg is rolled out EXACTLY ONCE per seed (never re-selected)."""
    train_w, val_w, test_w = windows
    calls = {"test": 0, "val": 0}

    real_builder = make_env_builder

    def _counting_builder(panel, cfg, tw, vw, test_window=None, embargo=0, lookback=0):  # noqa: ANN001
        bundle = real_builder(
            panel, cfg, tw, vw, test_window=test_window, embargo=embargo, lookback=lookback
        )(_identity_reward)

        class _Counting:
            def __init__(self, b):
                self._b = b
                self.test_window = b.test_window

            def train_env(self):
                return self._b.train_env()

            def val_returns(self, policy):
                calls["val"] += 1
                return self._b.val_returns(policy)

            def test_returns(self, policy):
                calls["test"] += 1
                return self._b.test_returns(policy)

        return lambda reward_fn: _Counting(bundle)  # noqa: ARG005

    run_campaign.evaluate_winner_on_test(
        _winner_record(),
        synthetic_panel,
        env_cfg,
        [0, 1],
        test_w,
        train_window=train_w,
        val_window=val_w,
        embargo=21,
        trainer=_fake_trainer_factory(synthetic_panel),
        env_builder=_counting_builder,
        archive_root=str(tmp_path),
        arm="distributional",
    )
    assert calls["test"] == 2   # exactly once per seed
    assert calls["val"] == 0    # the TEST stage never re-selects on validation


def test_resume_skips_already_done_records(synthetic_panel: Panel, env_cfg, windows, tmp_path):
    """--resume (done_ids) skips (arm, seed) records already archived."""
    train_w, val_w, test_w = windows
    common = dict(
        winner=_winner_record(),
        panel=synthetic_panel,
        cfg=env_cfg,
        test_window=test_w,
        train_window=train_w,
        val_window=val_w,
        embargo=21,
        trainer=_fake_trainer_factory(synthetic_panel),
        env_builder=make_env_builder,
        archive_root=str(tmp_path),
        arm="distributional",
    )
    # First pass: seeds 0,1.
    run_campaign.evaluate_winner_on_test(seeds=[0, 1], **common)
    done = {r["run_id"] for r in load_all(tmp_path)}
    assert done == {"distributional-s0", "distributional-s1"}

    # Resume with seeds 0,1,2 but done_ids covering 0,1 -> only seed 2 is written.
    written = run_campaign.evaluate_winner_on_test(seeds=[0, 1, 2], done_ids=done, **common)
    assert [r["run_id"] for r in written] == ["distributional-s2"]
    assert {r["run_id"] for r in load_all(tmp_path)} == {
        "distributional-s0", "distributional-s1", "distributional-s2"
    }


def test_search_arm_comment_stub_winner_is_not_testable():
    """A search-arm winner whose source is a comment stub cannot be re-instantiated (FLAGGED)."""
    with pytest.raises(ValueError, match="executable reward|reward_source"):
        run_campaign._reinstantiate_frozen_winner("# bayes_opt coeffs=[1.0, 0.0, 0.0]\n")


def test_optional_fields_are_documented_and_additive():
    """OPTIONAL_FIELDS exist and stay DISJOINT from REQUIRED_FIELDS (additive, back-compatible)."""
    assert "test_returns" in OPTIONAL_FIELDS and "per_period_pnl" in OPTIONAL_FIELDS
    assert set(OPTIONAL_FIELDS).isdisjoint(set(REQUIRED_FIELDS))


def test_search_parallel_arm_builds_campaign_opts_and_calls_runner():
    """The reflect-on-BEST PARALLEL search wiring builds ``opts`` at the CAMPAIGN budget (50k — the worker
    couples buffer==train_steps) with the campaign's OWN Opus author mapped into the flat keys the parallel
    driver reads, and hands ONE arm to ``run_parallel``. Uses the REAL ``build_parallel_opts`` (so the opts
    construction is covered) + an injected runner (no SAC, no LLM)."""
    captured: dict = {}

    def _fake_runner(arms, opts, n_gpu, n_cpu, root):
        captured.update(arms=list(arms), opts=opts, n_gpu=n_gpu, n_cpu=n_cpu, root=str(root))
        return [{"arm": arms[0], "matched_budget_ok": True}]

    agent_cfg = {
        "train_steps_per_candidate": 200000, "buffer_size": 50000, "batch_size": 256,
        "normalize_obs": True, "device": "cpu",
    }
    run_campaign._search_parallel_arm(
        "distributional",
        agent_cfg=agent_cfg,
        env_cfg=load_config("environment"),
        synthetic=True,
        candidates=4,
        generations=2,
        n_trials=40,
        pass_mode="B",
        provider="anthropic",
        llm_cfg={"model_snapshot": "claude-opus-4-8", "api_key_env": "ANTHROPIC_API_KEY", "temperature": 1.0},
        search_seed=7,
        search_root=Path("ignored"),
        n_gpu=4,
        n_cpu=0,
        runner=_fake_runner,  # inject the scheduler; build_parallel_opts is the REAL shared builder
    )
    assert captured["arms"] == ["distributional"]
    assert captured["n_gpu"] == 4 and captured["n_cpu"] == 0
    opts = captured["opts"]
    # campaign budget B*=200k threaded straight in; parallel.train_candidate CAPS buffer at 50k
    # (ADR-025 decoupling), so the replay RAM stays bounded while the step budget is the full B*.
    assert opts["train_steps"] == 200000
    # the campaign's OWN reward-author (Opus), mapped into the flat key the parallel driver reads
    assert opts["model"] == "claude-opus-4-8"
    assert opts["api_key_env"] == "ANTHROPIC_API_KEY"
    assert opts["seed"] == 7 and opts["candidates"] == 4 and opts["generations"] == 2
    assert opts["batch_size"] == 256 and opts["normalize_obs"] is True


def test_build_parser_exposes_search_gpu_defaulting_to_serial():
    """``--search-gpu`` exists and DEFAULTS to None (=> serial reflect-on-last); the reflect-on-best
    parallel path is OFF unless explicitly opted in (amendment-gated)."""
    parser = run_campaign.build_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "search_gpu" in actions and actions["search_gpu"].default is None
    assert "search_cpu" in actions and actions["search_cpu"].default == 0


def test_build_parser_baselines_override_defaults_to_config_path():
    """R97: ``--baselines`` exists, DEFAULTS to None (=> the frozen config h1_baselines path is
    byte-identical), and parses a name list for the report-only secondary-panel invocation."""
    parser = run_campaign.build_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "baselines" in actions and actions["baselines"].default is None
    args = parser.parse_args(
        ["--baselines-only", "--baselines", "differential_downside_ratio", "log_growth"])
    assert args.baselines == ["differential_downside_ratio", "log_growth"]
    assert args.baselines_only is True


# --------------------------------------------------------------------------- #
# Graceful-shutdown flag (campaign-robustness §A): the SHUTDOWN Event stops the #
# arm loop at a boundary NON-DESTRUCTIVELY, via an injected fake runner.        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def _clear_shutdown():
    """Ensure the module-level SHUTDOWN Event is clear before AND after each test (no leakage)."""
    run_campaign.SHUTDOWN.clear()
    yield
    run_campaign.SHUTDOWN.clear()


def _campaign_kwargs(tmp_path, arms):
    """Minimal synthetic-panel kwargs for run_headline_campaign (no gold, no torch).

    ``min_candidates=0`` DISABLES the C3b winner-selection floor: these wiring tests inject fake
    searches that archive no records, so with the production default (full budget) every arm would
    stop at ``search_incomplete`` before the stage under test. The floor itself has dedicated tests
    (``test_select_floor_*`` below) that run with it ENABLED.
    """
    return dict(
        arms=list(arms),
        seeds=[0],
        search_seed=0,
        candidates=2,
        train_steps=10,
        n_trials=2,
        output_dir=str(tmp_path / "camp"),
        synthetic=True,
        embargo=21,
        pass_mode="A",
        provider="stub",
        generations=1,
        agent_cfg={"train_steps_per_candidate": 10, "buffer_size": 10, "device": "cpu"},
        min_candidates=0,
    )


def test_shutdown_preset_skips_all_arms_without_searching(tmp_path, monkeypatch, _clear_shutdown):
    """If SHUTDOWN is already set, the arm loop skips EVERY arm at the first boundary and runs no search.

    This is the core graceful-shutdown contract: a set flag => the loop exits at the next (arm) boundary
    WITHOUT issuing any (paid) search or any training. We inject spies for the search + test stages and a
    fake trainer factory (no torch); they must never be called, and every arm is reported
    ``skipped_shutdown`` — a non-destructive stop the operator resumes with --resume.
    """
    search_calls: list[str] = []
    test_calls: list[str] = []

    def _spy_search(arm, **_kw):  # noqa: ANN001
        search_calls.append(arm)
        return {"arm": arm}

    def _spy_test(*_a, **_kw):  # noqa: ANN001
        test_calls.append(_kw.get("arm", "?"))
        return []

    monkeypatch.setattr(run_campaign, "run_winner_search", _spy_search)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", _spy_test)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    run_campaign.SHUTDOWN.set()  # interrupt BEFORE the loop starts
    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional", "scalar"]))

    assert search_calls == []  # no paid search issued
    assert test_calls == []    # no training/test issued
    # The A.2 check fires on the FIRST arm and breaks the whole loop -> exactly one skipped_shutdown
    # summary, and the campaign exits without searching/testing anything (non-destructive).
    assert [s["status"] for s in summary["arms"]] == ["skipped_shutdown"]
    assert summary["arms"][0]["arm"] == "distributional"


def test_shutdown_after_arm1_completes_stops_arm2_at_boundary(tmp_path, monkeypatch, _clear_shutdown):
    """A flag SET as arm 1 finishes lets arm 1 COMPLETE, then stops arm 2 at the between-arms boundary.

    The injected fake test leg for arm 1 sets SHUTDOWN once arm 1 is fully tested. Arm 2 is then reported
    ``skipped_shutdown`` at the A.2 boundary and its search is NEVER issued — a cooperative, non-
    destructive exit bounded to the un-started arm (the operator resumes the rest with --resume).
    """
    searched: list[str] = []
    tested: list[str] = []

    def _fake_search(arm, **_kw):  # noqa: ANN001
        searched.append(arm)
        return {"arm": arm}

    def _fake_select(_root):  # noqa: ANN001 - canned winner so select/freeze/test proceed
        return {"candidate_id": "distributional-g0-c0", "generation": 0,
                "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
                "feedback_block": "", "metrics": {"val_fitness": 0.5}}

    def _fake_test(*_a, **_kw):  # noqa: ANN001
        tested.append(_kw.get("arm", "?"))
        run_campaign.SHUTDOWN.set()  # interrupt arrives as arm 1's test leg completes
        return [{"run_id": "distributional-s0"}]

    monkeypatch.setattr(run_campaign, "run_winner_search", _fake_search)
    monkeypatch.setattr(run_campaign, "select_winner", _fake_select)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", _fake_test)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional", "scalar"]))
    statuses = {s["arm"]: s["status"] for s in summary["arms"]}

    assert searched == ["distributional"]      # only arm 1's search ran; arm 2's never issued
    assert tested == ["distributional"]        # arm 1 ran its test leg to completion
    assert statuses["distributional"] == "tested"
    assert statuses["scalar"] == "skipped_shutdown"  # arm 2 stopped at the boundary, non-destructively


def test_shutdown_after_freeze_defers_test_leg_nondestructively(tmp_path, monkeypatch, _clear_shutdown):
    """A flag SET during arm 1's SEARCH defers arm 1's (long) test leg at the A.3 boundary.

    This exercises the SEARCH->TEST sub-stage cut: search+select+freeze are done, but the flag is set
    before the test leg, so the test leg is NOT issued and arm 1 is reported ``frozen_test_deferred``
    (on --resume the frozen winner loads and the test seeds resume). The test leg must never run.
    """
    tested: list[str] = []

    def _fake_search(arm, **_kw):  # noqa: ANN001
        run_campaign.SHUTDOWN.set()  # interrupt during search, BEFORE the test leg
        return {"arm": arm}

    def _fake_select(_root):  # noqa: ANN001
        return {"candidate_id": "distributional-g0-c0", "generation": 0,
                "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
                "feedback_block": "", "metrics": {"val_fitness": 0.5}}

    monkeypatch.setattr(run_campaign, "run_winner_search", _fake_search)
    monkeypatch.setattr(run_campaign, "select_winner", _fake_select)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test",
                        lambda *a, **k: tested.append(k.get("arm", "?")))  # must NOT be called
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional", "scalar"]))

    assert tested == []  # the test leg was deferred, never issued
    assert summary["arms"][0]["arm"] == "distributional"
    assert summary["arms"][0]["status"] == "frozen_test_deferred"


# --------------------------------------------------------------------------- #
# (H1) BASELINE stage — the pre-registered "beat-the-human" hand rewards        #
# --------------------------------------------------------------------------- #
def test_baseline_reward_builder_resolves_canon_and_rejects_unknown():
    """``_baseline_reward_builder`` returns the canonical REWARD_CANON callable, fail-loud on a bad name."""
    from src.baselines import rewards as R

    build = run_campaign._baseline_reward_builder("return_minus_cvar")
    assert build("# ignored stub source\n") is R.return_minus_cvar  # the canonical callable, not re-derived
    # A stale/typo name must raise (never silently no-op) — protects against config drift.
    with pytest.raises(AttributeError):
        run_campaign._baseline_reward_builder("sharpe_episodic")  # dropped name, not in the canon


def test_baseline_winner_record_routes_by_name_not_source():
    """The baseline 'winner' record carries reward_kind='baseline' + reward_name + a stub source."""
    rec = run_campaign._baseline_winner_record("differential_sharpe")
    assert rec["reward_kind"] == "baseline"
    assert rec["reward_name"] == "differential_sharpe"
    assert rec["arm"] == "baseline_differential_sharpe"
    assert rec["reward_source"].strip() == "# baseline:differential_sharpe"
    # val_fitness is NaN (a baseline is never validation-selected).
    assert np.isnan(rec["metrics"]["val_fitness"])


def test_evaluate_baselines_on_test_serial_dispatches_4x_and_archives_disjoint(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """The SERIAL baseline stage runs each of the 4 H1 baselines at the seeds, archived to baseline_<name>/.

    Uses the SAME fake trainer (no torch) as the winner tests. Asserts: one record per (baseline, seed)
    with the test metrics + frozen marker, under a DISJOINT test/baseline_<name>/ subtree (not test/<arm>/).
    """
    train_w, val_w, test_w = windows
    seeds = [0, 1, 2]
    baselines = ["raw_return", "return_minus_variance", "return_minus_cvar", "differential_sharpe"]
    test_root = tmp_path / "test"

    written = run_campaign.evaluate_baselines_on_test(
        baselines,
        panel=synthetic_panel,
        panel_descriptor={"synthetic": True},
        env_cfg=env_cfg,
        agent_cfg=None,
        seeds=seeds,
        train_window=train_w,
        val_window=val_w,
        test_window=test_w,
        embargo=21,
        lookback=int(env_cfg["state"]["lookback_days"]),
        test_root=test_root,
        trainer=_fake_trainer_factory(synthetic_panel),
        resume=False,
        n_gpu=0,  # serial path (no spawn / no torch)
    )

    # 4 baselines x 3 seeds = 12 records dispatched.
    assert len(written) == len(baselines) * len(seeds)
    # Each baseline got its OWN disjoint subtree test/baseline_<name>/.
    for name in baselines:
        sub = test_root / f"baseline_{name}"
        assert sub.is_dir()
        recs = load_all(sub)
        assert {r["run_id"] for r in recs} == {f"baseline_{name}-s{s}" for s in seeds}
        for r in recs:
            assert r["arm"] == f"baseline_{name}"
            assert r["frozen"] is True
            m = r["metrics"]
            assert "test_sharpe" in m and "test_cvar05" in m
            assert len(m["test_returns"]) == (test_w[1] - test_w[0])
            assert np.isfinite(np.asarray(r["test_returns"], dtype=float)).all()
            assert all(f in r for f in REQUIRED_FIELDS)


def test_evaluate_baselines_on_test_resume_skips_done_cells(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """--resume skips already-archived (baseline, seed) cells (no duplicate training)."""
    train_w, val_w, test_w = windows
    test_root = tmp_path / "test"
    common = dict(
        panel=synthetic_panel,
        panel_descriptor={"synthetic": True},
        env_cfg=env_cfg,
        agent_cfg=None,
        train_window=train_w,
        val_window=val_w,
        test_window=test_w,
        embargo=21,
        lookback=int(env_cfg["state"]["lookback_days"]),
        test_root=test_root,
        trainer=_fake_trainer_factory(synthetic_panel),
        n_gpu=0,
    )
    # First pass: seeds 0,1 for one baseline.
    run_campaign.evaluate_baselines_on_test(["raw_return"], seeds=[0, 1], resume=False, **common)
    done = {r["run_id"] for r in load_all(test_root / "baseline_raw_return")}
    assert done == {"baseline_raw_return-s0", "baseline_raw_return-s1"}

    # Resume with seeds 0,1,2 -> only seed 2 is newly written.
    written = run_campaign.evaluate_baselines_on_test(["raw_return"], seeds=[0, 1, 2], resume=True, **common)
    assert [r["run_id"] for r in written] == ["baseline_raw_return-s2"]
    assert {r["run_id"] for r in load_all(test_root / "baseline_raw_return")} == {
        "baseline_raw_return-s0", "baseline_raw_return-s1", "baseline_raw_return-s2"
    }


def test_run_headline_campaign_runs_baseline_stage_via_fakes(tmp_path, monkeypatch, _clear_shutdown):
    """run_headline_campaign dispatches the H1 baseline stage after the arms (injected fakes, no torch).

    Spies the baseline stage to confirm it is called with EXACTLY the configured baseline_names, and that
    a per-(baseline,seed) record set is produced — without real SAC training. The arm leg is faked so the
    test exercises ONLY the new wiring (the arm path is covered by the shutdown tests above)."""
    captured: dict = {}

    def _fake_search(arm, **_kw):  # noqa: ANN001
        return {"arm": arm}

    def _fake_select(_root):  # noqa: ANN001
        return {"candidate_id": "distributional-g0-c0", "generation": 0,
                "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
                "feedback_block": "", "metrics": {"val_fitness": 0.5}}

    def _spy_baselines(names, **_kw):  # noqa: ANN001
        captured["names"] = list(names)
        captured["n_gpu"] = _kw.get("n_gpu")
        # M19 (2026-07-05): the stage status now counts records ON DISK unconditionally, so the
        # fake must leave the archive a COMPLETED stage would (one record per (baseline, seed)) —
        # else the new no-shutdown-shortfall branch correctly reports a test_failure_wave.
        base = dict(
            arm="baseline", seed=0, fold=0, generation=0, reward_source_hash="h",
            feedback_block="", wall_clock=0.0, env_fingerprint="x",
        )
        for n in names:
            write_run(
                {**base, "run_id": f"baseline_{n}-s0", "candidate_id": f"baseline_{n}-s0",
                 "metrics": {"val_fitness": 0.0}},
                Path(_kw["test_root"]) / f"baseline_{n}",
            )
        return [{"run_id": f"baseline_{n}-s0"} for n in names]

    monkeypatch.setattr(run_campaign, "run_winner_search", _fake_search)
    monkeypatch.setattr(run_campaign, "select_winner", _fake_select)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "evaluate_baselines_on_test", _spy_baselines)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["baseline_names"] = ["raw_return", "return_minus_cvar"]
    summary = run_campaign.run_headline_campaign(**kwargs)

    assert captured["names"] == ["raw_return", "return_minus_cvar"]  # read straight from baseline_names
    statuses = {s["arm"]: s["status"] for s in summary["arms"]}
    assert statuses.get("h1_baselines") == "tested"
    base_summary = next(s for s in summary["arms"] if s["arm"] == "h1_baselines")
    assert base_summary["n_records_written"] == 2


def test_run_headline_campaign_skips_baselines_when_none_configured(tmp_path, monkeypatch, _clear_shutdown):
    """No h1_baselines => the baseline stage is never called (back-compatible default)."""
    called: list = []

    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner",
                        lambda _r: {"candidate_id": "c", "generation": 0, "reward_source": _WINNER_SOURCE,
                                    "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5}})
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "evaluate_baselines_on_test",
                        lambda *a, **k: called.append(1) or [])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    # _campaign_kwargs does not set baseline_names -> defaults to None -> stage skipped.
    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional"]))
    assert called == []
    assert not any(s["arm"] == "h1_baselines" for s in summary["arms"])


# --------------------------------------------------------------------------- #
# Freeze ENFORCEMENT (verify-or-refuse): the REAL campaign must REFUSE an        #
# un-frozen / drifted pre-registration, ALLOW it under --allow-unfrozen, and     #
# STAMP the canonical hash into the campaign summary. Driven by a FAKE freeze    #
# module (no real freeze, no torch) so the frozen / drift / unfrozen branches    #
# are all exercised deterministically.                                          #
# --------------------------------------------------------------------------- #
import types as _types  # noqa: E402


class _FakeFreezeStatus:
    """Mimics scripts.freeze.FreezeStatus (only the fields enforce_freeze reads)."""

    def __init__(self, *, hash_, recorded_hash, already_frozen, phase0="P0-OK"):
        self.hash = hash_
        self.recorded_hash = recorded_hash
        self.already_frozen = already_frozen
        self.phase0_marker = phase0
        self.checks = []


def _install_fake_freeze(monkeypatch, *, status=None, fail_verify_msg=None):
    """Put a FAKE ``freeze`` module on sys.modules so ``enforce_freeze``'s ``import freeze`` gets it.

    The fake carries its own FreezeError class (which ``enforce_freeze`` catches via
    ``except _freeze.FreezeError``) and a ``verify`` that either returns ``status`` or — when
    ``fail_verify_msg`` is set — raises THIS module's FreezeError with that message (so the raised
    type is exactly the one enforce_freeze will catch).
    """
    import sys as _sys

    fake = _types.ModuleType("freeze")

    class FreezeError(RuntimeError):
        ...

    fake.FreezeError = FreezeError  # type: ignore[attr-defined]

    def _verify(root=None):  # noqa: ANN001, ARG001
        if fail_verify_msg is not None:
            raise FreezeError(fail_verify_msg)
        return status

    fake.verify = _verify  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "freeze", fake)
    return fake


def test_enforce_freeze_refuses_when_unfrozen(monkeypatch):
    """An un-frozen prereg (frozen: false / freeze_hash: null) REFUSES the real run."""
    _install_fake_freeze(
        monkeypatch,
        status=_FakeFreezeStatus(hash_="abc123", recorded_hash=None, already_frozen=False),
    )
    with pytest.raises(run_campaign.CampaignNotFrozenError, match="NOT frozen"):
        run_campaign.enforce_freeze(allow_unfrozen=False)


def test_enforce_freeze_refuses_on_drift(monkeypatch):
    """A FROZEN prereg whose recorded hash != the recomputed hash (drift) REFUSES the real run."""
    _install_fake_freeze(
        monkeypatch,
        status=_FakeFreezeStatus(hash_="NEWHASH", recorded_hash="OLDHASH", already_frozen=True),
    )
    with pytest.raises(run_campaign.CampaignNotFrozenError, match="DRIFT"):
        run_campaign.enforce_freeze(allow_unfrozen=False)


def test_enforce_freeze_accepts_and_stamps_when_frozen(monkeypatch):
    """A FROZEN, un-drifted prereg returns the stamp (frozen + enforced + the canonical hash)."""
    _install_fake_freeze(
        monkeypatch,
        status=_FakeFreezeStatus(hash_="HASH-OK", recorded_hash="HASH-OK", already_frozen=True),
    )
    stamp = run_campaign.enforce_freeze(allow_unfrozen=False)
    assert stamp["frozen"] is True
    assert stamp["enforced"] is True
    assert stamp["freeze_hash"] == "HASH-OK"
    assert stamp["recorded_hash"] == "HASH-OK"


def test_enforce_freeze_allow_unfrozen_downgrades_to_warning(monkeypatch):
    """--allow-unfrozen turns the refusal into a warning + enforced=False (dev escape hatch)."""
    _install_fake_freeze(
        monkeypatch,
        status=_FakeFreezeStatus(hash_="abc", recorded_hash=None, already_frozen=False),
    )
    stamp = run_campaign.enforce_freeze(allow_unfrozen=True)
    assert stamp["enforced"] is False
    assert stamp["frozen"] is False
    assert stamp["freeze_hash"] == "abc"  # the hash is still recorded for provenance


def test_enforce_freeze_propagates_verify_failure_as_refusal(monkeypatch):
    """A failing prose<->yaml / Phase-0 gate (FreezeError) REFUSES with a clear message."""
    _install_fake_freeze(monkeypatch, fail_verify_msg="Phase-0 precondition NOT met")
    with pytest.raises(run_campaign.CampaignNotFrozenError, match="verify FAILED"):
        run_campaign.enforce_freeze(allow_unfrozen=False)


def test_enforce_freeze_verify_failure_under_allow_unfrozen_warns(monkeypatch):
    """--allow-unfrozen downgrades even a verify FAILURE to a warning (enforced=False)."""
    _install_fake_freeze(monkeypatch, fail_verify_msg="prose<->yaml mismatch")
    stamp = run_campaign.enforce_freeze(allow_unfrozen=True)
    assert stamp["enforced"] is False and stamp["frozen"] is False


def test_run_headline_campaign_stamps_freeze_into_summary(tmp_path, monkeypatch, _clear_shutdown):
    """The freeze stamp threaded into run_headline_campaign lands in campaign_summary.json."""
    import json

    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner",
                        lambda _r: {"candidate_id": "c", "generation": 0, "reward_source": _WINNER_SOURCE,
                                    "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5}})
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    stamp = {"frozen": True, "enforced": True, "freeze_hash": "DEADBEEF", "recorded_hash": "DEADBEEF"}
    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["freeze_stamp"] = stamp
    summary = run_campaign.run_headline_campaign(**kwargs)

    assert summary["freeze"] == stamp
    # ...and it is persisted to disk in the summary artifact.
    on_disk = json.loads((Path(kwargs["output_dir"]) / "campaign_summary.json").read_text())
    assert on_disk["freeze"]["freeze_hash"] == "DEADBEEF"
    assert on_disk["freeze"]["enforced"] is True


def test_run_headline_campaign_default_freeze_stamp_is_unenforced(tmp_path, monkeypatch, _clear_shutdown):
    """With no freeze_stamp (the test/dry-run path), the summary records enforced=False (back-compat)."""
    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner",
                        lambda _r: {"candidate_id": "c", "generation": 0, "reward_source": _WINNER_SOURCE,
                                    "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5}})
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional"]))
    assert summary["freeze"]["enforced"] is False


def test_build_parser_exposes_allow_unfrozen_defaulting_off():
    """``--allow-unfrozen`` exists and DEFAULTS to False (the real run refuses an unfrozen prereg)."""
    parser = run_campaign.build_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "allow_unfrozen" in actions and actions["allow_unfrozen"].default is False


def test_install_signal_handlers_is_idempotent_and_safe():
    """The handler installer runs without error and leaves SHUTDOWN clear (no signal delivered).

    Saves + restores the process SIGINT/SIGTERM handlers so installing them here does not leak into
    the rest of the test session (e.g. pytest-timeout / the runner's own Ctrl-C handling).
    """
    import signal

    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_campaign.SHUTDOWN.clear()
        run_campaign._install_signal_handlers()
        run_campaign._install_signal_handlers()  # idempotent
        assert not run_campaign.SHUTDOWN.is_set()
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


# --------------------------------------------------------------------------- #
# (H3) SINGLE-SHOT stage — the distributional arm re-run at generations=1      #
# (DEEP_H3 §1, canonical contrast). Drives the FULL SEARCH->SELECT->FREEZE->TEST #
# chain via an INJECTED fake arm_runner/trainer (no SAC, no LLM).               #
# --------------------------------------------------------------------------- #
def _fake_singleshot_arm_runner(captured: dict, archive_root: str, candidates: int):
    """A fake ``run_arm`` for the H3 single-shot search: records the call kwargs (esp. ``generations``)
    and ARCHIVES ``candidates`` synthetic candidate records under ``<archive_root>/<arm>/`` — exactly the
    on-disk contract ``select_winner`` then reads — so the whole single-shot chain runs with NO training.
    The best ``val_fitness`` is deterministic (the last candidate) so the selected winner is known.
    """
    def _runner(arm, *, generations, archive_root, **kw):  # noqa: ANN001, ARG001
        captured["arm"] = arm
        captured["generations"] = generations
        captured["candidates"] = kw.get("candidates")
        captured["seed"] = kw.get("seed")
        arm_root = Path(archive_root) / arm
        for i in range(candidates):
            write_run(
                {
                    "run_id": f"{arm}-g0-c{i}", "arm": arm, "seed": 0, "fold": 0,
                    "candidate_id": f"{arm}-g0-c{i}", "generation": 0,
                    "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
                    "feedback_block": "", "metrics": {"val_fitness": 0.1 * (i + 1)},
                    "wall_clock": 0.0, "env_fingerprint": "x",
                },
                str(arm_root),
            )
        return {"arm": arm}

    return _runner


def test_run_h3_singleshot_dispatches_gen1_and_archives_to_singleshot_roots(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """The H3 stage searches the DISTRIBUTIONAL arm at generations=1 (best-of-N, no reflection) via an
    injected fake arm_runner, then SELECT->FREEZE->TEST into the DISJOINT ``*_h3_singleshot/`` roots."""
    train_w, val_w, test_w = windows
    seeds = [0, 1, 2]
    captured: dict = {}
    out = tmp_path / "camp"

    summary = run_campaign.run_h3_singleshot(
        panel=synthetic_panel,
        panel_descriptor={"synthetic": True},
        env_cfg=env_cfg,
        agent_cfg=None,
        seeds=seeds,
        search_seed=7,
        candidates=4,
        train_steps=10,
        n_trials=4,
        train_window=train_w,
        val_window=val_w,
        test_window=test_w,
        embargo=21,
        lookback=int(env_cfg["state"]["lookback_days"]),
        output_dir=out,
        synthetic=True,
        pass_mode="A",
        provider="stub",
        singleshot_generations=1,
        trainer=_fake_trainer_factory(synthetic_panel),
        arm_runner=_fake_singleshot_arm_runner(captured, str(out / "search_h3_singleshot"), candidates=4),
        n_gpu=0,
    )

    # (1) SEARCH dispatched the DISTRIBUTIONAL arm at generations=1 (the canonical single-shot contrast).
    assert captured["arm"] == "distributional"
    assert captured["generations"] == 1            # best-of-N, NO reflection (the whole budget in one gen)
    assert captured["candidates"] == 4 and captured["seed"] == 7
    # The stage summary reports the single-shot arm + its generation count.
    assert summary["arm"] == "distributional_singleshot"
    assert summary["status"] == "tested"
    assert summary["singleshot_generations"] == 1
    assert summary["n_seeds_written"] == len(seeds)

    # (2/3) FREEZE wrote the single-shot winner under the SEPARATE frozen root (NOT the headline frozen/).
    assert not (out / "frozen").exists()           # the headline frozen tree is untouched
    frozen = load_all(out / "frozen_h3_singleshot")
    assert len(frozen) == 1 and frozen[0]["frozen"] is True
    assert frozen[0]["run_id"] == "distributional-winner"

    # (4) TEST archived the winner's seeds under the SEPARATE test root test_h3_singleshot/distributional/.
    assert not (out / "test").exists()             # the headline test tree is untouched
    test_recs = load_all(out / "test_h3_singleshot" / "distributional")
    assert {r["run_id"] for r in test_recs} == {f"distributional-s{s}" for s in seeds}
    for r in test_recs:
        assert r["arm"] == "distributional" and r["frozen"] is True
        m = r["metrics"]
        assert "test_sharpe" in m and "test_cvar05" in m
        assert len(m["test_returns"]) == (test_w[1] - test_w[0])
        assert all(f in r for f in REQUIRED_FIELDS)


def test_run_h3_singleshot_resume_loads_frozen_winner_and_skips_done_seeds(
    synthetic_panel: Panel, env_cfg, windows, tmp_path
):
    """--resume LOADS the existing frozen single-shot winner (NO paid re-search) and skips done test seeds."""
    train_w, val_w, test_w = windows
    out = tmp_path / "camp"
    common = dict(
        panel=synthetic_panel,
        panel_descriptor={"synthetic": True},
        env_cfg=env_cfg,
        agent_cfg=None,
        search_seed=7,
        candidates=4,
        train_steps=10,
        n_trials=4,
        train_window=train_w,
        val_window=val_w,
        test_window=test_w,
        embargo=21,
        lookback=int(env_cfg["state"]["lookback_days"]),
        output_dir=out,
        synthetic=True,
        trainer=_fake_trainer_factory(synthetic_panel),
        n_gpu=0,
    )

    # First pass: seeds 0,1 — a real (fake) search runs and freezes the winner.
    c1: dict = {}
    run_campaign.run_h3_singleshot(
        seeds=[0, 1],
        resume=False,
        arm_runner=_fake_singleshot_arm_runner(c1, str(out / "search_h3_singleshot"), candidates=4),
        **common,
    )
    assert c1["generations"] == 1
    done = {r["run_id"] for r in load_all(out / "test_h3_singleshot" / "distributional")}
    assert done == {"distributional-s0", "distributional-s1"}

    # Resume with seeds 0,1,2 -> the frozen winner is LOADED (arm_runner must NOT be called), only seed 2 new.
    c2: dict = {}

    def _must_not_search(*_a, **_kw):  # noqa: ANN001
        c2["called"] = True
        return {"arm": "distributional"}

    summary = run_campaign.run_h3_singleshot(
        seeds=[0, 1, 2],
        resume=True,
        arm_runner=_must_not_search,
        **common,
    )
    assert "called" not in c2                       # NO re-search on resume (frozen winner loaded)
    assert summary["n_seeds_written"] == 1          # only the new seed 2 was tested
    assert {r["run_id"] for r in load_all(out / "test_h3_singleshot" / "distributional")} == {
        "distributional-s0", "distributional-s1", "distributional-s2"
    }


def test_run_headline_campaign_wires_h3_singleshot_stage_when_enabled(tmp_path, monkeypatch, _clear_shutdown):
    """run_headline_campaign dispatches the H3 single-shot stage AFTER the arms when run_h3_singleshot=True,
    threading the configured h3_singleshot_generations — injected fakes, no torch."""
    captured: dict = {}

    def _fake_search(arm, **_kw):  # noqa: ANN001
        return {"arm": arm}

    def _fake_select(_root):  # noqa: ANN001
        return {"candidate_id": "distributional-g0-c0", "generation": 0,
                "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
                "feedback_block": "", "metrics": {"val_fitness": 0.5}}

    def _spy_h3(*, singleshot_generations, **_kw):  # noqa: ANN001
        captured["singleshot_generations"] = singleshot_generations
        captured["seeds"] = list(_kw.get("seeds"))
        return {"arm": "distributional_singleshot", "status": "tested",
                "n_seeds_written": len(_kw.get("seeds")), "singleshot_generations": singleshot_generations}

    monkeypatch.setattr(run_campaign, "run_winner_search", _fake_search)
    monkeypatch.setattr(run_campaign, "select_winner", _fake_select)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    # Spy the module-level stage fn: the wiring resolves it via sys.modules so the gate param never shadows it.
    monkeypatch.setattr(run_campaign, "run_h3_singleshot", _spy_h3)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["run_h3_singleshot"] = True
    kwargs["h3_singleshot_generations"] = 1
    summary = run_campaign.run_headline_campaign(**kwargs)

    assert captured["singleshot_generations"] == 1   # the configured single-shot generation count threaded
    statuses = {s["arm"]: s["status"] for s in summary["arms"]}
    assert statuses.get("distributional_singleshot") == "tested"


def test_run_headline_campaign_skips_h3_singleshot_when_disabled(tmp_path, monkeypatch, _clear_shutdown):
    """run_h3_singleshot defaults to False => the single-shot stage is NEVER dispatched (back-compatible)."""
    called: list = []

    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner",
                        lambda _r: {"candidate_id": "c", "generation": 0, "reward_source": _WINNER_SOURCE,
                                    "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5}})
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "run_h3_singleshot", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    # _campaign_kwargs does not set run_h3_singleshot -> defaults to False -> stage skipped.
    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional"]))
    assert called == []
    assert not any(s["arm"] == "distributional_singleshot" for s in summary["arms"])


# --------------------------------------------------------------------------- #
# M1 — window-drift guard: resolved windows must match the recorded set        #
# --------------------------------------------------------------------------- #
# The three unit tests below deliberately pin the FROZEN pre-Split-C univ3 reference windows
# ([60,2517]/[2577,3272]/[3332,5283]) as a stable test vector — the guard is suffix-keyed, so the
# univ3 row stays valid alongside the ACTIVE univ5 row (asserted against config + real gold below).
def _expected_splits(tw, vw, te, suffix="univ3"):
    return {"expected_windows": {suffix: {"train": list(tw), "val": list(vw), "test": list(te)}}}


def test_assert_expected_windows_passes_on_match() -> None:
    """Resolved windows equal to the recorded set for the suffix -> no raise (M1)."""
    tw, vw, te = (60, 2517), (2577, 3272), (3332, 5283)
    run_campaign._assert_expected_windows("univ3", tw, vw, te, _expected_splits(tw, vw, te))


def test_assert_expected_windows_raises_on_drift() -> None:
    """A shifted window (searchsorted-clamp drift) fails loud with a naming message (M1)."""
    tw, vw, te = (60, 2517), (2577, 3272), (3332, 5283)
    splits = _expected_splits(tw, vw, te)
    drifted = (60, 2500)  # train end moved -> a new panel calendar
    with pytest.raises(run_campaign.WindowDriftError, match="drift"):
        run_campaign._assert_expected_windows("univ3", drifted, vw, te, splits)


def test_assert_expected_windows_raises_when_suffix_unrecorded() -> None:
    """A new suffix with no recorded windows must RAISE (record + review before running; M1)."""
    tw, vw, te = (60, 2517), (2577, 3272), (3332, 5283)
    splits = _expected_splits(tw, vw, te, suffix="univ3")
    with pytest.raises(run_campaign.WindowDriftError, match="no expected windows recorded"):
        run_campaign._assert_expected_windows("univ9", tw, vw, te, splits)


def test_expected_windows_config_records_ratified_split_c() -> None:
    """The freeze-bound config records the RATIFIED windows for BOTH suffixes (M1, ADR-044/051).

    univ5 (ACTIVE, Split C) = train [60,3021] / val [3081,3775] / test [3835,5406] and univ3
    (the FROZEN pre-Split-C reference) = [60,2517]/[2577,3272]/[3332,5283]. An accidental edit
    to config/inference.yaml: splits.expected_windows fails HERE, before it can reach the guard.
    """
    expected = load_config("inference")["splits"]["expected_windows"]
    assert expected["univ5"] == {"train": [60, 3021], "val": [3081, 3775], "test": [3835, 5406]}
    assert expected["univ3"] == {"train": [60, 2517], "val": [2577, 3272], "test": [3332, 5283]}


_GOLD_UNIV5 = Path(__file__).resolve().parents[1] / "data" / "gold" / "returns_panel_univ5.parquet"


@pytest.mark.skipif(not _GOLD_UNIV5.exists(), reason="frozen univ5 gold panel not present (licensed data)")
def test_resolve_windows_on_real_gold_match_recorded_univ5() -> None:
    """ACTIVE path end-to-end: resolve_windows on the REAL univ5 panel equals the recorded set (M1).

    Loads the dev top-30 to the settled 2026-06-30 cutoff (ADR-051), resolves the Split-C calendar
    splits, and asserts the integer windows equal config/inference.yaml: splits.expected_windows.univ5
    — the drift guard itself must pass on them (the exact pre-run check the headline campaign makes).
    """
    from src.data.loaders import load_gold_panel

    splits = load_config("inference")["splits"]
    panel = load_gold_panel("development", end="2026-06-30").panel
    train_w, val_w, test_w = run_campaign.resolve_windows(panel, 60, splits, embargo=21)
    assert (list(train_w), list(val_w), list(test_w)) == ([60, 3021], [3081, 3775], [3835, 5406])
    run_campaign._assert_expected_windows("univ5", train_w, val_w, test_w, splits)  # no raise


# --------------------------------------------------------------------------- #
# C3a — no exit-0 husk runs: campaign_exit_status / incomplete_arms            #
# --------------------------------------------------------------------------- #
def _summ(arm, status, **extra):
    return {"arm": arm, "status": status, **extra}


def test_campaign_exit_status_clean_run_is_zero() -> None:
    """Every arm tested (+ tested report-only stages) -> exit 0, nothing incomplete."""
    summaries = [_summ("distributional", "tested"), _summ("scalar", "tested"),
                 _summ("h1_baselines", "tested")]
    assert run_campaign.campaign_exit_status(summaries, ["distributional", "scalar"]) == 0
    assert run_campaign.incomplete_arms(summaries, ["distributional", "scalar"]) == []


@pytest.mark.parametrize("bad_status", [
    "no_candidates", "test_failure_wave", "winner_not_testable", "search_incomplete", "error",
    "some_future_status",  # fail-loud default: an unknown non-tested status must gate too
])
def test_campaign_exit_status_failure_statuses_exit_3(bad_status) -> None:
    """Any FAILURE status (or an unknown status) -> EXIT_INCOMPLETE (3): the supervisor relaunches."""
    summaries = [_summ("distributional", "tested"), _summ("scalar", bad_status)]
    code = run_campaign.campaign_exit_status(summaries, ["distributional", "scalar"])
    assert code == run_campaign.EXIT_INCOMPLETE == 3
    assert ("scalar", bad_status) in run_campaign.incomplete_arms(summaries, ["distributional", "scalar"])


def test_campaign_exit_status_missing_arm_is_a_husk() -> None:
    """An expected arm with NO summary row on a NON-interrupted run is the silent-drop husk -> exit 3."""
    summaries = [_summ("distributional", "tested")]
    assert run_campaign.campaign_exit_status(summaries, ["distributional", "scalar"]) == 3
    assert ("scalar", "missing") in run_campaign.incomplete_arms(summaries, ["distributional", "scalar"])


def test_campaign_exit_status_operator_interrupt_stays_zero() -> None:
    """skipped_shutdown / frozen_test_deferred are DELIBERATE stops: incomplete (table + flag) but exit 0,
    so the supervisor's graceful-shutdown contract (exit 0 = do not fight the operator) is preserved —
    including for the arms the interrupt never reached (no summary row)."""
    summaries = [_summ("distributional", "tested"), _summ("scalar", "frozen_test_deferred"),
                 _summ("placebo", "skipped_shutdown")]
    expected = ["distributional", "scalar", "placebo", "scalar_cvar5"]  # scalar_cvar5 never reached
    assert run_campaign.campaign_exit_status(summaries, expected) == 0
    bad = dict(run_campaign.incomplete_arms(summaries, expected))
    assert bad["scalar"] == "frozen_test_deferred" and bad["placebo"] == "skipped_shutdown"
    assert bad["scalar_cvar5"] == "missing"  # still PRINTED/flagged, just not a failure exit


def test_campaign_exit_status_failure_beats_interrupt() -> None:
    """A failed arm on an interrupted run is still a failure: the interrupt excuses MISSING arms only."""
    summaries = [_summ("distributional", "test_failure_wave"), _summ("scalar", "skipped_shutdown")]
    assert run_campaign.campaign_exit_status(summaries, ["distributional", "scalar"]) == 3


@pytest.mark.parametrize("shutting_down,n_done,n_expected,expected", [
    (False, 0, 3, "test_failure_wave"),      # M19: a no-shutdown shortfall is a FAILURE WAVE, never success
    (False, 2, 3, "test_failure_wave"),      # M19: partial no-shutdown shortfall -> same (husk guard)
    (False, 3, 3, "tested"),
    (True, 3, 3, "tested"),                  # shutdown but every seed finished -> tested (flag set at end)
    (True, 4, 3, "tested"),                  # more on disk than requested (stale/resume) -> tested
    (True, 2, 3, "frozen_test_deferred"),    # genuine mid-leg drain left a seed un-run
    (True, 0, 3, "frozen_test_deferred"),
])
def test_stage_completion_status(shutting_down, n_done, n_expected, expected) -> None:
    """A completed TEST stage is 'tested'; a shutdown that left seeds UN-RUN is 'frozen_test_deferred';
    a shortfall with NO shutdown is a 'test_failure_wave' (M19, 2026-07-05 — a total parallel H1/H3
    seed-failure cascade must not bank a husk 'tested (0)' and exit 0).

    This pins the distinction the three post-leg sites (arms / H1 baselines / H3 single-shot) share via
    ``stage_completion_status``: the interrupted status is reserved for a REAL partial drain, so a leg
    that returned complete results is never re-labelled just because the flag was set as it finished
    (spec §J / §A.3; regression guard for test_shutdown_after_arm1_completes_stops_arm2_at_boundary).
    """
    assert run_campaign.stage_completion_status(
        shutting_down=shutting_down, n_done=n_done, n_expected=n_expected) == expected


def test_run_headline_campaign_summary_carries_gate_and_main_would_exit_3(
    tmp_path, monkeypatch, _clear_shutdown
):
    """End-to-end through run_headline_campaign: a no_candidates arm lands in the summary as
    all_arms_tested=False + exit_code=3 (what main() raises as SystemExit)."""
    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner", lambda _r: None)  # -> no_candidates
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional"]))
    assert summary["all_arms_tested"] is False
    assert summary["exit_code"] == run_campaign.EXIT_INCOMPLETE
    assert [s["status"] for s in summary["arms"]] == ["no_candidates"]


def test_run_headline_campaign_clean_summary_flags_complete(tmp_path, monkeypatch, _clear_shutdown):
    """A fully-tested run reports all_arms_tested=True and exit_code=0."""
    monkeypatch.setattr(run_campaign, "run_winner_search", lambda arm, **k: {"arm": arm})
    monkeypatch.setattr(run_campaign, "select_winner",
                        lambda _r: {"candidate_id": "c", "generation": 0, "reward_source": _WINNER_SOURCE,
                                    "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5}})
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    summary = run_campaign.run_headline_campaign(**_campaign_kwargs(tmp_path, ["distributional"]))
    assert summary["all_arms_tested"] is True and summary["exit_code"] == 0


# --------------------------------------------------------------------------- #
# C3b — winner-selection floor: a PARTIAL search pool can never freeze a winner #
# --------------------------------------------------------------------------- #
def test_select_floor_ok_pure_logic() -> None:
    """Resolved slots = accepted + LEDGERED failures; unresolved slots (llm_error skips) leave it short."""
    assert run_campaign.select_floor_ok(30, 0, 30) is True    # full pool, no failures
    assert run_campaign.select_floor_ok(27, 3, 30) is True    # sandbox rejects RESOLVE their slots
    assert run_campaign.select_floor_ok(27, 0, 30) is False   # 3 unresolved (llm_error / crash) -> short
    assert run_campaign.select_floor_ok(0, 30, 30) is False   # all-failures pool: nothing to select
    assert run_campaign.select_floor_ok(5, 0, 3) is True      # over-resolved (resumed re-run) never blocks


def test_select_floor_blocks_partial_pool_and_reports_resumable_status(
    tmp_path, monkeypatch, _clear_shutdown
):
    """A search that archived FEWER records than the budget (e.g. llm_error skips) must NOT select or
    freeze: the arm is reported ``search_incomplete`` (resumable) and neither select_winner nor
    freeze_winner nor the test leg run. The fake search archives 1 accepted record of a 2-budget."""
    from src.io.results import write_run as _wr

    selected: list = []
    frozen: list = []
    tested: list = []

    def _short_search(arm, *, archive_root, **_kw):  # noqa: ANN001 - archives 1 < candidates=2
        _wr({"run_id": f"{arm}-g0-c0", "arm": arm, "seed": 0, "fold": 0,
             "candidate_id": f"{arm}-g0-c0", "generation": 0, "reward_source": _WINNER_SOURCE,
             "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5},
             "wall_clock": 0.0, "env_fingerprint": "x"}, str(Path(archive_root) / arm))
        return {"arm": arm}

    monkeypatch.setattr(run_campaign, "run_winner_search", _short_search)
    monkeypatch.setattr(run_campaign, "select_winner", lambda _r: selected.append(1) or None)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: frozen.append(1))
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: tested.append(1) or [])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["min_candidates"] = None  # production default: floor = the FULL candidates budget (2)
    summary = run_campaign.run_headline_campaign(**kwargs)

    assert selected == [] and frozen == [] and tested == []  # nothing selected/frozen/tested
    s = summary["arms"][0]
    assert s["status"] == "search_incomplete" and s["n_accepted"] == 1 and s["required"] == 2
    assert summary["exit_code"] == run_campaign.EXIT_INCOMPLETE  # resumable failure, not a husk


def test_select_floor_counts_ledgered_failures_as_resolved(tmp_path, monkeypatch, _clear_shutdown):
    """A pool of (budget-1) accepted + 1 LEDGERED sandbox failure is COMPLETE: failures replay on
    --resume and can never be re-filled, so they must not brick the arm (mirrors the parallel
    matched_budget_ok semantics)."""
    import json as _json

    from src.io.results import write_run as _wr

    def _search_with_ledgered_failure(arm, *, archive_root, **_kw):  # noqa: ANN001
        arm_root = Path(archive_root) / arm
        _wr({"run_id": f"{arm}-g0-c0", "arm": arm, "seed": 0, "fold": 0,
             "candidate_id": f"{arm}-g0-c0", "generation": 0, "reward_source": _WINNER_SOURCE,
             "reward_source_hash": "h", "feedback_block": "", "metrics": {"val_fitness": 0.5},
             "wall_clock": 0.0, "env_fingerprint": "x"}, str(arm_root))
        (arm_root / "failures.jsonl").write_text(
            _json.dumps({"candidate_id": f"{arm}-g0-c1", "error": "sandbox: forbidden import"}) + "\n",
            encoding="utf-8",
        )
        return {"arm": arm}

    monkeypatch.setattr(run_campaign, "run_winner_search", _search_with_ledgered_failure)
    monkeypatch.setattr(run_campaign, "freeze_winner", lambda *a, **k: None)
    monkeypatch.setattr(run_campaign, "evaluate_winner_on_test", lambda *a, **k: [{"run_id": "x"}])
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["min_candidates"] = None  # floor = full budget (2): 1 accepted + 1 ledgered failure passes
    summary = run_campaign.run_headline_campaign(**kwargs)
    assert summary["arms"][0]["status"] == "tested"
    assert summary["exit_code"] == 0


def test_search_pool_counts_dedupes_across_ledgers(tmp_path) -> None:
    """_search_pool_counts unions failures across BOTH ledger layouts, de-duplicated by candidate_id
    (a completed arm writes the same failure to the crash-robust per-prefix ledger AND failures.jsonl)."""
    import json as _json

    from src.io.results import write_run as _wr

    root = tmp_path / "arm"
    _wr({"run_id": "a-g0-c0", "arm": "a", "seed": 0, "fold": 0, "candidate_id": "a-g0-c0",
         "generation": 0, "reward_source": _WINNER_SOURCE, "reward_source_hash": "h",
         "feedback_block": "", "metrics": {"val_fitness": 0.1}, "wall_clock": 0.0,
         "env_fingerprint": "x"}, str(root))
    row = _json.dumps({"candidate_id": "a-g0-c1", "error": "sandbox"}) + "\n"
    (root / "failures.jsonl").write_text(row, encoding="utf-8")           # end-of-arm ledger
    (root / "a-s0-a.failures.jsonl").write_text(row, encoding="utf-8")    # per-prefix append ledger
    n_accepted, n_failed = run_campaign._search_pool_counts(root)
    assert n_accepted == 1
    assert n_failed == 1  # the SAME candidate_id in both files counts once


# --------------------------------------------------------------------------- #
# M4 — resume threads through the SERIAL search fallback down to run_arm       #
# --------------------------------------------------------------------------- #
def test_run_winner_search_threads_resume_to_arm_runner() -> None:
    """run_winner_search(resume=...) reaches the injected arm_runner as run_arm's resume kwarg."""
    captured: dict = {}

    def _capture_runner(arm, **kw):  # noqa: ANN001
        captured["arm"] = arm
        captured.update(kw)
        return {"arm": arm}

    run_campaign.run_winner_search(
        "distributional", search_seed=3, archive_root="ignored", synthetic=True,
        candidates=2, generations=1, train_steps=10, n_trials=2,
        resume=True, arm_runner=_capture_runner,
    )
    assert captured["arm"] == "distributional"
    assert captured["resume"] is True and captured["seed"] == 3

    run_campaign.run_winner_search(
        "distributional", search_seed=3, archive_root="ignored", synthetic=True,
        candidates=2, generations=1, train_steps=10, n_trials=2,
        arm_runner=_capture_runner,
    )
    assert captured["resume"] is False  # default stays a fresh search (back-compatible)


def test_headline_serial_search_receives_campaign_resume(tmp_path, monkeypatch, _clear_shutdown):
    """run_headline_campaign(resume=True) threads resume into the serial run_winner_search call."""
    seen: dict = {}

    def _spy_search(arm, **kw):  # noqa: ANN001
        seen["resume"] = kw.get("resume")
        return {"arm": arm}

    monkeypatch.setattr(run_campaign, "run_winner_search", _spy_search)
    monkeypatch.setattr(run_campaign, "select_winner", lambda _r: None)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["resume"] = True
    run_campaign.run_headline_campaign(**kwargs)
    assert seen["resume"] is True


# --------------------------------------------------------------------------- #
# m10 — config-as-truth resume default + the C3b/M7-adjacent CLI surface        #
# --------------------------------------------------------------------------- #
def test_resolve_resume_config_default_and_cli_overrides() -> None:
    """config/campaign.yaml `resume:` is the default; --resume forces on; --no-resume beats everything."""
    assert run_campaign.resolve_resume(False, False, True) is True     # config-as-truth default
    assert run_campaign.resolve_resume(False, False, False) is False
    assert run_campaign.resolve_resume(True, False, False) is True     # explicit --resume wins over cfg
    assert run_campaign.resolve_resume(False, True, True) is False     # --no-resume forces fresh
    assert run_campaign.resolve_resume(True, True, True) is False      # --no-resume beats --resume too


def test_build_parser_exposes_min_candidates_and_no_resume() -> None:
    """--min-candidates defaults to None (=> full-budget floor) and --no-resume defaults off."""
    parser = run_campaign.build_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "min_candidates" in actions and actions["min_candidates"].default is None
    assert "no_resume" in actions and actions["no_resume"].default is False


def test_run_headline_campaign_baselines_only_mode(tmp_path, monkeypatch, _clear_shutdown):
    """Throughput lever E (2026-07-06): --baselines-only runs ONLY the H1 stage — no arm search, no
    H3 — and writes its OWN sentinel (baselines_summary.json; never campaign_summary.json, which a
    concurrent serial-search campaign owns) with all_arms_tested NULLED (an empty-arm-set 'True'
    would fool the watcher) and exit_code 0 when the stage is fully tested."""
    import json as _json

    search_called: list = []
    captured: dict = {}

    def _spy_baselines(names, **_kw):  # noqa: ANN001
        captured["names"] = list(names)
        base = dict(
            arm="baseline", seed=0, fold=0, generation=0, reward_source_hash="h",
            feedback_block="", wall_clock=0.0, env_fingerprint="x",
        )
        for n in names:
            write_run(
                {**base, "run_id": f"baseline_{n}-s0", "candidate_id": f"baseline_{n}-s0",
                 "metrics": {"val_fitness": 0.0}},
                Path(_kw["test_root"]) / f"baseline_{n}",
            )
        return [{"run_id": f"baseline_{n}-s0"} for n in names]

    monkeypatch.setattr(run_campaign, "run_winner_search",
                        lambda arm, **k: search_called.append(arm) or {"arm": arm})
    monkeypatch.setattr(run_campaign, "evaluate_baselines_on_test", _spy_baselines)
    monkeypatch.setattr(run_campaign, "make_agent_trainer_factory", lambda: (lambda *a, **k: None))

    kwargs = _campaign_kwargs(tmp_path, ["distributional"])
    kwargs["baseline_names"] = ["raw_return", "return_minus_cvar"]
    kwargs["baselines_only"] = True
    kwargs["run_h3_singleshot"] = True  # must be forced OFF by the mode
    summary = run_campaign.run_headline_campaign(**kwargs)

    assert search_called == []  # NO arm search ran
    assert captured["names"] == ["raw_return", "return_minus_cvar"]
    assert summary["baselines_only"] is True and summary["all_arms_tested"] is None
    assert summary["exit_code"] == 0  # the h1_baselines row is 'tested'
    assert not any(s["arm"].endswith("singleshot") for s in summary["arms"])  # H3 forced off
    out_dir = Path(kwargs["output_dir"])
    assert (out_dir / "baselines_summary.json").is_file()
    assert not (out_dir / "campaign_summary.json").is_file()  # the campaign's sentinel untouched
    on_disk = _json.loads((out_dir / "baselines_summary.json").read_text(encoding="utf-8"))
    assert on_disk["baselines_only"] is True


def test_resolve_baseline_names_guards_and_default():
    """row 30l: the R97 laptop guard, now directly testable — (a) refusal without
    --baselines-only, (b) unknown-name rejection, (c) the byte-identical config default,
    (d) a valid override passes."""
    import pytest

    camp = {"h1_baselines": ["raw_return", "return_minus_variance"]}
    assert run_campaign.resolve_baseline_names(None, False, camp) == [
        "raw_return", "return_minus_variance"]
    with pytest.raises(SystemExit, match="requires --baselines-only"):
        run_campaign.resolve_baseline_names(["log_growth"], False, camp)
    with pytest.raises(SystemExit, match="unknown REWARD_CANON key"):
        run_campaign.resolve_baseline_names(["not_a_reward"], True, camp)
    assert run_campaign.resolve_baseline_names(
        ["differential_downside_ratio", "log_growth"], True, camp) == [
        "differential_downside_ratio", "log_growth"]
