"""Tests for the SAC trainer + C1 evaluator + the full loop (MASTER_EXECUTION_PLAN P3).

These are SLOW (they train a real Stable-Baselines3 SAC for a few hundred steps), so they
are marked ``slow`` and deselected by the default ``-m "not slow"`` run. They verify:
  - train_agent returns an obs-normalising, predicting policy (NormalizedPolicy);
  - the memory-safe buffer is honoured (buffer_size == train_steps; ADR-025);
  - evaluate_reward (the C1 search-arm adapter) trains + scores to a finite fitness;
  - a 2-candidate / 1-generation run_loop with the REAL trainer + runner archives
    reloadable candidates, the winner is selected on validation fitness, and the
    distributional arm's feedback block carries the tail statistics.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.evaluator import evaluate_reward
from src.agents.trainer import NormalizedPolicy, make_agent_trainer, train_agent
from src.data.panel import Panel
from src.env.runner import make_env_builder
from src.selection.fitness import held_out_fitness
from src.utils.config import load_config

pytestmark = pytest.mark.slow

_VALID_REWARD_SRC = """
def reward(weights, returns, prev_weights, port_ret, info):
    total = float(port_ret) - 0.001 * float(np.sum(np.abs(weights - prev_weights)))
    return total, {"port_ret": float(port_ret)}, None
"""


def _reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
    pr = float(port_ret)
    return pr, {"port_ret": pr}, None


@pytest.fixture
def env_cfg():
    return load_config("environment")


@pytest.fixture
def small_train_cfg():
    # Tiny budget for a fast slow-test; buffer auto-sizes to train_steps (ADR-025).
    return {"train_steps_per_candidate": 400, "learning_starts": 100, "normalize_obs": True}


def test_train_agent_returns_normalizing_policy_and_finite_rollout(
    synthetic_panel: Panel, env_cfg, small_train_cfg
) -> None:
    """train_agent returns a NormalizedPolicy whose rollout yields finite returns."""
    lookback = int(env_cfg["state"]["lookback_days"])
    builder = make_env_builder(synthetic_panel, env_cfg, (lookback, 400), (400, synthetic_panel.T))
    bundle = builder(_reward)
    policy = train_agent(bundle.train_env(), small_train_cfg, seed=0)

    assert isinstance(policy, NormalizedPolicy)
    assert np.isfinite(policy._mean).all() and np.isfinite(policy._std).all()
    assert (policy._std > 0).all()
    val_rets = bundle.val_returns(policy)
    assert val_rets.shape == (synthetic_panel.T - 400,)
    assert np.isfinite(val_rets).all()


def test_buffer_size_is_memory_safe(synthetic_panel: Panel, env_cfg) -> None:
    """The replay buffer is sized to train_steps, not SB3's 1e6 default (ADR-025)."""
    from src.agents.trainer import resolve_agent_kwargs

    kwargs, train_steps = resolve_agent_kwargs({"train_steps_per_candidate": 400}, seed=0)
    assert kwargs["buffer_size"] == train_steps == 400  # not 1_000_000


def test_train_agent_without_norm_returns_raw_model(
    synthetic_panel: Panel, env_cfg
) -> None:
    """With normalize_obs off, train_agent returns the raw SB3 model (still predicts)."""
    lookback = int(env_cfg["state"]["lookback_days"])
    builder = make_env_builder(synthetic_panel, env_cfg, (lookback, 300), (300, synthetic_panel.T))
    bundle = builder(_reward)
    cfg = {"train_steps_per_candidate": 300, "normalize_obs": False}
    policy = train_agent(bundle.train_env(), cfg, seed=1)
    assert not isinstance(policy, NormalizedPolicy)
    rets = bundle.val_returns(policy)
    assert np.isfinite(rets).all()


def test_evaluate_reward_trains_and_scores_C1(
    synthetic_panel: Panel, env_cfg, small_train_cfg
) -> None:
    """C1 acceptance: the search-arm evaluator trains the fixed SAC and returns a fitness."""
    lookback = int(env_cfg["state"]["lookback_days"])
    env_builder = make_env_builder(
        synthetic_panel, env_cfg, (lookback, 400), (400, synthetic_panel.T)
    )
    agent_trainer = make_agent_trainer(small_train_cfg, seed=0)
    fitness = evaluate_reward(
        _reward, env_builder, agent_trainer, held_out_fitness, n_trials=5
    )
    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_loop_end_to_end_with_real_trainer(
    synthetic_panel: Panel, env_cfg, small_train_cfg, tmp_path
) -> None:
    """A 2-candidate / 1-gen run_loop with the REAL trainer + runner archives candidates."""
    from src.feedback.measurement import ReturnDistribution
    from src.io.results import load_run
    from src.llm.client import FakeTransport, LLMClient
    from src.llm.loop import run_loop

    lookback = int(env_cfg["state"]["lookback_days"])
    env_builder = make_env_builder(
        synthetic_panel, env_cfg, (lookback, 420), (420, synthetic_panel.T)
    )
    agent_trainer = make_agent_trainer(small_train_cfg, seed=0)
    llm = LLMClient({"model": "stub"}, transport=FakeTransport(response=_VALID_REWARD_SRC))
    loop_cfg = {
        "generations": 1,
        "candidates_per_gen": 2,
        "budget": 2,
        "seed": 0,
        "n_trials": 2,
        "model": "stub",
        "run_prefix": "itest",
        "fold": 0,
    }

    archive = run_loop(
        "distributional",
        env_builder,
        llm,
        agent_trainer,
        ReturnDistribution,
        held_out_fitness,
        loop_cfg,
        tmp_path,
    )

    assert len(archive.candidates) == 2
    assert archive.winner() is not None
    assert all(np.isfinite(c.val_fitness) for c in archive.candidates)
    # The distributional arm's feedback block carries the tail statistics (CVaR etc.).
    assert "cvar" in archive.candidates[0].feedback_block.lower()
    # Artifacts round-trip through the results IO (replay, audit C-2).
    rec = load_run("itest-distributional-g0-c0", tmp_path)
    assert rec["arm"] == "distributional"
