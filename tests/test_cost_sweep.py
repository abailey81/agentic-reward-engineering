"""FAST tests for the transaction-cost robustness sweep (Rank 15, scripts/cost_sweep.py).

These are torch-free: a deterministic fake policy rolls through the REAL ``PortfolioEnv``, so
they pin the load-bearing invariants of the cost-sweep arm:

  - the ``cost_bps`` OVERRIDE sets ``self.cost = cost_bps*1e-4`` and changes the realized cost
    by exactly that factor (and ``cost_bps=None`` reproduces the config-headline behaviour);
  - ANALYTIC re-pricing ``net_c = gross − c·turnover`` matches RE-ROLLING the SAME frozen
    policy through a ``cost_bps``-overridden env to ~1e-12 on a small fixture, AND a
    higher cost yields a strictly lower (or equal) net whenever any trading happens;
  - the sweep winner-identity table has exactly ONE row per cost level, aligned to the grid;
  - the analytic re-price at the headline bps reproduces the headline net series.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.panel import Panel
from src.env.portfolio_env import PortfolioEnv, project_simplex
from src.env.runner import make_env_builder, rollout_port_series
from src.utils.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import cost_sweep  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes / fixtures (no torch)                                                  #
# --------------------------------------------------------------------------- #
class _FakePolicy:
    """Predicts a fixed non-trivial action so turnover is genuinely nonzero. No torch."""

    def __init__(self, n_act: int, action: np.ndarray | None = None) -> None:
        self.n_act = n_act
        self._action = (
            np.asarray(action, dtype=np.float32)
            if action is not None
            else np.linspace(-1.0, 1.0, n_act, dtype=np.float32)
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True):  # noqa: ARG002
        return self._action, None


def _identity_reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
    pr = float(port_ret)
    return pr, {"port_ret": pr}, None


@pytest.fixture
def env_cfg():
    return load_config("environment")


def _make_panel(n: int = 4, n_days: int = 200, seed: int = 7) -> Panel:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((n_days, n)) * 0.01
    return Panel(
        returns=returns,
        vix=np.full(n_days, 20.0),
        dates=np.arange(n_days),
        asset_ids=np.arange(n),
    )


# --------------------------------------------------------------------------- #
# (1) cost_bps override -> self.cost + realized cost                           #
# --------------------------------------------------------------------------- #
def test_cost_bps_override_sets_self_cost(env_cfg):
    """cost_bps=c sets self.cost = c*1e-4; None falls back to the config headline."""
    panel = _make_panel()
    lookback = int(env_cfg["state"]["lookback_days"])

    default_env = PortfolioEnv(panel, env_cfg, _identity_reward, start=lookback, end=lookback + 5)
    headline = float(env_cfg["costs"]["headline_bps"]) * 1e-4
    assert default_env.cost == pytest.approx(headline, abs=1e-18)
    assert default_env.cost_bps is None

    for bps in (0.0, 5.0, 10.0, 25.0, 50.0):
        env = PortfolioEnv(
            panel, env_cfg, _identity_reward, start=lookback, end=lookback + 5, cost_bps=bps
        )
        assert env.cost == pytest.approx(bps * 1e-4, abs=1e-18)
        assert env.cost_bps == pytest.approx(bps)


def test_cost_bps_override_scales_realized_cost(env_cfg):
    """The realized info['cost'] scales linearly in cost_bps for a FIXED policy step.

    Same panel + same action -> identical turnover at every cost level, so the per-step cost
    is exactly (cost_bps*1e-4) * turnover. Charging 2x the bps charges 2x the cost.
    """
    panel = _make_panel()
    lookback = int(env_cfg["state"]["lookback_days"])
    action = np.array([2.0, -1.0, 0.5, 0.0, -0.3], dtype=np.float64)  # N=4 -> n_act=5

    def _first_step(bps):
        env = PortfolioEnv(
            panel, env_cfg, _identity_reward, start=lookback, end=lookback + 1, cost_bps=bps
        )
        env.reset(seed=0)
        _o, _r, _t, _tr, info = env.step(action)
        return info

    info10 = _first_step(10.0)
    info20 = _first_step(20.0)
    info0 = _first_step(0.0)

    # Turnover is policy-determined, identical across cost levels.
    assert info10["turnover"] == pytest.approx(info20["turnover"], abs=1e-15)
    # Cost is linear in bps; gross is cost-independent; net = gross - cost.
    assert info10["cost"] == pytest.approx(10.0 * 1e-4 * info10["turnover"], abs=1e-15)
    assert info20["cost"] == pytest.approx(2.0 * info10["cost"], abs=1e-15)
    assert info0["cost"] == pytest.approx(0.0, abs=1e-18)
    assert info10["gross"] == pytest.approx(info20["gross"], abs=1e-15)
    assert info10["port_ret"] == pytest.approx(info10["gross"] - info10["cost"], abs=1e-15)


def test_make_env_builder_threads_cost_bps(env_cfg):
    """make_env_builder(cost_bps=c) -> every env it builds is priced at c (and bundle records it)."""
    panel = _make_panel(n_days=300)
    lookback = int(env_cfg["state"]["lookback_days"])
    # Embargo-respecting: val starts >= 120+21, test starts >= 200+21.
    builder = make_env_builder(
        panel, env_cfg, (lookback, 120), (150, 200), test_window=(230, 300),
        embargo=21, cost_bps=25.0,
    )
    bundle = builder(_identity_reward)
    assert bundle.cost_bps == pytest.approx(25.0)
    assert bundle.train_env().cost == pytest.approx(25.0 * 1e-4, abs=1e-18)


# --------------------------------------------------------------------------- #
# (2) analytic re-pricing == re-roll through cost_bps-overridden env (~1e-12)   #
# --------------------------------------------------------------------------- #
def test_analytic_reprice_matches_reroll(env_cfg):
    """net_c = gross − c·turnover (analytic) matches re-rolling a cost_bps-overridden env.

    Roll the frozen policy ONCE at any cost to capture (gross, turnover) — these are
    cost-independent — then for every grid level assert the analytic net equals the net the
    env produces when re-rolled at that exact cost_bps, to machine precision.
    """
    panel = _make_panel(n=4, n_days=200, seed=11)
    lookback = int(env_cfg["state"]["lookback_days"])
    test_window = (lookback, 200)
    policy = _FakePolicy(panel.N + 1, action=np.array([1.5, -0.5, 0.2, 0.8, -1.0]))

    # One reference roll to grab the cost-independent gross/turnover.
    ref_env = PortfolioEnv(panel, env_cfg, _identity_reward, start=lookback, end=200, cost_bps=10.0)
    ref = rollout_port_series(ref_env, policy)
    gross, turnover = ref["gross"], ref["turnover"]

    grid = [float(b) for b in env_cfg["costs"]["grid_bps"]]
    assert grid  # non-empty (config declares [0,5,10,25,50])
    for bps in grid:
        analytic = cost_sweep.reprice_analytic(gross, turnover, bps)
        rerolled = cost_sweep.reprice_reroll(
            panel, env_cfg, test_window, _identity_reward, policy, bps
        )
        # Gross/turnover are identical to the reference roll (policy-determined).
        assert np.allclose(rerolled["gross"], gross, atol=1e-12, rtol=0.0)
        assert np.allclose(rerolled["turnover"], turnover, atol=1e-12, rtol=0.0)
        # The analytic net equals the re-rolled net to ~1e-12.
        assert np.allclose(analytic, rerolled["net"], atol=1e-12, rtol=0.0)

    # Monotonicity sanity: a higher cost never raises the net (turnover >= 0 everywhere).
    net0 = cost_sweep.reprice_analytic(gross, turnover, 0.0)
    net50 = cost_sweep.reprice_analytic(gross, turnover, 50.0)
    assert np.all(net50 <= net0 + 1e-15)
    assert turnover.max() > 0.0  # the fixture really does trade


def test_headline_reprice_reproduces_headline_net(env_cfg):
    """Analytic re-price at the headline bps reproduces the default-cost env's net series."""
    panel = _make_panel(n=4, n_days=180, seed=3)
    lookback = int(env_cfg["state"]["lookback_days"])
    policy = _FakePolicy(panel.N + 1)
    headline_bps = float(env_cfg["costs"]["headline_bps"])

    # Default env (cost_bps=None) == headline cost.
    default_env = PortfolioEnv(panel, env_cfg, _identity_reward, start=lookback, end=180)
    series = rollout_port_series(default_env, policy)
    analytic = cost_sweep.reprice_analytic(series["gross"], series["turnover"], headline_bps)
    assert np.allclose(analytic, series["net"], atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# (3) the sweep table has one row per cost level                               #
# --------------------------------------------------------------------------- #
def _decomp_record(arm: str, gross, turnover, seed: int = 0) -> dict:
    """A minimal frozen-winner-shaped record carrying the gross/turnover decomposition."""
    net = (np.asarray(gross) - 10.0 * 1e-4 * np.asarray(turnover)).tolist()
    return {
        "run_id": f"{arm}-s{seed}",
        "arm": arm,
        "seed": seed,
        "metrics": {
            "test_returns": net,
            "test_gross": list(np.asarray(gross, dtype=float)),
            "test_turnover": list(np.asarray(turnover, dtype=float)),
        },
    }


def test_cost_curve_and_winner_table_one_row_per_level():
    """cost_curve re-prices analytically; winner_by_cost has exactly one row per cost level."""
    rng = np.random.default_rng(0)
    t = 64
    # Two arms with distinct gross/turnover profiles.
    g_a = rng.standard_normal(t) * 0.01 + 0.001
    tn_a = np.abs(rng.standard_normal(t)) * 0.2
    g_b = rng.standard_normal(t) * 0.01 + 0.0005
    tn_b = np.abs(rng.standard_normal(t)) * 0.05  # trades less
    records = [
        _decomp_record("distributional", g_a, tn_a),
        _decomp_record("scalar", g_b, tn_b),
    ]
    grid = [0.0, 5.0, 10.0, 25.0, 50.0]

    curve = cost_sweep.cost_curve(records, grid, arms=("distributional", "scalar"))
    assert curve["grid_bps"] == grid
    for arm in ("distributional", "scalar"):
        assert curve["arms"][arm]["source"] == "analytic"
        assert len(curve["arms"][arm]["sharpe"]) == len(grid)
        assert len(curve["arms"][arm]["cvar"]) == len(grid)

    rows = cost_sweep.winner_by_cost(curve)
    # Exactly one row per cost level, aligned to the grid.
    assert len(rows) == len(grid)
    assert [r["bps"] for r in rows] == grid
    for r in rows:
        assert r["winner"] in ("distributional", "scalar")
        assert set(r["per_arm"].keys()) == {"distributional", "scalar"}

    # The markdown table also has exactly one winner-row per cost level.
    md = cost_sweep.sweep_markdown(curve, rows)
    assert "Winner identity vs cost" in md
    for b in grid:
        assert f"| {b:g} |" in md


def test_cost_curve_skips_arm_without_decomposition():
    """An arm whose records carry only NET test_returns is reported as skipped (not fabricated)."""
    rng = np.random.default_rng(1)
    net_only = {
        "run_id": "placebo-s0",
        "arm": "placebo",
        "seed": 0,
        "metrics": {"test_returns": list(rng.standard_normal(40) * 0.01)},
    }
    curve = cost_sweep.cost_curve([net_only], [0.0, 10.0, 50.0], arms=("placebo",))
    assert "placebo" not in curve["arms"]
    assert "placebo" in curve["skipped"]


def test_arm_decomposition_averages_across_seeds():
    """arm_test_decomposition pools per-seed (gross, turnover) by the per-period mean."""
    g0 = np.array([0.01, 0.02, 0.03])
    g1 = np.array([0.03, 0.02, 0.01])
    tn0 = np.array([0.1, 0.2, 0.3])
    tn1 = np.array([0.3, 0.2, 0.1])
    records = [
        _decomp_record("distributional", g0, tn0, seed=0),
        _decomp_record("distributional", g1, tn1, seed=1),
    ]
    decomp = cost_sweep.arm_test_decomposition(records, "distributional")
    assert decomp is not None
    assert np.allclose(decomp["gross"], (g0 + g1) / 2.0)
    assert np.allclose(decomp["turnover"], (tn0 + tn1) / 2.0)


def test_softmax_invariant_action_keeps_test_helpers_importable():
    """A trivial guard that the env helpers used by the sweep import + project cleanly."""
    w = project_simplex(np.array([0.0, 0.0, 0.0]), "softmax")
    assert w.sum() == pytest.approx(1.0)
