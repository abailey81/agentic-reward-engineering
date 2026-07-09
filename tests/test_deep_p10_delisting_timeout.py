"""Regression lock for two load-bearing PortfolioEnv behaviours verified first-hand by audit P10.

Both behaviours were previously guaranteed only by an inline code comment; this module pins them
with executable assertions (additive — it neither weakens nor duplicates the existing env tests).

1. **Delisting = ``liquidate_to_cash`` zero-fill, consumed correctly by the env.** A survivorship-free
   top-30 contains names that die mid-window; the default policy
   (``src/data/loaders.py::load_gold_panel(on_missing="liquidate_to_cash")``) PRESERVES the dead name's
   realised return on its delisting DAY and ZERO-FILLS every post-event row (``0.0``) — it never
   forward-fills (a bare ``.ffill()`` would repeat the final daily return and fabricate compounding
   wealth on a dead name; loaders final-audit #20). Here we build a panel already in that post-loader
   state and step a scripted episode, asserting the env consumes it correctly: the delisting-day return
   is felt, the dead name then contributes exactly ``0`` (no ffill leak), the weight vector stays length
   ``N+1`` and on the simplex as a column dies (no desync), and the episode only ever *truncates* at the
   window edge. (The loader policy itself is exercised against the real gold in ``tests/test_loaders.py``;
   this locks the ENV's consumption of the zero-filled panel.)

2. **Window-edge truncation preserves the SB3 value bootstrap (TimeLimit semantics).** The ``[start, end)``
   walk-forward window running out of data is a Gymnasium *truncation*, not a *termination* (the MDP has
   no absorbing state). Reporting it as ``truncated`` (not ``terminated``) is load-bearing: SB3's
   ``DummyVecEnv`` synthesises ``info['TimeLimit.truncated']`` from it (verified against the installed
   ``stable_baselines3`` 2.8.0, ``dummy_vec_env.py:66``), and ``ReplayBuffer`` stores that as ``timeouts``
   and samples ``dones * (1 - timeouts)`` (``buffers.py:278`` / ``:322``), so the value bootstrap
   ``reward + gamma * Q(next)`` is KEPT at every window edge — reporting the edge as ``terminated`` would
   zero it once per episode and bias the critic (``portfolio_env.step`` inline note).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.panel import Panel
from src.env.portfolio_env import PortfolioEnv, project_simplex
from src.utils.config import load_config


def _port_ret_reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
    """Stateless reward = realized portfolio return (mirrors the existing env-test helper)."""
    return float(port_ret), {}, None


# --------------------------------------------------------------------------- #
# (1) Delisting = liquidate_to_cash zero-fill, consumed correctly by the env   #
# --------------------------------------------------------------------------- #
def _delisting_panel(lookback: int, n_assets: int = 3, dead_idx: int = 2) -> tuple[Panel, int, int, int]:
    """A deterministic panel whose asset ``dead_idx`` delists mid-window.

    Mirrors the state ``load_gold_panel(on_missing="liquidate_to_cash")`` produces: the dead name's
    real return is preserved on its DELISTING DAY (row ``lookback+1``) and every post-event row is
    ZERO-FILLED (``0.0``), NOT forward-filled. The window rows are engineered so each step isolates one
    behaviour (only the named asset has a non-zero return that day)::

        step 0 (row lookback+0): ordinary, all alive
        step 1 (row lookback+1): DELISTING DAY -> only the dead asset moves (-0.30)
        step 2 (row lookback+2): post-event    -> only LIVE asset 0 moves (+0.10); dead == 0
        step 3 (row lookback+3): post-event    -> only LIVE asset 1 moves (+0.05); dead == 0
        step 4 (row lookback+4): post-event, the WINDOW EDGE (truncation)
    """
    k = 5
    n_days = lookback + k
    returns = np.full((n_days, n_assets), 0.001, dtype=np.float64)  # benign warmup + baseline
    returns[lookback + 0] = [0.001, 0.001, 0.001]  # ordinary day, all alive
    returns[lookback + 1] = [0.0, 0.0, -0.30]      # delisting DAY: the dead asset's realised drop
    returns[lookback + 2] = [0.10, 0.0, 0.0]       # post-event: only live asset 0 moves
    returns[lookback + 3] = [0.0, 0.05, 0.0]        # post-event: only live asset 1 moves
    returns[lookback + 4] = [0.02, 0.0, 0.0]        # post-event, window edge
    # liquidate_to_cash tail: the dead name is 0.0 from the FIRST post-event row onward (never ffill'd).
    returns[lookback + 2 :, dead_idx] = 0.0
    vix = np.full(n_days, 20.0, dtype=np.float64)
    dates = np.arange(n_days)  # integer index (Panel accepts int dates; never reaches a reward)
    panel = Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(n_assets, dtype=np.int64))
    return panel, lookback, k, dead_idx


def test_delisting_zero_fill_is_consumed_correctly_by_env_p10() -> None:
    """P10-(1): a delisted (zero-filled) name is consumed correctly across a scripted episode.

    (a) the DELISTING-DAY return is still consumed; (b) post-event the dead name contributes EXACTLY 0
    to the gross return (no ffill leak); (c) the weight vector stays length ``N+1`` and sums to 1 at every
    step as a column dies (no desync); (d) the episode TERMINATES never and TRUNCATES only at the edge.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    panel, lb, k, dead = _delisting_panel(lookback)

    # Data-level premise: liquidate_to_cash preserves the delisting-day return and ZERO-FILLS the tail
    # (NOT a forward-fill that would leave -0.30 on every post-event row).
    assert panel.returns[lb + 1, dead] == pytest.approx(-0.30)     # delisting-day return preserved
    assert np.all(panel.returns[lb + 2 :, dead] == 0.0)            # zero-filled tail, not the -0.30 leak

    env = PortfolioEnv(panel, cfg, _port_ret_reward, start=lb, end=lb + k)
    n_act = env.N + 1
    assert env.N == 3 and n_act == 4
    env.reset(seed=0)

    actions = [
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        np.array([1.0, 0.0, 2.0, -1.0], dtype=np.float64),  # weight on the dead asset -> a clear signal
        np.array([2.0, -1.0, 0.5, 0.3], dtype=np.float64),
        np.array([0.0, 1.0, -0.5, 0.2], dtype=np.float64),
        np.array([-1.0, 0.5, 0.0, 1.0], dtype=np.float64),
    ]
    flags: list[tuple[bool, bool]] = []
    for step, a in enumerate(actions):
        w = project_simplex(a, env.projection)  # identical to the env's own projection of this action
        _obs, _r, terminated, truncated, info = env.step(a)
        flags.append((terminated, truncated))

        # (c) weights stay aligned: length N+1 (incl. cash) and on the simplex, EVERY step.
        assert len(info["weights"]) == n_act
        assert info["weights"].sum() == pytest.approx(1.0, abs=1e-12)
        assert len(info["prev_weights"]) == n_act
        assert env.w_prev.shape == (n_act,)

        if step == 1:
            # (a) DELISTING DAY: only the dead asset moved (-0.30); gross must CONSUME that realised loss.
            assert info["gross"] == pytest.approx(-0.30 * float(w[dead]), abs=1e-12)
            assert info["gross"] < 0.0
        if step == 2:
            # (b) POST-EVENT: the dead asset's return is 0, so it contributes EXACTLY 0 even though it
            # still carries weight; gross reflects only the live asset (no leaked -0.30 * w[dead] term).
            assert float(w[dead]) > 0.0                                  # dead name still carries weight...
            assert float(w[dead]) * panel.returns[lb + 2, dead] == 0.0   # ...but contributes exactly 0
            assert info["gross"] == pytest.approx(0.10 * float(w[0]), abs=1e-12)

    # (d) truncation semantics: terminated NEVER, truncated ONLY at the window edge.
    assert [t for (t, _tr) in flags] == [False] * k
    assert [tr for (_t, tr) in flags] == [False, False, False, False, True]
    assert len(flags) == k


# --------------------------------------------------------------------------- #
# (2) Window-edge truncation preserves the SB3 value bootstrap                  #
# --------------------------------------------------------------------------- #
def _clean_env(n_assets: int = 2, k: int = 4) -> tuple[PortfolioEnv, int]:
    """A minimal all-alive env whose ONLY episode end is the window-edge truncation."""
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n_days = lookback + k
    returns = np.full((n_days, n_assets), 0.001, dtype=np.float64)
    vix = np.full(n_days, 20.0, dtype=np.float64)
    dates = np.arange(n_days)
    panel = Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(n_assets, dtype=np.int64))
    return PortfolioEnv(panel, cfg, _port_ret_reward, start=lookback, end=lookback + k), k


def test_window_edge_is_gymnasium_truncation_not_termination_p10() -> None:
    """P10-(2) env contract: the window edge is a Gymnasium TRUNCATION, never a termination.

    ``terminated`` is False on EVERY step; ``truncated`` fires exactly once, at the edge — the flag SB3
    reads to keep the value bootstrap alive (portfolio_env.step inline note).
    """
    env, k = _clean_env()
    env.reset(seed=0)
    flags: list[tuple[bool, bool]] = []
    for _ in range(k):
        _obs, _r, terminated, truncated, _info = env.step(np.zeros(env.N + 1))
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        flags.append((terminated, truncated))
    assert flags[:-1] == [(False, False)] * (k - 1)  # no premature end mid-window
    assert flags[-1] == (False, True)                # edge = truncation, not termination


def test_dummyvecenv_synthesizes_timelimit_truncated_at_edge_p10() -> None:
    """P10-(2) SB3 mapping: ``DummyVecEnv`` turns the env's truncation into ``info['TimeLimit.truncated']``.

    Verified against the installed stable_baselines3 2.8.0 (``dummy_vec_env.py:66``:
    ``info['TimeLimit.truncated'] = truncated and not terminated`` — no TimeLimit/Monitor wrapper needed).
    ``dones AND TimeLimit.truncated`` being True at the edge is only possible if the inner env reported
    ``truncated=True, terminated=False`` (else the ``and not terminated`` would be False) — so this also
    re-confirms the env contract THROUGH SB3.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    _probe, k = _clean_env()
    n_act = _probe.N + 1
    # SB3 requires a factory that returns a FRESH env instance per call.
    venv = DummyVecEnv([lambda: _clean_env()[0]])
    venv.reset()
    actions = np.zeros((1, n_act), dtype=np.float32)
    seen: list[tuple[bool, bool]] = []
    for _ in range(k):
        _obs, _rew, dones, infos = venv.step(actions)
        seen.append((bool(dones[0]), bool(infos[0]["TimeLimit.truncated"])))

    assert seen[0] == (False, False)                       # mid-window: not done, not a timeout
    assert seen[-1] == (True, True)                        # edge: done AND flagged as a timeout (truncation)
    assert all((not d and not tl) for (d, tl) in seen[:-1])


def test_replay_buffer_timeout_flag_preserves_value_bootstrap_p10() -> None:
    """P10-(2) bootstrap: the ``timeouts`` flag makes ``dones*(1-timeouts)=0`` at the window edge.

    A ``TimeLimit.truncated`` transition sets the buffer's ``timeouts`` flag, so the sampled ``dones``
    are ``dones*(1-timeouts) = 0`` at the edge — SB3 keeps the value bootstrap (``reward + gamma*Q(next)``);
    a genuine termination (no timeout) keeps ``dones = 1`` and drops it. Verified against the installed
    stable_baselines3 2.8.0 (``buffers.py:278`` stores the flag; ``buffers.py:322`` returns
    ``dones*(1-timeouts)``).
    """
    import gymnasium as gym
    from stable_baselines3.common.buffers import ReplayBuffer

    obs_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    buf = ReplayBuffer(10, obs_space, act_space, device="cpu", handle_timeout_termination=True)

    obs = np.zeros((1, 2), dtype=np.float32)
    action = np.zeros((1, 1), dtype=np.float32)
    reward = np.zeros((1,), dtype=np.float32)
    done = np.array([True])

    buf.add(obs, obs, action, reward, done, [{"TimeLimit.truncated": True}])   # pos 0: WINDOW-EDGE timeout
    buf.add(obs, obs, action, reward, done, [{"TimeLimit.truncated": False}])  # pos 1: genuine termination

    # buffers.py:278 — the per-transition timeout flag (both are episode-ends -> dones==1).
    assert buf.dones[0, 0] == 1.0 and buf.dones[1, 0] == 1.0
    assert buf.timeouts[0, 0] == 1.0  # the truncation is a timeout
    assert buf.timeouts[1, 0] == 0.0  # the true terminal is not

    # buffers.py:322 — sampling returns dones*(1-timeouts): 0 at the edge (bootstrap KEPT), 1 at a true end.
    edge = buf._get_samples(np.array([0]))
    term = buf._get_samples(np.array([1]))
    assert float(edge.dones.reshape(-1)[0]) == 0.0  # value bootstrap PRESERVED at the window edge
    assert float(term.dones.reshape(-1)[0]) == 1.0  # bootstrap dropped only at a genuine termination
