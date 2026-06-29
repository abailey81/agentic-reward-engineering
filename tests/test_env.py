"""Tests for src/env/portfolio_env.py — construction, stepping, no-NaN (F.1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.panel import Panel
from src.env.portfolio_env import PortfolioEnv, project_simplex
from src.utils.config import load_config


def _port_ret_reward(weights, returns, prev_weights, port_ret, info):
    """A reward returning the portfolio return; round-trips a step-counter state."""
    state = info.get("reward_state")
    count = 0 if state is None else int(state) + 1
    return float(port_ret), {"port_ret": float(port_ret)}, count


@pytest.fixture
def env(synthetic_panel: Panel) -> PortfolioEnv:
    cfg = load_config("environment")
    return PortfolioEnv(synthetic_panel, cfg, _port_ret_reward)


def test_env_constructs_with_injected_reward_fn(env: PortfolioEnv) -> None:
    """PortfolioEnv accepts (panel, cfg, reward_fn) and exposes gym spaces (A-4)."""
    assert env.action_space.shape == (env.N + 1,)
    assert env.observation_space.shape[0] == env._obs_dim()
    assert np.dtype(env.observation_space.dtype) == np.float32


def test_reset_returns_obs_in_observation_space(env: PortfolioEnv) -> None:
    """reset returns (obs, info); obs lies in the observation space."""
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32


def test_action_always_on_simplex_under_frozen_projection(env: PortfolioEnv) -> None:
    """Every projected action is non-negative and sums to 1 (frozen projection, C-8)."""
    rng = np.random.default_rng(1)
    env.reset(seed=0)
    for _ in range(50):
        action = rng.standard_normal(env.N + 1) * 5.0
        w = project_simplex(action, env.projection)
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0)
        env.step(action)


def test_log_wealth_has_no_nans_across_full_panel(env: PortfolioEnv) -> None:
    """Stepping through the whole panel produces no NaN in obs or log-wealth."""
    obs, _ = env.reset(seed=0)
    assert np.isfinite(obs).all()
    rng = np.random.default_rng(2)
    terminated = truncated = False
    # The window-exhaustion boundary is a Gymnasium TRUNCATION, not a termination (audit 2026-06-20):
    # the MDP has no absorbing terminal state, so the episode ends on ``truncated`` and ``terminated``
    # stays False throughout (SB3 then bootstraps the boundary value — see portfolio_env.step).
    steps = 0
    while not (terminated or truncated):
        action = rng.standard_normal(env.N + 1)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(env.log_wealth)
        steps += 1
    assert truncated and not terminated  # ended by truncation at the window edge, never termination
    assert steps > 0
    assert np.isfinite(env.log_wealth)


def test_step_returns_gymnasium_five_tuple(env: PortfolioEnv) -> None:
    """step returns (obs, reward, terminated, truncated, info) with reward a float."""
    env.reset(seed=0)
    out = env.step(np.zeros(env.N + 1))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_state_round_trips_via_info(env: PortfolioEnv) -> None:
    """reward_state from one step is delivered to the next via info (stateful, B-4)."""
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.zeros(env.N + 1))
    assert info0["reward_state"] == 0
    _, _, _, _, info1 = env.step(np.zeros(env.N + 1))
    assert info1["reward_state"] == 1
    _, _, _, _, info2 = env.step(np.zeros(env.N + 1))
    assert info2["reward_state"] == 2


def test_cost_is_half_l1_drifted_turnover() -> None:
    """cost == 0.5 * c * |w - w_held|_1 and info['turnover'] match a hand-computed example.

    Half-L1-DRIFTED transaction cost ("Dynamics & accounting"):
    the book the agent HOLDS has already drifted by the return it earned (r_{t-1}) before
    the agent trades, so cost is paid only on the gap to that drifted book, and on the
    ONE-WAY (½) turnover.

    TWO steps are needed to exercise the drift at all: step 1 establishes a position out of
    the uniform reset book, that position drifts through r0, and step 2 trades against the
    drifted result. The two return rows are deliberately far apart in size so the assertion
    is DECISIVE about the index — drifting by the contemporaneous row r1 instead of the
    earned row r0 (the #92 look-ahead this test used to lock in) misses by far more than
    the 1e-12 tolerance.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n = 2  # two risky assets + cash -> n_act = 3
    t = lookback + 2  # two tradeable rows

    rng = np.random.default_rng(0)
    returns = rng.standard_normal((t, n)) * 1e-3
    r0 = np.array([0.10, -0.05], dtype=np.float64)  # earned by the step-1 book
    r1 = np.array([-0.02, 0.03], dtype=np.float64)  # realized during step 2
    returns[lookback] = r0
    returns[lookback + 1] = r1
    panel = Panel(
        returns=returns,
        vix=np.full(t, 20.0),
        dates=np.arange(t),
        asset_ids=np.arange(n),
    )

    c = float(cfg["costs"]["headline_bps"]) * 1e-4

    env = PortfolioEnv(panel, cfg, _port_ret_reward, start=lookback, end=t)
    env.reset(seed=0)

    # Known raw actions; the env softmax-projects them -> the same w we recompute here.
    action1 = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    action2 = np.array([-0.5, 1.5, 0.25], dtype=np.float64)
    w1 = project_simplex(action1, env.projection)
    w2 = project_simplex(action2, env.projection)

    # Step 1: nothing has drifted yet, so the held book is the uniform reset book.
    w_reset = np.full(n + 1, 1.0 / (n + 1), dtype=np.float64)
    _o, _r, _te, _tr, info1 = env.step(action1)
    assert info1["turnover"] == pytest.approx(0.5 * float(np.abs(w1 - w_reset).sum()), abs=1e-12)
    assert info1["gross"] == pytest.approx(float(w1[:n] @ r0), abs=1e-12)

    # Step 2: w1 held through day 0, so it drifted by r0 — the return it actually earned.
    growth0 = np.array([1.0 + r0[0], 1.0 + r0[1], 1.0], dtype=np.float64)  # cash grows at 1.0
    w_held = w1 * growth0 / float(w1 @ growth0)
    expected_turnover = 0.5 * float(np.abs(w2 - w_held).sum())
    expected_cost = c * expected_turnover
    expected_gross = float(w2[:n] @ r1)
    expected_port_ret = expected_gross - expected_cost

    _obs, _reward, _term, _trunc, info = env.step(action2)

    assert info["turnover"] == pytest.approx(expected_turnover, abs=1e-12)
    assert info["cost"] == pytest.approx(expected_cost, abs=1e-12)
    assert info["gross"] == pytest.approx(expected_gross, abs=1e-12)
    assert info["port_ret"] == pytest.approx(expected_port_ret, abs=1e-12)
    # The drift makes turnover STRICTLY less than the naive full-undrifted L1 (the old bug),
    # confirming both the ½ factor and the drift are applied (not just one of them).
    assert expected_turnover < float(np.abs(w2 - w1).sum())
    # DECISIVE on the INDEX (#92): drifting by the contemporaneous row r1 is a materially
    # different charge, so this test can no longer pass under the look-ahead ledger.
    growth1 = np.array([1.0 + r1[0], 1.0 + r1[1], 1.0], dtype=np.float64)
    w_lookahead = w1 * growth1 / float(w1 @ growth1)
    turnover_lookahead = 0.5 * float(np.abs(w2 - w_lookahead).sum())
    assert abs(turnover_lookahead - expected_turnover) > 1e-6


def test_turnover_is_zero_when_target_equals_drifted_weights() -> None:
    """Targeting the book you already hold costs exactly nothing — buy-and-hold is free.

    This isolates the drift term: not trading means w == w_held, so the half-L1-drifted
    turnover is exactly 0 even though w != w_prev (the raw, undrifted full-L1 model would
    wrongly charge a cost here).

    It is also the property #92 RESTORED. The held book is a function of w_prev and the
    already-observed row r_{t-1}, so a policy can compute this target and hit it; under the
    previous drift-by-r_t ledger the zero-turnover target depended on the unobserved
    contemporaneous return, so buy-and-hold was unreachable at any price and paid a measured
    0.139 %/yr floor it could neither avoid nor predict.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n = 2
    t = lookback + 2
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((t, n)) * 1e-3
    r0 = np.array([0.20, 0.08], dtype=np.float64)
    returns[lookback] = r0
    returns[lookback + 1] = np.array([-0.11, 0.04], dtype=np.float64)
    panel = Panel(returns=returns, vix=np.full(t, 20.0), dates=np.arange(t), asset_ids=np.arange(n))

    env = PortfolioEnv(panel, cfg, _port_ret_reward, start=lookback, end=t)
    env.reset(seed=0)

    # Step 1 establishes a position; softmax(log p) == p pins it exactly on the simplex.
    w1 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    env.step(np.log(w1))

    # It drifted by the return it EARNED, r0. Target exactly that -> no trade, no cost.
    growth0 = np.array([1.0 + r0[0], 1.0 + r0[1], 1.0], dtype=np.float64)
    w_held = w1 * growth0 / float(w1 @ growth0)

    _obs, _reward, _term, _trunc, info = env.step(np.log(w_held))
    assert info["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info["cost"] == pytest.approx(0.0, abs=1e-12)


def test_cost_ledger_has_no_contemporaneous_look_ahead() -> None:
    """Corrupting returns[t] must not move the turnover charged at step t (#92).

    The OBSERVATION's no-look-ahead property is tested adversarially in
    ``tests/test_env_nolookahead.py``; the COST LEDGER was the unguarded flank, and the
    paper's no-look-ahead claim (CH4) is scoped to the observation only. Until #92 the
    charged turnover was a function of ``returns[t]`` itself — the contemporaneous row the
    agent cannot see when it acts — so an identical policy paid a different price depending
    on a return that had not been revealed yet. Only ``gross`` may depend on ``returns[t]``.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n = 2
    t = lookback + 2

    def turnover_under(second_row: np.ndarray) -> float:
        rng = np.random.default_rng(7)
        returns = rng.standard_normal((t, n)) * 1e-3
        returns[lookback] = np.array([0.05, -0.02], dtype=np.float64)
        returns[lookback + 1] = second_row
        panel = Panel(
            returns=returns, vix=np.full(t, 20.0), dates=np.arange(t), asset_ids=np.arange(n)
        )
        env = PortfolioEnv(panel, cfg, _port_ret_reward, start=lookback, end=t)
        env.reset(seed=0)
        env.step(np.array([2.0, -1.0, 0.5], dtype=np.float64))
        _o, _r, _te, _tr, info = env.step(np.array([-0.5, 1.5, 0.25], dtype=np.float64))
        return float(info["turnover"])

    benign = turnover_under(np.array([0.01, 0.01], dtype=np.float64))
    violent = turnover_under(np.array([-0.60, 0.45], dtype=np.float64))
    assert benign == pytest.approx(violent, abs=1e-15)


# --- V15a: untrusted reward must NOT corrupt SHARED env state across steps/candidates ---


def _make_env(reward_fn, n: int = 4, extra_rows: int = 6):
    """Build a small deterministic env around ``reward_fn`` (V15 regression helper)."""
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    t = lookback + extra_rows
    rng = np.random.default_rng(0)
    returns = (rng.standard_normal((t, n)) * 1e-2).astype(np.float64)
    panel = Panel(returns=returns, vix=np.full(t, 20.0), dates=np.arange(t), asset_ids=np.arange(n))
    env = PortfolioEnv(panel, cfg, reward_fn, start=lookback, end=t)
    return env, panel, lookback


def test_reward_inplace_write_to_returns_cannot_corrupt_shared_panel() -> None:
    """A reward doing ``returns[:] = 0`` must NOT mutate the shared gold panel (V15a).

    The realized-return vector is a row of the frozen, shared panel; before the fix it was passed
    as a writable view, so an in-place write zeroed that row and corrupted every later step/candidate
    that replays it (a determinism / no-cross-contamination violation, not an RCE). The reward now
    receives a READ-ONLY copy, so the write raises inside the reward (caught by safe_call) and the
    panel is untouched.
    """

    def malicious(weights, returns, prev_weights, port_ret, info):
        returns[:] = 0.0  # in-place write attempt on a shared-panel row
        return float(port_ret), {}, None

    env, panel, lookback = _make_env(malicious)
    env.reset(seed=0)
    row_before = panel.returns[lookback].copy()
    env.step(np.zeros(env.N + 1))
    np.testing.assert_array_equal(
        panel.returns[lookback], row_before
    )  # shared panel row is intact
    assert np.count_nonzero(panel.returns[lookback]) > 0  # was NOT zeroed


def test_reward_inplace_write_to_weights_cannot_corrupt_env_state() -> None:
    """A reward writing in place to ``weights`` / ``prev_weights`` must not corrupt env state (V15a).

    ``weights`` becomes the env's next ``w_prev``; before the fix an in-place ``weights[:] = -7``
    leaked negative weights into the env (no longer a valid simplex). The reward now gets read-only
    views, so the write raises (caught) and the env keeps a valid simplex weight.
    """

    def malicious(weights, returns, prev_weights, port_ret, info):
        weights[:] = -7.0
        prev_weights[:] = -7.0
        return float(port_ret), {}, None

    env, _panel, _lookback = _make_env(malicious)
    env.reset(seed=0)
    env.step(np.full(env.N + 1, 0.3))
    assert np.all(env.w_prev >= 0.0)  # still a valid simplex point, not corrupted to -7
    assert env.w_prev.sum() == pytest.approx(1.0)


def test_reward_cannot_pollute_env_info_dict() -> None:
    """A reward injecting/clobbering an ``info`` key must not affect the env's returned info (V15a).

    The reward is handed a SHALLOW COPY of info, so its mutations stay local; the env's own logging
    dict (returned to SB3) is unpolluted and carries exactly the env-emitted keys.
    """

    def malicious(weights, returns, prev_weights, port_ret, info):
        info["INJECTED_BY_REWARD"] = 999
        info["port_ret"] = -12345.0  # try to clobber a key the env sets after the call
        return float(port_ret), {}, None

    env, _panel, _lookback = _make_env(malicious)
    env.reset(seed=0)
    _obs, _r, _term, _trunc, info = env.step(np.zeros(env.N + 1))
    assert "INJECTED_BY_REWARD" not in info  # reward's injected key did not leak out
    assert info["port_ret"] != -12345.0  # env's own value, not the reward's clobber


def test_stateful_reward_still_round_trips_after_protection() -> None:
    """The array/info protection must NOT break stateful-reward round-tripping (V15a guard).

    The reward persists its OWN state via the returned ``reward_state`` (not by mutating info), and
    reads the prior state from ``info['reward_state']`` — both still work through the shallow copy.
    """
    env, _panel, _lookback = _make_env(_port_ret_reward)
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.zeros(env.N + 1))
    _, _, _, _, info1 = env.step(np.zeros(env.N + 1))
    _, _, _, _, info2 = env.step(np.zeros(env.N + 1))
    assert (info0["reward_state"], info1["reward_state"], info2["reward_state"]) == (0, 1, 2)


def test_benign_reward_numerics_unchanged_by_protection() -> None:
    """A BENIGN reward must see identical inputs and produce identical outputs post-fix (V15a guard).

    The protection (read-only arrays + shallow-copied info) must change NOTHING for legitimate
    reward code. This captures the exact values a benign reward observes and the fed/realized
    quantities, pinning that the boundary is transparent to honest rewards.
    """
    seen: dict = {}

    def benign(weights, returns, prev_weights, port_ret, info):
        # Reads only (the legitimate case): record what the reward observes + compute a real total.
        # Per the contract `weights` is length N+1 (incl. cash) while `returns` is the N risky assets,
        # so a real reward slices `weights[:returns.size]` (mirrors the env's `w[:N] @ r_t`).
        seen["weights"] = np.array(weights, copy=True)
        seen["returns"] = np.array(returns, copy=True)
        seen["prev_weights"] = np.array(prev_weights, copy=True)
        seen["port_ret"] = float(port_ret)
        seen["reward_state_in"] = info.get("reward_state")
        risky = weights[: returns.size]
        total = float(port_ret - 0.5 * float(np.var(returns)))
        return total, {"pnl": float(np.sum(risky * returns))}, None

    env, panel, lookback = _make_env(benign)
    env.reset(seed=0)
    action = np.array([2.0, -1.0, 0.5, 0.3, 0.0], dtype=np.float64)  # length N+1 == 5
    w = project_simplex(action, env.projection)
    obs, reward, _term, _trunc, info = env.step(action)

    # The reward saw the true realized return row and the true projected weights.
    np.testing.assert_array_equal(seen["returns"], panel.returns[lookback])
    np.testing.assert_array_equal(seen["weights"], w)
    np.testing.assert_array_equal(seen["prev_weights"], np.full(env.N + 1, 1.0 / (env.N + 1)))
    # The realized port_ret fed to the reward equals the env-computed port_ret, and the agent's
    # reward equals the benign total (= port_ret - 0.5*var) — i.e. numerics are transparent.
    assert seen["port_ret"] == pytest.approx(info["port_ret"], abs=0.0)
    assert reward == pytest.approx(info["port_ret"] - 0.5 * float(np.var(panel.returns[lookback])))
    assert np.isfinite(obs).all()


# --- (1) RESET DETERMINISM: byte-identical obs on repeated seeded reset (PD-6) -----------------


def test_reset_with_same_seed_gives_byte_identical_obs() -> None:
    """env.reset(seed=S) twice => byte-identical obs (np.array_equal, exact) (PD-6 determinism).

    Determinism is load-bearing for the pre-registered study: a fixed (panel, seed) must reproduce
    the SAME initial observation bit-for-bit across resets, so the archive replays identically. We
    also confirm the reset previous-weights are a valid UNIFORM simplex point summing to 1.
    """
    env, _panel, _lookback = _make_env(_port_ret_reward)
    obs_a, info_a = env.reset(seed=4321)
    # Mutate runtime state between resets so we know the second reset truly re-initialises.
    env.step(np.full(env.N + 1, 0.7))
    env.step(np.full(env.N + 1, -0.2))
    obs_b, info_b = env.reset(seed=4321)

    assert np.array_equal(obs_a, obs_b)  # EXACT, not approx — determinism is bit-for-bit
    assert obs_a.dtype == obs_b.dtype == np.float32
    assert info_a == info_b == {}
    # reset w_prev is the UNIFORM simplex (every weight == 1/(N+1)) and sums to exactly 1.
    n_act = env.N + 1
    np.testing.assert_array_equal(env.w_prev, np.full(n_act, 1.0 / n_act, dtype=np.float64))
    assert env.w_prev.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.all(env.w_prev >= 0.0)


def test_reset_determinism_full_episode_byte_identical() -> None:
    """Two seeded resets followed by the SAME action sequence give byte-identical obs every step.

    Stronger than a single-step check: the WHOLE trajectory is deterministic, so the per-seed
    rollouts the headline inference relies on replay exactly (PD-6).
    """
    env, _panel, _lookback = _make_env(_port_ret_reward, extra_rows=8)
    actions = [np.full(env.N + 1, v) for v in (0.5, -0.3, 1.2, 0.0, -0.7)]

    def _roll() -> list[np.ndarray]:
        env.reset(seed=999)
        out = []
        for a in actions:
            obs, _r, _term, trunc, _info = env.step(a)
            out.append(np.array(obs, copy=True))
            if trunc:
                break
        return out

    first = _roll()
    second = _roll()
    assert len(first) == len(second)
    for o1, o2 in zip(first, second):
        assert np.array_equal(o1, o2)  # byte-identical across the whole replayed episode


# --- (2) V15a: arrays handed to the reward are DETACHED READ-ONLY COPIES (regression-lock) -----


def test_reward_arrays_are_readonly_detached_copies() -> None:
    """The arrays handed to the reward are READ-ONLY copies: writeable flag False AND base is None.

    The V15a determinism boundary requires the reward to receive arrays it cannot write through
    to env/panel memory. A read-only *view* would still expose a writable parent via ``.base``;
    a detached COPY (``base is None``) makes the boundary self-sufficient. Lock both invariants.
    """
    captured: dict = {}

    def inspector(weights, returns, prev_weights, port_ret, info):
        captured["weights_writeable"] = weights.flags.writeable
        captured["returns_writeable"] = returns.flags.writeable
        captured["prev_writeable"] = prev_weights.flags.writeable
        captured["weights_base"] = weights.base
        captured["returns_base"] = returns.base
        captured["prev_base"] = prev_weights.base
        return float(port_ret), {}, None

    env, _panel, _lookback = _make_env(inspector)
    env.reset(seed=0)
    env.step(np.full(env.N + 1, 0.3))

    assert captured["weights_writeable"] is False
    assert captured["returns_writeable"] is False
    assert captured["prev_writeable"] is False
    # base is None => a detached copy, not a (read-only) view onto a writable parent.
    assert captured["weights_base"] is None
    assert captured["returns_base"] is None
    assert captured["prev_base"] is None


def test_reward_mutation_attempts_cannot_corrupt_panel_across_full_episode() -> None:
    """A reward attempting every in-place write must NOT corrupt the SHARED panel over a full episode.

    Regression-locks the V15a fix end-to-end: across an entire episode a malicious reward attempting
    ``returns[:]=0`` / ``weights[:]=-7`` / ``prev_weights[:]=-7`` / ``info['HACK']=1`` /
    ``info['port_ret']=-123`` leaves (a) every panel row it touched unchanged, (b) the env's w_prev a
    valid simplex, and (c) the env's own info/port_ret/cost accounting intact (NOT the reward's clobber).
    """

    def malicious(weights, returns, prev_weights, port_ret, info):
        returns[:] = 0.0
        weights[:] = -7.0
        prev_weights[:] = -7.0
        info["HACK"] = 1
        info["port_ret"] = -123.0
        return 1e9, {"junk": 1.0}, None  # also an inflated total (see port_ret accounting test)

    env, panel, lookback = _make_env(malicious, extra_rows=6)
    panel_before = panel.returns.copy()
    env.reset(seed=0)

    rng = np.random.default_rng(3)
    truncated = False
    last_info: dict = {}
    while not truncated:
        action = rng.standard_normal(env.N + 1)
        _obs, _r, _term, truncated, last_info = env.step(action)
        # (b) env's next w_prev is always a valid simplex (never the reward's -7 leak).
        assert np.all(env.w_prev >= 0.0)
        assert env.w_prev.sum() == pytest.approx(1.0, abs=1e-12)
        # (c) env's own accounting survives the reward's info clobber.
        assert "HACK" not in last_info
        assert last_info["port_ret"] != -123.0
        assert np.isfinite(last_info["port_ret"])
        assert last_info["cost"] >= 0.0

    # (a) the whole shared panel is byte-identical to before the episode — nothing was zeroed.
    np.testing.assert_array_equal(panel.returns, panel_before)
    assert np.count_nonzero(panel.returns[lookback]) > 0


# --- (3) SIMPLEX PROJECTION: softmax interior vs l1-normalize corner reachability --------------


def test_softmax_projection_never_reaches_exact_cash_corner() -> None:
    """softmax can NEVER reach an exact cash corner over the REACHABLE (bounded) action range.

    Disclosed limitation (project_simplex docstring): softmax maps onto the OPEN interior, so the
    full "flee to cash" allocation (w_cash == 1, risky == 0) is structurally unreachable. The env
    bounds raw logits to ``action.bound`` (==10.0), so the largest cash dominance the agent can
    request is a logit-gap of 2*bound; even there w_cash < 1 strictly and every risky weight > 0.

    (For UNBOUNDED logits >~ 745 float64 underflow makes ``exp(-gap) == 0`` exactly, so the open-
    interior guarantee is mathematical and holds over the action space SAC actually explores, not at
    pathological out-of-bound magnitudes — documented here so the limitation is not overstated.)
    """
    cfg = load_config("environment")
    bound = float(cfg["action"]["bound"])
    n_risky = 5
    action = np.full(n_risky + 1, -bound, dtype=np.float64)
    action[-1] = bound  # maximal reachable cash dominance (logit-gap = 2*bound)
    w = project_simplex(action, kind="softmax")

    assert w[-1] < 1.0  # never an exact cash corner over the reachable range
    assert np.all(w[:n_risky] > 0.0)  # every risky weight strictly positive (open interior)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    assert np.all(w >= 0.0)


def test_l1_normalize_projection_can_reach_exact_corner_and_zeros() -> None:
    """l1_normalize_of_clipped CAN hit exact zeros / the cash corner (the softmax cannot)."""
    n_risky = 5
    # All risky logits <= 0 (clip to 0) and a positive cash logit => exact 100%-cash corner.
    action = np.array([-3.0, 0.0, -1.0, -2.0, -0.5, 4.0], dtype=np.float64)
    w = project_simplex(action, kind="l1_normalize_of_clipped")
    assert w[-1] == 1.0  # EXACT cash corner reached
    np.testing.assert_array_equal(w[:n_risky], np.zeros(n_risky))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    # A mixed action with some clipped-to-zero entries gives EXACT zeros for those entries.
    action2 = np.array([2.0, -1.0, 3.0, -5.0, 0.0, 1.0], dtype=np.float64)
    w2 = project_simplex(action2, kind="l1_normalize_of_clipped")
    assert w2[1] == 0.0 and w2[3] == 0.0  # negatives clipped to exact zero
    np.testing.assert_allclose(w2.sum(), 1.0, atol=1e-12)
    assert np.all(w2 >= 0.0)


def test_both_projections_are_valid_simplex_points() -> None:
    """Both frozen projections always yield weights >= 0 summing to 1 (tight atol) (C-8)."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        action = rng.standard_normal(9) * 7.0
        for kind in ("softmax", "l1_normalize_of_clipped"):
            w = project_simplex(action, kind=kind)
            assert np.all(w >= 0.0)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    # The all-zero degenerate input falls back to uniform under l1-normalize (no NaN).
    w0 = project_simplex(np.zeros(4), kind="l1_normalize_of_clipped")
    np.testing.assert_allclose(w0, np.full(4, 0.25), atol=1e-12)


# --- (6) PORT_RET ACCOUNTING: env computes port_ret/cost, never trusts the reward's total -------


def test_port_ret_and_cost_are_env_computed_not_reward_reported() -> None:
    """info['port_ret']/info['cost'] are computed by the ENV (gross - cost), never the reward's total.

    A reward returning a wildly inflated ``total`` must not bleed into the env's accounting: the
    env nets gross minus its own half-L1-drifted cost and ignores the reward's reported number for
    bookkeeping. The agent's REWARD is the (inflated) total; the env's LEDGER (port_ret/cost/gross)
    is independent.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n = 2
    t = lookback + 1
    rng = np.random.default_rng(5)
    returns = rng.standard_normal((t, n)) * 1e-3
    r0 = np.array([0.10, -0.05], dtype=np.float64)
    returns[lookback] = r0
    panel = Panel(returns=returns, vix=np.full(t, 20.0), dates=np.arange(t), asset_ids=np.arange(n))

    inflated = 987654.0

    def liar(weights, returns_, prev_weights, port_ret, info):
        # Returns an inflated total AND attempts to clobber the env's port_ret key.
        info["port_ret"] = inflated
        return inflated, {"port_ret": inflated}, None

    env = PortfolioEnv(panel, cfg, liar, start=lookback, end=t)
    env.reset(seed=0)

    action = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    w = project_simplex(action, env.projection)
    c = float(cfg["costs"]["headline_bps"]) * 1e-4
    # One step from reset: nothing has drifted yet, so the held book is the uniform book (#92).
    w_held = np.full(n + 1, 1.0 / (n + 1), dtype=np.float64)
    expected_turnover = 0.5 * float(np.abs(w - w_held).sum())
    expected_cost = c * expected_turnover
    expected_gross = float(w[:n] @ r0)
    expected_port_ret = expected_gross - expected_cost

    _obs, reward, _term, _trunc, info = env.step(action)

    # The env's ledger is its OWN closed-form computation, not the reward's inflated number.
    assert info["port_ret"] == pytest.approx(expected_port_ret, abs=1e-12)
    assert info["cost"] == pytest.approx(expected_cost, abs=1e-12)
    assert info["gross"] == pytest.approx(expected_gross, abs=1e-12)
    assert info["port_ret"] != pytest.approx(inflated)
    # The agent's reward IS the (inflated) total the reward returned — accounting and reward are decoupled.
    assert reward == pytest.approx(inflated)


@pytest.mark.parametrize("bad", ["all_nan", "all_inf", "one_inf"])
def test_non_finite_action_is_refused_not_silently_trained_on(env: PortfolioEnv, bad: str) -> None:
    """A NaN/inf action must RAISE, not poison the rollout while looking healthy.

    Demonstrated before the guard existed (deep review 2026-07-26): ``step(nan_action)`` did NOT
    raise -- it produced ``port_ret=NaN``, fed a NaN OBSERVATION back to the agent, and ``safe_call``
    substituted a SAFE_DEFAULT reward of 0.0, so training looked healthy while the policy learned
    from poison. The ``port_growth <= 0.0`` wipeout guard cannot catch this: ``port_growth`` is
    ``w @ growth`` on an all-NaN ``w``, hence NaN, and every NaN comparison is False (since #92 the
    guard keys off the post-trade book, so it sees the poison and still cannot act on it).

    ``one_inf`` is included deliberately: the softmax turns ANY single non-finite entry into an
    ALL-NaN weight vector, because ``max(a)`` is ``inf`` and ``inf - inf`` NaNs the max element
    itself. The corruption is total, not partial.
    """
    env.reset(seed=0)
    n = env.N + 1
    action = {
        "all_nan": np.full(n, np.nan),
        "all_inf": np.full(n, np.inf),
        "one_inf": np.array([np.inf] + [0.0] * (n - 1)),
    }[bad]

    # `errstate` only silences NumPy's expected "invalid value in subtract" from the deliberate
    # inf - inf inside the softmax; the RAISE below is what is being asserted.
    with np.errstate(invalid="ignore"), pytest.raises(FloatingPointError, match="non-finite weights"):
        env.step(action)


def test_finite_actions_still_step_normally(env: PortfolioEnv) -> None:
    """Positive control for the guard above: an ordinary rollout is untouched."""
    env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
