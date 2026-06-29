"""No-look-ahead and reward-timing tests for PortfolioEnv (F.1, audit C-5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.panel import Panel
from src.env.portfolio_env import PortfolioEnv, project_simplex
from src.utils.config import load_config


def _port_ret_reward(weights, returns, prev_weights, port_ret, info):
    """Reward = realized portfolio return (stateless)."""
    return float(port_ret), {}, None


def _make_env(panel: Panel) -> PortfolioEnv:
    cfg = load_config("environment")
    return PortfolioEnv(panel, cfg, _port_ret_reward)


def _hand_panel(n_days: int = 80, n_assets: int = 3, seed: int = 7) -> Panel:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    vix = 10.0 + np.abs(rng.normal(0.0, 5.0, size=n_days))
    dates = np.arange("2010-01-04", n_days, dtype="datetime64[D]")
    asset_ids = np.arange(n_assets, dtype=np.int64)
    caps = np.full((n_days, n_assets), 1e9)
    return Panel(returns=returns, vix=vix, dates=dates, asset_ids=asset_ids, market_caps=caps)


def test_truncation_invariance_over_all_columns() -> None:
    """obs on the panel truncated at t equals obs from the full panel, all columns.

    _obs reads only data with index < t (returns) and t-1 (vix), so building it on
    ``panel.slice(0, t+1)`` (which keeps rows 0..t) must reproduce every column.
    """
    panel = _hand_panel()
    full = _make_env(panel)
    for t in range(full.start, full.T):
        full.t = t
        obs_full = full._obs()

        sub = panel.slice(0, t + 1)  # keeps rows 0..t inclusive
        env_sub = _make_env(sub)
        env_sub.t = t
        env_sub.w_prev = full.w_prev
        obs_sub = env_sub._obs()

        np.testing.assert_array_equal(obs_full, obs_sub)


def test_perturbation_invariance_future_rows_do_not_change_past_obs() -> None:
    """Perturbing future panel rows leaves all past observations unchanged."""
    panel = _hand_panel()
    env = _make_env(panel)
    t = env.start + 5
    env.t = t
    obs_before = env._obs().copy()

    perturbed_returns = panel.returns.copy()
    perturbed_returns[t:] += 100.0  # corrupt every row at or after t
    perturbed_vix = panel.vix.copy()
    perturbed_vix[t:] += 1000.0
    perturbed = Panel(
        returns=perturbed_returns,
        vix=perturbed_vix,
        dates=panel.dates,
        asset_ids=panel.asset_ids,
        market_caps=panel.market_caps,
    )
    env_p = _make_env(perturbed)
    env_p.t = t
    obs_after = env_p._obs()
    np.testing.assert_array_equal(obs_before, obs_after)


def test_obs_never_reads_future_rows() -> None:
    """_obs is invariant to any modification of panel rows with index >= t."""
    panel = _hand_panel()
    env = _make_env(panel)
    rng = np.random.default_rng(0)
    for t in range(env.start, env.T):
        env.t = t
        base = env._obs().copy()
        corrupt = panel.returns.copy()
        corrupt[t:] = rng.normal(0.0, 10.0, size=corrupt[t:].shape)
        cvix = panel.vix.copy()
        cvix[t:] = rng.normal(500.0, 1.0, size=cvix[t:].shape)
        cpanel = Panel(
            returns=corrupt, vix=cvix, dates=panel.dates,
            asset_ids=panel.asset_ids, market_caps=panel.market_caps,
        )
        env_c = _make_env(cpanel)
        env_c.t = t
        np.testing.assert_array_equal(base, env_c._obs())


def test_reward_timing_uses_returns_at_t() -> None:
    """On a known-return panel, the reward at t uses returns[t] (C-5).

    With a zero transaction cost and a known projected weight, the portfolio return
    (and hence the reward) must equal w[:N] @ returns[t] exactly.
    """
    panel = _hand_panel()
    cfg = load_config("environment")
    # zero out cost so reward == gross == w @ returns[t]
    env = PortfolioEnv(panel, cfg, _port_ret_reward)
    env.cost = 0.0
    env.reset(seed=0)

    action = np.zeros(env.N + 1)
    w = project_simplex(action, env.projection)
    t = env.t
    expected = float(w[: env.N] @ panel.returns[t])
    _, reward, _, _, info = env.step(action)
    assert reward == pytest.approx(expected)
    assert info["gross"] == pytest.approx(expected)


def test_reward_timing_shifting_action_shifts_reward() -> None:
    """Shifting the action by one step changes the reward as expected (C-5).

    Applying action A at step t consumes returns[t]; applying the same A one step
    later consumes returns[t+1]. With distinct return rows the rewards differ and
    each matches the contemporaneous realized return.
    """
    panel = _hand_panel()
    cfg = load_config("environment")

    # Make a clearly asymmetric action so different asset weights matter.
    raw = np.array([5.0, -5.0, 0.0, 0.0])  # N=3 + cash

    env_a = PortfolioEnv(panel, cfg, _port_ret_reward)
    env_a.cost = 0.0
    env_a.reset(seed=0)
    w = project_simplex(raw, env_a.projection)
    t0 = env_a.t
    _, reward_t0, _, _, _ = env_a.step(raw)
    assert reward_t0 == pytest.approx(float(w[: env_a.N] @ panel.returns[t0]))

    env_b = PortfolioEnv(panel, cfg, _port_ret_reward)
    env_b.cost = 0.0
    env_b.reset(seed=0)
    env_b.step(np.zeros(env_b.N + 1))  # consume t0 with a neutral action
    t1 = env_b.t
    _, reward_t1, _, _, _ = env_b.step(raw)
    assert reward_t1 == pytest.approx(float(w[: env_b.N] @ panel.returns[t1]))
    assert reward_t0 != pytest.approx(reward_t1)


# --- (4) NO-LOOKAHEAD: strict window bounds + lagged VIX, with POISONED future rows -------------


def _poisoned_panel(n_assets: int = 3) -> tuple[Panel, int]:
    """A deterministic panel whose rows at index >= ``t_poison`` are sentinel-poisoned.

    ``t_poison`` is set to the config lookback so the strictly-past window ``[t-lookback:t]`` (and
    every realized-vol window, all <= lookback) lies entirely in the benign rows. Returns/vix at and
    beyond ``t_poison`` carry huge sentinel values; an obs built at ``t == t_poison`` reads strictly
    the PAST (rows < t for returns, the t-1 close for vix) and so must be unaffected by the poison.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    t_poison = lookback  # first decision index; window [0:lookback] is benign
    n_days = lookback + 10
    returns = np.full((n_days, n_assets), 0.001, dtype=np.float64)
    vix = np.full(n_days, 20.0, dtype=np.float64)
    returns[t_poison:] = 1e6  # future poison (index >= t)
    vix[t_poison:] = -1e9  # future poison
    dates = np.arange("2010-01-04", n_days, dtype="datetime64[D]")
    panel = Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(n_assets, dtype=np.int64))
    return panel, t_poison


def test_obs_unaffected_by_poisoned_future_rows_at_decision_index() -> None:
    """At t == t_poison the obs is finite and ignores the poisoned rows at index >= t (no look-ahead)."""
    panel, t_poison = _poisoned_panel()
    env = _make_env(panel)
    env.t = t_poison
    obs = env._obs()
    assert np.isfinite(obs).all()  # no 1e6 / -1e9 poison leaked in
    # Every value is bounded — the only inputs are the benign 0.001 returns, their vol, the lagged
    # vix (20.0), the cash marker (1.0) and the uniform w_prev. None approaches the 1e6 sentinel.
    assert np.max(np.abs(obs)) < 100.0


def test_returns_and_vol_windows_read_strictly_before_t() -> None:
    """The lookback and EVERY realized-vol window read returns[t-w:t] — never index >= t.

    Construct a panel whose row exactly at ``t`` is a huge sentinel; the obs at ``t`` must equal the
    obs built when that row is benign, proving returns[t] (the future realisation) is never read by
    _obs (it is consumed only in step()).
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n_assets = 3
    n_days = lookback + 20
    t = lookback + 5  # >= lookback so [t-lookback:t] is a fully-populated, strictly-past window
    base = np.full((n_days, n_assets), 0.002, dtype=np.float64)
    vix = np.full(n_days, 18.0, dtype=np.float64)
    dates = np.arange("2010-01-04", n_days, dtype="datetime64[D]")
    panel_clean = Panel(returns=base.copy(), vix=vix.copy(), dates=dates, asset_ids=np.arange(n_assets))

    poisoned = base.copy()
    poisoned[t] = 1e9  # corrupt EXACTLY the current row (the future realisation)
    panel_dirty = Panel(returns=poisoned, vix=vix.copy(), dates=dates, asset_ids=np.arange(n_assets))

    env_clean = _make_env(panel_clean)
    env_clean.t = t
    env_dirty = _make_env(panel_dirty)
    env_dirty.t = t
    # Identical obs => returns[t] (and anything >= t) is invisible to _obs; only [t-w:t] is read.
    np.testing.assert_array_equal(env_clean._obs(), env_dirty._obs())


def test_vix_in_obs_is_lagged_t_minus_one_close() -> None:
    """The VIX exposed in the obs is the t-1 close (lagged), never the contemporaneous/future value.

    On the synthetic (contemporaneous) convention the env lags to vix[t-1]; poisoning vix[t] (and
    beyond) must not change the obs, and the exposed vix value must equal vix[t-1] exactly.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n_assets = 3
    n_days = lookback + 10
    returns = np.full((n_days, n_assets), 0.001, dtype=np.float64)
    # Distinct, identifiable vix levels so we can pin which row the obs read.
    vix = np.arange(n_days, dtype=np.float64) * 10.0  # vix[t] == 10*t
    dates = np.arange("2010-01-04", n_days, dtype="datetime64[D]")
    panel = Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(n_assets))
    assert panel.vix_prelagged is False  # synthetic convention => env lags to vix[t-1]

    env = _make_env(panel)
    t = lookback + 3
    env.t = t
    obs = env._obs()
    # Obs layout: [lookback*N returns | vol_windows*N | vix(1) | cash(1) | prev_w(N+1)].
    n_ret = lookback * n_assets
    n_vol = len(env.vol_windows) * n_assets
    vix_value = obs[n_ret + n_vol]
    assert vix_value == pytest.approx(vix[t - 1])  # the t-1 close, NOT vix[t]
    assert vix_value != pytest.approx(vix[t])  # explicitly NOT the contemporaneous value

    # Poison vix at and beyond t: obs is unchanged because only the t-1 close is read.
    poisoned = vix.copy()
    poisoned[t:] = 1e12
    panel_p = Panel(returns=returns, vix=poisoned, dates=dates, asset_ids=np.arange(n_assets))
    env_p = _make_env(panel_p)
    env_p.t = t
    np.testing.assert_array_equal(obs, env_p._obs())


def test_vix_prelagged_panel_reads_row_t_still_no_lookahead() -> None:
    """On a PRELAGGED gold panel row t already holds the t-1 close, so the env reads vix[t] (== t index).

    This is the gold convention (vix_prelagged=True): the obs vix equals vix[t] (which IS the t-1
    close already), and poisoning rows STRICTLY beyond t (index > t) leaves the obs unchanged — the
    boundary row read is at index t, never the future row t+1.
    """
    cfg = load_config("environment")
    lookback = int(cfg["state"]["lookback_days"])
    n_assets = 3
    n_days = lookback + 10
    returns = np.full((n_days, n_assets), 0.001, dtype=np.float64)
    vix = np.arange(n_days, dtype=np.float64) * 10.0
    dates = np.arange("2010-01-04", n_days, dtype="datetime64[D]")
    panel = Panel(
        returns=returns, vix=vix, dates=dates,
        asset_ids=np.arange(n_assets), vix_prelagged=True,
    )

    env = _make_env(panel)
    t = lookback + 3
    env.t = t
    obs = env._obs()
    n_ret = lookback * n_assets
    n_vol = len(env.vol_windows) * n_assets
    assert obs[n_ret + n_vol] == pytest.approx(vix[t])  # prelagged: read row t directly

    # Poison STRICTLY future rows (> t): the prelagged read at index t must be unaffected.
    poisoned = vix.copy()
    poisoned[t + 1 :] = 1e12
    panel_p = Panel(
        returns=returns, vix=poisoned, dates=dates,
        asset_ids=np.arange(n_assets), vix_prelagged=True,
    )
    env_p = _make_env(panel_p)
    env_p.t = t
    np.testing.assert_array_equal(obs, env_p._obs())


def test_zero_lookback_refused_because_the_vix_channel_would_read_the_future() -> None:
    """``lookback_days = 0`` must RAISE, not build an env whose obs leaks the panel's LAST row.

    At ``lookback=0`` the default ``start`` is 0, so on the contemporaneous (synthetic) convention
    ``_obs`` reads ``vix[t - 1]`` = ``vix[-1]`` -- NumPy's NEGATIVE index, i.e. the panel's FINAL row:
    the future. DEMONSTRATED before the guard existed (deep review 2026-07-26): an obs built at t=0
    contained ``vix[-1]`` and NOT ``vix[0]``. ``_obs``'s ``min(vix_idx, T-1)`` clamp bounds the index
    only from ABOVE, so nothing caught it. Same negative-index class as final-audit #28, which closed
    it for ``realized_vol_windows`` but left the lookback itself open.
    """
    import copy

    cfg = copy.deepcopy(load_config("environment"))
    cfg["state"]["lookback_days"] = 0
    cfg["state"]["realized_vol_windows"] = []  # else the #28 guard fires first and masks this one

    with pytest.raises(ValueError, match="lookback_days must be >= 1"):
        PortfolioEnv(_hand_panel(), cfg, _port_ret_reward)
