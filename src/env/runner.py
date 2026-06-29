"""Environment runner — the ``env_builder`` keystone the discovery loop injects.

The reward-discovery loop (``src/llm/loop.py``) and the search arms
(``src/search/*``) are written against an injected ``env_builder(reward_fn) -> env``
where ``env`` exposes::

    train_env()            -> a PortfolioEnv on the TRAIN window (for SAC training)
    val_returns(policy)    -> realized VALIDATION portfolio returns (for selection)
    train_returns(policy)  -> realized TRAINING portfolio returns (for measurement)
    test_returns(policy)   -> realized held-out TEST returns (FINAL inference ONLY; sealed)

Nothing in the repo implemented that contract (it was only faked in tests); this
module provides it (MASTER_EXECUTION_PLAN P3).

Design (literature-anchored)
----------------------------
- **One anonymised Panel, two integer windows.** The train/val windows are
  half-open ``[start, end)`` slices into a *single* survivorship-free, point-in-time
  Panel (Brown et al. 1992 — survivorship-free; Cont 2001 — the real fat tails are
  the point). The validation observations' lookback history is therefore drawn from
  the (strictly past) training-period rows — no look-ahead (audit C-5; the env's
  ``_obs`` only ever reads ``returns[t-lookback:t]``).
- **Two splits, two roles (audit B-2/B-3).** ``train_returns`` feeds the
  *measurement* of the realized-return distribution (the contribution channel);
  ``val_returns`` feeds the *held-out fitness* (selection). They are deliberately on
  different splits so the loop cannot tune-then-select on the same data.
- **Deterministic walk-forward rollout** (Sood 2023 backtest protocol): the policy
  is rolled out greedily (``deterministic=True``) over the window, and the realized
  per-step portfolio return (net of turnover cost) is collected from
  ``info["port_ret"]``.

The rollout is policy-agnostic: it calls ``policy.predict(obs, deterministic=True)``,
so it works with a raw Stable-Baselines3 model OR the obs-normalising
``NormalizedPolicy`` returned by ``src.agents.trainer`` (which carries train-only
observation statistics, deep-research §2).
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from src.data.panel import Panel
from src.env.portfolio_env import PortfolioEnv
from src.reward.contract import RewardFn

__all__ = [
    "EnvBundle",
    "make_env_builder",
    "rollout_port_returns",
    "rollout_port_series",
    "rollout_port_diagnostics",
    "Window",
]

#: A half-open ``[start, end)`` integer window into a Panel's time axis.
Window = tuple[int, int]

_LOG = logging.getLogger(__name__)


def rollout_port_returns(env: PortfolioEnv, policy: Any) -> np.ndarray:
    """Deterministically roll ``policy`` through ``env``; return realized port returns.

    Parameters
    ----------
    env : PortfolioEnv
        A portfolio env over the desired window (train or val).
    policy : Any
        Anything exposing ``predict(obs, deterministic=True) -> (action, state)`` —
        a Stable-Baselines3 model or a ``NormalizedPolicy``.

    Returns
    -------
    np.ndarray
        1-D array of realized per-step portfolio returns ``info["port_ret"]`` (gross
        minus proportional turnover cost), one per traded step over ``[start, end)``.
    """
    from src.sandbox.executor import reset_failure_flag, safe_call_count, safe_default_count

    reset_failure_flag()  # P0-2: surface a reward that fails on real data mid-rollout
    obs, _info = env.reset()
    rets: list[float] = []
    done = False
    while not done:
        action, _state = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(np.asarray(action).ravel())
        rets.append(float(info["port_ret"]))
        done = bool(terminated or truncated)
    n_sd = safe_default_count()  # whole-window count, not last-call (R66): surfaces partial failures
    if n_sd:
        n_call = safe_call_count()
        _LOG.warning(
            "reward substituted SAFE_DEFAULT on %d/%d rollout steps (%.1f%%); candidate degraded (R66)",
            n_sd, n_call, 100.0 * n_sd / max(n_call, 1),
        )
    return np.asarray(rets, dtype=float)


def rollout_port_series(env: PortfolioEnv, policy: Any) -> dict[str, np.ndarray]:
    """Deterministically roll ``policy`` through ``env``; return the gross/turnover/net series.

    Like :func:`rollout_port_returns` but additionally collects the per-step ``info['gross']``
    and ``info['turnover']`` decomposition the env emits (Rank 4). This is the input to the
    cost-sweep's ANALYTIC re-pricing (``scripts/cost_sweep.py``): for any alternative
    per-unit-turnover cost ``c`` (fraction) the net return is exactly
    ``net_c = gross - c * turnover`` — no retraining and no re-rolling needed, because the
    *policy* (hence the realized ``gross`` and ``turnover`` at every step) does not depend on
    the cost level (the env charges the cost only AFTER the action is chosen, audit C-5).

    Parameters
    ----------
    env : PortfolioEnv
        A portfolio env over the desired window. ``env.cost`` (the headline or a
        ``cost_bps``-overridden level) is irrelevant to ``gross``/``turnover`` — only to the
        ``net`` returned here — so the gross/turnover series are reusable across cost levels.
    policy : Any
        Anything exposing ``predict(obs, deterministic=True) -> (action, state)``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"gross": ..., "turnover": ..., "net": ...}``, each a 1-D array of length
        ``end - start``. ``net`` equals ``gross - env.cost * turnover`` step-for-step (the
        same value :func:`rollout_port_returns` returns), so this is a strict superset.
    """
    from src.sandbox.executor import reset_failure_flag, safe_call_count, safe_default_count

    reset_failure_flag()
    obs, _info = env.reset()
    gross: list[float] = []
    turnover: list[float] = []
    net: list[float] = []
    done = False
    while not done:
        action, _state = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(np.asarray(action).ravel())
        gross.append(float(info["gross"]))
        turnover.append(float(info["turnover"]))
        net.append(float(info["port_ret"]))
        done = bool(terminated or truncated)
    n_sd = safe_default_count()  # whole-window count, not last-call (R66): surfaces partial failures
    if n_sd:
        n_call = safe_call_count()
        _LOG.warning(
            "reward substituted SAFE_DEFAULT on %d/%d rollout steps (%.1f%%); candidate degraded (R66)",
            n_sd, n_call, 100.0 * n_sd / max(n_call, 1),
        )
    return {
        "gross": np.asarray(gross, dtype=float),
        "turnover": np.asarray(turnover, dtype=float),
        "net": np.asarray(net, dtype=float),
    }


def rollout_port_diagnostics(env: PortfolioEnv, policy: Any) -> dict[str, Any]:
    """Deterministically roll ``policy`` through ``env``; return the FULL per-step diagnostic set.

    A strict superset of :func:`rollout_port_series`: it additionally collects the per-step portfolio
    ``info['weights']`` (the post-action simplex allocation) and ``info['components']`` (the reward
    decomposition) the env already emits every step. These are **pure reads of already-computed values**
    — the rollout adds no randomness and changes no arithmetic — so this is determinism-safe, report-only
    capture (the determinism envelope). It is used ONLY on the sealed test leg's frozen winner, to
    archive the exposure / allocation diagnostics the equity, drawdown and allocation figures need: a
    frozen, replay-only campaign can only ever plot a metric it LOGGED at run time (M1/M1b/M3).

    Returns
    -------
    dict[str, Any]
        ``{"gross", "turnover", "net"}`` exactly as :func:`rollout_port_series`, plus
        ``"weights"`` — a ``(T, N)`` float array of per-step allocations (``None`` if the env emits
        none) — and ``"components"`` — a list of the per-step ``info['components']`` (``None`` if the
        env emits none). ``net`` is identical to the other two rollouts step-for-step.
    """
    from src.sandbox.executor import reset_failure_flag, safe_call_count, safe_default_count

    reset_failure_flag()
    obs, _info = env.reset()
    gross: list[float] = []
    turnover: list[float] = []
    net: list[float] = []
    weights: list[np.ndarray] = []
    components: list[Any] = []
    any_weights = False
    any_components = False
    done = False
    while not done:
        action, _state = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(np.asarray(action).ravel())
        gross.append(float(info["gross"]))
        turnover.append(float(info["turnover"]))
        net.append(float(info["port_ret"]))
        w = info.get("weights")
        if w is not None:
            any_weights = True
            weights.append(np.asarray(w, dtype=float).ravel())
        comp = info.get("components")
        if comp is not None:
            any_components = True
        components.append(comp)
        done = bool(terminated or truncated)
    n_sd = safe_default_count()  # whole-window count, not last-call (R66): surfaces partial failures
    if n_sd:
        n_call = safe_call_count()
        _LOG.warning(
            "reward substituted SAFE_DEFAULT on %d/%d rollout steps (%.1f%%); candidate degraded (R66)",
            n_sd, n_call, 100.0 * n_sd / max(n_call, 1),
        )
    # A (T, N) matrix only when EVERY step emitted an equal-length weight vector (the fixed env always
    # does); otherwise fall back to None so a mismatched/absent emission never raises here.
    #
    # NOTE: 2026-07-27 (deep review, finding #93): the `len(weights) == len(net)` clause was missing, and
    # without it this guard failed in the wrong direction under the exact condition it exists for.
    # `weights.append` is conditional on the step emitting one, while `net.append` is not — so a
    # PARTIALLY-emitting env yielded a SHORT matrix that satisfied both remaining clauses (all collected
    # rows equal-length, non-empty) and was returned as if aligned. REPRODUCED: an env emitting weights
    # on 4 of 5 steps returned weights (4, 3) against net (5,), i.e. row k of the matrix is step k+1 of
    # the return series. That silently off-by-one pairing feeds the SEALED TEST LEG's exposure,
    # allocation and drawdown figures, which a replay-only campaign cannot recompute. Unreachable with
    # `PortfolioEnv` (it emits on every step) — but this guard's whole purpose is tolerating envs that
    # do not, so "the real env is fine" is not the property it claims. Length-agreement is the invariant.
    weights_arr: np.ndarray | None = None
    if (
        any_weights
        and len(weights) == len(net)
        and len({int(a.size) for a in weights}) == 1
        and weights[0].size
    ):
        weights_arr = np.asarray(weights, dtype=float)
    return {
        "gross": np.asarray(gross, dtype=float),
        "turnover": np.asarray(turnover, dtype=float),
        "net": np.asarray(net, dtype=float),
        "weights": weights_arr,
        "components": components if any_components else None,
    }


class EnvBundle:
    """Train/val/test env factory + rollout for ONE reward function (the loop's ``env``).

    Holds the shared Panel, the env config, the bound ``reward_fn``, and up to three
    integer windows. Each accessor builds a *fresh* ``PortfolioEnv`` (envs are cheap;
    a fresh env guarantees a clean reset and avoids cross-rollout state leakage).

    The ``test_window`` is **optional and defaults to ``None``**: every search/selection
    bundle is built without it, so the held-out 2020-2026 test leg is *physically*
    unreachable during selection (``test_returns`` raises ``RuntimeError``). Only the
    campaign's final inference (the frozen winner, touched exactly once) constructs a
    bundle *with* a test window (PREREGISTRATION.md §10: select-on-val -> freeze -> test).
    """

    def __init__(
        self,
        panel: Panel,
        cfg: Any,
        reward_fn: RewardFn,
        train_window: Window,
        val_window: Window,
        test_window: Window | None = None,
        cost_bps: float | None = None,
    ) -> None:
        self.panel = panel
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        #: Optional per-unit-turnover cost OVERRIDE in bps (costs.grid_bps sweep). ``None``
        #: (default) -> the config headline cost; every existing bundle is unchanged.
        self.cost_bps = cost_bps

    def _env(self, window: Window) -> PortfolioEnv:
        start, end = window
        return PortfolioEnv(
            self.panel,
            self.cfg,
            self.reward_fn,
            start=start,
            end=end,
            cost_bps=self.cost_bps,
        )

    def train_env(self) -> PortfolioEnv:
        """A fresh PortfolioEnv over the TRAIN window (the SAC training env)."""
        return self._env(self.train_window)

    def val_returns(self, policy: Any) -> np.ndarray:
        """Realized VALIDATION portfolio returns for ``policy`` (selection signal)."""
        return rollout_port_returns(self._env(self.val_window), policy)

    def train_returns(self, policy: Any) -> np.ndarray:
        """Realized TRAINING portfolio returns for ``policy`` (measurement signal)."""
        return rollout_port_returns(self._env(self.train_window), policy)

    def test_returns(self, policy: Any) -> np.ndarray:
        """Realized held-out TEST portfolio returns for ``policy`` — FINAL INFERENCE ONLY.

        The test split (2020-2026) is **sealed**: it is touched exactly once, on the
        frozen winner, at final inference (select-on-validation -> freeze -> test-once;
        PREREGISTRATION.md §10). A bundle built without a ``test_window`` — i.e. every
        bundle the discovery loop and the search arms ever see — raises, so selection
        cannot read the test leg even by accident.

        Raises
        ------
        RuntimeError
            If this bundle has no ``test_window`` (every selection/search bundle).
        """
        if self.test_window is None:
            raise RuntimeError(
                "test split sealed until final inference: this EnvBundle has no "
                "test_window (selection must never touch the 2020-2026 test leg; only "
                "run_campaign's frozen-winner evaluation builds a test bundle)"
            )
        return rollout_port_returns(self._env(self.test_window), policy)

    def test_series(self, policy: Any) -> dict[str, np.ndarray]:
        """Realized held-out TEST gross/turnover/net series — FINAL INFERENCE ONLY.

        The gross/turnover superset of :meth:`test_returns` (``net`` is identical), so the
        cost sweep can persist the per-step ``gross`` + ``turnover`` and re-price the frozen
        winner at any ``costs.grid_bps`` level analytically. Subject to the SAME seal — a
        bundle without a ``test_window`` raises (the test leg stays unreachable in selection).

        Returns
        -------
        dict[str, np.ndarray]
            ``{"gross", "turnover", "net"}`` over the test window (see
            :func:`rollout_port_series`).

        Raises
        ------
        RuntimeError
            If this bundle has no ``test_window`` (every selection/search bundle).
        """
        if self.test_window is None:
            raise RuntimeError(
                "test split sealed until final inference: this EnvBundle has no "
                "test_window (selection must never touch the 2020-2026 test leg; only "
                "run_campaign's frozen-winner evaluation builds a test bundle)"
            )
        return rollout_port_series(self._env(self.test_window), policy)

    def test_diagnostics(self, policy: Any) -> dict[str, Any]:
        """Realized held-out TEST diagnostics (net/gross/turnover + weights + components) — INFERENCE ONLY.

        The full-diagnostic superset of :meth:`test_series` (``net`` is identical), adding the per-step
        allocation ``weights`` and reward ``components`` so the sealed-leg record can archive the
        exposure / allocation figures' data (M1/M1b/M3). Subject to the SAME seal: a bundle without a
        ``test_window`` raises, so the test leg stays unreachable during selection.

        Raises
        ------
        RuntimeError
            If this bundle has no ``test_window`` (every selection/search bundle).
        """
        if self.test_window is None:
            raise RuntimeError(
                "test split sealed until final inference: this EnvBundle has no "
                "test_window (selection must never touch the 2020-2026 test leg; only "
                "run_campaign's frozen-winner evaluation builds a test bundle)"
            )
        return rollout_port_diagnostics(self._env(self.test_window), policy)


def make_env_builder(
    panel: Panel,
    cfg: Any,
    train_window: Window,
    val_window: Window,
    test_window: Window | None = None,
    embargo: int = 0,
    cost_bps: float | None = None,
    lookback: int = 0,
) -> Callable[[RewardFn], EnvBundle]:
    """Return ``env_builder(reward_fn) -> EnvBundle`` bound to a panel + windows.

    Parameters
    ----------
    panel : Panel
        The anonymised, survivorship-free, point-in-time Panel (shared by all windows).
    cfg : Any
        The environment config (``config/environment.yaml``: lookback, vol windows,
        action projection + bound, cost).
    train_window, val_window : (int, int)
        Half-open ``[start, end)`` windows. ``train_window[0]`` must be ``>= lookback``
        (the env enforces this); ``val_window[0]`` must be ``>= train_window[1] + embargo``
        so train and val are disjoint *and embargoed* (selection on a different split,
        audit B-3; Lopez de Prado 2018 purge+embargo).
    test_window : (int, int) or None, default None
        The held-out TEST window. ``None`` for every search/selection builder (so the
        bundle's ``test_returns`` stays sealed). When given (the campaign's final-inference
        builder), it must satisfy ``test_window[0] >= val_window[1] + embargo``.
    embargo : int, default 0
        Trading-day embargo enforced at *both* split boundaries (config/inference.yaml
        ``embargo_trading_days`` == config/data.yaml ``embargo_days`` == 21). The default
        of 0 preserves the legacy 2-window callers (run_prototype / parallel) unchanged;
        the campaign passes 21 explicitly.
    cost_bps : float or None, default None
        Optional per-unit-turnover cost OVERRIDE in bps (the ``costs.grid_bps`` robustness
        sweep). ``None`` (default) -> the config headline cost, so every existing caller is
        unchanged; when given, every env the returned builder makes is priced at ``cost_bps``.
    lookback : int, default 0
        The FEATURE lookback (``state.lookback_days``), enforced as part of the inter-split
        purge: the guard below requires ``max(embargo, lookback)`` sessions between splits,
        because an embargo shorter than the lookback still lets the downstream window's first
        observations read prior-split returns (R18). **Pass it.** The ``0`` default exists only
        to preserve the legacy embargo-only signature, and silently degrades the purge to
        embargo-only — undocumented until 2026-07-27 (deep review, finding #94), though all
        seven call sites were verified to pass it, so the guard is armed everywhere in practice.

    Returns
    -------
    Callable
        The ``env_builder`` the discovery loop / search arms inject.
    """
    # The inter-split purge must cover the FEATURE LOOKBACK, not just the embargo, else the downstream
    # window's first observations read prior-split returns (R18). The guard enforces max(embargo, lookback)
    # so the leakage invariant does NOT rest solely on resolve_windows producing a 60-session gap — a
    # future revert is caught here. ``lookback=0`` (default) preserves the legacy embargo-only callers.
    purge = max(int(embargo), int(lookback))
    if val_window[0] < train_window[1] + purge:
        raise ValueError(
            f"val_window {val_window} must start at/after train_window end "
            f"{train_window[1]} + max(embargo {embargo}, lookback {lookback}) (disjoint, purged splits, R18)"
        )
    if test_window is not None and test_window[0] < val_window[1] + purge:
        raise ValueError(
            f"test_window {test_window} must start at/after val_window end "
            f"{val_window[1]} + max(embargo {embargo}, lookback {lookback}) (sealed, purged test leg, R18)"
        )

    def _builder(reward_fn: RewardFn) -> EnvBundle:
        return EnvBundle(
            panel,
            cfg,
            reward_fn,
            train_window,
            val_window,
            test_window,
            cost_bps=cost_bps,
        )

    return _builder
