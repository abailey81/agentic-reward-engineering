"""H4 parametric reward family — the BO-over-template search space (FINAL_PLAN B.5 / ADR-010).

Both H4 search arms draw from this FIXED six-term linear reward family rather than free-form
code: the Bayesian-optimization arm (H4b) GP-EI-tunes its six weights, and the random-search
control (H4a, ``src/search/random_search.py``) draws the same six weights from a coarse grid over
the same ranges at the same fixed discrete dims. Comparing the LLM's free-form code against the
best member of this family isolates "reward-form richness" from "search procedure"
(PREREGISTRATION §4a). The family's vertices recover the hand-designed canon (return / turnover /
drawdown / CVaR / volatility), so H4b is a like-for-like comparison of free-form LLM code against
the best member of this PARAMETRIC family. It is NOT a like-for-like SEARCH-SPACE comparison: the
LLM authors strictly richer free-form code, so a positive H4 reflects open-ended-language plus
procedure, not procedure alone. H4a (random search over the SAME family)
is the procedure-only control at comparable expressive power.

The archived pre-merge version (``archive/pre_merge_repo_B/src_flat/reward_family.py``) used the
OLD ``RewardContext`` contract and is NOT compatible with the live engine. This is the
LIVE-contract implementation (authored per the P5 API audit — the function existed only as an
injected parameter name in ``src/agents/evaluator.py``):

    reward(weights, returns, prev_weights, port_ret, info) -> (total, components, reward_state)

The six weights are searched over the frozen ranges in ``config/eureka_loop.yaml: reward_family``.
The two discrete dims (``cvar_alpha``, ``window``) are FIXED for the continuous-box BO
(``bayes_opt_over_template`` handles continuous boxes only); the prototype fixes ``alpha=0.05``,
``window=20`` and searches the six weights.

    r = w_return*net_return + w_log*log1p(net_return)
        - w_turnover*turnover - w_drawdown*drawdown
        - w_cvar*max(0, -CVaR_alpha) - w_vol*sigma

Conventions (stated explicitly — loop 79, finding #52; Stefan's criterion 5 "clarity of what is
measured"). These were previously left implicit, and the config comment asserted the WRONG one:

    turnover : the **TWO-WAY** L1 change ``sum|w - w_prev|``, range ``[0, 2]`` (MEASURED 2.0 for a
               fully disjoint rebalance). NOTE: This DIFFERS from the H1 canon member
               ``rewards.py::return_minus_turnover``, which uses the ONE-WAY ``0.5*sum|w - w_prev|``
               in ``[0, 1]`` (Garleanu-Pedersen). Both are legitimate; they are simply not the same
               quantity, and each family's weight absorbs the factor of 2. Do not "harmonise" one to
               the other without a pre-registration amendment — the H4 box is registered.
    drawdown : LOG-wealth shortfall from the running peak (``peak - cum``), >= 0.
    CVaR     : mean of the worst ``ceil(alpha * n)`` returns in the rolling window — the stack-wide
               convention (``inference.bootstrap.cvar`` / ``measurement._empirical_cvar``); only a
               genuine loss is penalised, via ``max(0, -CVaR)``.
    sigma    : rolling ``np.std(..., ddof=0)`` — a penalty SCALE, not an estimator (same deliberate
               choice as ``rewards.py::return_minus_variance``).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["WEIGHT_KEYS", "params_to_reward", "params_to_source", "family_bounds"]

#: The six family weights, in coefficient-vector order (matches ``family_bounds`` rows).
WEIGHT_KEYS: tuple[str, ...] = (
    "w_return",
    "w_log",
    "w_turnover",
    "w_drawdown",
    "w_cvar",
    "w_vol",
)

#: Frozen ranges (mirror ``config/eureka_loop.yaml: reward_family.weights``); used if no cfg given.
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "w_return": (0.0, 2.0),
    "w_log": (0.0, 2.0),
    "w_turnover": (0.0, 0.02),
    "w_drawdown": (0.0, 0.1),
    "w_cvar": (0.0, 5.0),
    "w_vol": (0.0, 2.0),
}


def family_bounds(cfg: Any = None) -> np.ndarray:
    """Return the ``(6, 2)`` ``[low, high]`` box for the six family weights (frozen ranges).

    Reads ``cfg["reward_family"]["weights"]`` when provided (each entry a ``{low, high}`` dict or
    a ``[low, high]`` pair); otherwise uses the frozen defaults. The row order is :data:`WEIGHT_KEYS`.
    """
    weights = None
    if cfg is not None:
        rf = cfg.get("reward_family") if isinstance(cfg, dict) else getattr(cfg, "reward_family", None)
        if rf is not None:
            weights = rf.get("weights") if isinstance(rf, dict) else getattr(rf, "weights", None)
    rows: list[tuple[float, float]] = []
    for key in WEIGHT_KEYS:
        if weights is not None and key in weights:
            w = weights[key]
            lo, hi = (w["low"], w["high"]) if isinstance(w, dict) else (w[0], w[1])
        else:
            lo, hi = _DEFAULT_BOUNDS[key]
        rows.append((float(lo), float(hi)))
    return np.asarray(rows, dtype=float)


def params_to_reward(
    coeffs: Sequence[float],
    cvar_alpha: float = 0.05,
    window: int = 20,
) -> Callable:
    """Build a LIVE-contract reward callable from a 6-vector of family weights.

    Parameters
    ----------
    coeffs : Sequence[float]
        The six weights ``(w_return, w_log, w_turnover, w_drawdown, w_cvar, w_vol)`` — the BO
        candidate (``np.ndarray`` from ``bayes_opt_over_template``).
    cvar_alpha : float
        Tail level for the rolling-CVaR penalty term (fixed dim).
    window : int
        Rolling window for the volatility / CVaR / drawdown terms (fixed dim).

    Returns
    -------
    Callable
        A stateful reward conforming to the contract (rolling stats via ``reward_state``).
    """
    c = [float(x) for x in np.asarray(coeffs, dtype=float).ravel()]
    if len(c) != len(WEIGHT_KEYS):
        raise ValueError(f"reward family expects {len(WEIGHT_KEYS)} weights, got {len(c)}")
    w_return, w_log, w_turnover, w_drawdown, w_cvar, w_vol = c
    alpha = float(cvar_alpha)
    win = int(window)

    def reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        state = info.get("reward_state")
        if state is None:
            hist, peak, cum = [], 0.0, 0.0
        else:
            hist, peak, cum = list(state[0]), float(state[1]), float(state[2])
        r = float(port_ret)
        hist.append(r)
        if len(hist) > win:
            hist = hist[-win:]
        arr = np.asarray(hist, dtype=float)
        turnover = float(np.sum(np.abs(np.asarray(weights) - np.asarray(prev_weights))))
        # Clip port_ret > -1 before log1p (mirrors rewards.py return_minus_drawdown/log_growth): a
        # <= -100% step gives log1p(<=-1) = -inf/NaN, which here would also PERMANENTLY poison the
        # stateful cum/peak/drawdown carry for the rest of the rollout (critical-review 2026-06-20).
        cum = cum + float(np.log1p(max(r, -0.9999)))
        peak = max(peak, cum)
        drawdown = peak - cum
        sigma = float(np.std(arr)) if arr.size > 1 else 0.0  # ddof=0: penalty SCALE, not an estimator (see rewards.py return_minus_variance note)
        k = max(1, int(np.ceil(alpha * arr.size)))
        cvar = float(np.mean(np.sort(arr)[:k]))
        total = (
            w_return * r
            + w_log * float(np.log1p(max(r, -0.9999)))  # same -1 clip as the cum term above
            - w_turnover * turnover
            - w_drawdown * drawdown
            - w_cvar * max(0.0, -cvar)
            - w_vol * sigma
        )
        components = {
            "return": r,
            "turnover": turnover,
            "drawdown": float(drawdown),
            "cvar": cvar,
            "sigma": sigma,
        }
        return float(total), components, (hist, peak, cum)

    return reward


def params_to_source(
    coeffs: Sequence[float],
    cvar_alpha: float = 0.05,
    window: int = 20,
) -> str:
    """Materialize the H4 family member at ``coeffs`` as EXECUTABLE reward source (Rank 2c).

    The BO arm (H4b) searches over the parametric family and only ever holds an in-memory
    closure (:func:`params_to_reward`). For the FROZEN winner to round-trip through the sealed
    TEST leg on the IDENTICAL path as the LLM / random-search arms, its archived
    ``reward_source`` must be a runnable ``def reward(...)`` — NOT a ``# coeffs=[...]`` comment
    stub (which ``run_campaign._reinstantiate_frozen_winner`` -> ``validate_once`` cannot
    rehydrate, recording the arm ``winner_not_testable``).

    This emits the SAME six-term family computation as :func:`params_to_reward`, with the six
    weights, the (fixed) CVaR level ``cvar_alpha`` and the rolling ``window`` substituted in as
    literals. The emitted source:

      - obeys the canonical contract ``reward(weights, returns, prev_weights, port_ret, info)
        -> (total, components, reward_state)`` (``validate_signature`` accepts it);
      - passes ``ast_gate`` (numpy-only attributes, no banned IO/FFI, no dunders, no forbidden
        calls), so ``validate_once`` admits it;
      - threads the SAME ``reward_state`` shape ``(hist, peak, cum)`` and the same numpy ops as
        the closure, so it reproduces ``params_to_reward(coeffs, cvar_alpha, window)`` output
        for output on any call sequence (to floating-point identity — the coefficients are
        emitted via ``repr`` for an exact round-trip).

    Parameters mirror :func:`params_to_reward`. Returns the reward source string to archive as
    the BO candidate's / winner's ``reward_source``.
    """
    c = [float(x) for x in np.asarray(coeffs, dtype=float).ravel()]
    if len(c) != len(WEIGHT_KEYS):
        raise ValueError(f"reward family expects {len(WEIGHT_KEYS)} weights, got {len(c)}")
    w_return, w_log, w_turnover, w_drawdown, w_cvar, w_vol = c
    alpha = float(cvar_alpha)
    win = int(window)
    # repr() on the floats/int gives an exact round-trip literal, so the materialized source is
    # bit-identical to the closure; the body is a verbatim transcription of params_to_reward.
    return (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        f"    w_return, w_log, w_turnover, w_drawdown, w_cvar, w_vol = (\n"
        f"        {w_return!r}, {w_log!r}, {w_turnover!r}, {w_drawdown!r}, {w_cvar!r}, {w_vol!r},\n"
        f"    )\n"
        f"    alpha = {alpha!r}\n"
        f"    win = {win!r}\n"
        "    state = info.get('reward_state')\n"
        "    if state is None:\n"
        "        hist, peak, cum = [], 0.0, 0.0\n"
        "    else:\n"
        "        hist, peak, cum = list(state[0]), float(state[1]), float(state[2])\n"
        "    r = float(port_ret)\n"
        "    hist.append(r)\n"
        "    if len(hist) > win:\n"
        "        hist = hist[-win:]\n"
        "    arr = np.asarray(hist, dtype=float)\n"
        "    turnover = float(np.sum(np.abs(np.asarray(weights) - np.asarray(prev_weights))))\n"
        "    cum = cum + float(np.log1p(max(r, -0.9999)))\n"
        "    peak = max(peak, cum)\n"
        "    drawdown = peak - cum\n"
        "    sigma = float(np.std(arr)) if arr.size > 1 else 0.0\n"
        "    k = max(1, int(np.ceil(alpha * arr.size)))\n"
        "    cvar = float(np.mean(np.sort(arr)[:k]))\n"
        "    total = (\n"
        "        w_return * r\n"
        "        + w_log * float(np.log1p(max(r, -0.9999)))\n"
        "        - w_turnover * turnover\n"
        "        - w_drawdown * drawdown\n"
        "        - w_cvar * max(0.0, -cvar)\n"
        "        - w_vol * sigma\n"
        "    )\n"
        "    components = {\n"
        "        'return': r,\n"
        "        'turnover': turnover,\n"
        "        'drawdown': float(drawdown),\n"
        "        'cvar': cvar,\n"
        "        'sigma': sigma,\n"
        "    }\n"
        "    return float(total), components, (hist, peak, cum)\n"
    )
