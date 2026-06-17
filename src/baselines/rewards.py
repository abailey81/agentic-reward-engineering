"""Hand-designed reward canon for portfolio allocation agents.

Purpose
-------
This module holds the *fixed*, human-authored reward functions against which
LLM-discovered and search-discovered rewards are benchmarked (FINAL_PLAN F.6).
Every reward in this module obeys a single, audited contract so that any agent
training loop can swap rewards without code changes. The contract is the unit
of comparison for the entire dissertation: discovered rewards must satisfy the
same signature, so the baselines define the *shape* of all legal rewards.

Reward contract (audit B-4)
---------------------------
Every reward is a callable::

    def reward(weights, returns, prev_weights, port_ret, info)
        -> tuple[float, dict[str, float], object]

where:
    weights      : portfolio weights chosen for the current step (simplex).
    returns      : per-asset realized returns for the current step.
    prev_weights : portfolio weights held into the current step (for turnover).
    port_ret     : realized scalar portfolio return for the current step.
    info         : free-form dict of auxiliary signals (e.g. risk estimates).

returns a triple:
    total        : scalar reward fed to the RL algorithm.
    components   : dict mapping named sub-terms -> their scalar contribution
                   (for diagnostics, plotting, and audit trails).
    reward_state : opaque carry object threaded between consecutive calls.
                   STATELESS rewards return it unchanged (or ``None``);
                   STATEFUL rewards (``differential_sharpe``) mutate/return it.

The ``reward_state`` slot is what makes online/recursive rewards expressible
without hidden globals — it is the *only* legal place to keep memory between
steps, which keeps every reward replayable and auditable (audit B-4).

Reward canon (FINAL_PLAN F.6; EXPANDED 4 -> 11 on 2026-07-26)
--------------------------------------------------------------
The canon is :data:`REWARD_CANON` at the foot of this module. Since the 2026-07-26
expansion it **IS** the registered H1 / N6 comparator family — there is no longer a
"frozen four" subset distinct from a wider secondary panel::

    config/preregistration.yaml::h1_baselines
      == config/campaign.yaml::h1_baselines          (bound by scripts/freeze.py::assert_h1_baselines_match)
      == config/eureka_loop.yaml::baseline_rewards   (bound by tests/test_baselines.py)
      == REWARD_CANON.keys()

    raw_return                  : the bare portfolio return (myopic, risk-NEUTRAL floor).
    return_minus_variance       : return penalized by a variance proxy.
    return_minus_cvar           : return penalized by tail risk (CVaR).
    differential_sharpe         : differential (online) Sharpe ratio (Moody, Wu, Liao & Saffell 1998;
                                  Moody & Saffell 2001), STATEFUL via ``reward_state``.
    differential_downside_ratio : the DOWNSIDE companion of the above (Moody & Saffell 2001,
                                  eqs. 19-24), STATEFUL.
    mean_variance_utility       : Markowitz quadratic utility r - 0.5*lambda*var (Markowitz 1952).
    return_minus_drawdown       : running log-wealth drawdown penalty (Chekhlov et al. 2005), STATEFUL.
    return_minus_downside       : Sortino downside semi-deviation (Sortino & van der Meer 1991).
    return_minus_turnover       : transaction-cost / turnover penalty (Garleanu & Pedersen 2013).
    log_growth                  : growth-optimal Kelly log return (Kelly 1956; Thorp 1971).
    volatility_scaled_return    : volatility-TARGETED return (Zhang, Zohren & Roberts 2020).

Tests (tests/test_baselines.py)
-------------------------------
    - test_rewards_obey_contract: each reward returns (total, components, state)
      under the contract.
    - test_differential_sharpe_sequence / test_differential_downside_ratio_sequence:
      the two STATEFUL rewards reproduce hand-computed update sequences.
    - test_differential_downside_ratio_asymmetry: DDR penalizes downside asymmetrically.
    - test_secondary_panel_config_matches_reward_canon: the config list == REWARD_CANON
      in BOTH directions (so a name can never drift out of the registered family).
    - test_volatility_scaled_return_targets_volatility,
      test_return_minus_variance_penalizes_volatility, test_return_minus_cvar_runs.
"""

from __future__ import annotations


from typing import Any

import numpy as np


def raw_return(
    weights: Any,
    returns: Any,
    prev_weights: Any,
    port_ret: float,
    info: dict[str, Any],
) -> tuple[float, dict[str, float], object]:
    """Bare portfolio return reward (myopic baseline).

    This is the **field-standard FinRL-default reward**: FinRL's portfolio/stock-trading environments
    reward the (risk-neutral) one-step change in wealth — ``StockPortfolioEnv`` rewards the new portfolio
    value ``V_{t-1}(1 + Σ wᵢ rᵢ)`` and ``StockTradingEnv`` rewards ``V_t − V_{t-1}`` (Liu et al. 2020,
    arXiv:2011.09607; 2021, arXiv:2111.09395). Including it names the most-cited DRL-finance reward as the
    panel FLOOR; it is risk-NEUTRAL, so beating it on a risk-sensitive objective is near-automatic (the
    binding "human bar" in H1's max-over-panel is ``return_minus_cvar`` / ``differential_sharpe``).

    Algorithm sketch
    -----------------
    total = port_ret; components = {"raw_return": port_ret}; state unchanged.
    Stateless: reward_state is ``None`` and ignored.

    FINAL_PLAN F.6 (reward canon, raw return).
    """
    total = float(port_ret)
    components = {"raw_return": total}
    return total, components, info.get("reward_state")


def return_minus_variance(
    weights: Any,
    returns: Any,
    prev_weights: Any,
    port_ret: float,
    info: dict[str, Any],
) -> tuple[float, dict[str, float], object]:
    """Return penalized by a variance proxy.

    Algorithm sketch
    -----------------
    var = population variance over a rolling window of realized portfolio
    returns carried in ``reward_state``; total = port_ret - lambda * var.
    components = {"return": port_ret, "variance_penalty": -lambda * var}.

    The rolling window of realized portfolio returns is carried in
    ``reward_state`` (a list) so the variance is estimated online without hidden
    globals. ``info`` may supply ``lambda`` (risk-aversion) and ``window``.

    FINAL_PLAN F.6 (reward canon, mean-variance).
    """
    lam = float(info.get("lambda", 1.0))
    window = int(info.get("window", 20))

    state = info.get("reward_state")
    history: list[float] = list(state) if state is not None else []
    history.append(float(port_ret))
    if len(history) > window:
        history = history[-window:]

    # Population variance (ddof=0) over the rolling window (0.0 until we have >= 2 obs). DELIBERATE
    # convention, not drift (audit 2026-07-02): a reward PENALTY needs a scale — lambda absorbs the
    # ~(n-1)/n factor — while the INFERENTIAL machinery (deflated_sharpe._sample_moments, the allocators)
    # uses ddof=1 where the statistics demand an unbiased estimator. The two are intentionally different.
    var = float(np.var(history)) if len(history) >= 2 else 0.0
    penalty = lam * var
    total = float(port_ret) - penalty
    components = {
        "return": float(port_ret),
        "variance": var,
        "variance_penalty": -penalty,
    }
    return total, components, history


def return_minus_cvar(
    weights: Any,
    returns: Any,
    prev_weights: Any,
    port_ret: float,
    info: dict[str, Any],
) -> tuple[float, dict[str, float], object]:
    """Return penalized by tail risk (Conditional Value-at-Risk).

    Algorithm sketch
    -----------------
    cvar = CVaR estimate at level alpha (from ``info`` history or a rolling
    window of portfolio returns); total = port_ret - lambda * cvar.
    components = {"return": port_ret, "cvar_penalty": -lambda * cvar}.

    The recent realized portfolio returns are carried in ``reward_state`` (a
    rolling list); CVaR penalizes the lower tail (mean of the worst ``alpha``
    fraction). ``info`` may supply ``lambda``, ``alpha`` (tail level), and
    ``window``.

    FINAL_PLAN F.6 (reward canon, mean-CVaR).
    """
    lam = float(info.get("lambda", 1.0))
    alpha = float(info.get("alpha", 0.05))
    window = int(info.get("window", 50))

    state = info.get("reward_state")
    history: list[float] = list(state) if state is not None else []
    history.append(float(port_ret))
    if len(history) > window:
        history = history[-window:]

    arr = np.asarray(history, dtype=float)
    # Lower-tail Value-at-Risk threshold; CVaR = mean of returns at or below it. Estimator note (audit
    # 2026-07-02, closes a recurring false-positive): for the type-7 quantile, #{r <= Q(alpha)} =
    # floor((n-1)*alpha)+1 == ceil(n*alpha) at essentially every n, so this IS the standard
    # mean-of-worst-ceil(alpha*n) empirical CVaR used by the rest of the stack (reward_family,
    # measurement._empirical_cvar) — ~3 points at window=50/alpha=0.05 (1 during early warm-up), a
    # noisy few-point tail mean inherent to the window/alpha, not to the estimator choice.
    var_threshold = float(np.quantile(arr, alpha))
    tail = arr[arr <= var_threshold]
    cvar_loss = -float(tail.mean()) if tail.size > 0 else 0.0
    # Only penalize genuine downside (positive expected loss).
    tail_penalty = lam * max(cvar_loss, 0.0)
    total = float(port_ret) - tail_penalty
    components = {
        "return": float(port_ret),
        "cvar": cvar_loss,
        "cvar_penalty": -tail_penalty,
    }
    return total, components, history


def differential_sharpe(
    weights: Any,
    returns: Any,
    prev_weights: Any,
    port_ret: float,
    info: dict[str, Any],
) -> tuple[float, dict[str, float], object]:
    """Differential (online incremental) Sharpe ratio.

    The canonical risk-sensitive direct-RL trading reward. STATEFUL: it accumulates exponential moving
    averages of the first and second moments of the portfolio return in ``reward_state`` and emits the
    *marginal* contribution of the current return to the Sharpe ratio (audit B-4). It is the canonical
    example of a reward that legally uses the reward_state carry slot instead of hidden globals.

    Citation
    --------
    The differential-Sharpe derivation is from **Moody, Wu, Liao & Saffell (1998)**, "Performance
    functions and reinforcement learning for trading systems and portfolios," *J. Forecasting* 17(5-6):
    441-470 (FOUR authors). The canonical reference is **Moody & Saffell (2001)**, "Learning to Trade via
    Direct Reinforcement," *IEEE Trans. Neural Networks* 12(4):875-889 (the two-author paper) — cite 2001
    as primary. (The earlier "Moody & Saffell 1998" attribution conflated the two and is corrected here.)

    Algorithm sketch (Moody, Wu, Liao & Saffell 1998; Moody & Saffell 2001)
    -----------------------------------------
    State carries A (EMA of return) and B (EMA of squared return), with decay
    eta. On each step with return R_t:

        dA = R_t - A_{t-1}
        dB = R_t^2 - B_{t-1}
        D_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB)
              / (B_{t-1} - A_{t-1}^2) ** 1.5      # differential Sharpe
        A_t = A_{t-1} + eta * dA
        B_t = B_{t-1} + eta * dB

    total = D_t; components = {"dsr": D_t, "A": A_t, "B": B_t};
    reward_state = updated (A_t, B_t, eta) carry.

    Worked example
    --------------
    Let eta = 0.1, and initialize A_0 = 0, B_0 = 0 (warm-up returns D = 0 / a
    sentinel until B - A^2 > 0). Feed R_1 = 0.02 then R_2 = -0.01:

      Step 1 (R_1 = 0.02):
        dA = 0.02 - 0 = 0.02
        dB = 0.0004 - 0 = 0.0004
        denom = (0 - 0) ** 1.5 = 0  -> warm-up: D_1 = 0 (guarded).
        A_1 = 0 + 0.1 * 0.02 = 0.002
        B_1 = 0 + 0.1 * 0.0004 = 0.00004

      Step 2 (R_2 = -0.01):
        dA = -0.01 - 0.002 = -0.012
        dB = 0.0001 - 0.00004 = 0.00006
        denom = (B_1 - A_1**2) ** 1.5
              = (0.00004 - 0.000004) ** 1.5
              = (0.000036) ** 1.5
        D_2 = (B_1 * dA - 0.5 * A_1 * dB) / denom
            = (0.00004 * -0.012 - 0.5 * 0.002 * 0.00006) / (0.000036 ** 1.5)
        A_2 = 0.002 + 0.1 * -0.012 = 0.0008
        B_2 = 0.00004 + 0.1 * 0.00006 = 0.000046

    The test in tests/test_baselines.py replays exactly this A/B/eta sequence.

    FINAL_PLAN F.6 (reward canon, differential Sharpe; audit B-4 statefulness).
    """
    r = float(port_ret)

    state = info.get("reward_state")
    if state is None:
        eta = float(info.get("eta", 0.1))
        a_prev, b_prev = 0.0, 0.0
    else:
        a_prev = float(state["A"])
        b_prev = float(state["B"])
        eta = float(state["eta"])

    d_a = r - a_prev
    d_b = r * r - b_prev

    denom_base = b_prev - a_prev * a_prev
    # Warm-up guard: with A_0 = B_0 = 0 (or any degenerate variance) the
    # differential Sharpe is undefined; emit 0.0 until variance is positive.
    # Disclosure (audit 2026-07-02): this warm-up 0.0 IS the live reward SAC receives on the first
    # step(s) — one step in thousands for the frozen H1 `differential_sharpe` baseline (benign, but
    # stated rather than silent; standard practice for the online DSR, whose D_1 is undefined).
    if denom_base > 0.0:
        dsr = (b_prev * d_a - 0.5 * a_prev * d_b) / (denom_base**1.5)
    else:
        dsr = 0.0

    a_new = a_prev + eta * d_a
    b_new = b_prev + eta * d_b

    total = float(dsr)
    components = {"dsr": total, "A": float(a_new), "B": float(b_new)}
    new_state = {"A": float(a_new), "B": float(b_new), "eta": eta}
    return total, components, new_state


# =========================================================================== #
# Extended canon (block B8): more published / canonical reward designs run as   #
# HAND-CRAFTED BASELINES so the LLM-discovered rewards are compared not only to  #
# scalar-feedback LLM rewards but to the standard literature designs.            #
# 2026-07-26: these are NO LONGER "secondary" — the H1 canon was expanded      #
# 4 -> 11, so every member below is part of the registered H1/N6 comparator      #
# family. They remain DISJOINT from the H2 arm roster (config/arms.yaml): H2     #
# contrasts FEEDBACK arms, H1/N6 contrasts the LLM winner against these HUMAN    #
# rewards. The m=6 fed vector and H2's multiplicity are untouched.               #
# =========================================================================== #
def differential_downside_ratio(
    weights: Any,
    returns: Any,
    prev_weights: Any,
    port_ret: float,
    info: dict[str, Any],
) -> tuple[float, dict[str, float], object]:
    """Differential (online incremental) downside deviation ratio (DDR).

    The DOWNSIDE companion of :func:`differential_sharpe` from the SAME canonical paper — the
    tail-asymmetric member of the human canon (R97, 2026-07-22): it "rewards the presence of large
    average positive returns and penalizes risky returns, where 'risky' now refers to downside
    returns" (Moody & Saffell 2001, p. 879). A hand-written panel facing a distributional-vs-scalar
    tail-feedback study MUST include the canonical downside-sensitive online reward, not only the
    symmetric-Sharpe half of its own primary citation. STATEFUL via ``reward_state``.

    Citation (formulas transcribed FIRST-HAND from the on-disk PDF, p. 879)
    ----------------------------------------------------------------------
    **Moody & Saffell (2001)**, "Learning to Trade via Direct Reinforcement," *IEEE Trans. Neural
    Networks* 12(4):875-889, eqs. (19)-(24). DD is the square root of the mean SQUARED NEGATIVE
    return (19); DDR = Average(R)/DD (20); the differential form is the first-order term of the
    expansion in the adaptation rate eta (22). Footnote 7 there notes the DD "tracks the Sterling
    ratio effectively" (White), so this member also proxies the drawdown-RATIO axis
    (Calmar/Sterling; cf. Almahdi & Yang 2017) alongside the return_minus_drawdown PENALTY form.

    Algorithm sketch (Moody & Saffell 2001, eqs. (21), (23), (24))
    --------------------------------------------------------------
    State carries A (EMA of return) and DD2 (EMA of squared negative return), decay eta:

        A_t   = A_{t-1}   + eta * (R_t - A_{t-1})
        DD2_t = DD2_{t-1} + eta * (min{R_t, 0}^2 - DD2_{t-1})           # (21)

        D_t = (R_t - 0.5*A_{t-1}) / DD_{t-1}                            # R_t > 0   (23)
        D_t = (DD2_{t-1}*(R_t - 0.5*A_{t-1}) - 0.5*A_{t-1}*R_t^2)
              / DD_{t-1}^3                                              # R_t <= 0  (24)

    with DD_{t-1} = sqrt(DD2_{t-1}). Per (24), a positive return carries NO penalty however large
    (unlike the variance-based DSR) while a negative return is penalized through the cubic
    denominator — the asymmetry that makes this the tail-sensitive human reward.

    Worked example (replayed exactly in tests/test_baselines.py)
    ------------------------------------------------------------
    eta = 0.1, A_0 = DD2_0 = 0. Feed R = 0.02, -0.01, 0.005, -0.02:
      Step 1 (0.02):  DD_0 = 0 -> warm-up D_1 = 0; A_1 = 0.002, DD2_1 = 0 (positive return).
      Step 2 (-0.01): DD_1 = 0 -> warm-up D_2 = 0; A_2 = 0.0008, DD2_2 = 1e-05.
      Step 3 (0.005): R > 0: D_3 = (0.005 - 0.0004)/sqrt(1e-05) ~= 1.454653;
                      A_3 = 0.00122, DD2_3 = 9e-06.
      Step 4 (-0.02): R <= 0: D_4 = (9e-06*(-0.02 - 0.00061) - 0.00061*0.0004)/(9e-06)**1.5
                      ~= -15.907037; A_4 = -0.000902, DD2_4 = 4.81e-05.

    Warm-up disclosure (mirrors the DSR's): D is undefined until a NEGATIVE return has entered
    DD2 (DD_{t-1} = 0), so 0.0 is emitted during warm-up — the live reward SAC receives on those
    early steps, stated rather than silent.

    Registered status (CORRECTED 2026-07-26 — this paragraph previously asserted the OPPOSITE)
    ---------------------------------------------------------------------------------------------
    Since the H1 canon expansion 4 -> 11 (R108, ratified together with ``n6_h1_confirmatory_node``),
    this member **IS** part of the registered H1 family and is one of the ELEVEN legs of CONFIRMATORY
    node **N6**, whose intersection-union p-value is the MAX over the legs. It is no longer a
    "secondary panel" member, and there is no longer a "frozen H1 four" for it to sit outside of (see
    the module header and ``config/eureka_loop.yaml``, both of which say so).
    On multiplicity, precisely: adding legs does NOT inflate alpha — an IUT is level-alpha for free
    (Berger 1982) — but it does make N6 strictly HARDER to reject, since the LLM winner must beat
    EVERY member. So the old "(multiplicity unchanged)" was right about FWER and wrong about
    membership; the membership claim is what made it dangerous.
    """
    r = float(port_ret)

    state = info.get("reward_state")
    if state is None:
        eta = float(info.get("eta", 0.1))
        a_prev, dd2_prev = 0.0, 0.0
    else:
        a_prev = float(state["A"])
        dd2_prev = float(state["DD2"])
        eta = float(state["eta"])

    # Warm-up guard: DD_{t-1} = 0 (no negative return seen yet, or degenerate state) leaves both
    # branches of (23)/(24) undefined; emit 0.0 until DD2 is strictly positive.
    if dd2_prev > 0.0:
        dd_prev = dd2_prev**0.5
        if r > 0.0:
            ddr = (r - 0.5 * a_prev) / dd_prev  # (23)
        else:
            ddr = (dd2_prev * (r - 0.5 * a_prev) - 0.5 * a_prev * r * r) / (dd_prev**3)  # (24)
    else:
        ddr = 0.0

    neg_sq = min(r, 0.0) ** 2
    a_new = a_prev + eta * (r - a_prev)
    dd2_new = dd2_prev + eta * (neg_sq - dd2_prev)

    total = float(ddr)
    components = {"ddr": total, "A": float(a_new), "DD2": float(dd2_new)}
    new_state = {"A": float(a_new), "DD2": float(dd2_new), "eta": eta}
    return total, components, new_state


def mean_variance_utility(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Markowitz quadratic utility ``r - 0.5*lambda*var`` (Markowitz 1952, *J. Finance*).

    The canonical mean-variance objective (note the 0.5 coefficient that distinguishes it from the plain
    ``return_minus_variance``). ``var`` is the population variance over a rolling window of realised
    portfolio returns carried in ``reward_state``; ``info`` may supply ``lambda`` (risk aversion) and
    ``window``.
    """
    lam = float(info.get("lambda", 1.0))
    window = int(info.get("window", 20))
    state = info.get("reward_state")
    hist: list[float] = list(state) if state is not None else []
    hist.append(float(port_ret))
    if len(hist) > window:
        hist = hist[-window:]
    var = float(np.var(hist)) if len(hist) >= 2 else 0.0
    penalty = 0.5 * lam * var
    total = float(port_ret) - penalty
    return total, {"return": float(port_ret), "variance": var, "mv_penalty": -penalty}, hist


def return_minus_drawdown(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Return penalised by the running DRAWDOWN (Chekhlov, Uryasev & Zabarankin 2005, *IJTAF*).

    Drawdown-aware allocation: accumulate log-wealth, track its running peak, and penalise the current
    shortfall from the peak. STATEFUL via ``reward_state = (cum_log, peak)``. ``info`` may supply
    ``lambda``.
    """
    lam = float(info.get("lambda", 1.0))
    state = info.get("reward_state")
    cum = float(state[0]) if state is not None else 0.0
    peak = float(state[1]) if state is not None else 0.0
    # Clip port_ret > -1 before log1p: a <= -100% step would give log1p(<=-1) = -inf/NaN -> a non-finite
    # reward (silently SAFE_DEFAULT-substituted downstream). A full-wipeout step is floored at -99.99%.
    cum = cum + float(np.log1p(max(float(port_ret), -0.9999)))
    peak = max(peak, cum)
    drawdown = peak - cum  # >= 0 (log-wealth shortfall from the peak)
    total = float(port_ret) - lam * drawdown
    return total, {"return": float(port_ret), "drawdown": drawdown, "dd_penalty": -lam * drawdown}, (cum, peak)


def return_minus_downside(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Return penalised by DOWNSIDE semi-deviation (Sortino & van der Meer 1991, *J. Portfolio Mgmt*).

    Penalises only volatility BELOW a target (downside risk), the Sortino philosophy: a rolling
    downside semi-deviation about ``target`` (default 0) carried in ``reward_state``. ``info`` may supply
    ``lambda``, ``target``, ``window``.
    """
    lam = float(info.get("lambda", 1.0))
    target = float(info.get("target", 0.0))
    window = int(info.get("window", 20))
    state = info.get("reward_state")
    hist: list[float] = list(state) if state is not None else []
    hist.append(float(port_ret))
    if len(hist) > window:
        hist = hist[-window:]
    arr = np.asarray(hist, dtype=float)
    short = np.minimum(arr - target, 0.0)
    dsd = float(np.sqrt(np.mean(short**2))) if arr.size >= 1 else 0.0
    total = float(port_ret) - lam * dsd
    return total, {"return": float(port_ret), "downside_dev": dsd, "downside_penalty": -lam * dsd}, hist


def return_minus_turnover(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Net return penalised by TURNOVER / transaction cost (Gârleanu & Pedersen 2013, *J. Finance*).

    The transaction-cost-aware objective: ``port_ret - kappa * turnover``, where turnover is the one-way
    L1 weight change ``0.5*sum|w - w_prev|``. Stateless. ``info`` may supply ``kappa``.
    """
    kappa = float(info.get("kappa", 1.0))
    w = np.asarray(weights, dtype=float)
    wp = np.asarray(prev_weights, dtype=float)
    n = min(w.size, wp.size)
    turnover = 0.5 * float(np.abs(w[:n] - wp[:n]).sum())
    total = float(port_ret) - kappa * turnover
    return total, {"return": float(port_ret), "turnover": turnover, "turnover_penalty": -kappa * turnover}, info.get("reward_state")


def log_growth(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Growth-optimal (Kelly) log return ``log(1 + port_ret)`` (Kelly 1956; Thorp 1971).

    Maximising expected log-wealth is the long-run growth-optimal criterion and is implicitly
    risk-averse (concavity penalises large drawdowns). Stateless.
    """
    # Clip port_ret > -1 before log1p (mirrors return_minus_drawdown): a <= -100% step would give
    # log1p(<=-1) = -inf/NaN. Unreachable on the finite gold panels (worst single-asset day ~ -0.90, and
    # the env raises on a drifted wipeout first), but kept consistent + defensive (critical-review 2026-06-20).
    total = float(np.log1p(max(float(port_ret), -0.9999)))
    return total, {"log_return": total, "return": float(port_ret)}, info.get("reward_state")


def volatility_scaled_return(
    weights: Any, returns: Any, prev_weights: Any, port_ret: float, info: dict[str, Any]
) -> tuple[float, dict[str, float], object]:
    """Volatility-TARGETED (risk-scaled) return (Zhang, Zohren & Roberts 2020, *J. Financial Data Science*).

    The canonical deep-RL-for-trading reward that TARGETS a fixed volatility: scale the raw return by
    ``sigma_target / sigma_realized``, where ``sigma_realized`` is a rolling std of recent returns carried
    in ``reward_state``. Risk-targeting normalises the signal across volatility regimes (the position is
    implicitly levered down in turbulent markets, up in calm ones). STATEFUL via a rolling window; ``info``
    may supply ``sigma_target``, ``window``. The scale is capped to bound the leverage-like blow-up when the
    realised vol is near zero (mirrors the defensive clipping in the other canon members).
    """
    sigma_target = float(info.get("sigma_target", 0.01))   # per-step target volatility
    window = int(info.get("window", 20))
    state = info.get("reward_state")
    hist: list[float] = list(state) if state is not None else []
    hist.append(float(port_ret))
    if len(hist) > window:
        hist = hist[-window:]
    arr = np.asarray(hist, dtype=float)
    sigma = float(np.std(arr)) if arr.size >= 2 else sigma_target
    scale = min(sigma_target / sigma, 10.0) if sigma > 1e-9 else 1.0   # cap the leverage-like scaling
    total = float(port_ret) * scale
    return total, {"return": float(port_ret), "vol": sigma, "vol_scale": scale}, hist


#: The full hand-crafted reward canon (name -> callable). Since the 2026-07-26 expansion (4 -> 11)
#: this IS the registered H1 family and the comparator set of CONFIRMATORY node N6 — "did the LLM beat
#: the standard literature rewards?" is now a confirmatory question, not a secondary one, so every
#: member below carries a leg of N6's intersection-union test. Stateful rewards thread ``reward_state``.
REWARD_CANON: dict[str, Any] = {
    "raw_return": raw_return,
    "return_minus_variance": return_minus_variance,
    "return_minus_cvar": return_minus_cvar,
    "differential_sharpe": differential_sharpe,
    "differential_downside_ratio": differential_downside_ratio,
    "mean_variance_utility": mean_variance_utility,
    "return_minus_drawdown": return_minus_drawdown,
    "return_minus_downside": return_minus_downside,
    "return_minus_turnover": return_minus_turnover,
    "log_growth": log_growth,
    "volatility_scaled_return": volatility_scaled_return,
}
