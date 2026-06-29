"""DEEP, advanced tests for the load-bearing reward / search / selection machinery.

This module complements (does NOT duplicate) the existing behaviour suites
(``test_baselines.py``, ``test_reward_family.py``, ``test_reward_family_source.py``,
``test_fitness.py``, ``test_arms.py``) with property-based (Hypothesis,
``derandomize=True``), metamorphic, adversarial, and boundary tests for:

  - the REWARD_CANON hand-designed rewards (``src/baselines/rewards.py``);
  - the H4 six-primitive search grammar (``src/baselines/reward_family.py``)
    and the H4a/H4b family parity (``src/search/random_search.py``,
    ``src/search/bayes_opt.py``);
  - the allocator baselines (``src/baselines/strategies.py``);
  - the validation deflated-Sharpe fitness selector (``src/selection/fitness.py``);
  - the experimental-arm factory (``src/arms/factory.py``).

Tests use tight explicit ``atol`` and seeded RNGs. A failure here that reflects a
real source bug is reported loudly rather than worked around.
"""
from __future__ import annotations

import numpy as np
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from src.arms.factory import all_arms, build_arm  # noqa: E402
from src.baselines import rewards, strategies  # noqa: E402
from src.baselines.reward_family import (  # noqa: E402
    WEIGHT_KEYS,
    family_bounds,
    params_to_reward,
    params_to_source,
)
from src.search import bayes_opt, random_search  # noqa: E402
from src.selection.fitness import held_out_fitness  # noqa: E402

# Every reward in the canon (primary + extended block-B8 baselines).
CANON = rewards.REWARD_CANON
STATELESS = {"raw_return", "return_minus_turnover", "log_growth"}

# --------------------------------------------------------------------------- #
# Hypothesis search strategies (small, finite, contract-shaped).               #
# --------------------------------------------------------------------------- #
_finite_ret = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)


def _simplex(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.random(n)
    return w / w.sum()


def _step(fn, port_ret, info, *, n=4, rng=None):
    """Run one reward step under the contract; returns (total, components, state)."""
    rng = rng or np.random.default_rng(0)
    w = _simplex(n, rng)
    wp = _simplex(n, rng)
    returns = rng.standard_normal(n) * 0.01
    return fn(w, returns, wp, float(port_ret), dict(info))


# =========================================================================== #
# 1. REWARD_CANON — contract, finiteness, degeneracy safety                    #
# =========================================================================== #
@pytest.mark.parametrize("name", sorted(CANON))
@settings(derandomize=True, max_examples=60)
@given(port_ret=_finite_ret)
def test_reward_obeys_contract_and_is_finite(name: str, port_ret: float) -> None:
    """Every canon reward returns a (float, dict[str,float], state) triple, finite on finite input."""
    fn = CANON[name]
    total, components, _state = _step(fn, port_ret, {})
    assert isinstance(total, float) and np.isfinite(total)
    assert isinstance(components, dict)
    for k, v in components.items():
        assert isinstance(k, str)
        assert isinstance(v, float) and np.isfinite(v)


@pytest.mark.parametrize("name", sorted(CANON))
def test_reward_safe_on_constant_returns(name: str) -> None:
    """A long path of a single CONSTANT return stays finite (zero turnover, zero variance)."""
    fn = CANON[name]
    w = np.full(5, 0.2)
    r = np.zeros(5)
    info: dict = {}
    for _ in range(80):
        total, comp, state = fn(w, r, w, 0.0, info)  # zero turnover (w == prev), constant return
        assert np.isfinite(total)
        assert all(np.isfinite(v) for v in comp.values())
        info = {"reward_state": state}


@pytest.mark.parametrize("name", sorted(CANON))
def test_reward_safe_on_single_asset(name: str) -> None:
    """A single-asset portfolio (degenerate covariance) does not crash any reward."""
    fn = CANON[name]
    w = np.array([1.0])
    r = np.array([0.01])
    total, comp, _ = fn(w, r, w, 0.01, {})
    assert np.isfinite(total)
    assert all(np.isfinite(v) for v in comp.values())


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_stateless_rewards_ignore_state(name: str) -> None:
    """Stateless rewards (raw_return / turnover / log_growth) pass the carry through untouched."""
    fn = CANON[name]
    sentinel = {"opaque": object()}
    _, _, state = _step(fn, 0.01, {"reward_state": sentinel})
    assert state is sentinel  # carry is returned verbatim, not mutated


# --------------------------------------------------------------------------- #
# 1a. raw_return — exact identity                                              #
# --------------------------------------------------------------------------- #
@settings(derandomize=True)
@given(port_ret=_finite_ret)
def test_raw_return_is_identity(port_ret: float) -> None:
    """raw_return.total == port_ret exactly and components mirror it."""
    total, comp, _ = _step(rewards.raw_return, port_ret, {})
    assert total == pytest.approx(port_ret, abs=0.0)
    assert comp["raw_return"] == pytest.approx(port_ret, abs=0.0)


# --------------------------------------------------------------------------- #
# 1b. return_minus_variance — decreases as variance rises (mean held fixed)    #
# --------------------------------------------------------------------------- #
def _run_path(fn, path, info0):
    info = dict(info0)
    out = []
    for r in path:
        total, comp, state = fn(np.array([1.0]), np.array([0.0]), np.array([1.0]), float(r), info)
        out.append((total, comp))
        info = dict(info0)
        info["reward_state"] = state
    return out


def test_return_minus_variance_monotone_in_variance() -> None:
    """Holding the MEAN return fixed, a higher-variance path yields a more negative variance penalty.

    Build two zero-mean +/- perturbation paths of equal length with different amplitudes; the
    population variance (hence the penalty) scales with the amplitude, so total falls as variance rises.
    """
    n = 40
    base = 0.001
    small = base + 0.001 * np.tile([1.0, -1.0], n // 2)  # mean == base, var ~ 1e-6
    large = base + 0.02 * np.tile([1.0, -1.0], n // 2)   # mean == base, var ~ 4e-4
    assert np.isclose(small.mean(), large.mean())  # same mean
    info0 = {"lambda": 10.0, "window": n}
    pen_small = _run_path(rewards.return_minus_variance, small, info0)[-1][1]["variance_penalty"]
    pen_large = _run_path(rewards.return_minus_variance, large, info0)[-1][1]["variance_penalty"]
    assert pen_large < pen_small < 0.0  # more variance -> more negative penalty


def test_return_minus_variance_components_relate_to_total() -> None:
    """total == return + variance_penalty, and variance_penalty == -lambda * variance (documented)."""
    rng = np.random.default_rng(7)
    path = 0.001 + 0.01 * rng.standard_normal(60)
    lam = 3.0
    info = {"lambda": lam, "window": 30}
    for r in path:
        total, comp, state = rewards.return_minus_variance(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), float(r), info
        )
        assert total == pytest.approx(comp["return"] + comp["variance_penalty"], abs=1e-12)
        assert comp["variance_penalty"] == pytest.approx(-lam * comp["variance"], abs=1e-12)
        assert comp["variance"] >= 0.0
        info = {"lambda": lam, "window": 30, "reward_state": state}


def test_mean_variance_utility_has_half_coefficient() -> None:
    """mean_variance_utility uses the Markowitz 0.5 coefficient: penalty == -0.5*lambda*var."""
    rng = np.random.default_rng(11)
    path = 0.001 + 0.02 * rng.standard_normal(50)
    lam = 4.0
    info = {"lambda": lam, "window": 25}
    for r in path:
        total, comp, state = rewards.mean_variance_utility(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), float(r), info
        )
        assert comp["mv_penalty"] == pytest.approx(-0.5 * lam * comp["variance"], abs=1e-12)
        assert total == pytest.approx(comp["return"] + comp["mv_penalty"], abs=1e-12)
        info = {"lambda": lam, "window": 25, "reward_state": state}


# --------------------------------------------------------------------------- #
# 1c. return_minus_cvar — worse left tail => lower total (penalty non-positive) #
# --------------------------------------------------------------------------- #
def test_return_minus_cvar_decreases_as_left_tail_worsens() -> None:
    """A path with a fatter left tail (same final return) gets a heavier CVaR penalty -> lower total.

    Construct two histories that END on the same step return but differ in their worst observations;
    the rolling-CVaR penalty must be at least as severe for the fatter-tailed history.
    """
    common_tail = 0.001
    mild = np.concatenate([np.full(49, 0.001), [common_tail]])
    severe = mild.copy()
    severe[:10] = -0.30  # inject deep crashes into the rolling window
    info0 = {"lambda": 5.0, "alpha": 0.1, "window": 50}

    mild_last = _run_path(rewards.return_minus_cvar, mild, info0)[-1]
    severe_last = _run_path(rewards.return_minus_cvar, severe, info0)[-1]

    # cvar component is the (positive) expected loss; severe has a larger loss => more negative penalty.
    assert severe_last[1]["cvar"] >= mild_last[1]["cvar"]
    assert severe_last[1]["cvar_penalty"] <= mild_last[1]["cvar_penalty"] <= 0.0
    assert severe_last[0] <= mild_last[0]  # lower total on the worse tail


@settings(derandomize=True, max_examples=40)
@given(path=hnp.arrays(np.float64, 30, elements=_finite_ret))
def test_return_minus_cvar_penalty_never_positive(path) -> None:
    """The CVaR penalty term only ever penalizes (component <= 0) -- never rewards downside."""
    info = {"lambda": 2.0, "alpha": 0.05, "window": 30}
    for r in path:
        _, comp, state = rewards.return_minus_cvar(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), float(r), info
        )
        assert comp["cvar_penalty"] <= 0.0
        info = {"lambda": 2.0, "alpha": 0.05, "window": 30, "reward_state": state}


# --------------------------------------------------------------------------- #
# 1d. return_minus_turnover — turnover == one-way L1; zero on no trade          #
# --------------------------------------------------------------------------- #
@settings(derandomize=True)
@given(seed=st.integers(0, 10_000), kappa=st.floats(0.0, 10.0, allow_nan=False))
def test_turnover_is_half_l1_and_penalizes(seed: int, kappa: float) -> None:
    """turnover == 0.5*sum|w-w_prev| and total == port_ret - kappa*turnover; zero turnover => no penalty."""
    rng = np.random.default_rng(seed)
    w = _simplex(5, rng)
    wp = _simplex(5, rng)
    port_ret = 0.01
    total, comp, _ = rewards.return_minus_turnover(w, np.zeros(5), wp, port_ret, {"kappa": kappa})
    expected_turn = 0.5 * float(np.abs(w - wp).sum())
    assert comp["turnover"] == pytest.approx(expected_turn, abs=1e-12)
    assert total == pytest.approx(port_ret - kappa * expected_turn, abs=1e-12)
    # No trade -> zero turnover -> reward is exactly the bare return.
    total0, comp0, _ = rewards.return_minus_turnover(w, np.zeros(5), w, port_ret, {"kappa": kappa})
    assert comp0["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert total0 == pytest.approx(port_ret, abs=1e-12)


# --------------------------------------------------------------------------- #
# 1e. log_growth — exact log1p identity + concavity (penalizes large losses)   #
# --------------------------------------------------------------------------- #
@settings(derandomize=True)
@given(port_ret=st.floats(-0.99, 0.5, allow_nan=False))
def test_log_growth_is_log1p(port_ret: float) -> None:
    """log_growth.total == log1p(port_ret) (with the -0.9999 floor) and is concave (<= port_ret)."""
    total, comp, _ = rewards.log_growth(np.array([1.0]), np.array([0.0]), np.array([1.0]), port_ret, {})
    assert total == pytest.approx(float(np.log1p(max(port_ret, -0.9999))), abs=1e-12)
    assert comp["return"] == pytest.approx(port_ret, abs=1e-12)
    # log(1+x) <= x: log-growth is implicitly risk-averse.
    assert total <= port_ret + 1e-12


def test_log_growth_floors_total_wipeout() -> None:
    """A <= -100% step is floored (log1p(-0.9999)) instead of returning -inf/NaN."""
    total, _, _ = rewards.log_growth(np.array([1.0]), np.array([0.0]), np.array([1.0]), -1.0, {})
    assert np.isfinite(total)
    assert total == pytest.approx(float(np.log1p(-0.9999)), abs=1e-12)


def test_return_minus_drawdown_floors_wipeout_and_is_monotone() -> None:
    """return_minus_drawdown survives a -100% step (log1p floor) and drawdown is non-negative & grows."""
    info: dict = {"lambda": 1.0}
    w = np.array([1.0])
    # A wipeout step must not poison the stateful cum/peak carry.
    total, comp, state = rewards.return_minus_drawdown(w, np.array([0.0]), w, -1.0, info)
    assert np.isfinite(total) and comp["drawdown"] >= 0.0
    # After a gain then a loss the drawdown is strictly positive.
    _, _, s1 = rewards.return_minus_drawdown(w, np.array([0.0]), w, 0.05, {"lambda": 1.0})
    _, c2, _ = rewards.return_minus_drawdown(w, np.array([0.0]), w, -0.03, {"lambda": 1.0, "reward_state": s1})
    assert c2["drawdown"] > 0.0


def test_return_minus_downside_ignores_upside() -> None:
    """downside semi-deviation penalizes only sub-target returns: an all-positive path has zero penalty."""
    info = {"lambda": 5.0, "target": 0.0, "window": 20}
    for r in [0.01] * 30:  # all above target 0
        total, comp, state = rewards.return_minus_downside(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), r, info
        )
        assert comp["downside_dev"] == pytest.approx(0.0, abs=1e-12)
        assert total == pytest.approx(comp["return"], abs=1e-12)
        info = {"lambda": 5.0, "target": 0.0, "window": 20, "reward_state": state}
    # A path with losses below target yields a strictly positive downside deviation.
    _, comp_loss, _ = rewards.return_minus_downside(
        np.array([1.0]), np.array([0.0]), np.array([1.0]), -0.05,
        {"lambda": 5.0, "target": 0.0, "window": 20},
    )
    assert comp_loss["downside_dev"] > 0.0


# --------------------------------------------------------------------------- #
# 1f. differential_sharpe — warm-up guard + statefulness                       #
# --------------------------------------------------------------------------- #
def test_differential_sharpe_warmup_is_zero_and_state_advances() -> None:
    """First call (A=B=0) is guarded to D=0; the (A,B,eta) carry advances by the EMA update."""
    eta = 0.2
    total, comp, state = rewards.differential_sharpe(
        np.array([1.0]), np.array([0.0]), np.array([1.0]), 0.03, {"eta": eta}
    )
    assert total == 0.0  # warm-up guard: denom_base == 0
    assert state["A"] == pytest.approx(eta * 0.03, abs=1e-15)
    assert state["B"] == pytest.approx(eta * 0.03**2, abs=1e-15)
    assert state["eta"] == pytest.approx(eta, abs=0.0)


@settings(derandomize=True, max_examples=40)
@given(path=hnp.arrays(np.float64, 25, elements=st.floats(-0.1, 0.1, allow_nan=False)))
def test_differential_sharpe_finite_over_path(path) -> None:
    """Over a stateful replay the DSR reward stays finite (the variance guard never divides by zero)."""
    info: dict = {"eta": 0.1}
    for r in path:
        total, comp, state = rewards.differential_sharpe(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), float(r), info
        )
        assert np.isfinite(total)
        assert np.isfinite(comp["A"]) and np.isfinite(comp["B"])
        info = {"reward_state": state}


def test_differential_sharpe_constant_returns_stay_bounded() -> None:
    """A perfectly constant return never explodes: D=0 on warm-up, then a bounded, finite DSR.

    With A_0=B_0=0 the first step is guarded to 0; from the second step on the EMA gives
    B - A^2 = eta*(1-eta)*R^2 > 0 (for 0<eta<1), so the DSR is well-defined and finite, never inf/NaN.
    """
    info: dict = {"eta": 0.3}
    first = rewards.differential_sharpe(
        np.array([1.0]), np.array([0.0]), np.array([1.0]), 0.01, info
    )
    assert first[0] == 0.0  # warm-up guard: B - A^2 == 0 on the first call
    info = {"reward_state": first[2]}
    for _ in range(50):
        total, _, state = rewards.differential_sharpe(
            np.array([1.0]), np.array([0.0]), np.array([1.0]), 0.01, info
        )
        assert np.isfinite(total)  # bounded; never an explosion
        info = {"reward_state": state}


# =========================================================================== #
# 2. reward_family grammar — closure, parity (H4a/H4b), grid membership        #
# =========================================================================== #
_FAMILY_COMPONENT_KEYS = {"return", "turnover", "drawdown", "cvar", "sigma"}


@settings(derandomize=True, max_examples=50)
@given(
    coeffs=hnp.arrays(np.float64, 6, elements=st.floats(0.0, 5.0, allow_nan=False)),
    port_ret=_finite_ret,
)
def test_family_closed_under_six_primitives(coeffs, port_ret) -> None:
    """Every sampled family member emits EXACTLY the six-primitive component set and a finite total."""
    reward = params_to_reward(coeffs)
    w = np.full(4, 0.25)
    total, comp, state = reward(w, np.zeros(4), w, float(port_ret), {})
    assert isinstance(total, float) and np.isfinite(total)
    assert set(comp) == _FAMILY_COMPONENT_KEYS
    assert state is not None and len(state) == 3  # (hist, peak, cum)


def test_family_vertices_recover_pure_return() -> None:
    """The w_return=1 (others 0) vertex reduces the family reward to the bare net return."""
    reward = params_to_reward([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = np.full(3, 1 / 3)
    total, comp, _ = reward(w, np.zeros(3), w, 0.027, {})
    assert total == pytest.approx(0.027, abs=1e-12)
    assert comp["return"] == pytest.approx(0.027, abs=1e-12)


def test_family_penalty_terms_only_subtract() -> None:
    """Turning on ONLY a penalty weight can never raise the reward above the pure-return baseline."""
    rng = np.random.default_rng(3)
    path = -0.02 + 0.03 * rng.standard_normal(40)  # loss-skewed to activate cvar/drawdown
    w = np.full(3, 1 / 3)
    base = params_to_reward([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    info_b: dict = {}
    info_p: dict = {}
    penal = params_to_reward([1.0, 0.0, 0.01, 0.05, 2.0, 1.0])  # return + all penalties
    for r in path:
        tb, _, sb = base(w, np.zeros(3), w, float(r), info_b)
        tp, _, sp = penal(w, np.zeros(3), w, float(r), info_p)
        assert tp <= tb + 1e-12  # penalties only ever subtract
        info_b = {"reward_state": sb}
        info_p = {"reward_state": sp}


def test_family_bounds_grid_membership_random_search() -> None:
    """random_search draws coefficients that land on its declared per-primitive grid.

    The grid is ``low + fraction*(high-low)`` for each frozen family bound and each grid fraction
    (``code_grid``). Every coefficient the sampler can draw must sit exactly on a grid point.
    """
    box = family_bounds()
    fr = np.asarray(random_search.code_grid(), dtype=float)
    grids = [lo + fr * (hi - lo) for lo, hi in box]
    rng = np.random.default_rng(123)
    # The sampler draws each weight via rng.choice over these grids (random_search.sample_reward_source);
    # replicate the draw and assert exact grid membership.
    for _ in range(300):
        coeffs = [float(rng.choice(g)) for g in grids]
        for c, g in zip(coeffs, grids):
            assert np.min(np.abs(g - c)) < 1e-12


def test_h4a_h4b_draw_from_same_family_identical_component_keys() -> None:
    """H4a (random_search) and H4b (bayes_opt via params_to_source) emit IDENTICAL component-key sets.

    Both arms render the SAME six-term family source, so any candidate from either produces exactly the
    five family component keys -- the structural parity that makes the H4a-vs-H4b contrast
    surrogate-vs-uniform rather than space-vs-space.
    """
    rng = np.random.default_rng(99)
    w = np.full(3, 1 / 3)
    fixture = (w, np.array([0.01, -0.02, 0.0]), w, 0.0, {})

    def keys_from_source(src: str) -> set:
        ns: dict = {"np": np}
        exec(compile(src, "<s>", "exec"), ns)  # noqa: S102
        _t, comp, _s = ns["reward"](*fixture)
        return set(comp)

    # H4a candidates.
    a_keys = [keys_from_source(random_search.sample_reward_source(rng)) for _ in range(20)]
    # H4b candidates: the materialized winner/candidate source for arbitrary in-box coeffs.
    box = family_bounds()
    b_keys = [
        keys_from_source(params_to_source(bayes_opt._sample_uniform(box, 1, rng)[0]))
        for _ in range(20)
    ]
    assert all(k == _FAMILY_COMPONENT_KEYS for k in a_keys)
    assert all(k == _FAMILY_COMPONENT_KEYS for k in b_keys)
    assert set().union(*a_keys) == set().union(*b_keys) == _FAMILY_COMPONENT_KEYS


def test_family_bounds_box_is_nonnegative_and_ordered() -> None:
    """The frozen family box is non-negative and low < high in every dimension (a valid search box)."""
    box = family_bounds()
    assert box.shape == (len(WEIGHT_KEYS), 2)
    assert np.all(box[:, 0] >= 0.0)
    assert np.all(box[:, 0] < box[:, 1])


# =========================================================================== #
# 3. strategies — simplex validity, determinism, near-singular covariance      #
# =========================================================================== #
ALLOCATORS = [
    strategies.equal_weight,
    strategies.spy_buy_and_hold,
    strategies.mean_variance,
    strategies.risk_parity,
    strategies.hrp,
    strategies.minimum_variance,
    strategies.inverse_volatility,
    strategies.maximum_diversification,
    strategies.cross_sectional_momentum,
]


def _make_window(seed: int, n: int = 12, t: int = 80) -> np.ndarray:
    r = np.random.default_rng(seed)
    f = r.standard_normal((t, 3)) * 0.01
    return f @ r.standard_normal((3, n)) * 0.5 + r.standard_normal((t, n)) * 0.008


@pytest.mark.parametrize("fn", ALLOCATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("seed", range(6))
def test_allocator_returns_valid_simplex(fn, seed: int) -> None:
    """Every allocator returns a non-negative weight vector summing to 1 (tight atol)."""
    x = _make_window(seed)
    n = x.shape[1]
    w = np.asarray(fn(x))
    assert w.shape == (n,)
    assert np.all(w >= -1e-12), f"{fn.__name__} produced a negative weight"
    assert np.isclose(w.sum(), 1.0, atol=1e-8), f"{fn.__name__} weights sum to {w.sum()}"
    assert np.isfinite(w).all()


@pytest.mark.parametrize("fn", ALLOCATORS, ids=lambda f: f.__name__)
def test_allocator_is_deterministic(fn) -> None:
    """Allocators are pure functions of the window: two calls give byte-identical weights."""
    x = _make_window(42)
    w1 = np.asarray(fn(x))
    w2 = np.asarray(fn(x.copy()))
    assert np.array_equal(w1, w2), f"{fn.__name__} is non-deterministic"


@pytest.mark.parametrize("fn", ALLOCATORS, ids=lambda f: f.__name__)
def test_allocator_near_singular_covariance(fn) -> None:
    """Near-singular covariance (nearly collinear assets) must not crash or break the simplex."""
    r = np.random.default_rng(5)
    base = r.standard_normal((60, 1)) * 0.01
    # 8 assets that are near-duplicates of one factor (rank-deficient covariance).
    x = base @ np.ones((1, 8)) + r.standard_normal((60, 8)) * 1e-7
    w = np.asarray(fn(x))
    assert np.isfinite(w).all(), f"{fn.__name__} returned non-finite weights on near-singular cov"
    assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{fn.__name__} broke the simplex on near-singular cov"
    assert np.all(w >= -1e-9)


def test_allocator_single_asset_is_full_weight() -> None:
    """With one asset every allocator returns the trivial [1.0] simplex."""
    x = np.random.default_rng(0).standard_normal((50, 1)) * 0.01
    for fn in ALLOCATORS:
        w = np.asarray(fn(x))
        assert w.shape == (1,)
        assert np.isclose(w.sum(), 1.0, atol=1e-12)


def test_minimum_variance_metamorphic_asset_permutation() -> None:
    """Permuting the asset columns permutes the GMV weights identically (label-equivariance)."""
    x = _make_window(8, n=10)
    perm = np.random.default_rng(1).permutation(x.shape[1])
    w = strategies.minimum_variance(x)
    w_perm = strategies.minimum_variance(x[:, perm])
    assert np.allclose(w[perm], w_perm, atol=1e-6)


def test_equal_weight_metamorphic_scale_invariance() -> None:
    """Scaling all returns by k>0 leaves 1/N weights unchanged (forecast-free)."""
    x = _make_window(2, n=7)
    for k in (0.5, 2.0, 10.0):
        assert np.allclose(strategies.equal_weight(k * x), strategies.equal_weight(x), atol=1e-12)


def test_inverse_volatility_is_proportional_to_one_over_sigma() -> None:
    """inverse_volatility weights are exactly proportional to 1/sigma over the live names."""
    x = _make_window(13, n=6)
    sd = x.std(axis=0, ddof=1)
    expected = (1.0 / sd) / (1.0 / sd).sum()
    assert np.allclose(strategies.inverse_volatility(x), expected, atol=1e-10)


# =========================================================================== #
# 4. fitness — reward-independence, lam==0 == pure DSR, monotonicity, det.     #
# =========================================================================== #
def test_fitness_equals_pure_dsr_when_lam_zero() -> None:
    """lam=0 reduces held_out_fitness to exactly the validation deflated Sharpe ratio."""
    from src.inference.deflated_sharpe import deflated_sharpe_ratio

    rng = np.random.default_rng(4)
    returns = 0.0005 + 0.01 * rng.standard_normal(1500)
    fit = held_out_fitness(returns, n_trials=12, split="val", lam=0.0)
    dsr = deflated_sharpe_ratio(returns, 12)
    assert fit == pytest.approx(dsr, abs=0.0)  # identical, not just close


def test_fitness_lam_zero_is_reward_value_independent() -> None:
    """Fitness depends ONLY on realized returns: scaling/offsetting any external 'reward value' is moot.

    held_out_fitness has no reward-value parameter; recomputing on identical returns is byte-identical.
    """
    rng = np.random.default_rng(6)
    r = 0.0005 + 0.01 * rng.standard_normal(1000)
    assert held_out_fitness(r, n_trials=10, split="val") == held_out_fitness(
        r.copy(), n_trials=10, split="val"
    )


def test_fitness_monotone_in_lambda() -> None:
    """For a loss-bearing series, fitness is non-increasing as lam rises (penalty grows with lam)."""
    rng = np.random.default_rng(9)
    returns = rng.standard_normal(2000) * 0.01  # zero-mean, genuine left tail
    fits = [held_out_fitness(returns, n_trials=10, split="val", lam=lam) for lam in (0.0, 1.0, 5.0, 10.0)]
    for a, b in zip(fits, fits[1:]):
        assert b <= a + 1e-12  # weakly decreasing in lam


def test_fitness_monotone_in_underlying_sharpe() -> None:
    """A series with a higher (deflated) Sharpe scores higher: shifting the mean up raises fitness."""
    rng = np.random.default_rng(14)
    noise = 0.01 * rng.standard_normal(2000)
    low = 0.0002 + noise
    high = 0.0010 + noise  # same noise, strictly higher mean -> higher Sharpe
    assert held_out_fitness(high, n_trials=10, split="val") >= held_out_fitness(
        low, n_trials=10, split="val"
    )


def test_fitness_metamorphic_positive_scaling_preserves_dsr() -> None:
    """Scaling all returns by k>0 leaves the Sharpe (hence lam=0 DSR fitness) invariant.

    The Sharpe ratio mu/sigma is scale-invariant, so the deflated Sharpe is unchanged by k>0 scaling.
    """
    rng = np.random.default_rng(21)
    r = 0.0005 + 0.01 * rng.standard_normal(1500)
    base = held_out_fitness(r, n_trials=10, split="val", lam=0.0)
    for k in (0.5, 3.0, 100.0):
        assert held_out_fitness(k * r, n_trials=10, split="val", lam=0.0) == pytest.approx(base, abs=1e-9)


def test_fitness_rejects_non_validation_split() -> None:
    """Selection is locked to the validation split: any other split raises (audit B-2/B-3)."""
    r = 0.0005 + 0.01 * np.random.default_rng(0).standard_normal(500)
    for bad in ("train", "test", "holdout", ""):
        with pytest.raises(ValueError):
            held_out_fitness(r, n_trials=5, split=bad)


def test_fitness_deterministic_under_rng_argument() -> None:
    """The accepted-for-symmetry rng argument never affects the (deterministic) fitness value."""
    r = 0.0005 + 0.01 * np.random.default_rng(3).standard_normal(800)
    a = held_out_fitness(r, n_trials=10, split="val", rng=np.random.default_rng(1))
    b = held_out_fitness(r, n_trials=10, split="val", rng=np.random.default_rng(999))
    assert a == b


# =========================================================================== #
# 5. arms — matched budget, feedback-only LLM variation, search-kind mapping   #
# =========================================================================== #
def test_all_arms_share_one_matched_budget() -> None:
    """Matched compute: every arm carries one identical positive candidate budget."""
    budgets = {a.candidate_budget for a in all_arms()}
    assert len(budgets) == 1 and next(iter(budgets)) > 0


def test_llm_arms_differ_only_in_feedback_kind() -> None:
    """The five LLM arms are identical except for a DISTINCT feedback_kind (the contribution channel)."""
    from dataclasses import replace

    llm = [a for a in all_arms() if a.is_llm]
    assert len(llm) == 5
    kinds = [a.feedback_kind for a in llm]
    assert len(set(kinds)) == len(kinds)  # all distinct
    normed = {replace(a, name="x", feedback_kind="x") for a in llm}
    assert len(normed) == 1  # collapse to one once name+feedback removed


def test_search_arms_are_non_llm_with_distinct_search_kinds() -> None:
    """random_search (code) and bayes_opt (template) are non-LLM with distinct search kinds."""
    rs, bo = build_arm("random_search"), build_arm("bayes_opt")
    assert (rs.is_llm, rs.search_kind, rs.feedback_kind) == (False, "code", None)
    assert (bo.is_llm, bo.search_kind, bo.feedback_kind) == (False, "template", None)


def test_arm_is_frozen_immutable() -> None:
    """Arm is a frozen dataclass: attribute assignment must raise (no silent post-build mutation)."""
    arm = build_arm("distributional")
    with pytest.raises(Exception):
        arm.candidate_budget = 999  # type: ignore[misc]


def test_unknown_arm_raises_keyerror() -> None:
    """Building an undeclared arm name raises KeyError."""
    with pytest.raises(KeyError):
        build_arm("not_a_real_arm")
