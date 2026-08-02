"""Coverage-hardening tests for src/baselines, src/search, src/arms.

These are REAL deterministic (seeded) property tests that target the previously
uncovered guard/error/degenerate branches in the allocator canon, the arm
factory, the random-search and Bayesian-optimisation samplers, and the H4
reward family. Each asserts a genuine property (valid simplex, determinism,
fallback behaviour, raises on bad input) rather than merely executing a line.

New file (does not touch src/, conftest, or existing tests). Reuses the shared
``rng`` / ``seed`` conftest fixtures where useful; local fixtures otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.arms.factory import (
    Arm,
    all_arms,
    assert_fixed_agent_across_arms,
    build_arm,
    schema_arm_for,
)
from src.baselines.reward_family import params_to_reward, params_to_source
from src.baselines.strategies import (
    STRATEGY_CANON,
    cross_sectional_momentum,
    hrp,
    inverse_volatility,
    maximum_diversification,
    mean_variance,
    minimum_variance,
    risk_parity,
    spy_buy_and_hold,
)
from src.search.bayes_opt import (
    bayes_opt_over_template,
    random_search_over_template,
)
from src.search.random_search import (
    code_grid,
    random_search_over_code,
    sample_reward_source,
)
from src.utils.config import DotDict

# Allocators that operate over the live sub-panel and exclude zero-variance names.
_LIVE_ALLOCATORS = (
    mean_variance,
    risk_parity,
    hrp,
    minimum_variance,
    inverse_volatility,
    maximum_diversification,
    cross_sectional_momentum,
)

ATOL = 1e-9


def _assert_simplex(w: np.ndarray, n: int) -> None:
    """Assert ``w`` is a length-``n`` long-only simplex weight vector."""
    w = np.asarray(w, dtype=float)
    assert w.shape == (n,)
    assert np.all(w >= -ATOL), f"negative weight: {w}"
    assert np.all(np.isfinite(w)), f"non-finite weight: {w}"
    assert abs(float(w.sum()) - 1.0) <= 1e-8, f"weights not summing to 1: {w.sum()}"


# --------------------------------------------------------------------------- #
# Fixtures (local — NOT in conftest)                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def live_window(rng: np.random.Generator) -> np.ndarray:
    """A healthy (time, assets) window where every asset has real variation."""
    return 0.01 * rng.standard_normal((60, 5))


# --------------------------------------------------------------------------- #
# strategies.py — happy-path simplex over all allocators                        #
# --------------------------------------------------------------------------- #
def test_canon_all_return_simplex(live_window: np.ndarray) -> None:
    """Every allocator in the canon returns valid simplex weights on a live window."""
    n = live_window.shape[1]
    for name, fn in STRATEGY_CANON.items():
        w = fn(live_window)
        _assert_simplex(w, n)


def test_canon_deterministic(live_window: np.ndarray) -> None:
    """Allocators are deterministic: same input -> byte-identical weights."""
    for fn in STRATEGY_CANON.values():
        w1 = fn(live_window)
        w2 = fn(live_window)
        np.testing.assert_array_equal(w1, w2)


# --------------------------------------------------------------------------- #
# strategies.py — single-asset short-circuit (lines 46, 157, 184, 227, 296, ...) #
# --------------------------------------------------------------------------- #
def test_single_asset_returns_unit_weight() -> None:
    """A single-asset (1-column or 1-D) window short-circuits to weight [1.0]."""
    col = (0.01 * np.arange(1, 31, dtype=float)).reshape(-1, 1)
    flat = 0.01 * np.arange(1, 31, dtype=float)  # 1-D -> reshaped to (T, 1) by _as_window
    for fn in (mean_variance, risk_parity, hrp, minimum_variance,
               maximum_diversification, cross_sectional_momentum, spy_buy_and_hold):
        np.testing.assert_allclose(fn(col), [1.0], atol=ATOL)
        # 1-D path exercises _as_window's reshape (strategies.py line 46).
        np.testing.assert_allclose(fn(flat), [1.0], atol=ATOL)


def test_inverse_volatility_single_live_asset() -> None:
    """inverse_volatility on a 1-D live series returns the unit simplex."""
    series = 0.01 * np.array([1.0, -2.0, 0.5, 1.5, -0.3, 0.8])
    np.testing.assert_allclose(inverse_volatility(series), [1.0], atol=ATOL)


# --------------------------------------------------------------------------- #
# strategies.py — all-dead window: nl==0 -> uniform 1/N fallback                #
# (lines 113, 226-227, 391-392, 413-414, 433-434, 473-474)                      #
# --------------------------------------------------------------------------- #
def test_all_zero_variance_falls_back_to_equal_weight() -> None:
    """When NO asset has variation (all delisted/zero-fill) allocators fall back to 1/N."""
    n = 4
    dead = np.zeros((30, n), dtype=float)  # every column is zero-variance -> nl == 0
    for fn in _LIVE_ALLOCATORS:
        w = fn(dead)
        _assert_simplex(w, n)
        np.testing.assert_allclose(w, np.full(n, 1.0 / n), atol=ATOL)


# --------------------------------------------------------------------------- #
# strategies.py — exactly ONE live asset among dead names                       #
# (lines 114-117, 169-170, 228-231, 393-396, 435-438)                           #
# --------------------------------------------------------------------------- #
def test_single_live_asset_among_dead_gets_full_weight() -> None:
    """With exactly one varying asset (rest zero-fill), live allocators put all weight on it."""
    n = 4
    live_idx = 2
    arr = np.zeros((40, n), dtype=float)
    arr[:, live_idx] = 0.01 * np.array([1.0, -1.0] * 20)  # the only varying column
    for fn in (mean_variance, risk_parity, hrp, minimum_variance, maximum_diversification):
        w = fn(arr)
        _assert_simplex(w, n)
        expected = np.zeros(n)
        expected[live_idx] = 1.0
        np.testing.assert_allclose(w, expected, atol=ATOL)


# --------------------------------------------------------------------------- #
# strategies.py — _as_window guard (line 48): >2-D raises                       #
# --------------------------------------------------------------------------- #
def test_window_ndim_guard_raises() -> None:
    """A 3-D returns block is rejected by _as_window (via any allocator)."""
    bad = np.zeros((2, 3, 4), dtype=float)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        spy_buy_and_hold(bad)


# --------------------------------------------------------------------------- #
# strategies.py — mean_variance with NO positive mu -> GMV fallback (line 186)   #
# --------------------------------------------------------------------------- #
def test_mean_variance_no_upside_uses_gmv(rng: np.random.Generator) -> None:
    """With all-negative mean returns the tangency is undefined -> long-only GMV fallback (simplex)."""
    # Center each asset's returns at a negative mean so mu has no positive entry.
    arr = -0.02 + 0.005 * rng.standard_normal((80, 4))
    w = mean_variance(arr)
    _assert_simplex(w, 4)


def test_mean_variance_positive_signal_simplex(rng: np.random.Generator) -> None:
    """With a positive-mean asset the max-Sharpe branch (lines 187-200) yields a simplex."""
    arr = 0.005 + 0.01 * rng.standard_normal((80, 4))
    w = mean_variance(arr)
    _assert_simplex(w, 4)


# --------------------------------------------------------------------------- #
# strategies.py — cross_sectional_momentum tertile selection (lines 471-479)     #
# --------------------------------------------------------------------------- #
def test_cross_sectional_momentum_picks_top_tertile() -> None:
    """CSM equal-weights the top-tertile past performers; losers get zero."""
    n = 6
    arr = np.zeros((30, n), dtype=float)
    # Distinct per-asset constant drifts so the cumulative ranking is unambiguous.
    drifts = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    rng = np.random.default_rng(7)
    arr = drifts[None, :] + 0.0005 * rng.standard_normal((30, n))
    w = cross_sectional_momentum(arr)
    _assert_simplex(w, n)
    k = max(1, n // 3)
    assert int(np.count_nonzero(w)) == k
    # The top-k by drift are 4 and 5 (highest two of six).
    top = set(np.argsort(w)[-k:].tolist())
    assert top == {4, 5}


# --------------------------------------------------------------------------- #
# strategies.py — risk_parity equal-risk-contribution property                  #
# --------------------------------------------------------------------------- #
def test_risk_parity_equalises_risk_contributions(rng: np.random.Generator) -> None:
    """Risk-parity weights give ~equal marginal risk contributions w_i*(Σw)_i."""
    arr = 0.01 * rng.standard_normal((300, 4))
    w = risk_parity(arr)
    _assert_simplex(w, 4)
    cov = np.cov(arr, rowvar=False)
    rc = w * (cov @ w)  # per-asset risk contribution
    rc = rc / rc.sum()
    # Equal risk contribution => each ~ 1/n; allow loose tol (finite-sample cov).
    assert np.max(np.abs(rc - 0.25)) < 0.05, rc


# --------------------------------------------------------------------------- #
# arms/factory.py — _resolve_cfg DotDict branch (line 124) + happy build         #
# --------------------------------------------------------------------------- #
def test_build_arm_accepts_dotdict_cfg() -> None:
    """build_arm accepts a DotDict cfg directly (the isinstance(DotDict) branch, line 124)."""
    cfg = DotDict(
        {
            "matched_budget": 12,
            "arms": {
                "distributional": {"feedback": "full_tail_set"},
                "random_search": {"search": "code"},
            },
        }
    )
    a = build_arm("distributional", cfg)
    assert a.is_llm is True
    assert a.feedback_kind == "full_tail_set"
    assert a.candidate_budget == 12
    b = build_arm("random_search", cfg)
    assert b.is_llm is False
    assert b.search_kind == "code"


def test_build_arm_accepts_plain_dict_cfg() -> None:
    """build_arm coerces a plain dict cfg via DotDict (line 124)."""
    cfg = {"matched_budget": 6, "arms": {"scalar": {"feedback": "scalar_only"}}}
    a = build_arm("scalar", cfg)
    assert a.feedback_kind == "scalar_only"
    assert a.candidate_budget == 6


def test_build_arm_unknown_name_raises() -> None:
    """An unknown arm name raises KeyError."""
    cfg = DotDict({"matched_budget": 4, "arms": {"scalar": {"feedback": "scalar_only"}}})
    with pytest.raises(KeyError, match="unknown arm"):
        build_arm("nope", cfg)


def test_build_arm_unknown_feedback_raises() -> None:
    """An LLM arm with an unknown feedback kind raises ValueError (line 174)."""
    cfg = DotDict({"matched_budget": 4, "arms": {"weird": {"feedback": "telepathy"}}})
    with pytest.raises(ValueError, match="unknown feedback"):
        build_arm("weird", cfg)


def test_build_arm_unknown_search_kind_raises() -> None:
    """A search arm with an unknown search kind raises ValueError (line 191)."""
    cfg = DotDict({"matched_budget": 4, "arms": {"weird": {"search": "telekinesis"}}})
    with pytest.raises(ValueError, match="unknown search kind"):
        build_arm("weird", cfg)


def test_build_arm_neither_feedback_nor_search_raises() -> None:
    """An arm declaring neither feedback nor search raises ValueError (line 204)."""
    cfg = DotDict({"matched_budget": 4, "arms": {"weird": {"note": "empty"}}})
    with pytest.raises(ValueError, match="neither 'feedback' nor 'search'"):
        build_arm("weird", cfg)


def test_build_arm_search_llm_flag_passthrough() -> None:
    """A search arm with llm: true carries is_llm True (the spec.get('llm') branch)."""
    cfg = DotDict({"matched_budget": 4, "arms": {"odd": {"search": "template", "llm": True}}})
    a = build_arm("odd", cfg)
    assert a.is_llm is True
    assert a.search_kind == "template"


# --------------------------------------------------------------------------- #
# arms/factory.py — schema_arm_for guards (lines 245, 248)                        #
# --------------------------------------------------------------------------- #
def test_schema_arm_for_non_llm_raises() -> None:
    """schema_arm_for on a non-LLM arm raises ValueError (line 245)."""
    search_arm = Arm(name="random_search", feedback_kind=None, is_llm=False,
                     search_kind="code", candidate_budget=4)
    with pytest.raises(ValueError, match="not an LLM arm"):
        schema_arm_for(search_arm)


def test_schema_arm_for_unknown_feedback_kind_raises() -> None:
    """An LLM arm whose feedback_kind has no schema mapping raises ValueError (line 248)."""
    bad = Arm(name="x", feedback_kind="mystery", is_llm=True,
              search_kind=None, candidate_budget=4)
    with pytest.raises(ValueError, match="no schema arm"):
        schema_arm_for(bad)


def test_schema_arm_for_valid_llm_arm() -> None:
    """schema_arm_for maps a real LLM feedback_kind to its schema arm string."""
    good = Arm(name="distributional", feedback_kind="full_tail_set", is_llm=True,
               search_kind=None, candidate_budget=4)
    assert schema_arm_for(good) == "distributional"


# --------------------------------------------------------------------------- #
# arms/factory.py — assert_fixed_agent_across_arms guards (lines 313, 316, 320, 326)
# --------------------------------------------------------------------------- #
def test_assert_fixed_agent_empty_arms_raises() -> None:
    """No arms -> ValueError."""
    with pytest.raises(ValueError, match="no arms to verify"):
        assert_fixed_agent_across_arms([], {})


def test_assert_fixed_agent_non_distinct_seeds_raises() -> None:
    """Equal seeds cannot prove seed-only dependence -> ValueError."""
    arms = all_arms()
    with pytest.raises(ValueError, match="seeds must be distinct"):
        assert_fixed_agent_across_arms(arms, {}, seeds=(3, 3))


def test_assert_fixed_agent_differing_budget_raises() -> None:
    """Arms with differing candidate budgets -> matched-compute AssertionError (line ~306)."""
    a = Arm(name="a", feedback_kind="full_tail_set", is_llm=True, search_kind=None, candidate_budget=4)
    b = Arm(name="b", feedback_kind="scalar_only", is_llm=True, search_kind=None, candidate_budget=8)
    with pytest.raises(AssertionError, match="matched compute"):
        assert_fixed_agent_across_arms([a, b], {})


def test_assert_fixed_agent_duplicate_feedback_kind_raises() -> None:
    """Two LLM arms sharing a feedback_kind -> feedback-isolation AssertionError (line 326)."""
    a = Arm(name="a", feedback_kind="full_tail_set", is_llm=True, search_kind=None, candidate_budget=4)
    b = Arm(name="b", feedback_kind="full_tail_set", is_llm=True, search_kind=None, candidate_budget=4)
    with pytest.raises(AssertionError, match="DISTINCT feedback_kind"):
        assert_fixed_agent_across_arms([a, b], {})


def _two_distinct_llm_arms() -> list[Arm]:
    return [
        Arm(name="a", feedback_kind="full_tail_set", is_llm=True, search_kind=None, candidate_budget=4),
        Arm(name="b", feedback_kind="scalar_only", is_llm=True, search_kind=None, candidate_budget=4),
    ]


def test_assert_fixed_agent_train_steps_differ_raises(monkeypatch) -> None:
    """If resolve_agent_kwargs returns differing train-steps across seeds -> AssertionError (line 313)."""
    import src.agents.trainer as trainer

    def fake(cfg, seed):  # noqa: ANN001
        return {"policy": "MlpPolicy", "seed": seed}, 100 + int(seed)  # steps vary by seed

    monkeypatch.setattr(trainer, "resolve_agent_kwargs", fake)
    with pytest.raises(AssertionError, match="train-step budget differs"):
        assert_fixed_agent_across_arms(_two_distinct_llm_arms(), {}, seeds=(0, 1))


def test_assert_fixed_agent_kwargs_vary_beyond_seed_raises(monkeypatch) -> None:
    """If a non-seed kwarg differs across seeds -> AssertionError (line 316)."""
    import src.agents.trainer as trainer

    def fake(cfg, seed):  # noqa: ANN001
        return {"policy": "MlpPolicy", "seed": seed, "learning_rate": 1e-3 * (1 + seed)}, 100

    monkeypatch.setattr(trainer, "resolve_agent_kwargs", fake)
    with pytest.raises(AssertionError, match="vary by more than the seed"):
        assert_fixed_agent_across_arms(_two_distinct_llm_arms(), {}, seeds=(0, 1))


def test_assert_fixed_agent_non_mlp_policy_raises(monkeypatch) -> None:
    """If the resolved policy is not MlpPolicy -> AssertionError (line 320)."""
    import src.agents.trainer as trainer

    def fake(cfg, seed):  # noqa: ANN001
        return {"policy": "CnnPolicy", "seed": seed}, 100

    monkeypatch.setattr(trainer, "resolve_agent_kwargs", fake)
    with pytest.raises(AssertionError, match="fixed SB3-SAC MlpPolicy"):
        assert_fixed_agent_across_arms(_two_distinct_llm_arms(), {}, seeds=(0, 1))


def test_assert_fixed_agent_happy_path_returns_shared_kwargs() -> None:
    """On the real frozen roster the invariant holds and returns shared kwargs/budget/steps."""
    arms = all_arms()
    out = assert_fixed_agent_across_arms(arms, {"train_steps": 100}, seeds=(0, 1))
    assert out["agent_kwargs"]["policy"] == "MlpPolicy"
    assert "seed" not in out["agent_kwargs"]
    assert out["train_steps"] == 100
    assert out["candidate_budget"] == arms[0].candidate_budget


# --------------------------------------------------------------------------- #
# search/random_search.py — budget guards (lines 182, 187-191, 239, 241)         #
# --------------------------------------------------------------------------- #
def test_random_search_none_cfg_raises() -> None:
    """A None cfg has no budget -> ValueError (line 182)."""
    with pytest.raises(ValueError, match="requires a cfg with a budget"):
        random_search_over_code(None, lambda r: 0.0, None)


def test_random_search_dict_without_budget_key_raises() -> None:
    """A mapping cfg missing every budget key -> KeyError (lines 187-191)."""
    with pytest.raises(KeyError, match="matched_budget"):
        random_search_over_code(None, lambda r: 0.0, {"unrelated": 1})


def test_random_search_nonpositive_budget_raises() -> None:
    """Budget <= 0 -> ValueError (line 239)."""
    with pytest.raises(ValueError, match="budget must be positive"):
        random_search_over_code(None, lambda r: 0.0, {"matched_budget": 0})


def test_random_search_int_budget_and_default_rng() -> None:
    """An int cfg is the budget directly; rng=None creates a default generator (line 241)."""
    out = random_search_over_code(None, lambda r: 1.0, 3, rng=None)
    assert out["n_evaluated"] == 3
    assert out["budget"] == 3
    assert out["best_source"] is not None


def test_random_search_deterministic_and_finds_best(seed: int) -> None:
    """Seeded random search is reproducible and returns the candidate with the max score."""
    # Deterministic fitness = number of 'np.log1p' occurrences (varies with sampled weights),
    # actually score on length so it is a real ordering over distinct sources.
    def fitness(reward) -> float:
        return float(reward(  # exercise the validated callable end-to-end
            np.array([0.5, 0.5]), np.array([0.01, -0.01]),
            np.array([0.4, 0.6]), 0.002, {})[0])

    cfg = DotDict({"matched_budget": 8})
    out1 = random_search_over_code(None, fitness, cfg, rng=np.random.default_rng(seed))
    out2 = random_search_over_code(None, fitness, cfg, rng=np.random.default_rng(seed))
    assert out1["best_source"] == out2["best_source"]
    assert out1["best_score"] == out2["best_score"]
    # best_score is the max over the archive.
    assert out1["best_score"] == max(rec["score"] for rec in out1["archive"])


def test_sample_reward_source_gate_clean_and_deterministic(seed: int) -> None:
    """sample_reward_source is deterministic given a seeded rng and renders runnable family code."""
    s1 = sample_reward_source(np.random.default_rng(seed))
    s2 = sample_reward_source(np.random.default_rng(seed))
    assert s1 == s2
    assert s1.startswith("def reward(")


def test_code_grid_fractions_include_zero_and_one() -> None:
    """The grid fractions span 0 (term off) to 1 (full high bound)."""
    g = code_grid()
    assert 0.0 in g
    assert 1.0 in g
    assert all(0.0 <= f <= 1.0 for f in g)


def test_sample_reward_source_all_zero_revives_return_term() -> None:
    """When the grid draws all-zero weights the sampler revives w_return (lines 156-158).

    A grid of only {0.0} (cfg with zero-width? no — use a stub rng whose choice picks index 0,
    i.e. the 0.0 fraction for every weight) triggers the all-zero revive branch.
    """
    class _ZeroRng:
        # Mimics np.random.Generator.choice by always returning the first grid entry (0.0).
        def choice(self, g):  # noqa: ANN001
            return g[0]

    src = sample_reward_source(_ZeroRng())  # type: ignore[arg-type]
    assert src.startswith("def reward(")
    # The first weight (w_return) must have been bumped to its grid max (non-zero).
    assert "0.0, 0.0, 0.0, 0.0, 0.0, 0.0" not in src


# --------------------------------------------------------------------------- #
# search/bayes_opt.py — budget guards (lines 68, 73, 84) + samplers               #
# --------------------------------------------------------------------------- #
def test_bayes_opt_none_cfg_raises() -> None:
    """None cfg -> ValueError (line 68)."""
    with pytest.raises(ValueError, match="requires a cfg with a budget"):
        bayes_opt_over_template(lambda x: 0.0, [(0.0, 1.0)], None)


def test_bayes_opt_dict_without_budget_raises() -> None:
    """Mapping cfg missing every budget key -> KeyError (line 73)."""
    with pytest.raises(KeyError, match="matched_budget"):
        bayes_opt_over_template(lambda x: 0.0, [(0.0, 1.0)], {"foo": 1})


def test_bayes_opt_bad_bounds_shape_raises() -> None:
    """Bounds not (d, 2) -> ValueError (line 84)."""
    with pytest.raises(ValueError, match=r"\(d, 2\)"):
        bayes_opt_over_template(lambda x: 0.0, [[0.0, 1.0, 2.0]], 4)


def test_bayes_opt_inverted_bounds_raises() -> None:
    """low >= high -> ValueError (line 89)."""
    with pytest.raises(ValueError, match="low < high"):
        bayes_opt_over_template(lambda x: 0.0, [(1.0, 1.0)], 4)


def test_bayes_opt_nonpositive_budget_raises() -> None:
    """Budget <= 0 -> ValueError (line 235)."""
    with pytest.raises(ValueError, match="budget must be positive"):
        bayes_opt_over_template(lambda x: 0.0, [(0.0, 1.0)], {"matched_budget": -1})


def test_bayes_opt_int_budget_default_rng_and_n_init_clamp() -> None:
    """Int budget + rng=None default generator; n_init is clamped to the budget (line 239)."""
    calls = {"n": 0}

    def obj(x: np.ndarray) -> float:
        calls["n"] += 1
        return float(-((x[0] - 0.3) ** 2))

    out = bayes_opt_over_template(obj, [(0.0, 1.0)], 4, n_init=99, rng=None)
    assert out["n_evaluated"] == 4  # total eval == budget
    assert out["n_init"] == 4       # clamped down to the budget
    assert calls["n"] == 4
    assert 0.0 <= float(out["best_coeffs"][0]) <= 1.0


def test_bayes_opt_concave_objective_recovers_optimum(seed: int) -> None:
    """On a concave objective BO gets close to the known optimum within the budget."""
    def obj(x: np.ndarray) -> float:
        return float(-((x[0] - 0.3) ** 2) - ((x[1] + 0.1) ** 2))

    out = bayes_opt_over_template(
        obj, [(-1.0, 1.0), (-1.0, 1.0)], {"matched_budget": 25},
        n_init=5, rng=np.random.default_rng(seed),
    )
    bx, by = out["best_coeffs"]
    # Optimum at (0.3, -0.1); a 25-eval GP-EI run should land in a loose ball.
    assert abs(float(bx) - 0.3) < 0.25
    assert abs(float(by) + 0.1) < 0.25


def test_random_search_over_template_guards_and_run(seed: int) -> None:
    """random_search_over_template: budget guard (lines 168/170) + deterministic best."""
    with pytest.raises(ValueError, match="budget must be positive"):
        random_search_over_template(lambda x: 0.0, [(0.0, 1.0)], {"matched_budget": 0})

    def obj(x: np.ndarray) -> float:
        return float(-((x[0] - 0.5) ** 2))

    out1 = random_search_over_template(obj, [(0.0, 1.0)], 10, rng=np.random.default_rng(seed))
    out2 = random_search_over_template(obj, [(0.0, 1.0)], 10, rng=np.random.default_rng(seed))
    assert out1["n_evaluated"] == 10
    np.testing.assert_array_equal(out1["best_coeffs"], out2["best_coeffs"])
    assert out1["best_score"] == max(h["score"] for h in out1["history"])


def test_random_search_over_template_int_default_rng() -> None:
    """rng=None path of random_search_over_template (line 170) runs and matches budget."""
    out = random_search_over_template(lambda x: float(x[0]), [(0.0, 1.0)], 5, rng=None)
    assert out["n_evaluated"] == 5


# --------------------------------------------------------------------------- #
# baselines/reward_family.py — params_to_source / params_to_reward parity        #
# + the wrong-length ValueError (line 184 in params_to_source)                   #
# --------------------------------------------------------------------------- #
def test_params_to_source_wrong_length_raises() -> None:
    """params_to_source rejects a coeff vector of the wrong length (line 184)."""
    with pytest.raises(ValueError, match="expects 6 weights"):
        params_to_source([0.1, 0.2, 0.3])


def test_params_to_reward_wrong_length_raises() -> None:
    """params_to_reward rejects a coeff vector of the wrong length (mirror guard)."""
    with pytest.raises(ValueError, match="expects 6 weights"):
        params_to_reward([0.1, 0.2])


def test_source_and_closure_parity() -> None:
    """The materialised source reward and the closure produce identical output sequences.

    Guarantees the H4 winner round-trips through the sealed test leg byte-identically.
    """
    coeffs = [1.3, 0.4, 0.01, 0.05, 2.0, 0.7]
    closure = params_to_reward(coeffs, cvar_alpha=0.05, window=20)
    src = params_to_source(coeffs, cvar_alpha=0.05, window=20)
    ns: dict = {"np": np}
    exec(src, ns)  # noqa: S102 - trusted, materialised-by-us family source
    materialised = ns["reward"]

    rng = np.random.default_rng(2024)
    state_c = None
    state_m = None
    for _ in range(30):
        w = np.array([0.5, 0.3, 0.2])
        pw = np.array([0.4, 0.4, 0.2])
        pr = float(0.01 * rng.standard_normal())
        tc, cc, state_c = closure(w, None, pw, pr, {"reward_state": state_c})
        tm, cm, state_m = materialised(w, None, pw, pr, {"reward_state": state_m})
        assert tc == tm, (tc, tm)
        assert cc == cm
