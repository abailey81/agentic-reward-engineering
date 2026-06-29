"""Tests for the MATERIALIZED executable BO reward source (Rank 2c; src/baselines/reward_family.py).

The Bayesian-optimization arm (H4b) optimizes the six coefficients of the parametric reward
family and, in memory, only ever holds a closure (``params_to_reward``). For the FROZEN BO winner
to round-trip through the sealed TEST leg on the IDENTICAL path as the LLM / random-search arms,
its archived ``reward_source`` must be EXECUTABLE ``def reward(...)`` source — not a
``# coeffs=[...]`` comment stub (which ``run_campaign._reinstantiate_frozen_winner`` ->
``validate_once`` cannot rehydrate, recording the arm ``winner_not_testable`` and breaking the H4
held-out comparison).

``params_to_source(coeffs, cvar_alpha, window)`` emits that executable source. These tests lock in
the load-bearing invariants:

  - the emitted source obeys the contract signature, passes ``ast_gate`` AND ``validate_once``
    (the SAME gate the LLM / random-search sources pass), and runs on a small fixture;
  - it reproduces the in-memory BO candidate (``params_to_reward``) ``(total, components)`` to
    ~1e-12 over a stateful replay (it is in fact bit-identical — coefficients are emitted via
    ``repr``), so swapping the archived stub for the materialized source changes NOTHING the agent
    optimizes;
  - a BO winner record carrying ``params_to_source(...)`` as its ``reward_source`` rehydrates
    through ``run_campaign._reinstantiate_frozen_winner`` WITHOUT raising — i.e. NO
    ``winner_not_testable``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.baselines.reward_family import (
    WEIGHT_KEYS,
    family_bounds,
    params_to_reward,
    params_to_source,
)
from src.reward.contract import validate_signature
from src.sandbox.executor import ast_gate, validate_once
from src.search.bayes_opt import bayes_opt_over_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_campaign  # noqa: E402


# Anonymized contract fixture (no tickers/dates) — matches the sandbox fixtures elsewhere.
_FIXTURE: tuple = (
    np.array([0.5, 0.5], dtype=float),
    np.array([0.01, -0.02], dtype=float),
    np.array([0.5, 0.5], dtype=float),
    0.0,
    {},
)

# A representative BO candidate inside the family box (non-trivial in every term).
_COEFFS = np.array([1.3, 0.4, 0.013, 0.07, 2.1, 0.9], dtype=float)


def _compile_source(src: str):
    """Compile materialized source in a numpy-only namespace (mirrors the sandbox)."""
    ns: dict = {"np": np}
    exec(compile(src, "<materialized_reward>", "exec"), ns)  # noqa: S102
    return ns["reward"]


def _replay(reward, path, *, w, prev, returns) -> list[tuple[float, dict]]:
    """Replay a return ``path`` through ``reward`` threading reward_state; collect (total, comps)."""
    info: dict = {}
    out: list[tuple[float, dict]] = []
    for r in path:
        total, comps, state = reward(w, returns, prev, float(r), info)
        info["reward_state"] = state
        out.append((float(total), dict(comps)))
    return out


# --------------------------------------------------------------------------- #
# (a) The materialized source is EXECUTABLE: contract + gate + validate_once   #
# --------------------------------------------------------------------------- #
def test_params_to_source_is_a_def_not_a_comment_stub() -> None:
    """The emitted source is real reward CODE (a ``def reward(``), not a ``# coeffs`` stub."""
    src = params_to_source(_COEFFS)
    assert src.lstrip().startswith("def reward(")
    assert "coeffs=" not in src  # NOT the old non-executable comment stub
    assert not src.lstrip().startswith("#")


def test_params_to_source_obeys_contract_signature() -> None:
    """The compiled reward has exactly (weights, returns, prev_weights, port_ret, info)."""
    reward = _compile_source(params_to_source(_COEFFS))
    assert validate_signature(reward)


def test_params_to_source_passes_ast_gate() -> None:
    """The materialized source passes the sandbox AST gate (numpy-only, no banned attrs)."""
    assert ast_gate(params_to_source(_COEFFS)) is True


def test_params_to_source_passes_validate_once_and_runs() -> None:
    """validate_once admits the source (gate + one-shot contract run on a fixture)."""
    reward = validate_once(params_to_source(_COEFFS), _FIXTURE)
    assert callable(reward)
    total, comps, state = reward(*_FIXTURE)
    assert isinstance(total, float) and np.isfinite(total)
    assert isinstance(comps, dict) and {"return", "turnover", "drawdown", "cvar", "sigma"} <= set(comps)
    assert state is not None  # stateful (hist, peak, cum)


def test_params_to_source_rejects_wrong_length() -> None:
    """Wrong-length coefficient vectors raise (mirrors params_to_reward)."""
    with pytest.raises(ValueError):
        params_to_source([1.0, 2.0])


# --------------------------------------------------------------------------- #
# (b) The materialized source REPRODUCES the in-memory BO candidate (~1e-12)   #
# --------------------------------------------------------------------------- #
def test_materialized_source_matches_closure_on_fixture() -> None:
    """Single-step (total, components) match params_to_reward to ~1e-12 on the fixture."""
    closure = params_to_reward(_COEFFS)
    materialized = _compile_source(params_to_source(_COEFFS))
    tc, cc, _ = closure(*_FIXTURE)
    tm, cm, _ = materialized(*_FIXTURE)
    assert tm == pytest.approx(tc, abs=1e-12)
    assert set(cm) == set(cc)
    for k in cc:
        assert cm[k] == pytest.approx(cc[k], abs=1e-12)


def test_materialized_source_matches_closure_over_stateful_replay() -> None:
    """Over a 60-step stateful replay the materialized source reproduces the closure to ~1e-12.

    This is the property that makes swapping the archived comment stub for ``params_to_source``
    output a no-op for the agent: the rehydrated reward IS the in-memory BO candidate.
    """
    rng = np.random.default_rng(0)
    path = rng.normal(0.0, 0.02, size=60)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    prev = np.array([0.34, 0.33, 0.33], dtype=float)
    returns = np.array([0.01, -0.02, 0.0], dtype=float)

    closure_out = _replay(params_to_reward(_COEFFS), path, w=w, prev=prev, returns=returns)
    materialized_out = _replay(
        _compile_source(params_to_source(_COEFFS)), path, w=w, prev=prev, returns=returns
    )
    assert len(closure_out) == len(materialized_out) == len(path)
    for (tc, cc), (tm, cm) in zip(closure_out, materialized_out):
        assert tm == pytest.approx(tc, abs=1e-12)
        for k in cc:
            assert cm[k] == pytest.approx(cc[k], abs=1e-12)


def test_materialized_source_matches_closure_with_nondefault_alpha_window() -> None:
    """The (alpha, window) discrete dims are baked in faithfully (non-default values)."""
    closure = params_to_reward(_COEFFS, cvar_alpha=0.1, window=5)
    materialized = _compile_source(params_to_source(_COEFFS, cvar_alpha=0.1, window=5))
    rng = np.random.default_rng(3)
    path = rng.normal(0.0, 0.03, size=12)  # > window, to exercise the rolling truncation
    w = np.full(4, 0.25)
    prev = np.full(4, 0.25)
    returns = np.array([0.01, -0.02, 0.0, 0.0])
    for (tc, cc), (tm, cm) in zip(
        _replay(closure, path, w=w, prev=prev, returns=returns),
        _replay(materialized, path, w=w, prev=prev, returns=returns),
    ):
        assert tm == pytest.approx(tc, abs=1e-12)
        for k in cc:
            assert cm[k] == pytest.approx(cc[k], abs=1e-12)


def test_materialized_source_matches_bo_winner_from_real_bo_run() -> None:
    """End-to-end: materialize a REAL BO winner's coeffs and reproduce its closure output.

    Runs the actual ``bayes_opt_over_template`` over the family box on a cheap deterministic
    template objective (no agent training), takes the winning coefficient vector, materializes
    it, and confirms the materialized reward reproduces the in-memory family closure for that
    winner to ~1e-12 — i.e. the archived source IS the BO candidate the search selected.
    """
    bounds = family_bounds()

    def template_eval(coeffs: np.ndarray) -> float:
        # Deterministic stand-in for the fixed-agent fitness (the search arm injects the real
        # evaluator in production); peaks inside the box so BO has a real optimum to find.
        reward = params_to_reward(coeffs)
        _t, comps, _s = reward(*_FIXTURE)
        return -float(comps["cvar"]) - 0.01 * float(np.sum(np.asarray(coeffs)))

    out = bayes_opt_over_template(
        template_eval, bounds, {"matched_budget": 12}, n_init=5, rng=np.random.default_rng(0)
    )
    best = np.asarray(out["best_coeffs"], dtype=float)
    assert best.shape == (len(WEIGHT_KEYS),)

    closure = params_to_reward(best)
    materialized = _compile_source(params_to_source(best))
    rng = np.random.default_rng(11)
    path = rng.normal(0.0, 0.02, size=30)
    w = np.array([0.2, 0.3, 0.5])
    prev = np.array([0.34, 0.33, 0.33])
    returns = np.array([0.01, -0.02, 0.0])
    for (tc, cc), (tm, cm) in zip(
        _replay(closure, path, w=w, prev=prev, returns=returns),
        _replay(materialized, path, w=w, prev=prev, returns=returns),
    ):
        assert tm == pytest.approx(tc, abs=1e-12)
        for k in cc:
            assert cm[k] == pytest.approx(cc[k], abs=1e-12)


# --------------------------------------------------------------------------- #
# (c) A BO winner with materialized source rehydrates in run_campaign         #
# --------------------------------------------------------------------------- #
def test_reinstantiate_frozen_bo_winner_does_not_raise() -> None:
    """run_campaign._reinstantiate_frozen_winner rehydrates a materialized BO source.

    The whole point of Rank 2c: a BO frozen winner whose ``reward_source`` is
    ``params_to_source(coeffs)`` round-trips through the SAME ``validate_once`` path the LLM /
    random-search winners use — so it is NO LONGER recorded ``winner_not_testable``.
    """
    src = params_to_source(_COEFFS)
    reward = run_campaign._reinstantiate_frozen_winner(src)
    assert callable(reward)
    # The rehydrated reward reproduces the in-memory BO candidate (the test leg sees the SAME fn).
    closure = params_to_reward(_COEFFS)
    tc, cc, _ = closure(*_FIXTURE)
    tr, cr, _ = reward(*_FIXTURE)
    assert tr == pytest.approx(tc, abs=1e-12)
    for k in cc:
        assert cr[k] == pytest.approx(cc[k], abs=1e-12)


def test_comment_stub_still_not_testable_but_materialized_source_is() -> None:
    """Contrast: the OLD ``# coeffs`` stub still raises; the materialized source does not.

    Confirms the fix is the source-MATERIALIZATION (not a loosening of the gate): a raw comment
    stub remains non-rehydratable (the existing ``winner_not_testable`` guard is intact), while
    ``params_to_source`` output rehydrates cleanly.
    """
    stub = f"# bayes_opt coeffs={list(_COEFFS)}\n"
    with pytest.raises(ValueError, match="executable reward|reward_source"):
        run_campaign._reinstantiate_frozen_winner(stub)
    # ... whereas the materialized source rehydrates.
    assert callable(run_campaign._reinstantiate_frozen_winner(params_to_source(_COEFFS)))
