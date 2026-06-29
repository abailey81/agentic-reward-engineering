"""Tests for the keyless stub reward-designer (P4, Pass A machinery validation).

Verifies that every emitted candidate is gate-passing, contract-conforming, finite, and
that the sequence is deterministic by seed and varied across archetypes. These run the
real sandbox (validate_once spawns a killable child — C2/ADR-028), so they double as a
cross-platform check that validation works under multiprocessing on this host.
"""
from __future__ import annotations

import numpy as np

from src.llm.stub_designer import ARCHETYPES, StubDesignerTransport
from src.sandbox.executor import ast_gate, validate_once


def _fixture() -> tuple:
    return (
        np.array([0.4, 0.3, 0.3]),
        np.array([0.01, -0.02, 0.005]),
        np.array([0.5, 0.3, 0.2]),
        0.0,
        {},
    )


def test_every_emitted_candidate_gates_validates_and_is_finite() -> None:
    """The stub emits AST-gate-passing, contract-conforming, finite reward code."""
    stub = StubDesignerTransport(seed=0)
    fixture = _fixture()
    for _ in range(2 * len(ARCHETYPES)):
        src = stub("system", "user")
        assert ast_gate(src) is True
        fn = validate_once(src, fixture, timeout_s=3.0)  # raises SandboxError on any failure
        total, components, _state = fn(*fixture)
        assert np.isfinite(float(total))
        assert isinstance(components, dict)


def test_sequence_is_deterministic_by_seed() -> None:
    """The same seed reproduces the same candidate sequence (replay, audit C-2)."""
    a = [StubDesignerTransport(seed=7)("s", "u") for _ in range(len(ARCHETYPES))]
    b = [StubDesignerTransport(seed=7)("s", "u") for _ in range(len(ARCHETYPES))]
    assert a == b


def test_archetypes_vary_across_a_cycle() -> None:
    """One full cycle yields distinct reward bodies (spans the archetype library)."""
    stub = StubDesignerTransport(seed=1)
    out = [stub("s", "u") for _ in range(len(ARCHETYPES))]
    assert len(set(out)) == len(ARCHETYPES)


def test_stateful_archetype_threads_reward_state() -> None:
    """A stateful archetype accumulates across steps via info['reward_state']."""
    stub = StubDesignerTransport(seed=0)
    # Indices 0,1 are stateless; index 2 is rolling-variance (stateful).
    stub("s", "u")
    stub("s", "u")
    src = stub("s", "u")
    assert "reward_state" in src
    fixture = _fixture()
    fn = validate_once(src, fixture, timeout_s=3.0)
    _t1, _c1, state1 = fn(*fixture)
    assert state1 is not None  # carries accumulated history
    args2 = (fixture[0], fixture[1], fixture[2], 0.01, {"reward_state": state1})
    t2, _c2, state2 = fn(*args2)
    assert np.isfinite(float(t2))
    assert state2 is not None


def test_records_calls_like_a_transport() -> None:
    """The stub records (system, user) pairs so tests/forensics can inspect them."""
    stub = StubDesignerTransport(seed=0)
    stub("SYS", "USER")
    assert stub.calls == [("SYS", "USER")]
