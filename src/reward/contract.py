"""Reward contract — the single signature every reward obeys (F.5, audit A-3/B-4).

Reward code is produced by the LLM (untrusted), by hand-designed baselines, and by
the search baselines, and must all plug into the same fixed agent. This module
defines the ONE contract they share and a cheap signature validator.

The contract signature (used everywhere, EXACTLY):

    def reward(weights, returns, prev_weights, port_ret, info)
        -> tuple[float, dict[str, float], object]

Semantics (audit A-3, B-4):
  - returns (total, components, reward_state).
  - the agent optimizes `total` (a float);
  - `components` is a {str: float} dict, LOGGED only, never optimized;
  - `reward_state` is an opaque object round-tripped via `info["reward_state"]`
    so rewards can be STATEFUL (e.g. differential Sharpe / Moody-Saffell).
  - all array arguments are ANONYMIZED numpy arrays — no tickers, no dates ever
    reach a reward.
  - rewards may use numpy only; imports are allowlisted (see ALLOWED_IMPORTS).

Audit refs: A-3 (dict->scalar total + logged components), B-4 (stateful via info).

Tests (covered under test_sandbox.py and the contract checks):
  - validate_signature accepts a conforming reward and rejects a non-conforming one.
  - SAFE_DEFAULT is a finite float substituted by the sandbox on failure.
  - ALLOWED_IMPORTS is exactly the numpy allowlist.
"""

from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable

import numpy as np

# Imports an LLM-authored reward is permitted to use. The sandbox AST-gate
# (src/sandbox/executor.py) rejects anything outside this set.
ALLOWED_IMPORTS: set[str] = {"numpy", "np"}

# Finite scalar substituted by the sandbox when a reward raises / returns NaN/inf.
SAFE_DEFAULT: float = 0.0


@runtime_checkable
class RewardFn(Protocol):
    """Structural type for the reward contract (F.5).

    Any callable matching this signature is a valid reward. The agent optimizes
    the returned `total`; `components` are logged; `reward_state` is round-tripped
    via `info` for stateful rewards.
    """

    def __call__(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        prev_weights: np.ndarray,
        port_ret: float,
        info: dict,
    ) -> tuple[float, dict[str, float], object]:
        """Compute (total, components, reward_state) for one step. See module docs."""
        ...


def validate_signature(fn: object) -> bool:
    """Check that `fn` conforms to the reward contract signature.

    Algorithm:
        Inspect the callable's parameters and confirm they are exactly
        (weights, returns, prev_weights, port_ret, info) in order; this is a cheap
        static check used before a candidate is admitted to the sandbox.

    Args:
        fn: The candidate reward callable.

    Returns:
        True iff `fn` matches the contract signature.
    """
    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    expected = ("weights", "returns", "prev_weights", "port_ret", "info")
    params = list(sig.parameters.values())

    # Reject *args / **kwargs forms — the contract is exactly five positional params.
    names: list[str] = []
    for p in params:
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return False
        names.append(p.name)

    return tuple(names) == expected
