"""Edge/degrade-path tests for the multiple-testing corrections (src/inference/multiple_testing.py).

Covers the empty-family and malformed-input guards not exercised by the main behaviour suite
(test_inference.py): BH/RW on an empty p-value family, and the Romano-Wolf bootstrap-shape check.
Deterministic (no RNG). Asserts real return-contract behaviour, not smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.multiple_testing import benjamini_hochberg, romano_wolf


def test_benjamini_hochberg_empty_family_returns_empty() -> None:
    # m == 0 -> empty boolean array, no rejections (line 45).
    out = benjamini_hochberg(np.array([]), q=0.1)
    assert out.dtype == bool
    assert out.size == 0


def test_romano_wolf_empty_family_returns_empty() -> None:
    # m == 0 -> empty boolean array before any shape check (line 94).
    out = romano_wolf(np.array([]), np.empty((5, 0)), alpha=0.05)
    assert out.dtype == bool
    assert out.size == 0


def test_romano_wolf_rejects_malformed_bootstrap_shape() -> None:
    # boot_stats not (n_boot, n_hypotheses) -> ValueError (line 96).
    stats = np.array([3.0, 1.5, 0.2])
    with pytest.raises(ValueError, match="shape"):
        romano_wolf(stats, np.ones((10, 2)), alpha=0.05)  # 2 != 3 hypotheses
    with pytest.raises(ValueError, match="shape"):
        romano_wolf(stats, np.ones(10), alpha=0.05)       # 1-D, not 2-D


def test_romano_wolf_stops_at_first_non_rejection() -> None:
    # One dominant statistic well above the bootstrap max quantile, the rest buried in the null:
    # stepdown rejects only the leader and stops (the else/break path, lines 110-111).
    rng = np.random.default_rng(20260701)
    n_boot = 4000
    stats = np.array([8.0, 0.10, 0.05])           # only the first is a real signal
    boot = rng.standard_normal((n_boot, 3))        # centred null draws
    rej = romano_wolf(stats, boot, alpha=0.05)
    assert rej.tolist() == [True, False, False]
