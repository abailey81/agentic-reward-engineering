"""Leakage invariances for the gold cash-row feature builder (data_pipeline/src/features.py).

The module docstring asserts these two invariances are unit-tested here; this file makes that true
(supervisor leakage/rigor audit 2026-06-20 found the cited test absent). The VIX ``shift(1)`` and the
``rolling_vol_shifted`` lag are the pipeline-side transforms that BAKE the no-look-ahead guarantee into
the frozen gold parquet, so they must be checked, not assumed.

Run isolated (the repo-root conftest binds the ENGINE's ``src``; here ``src`` is data_pipeline's):
    PYTHONPATH=data_pipeline python -m pytest data_pipeline/tests/test_features.py --noconftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``src`` resolve to data_pipeline's package (features.py uses ``from .config import get``). Under
# pytest the ENGINE's ``src`` (repo root) is already bound, so evict it and put data_pipeline first.
_DP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DP))
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src.config import get  # noqa: E402
from src.features import build_cash_features, rolling_vol_shifted  # noqa: E402

_VS, _VL = 2, 4  # small explicit vol windows so a short fixture warms up


def _returns(t: int = 10, n: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, n)) * 0.01


# --------------------------------------------------------------------------- #
# rolling_vol_shifted: row t is a function of returns STRICTLY before t        #
# --------------------------------------------------------------------------- #
def test_rolling_vol_shifted_is_strictly_past_and_ddof1() -> None:
    r = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    v = rolling_vol_shifted(r, 2)
    assert np.isnan(v[0]) and np.isnan(v[1])  # need 2 prior obs THEN a shift -> first value at row 2
    assert v[2] == pytest.approx(np.std(r[0:2], ddof=1))  # std of the two obs ending at t-1
    assert v[3] == pytest.approx(np.std(r[1:3], ddof=1))
    assert v[4] == pytest.approx(np.std(r[2:4], ddof=1))


def test_rolling_vol_window_must_be_at_least_2() -> None:
    with pytest.raises(ValueError):
        rolling_vol_shifted(np.zeros(5), 1)


# --------------------------------------------------------------------------- #
# VIX column uses the close of t-1 (the leakage-critical shift)               #
# --------------------------------------------------------------------------- #
def test_vix_feature_uses_t_minus_1_close() -> None:
    t = 8
    vix_scale = float(get("environment.state.vix_scale"))
    ar = _returns(t)
    vix = np.arange(10.0, 10.0 + t)  # distinct levels so the off-by-one is visible
    feats = build_cash_features(ar, vix, vol_short=_VS, vol_long=_VL)
    vix_col = feats[:, 2]
    assert np.isnan(vix_col[0])  # row 0 has no t-1 close
    for row in range(1, t):
        assert vix_col[row] == pytest.approx(vix[row - 1] / vix_scale)  # t reads the t-1 close, NEVER t


# --------------------------------------------------------------------------- #
# truncation invariance: features[:m] on the full series == on the truncation #
# --------------------------------------------------------------------------- #
def test_truncation_invariance() -> None:
    t, m = 12, 7
    ar, vix = _returns(t), np.arange(15.0, 15.0 + t)
    full = build_cash_features(ar, vix, vol_short=_VS, vol_long=_VL)
    trunc = build_cash_features(ar[:m], vix[:m], vol_short=_VS, vol_long=_VL)
    np.testing.assert_allclose(full[:m], trunc, equal_nan=True)


# --------------------------------------------------------------------------- #
# future-perturbation invariance: changing rows >= k leaves features <= k     #
# --------------------------------------------------------------------------- #
def test_future_perturbation_invariance() -> None:
    t, k = 12, 6
    ar, vix = _returns(t), np.arange(15.0, 15.0 + t)
    base = build_cash_features(ar, vix, vol_short=_VS, vol_long=_VL)
    ar2, vix2 = ar.copy(), vix.copy()
    ar2[k:] += 5.0  # large perturbation of the FUTURE
    vix2[k:] += 100.0
    pert = build_cash_features(ar2, vix2, vol_short=_VS, vol_long=_VL)
    # every feature at row t depends only on inputs through t-1, so rows <= k are untouched by inputs >= k.
    np.testing.assert_allclose(base[: k + 1], pert[: k + 1], equal_nan=True)
    assert not np.allclose(base[k + 1 :], pert[k + 1 :], equal_nan=True)  # the future DID move
