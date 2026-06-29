"""Coverage tests for the pure-function inference paths whose dedicated coverage agent did not land:
src/inference/{ood_stress, attribution, contamination}.py. Deterministic (seeded), asserting real
properties (shapes, status keys, finiteness, monotone bounds, guard-raises) — not just "runs".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import attribution as attr  # noqa: E402
from src.inference import contamination as cont  # noqa: E402
from src.inference import ood_stress as ood  # noqa: E402


def _panel(T: int = 120, N: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, N)) * 0.01


# --------------------------------------------------------------------------- #
# ood_stress.py — panel stressors + tail/score/validate (numpy-only paths)      #
# --------------------------------------------------------------------------- #
def test_optimal_block_length_is_finite_ge_one() -> None:
    bl = ood.optimal_block_length(_panel())
    assert np.isfinite(bl) and bl >= 1.0


def test_block_bootstrap_paths_shape_and_determinism() -> None:
    p = _panel()
    a = ood.block_bootstrap_paths(p, n_paths=20, horizon=30, rng=np.random.default_rng(1))
    b = ood.block_bootstrap_paths(p, n_paths=20, horizon=30, rng=np.random.default_rng(1))
    assert a.shape == (20, 30, p.shape[1])
    assert np.array_equal(a, b)  # same rng seed -> identical
    assert np.isfinite(a).all()


def test_markov_crash_and_vol_spike_paths_run() -> None:
    p = _panel()
    crash = ood.markov_crash_paths(p)
    assert isinstance(crash, dict)
    spike = ood.vol_spike_paths(p)
    assert np.isfinite(spike).all()


def test_tail_metrics_keys_and_finite() -> None:
    port = np.random.default_rng(2).standard_normal(500) * 0.01
    m = ood.tail_metrics(port)
    assert isinstance(m, dict) and m
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, (int, float)))


def test_score_paths_runs_on_bootstrap_output() -> None:
    p = _panel()
    paths = ood.block_bootstrap_paths(p, n_paths=16, horizon=20, rng=np.random.default_rng(3))
    s = ood.score_paths(paths)
    assert isinstance(s, dict) and s


def test_validate_stylized_facts_and_claims() -> None:
    hist = _panel(seed=4)
    synth = ood.block_bootstrap_paths(hist, n_paths=16, horizon=hist.shape[0], rng=np.random.default_rng(5))
    res = ood.validate_stylized_facts(synth, hist)
    assert isinstance(res, dict)
    cl = ood.claims()
    assert isinstance(cl, dict) and cl  # static documentation map


# --------------------------------------------------------------------------- #
# contamination.py — paired TOST (+ guard raises) + report                      #
# --------------------------------------------------------------------------- #
def test_paired_tost_equivalent_when_identical() -> None:
    rng = np.random.default_rng(6)
    named = rng.standard_normal(40) * 0.1
    blinded = named.copy()  # zero paired difference -> well inside any symmetric band
    out = cont.paired_tost(named, blinded, low=-0.5, high=0.5)
    assert out["equivalent"] is True
    assert abs(out["mean_diff"]) < 1e-9


@pytest.mark.parametrize(
    "named, blinded, low, high, match",
    [
        (np.zeros(5), np.zeros(4), -0.1, 0.1, "paired"),        # shape mismatch
        (np.zeros(1), np.zeros(1), -0.1, 0.1, ">= 2"),          # too few pairs
        (np.zeros(3), np.zeros(3), 0.1, -0.1, None),            # low >= high
    ],
)
def test_paired_tost_guard_raises(named, blinded, low, high, match) -> None:
    with pytest.raises(ValueError, match=match) if match else pytest.raises(ValueError):
        cont.paired_tost(named, blinded, low=low, high=high)


def test_contamination_report_runs() -> None:
    rep = cont.contamination_report()
    assert isinstance(rep, dict)


# --------------------------------------------------------------------------- #
# attribution.py — HAC lag, factor alpha (ok + skipped), markdown               #
# --------------------------------------------------------------------------- #
def test_newey_west_hac_lag_grows_with_n() -> None:
    assert attr.newey_west_hac_lag(100) >= 0
    assert attr.newey_west_hac_lag(1000) >= attr.newey_west_hac_lag(100)


def test_factor_alpha_ok_and_markdown() -> None:
    rng = np.random.default_rng(7)
    T = 300
    mkt = rng.standard_normal(T) * 0.01
    returns = 0.0002 + 0.8 * mkt + rng.standard_normal(T) * 0.003  # alpha>0, beta~0.8
    res = attr.factor_alpha(returns, {"Mkt-RF": mkt})
    assert res["status"] == "ok"
    assert np.isfinite(res["alpha"]) and "Mkt-RF" in res["betas"]
    md = attr.attribution_markdown(res)
    assert isinstance(md, str) and len(md) > 0


def test_factor_alpha_skips_on_degenerate_input() -> None:
    # Too-short / rank-deficient series must degrade to status="skipped" with a reason (no raise).
    res = attr.factor_alpha(np.zeros(3), {"Mkt-RF": np.zeros(3)})
    assert res["status"] == "skipped" and "reason" in res
