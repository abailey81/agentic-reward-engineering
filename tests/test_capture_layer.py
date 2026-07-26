"""Tests for the report-only, determinism-safe run-diagnostics capture layer (2026-07-26).

Covers M1/M1b/M3 (test-path exposure / allocation / reward-component summaries) and the invariant that
matters most: :func:`rollout_port_diagnostics` returns net/gross/turnover **byte-identical** to
:func:`rollout_port_series` (the diagnostics rollout only ADDS passive per-step reads — it must not perturb
the trajectory).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.runner import rollout_port_diagnostics, rollout_port_series
from src.inference.exposure import alloc_snapshots, exposure_series
from src.orchestration.test_leg import _summarize_components, build_test_record


# --------------------------------------------------------------------------- #
# A minimal fake env + policy (no torch, no panel) exercising the rollout      #
# --------------------------------------------------------------------------- #
class _FakeEnv:
    """Emits a deterministic 3-step episode with the info keys the rollouts read."""

    def __init__(self, n_assets: int = 3) -> None:
        self.n = n_assets
        self._t = 0
        # a fixed per-step (gross, turnover, weights) script; port_ret = gross - 0.5*turnover
        self._script = [
            (0.010, 0.20, np.array([0.5, 0.5, 0.0])),
            (-0.020, 0.10, np.array([1.0, 0.0, 0.0])),
            (0.030, 0.05, np.array([0.34, 0.33, 0.33])),
        ]

    def reset(self):
        self._t = 0
        return np.zeros(self.n), {}

    def step(self, action):
        gross, turnover, w = self._script[self._t]
        self._t += 1
        port_ret = gross - 0.5 * turnover
        info = {
            "port_ret": port_ret, "gross": gross, "turnover": turnover,
            "weights": w.copy(), "components": {"risk": -abs(gross), "ret": gross},
        }
        terminated = self._t >= len(self._script)
        return np.zeros(self.n), 0.0, terminated, False, info


class _FakePolicy:
    def predict(self, obs, deterministic: bool = True):
        return np.zeros(3), None


def test_diagnostics_rollout_matches_series_rollout_byte_for_byte():
    """net/gross/turnover from the diagnostics rollout == the series rollout (no perturbation)."""
    series = rollout_port_series(_FakeEnv(), _FakePolicy())
    diag = rollout_port_diagnostics(_FakeEnv(), _FakePolicy())
    for k in ("net", "gross", "turnover"):
        np.testing.assert_array_equal(diag[k], series[k])
    # and it additionally captures the per-step weights + components
    assert diag["weights"].shape == (3, 3)
    assert diag["components"] is not None and len(diag["components"]) == 3


def test_diagnostics_rollout_weights_absent_is_none():
    """An env that emits no per-step weights yields weights=None (never raises)."""

    class _NoWeights(_FakeEnv):
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            info.pop("weights")
            info.pop("components")
            return obs, r, term, trunc, info

    diag = rollout_port_diagnostics(_NoWeights(), _FakePolicy())
    assert diag["weights"] is None and diag["components"] is None
    assert diag["net"].shape == (3,)


def test_exposure_series_values():
    w = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.34, 0.33, 0.33]])
    ex = exposure_series(w)
    assert ex["hhi"][1] == 1.0                      # fully concentrated
    assert ex["eff_n"][0] == pytest.approx(2.0)     # two equal positions
    assert ex["max_weight"][1] == 1.0
    assert all(t == pytest.approx(1.0) for t in ex["top5"])  # simplex sums to 1 within top-5


def test_exposure_series_degenerate_is_empty():
    assert exposure_series(np.array([]))["hhi"] == []
    assert exposure_series(np.zeros((0, 3)))["eff_n"] == []


def test_alloc_snapshots_shape_and_residual():
    w = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.34, 0.33, 0.33]])
    al = alloc_snapshots(w, top_k=2, n_snapshots=3)
    assert len(al["asset_idx"]) == 2
    assert al["asset_idx"] == sorted(al["asset_idx"])          # ascending for a readable heatmap
    assert len(al["weights"]) == len(al["steps"]) == len(al["other"])
    for row, other in zip(al["weights"], al["other"]):
        assert other == pytest.approx(1.0 - sum(row))          # residual mass is consistent


def test_alloc_snapshots_degenerate_is_empty():
    assert alloc_snapshots(np.array([]))["asset_idx"] == []


def test_summarize_components_means_and_guards():
    assert _summarize_components([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]) == {"a": 2.0, "b": 3.0}
    assert _summarize_components([1, 2, 3]) is None            # not dict-shaped
    assert _summarize_components("nonsense") is None
    assert _summarize_components([{"a": float("nan")}]) is None  # non-finite dropped -> empty -> None


def test_build_test_record_archives_diagnostics_and_is_back_compatible():
    winner = {"candidate_id": "c", "generation": 0, "reward_source": "",
              "feedback_block": "", "metrics": {"val_fitness": 0.1}}
    ex = {"hhi": [0.5], "eff_n": [2.0], "max_weight": [0.5], "top5": [1.0]}
    al = {"asset_idx": [0, 1], "steps": [0], "weights": [[0.5, 0.5]], "other": [0.0]}
    rec = build_test_record(
        winner=winner, arm="scalar", seed=7, reward_hash="h", env_fp="fp",
        test_returns=[0.01, -0.02, 0.03], test_exposure=ex, test_alloc=al, test_components={"risk": -0.02},
    )
    assert rec["metrics"]["test_exposure"] == ex
    assert rec["metrics"]["test_alloc"] == al
    assert rec["metrics"]["test_components"] == {"risk": -0.02}
    # omitting them keeps the record back-compatible (no keys added)
    rec2 = build_test_record(winner=winner, arm="scalar", seed=7, reward_hash="h", env_fp="fp",
                             test_returns=[0.01])
    assert "test_exposure" not in rec2["metrics"]
    assert "test_alloc" not in rec2["metrics"]
    assert "test_components" not in rec2["metrics"]
    assert "train_curve" not in rec2["metrics"]


# --------------------------------------------------------------------------- #
# M2 — the read-only training-curve recorder                                   #
# --------------------------------------------------------------------------- #
def test_train_curve_recorder_samples_read_only():
    """The recorder samples the SB3 logger/episode buffer every ``record_every`` steps, read-only."""
    from src.agents.trainer import _make_curve_recorder

    rec = _make_curve_recorder(record_every=2)

    class _Logger:
        name_to_value = {"train/critic_loss": 1.5, "train/actor_loss": -0.3, "train/ent_coef": 0.1}

    class _Model:
        logger = _Logger()
        ep_info_buffer = [{"r": 2.0}, {"r": 4.0}]

    rec.model = _Model()
    for step in range(1, 7):
        rec.num_timesteps = step
        assert rec._on_step() is True         # never halts training
    assert rec.curve["step"] == [2, 4, 6]     # sampled at the cadence
    assert rec.curve["critic_loss"] == [1.5, 1.5, 1.5]
    assert rec.curve["return"] == [3.0, 3.0, 3.0]  # mean of the episode buffer


def test_train_curve_recorder_tolerates_missing_logger_values():
    """Absent logger keys / empty episode buffer -> NaN, never a crash (early-training steps)."""
    from src.agents.trainer import _make_curve_recorder

    rec = _make_curve_recorder(record_every=1)

    class _Model:
        class logger:  # noqa: N801
            name_to_value: dict = {}
        ep_info_buffer: list = []

    rec.model = _Model()
    rec.num_timesteps = 1
    assert rec._on_step() is True
    assert np.isnan(rec.curve["critic_loss"][0])
    assert np.isnan(rec.curve["return"][0])


def test_build_test_record_archives_train_curve():
    winner = {"candidate_id": "c", "generation": 0, "reward_source": "",
              "feedback_block": "", "metrics": {"val_fitness": 0.1}}
    curve = {"step": [80, 160], "critic_loss": [5.0, 1.0], "actor_loss": [0.0, 0.0],
             "ent_coef": [0.1, 0.1], "return": [1.0, 2.0]}
    rec = build_test_record(winner=winner, arm="scalar", seed=7, reward_hash="h", env_fp="fp",
                            test_returns=[0.01], train_curve=curve)
    assert rec["metrics"]["train_curve"] == curve
