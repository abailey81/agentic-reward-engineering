"""Behaviour tests for the PopArt scale-dependence ablation harness (scripts/popart_ablation.py, T2.4).

The ablation re-trains the FROZEN winners with ``popart=False`` at one seed and reports whether the H2
ordering (distributional > scalar, on Sharpe AND CVaR-5%) survives removing the scaler. These tests cover:
  - the pure reducers (``ordering_from_returns`` / ``ordering_preserved`` / ``ablation_markdown``) on
    hand-built return vectors with a KNOWN ordering — no torch;
  - the driver (``run_popart_ablation``) end-to-end with INJECTED fake trainer / env_builder / winner
    loader, asserting it forces ``popart=False`` on the retrain, touches the test leg once per arm, and
    produces the ordering verdict + the realised-sigma audit (no GPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# scripts/ is not a package; make the module importable (mirrors the other script tests).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import popart_ablation as pa  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure reducers                                                                #
# --------------------------------------------------------------------------- #
def test_ordering_from_returns_ranks_by_sharpe_and_cvar() -> None:
    """A higher-mean / lighter-tail arm ranks above a worse one on both axes."""
    rng = np.random.default_rng(0)
    good = rng.normal(0.002, 0.01, size=400)          # higher mean, tight tail
    bad = rng.normal(-0.001, 0.03, size=400)          # negative mean, fat tail
    out = pa.ordering_from_returns({"distributional": good, "scalar": bad})
    assert out["sharpe_rank"][0] == "distributional"
    assert out["cvar_rank"][0] == "distributional"
    assert out["metrics"]["distributional"]["sharpe"] > out["metrics"]["scalar"]["sharpe"]
    assert out["metrics"]["distributional"]["cvar"] > out["metrics"]["scalar"]["cvar"]


def test_ordering_from_returns_skips_missing_arms() -> None:
    """An arm with ``None`` / empty returns is omitted (never fabricated), not crashed on."""
    out = pa.ordering_from_returns(
        {"distributional": np.array([0.01, -0.02, 0.0]), "scalar": None, "placebo": np.array([])}
    )
    assert "distributional" in out["metrics"]
    assert "scalar" not in out["metrics"]
    assert "placebo" not in out["metrics"]


def test_ordering_preserved_detects_hold_and_flip() -> None:
    """``preserved`` is True iff distributional beats scalar on BOTH Sharpe and CVaR.

    Streams must be non-constant (a zero-variance series has Sharpe 0 by the bootstrap guard); we use a
    fixed pair with a clear mean/tail separation and SWAP the labels to force the flip.
    """
    rng = np.random.default_rng(1)
    better = rng.normal(0.002, 0.01, size=300)   # higher mean, tighter tail
    worse = rng.normal(-0.001, 0.03, size=300)   # negative mean, fat tail
    hold = pa.ordering_from_returns({"distributional": better, "scalar": worse})
    v_hold = pa.ordering_preserved(hold)
    assert v_hold["preserved"] is True
    assert v_hold["sharpe_preserved"] is True and v_hold["cvar_preserved"] is True

    # Flip: scalar now holds the better stream -> distributional no longer leads on either axis.
    flip = pa.ordering_from_returns({"distributional": worse, "scalar": better})
    v_flip = pa.ordering_preserved(flip)
    assert v_flip["preserved"] is False


def test_ordering_preserved_inconclusive_when_arm_missing() -> None:
    """Missing one side of the contrast -> preserved=False (conservative) + a 'missing' note."""
    only_one = pa.ordering_from_returns({"distributional": np.array([0.01, 0.0, -0.01])})
    v = pa.ordering_preserved(only_one)
    assert v["preserved"] is False
    assert v["sharpe_delta"] is None
    assert "scalar" in v["missing"]


# --------------------------------------------------------------------------- #
# Driver (injected fakes; no torch / no GPU)                                   #
# --------------------------------------------------------------------------- #
class _FakeBundle:
    """A bundle whose ``test_returns`` is fixed per arm; records that the test leg was touched once."""

    def __init__(self, arm: str, returns_by_arm: dict, touch_log: list) -> None:
        self._arm = arm
        self._returns_by_arm = returns_by_arm
        self._touch_log = touch_log

    def train_env(self):  # noqa: ANN201
        return object()

    def test_returns(self, policy):  # noqa: ANN001, ARG002
        self._touch_log.append(self._arm)
        return np.asarray(self._returns_by_arm[self._arm], dtype=float)


def _fake_env_builder_factory(returns_by_arm: dict, touch_log: list):
    """Return an ``env_builder(panel, cfg, tw, vw, **kw) -> (reward_fn -> _FakeBundle)``.

    The bundle's arm is inferred from the reward_fn's tag set by the fake reward_builder below, so each
    arm gets its own deterministic test-return vector.
    """

    def _builder(panel, cfg, train_window, val_window, **kwargs):  # noqa: ANN001, ARG001
        def _bind(reward_fn):  # noqa: ANN001
            return _FakeBundle(reward_fn.arm_tag, returns_by_arm, touch_log)

        return _bind

    return _builder


class _FakePolicy:
    def __init__(self, popart_scale) -> None:  # noqa: ANN001
        self.popart_scale = popart_scale

    def predict(self, obs, deterministic=True):  # noqa: ANN001, ARG002
        return np.zeros(1, dtype=np.float32), None


def _fake_trainer_factory(seen_cfgs: list, sigma_by_flag: dict):
    """A ``trainer(agent_cfg, seed) -> (train_env -> _FakePolicy)`` that records the popart flag it saw.

    The returned policy's ``popart_scale`` depends on whether ``popart`` was on, so the test can assert the
    ablation passes ``popart=False`` on the ablation retrain and ``popart=True`` on the sigma-audit retrain.
    """

    def _factory(agent_cfg, seed):  # noqa: ANN001, ARG001
        popart_on = bool(agent_cfg.get("popart", True))
        seen_cfgs.append(popart_on)

        def _train(train_env):  # noqa: ANN001, ARG001
            return _FakePolicy(sigma_by_flag[popart_on])

        return _train

    return _factory


def _fake_reward_builder(source):  # noqa: ANN001
    """Re-instantiate a 'reward' tagged with its arm (parsed from the stub source) — no sandbox."""

    def _reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float(port_ret), {}, None

    # The stub source encodes the arm in a comment ``# arm:<name>`` (set by the fake winner loader).
    _reward.arm_tag = source.split("# arm:")[1].strip()  # type: ignore[attr-defined]
    return _reward


def test_run_popart_ablation_forces_popart_off_and_reports_ordering() -> None:
    """End-to-end: retrain with PopArt OFF per arm, touch the test leg once, verdict the H2 ordering."""
    arms = ("distributional", "scalar")
    # distributional gets the better (non-constant) stream -> ordering PRESERVED without PopArt.
    rng = np.random.default_rng(3)
    returns_by_arm = {
        "distributional": rng.normal(0.002, 0.01, size=300),
        "scalar": rng.normal(-0.001, 0.03, size=300),
    }
    touch_log: list = []
    seen_cfgs: list = []
    # popart=False retrain -> sigma marker {"popart":0.0}; popart=True audit -> sigma_max 1.0.
    sigma_by_flag = {False: {"popart": 0.0}, True: {"popart": 1.0, "sigma_max": 1.0, "sigma_last": 1.0}}

    winners = {
        arm: {"arm": arm, "reward_source": f"stub # arm:{arm}", "reward_source_hash": "x"}
        for arm in arms
    }

    result = pa.run_popart_ablation(
        frozen_root="unused",
        seed=7,
        panel=object(),
        cfg={},
        agent_cfg={"train_steps_per_candidate": 10, "popart": True},  # the ablation must OVERRIDE this
        train_window=(60, 250),
        val_window=(310, 360),
        test_window=(420, 600),
        embargo=21,
        lookback=60,
        trainer=_fake_trainer_factory(seen_cfgs, sigma_by_flag),
        env_builder=_fake_env_builder_factory(returns_by_arm, touch_log),
        reward_builder=_fake_reward_builder,
        winner_loader=lambda _root: winners,
        arms=arms,
        with_popart_audit=True,
    )

    # The ablation retrain forced popart=False (despite agent_cfg popart=True); the audit retrain used True.
    assert False in seen_cfgs, "ablation must retrain with popart=False"
    assert True in seen_cfgs, "sigma audit must retrain with popart=True"
    # Test leg touched once per arm in BOTH retrains (off + on) = 2 touches per arm.
    assert touch_log.count("distributional") == 2
    assert touch_log.count("scalar") == 2
    # The H2 ordering is preserved without PopArt on this constructed stream.
    assert result["ordering"]["preserved"] is True
    assert result["ordering"]["sharpe_delta"] > 0.0
    # The realised-sigma audit captured sigma_max == 1.0 for each arm (identity wrapper).
    assert result["sigma_audit"]["distributional"]["sigma_max"] == 1.0
    # The markdown renders without error and states the verdict.
    md = pa.ablation_markdown(result)
    assert "H2 ordering PRESERVED" in md
    assert "popart=False" in md


def test_run_popart_ablation_skips_arm_without_frozen_winner() -> None:
    """An arm with no frozen winner is reported as skipped (not crashed, not fabricated)."""
    result = pa.run_popart_ablation(
        frozen_root="unused",
        seed=0,
        panel=object(),
        cfg={},
        agent_cfg={"popart": True},
        train_window=(60, 250),
        val_window=(310, 360),
        test_window=(420, 600),
        embargo=21,
        lookback=60,
        trainer=_fake_trainer_factory([], {False: None, True: None}),
        env_builder=_fake_env_builder_factory({}, []),
        reward_builder=_fake_reward_builder,
        winner_loader=lambda _root: {},  # no winners
        arms=("distributional", "scalar"),
        with_popart_audit=False,
    )
    assert "distributional" in result["skipped"]
    assert "scalar" in result["skipped"]
    assert result["ordering"]["preserved"] is False
