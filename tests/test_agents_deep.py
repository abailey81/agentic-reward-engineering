"""DEEP, advanced tests for the agent layer (popart / trainer / evaluator / factory / runner).

These complement — they do NOT duplicate — the behaviour tests in ``tests/test_popart.py``,
``tests/test_popart_ablation.py``, ``tests/test_trainer.py``, ``tests/test_agents.py`` and
``tests/test_runner.py``. The emphasis here is on the *invariants* that make the agent layer
load-bearing for the dissertation:

  * PopArt (``src/agents/popart.py``) — the scale-normalisation algebra is exercised as a set of
    PROPERTIES: the affine output-invariance the design claims (the policy-preserving positive
    scaling), the running second-moment tracking a known stream within tolerance, the ``min_scale``
    clamp boundary, the documented SCALE-EQUIVARIANCE (rewards * k -> sigma * k), determinism (no
    RNG), and the realised-scale logging contract. Property-based via Hypothesis (``derandomize``
    profile from conftest), with metamorphic + boundary + closed-form checks. No torch.
  * trainer (``src/agents/trainer.py``) — config wiring is READ from cfg (ent_coef / seed / steps /
    buffer / tf32 / device), not hardcoded; NormalizedPolicy re-applies the frozen train stats
    exactly; a tiny real SAC construction is torch-guarded.
  * evaluator (``src/agents/evaluator.py``) — the C1 adapter runs the documented
    train->rollout->select pipeline with INJECTED fakes (no torch), forwarding the val-return vector.
  * factory (``src/agents/factory.py``) — dispatch + unknown-kind rejection + config pass-through.
  * runner (``src/env/runner.py``) — train_returns is finite, 1-D, correct length, deterministic,
    and reads ONLY ``info['port_ret']`` (no look-ahead beyond the env contract).

Torch / SB3-dependent assertions are guarded with ``pytest.importorskip`` so the import-light suite
still collects and runs; any actual training is kept to a TINY step budget.
"""
from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym

from src.agents.popart import (
    PopArtRewardScaler,
    realized_scale_stats,
    wrap_popart,
)


# --------------------------------------------------------------------------- #
# Test doubles (no torch)                                                      #
# --------------------------------------------------------------------------- #
class _StreamEnv(gym.Env):  # type: ignore[misc]
    """A minimal gym env that replays a FIXED, pre-supplied reward stream.

    Distinct from ``test_popart._DummyEnv`` (which emits a single constant): this one walks a list,
    so the running second-moment / equivariance / determinism properties can be checked against a
    KNOWN, arbitrary stream. ``info['port_ret']`` mirrors the raw reward so the byte-for-byte
    forwarding can also be asserted on a varying signal.
    """

    def __init__(self, rewards: list[float]) -> None:
        super().__init__()
        self._rewards = [float(r) for r in rewards]
        self._i = 0
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):  # noqa: ANN003, ANN201
        self._i = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001, ANN201
        r = self._rewards[self._i % len(self._rewards)]
        self._i += 1
        return np.zeros(1, dtype=np.float32), r, False, False, {"port_ret": r}


def _scale_after_stream(rewards, **kw):  # noqa: ANN001
    """Drive a fresh scaler through ``rewards`` and return its final ``sigma_last``."""
    env = PopArtRewardScaler(_StreamEnv(list(rewards)), **kw)
    env.reset()
    for _ in range(len(rewards)):
        env.step(0)
    return env.sigma_last


# ===========================================================================
# PopArt — the normalisation invariant / properties
# ===========================================================================
def test_popart_single_reward_normalises_to_unit_magnitude() -> None:
    """The documented bias-correction property: after ONE reward ``v`` the scaled output is ±1.

    The docstring promises the Adam-style debias makes ``sigma == |v|`` after a single reward, so a
    large opening reward is normalised to ``v / |v| = ±1`` on the SAME step (never entering the buffer
    at raw magnitude). Checked here above the ``min_scale`` floor so the clamp does not mask it.
    """
    for v in (12345.0, -8000.0, 3.5, -42.0):
        env = PopArtRewardScaler(_StreamEnv([v]), warmup=0, min_scale=1e-12)
        env.reset()
        _o, scaled, _t, _tr, _info = env.step(0)
        assert scaled == pytest.approx(np.sign(v) * 1.0, abs=1e-6), f"v={v} -> {scaled}"
        # sigma_last must equal |v| (the running scale the critic divided by).
        assert env.sigma_last == pytest.approx(abs(v), rel=1e-6)


def test_popart_affine_output_invariance_to_scale_update() -> None:
    """Metamorphic / affine-invariance: the NORMALISED-then-DENORMALISED reward equals the raw reward.

    PopArt's claim is that dividing the learning signal by ``sigma`` is a positive affine map of the
    value function, so multiplying the scaled output back by the SAME ``sigma`` recovers the raw signal
    exactly — regardless of how the running statistic has moved. This is the precise invariant the
    'policy is unchanged' argument rests on; we verify it step-by-step on a varying stream.
    """
    rng = np.random.default_rng(0)
    stream = (50.0 * rng.standard_normal(80)).tolist()
    env = PopArtRewardScaler(_StreamEnv(stream), warmup=0, min_scale=1e-12, beta=0.1)
    env.reset()
    for raw in stream:
        _o, scaled, _t, _tr, _info = env.step(0)
        # denormalise with the sigma actually in force this step -> recovers the raw reward.
        assert scaled * env.sigma_last == pytest.approx(raw, rel=1e-9, abs=1e-12)


def test_popart_running_second_moment_tracks_known_stream() -> None:
    """The running scale tracks ``sqrt(E[r^2])`` of a known iid stream within tolerance.

    With ``min_scale`` released and a long-memory beta, after many draws the bias-corrected EMA second
    moment must converge to the sample second moment, so ``sigma -> sqrt(mean(r^2))``. We feed a known
    constant-rms stream (scaled normals) and assert sigma lands near the closed form.
    """
    rng = np.random.default_rng(7)
    rms = 30.0
    stream = (rms * rng.standard_normal(20_000)).tolist()
    target = float(np.sqrt(np.mean(np.square(stream))))
    sigma = _scale_after_stream(stream, warmup=0, min_scale=1e-12, beta=2e-3)
    assert sigma == pytest.approx(target, rel=0.08), f"sigma={sigma} vs sqrt(E[r^2])={target}"


def test_popart_scale_equivariance_rewards_times_k_shifts_sigma_by_k() -> None:
    """SCALE-EQUIVARIANCE (the documented arm-dependent-scale behaviour): r*k -> sigma*k.

    This is the load-bearing T2.4 property: two arms whose rewards differ in magnitude by a factor ``k``
    are normalised against scales that differ by ``k``. Driving the SAME stream scaled by several ``k``
    (all well above the ``min_scale`` floor so the clamp is inactive) must scale ``sigma_last`` linearly.
    """
    rng = np.random.default_rng(11)
    base = (100.0 * rng.standard_normal(500)).tolist()
    sigma_base = _scale_after_stream(base, warmup=0, min_scale=1e-12, beta=1e-2)
    for k in (2.0, 5.0, 17.0, 0.5):
        scaled_stream = [k * r for r in base]
        sigma_k = _scale_after_stream(scaled_stream, warmup=0, min_scale=1e-12, beta=1e-2)
        assert sigma_k == pytest.approx(k * sigma_base, rel=1e-9), f"k={k}: {sigma_k} vs {k*sigma_base}"


def test_popart_scaled_output_is_scale_invariant_under_equivariance() -> None:
    """Corollary of equivariance: the SCALED learning reward is invariant to a global reward rescale.

    Because ``sigma`` scales with ``k`` (previous test) and the scaled output is ``raw/sigma``, feeding
    ``k * stream`` yields the IDENTICAL scaled sequence as the unscaled stream. This is exactly why the
    critic's conditioning (hence the learned policy) is invariant to the reward's absolute scale.
    """
    rng = np.random.default_rng(13)
    base = (100.0 * rng.standard_normal(300)).tolist()

    def _scaled_seq(stream):  # noqa: ANN001
        env = PopArtRewardScaler(_StreamEnv(list(stream)), warmup=0, min_scale=1e-12, beta=1e-2)
        env.reset()
        return np.array([env.step(0)[1] for _ in range(len(stream))])

    seq1 = _scaled_seq(base)
    seq_k = _scaled_seq([7.0 * r for r in base])
    np.testing.assert_allclose(seq1, seq_k, rtol=1e-9, atol=1e-12)


def test_popart_min_scale_clamp_boundary() -> None:
    """The ``min_scale`` clamp: a sub-floor reward yields sigma == min_scale (pure shrink, never amplify).

    Boundary behaviour at three regimes:
      * a tiny reward (rms << min_scale) -> sigma pinned exactly at the floor (identity-ish, never amplified);
      * a reward exactly at the floor -> sigma == floor;
      * a large reward (rms >> floor) -> sigma tracks the rms, above the floor.
    """
    floor = 2.5
    tiny = _scale_after_stream([1e-6] * 50, warmup=0, min_scale=floor)
    assert tiny == pytest.approx(floor), "sub-floor reward must clamp sigma to min_scale (no amplification)"

    at_floor = _scale_after_stream([floor] * 50, warmup=0, min_scale=floor)
    assert at_floor == pytest.approx(floor, rel=1e-6)

    big = _scale_after_stream([1000.0] * 50, warmup=0, min_scale=floor)
    assert big > floor and big == pytest.approx(1000.0, rel=1e-3)


def test_popart_warmup_holds_scale_at_one_then_releases() -> None:
    """``warmup`` (ablation knob): early rewards pass through unscaled while ``count < warmup`` (sigma==1).

    The gate is ``count < warmup`` (count incremented BEFORE the scale is read), so with warmup=5 the
    steps at count 1..4 are passed through unscaled and the scale engages at count==5 — we assert the
    documented hold (the pre-warmup steps are exactly the raw reward) and the post-warmup shrink.
    """
    big = 1.0e4
    env = PopArtRewardScaler(_StreamEnv([big] * 20), warmup=5, min_scale=1.0, beta=0.5)
    env.reset()
    outs = [env.step(0)[1] for _ in range(20)]
    # While count < warmup (steps 1..4) the raw reward is passed through (sigma == 1).
    assert all(o == pytest.approx(big) for o in outs[:4]), "warmup must pass rewards through unscaled"
    # Once the scale engages (count >= warmup) the large reward is shrunk well below its raw magnitude.
    assert abs(outs[-1]) < big / 100.0, "post-warmup the large reward must be shrunk"


def test_popart_is_deterministic_given_identical_stream() -> None:
    """No RNG: two scalers fed the byte-identical stream produce byte-identical scaled outputs + stats."""
    rng = np.random.default_rng(99)
    stream = (250.0 * rng.standard_normal(120)).tolist()

    def _run():  # noqa: ANN202
        env = PopArtRewardScaler(_StreamEnv(list(stream)), warmup=0, beta=3e-3)
        env.reset()
        outs = [env.step(0)[1] for _ in range(len(stream))]
        return outs, env.realized_scale_stats()

    o1, s1 = _run()
    o2, s2 = _run()
    assert o1 == o2  # exact equality (no float jitter): deterministic, replay-safe
    assert s1 == s2


def test_popart_forwards_info_byte_for_byte_on_varying_stream() -> None:
    """info['port_ret'] is forwarded UNCHANGED even as the learning reward is scaled (object-of-study seal)."""
    rng = np.random.default_rng(5)
    stream = (500.0 * rng.standard_normal(60)).tolist()
    env = PopArtRewardScaler(_StreamEnv(list(stream)), warmup=0, beta=1e-2)
    env.reset()
    for raw in stream:
        _o, scaled, _t, _tr, info = env.step(0)
        assert info["port_ret"] == raw  # forwarded byte-for-byte
        # The scaled learning signal generally differs once sigma has moved off 1 / the floor.
    # And on a large-magnitude stream the wrapper must have actually scaled somewhere.
    assert env.sigma_max > 1.0


def test_popart_sigma_max_is_running_maximum_monotone() -> None:
    """``sigma_max`` is the non-decreasing running max of the per-step sigma (the audit upper bound)."""
    # Front-load a huge reward then small ones: sigma_max must latch the peak and never fall.
    env = PopArtRewardScaler(_StreamEnv([1.0e5] + [0.1] * 100), warmup=0, beta=0.5, min_scale=1.0)
    env.reset()
    prev_max = 0.0
    peak_seen = 0.0
    for _ in range(101):
        env.step(0)
        assert env.sigma_max >= prev_max, "sigma_max must be non-decreasing"
        prev_max = env.sigma_max
        peak_seen = max(peak_seen, env.sigma_last)
    assert env.sigma_max == pytest.approx(peak_seen)
    assert env.sigma_max >= env.sigma_last  # the latched peak >= the final scale


def test_popart_realized_scale_stats_contract() -> None:
    """The realised-scale dict carries JSON-friendly floats with the documented keys + the off-marker."""
    env = PopArtRewardScaler(_StreamEnv([1000.0] * 30), warmup=0, beta=0.5)
    env.reset()
    for _ in range(30):
        env.step(0)
    stats = env.realized_scale_stats()
    assert set(stats) == {"popart", "sigma_max", "sigma_last", "raw_rms_max", "raw_rms_last", "count"}
    assert all(isinstance(v, float) for v in stats.values())  # JSON round-trip safe
    assert stats["popart"] == 1.0
    assert stats["count"] == 30.0
    # Module-level helper: unwraps a layered wrapper, marks an off env explicitly (never silent None).
    assert realized_scale_stats(env) == stats
    assert realized_scale_stats(_StreamEnv([1.0])) == {"popart": 0.0}


def test_popart_helper_unwraps_layered_wrapper() -> None:
    """``realized_scale_stats`` finds the scaler even when another gym.Wrapper sits on TOP of it."""
    scaler = PopArtRewardScaler(_StreamEnv([1000.0] * 10), warmup=0, beta=0.5)
    outer = gym.Wrapper(scaler)  # a benign wrapper layered above the scaler
    scaler.reset()
    for _ in range(10):
        scaler.step(0)
    out = realized_scale_stats(outer)
    assert out["popart"] == 1.0 and out["sigma_max"] > 1.0


def test_wrap_popart_reads_config_knobs() -> None:
    """wrap_popart wires beta / min_scale / warmup from cfg (config-driven, not hardcoded)."""
    env = _StreamEnv([1.0])
    scaler = wrap_popart(env, {"popart": True, "popart_beta": 0.25, "popart_min_scale": 4.0, "popart_warmup": 3})
    assert isinstance(scaler, PopArtRewardScaler)
    assert scaler.beta == pytest.approx(0.25)
    assert scaler.min_scale == pytest.approx(4.0)
    assert scaler.warmup == 3
    # Defaults (no keys) match the documented design (beta=1e-3, min_scale=1.0, warmup=0).
    default = wrap_popart(_StreamEnv([1.0]), {})
    assert default.beta == pytest.approx(1e-3)
    assert default.min_scale == pytest.approx(1.0)
    assert default.warmup == 0


def test_popart_scale_persists_across_episode_reset() -> None:
    """reset() does NOT reset the running scale (one continuing learner — the buffer is full-history)."""
    env = PopArtRewardScaler(_StreamEnv([1000.0] * 10), warmup=0, beta=0.5)
    env.reset()
    for _ in range(10):
        env.step(0)
    count_before, sigma_before = env.count, env.sigma_max
    env.reset()  # episode boundary
    assert env.count == count_before, "count must persist across reset (continuing learner)"
    assert env.sigma_max == sigma_before


# ===========================================================================
# trainer.py — config wiring (no torch needed for resolve_agent_kwargs)
# ===========================================================================
def test_resolve_agent_kwargs_reads_config_not_hardcoded() -> None:
    """Every SAC hyperparameter is READ from cfg (ent_coef / seed / lr / gamma / batch / device)."""
    from src.agents.trainer import resolve_agent_kwargs

    cfg = {
        "train_steps_per_candidate": 1234,
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "ent_coef": 0.07,
        "batch_size": 64,
        "device": "cpu",
        "verbose": 2,
    }
    kwargs, train_steps = resolve_agent_kwargs(cfg, seed=42)
    assert train_steps == 1234
    assert kwargs["learning_rate"] == pytest.approx(1e-3)
    assert kwargs["gamma"] == pytest.approx(0.95)
    assert kwargs["ent_coef"] == 0.07
    assert kwargs["batch_size"] == 64
    assert kwargs["device"] == "cpu"
    assert kwargs["verbose"] == 2
    assert kwargs["seed"] == 42  # seed argument wins, threaded into kwargs
    assert kwargs["policy"] == "MlpPolicy"


def test_resolve_agent_kwargs_ent_coef_defaults_to_auto() -> None:
    """ent_coef defaults to 'auto' (SAC temperature auto-tuning; the documented default)."""
    from src.agents.trainer import resolve_agent_kwargs

    kwargs, _ = resolve_agent_kwargs({}, seed=0)
    assert kwargs["ent_coef"] == "auto"


def test_resolve_agent_kwargs_buffer_sized_to_train_steps() -> None:
    """The replay buffer is sized to the step budget (ADR-025), and an explicit override wins."""
    from src.agents.trainer import resolve_agent_kwargs

    kwargs, train_steps = resolve_agent_kwargs({"train_steps_per_candidate": 777}, seed=0)
    assert kwargs["buffer_size"] == train_steps == 777  # not SB3's 1_000_000
    kwargs2, _ = resolve_agent_kwargs(
        {"train_steps_per_candidate": 777, "buffer_size": 100}, seed=0
    )
    assert kwargs2["buffer_size"] == 100  # explicit override honoured -- BELOW the cap; see next test


def test_replay_cap_clamps_the_frozen_budget_at_both_construction_sites() -> None:
    """The memory-safety cap is a true CEILING at the frozen 400k budget (R77), not a default.

    Nothing else in the suite exercised the cap ACTUALLY clamping: every other ``buffer_size``
    assertion uses a sub-cap value (777 / 321 / 100), so a refactor that dropped the ``min(...)``
    would leave the suite green while a 400k run allocated a ~5.6 GB replay buffer and OOM'd the
    15.6 GB laptop (ADR-025 EXTENDED). Both sites are asserted because the cap is documented as
    "enforced at BOTH" (``factory._policy_kwargs`` and ``trainer.resolve_agent_kwargs``);
    ``_policy_kwargs`` is used directly so the assertion needs no SB3 agent construction.
    """
    from src.agents.factory import _policy_kwargs, campaign_replay_cap
    from src.agents.trainer import resolve_agent_kwargs

    cap = campaign_replay_cap()

    # A ceiling cannot be overridden UPWARD -- that is the whole point of a RAM guard.
    assert _policy_kwargs({"train_steps_per_candidate": 400_000})["buffer_size"] == cap
    assert _policy_kwargs({"buffer_size": 400_000})["buffer_size"] == cap
    assert _policy_kwargs({"train_steps_per_candidate": 777})["buffer_size"] == 777  # no-op below

    kw_implicit, steps = resolve_agent_kwargs({"train_steps_per_candidate": 400_000}, seed=0)
    assert kw_implicit["buffer_size"] == cap
    assert steps == 400_000, "the cap bounds the BUFFER only -- it must never truncate the budget"
    kw_explicit, _ = resolve_agent_kwargs(
        {"train_steps_per_candidate": 400_000, "buffer_size": 400_000}, seed=0
    )
    assert kw_explicit["buffer_size"] == cap


def test_default_replay_cap_matches_the_registered_config() -> None:
    """``DEFAULT_REPLAY_CAP`` duplicates ``config/campaign.yaml agent.buffer_size`` -- bind them.

    ``campaign_replay_cap()`` falls back to the literal when the config cannot be read, so if the
    registered value were changed without updating the literal, that fallback would silently resolve
    a DIFFERENT buffer than the registered one -- a laptop/cluster parity break in the failure path.
    Nothing (no test, no freeze guard) bound the two before this.
    """
    from src.agents.factory import DEFAULT_REPLAY_CAP, campaign_replay_cap
    from src.utils.config import load_config

    registered = (load_config("campaign").get("agent") or {}).get("buffer_size")
    assert registered == DEFAULT_REPLAY_CAP, (
        f"config/campaign.yaml agent.buffer_size={registered!r} has drifted from "
        f"DEFAULT_REPLAY_CAP={DEFAULT_REPLAY_CAP!r}"
    )
    assert campaign_replay_cap() == registered


def test_resolve_agent_kwargs_train_steps_fallback_chain() -> None:
    """train_steps resolves train_steps_per_candidate -> train_steps -> 50000 (documented fallback)."""
    from src.agents.trainer import resolve_agent_kwargs

    _, a = resolve_agent_kwargs({"train_steps_per_candidate": 10}, seed=0)
    _, b = resolve_agent_kwargs({"train_steps": 20}, seed=0)
    _, c = resolve_agent_kwargs({}, seed=0)
    assert (a, b, c) == (10, 20, 50000)


def test_resolve_agent_kwargs_seed_is_deterministic() -> None:
    """Same cfg + seed -> identical resolved kwargs (determinism of the config resolution)."""
    from src.agents.trainer import resolve_agent_kwargs

    k1, s1 = resolve_agent_kwargs({"train_steps_per_candidate": 50}, seed=3)
    k2, s2 = resolve_agent_kwargs({"train_steps_per_candidate": 50}, seed=3)
    assert k1 == k2 and s1 == s2
    # A different seed flows through to the kwargs.
    k3, _ = resolve_agent_kwargs({"train_steps_per_candidate": 50}, seed=4)
    assert k3["seed"] == 4 and k1["seed"] == 3


def test_apply_tf32_is_safe_noop_without_torch() -> None:
    """_apply_tf32 never raises (torch absent / no CUDA -> documented safe no-op)."""
    from src.agents.trainer import _apply_tf32

    _apply_tf32(True)
    _apply_tf32(False)  # both directions must be safe to call


def test_normalized_policy_reapplies_frozen_train_stats_exactly() -> None:
    """NormalizedPolicy applies clip((obs-mean)/sqrt(var+eps), -clip, clip) with the FROZEN stats.

    The val-leakage defence (deep-research §2): evaluation obs are normalised with TRAIN-period stats.
    We check the exact closed form (including the clip) against a hand-computed value, using a stub
    model that simply echoes the normalised obs it receives — no torch.
    """
    from src.agents.trainer import NormalizedPolicy

    seen = {}

    class _EchoModel:
        def predict(self, obs, deterministic=True):  # noqa: ANN001, ARG002
            seen["obs"] = np.asarray(obs)
            return np.zeros(1), None

    mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    var = np.array([4.0, 9.0, 1.0], dtype=np.float32)
    eps = 1e-8
    clip = 2.0
    policy = NormalizedPolicy(_EchoModel(), mean=mean, var=var, epsilon=eps, clip_obs=clip)

    raw = np.array([5.0, 2.0, 3.0], dtype=np.float32)  # first dim will exceed clip
    policy.predict(raw)
    expected = np.clip((raw - mean) / np.sqrt(var + eps), -clip, clip)
    np.testing.assert_allclose(seen["obs"], expected, rtol=1e-6, atol=1e-7)
    # The high-leverage first dim ((5-1)/2 = 2.0) sits exactly at the clip edge; push it past:
    raw2 = np.array([100.0, 2.0, 3.0], dtype=np.float32)
    policy.predict(raw2)
    assert seen["obs"][0] == pytest.approx(clip)  # clipped, not 49.5


def test_normalized_policy_carries_popart_scale_marker() -> None:
    """The policy carries the realised PopArt scale dict through to the candidate/test record (T2.4)."""
    from src.agents.trainer import NormalizedPolicy

    marker = {"popart": 1.0, "sigma_max": 7.0, "sigma_last": 3.0, "count": 100.0}
    policy = NormalizedPolicy(
        object(), mean=np.zeros(2), var=np.ones(2), epsilon=1e-8, clip_obs=10.0, popart_scale=marker
    )
    assert policy.popart_scale == marker


@pytest.mark.slow
def test_train_agent_tiny_budget_constructs_and_predicts(synthetic_panel, env_cfg_fixture) -> None:
    """A TINY real SAC train (torch-guarded) returns a predicting NormalizedPolicy with finite stats."""
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from src.agents.trainer import NormalizedPolicy, train_agent
    from src.env.runner import make_env_builder

    cfg = env_cfg_fixture
    lookback = int(cfg["state"]["lookback_days"])
    builder = make_env_builder(synthetic_panel, cfg, (lookback, 200), (200, synthetic_panel.T))
    bundle = builder(_simple_reward)
    agent_cfg = {"train_steps_per_candidate": 120, "learning_starts": 16, "normalize_obs": True}
    policy = train_agent(bundle.train_env(), agent_cfg, seed=0)
    assert isinstance(policy, NormalizedPolicy)
    assert np.isfinite(policy._mean).all() and (policy._std > 0).all()
    # The realised PopArt scale was surfaced for the audit.
    assert policy.popart_scale is not None and "popart" in policy.popart_scale


# ===========================================================================
# factory.py — dispatch + config pass-through (no torch needed for the contract)
# ===========================================================================
def test_factory_constants_and_dispatch_validation() -> None:
    """Unknown kind is rejected with a clear ValueError BEFORE any backend import."""
    from src.agents import factory

    with pytest.raises(ValueError) as exc:
        factory.make_agent("not-a-kind", env=object(), cfg={})
    msg = str(exc.value)
    assert "not-a-kind" in msg and "headline" in msg and "distributional" in msg
    assert factory.HEADLINE_ALGO == "SAC" and factory.DISTRIBUTIONAL_ALGO == "TQC"


def test_factory_policy_kwargs_resolution() -> None:
    """_policy_kwargs reads the shared learner settings from cfg (held identical across SAC/TQC)."""
    from src.agents.factory import _policy_kwargs

    cfg = {
        "policy": "MlpPolicy",
        "learning_rate": 5e-4,
        "gamma": 0.97,
        "ent_coef": "auto",
        "batch_size": 128,
        "seed": 9,
        "train_steps_per_candidate": 321,
        "policy_kwargs": {"net_arch": [64, 64]},
    }
    kw = _policy_kwargs(cfg)
    assert kw["learning_rate"] == pytest.approx(5e-4)
    assert kw["gamma"] == pytest.approx(0.97)
    assert kw["batch_size"] == 128
    assert kw["seed"] == 9
    assert kw["buffer_size"] == 321  # buffer defaults to the step budget (ADR-025), not 1M
    assert kw["policy_kwargs"] == {"net_arch": [64, 64]}  # passed through when present


def test_factory_headline_constructs_tiny_sac(env_cfg_fixture) -> None:
    """Torch-guarded: make_headline_agent builds a real SB3 SAC bound to a tiny env."""
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from src.agents.factory import make_headline_agent

    env = _StreamBoxEnv()
    agent = make_headline_agent(env, {"buffer_size": 64, "learning_starts": 8, "seed": 0})
    assert isinstance(agent, sb3.SAC)


def test_factory_distributional_constructs_tiny_tqc() -> None:
    """Torch-guarded: make_distributional_agent builds a real sb3-contrib TQC; TQC-knobs routed."""
    pytest.importorskip("torch")
    sb3c = pytest.importorskip("sb3_contrib")
    from src.agents.factory import make_distributional_agent

    env = _StreamBoxEnv()
    agent = make_distributional_agent(
        env, {"buffer_size": 64, "learning_starts": 8, "seed": 0, "top_quantiles_to_drop_per_net": 2}
    )
    assert isinstance(agent, sb3c.TQC)


# ===========================================================================
# evaluator.py — the C1 adapter pipeline (injected fakes, no torch)
# ===========================================================================
class _FakeBundle:
    """Records the train->rollout->select ordering and returns a fixed val-return vector."""

    def __init__(self, val_returns, order_log) -> None:  # noqa: ANN001
        self._val = np.asarray(val_returns, dtype=float)
        self._order = order_log

    def train_env(self):  # noqa: ANN201
        self._order.append("train_env")
        return object()

    def val_returns(self, policy):  # noqa: ANN001, ARG002
        self._order.append("val_returns")
        return self._val


def test_evaluate_reward_runs_documented_pipeline_in_order() -> None:
    """evaluate_reward runs env_builder -> agent_trainer(train_env) -> fitness(val_returns, n_trials)."""
    from src.agents.evaluator import evaluate_reward

    order: list[str] = []
    val_vec = np.array([0.01, -0.02, 0.03, 0.0])
    seen = {}

    def env_builder(reward_fn):  # noqa: ANN001
        seen["reward_fn"] = reward_fn
        order.append("env_builder")
        return _FakeBundle(val_vec, order)

    def agent_trainer(train_env):  # noqa: ANN001
        order.append("agent_trainer")
        assert train_env is not None  # received the env from bundle.train_env()
        return "policy"

    def fitness_fn(val_returns, n_trials):  # noqa: ANN001
        order.append("fitness_fn")
        np.testing.assert_array_equal(val_returns, val_vec)
        assert n_trials == 5
        return 1.5

    fit = evaluate_reward("REWARD", env_builder, agent_trainer, fitness_fn, n_trials=5)
    assert fit == pytest.approx(1.5)
    assert seen["reward_fn"] == "REWARD"
    # Documented ordering: build env, build train_env, train, roll val, score.
    assert order == ["env_builder", "train_env", "agent_trainer", "val_returns", "fitness_fn"]


def test_evaluate_reward_with_returns_surfaces_val_vector() -> None:
    """evaluate_reward_with_returns ALSO returns the realised validation-return vector (for PBO/CSCV)."""
    from src.agents.evaluator import evaluate_reward, evaluate_reward_with_returns

    val_vec = np.array([0.05, -0.01, 0.02])

    def env_builder(reward_fn):  # noqa: ANN001, ARG001
        return _FakeBundle(val_vec, [])

    def agent_trainer(train_env):  # noqa: ANN001, ARG001
        return "policy"

    def fitness_fn(val_returns, n_trials):  # noqa: ANN001, ARG001
        return float(np.mean(val_returns))

    fit, returns = evaluate_reward_with_returns("R", env_builder, agent_trainer, fitness_fn, 3)
    np.testing.assert_array_equal(returns, val_vec)
    assert returns.ndim == 1
    assert fit == pytest.approx(np.mean(val_vec))
    # evaluate_reward delegates here and returns the SAME scalar (scalar-only contract).
    assert evaluate_reward("R", env_builder, agent_trainer, fitness_fn, 3) == pytest.approx(fit)


def test_make_reward_evaluator_is_single_arg_closure() -> None:
    """make_reward_evaluator returns a (reward_fn)->fitness closure (the random-search-over-code contract)."""
    from src.agents.evaluator import make_reward_evaluator

    captured = {}

    def env_builder(reward_fn):  # noqa: ANN001
        captured["reward_fn"] = reward_fn
        return _FakeBundle([0.0, 0.0], [])

    ev = make_reward_evaluator(env_builder, lambda _e: "p", lambda v, n: 0.42, n_trials=4)  # noqa: ARG005
    out = ev("CANDIDATE_REWARD")
    assert out == pytest.approx(0.42)
    assert captured["reward_fn"] == "CANDIDATE_REWARD"


def test_make_template_evaluator_maps_coeffs_to_reward_first() -> None:
    """make_template_evaluator maps a coefficient vector -> reward via params_to_reward, then evaluates."""
    from src.agents.evaluator import make_template_evaluator

    seen = {}

    def params_to_reward(coeffs):  # noqa: ANN001
        seen["coeffs"] = coeffs
        return f"reward_from_{coeffs}"

    def env_builder(reward_fn):  # noqa: ANN001
        seen["reward_fn"] = reward_fn
        return _FakeBundle([0.0, 0.0], [])

    ev = make_template_evaluator(
        params_to_reward, env_builder, lambda _e: "p", lambda v, n: 0.9, n_trials=2  # noqa: ARG005
    )
    out = ev([0.1, 0.2])
    assert out == pytest.approx(0.9)
    assert seen["coeffs"] == [0.1, 0.2]
    assert seen["reward_fn"] == "reward_from_[0.1, 0.2]"  # template instantiated BEFORE evaluation


# ===========================================================================
# runner.py — train_returns rollout (fake policy, no torch)
# ===========================================================================
class _FakePolicy:
    """Predicts a fixed zero action (-> uniform simplex). No torch. Counts predict calls."""

    def __init__(self, n_act: int) -> None:
        self.n_act = n_act
        self.calls = 0

    def predict(self, obs, deterministic=True):  # noqa: ANN001, ARG002
        self.calls += 1
        assert deterministic is True, "rollout must roll the policy DETERMINISTICALLY (walk-forward)"
        return np.zeros(self.n_act, dtype=np.float32), None


def _simple_reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
    pr = float(port_ret)
    return pr, {"port_ret": pr}, None


def test_train_returns_finite_1d_correct_length(synthetic_panel, env_cfg_fixture) -> None:
    """EnvBundle.train_returns -> finite, 1-D, length == (train_end - train_start)."""
    from src.env.runner import make_env_builder

    cfg = env_cfg_fixture
    lookback = int(cfg["state"]["lookback_days"])
    train_window = (lookback, 350)
    builder = make_env_builder(synthetic_panel, cfg, train_window, (350, synthetic_panel.T))
    bundle = builder(_simple_reward)
    policy = _FakePolicy(synthetic_panel.N + 1)
    rets = bundle.train_returns(policy)
    assert rets.ndim == 1
    assert rets.shape == (train_window[1] - train_window[0],)
    assert np.isfinite(rets).all()
    # One predict per traded step (the deterministic walk-forward contract).
    assert policy.calls == train_window[1] - train_window[0]


def test_train_returns_deterministic_given_seed(synthetic_panel, env_cfg_fixture) -> None:
    """Re-rolling the same (panel, reward, policy) over the train window is byte-identical."""
    from src.env.runner import make_env_builder

    cfg = env_cfg_fixture
    lookback = int(cfg["state"]["lookback_days"])
    builder = make_env_builder(synthetic_panel, cfg, (lookback, 350), (350, synthetic_panel.T))
    bundle = builder(_simple_reward)
    a = bundle.train_returns(_FakePolicy(synthetic_panel.N + 1))
    b = bundle.train_returns(_FakePolicy(synthetic_panel.N + 1))
    assert np.array_equal(a, b)


def test_train_returns_no_lookahead_beyond_env_contract(synthetic_panel, env_cfg_fixture) -> None:
    """train_returns reads ONLY info['port_ret']: a reward fed FUTURE-knowledge can't change the realised series.

    The realised return is the env's ``info['port_ret']`` (gross minus turnover cost), computed from the
    chosen action and the panel — INDEPENDENT of whatever scalar the reward function emits. Two rewards
    that emit wildly different totals (a clairvoyant-looking one vs. the identity) therefore yield the
    IDENTICAL realised train-return series, proving the rollout never lets the reward leak into the
    measured returns (no look-ahead beyond the env contract).
    """
    from src.env.runner import make_env_builder

    cfg = env_cfg_fixture
    lookback = int(cfg["state"]["lookback_days"])

    def _wild_reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        # Emit a huge, signal-laden total; the realised port_ret in info is untouched.
        return 1.0e6 * float(port_ret) + 12345.0, {"port_ret": float(port_ret)}, None

    builder = make_env_builder(synthetic_panel, cfg, (lookback, 350), (350, synthetic_panel.T))
    policy = _FakePolicy(synthetic_panel.N + 1)
    a = builder(_simple_reward).train_returns(policy)
    b = builder(_wild_reward).train_returns(_FakePolicy(synthetic_panel.N + 1))
    np.testing.assert_array_equal(a, b)


def test_train_and_val_returns_are_on_disjoint_windows(synthetic_panel, env_cfg_fixture) -> None:
    """train_returns and val_returns roll DIFFERENT splits (measurement vs selection, audit B-2/B-3)."""
    from src.env.runner import make_env_builder

    cfg = env_cfg_fixture
    lookback = int(cfg["state"]["lookback_days"])
    train_window = (lookback, 350)
    val_window = (350, synthetic_panel.T)
    bundle = make_env_builder(synthetic_panel, cfg, train_window, val_window)(_simple_reward)
    tr = bundle.train_returns(_FakePolicy(synthetic_panel.N + 1))
    va = bundle.val_returns(_FakePolicy(synthetic_panel.N + 1))
    assert tr.shape == (train_window[1] - train_window[0],)
    assert va.shape == (val_window[1] - val_window[0],)
    # Disjoint lengths from disjoint windows; not the same vector.
    assert tr.shape != va.shape or not np.array_equal(tr, va)


# --------------------------------------------------------------------------- #
# Local fixtures / helpers                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def env_cfg_fixture():
    from src.utils.config import load_config

    return load_config("environment")


class _StreamBoxEnv(gym.Env):  # type: ignore[misc]
    """A tiny continuous gym env (Box obs+action) sufficient to CONSTRUCT an SB3 SAC/TQC."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):  # noqa: ANN003, ANN201
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001, ANN201
        return np.zeros(3, dtype=np.float32), 0.0, False, True, {}
