"""Deep-P3 dynamics tests for PopArt scale normalization (src/agents/popart.py).

These probe the *running-statistic dynamics* under heavy-tailed / intermittent-huge rewards — a layer
BELOW the already-covered sqrt-overflow guard (tests/test_popart.py). Two things are pinned here:

1. **The scale is an UNCENTERED second moment** ``sqrt(EMA[r^2])`` (no mean term), so the classic
   "centred variance goes slightly negative under cancellation -> sqrt(neg)=NaN" failure mode is
   *structurally impossible*: ``sq_ema`` is a convex combination of non-negative squares, hence >= 0 by
   construction, and ``sigma >= min_scale > 0`` always. We assert this over a mixed-sign stream.
2. **NaN hardening (regression guard).** ``np.clip`` maps +-inf to +-cap but leaves ``nan`` as ``nan``.
   Before the P3 fix, a single NaN raw reward poisoned ``sq_ema`` irreversibly and the ``_scale`` backstop
   then silently pinned ``sigma`` at the floor — disabling scaling for the rest of the candidate and
   emitting a NaN learning signal. ``safe_call`` prevents a NaN in production, so this is defense-in-depth
   consistent with the module's existing fail-loud overflow guard.

Also characterises (pins, does not judge) the intended heavy-tail dynamics: a single spike is bounded and
sign-preserving, and the post-spike suppression of ordinary rewards stays FINITE and is surfaced by the
``sigma_max`` audit (T2.4).
"""
from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest

from src.agents.popart import PopArtRewardScaler, realized_scale_stats


class _ScriptEnv(gym.Env):  # type: ignore[misc]
    """Minimal env replaying a scripted reward list (isolates the scaler's arithmetic)."""

    metadata: dict = {}

    def __init__(self, rewards: list[float]) -> None:
        self._r = [float(x) for x in rewards]
        self._i = 0
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # noqa: ANN001, ANN201, ARG002
        self._i = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001, ANN201, ARG002
        r = self._r[self._i]
        self._i += 1
        return np.zeros(1, dtype=np.float32), r, False, False, {"port_ret": 0.0}


def _run(rewards, **kw):  # noqa: ANN001, ANN003
    """Step the scaler over ``rewards``; return (scaled_list, wrapper)."""
    w = PopArtRewardScaler(_ScriptEnv(rewards), **kw)
    w.reset()
    scaled = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fail-loud RuntimeWarning is asserted separately
        for _ in range(len(rewards)):
            _o, s, *_ = w.step(np.zeros(1, dtype=np.float32))
            scaled.append(float(s))
    return scaled, w


# --------------------------------------------------------------------------- #
# 1. Structural: uncentered second moment => no negative variance, sign kept   #
# --------------------------------------------------------------------------- #
def test_uncentered_second_moment_never_negative_and_sign_preserved() -> None:
    """sq_ema >= 0 and sigma >= min_scale at every step for a mixed-sign stream; sign is preserved.

    The scale is ``sqrt(EMA[r^2])`` (root second moment, NOT a centred std), so there is no ``E[X^2]-E[X]^2``
    cancellation that could push the radicand negative -> the ``sqrt(neg)=NaN`` class cannot arise here.
    """
    stream = [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, -10.0, 7.0, -0.001, 0.001] * 3
    w = PopArtRewardScaler(_ScriptEnv(stream), beta=0.1)
    w.reset()
    for raw in stream:
        _o, scaled, *_ = w.step(np.zeros(1, dtype=np.float32))
        assert w.sq_ema >= 0.0, "second moment must never be negative (uncentered => sum of squares)"
        assert np.isfinite(w.sq_ema)
        assert w.sigma_last >= w.min_scale > 0.0, "sigma is floored strictly positive"
        assert np.isfinite(scaled)
        if raw != 0.0:  # scaled = raw / sigma, sigma > 0 => sign is preserved exactly
            assert np.sign(scaled) == np.sign(raw)


# --------------------------------------------------------------------------- #
# 2. NaN hardening (the P3 fix) — regression guard                             #
# --------------------------------------------------------------------------- #
def test_nan_raw_does_not_poison_sq_ema_or_propagate() -> None:
    """A NaN raw reward is sanitized fail-loud: sq_ema stays finite, no NaN is emitted, PopArt stays live.

    Pre-fix this FAILED: ``np.clip(nan)==nan`` poisoned ``sq_ema`` forever and the ``_scale`` backstop pinned
    ``sigma`` at the floor, so the tail 1e4 rewards would pass through UNSCALED (scaled==1e4, not ~1).
    """
    # 0.5, 0.5, NaN, then a sustained 1e4 stream that PopArt must still shrink to ~O(1) if it is alive.
    seq = [0.5, 0.5, float("nan")] + [1.0e4] * 40
    scaled, w = _run(seq, beta=0.5)

    assert all(np.isfinite(s) for s in scaled), "no scaled output may be NaN/inf"
    assert scaled[2] == 0.0, "the NaN step maps to a neutral 0.0 learning signal"
    assert np.isfinite(w.sq_ema), "sq_ema must not be poisoned to NaN"
    assert w.raw_clamp_count == 1, "the NaN was counted once (fail-loud, same channel as overflow)"
    # PopArt still ACTIVE after the NaN: the sustained 1e4 tail is shrunk to O(1) (would be ~1e4 if disabled).
    tail = np.abs(scaled[-10:])
    assert tail.max() < 5.0, f"PopArt was silently disabled after NaN (tail max |scaled|={tail.max():.3g})"


def test_nan_raw_emits_fail_loud_warning() -> None:
    """The NaN sanitization uses the SAME fail-loud channel as the overflow clamp (counted + warned once)."""
    w = PopArtRewardScaler(_ScriptEnv([0.5, float("nan"), 0.5]))
    w.reset()
    w.step(np.zeros(1, dtype=np.float32))
    with pytest.warns(RuntimeWarning, match="overflow-safe cap"):
        _o, scaled, *_ = w.step(np.zeros(1, dtype=np.float32))
    assert scaled == 0.0 and np.isfinite(w.sq_ema)


# --------------------------------------------------------------------------- #
# 3. inf handling regression (must stay clamped-finite after the NaN edit)     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad, want_sign", [(float("inf"), 1.0), (float("-inf"), -1.0)])
def test_inf_raw_is_clamped_finite_and_sign_preserved(bad, want_sign) -> None:  # noqa: ANN001
    """+-inf are clamped to +-_RAW_CAP (finite) — behaviour unchanged by the NaN edit; sq_ema stays finite."""
    scaled, w = _run([0.5, 0.5, bad, 0.5, 0.5], beta=0.5)
    assert all(np.isfinite(s) for s in scaled)
    assert np.isfinite(w.sq_ema)
    assert w.raw_clamp_count == 1
    assert np.sign(scaled[2]) == want_sign, "an infinite reward keeps its sign through the clamp"


# --------------------------------------------------------------------------- #
# 4. Heavy-tail dynamics (pinned + auditable)                                  #
# --------------------------------------------------------------------------- #
def test_single_spike_is_bounded_sign_preserved_and_auditable() -> None:
    """A lone huge spike among tiny rewards is tamed to a BOUNDED, sign-preserving value; sigma_max surfaces it.

    This is the whole point of the wrapper: the raw 1e6 that drove the critic to ~1e6/(1-gamma) is normalized
    to O(1)-O(30) (never exploding), and the realised ``sigma_max`` >> 1 makes the rescale auditable (T2.4).
    """
    seq = [0.001] * 10 + [-1.0e6] + [0.001] * 10
    scaled, w = _run(seq)
    spike = scaled[10]
    assert np.isfinite(spike)
    assert spike < 0.0, "sign of the spike is preserved"
    assert abs(spike) < 1.0e3, f"spike must be tamed far below its 1e6 magnitude, got {spike:.3g}"
    stats = realized_scale_stats(w)
    assert stats["sigma_max"] > 100.0, "the large rescale is surfaced for the cross-arm audit"


def test_post_spike_suppression_stays_finite() -> None:
    """After a spike, sigma stays elevated ~1/beta steps, so ordinary rewards are suppressed — but FINITE.

    Pins the documented EMA behaviour (a spike transiently blinds the critic to small rewards) and proves the
    tiny/huge division never yields NaN/inf. It does NOT assert the suppression is desirable — that residual
    is exactly what the ``sigma_max`` log and the ``popart=False`` ablation exist to expose.
    """
    seq = [0.01] * 5 + [1.0e6] + [0.01] * 10
    scaled, _w = _run(seq)
    pre = abs(scaled[0])               # ~0.01 (sub-unit, min_scale identity)
    post = np.abs(scaled[6:])          # ordinary rewards right after the spike
    assert all(np.isfinite(post)), "post-spike scaled rewards must remain finite"
    assert post.max() < pre, "the elevated sigma suppresses ordinary rewards after a spike (documented)"
    assert post.max() < 1.0e-3, "suppression is strong (many orders) while the spike's EMA memory persists"
