"""PopArt-style value-target scale normalization for the fixed SB3-SAC critic (the engine gap).

Why this exists (the verified failure)
--------------------------------------
The reward is the *object of study* (H2) and is **never** rescaled in the data we analyse: every
realized-return number — selection fitness, the tail/EVT statistics, all inference — is read from
``info["port_ret"]`` (the realized portfolio return), NOT from the reward ``total`` the agent
maximizes (see ``src/env/runner.py``). But an LLM-authored reward is free to emit a ``total`` of
almost any *scale*: a Sharpe-style reward that divides a one-step return by a variance floored at
``1e-8`` produces a per-step target of order ``1e4`` (verified: prototype candidate ``scalar-g5-c3``
peaked at ``|total| = 1.146e4``). Feeding a target of magnitude ``R`` into SAC's Bellman backup
drives the Q-target toward ``R / (1 - gamma)`` — at ``gamma = 0.99`` that is ``~1e6`` — so the MSE
critic loss explodes to ``~5e6`` and the run diverges at the LAST training step
(``outputs/prototype/anomalies.jsonl``: ``critic_loss`` up to ``1.12e7``; ``critic_explosion`` logged
at step 25000). That is an *engine* pathology — a scale-sensitive learner — not a property of the
reward we want to study.

The principled fix (PopArt invariance, applied to the target only)
------------------------------------------------------------------
PopArt (van Hasselt et al., *Learning values across many orders of magnitude*, NeurIPS 2016) makes a
value learner invariant to the **scale** of its returns by learning in a *normalized* target space
while preserving the greedy policy. The exact, network-surgery form ("ART": adaptively rescale the
output layer so predictions are preserved when the statistics change) is fragile to bolt onto SB3's
compiled ``SAC.train`` loop. We implement the *intended* invariance the cleanest way
that does not touch SB3 internals: a running **scale** ``sigma`` (EMA of the reward's root second
moment) divides the reward the **critic** regresses on. Under a CONSTANT ``sigma`` the argument is
exact: SAC's critic target, its current-Q prediction, AND the actor's ``min_q`` objective all live in
that single scaled space, so dividing the reward by ``sigma`` rescales *every* Q by the same
``~1 / (sigma (1 - gamma))`` factor — a positive affine map of the value function whose argmax, hence
the optimal **policy**, is unchanged (van Hasselt et al. 2016; ``ent_coef="auto"`` re-adapts to the
normalized scale on its own). The IMPLEMENTED ``sigma``, however, is a running EMA (honest scope,
2026-07-03): it drifts during training, and each stored transition keeps the reward at its
COLLECTION-time scaling (SB3's replay buffer stores the scaled reward at ``add`` time), so the critic
regresses a mildly TIME-VARYING effective reward and the map is only **approximately**
policy-preserving — exactly policy-preserving in the settled-``sigma`` limit, and the exact IDENTITY
for a unit-scale reward (``min_scale=1`` clamp). The residual is not assumed away: the realised
per-candidate ``sigma_max``/``sigma_last`` are logged (R48) and the ``popart=False`` ablation
(``scripts/popart_ablation.py``) re-checks the H2 ordering with the wrapper off. What the wrapper buys
regardless is the *conditioning*: the regression target is held at unit scale, so the critic loss can
no longer reach ``1e6``.

Crucially this is **not** ``VecNormalize(norm_reward=True)`` and it does **not** alter the object of
study: only ``sigma`` divides the *learning* signal; ``info["port_ret"]`` (and the components dict) are
forwarded byte-for-byte, so ``EnvBundle.{val,train,test}_returns`` — every number that enters the
dissertation — are identical with or without this wrapper. We normalize **scale only** (no mean shift):
shifting the reward would change the entropy-vs-return trade-off and is unnecessary for the divergence,
which is purely a magnitude effect.

Design choices (each load-bearing)
----------------------------------
* **Scale, not standardize.** ``sigma = sqrt(EMA[r^2])`` (root second moment, not std). A floored
  variance pathology is a *magnitude* explosion; dividing by the RMS magnitude is exactly the quantity
  that tames it, and it leaves a genuinely tiny well-behaved reward (``sigma -> small``) untouched in
  *relative* terms (we clamp ``sigma`` below to avoid amplifying a near-zero reward into noise).
* **Deterministic, seed-free, replay-safe.** The statistic is a plain EMA over the rewards the agent
  actually sees, updated inside the env ``step``; it draws no RNG, so determinism/replay
  is preserved and the cross-arm architecture+hyperparameter equivalence (the arms differ only by seed
  and by the authored reward; see the T2.4 disclosure below for the effective-regularisation caveat) still
  holds — the wrapper is applied identically to every arm.
* **Bias-corrected EMA, scale from step 1 (NOT a warmup hold).** The scale must act from the FIRST
  reward: a warmup that passes early rewards through unscaled would let a ``1e4`` opening transition into
  the replay buffer, where it is resampled for many replay passes and re-explodes the critic (the buffer
  is a capped 50k FIFO window that still resamples an opening transition for ~17 passes before it is
  evicted at step 50k (budget-independent: the same at the frozen B*=400k as at the old 200k), ADR-025
  EXTENDED). We instead debias the EMA Adam-style — ``sq_hat = sq_ema / (1 - (1-beta)^count)``
  — so after a single reward ``v`` the estimate is exactly ``v^2`` and the scaled output is ``v/|v| = ±1``
  (perfectly normalized), not ``v/sqrt(beta*v^2)``. ``warmup`` therefore defaults to ``0``; it is kept as
  a knob only for ablations.

Disclosure — a residual SCALE-DEPENDENCE the design carries (T2.4)
-----------------------------------------------------------------
``min_scale`` clamps ``sigma >= 1.0`` (default), so this wrapper is the EXACT IDENTITY for a unit-scale
reward but a ``>1`` divisor for a large-magnitude one. This qualifies the cross-arm "fixed agent" claim,
which must be stated PRECISELY: what is held identical across the H2 arms is the agent **ARCHITECTURE and
hyperparameters** (the SB3-SAC network, ``learning_rate``, ``gamma``, ``ent_coef="auto"``, the PopArt knobs
— everything in :func:`resolve_agent_kwargs`); the arms differ only by seed AND by the authored reward. The
*effective entropy regularisation* is NOT held identical — it can vary with the authored reward magnitude —
so "fixed agent" means fixed architecture+hyperparameters, not a fixed effective regulariser. Two
consequences are worth stating plainly:

1. **Entropy regularisation re-adapts to the NORMALISED scale.** With ``ent_coef="auto"`` SAC tunes the
   temperature against the reward the critic sees, i.e. ``raw / sigma``. Two arms that author rewards of
   different *magnitude* are therefore regularised against different *normalised* scales unless their
   ``sigma`` coincide — a LATENT, scale-driven cross-arm difference inside an agent that is fixed in
   architecture and hyperparameters but whose effective entropy regularisation is reward-magnitude
   dependent. This is not a bug (PopArt is exactly what makes the critic scale-invariant; the alternative
   — letting a 1e4 reward explode the critic — is worse), but it IS a confound that must be *visible*.

2. **It is made auditable, not assumed away.** The exposure is bounded two ways, both of which REQUIRE the
   campaign run (not yet executed): (a) every training path logs the realised per-candidate scale
   (:meth:`PopArtRewardScaler.realized_scale_stats`: ``sigma_max``/``sigma_last``/``raw_rms_max``/
   ``raw_rms_last``/``count``, amendments R48 + P5) into the candidate AND test records. **NOTE (P5,
   2026-07-04): ``sigma_max`` audits only SUPRA-unit rescaling.** With ``min_scale=1.0`` it is pinned at
   ``1.0`` for ANY sub-unit reward (RMS <= 1), so ``sigma_max == 1.0`` across all arms means no arm was
   *shrunk* — NOT that the arms are scale-matched: two sub-unit arms of different magnitude both read
   ``sigma_max == 1.0`` while ``ent_coef="auto"`` adapts to their different raw scales. Since realized
   per-step rewards typically live in the sub-unit regime, the cross-arm scale-confound is audited on the
   UNCLAMPED ``raw_rms_max`` (whose spread quantifies the latent effective-regularisation difference), NOT
   on ``sigma_max``; (b) the ``popart=False`` ablation (``scripts/popart_ablation.py``) re-evaluates the
   FROZEN winners with PopArt off at one seed and confirms the H2 ordering is unchanged — but note (P5)
   that for a strictly sub-unit winner PopArt is the EXACT identity (``sigma == 1``), so ``raw / 1 == raw``
   bitwise and this ablation is a NO-OP there: it verifies that a *supra-unit* winner's ordering survives,
   while the ``raw_rms`` audit in (a) — not this ablation — is what surfaces the sub-unit-regime confound.
   Until the campaign runs, the cross-arm ``sigma_max``/``raw_rms_max`` tables and the ablation are
   SPECIFIED but UNEXECUTED.

This module is config-gated by ``popart`` (default on) and exercised by ``tests/test_popart.py``
(a large-magnitude reward no longer diverges the critic; ``port_ret`` is preserved exactly; the realised
``sigma_max``/``sigma_last`` are tracked and surfaced for the cross-arm audit).
"""
from __future__ import annotations

import warnings

from typing import Any

import numpy as np

import gymnasium as gym

__all__ = ["PopArtRewardScaler", "wrap_popart", "realized_scale_stats"]

# Largest |reward| whose square stays finite in float64 (``sqrt(max/2) ~= 9.5e153``). ``safe_call`` already
# rejects NON-finite rewards, but a FINITE reward of this magnitude would overflow ``raw*raw`` to ``inf`` in
# the second-moment EMA, pinning ``sigma`` at ``inf`` and silently zeroing the critic's learning signal for
# the rest of the candidate. The raw reward is clamped to this bound BEFORE squaring (a no-op for any real
# reward; counted + warned once when it fires) — see :meth:`PopArtRewardScaler.step`.
_RAW_CAP: float = float(np.sqrt(np.finfo(np.float64).max / 2.0))


class PopArtRewardScaler(gym.Wrapper):  # type: ignore[misc]
    """Divide the reward the agent learns on by an adaptive running **scale** (PopArt, scale-only).

    The wrapper sits between the :class:`~src.env.portfolio_env.PortfolioEnv` and the SAC learner. On
    every ``step`` it updates ``sigma = sqrt(EMA[r^2])`` from the *raw* reward and returns ``raw / sigma``
    as the learning signal, while forwarding the original ``info`` (so ``info["port_ret"]`` — the object
    of study — is unchanged). At CONSTANT ``sigma`` the transform is a positive scaling of the value
    function, leaving the optimal policy invariant (van Hasselt et al. 2016); the implemented EMA
    ``sigma`` drifts (buffered rewards keep their collection-time scaling), so it is APPROXIMATELY
    policy-preserving — see the module docstring's honest-scope note + the ``popart=False`` ablation —
    while the critic's target conditioning improves either way.

    Parameters
    ----------
    env : gym.Env
        The reward-bound ``PortfolioEnv`` to wrap.
    beta : float, default 1e-3
        EMA decay for the second-moment estimate ``EMA[r^2]`` (smaller = longer memory). ``1e-3``
        gives an effective horizon of ~1000 steps, matching the ``learning_starts`` warmup scale.
    epsilon : float, default 1e-8
        Numerical floor inside the square root.
    min_scale : float, default 1.0
        Lower clamp on ``sigma``. Clamping at ``1.0`` makes the wrapper a pure **shrink** of large
        rewards: a well-behaved reward already near unit scale is passed through essentially unchanged
        (``sigma`` never drops below 1, so a tiny reward is never *amplified* into noise), while a
        ``1e4``-scale reward is divided down to unit scale. This one-sided design keeps the common,
        already-conditioned case a no-op and only acts on the pathological large-magnitude case.
    warmup : int, default 0
        Number of initial rewards passed through unscaled (``sigma = 1``). Defaults to ``0`` — the
        bias-corrected EMA already gives a sound scale from the FIRST reward (``v -> ±1``), and any hold
        would leak a pathological opening reward into the (full-history) replay buffer. Kept for ablations.

    Attributes
    ----------
    sq_ema : float
        The (raw, un-debiased) running second-moment accumulator.
    count : int
        Number of rewards observed (drives the bias correction and the optional warmup gate).
    sigma_max : float
        The LARGEST realised scale ``sigma`` seen over the run (0.0 before any step). With the default
        ``min_scale=1.0`` this is ``1.0`` whenever the reward stayed SUB-UNIT (RMS <= 1) and ``>1`` only
        for a supra-unit reward the wrapper had to shrink. It therefore audits the SUPRA-unit rescaling
        only: ``sigma_max == 1.0`` across all arms means no arm was shrunk, NOT that the arms are
        scale-matched — two sub-unit arms of different magnitude both read ``sigma_max == 1.0`` while the
        critic (and ``ent_coef="auto"``) sees their different raw scales. The cross-arm
        entropy-regularisation confound (T2.4) is audited in the sub-unit regime by ``raw_rms_max`` below,
        NOT by ``sigma_max`` (P5, 2026-07-04).
    raw_rms_max : float
        The LARGEST UNCLAMPED bias-corrected RMS of the raw reward (0.0 before any step) — the effective
        scale ``ent_coef="auto"`` adapts to even when ``sigma`` is pinned at the ``min_scale`` floor. A
        spread in ``raw_rms_max`` across arms quantifies the latent effective-regularisation difference
        that ``sigma_max`` cannot see; it is the honest cross-arm scale-confound audit signal in the
        sub-unit regime. ``raw_rms_last`` is its end-of-training value.
    sigma_last : float
        The MOST RECENT realised scale (the scale in force at the end of training; ``0.0`` before any
        step). Together with ``sigma_max`` it bounds the per-candidate divisor the critic actually used.
    """

    def __init__(
        self,
        env: Any,
        beta: float = 1e-3,
        epsilon: float = 1e-8,
        min_scale: float = 1.0,
        warmup: int = 0,
    ) -> None:
        super().__init__(env)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.min_scale = float(min_scale)
        self.warmup = int(warmup)
        # Fail loud on a knob that would SILENTLY disable the scaler (2026-07-26 deep review). All four
        # are config-reachable (``popart_beta``/``popart_min_scale``/``popart_warmup``, via ``wrap_popart``),
        # and every out-of-range value degrades to the IDENTITY without ever raising: ``beta=0`` freezes
        # ``sq_ema`` at 0 so ``sigma`` pins at the floor; ``beta>1`` can drive ``sq_ema`` negative -> NaN
        # ``sigma`` -> ``_scale``'s backstop returns the floor; ``min_scale<=0`` voids BOTH the identity
        # guarantee and the div-by-zero floor. Each of those logs ``sigma_max == min_scale``, which is
        # INDISTINGUISHABLE from the healthy sub-unit reading the R48 cross-arm audit expects — a mistyped
        # knob would corrupt the audit instrument rather than fail. Reject at the boundary instead.
        # (NaN fails every comparison below, so it raises here too.)
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"popart beta must be in (0, 1]; got {self.beta!r}")
        if not (self.min_scale > 0.0 and np.isfinite(self.min_scale)):
            raise ValueError(f"popart min_scale must be finite and > 0; got {self.min_scale!r}")
        if not (self.epsilon > 0.0 and np.isfinite(self.epsilon)):
            raise ValueError(f"popart epsilon must be finite and > 0; got {self.epsilon!r}")
        if self.warmup < 0:
            raise ValueError(f"popart warmup must be >= 0; got {self.warmup!r}")
        self.sq_ema: float = 0.0
        self.count: int = 0
        # Realised-scale audit trail (T2.4): the cross-arm distribution of these is what makes the
        # "fixed-agent" claim (fixed architecture+hyperparameters; effective entropy regularisation may
        # vary with reward magnitude) falsifiable. Updated every step from the SAME sigma the critic divides by.
        self.sigma_max: float = 0.0
        self.sigma_last: float = 0.0
        # UNCLAMPED bias-corrected RMS of the raw reward — the effective scale the critic regresses on
        # BEFORE the ``min_scale`` clamp. Because ``min_scale=1.0`` pins ``sigma`` (hence ``sigma_max``) at
        # the floor for ANY sub-unit reward (RMS <= 1), ``sigma_max`` cannot distinguish two sub-unit arms
        # of different magnitude (e.g. RMS 1e-2 vs 5e-4 both read ``sigma_max=1.0`` while the critic sees a
        # 20x scale gap). ``raw_rms_max`` can, and is therefore the honest cross-arm scale-confound audit
        # signal in the sub-unit regime — the regime the realized per-step rewards typically live in (P5, 2026-07-04).
        self.raw_rms_max: float = 0.0
        self.raw_rms_last: float = 0.0
        # Fail-loud counter: number of steps whose raw reward was clamped to ``_RAW_CAP`` to prevent an
        # ``raw*raw`` overflow (0 on any well-behaved run; a non-zero value flags a pathological reward).
        self.raw_clamp_count: int = 0

    def _scale(self) -> float:
        """Current normalization scale ``sigma`` from the BIAS-CORRECTED second moment.

        ``1.0`` during the (default-empty) warmup; otherwise ``max(sqrt(sq_ema / (1-(1-beta)^count)), floor)``.
        The Adam-style debias makes the estimate exactly ``v^2`` after one reward ``v`` (so ``sigma=|v|`` and
        the scaled reward is ``±1``), removing the cold-start bias of a raw EMA.
        """
        if self.count < self.warmup or self.count == 0:
            return 1.0
        bias_correction = 1.0 - (1.0 - self.beta) ** self.count
        sq_hat = self.sq_ema / max(bias_correction, self.epsilon)
        sigma = max(float(np.sqrt(sq_hat + self.epsilon)), self.min_scale)
        # Backstop: a non-finite sigma (should be unreachable given the raw-reward clamp in ``step``) would
        # zero the critic signal; fall back to the floor rather than return ``inf``/``nan``.
        if not np.isfinite(sigma):
            return self.min_scale
        return sigma

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the inner env, update the running scale from the RAW reward, return ``raw / sigma``.

        ``info`` (hence ``info["port_ret"]`` and the logged components) is forwarded UNCHANGED — only
        the scalar the SAC critic regresses on is normalized.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = float(reward)
        # Guard (fail-loud): ``safe_call`` ensured ``raw`` is finite, but a finite ``|raw| >= _RAW_CAP``
        # (~9.5e153) would overflow ``raw*raw`` to ``inf``, pinning ``sigma`` at ``inf`` and silently training
        # the critic on a zeroed reward. Clamp before squaring — inert for any real reward — and count it loudly.
        if not -_RAW_CAP <= raw <= _RAW_CAP:
            self.raw_clamp_count += 1
            if self.raw_clamp_count == 1:
                warnings.warn(
                    f"PopArt: raw reward magnitude {abs(raw):.3e} exceeds the overflow-safe cap "
                    f"{_RAW_CAP:.3e}; clamping before squaring to keep the scale finite.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raw = float(np.clip(raw, -_RAW_CAP, _RAW_CAP))
            # ``np.clip`` maps ``+-inf`` to ``+-_RAW_CAP`` but leaves ``nan`` as ``nan`` (clip does NOT
            # sanitize NaN). A NaN reaching the EMA below would poison ``sq_ema`` IRREVERSIBLY (every later
            # ``(1-beta)*nan + ...`` stays NaN); ``_scale``'s backstop would then silently pin ``sigma`` at the
            # floor for the REST of the candidate (PopArt disabled, not recovered) while a NaN learning signal
            # leaks into the buffer on this step. ``safe_call`` already rejects non-finite rewards, so this is
            # defense-in-depth: map any non-finite residual to 0.0 (neutral, information-free) — never poison the
            # second moment, never propagate NaN. Already counted/warned above (fail-loud, same channel as overflow).
            if not np.isfinite(raw):
                raw = 0.0
        # Update the second-moment EMA from the raw reward BEFORE scaling (so the divisor reflects the
        # magnitude the critic would otherwise have to fit). Deterministic, no RNG. The scale (computed
        # below, AFTER this update) therefore already reflects the current reward -> a 1e4 opening reward
        # is normalized to ±1 on the SAME step, never entering the buffer at raw magnitude.
        self.count += 1
        self.sq_ema = (1.0 - self.beta) * self.sq_ema + self.beta * (raw * raw)
        sigma = self._scale()
        # Record the SAME sigma the critic divides by (computed once, reused) for the cross-arm audit (T2.4).
        self.sigma_last = sigma
        if sigma > self.sigma_max:
            self.sigma_max = sigma
        # Also track the UNCLAMPED bias-corrected RMS (the effective scale before the ``min_scale`` clamp):
        # for a sub-unit reward ``sigma`` is pinned at the floor and reveals nothing, but ``raw_rms`` still
        # varies with the authored magnitude, so it is the cross-arm scale-confound audit in the sub-unit
        # regime (P5). Recomputed here from the SAME bias-corrected second moment ``_scale`` uses so that
        # ``_scale`` itself stays byte-for-byte unchanged (no effect on the learning signal or dynamics).
        _bias = 1.0 - (1.0 - self.beta) ** self.count
        raw_rms = float(np.sqrt(max(self.sq_ema / max(_bias, self.epsilon), 0.0)))
        self.raw_rms_last = raw_rms
        if raw_rms > self.raw_rms_max:
            self.raw_rms_max = raw_rms
        scaled = raw / sigma
        return obs, float(scaled), bool(terminated), bool(truncated), info

    def realized_scale_stats(self) -> dict[str, float]:
        """The per-candidate realised-scale summary for the candidate/test record (T2.4 audit).

        Returns ``{"popart": 1.0, "sigma_max": ..., "sigma_last": ..., "raw_rms_max": ...,
        "raw_rms_last": ..., "count": ...}`` (JSON-friendly floats so it round-trips through the
        results-IO archive). ``sigma_max == 1.0`` (with the default ``min_scale=1.0``) means only that
        every reward was SUB-UNIT (RMS <= 1), i.e. the wrapper applied no SUPRA-unit shrink — it does
        **not** imply the arms are mutually scale-matched: two sub-unit arms of different magnitude both
        read ``sigma_max == 1.0`` while the critic (and ``ent_coef="auto"``) sees their different raw
        scales. The cross-arm scale-confound audit (P5) therefore compares ``raw_rms_max`` — the UNCLAMPED
        bias-corrected RMS, which is the effective scale the critic regresses on in the sub-unit regime —
        NOT ``sigma_max``. ``sigma_max > 1`` additionally quantifies any supra-unit shrink applied.
        """
        return {
            "popart": 1.0,
            "sigma_max": float(self.sigma_max),
            "sigma_last": float(self.sigma_last),
            "raw_rms_max": float(self.raw_rms_max),
            "raw_rms_last": float(self.raw_rms_last),
            "count": float(self.count),
        }

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        """Reset the inner env. The running scale PERSISTS across episodes (one continuing learner)."""
        return self.env.reset(**kwargs)


def wrap_popart(env: Any, cfg: Any) -> Any:
    """Wrap ``env`` in :class:`PopArtRewardScaler` using ``cfg`` knobs (no-op if ``popart`` is off).

    Reads (all optional, with the documented defaults): ``popart`` (bool, default True),
    ``popart_beta``, ``popart_min_scale``, ``popart_warmup``. Returns the bare ``env`` when ``popart``
    is False so the behaviour is identical to the pre-fix trainer for an explicit opt-out.
    """
    from src.utils.config import cfg_get

    if not bool(cfg_get(cfg, "popart", True)):
        return env
    return PopArtRewardScaler(
        env,
        beta=float(cfg_get(cfg, "popart_beta", 1e-3)),
        min_scale=float(cfg_get(cfg, "popart_min_scale", 1.0)),
        warmup=int(cfg_get(cfg, "popart_warmup", 0)),
    )


def realized_scale_stats(env: Any) -> dict[str, float] | None:
    """Realised-scale summary for whatever :func:`wrap_popart` returned (T2.4 audit logging).

    Returns the :meth:`PopArtRewardScaler.realized_scale_stats` dict when ``env`` is a PopArt scaler,
    or ``{"popart": 0.0}`` when PopArt was OFF (``wrap_popart`` returned the bare env). Either way the
    per-candidate record can carry an explicit, JSON-friendly marker of the scale the critic saw (or of
    the explicit opt-out) — never a silent ``None`` that an analyst could misread as "PopArt unverified".
    Robust to a future wrapper layered ON TOP of the scaler: it unwraps ``.env`` up to a small depth.
    """
    node = env
    for _ in range(8):  # bounded unwrap; the scaler sits directly on PortfolioEnv today (depth 0)
        if isinstance(node, PopArtRewardScaler):
            return node.realized_scale_stats()
        node = getattr(node, "env", None)
        if node is None:
            break
    return {"popart": 0.0}
