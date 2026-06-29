"""SAC trainer — builds the FIXED headline agent and trains it on a reward-bound env.

``make_agent_trainer(cfg, seed) -> (train_env -> policy)`` is the ``agent_trainer`` the
discovery loop (``src/llm/loop.py``) injects, and ``train_agent`` is the standalone
trainer the search-arm evaluator (``src/agents/evaluator.py``) uses. The agent is a
single SB3 SAC (Haarnoja et al. 2018; audit A-1) whose ARCHITECTURE and HYPERPARAMETERS
are held fixed across arms (see :func:`resolve_agent_kwargs`) so that performance
differences are attributable to the *reward*, not the learner. One precise caveat: with
``ent_coef="auto"`` and the PopArt scale-normalisation (fact 3 below), the *effective entropy
regularisation* can vary with the authored reward magnitude — so "fixed agent" means fixed
architecture+hyperparameters, not a fixed effective regulariser. This is made auditable by the
per-candidate ``sigma_max`` logging (``src/agents/popart.py``, amendment R48) and bounded by the
``popart=False`` ablation (``scripts/popart_ablation.py``); both the cross-arm ``sigma_max`` table
and the ablation REQUIRE the campaign run (not yet executed).

Two engineering facts, each load-bearing (MASTER_EXECUTION_PLAN §5, deep-research §2):

1. **Memory-safe replay buffer (ADR-025).** SB3's default ``buffer_size=1_000_000`` at
   the 1,893-dim observation needs ``1e6 x 1893 x 4 B x 2 ~= 15 GB`` of CPU RAM — it
   would OOM a 16 GB laptop. The buffer is sized to the (fixed, matched) per-candidate
   step budget instead (``buffer_size = train_steps``), ~0.76 GB at 50k
   (50000 x 1893 x 4 B x 2, the same obs+next_obs float32 allocation the 15 GB figure
   above assumes). We do **not** use ``optimize_memory_usage=True`` (its
   handle-timeout-termination caveat is unsafe at our split-boundary episode
   truncation — review m1).

2. **Train-only observation normalization (deep-research §2).** The 1,893-dim obs mixes
   returns (~1e-3), realized vol (~1e-2), lagged VIX (~10-80) and weights (0-1); SAC is
   sensitive to that scale spread. We wrap the *training* env in SB3 ``VecNormalize``
   (``norm_obs=True``), then **freeze** the running statistics and carry them on the
   returned :class:`NormalizedPolicy`, which re-applies the *same train-period* stats at
   evaluation time. We deliberately keep ``norm_reward=False`` — the reward is the object
   of study and must not be rescaled by the trainer (that would distort H2). Entropy
   auto-tuning (``ent_coef="auto"``) further adapts to the reward scale.

3. **PopArt-style value-target scale normalization (the engine gap; src/agents/popart.py).** An
   LLM-authored reward can emit a ``total`` of almost any scale (a Sharpe that divides a one-step
   return by a variance floored at ``1e-8`` peaks at ``|total| ~ 1e4``; verified prototype candidate
   ``scalar-g5-c3``). SAC's Bellman backup turns a target of magnitude ``R`` into a Q-target near
   ``R/(1-gamma) ~ 1e6`` (``gamma=0.99``) and the MSE critic loss explodes to ``~5e6`` at the LAST
   training step (``outputs/prototype/anomalies.jsonl``). We make the critic invariant to reward
   *scale* by dividing ONLY the learning signal by a running scale ``sqrt(EMA[r^2])`` — at constant
   scale a positive affine map of the value function, leaving the optimal policy unchanged (van
   Hasselt et al. 2016); the implemented EMA scale drifts (buffered rewards keep their
   collection-time scaling), so it is approximately policy-preserving — see ``src/agents/popart.py``'s
   honest-scope note + the ``popart=False`` ablation — while the regression target is held at unit
   scale. ``info["port_ret"]`` (the object of study) is
   forwarded UNCHANGED, so ``norm_reward`` stays False and every realized-return number we analyse is
   identical. Config-gated by ``popart`` (default on); applied identically across arms.

All behaviours are config-gated; ``normalize_obs`` and ``popart`` default on. ``learning_starts``
defaults to 1000 (the value the Phase-0 GATE validated; SB3's UNSET default is 100, so leaving it
unset would run a DIFFERENT warmup than the gate proved).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from src.agents.factory import campaign_replay_cap, make_headline_agent
from src.utils.config import cfg_get

__all__ = [
    "NormalizedPolicy",
    "resolve_agent_kwargs",
    "train_agent",
    "make_agent_trainer",
]

_LOG = logging.getLogger(__name__)


class NormalizedPolicy:
    """A trained SAC model plus the FROZEN train-period observation statistics.

    Re-applies the exact ``VecNormalize`` transform the policy was trained under
    (``clip((obs - mean) / sqrt(var + eps), -clip, clip)``) before delegating to the
    model, so evaluation observations are normalized with *train* statistics (no
    val-set leakage; deep-research §2). Exposes ``predict`` so it is a drop-in for a
    raw SB3 model in ``src.env.runner.rollout_port_returns``.
    """

    def __init__(
        self,
        model: Any,
        mean: np.ndarray,
        var: np.ndarray,
        epsilon: float,
        clip_obs: float,
        popart_scale: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.sqrt(np.asarray(var, dtype=np.float32) + float(epsilon))
        self._clip = float(clip_obs)
        #: T2.4 realised PopArt scale (sigma_max/sigma_last/count, or {"popart": 0.0} when off) the critic
        #: saw during training — copied into the candidate + test record for the cross-arm sigma audit.
        self.popart_scale = popart_scale

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        norm = (np.asarray(obs, dtype=np.float32) - self._mean) / self._std
        return np.clip(norm, -self._clip, self._clip)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]:
        """Normalize ``obs`` with the frozen train stats, then delegate to the model."""
        return self.model.predict(self._normalize(obs), deterministic=deterministic)


def resolve_agent_kwargs(cfg: Any, seed: int) -> tuple[dict[str, Any], int]:
    """Resolve the SAC kwargs (factory-supported keys) + the training-step budget.

    The buffer is sized to ``train_steps`` then HARD-CAPPED at the campaign replay cap
    (memory-safe, ADR-025 EXTENDED): ``buffer_size = min(requested_or_train_steps, cap)``.
    For any budget below the cap this is a no-op (buffer == budget); at B* >= 200k it clamps
    to the cap (default 50k) so no path OOMs the laptop. Hyperparameters are SB3 defaults
    unless overridden in ``cfg`` (kept IDENTICAL across feedback arms so only the reward
    differs — audit A-1 / the runtime equivalence test).
    """
    train_steps = int(
        cfg_get(cfg, "train_steps_per_candidate", cfg_get(cfg, "train_steps", 50000))
    )
    # min(requested-or-train_steps, cap): decouples the replay RAM from the step budget (ADR-025
    # EXTENDED, 2026-06-30). No-op for budgets < cap; clamps to 50k at B* >= 200k.
    buffer_size = min(int(cfg_get(cfg, "buffer_size", train_steps)), campaign_replay_cap())
    # SB3-SAC's UNSET default is learning_starts=100, but the Phase-0 GATE (scripts/smoke_test.py)
    # validated the agent with learning_starts=1000 — so the gate proved a DIFFERENT warmup than the
    # study ran (the gate validity bug). We resolve it here (config-driven, default 1000) so EVERY
    # training path (serial / SEARCH / TEST) trains under the GATED 1000-step warmup, and floor it at
    # 1000 so a config can never silently drop below the value the gate validated. (The critic-explosion
    # CURE is the PopArt scale-normalization in train_agent below; this fix aligns the warmup with the
    # gate and gives the buffer more pre-regression transitions.)
    learning_starts = max(int(cfg_get(cfg, "learning_starts", 1000)), 1000)
    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": float(cfg_get(cfg, "learning_rate", 3e-4)),
        "buffer_size": buffer_size,
        "batch_size": int(cfg_get(cfg, "batch_size", 256)),
        "learning_starts": learning_starts,
        "gamma": float(cfg_get(cfg, "gamma", 0.99)),
        "ent_coef": cfg_get(cfg, "ent_coef", "auto"),
        "seed": int(seed),
        "verbose": int(cfg_get(cfg, "verbose", 0)),
        "device": cfg_get(cfg, "device", "auto"),  # "cpu" | "cuda" | "auto" (heterogeneous pool)
    }
    return kwargs, train_steps


def _training_substitution_counts() -> tuple[int, int]:
    """Read (and loudly surface) the sandbox SAFE_DEFAULT substitution counters after ``model.learn``.

    WHY (training-time SAFE_DEFAULT visibility, 2026-07-03): during training the env invokes the
    reward via ``safe_call`` on EVERY step, so the executor's counters accumulate exactly this
    candidate's training-window substitutions (``train_agent`` resets them just before ``learn``).
    They used to be zeroed UNREAD by the first evaluation rollout's ``reset_failure_flag()`` — a
    reward that failed on a fraction of TRAINING steps (the agent then part-trains on the 0.0
    fallback signal) was invisible. Returns ``(n_substituted, n_calls)``; the caller attaches them
    to the returned policy so the loop / parallel worker can archive them (R66).
    """
    from src.sandbox.executor import safe_call_count, safe_default_count

    n_sd = safe_default_count()
    n_call = safe_call_count()
    if n_sd:
        _LOG.warning(
            "reward substituted SAFE_DEFAULT on %d/%d TRAINING steps (%.1f%%) — the candidate "
            "part-trained on the 0.0 fallback signal; archived as train_safe_default_count (R66)",
            n_sd,
            n_call,
            100.0 * n_sd / max(n_call, 1),
        )
    return n_sd, n_call


def _apply_tf32(enabled: bool) -> None:
    """Set float32 matmul precision on the (process-global) torch backends, idempotently.

    Called by :func:`train_agent` so the SAME setting applies in EVERY training path — the serial
    trainer, the SEARCH worker (``parallel.train_candidate``), and the parallel TEST worker — preventing
    a SEARCH-vs-TEST numerics asymmetry (the frozen winner must be EVALUATED under the precision it was
    SELECTED under; TF32 is active on Ampere/Ada GPUs). Torch absent / no CUDA -> a safe no-op.
    """
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    except Exception:  # noqa: BLE001
        pass


def _make_curve_recorder(record_every: int) -> Any:
    """A READ-ONLY SB3 callback sampling the training curve into ``.curve`` (M2 — the F9 data).

    Every ``record_every`` env steps it appends ``(step, critic_loss, actor_loss, ent_coef, rollout
    return)`` read from the SB3 logger + episode buffer. It draws NO randomness and mutates neither the
    model nor the env, so training is bit-identical whether it is attached or not (the determinism
    envelope is untouched; verified by the golden reproduction). ~``train_steps/record_every`` points, so
    the archived curve is a few hundred floats — a frozen, replay-only campaign can only plot what it logs.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _TrainCurveRecorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.record_every = max(1, int(record_every))
            self._t0 = time.monotonic()   # heartbeat only; never read by training
            self.curve: dict[str, list[float]] = {
                "step": [], "critic_loss": [], "actor_loss": [], "ent_coef": [], "return": [],
            }

        @staticmethod
        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _on_step(self) -> bool:  # read-only hook; always returns True (never halts training)
            if self.num_timesteps % self.record_every == 0:
                nv = getattr(self.model.logger, "name_to_value", {}) or {}
                ep = getattr(self.model, "ep_info_buffer", None)
                self.curve["step"].append(int(self.num_timesteps))
                self.curve["critic_loss"].append(self._f(nv.get("train/critic_loss")))
                self.curve["actor_loss"].append(self._f(nv.get("train/actor_loss")))
                self.curve["ent_coef"].append(self._f(nv.get("train/ent_coef")))
                self.curve["return"].append(
                    float(np.mean([e.get("r", float("nan")) for e in ep])) if ep else float("nan")
                )
                # ── PROGRESS HEARTBEAT (2026-07-27) ────────────────────────────────────────────
                # A cluster training was a BLACK BOX: `verbose: 0` and nothing else writes to
                # stdout, so a job's log stayed 0 bytes from dispatch until it finished. Live on
                # 2026-07-27 that made "slow" and "stuck" indistinguishable for an hour on the very
                # first CPU-lane run — and the campaign's trainings are ~8.5 h each, 42,128 of them.
                # Emitted from the EXISTING read-only recorder at its EXISTING cadence, so no new
                # callback, no new hook frequency, and nothing new in the determinism envelope:
                # this branch draws no randomness and touches neither model nor env.
                # ``flush=True`` is load-bearing — stdout is block-buffered when redirected to the
                # jobscript's log file, so an unflushed print would be invisible for hours, which
                # is the same block-buffering trap that hung the 2026-07-02 ladder launch.
                el = time.monotonic() - self._t0
                print(f"[train] step {int(self.num_timesteps)}/{int(self.locals.get('total_timesteps') or 0)} "
                      f"elapsed {el:.0f}s rate {(self.num_timesteps / el if el > 0 else 0):.1f} steps/s",
                      flush=True)
            return True

    return _TrainCurveRecorder()


def train_agent(env: Any, cfg: Any, seed: int, callback: Any = None) -> Any:
    """Build the fixed SAC agent and train it on ``env`` for the fixed step budget.

    Parameters
    ----------
    env : Any
        A reward-bound ``PortfolioEnv`` (the TRAIN-window env from ``EnvBundle``).
    cfg : Any
        Agent/training config (``train_steps_per_candidate``, ``buffer_size``,
        ``normalize_obs``, SB3 hyperparameters, ...).
    seed : int
        The run seed (determinism; ``src/utils/seeding.py`` seeds the wider stack).
    callback : Any, optional
        An SB3 ``BaseCallback`` (e.g. from ``src.utils.monitoring.make_training_callback``) streaming
        per-step training metrics to the live monitor. ``None`` (default) trains with no callback.

    Returns
    -------
    Any
        A :class:`NormalizedPolicy` when ``normalize_obs`` is on (default), else the raw
        SB3 model. Both expose ``predict(obs, deterministic=True)``.
    """
    kwargs, train_steps = resolve_agent_kwargs(cfg, seed)
    # Float32 matmul precision is set HERE (not as a per-scheduler side effect) so EVERY training path —
    # the serial trainer, the SEARCH worker, and the parallel TEST worker — selects AND evaluates the
    # fixed agent under IDENTICAL numerics (config key ``tf32``, default on for Ampere/Ada speed).
    _apply_tf32(bool(cfg_get(cfg, "tf32", True)))
    normalize_obs = bool(cfg_get(cfg, "normalize_obs", True))

    # PopArt-style value-target SCALE normalization (src/agents/popart.py): make the SAC critic
    # invariant to the reward's SCALE so a large-magnitude LLM reward (a variance-floored Sharpe can emit
    # |reward| ~ 1e4) cannot drive the Q-target to ~1e4/(1-gamma) ~ 1e6 and explode the critic loss. This
    # divides ONLY the learning signal by a running scale; info["port_ret"] (the object of study) is
    # forwarded unchanged, so norm_reward stays False and every realized-return number we analyse is
    # identical. The wrapper sits INSIDE VecNormalize (it wraps the PortfolioEnv; VecNormalize then
    # normalizes observations on top), and is applied identically across arms (not an H2 confound).
    from src.agents.popart import realized_scale_stats, wrap_popart

    # Training-time SAFE_DEFAULT visibility (2026-07-03): zero the executor's substitution counters so
    # the post-learn read below spans exactly THIS candidate's training window (the previous window's
    # counts were already consumed by its own end-of-window reader — runner rollout / prior train).
    from src.sandbox.executor import reset_failure_flag

    reset_failure_flag()

    # M2 (2026-07-26): attach a READ-ONLY training-curve recorder so F9's learning curves have data to
    # plot. Config-gated (``capture_train_curve``, default on); combined with any passed monitor callback
    # via CallbackList. Determinism-safe — it only reads the SB3 logger, so training is bit-identical
    # whether it is attached or not (verified by the golden reproduction).
    curve_rec = (
        _make_curve_recorder(max(1, train_steps // 80))
        if bool(cfg_get(cfg, "capture_train_curve", True))
        else None
    )
    if curve_rec is None:
        learn_cb = callback
    elif callback is None:
        learn_cb = curve_rec
    else:
        from stable_baselines3.common.callbacks import CallbackList

        learn_cb = CallbackList([callback, curve_rec])

    if not normalize_obs:
        popart_env = wrap_popart(env, cfg)
        model = make_headline_agent(popart_env, kwargs)
        model.learn(total_timesteps=train_steps, callback=learn_cb)
        n_sd, n_call = _training_substitution_counts()  # read BEFORE the eval rollouts re-zero them
        # T2.4: surface the realised PopArt scale (sigma_max/last) the critic actually used so the
        # cross-arm sigma distribution is auditable. Attached to the returned policy; the loop / TEST
        # worker copy it into the candidate + test record. No effect on training or on info["port_ret"].
        model.popart_scale = realized_scale_stats(popart_env)  # type: ignore[attr-defined]
        # Training-window SAFE_DEFAULT counts (R66): attached so the caller can archive them.
        model.train_safe_default_count = n_sd  # type: ignore[attr-defined]
        model.train_safe_call_count = n_call  # type: ignore[attr-defined]
        if curve_rec is not None:
            model.train_curve = curve_rec.curve  # type: ignore[attr-defined]  # M2 learning-curve data
        return model

    # Train-only observation normalization (deep-research §2). norm_reward=False: the
    # reward is the object of study and must not be rescaled by the trainer.
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    clip_obs = float(cfg_get(cfg, "clip_obs", 10.0))
    wrapped = wrap_popart(env, cfg)
    venv = VecNormalize(
        DummyVecEnv([lambda: wrapped]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=clip_obs,
        training=True,
    )
    model = make_headline_agent(venv, kwargs)
    model.learn(total_timesteps=train_steps, callback=learn_cb)
    n_sd, n_call = _training_substitution_counts()  # read BEFORE the eval rollouts re-zero them

    rms: Any = venv.obs_rms  # frozen train-period running mean/var (single-obs RunningMeanStd at runtime)
    policy = NormalizedPolicy(
        model=model,
        mean=rms.mean,
        var=rms.var,
        epsilon=float(venv.epsilon),
        clip_obs=clip_obs,
        # T2.4: ``wrapped`` is the PopArt scaler (or the bare env when popart is off); read the realised
        # scale it logged during training so the candidate/test record can carry the cross-arm sigma audit.
        popart_scale=realized_scale_stats(wrapped),
    )
    # Training-window SAFE_DEFAULT counts (R66, 2026-07-03): attached (same style as the popart attach on
    # the raw-model branch) so the loop / parallel worker can archive them with the candidate record.
    policy.train_safe_default_count = n_sd  # type: ignore[attr-defined]
    policy.train_safe_call_count = n_call  # type: ignore[attr-defined]
    if curve_rec is not None:
        policy.train_curve = curve_rec.curve  # type: ignore[attr-defined]  # M2 learning-curve data
    return policy


def _make_governor(cfg: Any) -> Any:
    """Build a thermal governor from ``cfg['thermal_guardian'] = {hi, lo, poll_secs}``; ``None`` when absent.

    Default OFF -> result-neutral (no behaviour change). On a 24/7 laptop campaign, enable it so training
    cooperatively pauses-and-cools on GPU overheat instead of throttling/aborting (the pause changes only
    wall-clock, never a weight, so determinism and every reported number are untouched)."""
    spec = cfg_get(cfg, "thermal_guardian", None)
    if not spec:
        return None
    try:
        from src.utils.guardian import ThermalGovernor

        kw = spec if isinstance(spec, dict) else {}
        return ThermalGovernor(
            hi=float(kw.get("hi", 88.0)),
            lo=float(kw.get("lo", 80.0)),
            poll_secs=float(kw.get("poll_secs", 5.0)),
        )
    except Exception:  # pragma: no cover - guardian/NVML absent -> silently disabled
        return None


def make_agent_trainer(cfg: Any, seed: int, monitor: Any = None) -> Callable[[Any], Any]:
    """Return the ``agent_trainer(train_env) -> policy`` closure the loop injects.

    Binds ``cfg`` + ``seed`` so the discovery loop only has to pass the train env. When ``monitor`` is
    given, a FRESH SB3 monitoring callback (per-step metrics + anomaly detection) is built for EACH
    candidate's training and streamed to the live :class:`src.utils.monitoring.RunMonitor`.
    """

    governor = _make_governor(cfg)  # config-gated thermal pause (None = off, result-neutral)

    def _trainer(train_env: Any) -> Any:
        callback = None
        if monitor is not None or governor is not None:
            from src.utils.monitoring import make_training_callback

            callback = make_training_callback(monitor, governor=governor)
        return train_agent(train_env, cfg, seed, callback=callback)

    return _trainer
