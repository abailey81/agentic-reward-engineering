"""LLM-driven reward-discovery evolutionary loop (FINAL_PLAN F.8, B.3).

Purpose
-------
The core discovery loop. For a given experimental *arm* it prompts an LLM to
propose reward functions, validates and trains them, scores their held-out
fitness, and reflects across generations -- accumulating proposals into a
:class:`CandidateArchive`. The loop is the shared machinery; arms differ only in
the prompt's serialized "arm block" (see :func:`src.feedback.schema.build_block`).

Dependency injection
--------------------
Every heavy collaborator is injected so the loop is unit-testable WITHOUT real
training, a real LLM, or torch installed:

    - ``llm``           : has ``complete(system, user) -> str`` (the reward source).
    - ``agent_trainer`` : ``(env) -> policy`` (a trained policy; fake in tests).
    - ``measurement``   : factory of a fresh distribution estimator (e.g.
                          ``ReturnDistribution``) exposing ``.fit(returns)`` and
                          ``.tail_stats() -> dict``.
    - ``fitness_fn``    : ``(val_returns) -> float`` (e.g. ``held_out_fitness``).
    - ``env_builder``   : ``(reward_fn) -> env`` where ``env`` exposes
                          ``train_env()`` and ``val_returns(policy) -> np.ndarray``.

Algorithm (FINAL_PLAN B.3) -- runs ONCE per call
-------------------------------------------------
For each generation::

    build prompt  (system + initial for gen0, else reflection carrying the arm's
                   feedback block from schema.build_block)
    src  = llm.complete(system, user)
    fn   = validate_once(src, fixture)          # skip + log on SandboxError
    pol  = agent_trainer(env_builder(fn).train_env())
    valr = env.val_returns(pol)                 # realized VALIDATION returns
    fit  = fitness_fn(valr)                      # validation Deflated Sharpe
    dist = measurement().fit(train_returns)      # measure TRAINING realized returns
    tail = dist.tail_stats()
    block = schema.build_block(arm, fit, tail)   # next feedback block
    write_run(record, archive_root)              # archive (replay, audit C-2)
    reflect: carry block into the next generation's prompt

Single-shot semantics
----------------------
A single-shot arm spends the WHOLE matched budget in ONE generation
(``generations == 1``); multi-generation arms split the same budget across
generations. The total budget is matched across arms (FINAL_PLAN F.9). The
winner is the candidate with the best validation fitness.

Tests (tests/test_loop.py)
--------------------------
    - a 2-candidate / 1-gen loop archives reloadable artifacts (load_run round-trips).
    - the reflection prompt for the distributional arm CONTAINS the tail-stat lines
      (and the scalar arm does NOT).
    - a candidate whose source fails the sandbox gate is logged and SKIPPED.
    - single-shot draws the full budget in one generation.
    - the winner is the best-fitness candidate.
"""

from __future__ import annotations


import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.feedback import schema
from src.sandbox.executor import (
    SandboxEnvironmentError,
    SandboxError,
    extract_reward_source,
    validate_once,
)
from src.utils.config import cfg_get

__all__ = ["CandidateRecord", "CandidateArchive", "run_loop"]

_LOG = logging.getLogger(__name__)


class _NoMon:
    """No-op monitor: every ``monitor.<m>(...)`` call is a safe no-op when no RunMonitor is injected
    (unit tests, offline runs), so ``run_loop`` can call the monitor unconditionally."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


_NULL_MONITOR = _NoMon()

#: System prompt used for every generation (arm-agnostic).
_SYSTEM_PROMPT = (
    "You are a reward-function engineer for a risk-sensitive portfolio RL agent. "
    "Return Python source defining `reward(weights, returns, prev_weights, "
    "port_ret, info)` using numpy only."
)

#: Initial (generation-0) user prompt; no prior feedback exists yet.
_INITIAL_PROMPT = (
    "Propose an initial reward function for the portfolio agent. "
    "Use numpy only and obey the reward contract."
)

#: Reflection-prompt preamble; the arm's feedback block is appended below it.
_REFLECTION_PREAMBLE = (
    "Reflect on the previous candidate's results and propose an improved reward "
    "function. Feedback from the previous candidate:"
)


@dataclass
class CandidateRecord:
    """One accepted candidate produced by :func:`run_loop`.

    Attributes
    ----------
    prompt : str
        The full rendered user prompt that produced this candidate.
    reward_source : str
        The raw reward source returned by the LLM.
    reward_hash : str
        SHA-256 hex digest of ``reward_source`` (provenance / dedup).
    feedback_block : str
        The serialized arm feedback block built from THIS candidate's results
        (carried forward into the next generation's reflection prompt).
    val_fitness : float
        Held-out (validation) fitness used for winner selection.
    tail_stats : dict
        The frozen tail-diagnostic set measured on TRAINING realized returns.
    generation : int
        The 0-based generation that produced this candidate.
    candidate_id : str
        Unique id within the run (``"{arm}-g{gen}-c{idx}"``).
    popart_scale : dict or None
        The realised PopArt scale the SAC critic saw while training THIS candidate (T2.4):
        ``{"popart": 1.0, "sigma_max", "sigma_last", "count"}`` when PopArt is on, ``{"popart": 0.0}``
        when off, or ``None`` if the trainer did not surface it (e.g. a fake test trainer). Logged so the
        CROSS-ARM ``sigma`` distribution is auditable — a unit ``sigma_max`` across arms shows the
        "fixed-agent" design carries no latent, scale-driven entropy-regularisation difference.
    """

    prompt: str
    reward_source: str
    reward_hash: str
    feedback_block: str
    val_fitness: float
    tail_stats: dict
    generation: int
    candidate_id: str
    popart_scale: dict | None = None


@dataclass
class CandidateArchive:
    """Accumulated per-candidate records produced by one :func:`run_loop` call.

    Attributes
    ----------
    arm : str
        Name of the experimental arm that produced these candidates.
    candidates : list[CandidateRecord]
        Accepted candidate records, in generation/draw order.
    failures : list[dict]
        Logged validation failures that were skipped (audit trail); each holds
        ``generation``, ``reward_source``, and ``error``.
    meta : dict
        Run-level metadata (budget spent, generations, seeds, model id, etc.).

    Notes
    -----
    The winning candidate is the accepted record with the highest
    :attr:`CandidateRecord.val_fitness`; see :meth:`winner`.
    """

    arm: str
    candidates: list[CandidateRecord] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def winner(self) -> CandidateRecord | None:
        """Return the accepted candidate with the best validation fitness.

        Returns
        -------
        CandidateRecord or None
            The best-fitness candidate, or ``None`` if none were accepted.
        """
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.val_fitness)


def _reward_hash(src: str) -> str:
    """Stable SHA-256 hex digest of reward source."""
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _diversity_directive(cidx: int, n: int) -> str:
    """A per-candidate exploration directive giving within-generation diversity by PROMPT VARIATION.

    Eureka samples ``n`` candidates per generation from the SAME prompt and relies on sampling
    ``temperature`` for variety; a reward-author that rejects the ``temperature`` parameter
    (e.g. Claude Opus 5) needs the variety injected another way. Appending a distinct directive
    per candidate index does that. The directive set is IDENTICAL across arms AND (R38 de-seed)
    names NO specific risk statistic — it asks the LLM to vary WHICH statistics it tracks, not to use
    CVaR/drawdown — so it neither differentially favours an arm nor pre-seeds the tail to the
    non-distributional arms; only the arm's feedback block introduces the tail. Also reused by the
    parallel scheduler (``src/orchestration/parallel.py``) so both run paths diversify identically.
    """
    return (
        f"[Exploration directive {cidx + 1}/{n}: propose a reward DISTINCT from the other "
        f"candidates this generation — vary which statistics of the return history you track, the "
        f"rolling window, and the functional form. Do not reuse a design you would give a different "
        f"candidate index.]"
    )


def _budget_for_generation(total_budget: int, generations: int, gen: int) -> int:
    """Split a matched total budget evenly across generations.

    Single-shot arms (``generations == 1``) get the WHOLE budget in one
    generation; multi-generation arms split it, with any remainder front-loaded.

    NOTE (F15, review 2026-07-02): :func:`run_loop` no longer accrues ``budget_spent`` from this
    PLAN — it accrues the actual draws (``candidates_per_gen`` per generation), which the plan
    over-stated whenever ``budget % generations != 0``. This helper remains the documented even-split
    planner (DEEP_H3 §"budget") for callers sizing a generation schedule.
    """
    base = total_budget // generations
    remainder = total_budget % generations
    return base + (1 if gen < remainder else 0)


def run_loop(
    arm: str,
    env_builder: Any,
    llm: Any,
    agent_trainer: Any,
    measurement: Any,
    fitness_fn: Any,
    cfg: Any,
    archive_root: Any,
) -> CandidateArchive:
    """Run the LLM reward-discovery loop once for one arm (FINAL_PLAN B.3).

    Parameters
    ----------
    arm : str
        Experimental arm name; selects the serialized feedback block
        (``"distributional"``, ``"scalar"``, ``"placebo"``, ``"scalar_cvar5"``).
    env_builder : Any
        Callable ``(reward_fn) -> env``; ``env`` exposes ``train_env()`` and
        ``val_returns(policy) -> np.ndarray``.
    llm : Any
        An ``LLMClient`` (or compatible) with ``complete(system, user) -> str``.
    agent_trainer : Any
        Callable ``(train_env) -> policy`` returning a trained policy.
    measurement : Any
        Zero-arg factory of a fresh distribution estimator exposing
        ``.fit(returns)`` and ``.tail_stats() -> dict`` (e.g. ``ReturnDistribution``).
    fitness_fn : Any
        Callable ``(val_returns) -> float`` (e.g. ``held_out_fitness``).
    cfg : Any
        Configuration carrying ``generations`` (default 1), ``candidates_per_gen``
        (default 1), ``budget`` (matched total; default 0), ``seed``, ``n_trials``,
        ``model`` (id), and ``run_prefix``.
    archive_root : Any
        Directory passed to :func:`src.io.results.write_run` for each candidate.

    Returns
    -------
    CandidateArchive
        The accumulated, reloadable archive of candidates and artifacts. The
        winner is :meth:`CandidateArchive.winner`.

    Notes
    -----
    Runs ONCE; single-shot arms (``generations == 1``) spend the whole matched
    budget in one generation. Invalid candidates are logged in ``failures`` and
    SKIPPED -- they never crash the loop (audit A-5). Every accepted candidate is
    archived via ``write_run`` so results replay rather than regenerate (C-2).
    """
    from src.io.results import write_run

    generations = int(cfg_get(cfg, "generations", 1))
    candidates_per_gen = int(cfg_get(cfg, "candidates_per_gen", 1))
    total_budget = int(cfg_get(cfg, "budget", 0))
    seed = cfg_get(cfg, "seed", 0)
    n_trials = int(cfg_get(cfg, "n_trials", 1))
    model_id = cfg_get(cfg, "model", cfg_get(getattr(llm, "cfg", None), "model", ""))
    run_prefix = cfg_get(cfg, "run_prefix", "run")
    env_fingerprint = cfg_get(cfg, "env_fingerprint", "injected")
    # Within-generation diversity by per-candidate PROMPT VARIATION (uniform across arms) — needed
    # when the reward-author rejects the ``temperature`` parameter (e.g. Claude Opus 5). Off by
    # default (temperature-honoring models like Gemini get diversity from sampling instead).
    diversity = bool(cfg_get(cfg, "diversity_prompt_variation", False))
    # Report-only mechanism sub-experiments (ADR-039), both DEFAULT-OFF so the frozen campaign path is
    # byte-identical: ``legible_render`` renders the distributional tail block in basis-points/decile form
    # (numeracy-bottleneck leg); ``extra_record_fields`` stamps driver-supplied discriminators
    # (``label``/``condition``) onto each archived record so analyze_campaign can split the named/legible legs.
    legible_render = bool(cfg_get(cfg, "legible_render", False))
    extra_record_fields = dict(cfg_get(cfg, "extra_record_fields", {}) or {})
    monitor = cfg_get(cfg, "monitor", None) or _NULL_MONITOR  # RunMonitor; no-op when absent (tests/offline)

    # ── Search-replay cache (resume) ──────────────────────────────────────────────────
    # On --resume a previously-archived candidate is REPLAYED from disk instead of re-calling
    # the (paid, non-deterministic) LLM and retraining: a mid-search resume becomes byte-faithful
    # to the original run (same candidates -> same winner) AND saves the Opus spend + GPU time.
    # Default OFF -> a fresh run is byte-for-byte unchanged. Successes reload via load_run(run_id);
    # sandbox FAILURES replay from a sibling failures-ledger, so a previously-rejected candidate is
    # NOT re-generated (which, against a non-deterministic author, could newly SUCCEED and silently
    # change the candidate set / winner).
    import json as _json
    from pathlib import Path as _Path

    from src.io.results import load_run as _load_run  # local import (mirrors the write_run import)

    resume = bool(cfg_get(cfg, "resume", False))
    _fail_ledger = _Path(archive_root) / f"{run_prefix}-{arm}.failures.jsonl"
    _cached_failures: dict[str, dict] = {}
    if resume and _fail_ledger.is_file():
        for _ln in _fail_ledger.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if not _ln:
                continue
            try:  # a torn last line just means that one candidate regenerates — never crash resume
                _f = _json.loads(_ln)
                _cached_failures[str(_f["candidate_id"])] = _f
            except Exception:
                pass

    # C3 (ADR-029): when the orchestrator supplies rendered prompts (system + initial with the
    # env interface filled, src/llm/prompts.py), use them; else fall back to the built-in minimal
    # prompts so unit tests need no prompt files. The REFLECTION body is composed from the arm's
    # feedback block (schema.build_block) either way — that block is the only thing that differs
    # across the five LLM arms and is what carries the tail diagnostics for the distributional arm.
    _prompts = cfg_get(cfg, "prompts", None)
    if _prompts is None:
        system_prompt, initial_prompt = _SYSTEM_PROMPT, _INITIAL_PROMPT
    else:
        system_prompt = _prompts["system"] if isinstance(_prompts, dict) else _prompts.system
        initial_prompt = _prompts["initial"] if isinstance(_prompts, dict) else _prompts.initial

    archive = CandidateArchive(arm=arm)
    archive.meta = {
        "generations": generations,
        "candidates_per_gen": candidates_per_gen,
        "budget": total_budget,
        "seed": seed,
        "model": model_id,
    }

    # Fixture passed to validate_once (anonymized arrays + scalar + info dict).
    # Real-ish per-step shapes (final-audit #12): a realistic asset count so a reward whose
    # allocation scales with the input surfaces at validate_once (under the Linux child's RLIMIT_AS)
    # rather than only at training time, where safe_call has no rlimit/timeout.
    # SHAPE PARITY (review batch 3 #1, 2026-07-03): the old equal-length arrays ("contract-
    # agnostic") INVERTED the gate — production calls reward(weights(N+1), returns(N), prev(N+1))
    # (portfolio_env.py:347-348) and the FROZEN prompt promises those exact shapes, so the
    # spec-faithful `weights[:-1] @ returns` was falsely REJECTED here while the sloppy
    # `weights @ returns` passed and then zero-trained via SAFE_DEFAULT. The fixture now mirrors
    # the production contract: returns has ONE FEWER element than weights/prev_weights.
    # DEGENERACY PARITY (audit 2026-07-19): non-degenerate returns + a production-shaped info dict. A
    # constant all-positive returns vector + empty info falsely rejected valid downside/tail/stateful
    # rewards at validate_once (empty left-tail subset, ÷0 std, or KeyError on info["reward_state"]) —
    # none of which occur in production (portfolio_env.py:323-327 always supplies the three info keys).
    _n_fix = max(2, int(cfg_get(cfg, "fixture_n_assets", 31)))
    _fix_w = np.full(_n_fix, 1.0 / _n_fix, dtype=float)
    _fix_prev = np.full(_n_fix, 1.0 / _n_fix, dtype=float)
    fixture: tuple = (
        _fix_w,                                           # weights (simplex over ~30 risky + cash)
        ((np.arange(_n_fix - 1) % 5) - 2) * 0.01,         # returns (N = n_act - 1): mixed-sign, non-zero var
        _fix_prev,                                        # prev_weights
        0.002,                                            # port_ret (non-zero)
        {"weights": _fix_w, "prev_weights": _fix_prev, "reward_state": None},  # info mirrors portfolio_env.py:323-327
    )

    # The feedback block fed into the next generation's reflection prompt. None at
    # generation 0 -> the initial prompt is used; reflection thereafter.
    prev_feedback_block: str | None = None
    budget_spent = 0
    # C3c (ops audit 2026-07-02): count the LLM-error SKIPS (API exhaustion / unparseable response,
    # the `continue` below). These slots are neither archived records nor ledgered sandbox failures,
    # so the arm's resolved pool is silently SHORT — the campaign's winner-selection floor (C3b,
    # run_campaign) needs the count surfaced in the run summary to distinguish "search incomplete,
    # --resume re-attempts these slots" from a genuinely exhausted budget.
    llm_error_skips = 0
    _failover_announced = False  # emit the api_key_failover anomaly ONCE per loop (2026-07-20)

    for gen in range(generations):
        # F15 (review 2026-07-02): accrue budget_spent from the ACTUAL draws. Every slot of the
        # cidx loop below consumes exactly one author draw (accepted, sandbox-rejected, llm-error
        # skipped, or resume-replayed), so the truthful spend is candidates_per_gen per generation —
        # NOT ``_budget_for_generation``'s even split of the cfg ``budget``, which silently
        # over-reported whenever budget % generations != 0 (callers floor candidates_per_gen =
        # budget // generations, so the remainder slots are never drawn). ``archive.meta`` keeps
        # ``budget`` as the configured plan and ``budget_spent`` as what actually happened. The M2/
        # ADR-026 once-per-generation accrual (not per-candidate) is preserved; only WHAT is accrued
        # changed.
        budget_spent += candidates_per_gen

        # 1. Build the prompt: initial at gen 0, else reflection with the arm block.
        if prev_feedback_block is None:
            user_prompt = initial_prompt
        else:
            user_prompt = f"{_REFLECTION_PREAMBLE}\n{prev_feedback_block}"

        # M5: reflect on the generation's BEST candidate (Eureka-faithful + parity with the parallel
        # path), not the last. Track the best WITHIN this generation; seed the next prompt at the boundary.
        gen_best_fitness: float | None = None
        gen_best_block: str | None = None

        for cidx in range(candidates_per_gen):
            candidate_id = f"{arm}-g{gen}-c{cidx}"
            cand_n = gen * candidates_per_gen + cidx  # 0-based candidate index WITHIN the arm
            monitor.candidate_start(arm, cand_n, gen)
            cand_t0 = time.perf_counter()

            # Search-replay cache: on resume, REPLAY an already-archived candidate (success or
            # sandbox-failure) instead of re-calling the LLM + retraining. Reproduces the exact
            # search (same feedback-block carry -> same winner) at zero API/GPU cost.
            if resume:
                _run_id = f"{run_prefix}-{candidate_id}"
                _hit = None
                try:
                    _hit = _load_run(_run_id, archive_root)
                except FileNotFoundError:
                    _hit = None  # not yet done -> fall through and generate
                # A corrupt/integrity-failed record (KeyError/ValueError) PROPAGATES — failing loud
                # beats silently regenerating (which would re-bill Opus and could desync the search).
                if _hit is not None:
                    _m = _hit.get("metrics", {}) or {}
                    _rec = CandidateRecord(
                        prompt=_hit.get("prompt", ""),
                        reward_source=_hit.get("reward_source", ""),
                        reward_hash=_hit["reward_source_hash"],
                        feedback_block=_hit["feedback_block"],
                        val_fitness=float(_m["val_fitness"]),
                        tail_stats=_m.get("tail_stats"),
                        generation=int(_hit["generation"]),
                        candidate_id=candidate_id,
                        popart_scale=_m.get("popart_scale"),
                    )
                    archive.candidates.append(_rec)
                    if gen_best_fitness is None or _rec.val_fitness > gen_best_fitness:
                        gen_best_fitness = _rec.val_fitness
                        gen_best_block = _rec.feedback_block
                    monitor.candidate_done(arm, cand_n, fitness=_rec.val_fitness, status="ok",
                                           secs=time.perf_counter() - cand_t0)
                    # Stream-faithful replay (2026-07-06): this slot was AUTHORED in the original
                    # run, so consume its author-stream position — a POSITIONAL (stub) transport
                    # keeps later fresh candidates byte-identical to an uninterrupted run; a real
                    # LLM transport no-ops (its calls cannot be replayed by position). Duck-typed:
                    # injected test fakes need not implement it.
                    _adv = getattr(llm, "advance_author_stream", None)
                    if _adv is not None:
                        _adv()
                    continue
                if candidate_id in _cached_failures:
                    _cf = _cached_failures[candidate_id]
                    archive.failures.append({
                        "generation": gen,
                        "candidate_id": candidate_id,
                        "reward_source": _cf.get("reward_source", ""),
                        "error": _cf.get("error", "cached failure"),
                    })
                    monitor.candidate_done(arm, cand_n, fitness=None, status="sandbox_reject",
                                           secs=time.perf_counter() - cand_t0)
                    _adv = getattr(llm, "advance_author_stream", None)
                    if _adv is not None:
                        _adv()  # a ledgered failure was authored too (one position)
                    continue

            # Per-candidate prompt variation -> within-generation diversity without temperature
            # (see _diversity_directive). cand_prompt is the EXACT prompt sent + archived (C-2).
            cand_prompt = user_prompt
            if diversity and candidates_per_gen > 1:
                cand_prompt = f"{user_prompt}\n\n{_diversity_directive(cidx, candidates_per_gen)}"

            # 2. Sample a candidate reward source from the LLM, salvaging fenced / prose-wrapped
            #    output so a well-formed reward is never rejected for FORMATTING (final-audit P0).
            _llm_t0 = time.perf_counter()
            try:
                src = extract_reward_source(llm.complete(system_prompt, cand_prompt))
            except Exception as exc:  # noqa: BLE001 — a 2-week unattended run must not die on one candidate
                # API exhaustion (rate-limit/outage AFTER the client's bounded retries) or an unparseable
                # response: LOG loudly + SKIP the candidate, never crash the arm. Deliberately NOT written
                # to the failures-ledger (unlike a sandbox reject) — an API failure is transient and no
                # generation happened, so a later --resume RE-ATTEMPTS this slot (the search-replay cache
                # regenerates only the missing candidate). A whole arm that fails this way reports
                # "no_candidates" downstream rather than aborting the campaign.
                _LOG.error("candidate %s: LLM generation failed (%s: %s) — skipped; --resume re-attempts it",
                           candidate_id, type(exc).__name__, str(exc)[:200])
                # C3c: count the skip + emit a monitor ANOMALY (kind "llm_error") so a wave of API
                # failures is loud on the dashboard/anomalies.jsonl instead of living only in a
                # transient log line. No-op when no RunMonitor is attached (_NULL_MONITOR).
                llm_error_skips += 1
                monitor.anomaly(
                    "llm_error",
                    f"candidate {candidate_id}: LLM generation failed "
                    f"({type(exc).__name__}: {str(exc)[:120]}) — slot skipped; --resume re-attempts it",
                    arm=arm, cand=cand_n,  # FIX B: explicit identity, not just in the free-text detail
                )
                monitor.candidate_done(arm, cand_n, fitness=None, status="llm_error",
                                       secs=time.perf_counter() - cand_t0)
                continue
            _arch = getattr(llm, "archive", None)  # LLMClient archives a ProvenanceRecord (with token usage)
            _u = (getattr(_arch[-1], "usage", None) or {}) if _arch else {}
            monitor.llm_call(arm, cand_n, secs=time.perf_counter() - _llm_t0,
                             in_tok=_u.get("input_tokens"), out_tok=_u.get("output_tokens"), model=model_id)
            # 2026-07-20 (key-failover visibility): a SUCCESSFUL failover raises no exception, so
            # without this it would live only in the driver log — invisible to anomalies.jsonl and
            # the ntfy push. Surface it ONCE per loop as a monitor anomaly (the operator must know
            # the run is now spending the fallback account).
            if (getattr(getattr(llm, "_transport", None), "_failed_over", False)
                    and not _failover_announced):
                _failover_announced = True
                monitor.anomaly(
                    "api_key_failover",
                    "the PRIMARY API key died (credit exhausted / revoked) — authoring switched "
                    "PERMANENTLY to the fallback key; all further spend is on the fallback account",
                    arm=arm, cand=cand_n)

            # 3. Validate; LOG + SKIP on failure (never crash the loop).
            try:
                reward_fn = validate_once(src, fixture)
                monitor.sandbox_result(arm, cand_n, ok=True)
            except SandboxEnvironmentError:
                # 2026-07-19 audit: a STARVED spawn environment is NOT a candidate defect. Do NOT
                # ledger it as a permanent rejection (that would poison this PAID candidate into the
                # frozen reject set --resume replays). Abort the authoring loop LOUDLY instead: the
                # supervisor relaunches, and --resume preserves every already-validated candidate
                # and re-attempts only this one on a (hopefully freed) box.
                _LOG.error("candidate %s: validation ENVIRONMENT starved — aborting authoring for a "
                           "clean supervisor relaunch (candidate NOT rejected)", candidate_id)
                raise
            except SandboxError as exc:
                _LOG.warning(
                    "candidate %s failed validation and was skipped: %s",
                    candidate_id,
                    exc,
                )
                monitor.sandbox_result(arm, cand_n, ok=False, reason=str(exc)[:120])
                monitor.candidate_done(arm, cand_n, fitness=None, status="sandbox_reject",
                                       secs=time.perf_counter() - cand_t0)
                archive.failures.append(
                    {
                        "generation": gen,
                        "candidate_id": candidate_id,
                        "reward_source": src,
                        "error": str(exc),
                    }
                )
                # Persist the failure so --resume REPLAYS it rather than re-generating a
                # previously-rejected candidate against the non-deterministic author (which could
                # newly succeed and change the set). Best-effort, append-only; never crash the loop.
                try:
                    with _fail_ledger.open("a", encoding="utf-8") as _fh:
                        _fh.write(_json.dumps({
                            "candidate_id": candidate_id, "generation": gen,
                            "reward_source": src, "error": str(exc)[:500],
                        }) + "\n")
                        _fh.flush()
                except Exception:
                    pass
                continue

            # 4-7. Train + evaluate, wrapped like the SandboxError path above (F10, review
            # 2026-07-02): an env/trainer crash on ONE candidate (CUDA OOM, a reward that NaNs the env
            # mid-rollout) used to kill the whole ARM process — exactly the fuel for an unbounded
            # supervisor restart loop (each cycle able to re-bill one LLM call; see supervisor
            # --max-total-restarts). The slot is LEDGERED as a failure — so --resume REPLAYS it instead
            # of re-billing the author, and the C3b winner-selection floor counts it as RESOLVED — and
            # the loop CONTINUES to the next candidate slot.
            try:
                # 4. Train the fixed agent on the (reward-bound) train env.
                env = env_builder(reward_fn)
                policy = agent_trainer(env.train_env())
                # T2.4: capture the realised PopArt scale the critic saw for this candidate (None for a
                # fake/raw trainer that does not surface it) so the cross-arm sigma distribution is auditable.
                popart_scale = getattr(policy, "popart_scale", None)
                # Training-time SAFE_DEFAULT visibility (R66, 2026-07-03): train_agent attaches the
                # training-window substitution counts (None for a fake trainer); archived below so a
                # candidate that part-trained on the 0.0 fallback signal is auditable from the record.
                train_sd_count = getattr(policy, "train_safe_default_count", None)
                train_call_count = getattr(policy, "train_safe_call_count", None)

                # 5. Evaluate on the VALIDATION split (realized val returns).
                val_returns = np.asarray(env.val_returns(policy), dtype=float)

                # 6. Held-out fitness (validation Deflated Sharpe), reward-independent.
                val_fitness = float(fitness_fn(val_returns, n_trials))

                # 7. Measure the TRAINING realized-return distribution -> tail stats.
                train_returns = np.asarray(env.train_returns(policy), dtype=float)
                dist = measurement()
                dist.fit(train_returns)
                tail_stats = dist.tail_stats()
            except Exception as exc:  # noqa: BLE001 — a 2-week unattended run must not die on one candidate
                _err = f"training_error: {type(exc).__name__}: {exc}"
                _LOG.error("candidate %s: %s — ledgered as a failure; the arm continues",
                           candidate_id, _err[:300])
                monitor.anomaly(
                    "training_error",
                    f"candidate {candidate_id}: {_err[:120]} — slot ledgered as a failure; "
                    f"the arm continues",
                    arm=arm, cand=cand_n,  # FIX B: explicit identity, not just in the free-text detail
                )
                monitor.candidate_done(arm, cand_n, fitness=None, status="error",
                                       secs=time.perf_counter() - cand_t0)
                archive.failures.append(
                    {
                        "generation": gen,
                        "candidate_id": candidate_id,
                        "reward_source": src,
                        "error": _err,
                    }
                )
                # Same crash-robust append ledger the sandbox path writes (best-effort, never crash
                # the loop) — the SAME jsonl shape, so the resume reader and _search_pool_counts
                # treat a training-crashed slot exactly like a sandbox-rejected one.
                try:
                    with _fail_ledger.open("a", encoding="utf-8") as _fh:
                        _fh.write(_json.dumps({
                            "candidate_id": candidate_id, "generation": gen,
                            "reward_source": src, "error": _err[:500],
                        }) + "\n")
                        _fh.flush()
                except Exception:
                    pass
                continue

            # 8. Build the next feedback block (carry state forward).
            #    Tail-carrying arms get tail_stats; scalar/placebo get None. placebo_shuffled (R32)
            #    is tail-carrying too, but its values are deranged by a candidate-seeded shuffle.
            tail_for_block = (
                tail_stats
                if arm in ("distributional", "scalar_cvar5", "placebo_shuffled")
                else None
            )
            feedback_block = schema.build_block(
                arm,
                val_fitness,
                tail_for_block,
                shuffle_seed=(
                    schema.shuffle_seed_from_id(candidate_id)
                    if arm == "placebo_shuffled"
                    else None
                ),
                legible=legible_render,
            )

            reward_h = _reward_hash(src)
            record = CandidateRecord(
                prompt=cand_prompt,
                reward_source=src,
                reward_hash=reward_h,
                feedback_block=feedback_block,
                val_fitness=val_fitness,
                tail_stats=tail_stats,
                generation=gen,
                candidate_id=candidate_id,
                popart_scale=popart_scale,
            )
            archive.candidates.append(record)

            # 9. Archive the candidate (replay, audit C-2). Conforms to the
            #    results-IO schema so load_run round-trips.
            run_id = f"{run_prefix}-{candidate_id}"
            write_run(
                {
                    # Driver-supplied top-level discriminators (e.g. label/condition for the report-only
                    # named-vs-blinded & legible sub-experiments). Spread FIRST so the canonical keys below
                    # always win — a driver can never clobber run_id/arm/metrics. Empty by default.
                    **extra_record_fields,
                    "run_id": run_id,
                    "arm": arm,
                    "seed": seed,
                    "fold": cfg_get(cfg, "fold", 0),
                    "candidate_id": candidate_id,
                    "generation": gen,
                    # Rank 14: persist the rendered prompt so the replay archive doesn't DROP it
                    # ("archive every prompt"). results.write_run dumps it to
                    # a prompt.txt sidecar; OPTIONAL_FIELDS so REQUIRED_FIELDS / round-trips unchanged.
                    "prompt": cand_prompt,
                    "reward_source": src,
                    "reward_source_hash": reward_h,
                    "feedback_block": feedback_block,
                    "metrics": {
                        "val_fitness": val_fitness,
                        "tail_stats": tail_stats,
                        # Realized validation returns archived so analyze_results can run the
                        # Sharpe/CVaR difference tests + FZ ES backtest on the winners (P6).
                        "val_returns": [float(x) for x in val_returns],
                        # T2.4: realised PopArt scale (sigma_max/last) the critic saw, for the cross-arm
                        # sigma audit. Optional/back-compatible (omitted when the trainer doesn't surface it).
                        **({"popart_scale": popart_scale} if popart_scale is not None else {}),
                        # R66 (2026-07-03): training-window SAFE_DEFAULT substitution counts attached by
                        # train_agent — additive/optional (omitted for trainers that don't surface them).
                        **(
                            {
                                "train_safe_default_count": int(train_sd_count),
                                "train_safe_call_count": int(train_call_count),
                            }
                            if train_sd_count is not None and train_call_count is not None
                            else {}
                        ),
                    },
                    "wall_clock": 0.0,
                    "env_fingerprint": env_fingerprint,
                },
                archive_root,
            )

            # 10. Reflect-on-BEST (M5): track the generation's best candidate; its feedback block
            #     seeds the NEXT generation's prompt (set at the generation boundary below).
            if gen_best_fitness is None or val_fitness > gen_best_fitness:
                gen_best_fitness = val_fitness
                gen_best_block = feedback_block
            monitor.candidate_done(arm, cand_n, fitness=val_fitness, status="ok",
                                   secs=time.perf_counter() - cand_t0)

        # Generation boundary: the generation's BEST candidate seeds the next prompt (M5 reflect-on-
        # best — Eureka-faithful, and consistent with the parallel reflect-on-best path).
        if gen_best_block is not None:
            prev_feedback_block = gen_best_block

    archive.meta["budget_spent"] = budget_spent
    archive.meta["accepted"] = len(archive.candidates)
    archive.meta["failed"] = len(archive.failures)
    # C3c: LLM-error skips are surfaced in the run meta/summary — accepted + failed + llm_error_skips
    # should equal the drawn budget; a non-zero count means the pool is resumably SHORT (the
    # campaign-level select floor, run_campaign C3b, refuses to freeze a winner from it).
    archive.meta["llm_error_skips"] = llm_error_skips
    return archive
