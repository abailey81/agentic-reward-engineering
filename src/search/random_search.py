"""Random search over the reward CODE space (LLM-free control; H4a).

Purpose
-------
A non-LLM control arm (FINAL_PLAN F.10, hypothesis H4a). It samples reward
functions uniformly at random from a fixed grammar (no model in the loop) and
scores them under the MATCHED candidate budget. Its role is to be a *procedure*
control at COMPARABLE expressive power to the Bayesian-optimisation arm (H4b):
both H4a and H4b draw from the SAME six-primitive reward family, so the H4a-vs-LLM
contrast isolates *proposal quality* rather than rewarding the LLM merely for a
richer reward language.

What this arm CAN and CANNOT isolate (state this precisely)
-------------------------------------------------------------------------------------
- It DOES give a procedure-only comparison against H4b: random-search and BO now
  search the identical six-term family at the identical fixed discrete dims, so
  any H4b-vs-H4a difference is surrogate-vs-uniform, not space-vs-space.
- It does NOT match the LLM's search SPACE: the LLM authors FREE-FORM code (tanh,
  ratios, regime switches, Sortino-style asymmetries — none reachable by a fixed
  six-term linear combination). So a positive H4a reflects *proposal quality at
  comparable-but-not-identical richness*, and the residual richness gap (free-form
  vs six-term) is part of what H4b measures. Do NOT claim H4a proves the LLM is a
  better optimiser over an identical space.
- Random search is a deliberately STRONG baseline (Bergstra & Bengio 2012): a weak
  control would make a positive H4a trivially true and therefore uninformative.

The code grammar (the SHARED six-primitive family)
--------------------------------------------------
Every sampled reward is the SAME six-term linear family the BO arm searches
(``src/baselines/reward_family.py``), rendered to executable source via
:func:`reward_family.params_to_source` so its form is byte-identical to the BO
arm's materialised winner (same numpy ops, same gate-clean status, same
``(hist, peak, cum)`` reward_state):

    r = w_return*net_return + w_log*log1p(net_return)
        - w_turnover*turnover - w_drawdown*drawdown
        - w_cvar*max(0, -CVaR_alpha) - w_vol*sigma

The six non-negative weights are drawn from a coarse per-primitive grid spanning
the SAME frozen ``[low, high]`` ranges as the BO box (``family_bounds``); the two
discrete dims (``cvar_alpha``, ``window``) are FIXED at the same values the BO arm
freezes (0.05, 20) so the two H4 arms share an identical search space. The grid is
non-negative-only, so every penalty term acts as a penalty (a *risk-aware* random
baseline, matching the LLM's and the family's risk-aware intent — disclose this; it
is favourable to the control, not unfair).

Algorithm (FINAL_PLAN F.10, H4a)
--------------------------------
    1. For each of ``candidate_budget`` units, sample a six-weight vector from the
       per-primitive grid and render the family reward source.
    2. Gate + validate the source via ``src.sandbox.executor`` (skip-and-log
       gate failures; they do NOT consume a budget unit — we keep sampling until
       the full matched budget of *valid* candidates has been evaluated).
    3. Score each validated reward with the injected ``fitness_fn`` (which maps a
       reward callable to a held-out score; deterministic in the tests).
    4. Track the best candidate; stop when the matched budget is exhausted.
    5. Return the best reward + an archive of every evaluated candidate.

The search BUDGET is unchanged by the grammar widening (still exactly
``matched_budget`` valid candidates), so matched compute with the LLM / BO arms
holds.

Tests (tests/test_search.py)
----------------------------
    - returns a valid (gate-passing) best reward and consumes EXACTLY the matched
      budget;
    - every sampled source passes ``ast_gate`` and is six-term family code.
"""

from __future__ import annotations


from typing import Any, Callable, Optional

import numpy as np

from src.baselines.reward_family import WEIGHT_KEYS, family_bounds, params_to_source
from src.sandbox.executor import SandboxEnvironmentError, SandboxError, ast_gate, validate_once
from src.utils.config import DotDict

__all__ = ["random_search_over_code", "sample_reward_source", "code_grid"]

#: Fractions of each weight's frozen ``high`` bound that the per-primitive grid
#: lands on (``low`` is 0.0 for all six family weights). Coarse and non-negative so
#: each penalty term always acts as a penalty (a risk-aware random baseline) while
#: still covering the SAME six-dimensional box as the BO arm. Keeping 0.0 lets the
#: sampler switch any term off (so it can recover sparser canon-like vertices).
_GRID_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

#: Fixed discrete dims, IDENTICAL to the values the BO arm freezes
#: (``reward_family.params_to_reward`` / ``config: reward_family``), so the two H4
#: arms search the identical space.
_FIXED_CVAR_ALPHA: float = 0.05
_FIXED_WINDOW: int = 20


def code_grid() -> tuple[float, ...]:
    """Return the fixed per-primitive grid fractions the random search draws from.

    Each of the six family weights is sampled as ``fraction * high`` for a fraction
    drawn uniformly from this tuple, where ``high`` is that weight's frozen upper
    bound (:func:`reward_family.family_bounds`). The fractions are returned (rather
    than absolute coefficients) because the absolute grid differs per primitive —
    the six weights have different ranges (e.g. ``w_turnover`` ∈ [0, 0.02] vs
    ``w_cvar`` ∈ [0, 5]).
    """
    return _GRID_FRACTIONS


def _weight_grids(cfg: Any = None) -> list[np.ndarray]:
    """Per-primitive coefficient grids over the frozen ``[low, high]`` family box.

    Returns one 1-D grid per family weight (row order :data:`WEIGHT_KEYS`), each
    equal to ``low + fraction * (high - low)`` for the fractions in
    :data:`_GRID_FRACTIONS`. Reads the frozen ranges from ``cfg`` when given
    (via :func:`reward_family.family_bounds`), else the family defaults.
    """
    box = family_bounds(cfg)  # (6, 2) [low, high], row order == WEIGHT_KEYS
    fr = np.asarray(_GRID_FRACTIONS, dtype=float)
    grids: list[np.ndarray] = []
    for lo, hi in box:
        grids.append(float(lo) + fr * (float(hi) - float(lo)))
    return grids


def sample_reward_source(
    rng: np.random.Generator, cfg: Any = None
) -> str:
    """Render one reward source string from the shared six-primitive family.

    Draws each of the six family weights ``(w_return, w_log, w_turnover,
    w_drawdown, w_cvar, w_vol)`` uniformly from its per-primitive grid over the
    frozen ranges, then materialises the family reward at those weights (and the
    fixed ``cvar_alpha=0.05``, ``window=20``) via
    :func:`reward_family.params_to_source`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness for the weight draws.
    cfg : DotDict or dict, optional
        Carries ``reward_family.weights`` ranges; defaults to the frozen family
        box when omitted. Passed through to :func:`reward_family.family_bounds`.

    Returns
    -------
    str
        A ``def reward(...)`` source string for the six-term family with
        grid-sampled weights. The source obeys the reward contract and passes the
        sandbox AST gate (it is the SAME source form the BO arm materialises).
    """
    grids = _weight_grids(cfg)
    coeffs = [float(rng.choice(g)) for g in grids]
    # Guarantee at least the return term so the reward is never identically zero
    # (an all-zero weight vector renders a constant-0 reward, which is degenerate).
    if all(c == 0.0 for c in coeffs):
        # WEIGHT_KEYS[0] is w_return; set it to its grid max so the reward is alive.
        coeffs[0] = float(grids[0].max())
    return params_to_source(
        coeffs, cvar_alpha=_FIXED_CVAR_ALPHA, window=_FIXED_WINDOW
    )


# WEIGHT_KEYS is imported for the docstring's coefficient-order contract and to
# pin the rendered family's weight ordering; reference it so linters see the use.
assert WEIGHT_KEYS[0] == "w_return"


def _default_fixture() -> tuple:
    """Small anonymized contract fixture for one-shot validation.

    SHAPE PARITY (review batch 3 #1, 2026-07-03): production calls
    ``reward(weights(N+1), returns(N), prev(N+1), ...)`` — weights carry the cash slot, returns
    only the risky leg (portfolio_env.py:347-348). The fixture mirrors that contract so a
    shape-aware reward is gated on the SAME shapes it will face in training.
    """
    weights = np.array([0.4, 0.3, 0.3], dtype=float)
    returns = np.array([0.01, -0.02], dtype=float)  # N = len(weights) - 1 (the risky leg)
    prev_weights = np.array([0.34, 0.33, 0.33], dtype=float)
    port_ret = 0.002
    # info mirrors portfolio_env.py:323-327 (weights, prev_weights, reward_state=None at reset) so a
    # stateful reward reading the prompt-documented info["reward_state"] is not falsely rejected here
    # (audit 2026-07-19). The returns above are already mixed-sign — the reference non-degenerate fixture.
    info: dict[str, Any] = {"weights": weights, "prev_weights": prev_weights, "reward_state": None}
    return (weights, returns, prev_weights, port_ret, info)


def _budget(cfg: Any) -> int:
    """Read the matched candidate budget from ``cfg``."""
    if cfg is None:
        raise ValueError("random_search_over_code requires a cfg with a budget")
    if isinstance(cfg, (DotDict, dict)):
        for key in ("matched_budget", "candidate_budget", "candidate_budget_total"):
            if key in cfg:
                return int(cfg[key])
        raise KeyError(
            "cfg must carry one of matched_budget / candidate_budget / "
            "candidate_budget_total"
        )
    return int(cfg)


def random_search_over_code(
    env_builder: Any,
    fitness_fn: Callable[[Any], float],
    cfg: Any,
    rng: Optional[np.random.Generator] = None,
    cache_lookup: Optional[Callable[[int, str], Optional[float]]] = None,
    on_evaluated: Optional[Callable[[int, str, float], None]] = None,
) -> dict[str, Any]:
    """Randomly search the reward code space under the matched budget (H4a).

    Parameters
    ----------
    env_builder : Any
        ACCEPTED AND NOT USED — retained only so the production call site keeps
        its positional shape. NOTE: This said "Forwarded to the injected
        ``fitness_fn`` when it accepts an environment" until 2026-07-27 (deep
        review #113); that was never true under ANY condition. ``env_builder``
        occurs exactly twice in this module — this signature and this docstring —
        and the single call is ``fitness_fn(reward)``, one argument, so no
        environment is ever passed on. In production (``run_prototype.py``) and
        in the tests alike, ``fitness_fn`` is a closure that has ALREADY bound
        its environment, which is why nothing was broken by the gap between the
        claim and the code. Nor is it signature parity with the sibling H4
        searchers: ``bayes_opt`` and ``dfo_toolkit`` (cma_es / tpe) do not take
        this parameter at all. Do not write a ``fitness_fn`` that expects an
        environment argument on the strength of the old sentence.
    fitness_fn : Callable[[reward], float]
        Maps a validated reward callable to a held-out fitness score. In
        production this trains the fixed headline agent on the candidate reward
        and scores its validation returns; in tests it is a deterministic
        closure (no agent training required).
    cfg : DotDict or dict or int
        Carries the matched candidate budget (``matched_budget`` /
        ``candidate_budget`` / ``candidate_budget_total``), or the budget int
        directly. When a mapping, it may also carry ``reward_family.weights`` to
        override the frozen family ranges the grammar draws from.
    rng : numpy.random.Generator, optional
        Source of randomness; a fresh default generator is created when ``None``.

    Returns
    -------
    dict
        ``{"best_source", "best_reward", "best_score", "archive", "n_evaluated",
        "budget"}`` where ``archive`` is a list of per-candidate records
        ``{"source", "score"}`` for every evaluated (validated) candidate.

    Notes
    -----
    Gate/validation failures are skipped without consuming a budget unit, so the
    search always evaluates EXACTLY ``budget`` valid candidates (matched compute
    is invariant to the grammar). The grammar renders the six-term reward family
    via ``reward_family.params_to_source``, which is gate-clean by construction
    (numpy-only allowlisted ops, no imports, no dunders); the assertion in the
    loop is a defensive invariant, not an expected branch.
    """
    budget = _budget(cfg)
    if budget <= 0:
        raise ValueError(f"matched budget must be positive; got {budget}")
    if rng is None:
        # 2026-07-06 (repro audit): the omitted-rng default is now DETERMINISTIC (seed 0), not OS
        # entropy — the exact footgun that once made the parallel BO winner non-reproducible
        # (final-audit #2). Every production caller passes a run-seeded generator explicitly.
        rng = np.random.default_rng(0)

    # Resolve the family ranges ONCE (cfg may carry reward_family.weights overrides);
    # a bare int / None budget falls back to the frozen family defaults.
    family_cfg = cfg if isinstance(cfg, (DotDict, dict)) else None

    fixture = _default_fixture()

    archive: list[dict[str, Any]] = []
    best_score = -np.inf
    best_source: Optional[str] = None
    best_reward: Optional[Any] = None

    # Defensive cap so a (hypothetical) pathological gate-failure streak cannot
    # loop forever; the grammar is gate-clean, so this is never reached.
    max_attempts = max(1000, budget * 100)
    attempts = 0

    while len(archive) < budget and attempts < max_attempts:
        attempts += 1
        source = sample_reward_source(rng, family_cfg)

        # Invariant: every sampled source passes the AST gate.
        assert ast_gate(source), "sampled source must pass ast_gate"

        try:
            reward = validate_once(source, fixture)
        except SandboxEnvironmentError:
            # CONTRACT (src/sandbox/executor.py::SandboxEnvironmentError): a starved spawn
            # environment is NOT a defect of the sampled source. `continue`-ing here would be
            # especially damaging on THIS path: the skip does not consume a budget unit, so a
            # persistently starved box spins up to `max_attempts` (>= 1000) times and then returns a
            # SHORT archive — silently breaking H4a's matched-budget control rather than failing.
            # Fail loud instead; the supervisor relaunches and --resume replays the archive.
            # (deep-review 2026-07-26, loop 2 — found by the repo-wide handler-order contract test.)
            raise
        except SandboxError:
            # Should not happen for the gate-clean grammar; skip without
            # consuming a budget unit if it ever does.
            continue

        # Resume/checkpoint hooks (2026-07-05 crash-resume hardening). The candidate DRAW above
        # always happens — so the rng stream is IDENTICAL with or without a cache hit and the
        # re-drawn sequence byte-matches the original run — but a cached score skips the expensive
        # training. ``cache_lookup(i, source)`` returns the archived score (the caller verifies the
        # archived source hash matches this re-drawn ``source`` and fails loud on drift) or ``None``;
        # ``on_evaluated(i, source, score)`` fires ONLY for fresh evaluations, letting the caller
        # checkpoint each candidate the moment it completes instead of after the whole arm.
        idx = len(archive)
        cached = cache_lookup(idx, source) if cache_lookup is not None else None
        if cached is not None:
            score = float(cached)
        else:
            score = float(fitness_fn(reward))
            if on_evaluated is not None:
                on_evaluated(idx, source, score)
        archive.append({"source": source, "score": score})

        if score > best_score:
            best_score = score
            best_source = source
            best_reward = reward

    if len(archive) < budget:
        # Budget-exhaustion is a search-loop failure, NOT a per-candidate
        # sandbox rejection, so raise RuntimeError rather than SandboxError
        # (which executor.py documents as "Raised by validate_once when a
        # candidate reward is rejected"). Using SandboxError here would let a
        # caller catching it to skip a bad candidate silently swallow this fatal
        # budget failure and misattribute it in logs/tracebacks.
        raise RuntimeError(
            f"exhausted {attempts} attempts before reaching the matched budget "
            f"of {budget} valid candidates"
        )

    return {
        "best_source": best_source,
        "best_reward": best_reward,
        "best_score": best_score,
        "archive": archive,
        "n_evaluated": len(archive),
        "budget": budget,
    }
