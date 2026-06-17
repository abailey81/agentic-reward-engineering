"""Experimental-arm factory for the reward-discovery study (FINAL_PLAN F.9).

Purpose
-------
Construct a runnable *arm* descriptor by name. The dissertation compares several
reward-discovery strategies under a single MATCHED candidate budget. To keep the
comparison fair, every arm shares the same headline agent and the same budget;
the arms differ only in *how* candidate rewards are proposed.

The nine arms (config/arms.yaml; FINAL_PLAN B.5; R32 added placebo_shuffled; cma_es/tpe added 2026-07-26)
-------------------------------------------------------------------------------------------------------
Five LLM arms — identical in every respect except the feedback block fed back to
the model (audit A-1: the feedback channel is the contribution):

    distributional : scalar + the full frozen tail-diagnostic set.
    scalar         : scalar performance number only.
    placebo        : scalar + an inert block matched in length/field-count.
    scalar_cvar5   : scalar + one downside number (CVaR 5%).
    placebo_shuffled : distributional's EXACT structure with the tail VALUES deranged (structure-vs-content control; R32).

Four non-LLM search baselines (the H4 optimiser portfolio; N4 confirmatory):

    random_search  : H4a — samples reward CODE from the same code space the LLM
                     uses (src/search/random_search.py).
    bayes_opt      : H4b — Bayesian optimization of the coefficients of a fixed
                     parametric reward template (src/search/bayes_opt.py).
    cma_es         : H4c -- CMA-ES over the SAME template, matched budget
                     (src/search/dfo_toolkit.py; evolution-strategy).
    tpe            : H4d -- TPE over the SAME template, matched budget
                     (src/search/dfo_toolkit.py; density-ratio).

Invariant
---------
Whatever the arm, the candidate budget is identical (matched compute) and the
headline agent is the fixed SB3 SAC learner (audit A-1) — only the proposal
mechanism varies. The five LLM arms differ ONLY in ``feedback_kind``, which is
mapped onto the ``src.feedback.schema`` arm string by ``FEEDBACK_KIND_TO_SCHEMA_ARM``
(e.g. ``full_tail_set`` → ``distributional``) — NOT a straight pass-through to
``build_block`` (whose ``arm`` vocabulary is the schema-arm names, not the feedback kinds).

Tests (tests/test_arms.py)
--------------------------
    - all nine arms build;
    - every arm shares the same candidate budget (matched compute);
    - the five LLM arms have ``is_llm=True`` and are identical except
      ``feedback_kind``;
    - random_search / bayes_opt have ``is_llm=False`` and the right
      ``search_kind``;
    - LLM ``feedback_kind`` values map to valid schema arms.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Optional

from src.feedback.schema import block_fields
from src.utils.config import DotDict, load_config

__all__ = ["Arm", "build_arm", "all_arms", "assert_fixed_agent_across_arms"]


@dataclass(frozen=True)
class Arm:
    """Immutable descriptor of one experimental arm.

    Attributes
    ----------
    name : str
        Arm identifier, matching a key under ``arms:`` in config/arms.yaml.
    feedback_kind : str or None
        For LLM arms, the feedback-block kind — one of ``"full_tail_set"``,
        ``"scalar_only"``, ``"scalar_plus_inert_block"``, ``"scalar_plus_cvar5"``, ``"scalar_plus_shuffled_tail"``.
        ``None`` for the non-LLM search arms.
    is_llm : bool
        ``True`` for the five feedback arms (which use the LLM reward designer),
        ``False`` for ``random_search`` and ``bayes_opt``.
    search_kind : str or None
        For non-LLM arms, ``"code"`` (random search over reward code) or
        ``"template"`` (Bayesian optimization over template coefficients).
        ``None`` for LLM arms.
    candidate_budget : int
        Total reward candidates evaluated by the arm. Identical across all arms
        (matched compute, the property that licenses the comparative claim).
    """

    name: str
    feedback_kind: Optional[str]
    is_llm: bool
    search_kind: Optional[str]
    candidate_budget: int


#: Validates and normalizes the ``feedback`` value stored in config/arms.yaml
#: into this module's ``feedback_kind`` vocabulary (an identity map; its values
#: are feedback kinds, NOT schema arms). It does NOT tie to
#: ``src.feedback.schema.build_block`` / ``block_fields`` — those arm strings are
#: ``distributional`` / ``scalar`` / ``placebo`` / ``scalar_cvar5``, which this
#: dict never produces. The mapping of ``feedback_kind`` onto the schema arm
#: string is performed separately by ``FEEDBACK_KIND_TO_SCHEMA_ARM`` below (the
#: actual single source of truth for that link; audit fix 2026-06-20).
_FEEDBACK_TO_KIND: dict[str, str] = {
    "full_tail_set": "full_tail_set",
    "scalar_only": "scalar_only",
    "scalar_plus_inert_block": "scalar_plus_inert_block",
    "scalar_plus_cvar5": "scalar_plus_cvar5",
    "scalar_plus_shuffled_tail": "scalar_plus_shuffled_tail",
}

#: Maps a ``feedback_kind`` onto the corresponding ``src.feedback.schema`` arm
#: string. Used to render the feedback block and cross-checked by the tests.
FEEDBACK_KIND_TO_SCHEMA_ARM: dict[str, str] = {
    "full_tail_set": "distributional",
    "scalar_only": "scalar",
    "scalar_plus_inert_block": "placebo",
    "scalar_plus_cvar5": "scalar_cvar5",
    "scalar_plus_shuffled_tail": "placebo_shuffled",
}


def _resolve_cfg(cfg: Any) -> DotDict:
    """Return the arms config, loading config/arms.yaml when ``cfg`` is None."""
    if cfg is None:
        return load_config("arms")
    if isinstance(cfg, DotDict):
        return cfg
    return DotDict(cfg)


def _matched_budget(cfg: DotDict) -> int:
    """Read the single matched candidate budget shared by every arm."""
    return int(cfg.require("matched_budget"))


def build_arm(name: str, cfg: Any = None) -> Arm:
    """Build the experimental arm identified by ``name``.

    Parameters
    ----------
    name : str
        Arm identifier. One of the five LLM arms (``distributional``,
        ``scalar``, ``placebo``, ``scalar_cvar5``, ``placebo_shuffled``) or the
        FOUR search arms (``random_search``, ``bayes_opt``, ``cma_es``, ``tpe``).
        The search roster grew 2 -> 4 with the H4 DFO portfolio (R108, the 7 -> 9
        arm reconciliation); this list is a convenience only — the authoritative
        roster is the ``arms:`` table in ``config/arms.yaml``, which this function
        reads dynamically, so a stale list here can never restrict what is built.
    cfg : DotDict or dict or None, optional
        Arms configuration carrying ``matched_budget`` and the ``arms`` table.
        When ``None`` (default), ``config/arms.yaml`` is loaded.

    Returns
    -------
    Arm
        A frozen descriptor wired to the matched budget. LLM arms carry only a
        differing ``feedback_kind``; search arms carry only a differing
        ``search_kind``.

    Raises
    ------
    KeyError
        If ``name`` is not present in the ``arms`` table.
    ValueError
        If an arm entry is malformed (e.g. an LLM arm with an unknown feedback
        kind, or an arm declaring neither feedback nor search).
    """
    config = _resolve_cfg(cfg)
    budget = _matched_budget(config)

    arms_table = config.require("arms")
    if name not in arms_table:
        raise KeyError(
            f"unknown arm {name!r}; available: {sorted(arms_table)}"
        )
    spec = dict(arms_table[name])

    # The five LLM (feedback) arms.
    if "feedback" in spec:
        feedback_raw = str(spec["feedback"])
        if feedback_raw not in _FEEDBACK_TO_KIND:
            raise ValueError(
                f"arm {name!r} has unknown feedback {feedback_raw!r}; "
                f"expected one of {sorted(_FEEDBACK_TO_KIND)}"
            )
        feedback_kind = _FEEDBACK_TO_KIND[feedback_raw]
        return Arm(
            name=name,
            feedback_kind=feedback_kind,
            is_llm=True,
            search_kind=None,
            candidate_budget=budget,
        )

    # The two non-LLM search baselines.
    if "search" in spec:
        search_kind = str(spec["search"])
        if search_kind not in ("code", "template"):
            raise ValueError(
                f"arm {name!r} has unknown search kind {search_kind!r}; "
                f"expected 'code' or 'template'"
            )
        is_llm = bool(spec.get("llm", False))
        return Arm(
            name=name,
            feedback_kind=None,
            is_llm=is_llm,
            search_kind=search_kind,
            candidate_budget=budget,
        )

    raise ValueError(
        f"arm {name!r} declares neither 'feedback' nor 'search' in config"
    )


def all_arms(cfg: Any = None) -> list[Arm]:
    """Build every arm declared in the config, in declaration order.

    Parameters
    ----------
    cfg : DotDict or dict or None, optional
        Arms configuration; loaded from ``config/arms.yaml`` when ``None``.

    Returns
    -------
    list of Arm
        One :class:`Arm` per entry under ``arms:`` in the config.
    """
    config = _resolve_cfg(cfg)
    return [build_arm(name, config) for name in config.require("arms")]


def schema_arm_for(arm: Arm) -> str:
    """Return the ``src.feedback.schema`` arm string for an LLM arm.

    Parameters
    ----------
    arm : Arm
        An LLM arm (``is_llm=True``).

    Returns
    -------
    str
        The matching ``build_block`` / ``block_fields`` arm string.

    Raises
    ------
    ValueError
        If ``arm`` is not an LLM arm or its ``feedback_kind`` is unknown.
    """
    if not arm.is_llm or arm.feedback_kind is None:
        raise ValueError(f"arm {arm.name!r} is not an LLM arm")
    schema_arm = FEEDBACK_KIND_TO_SCHEMA_ARM.get(arm.feedback_kind)
    if schema_arm is None:
        raise ValueError(f"no schema arm for feedback kind {arm.feedback_kind!r}")
    # Cross-check the schema arm is real (raises ValueError if not).
    block_fields(schema_arm)
    return schema_arm


def assert_fixed_agent_across_arms(
    arms: list[Arm], agent_cfg: Any, *, seeds: tuple[int, int] = (0, 1)
) -> dict[str, Any]:
    """TEST-ONLY invariant check — matched budget + a fixed, deterministic shared agent config.

    HONEST SCOPE (critical-review 2026-06-20). This is a TEST-time regression guard, NOT a runtime
    per-arm-override detector. In the frozen design there is a SINGLE shared ``agent_cfg`` and
    ``resolve_agent_kwargs(cfg, seed)`` takes NO arm argument, so a "per-arm hyperparameter override"
    cannot exist for this function to catch — the matched-agent property holds structurally, by
    construction. What this DOES verify (so a future refactor that breaks one is caught) is:

    1. **Matched compute** — every arm carries one and the same ``candidate_budget``.
    2. **Fixed, seed-only agent** — ``resolve_agent_kwargs`` yields the documented ``MlpPolicy`` SB3-SAC
       and depends on the SEED ALONE: at two seeds the network/lr/buffer/batch/gamma/ent-coef/device and
       the train-step budget are byte-identical (a determinism guard; it would fire only if a refactor
       made the kwargs seed- or otherwise non-deterministically dependent, NOT on a per-arm override).
    3. **Feedback is the only LLM difference** — the LLM arms each carry a DISTINCT ``feedback_kind``.

    To make it a TRUE per-arm guard one would have to resolve kwargs PER ARM (e.g. from an
    ``agent_cfg_for_arm`` mapping) and assert byte-equality across arms; the current architecture has no
    such per-arm config, so that is not wired.

    Parameters
    ----------
    arms : list of Arm
        The arms to verify (e.g. ``all_arms()``).
    agent_cfg : Any
        The shared agent/training config passed to ``resolve_agent_kwargs`` for every arm.
    seeds : (int, int)
        Two distinct seeds used to prove the agent config is seed-only-dependent.

    Returns
    -------
    dict
        ``{"agent_kwargs": <shared kwargs minus seed>, "candidate_budget": int, "train_steps": int}``.

    Raises
    ------
    AssertionError
        On any matched-compute / fixed-agent / feedback-isolation violation.
    ValueError
        If ``arms`` is empty or ``seeds`` are not distinct.
    """
    from src.agents.trainer import resolve_agent_kwargs

    if not arms:
        raise ValueError("no arms to verify")
    if seeds[0] == seeds[1]:
        raise ValueError("seeds must be distinct to prove seed-only dependence")

    budgets = {a.candidate_budget for a in arms}
    if len(budgets) != 1:
        raise AssertionError(
            f"matched compute VIOLATED: arms carry differing candidate budgets {sorted(budgets)}"
        )

    k0, steps0 = resolve_agent_kwargs(agent_cfg, seeds[0])
    k1, steps1 = resolve_agent_kwargs(agent_cfg, seeds[1])
    if steps0 != steps1:
        raise AssertionError(f"train-step budget differs across seeds ({steps0} != {steps1}) — not fixed")
    differing = {k for k in set(k0) | set(k1) if k0.get(k) != k1.get(k)}
    if differing != {"seed"}:
        raise AssertionError(
            f"agent kwargs vary by more than the seed {sorted(differing)} — the agent is NOT fixed"
        )
    if k0.get("policy") != "MlpPolicy":
        raise AssertionError(
            f"the headline agent must be the fixed SB3-SAC MlpPolicy; got {k0.get('policy')!r}"
        )

    llm = [a for a in arms if a.is_llm]
    if llm and len({a.feedback_kind for a in llm}) != len(llm):
        raise AssertionError(
            "the LLM arms must each carry a DISTINCT feedback_kind (only the feedback channel may differ)"
        )

    return {
        "agent_kwargs": {k: v for k, v in k0.items() if k != "seed"},
        "candidate_budget": next(iter(budgets)),
        "train_steps": steps0,
    }
