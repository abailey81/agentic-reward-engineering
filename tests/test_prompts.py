"""Tests for src/llm/prompts.py + the C3 loop wiring (P4).

Verifies the env-interface rendering (anonymised shapes + the exact reward contract), that
build_prompt_set fills {ENV_INTERFACE}, and that run_loop actually SENDS the injected prompts
(the C3 fix — previously loop.py used hardcoded contentless prompts and ignored prompts/).
"""
from __future__ import annotations

import numpy as np

from src.llm.prompts import build_prompt_set, render_env_interface
from src.utils.config import load_config


def test_render_env_interface_has_contract_and_shapes() -> None:
    cfg = load_config("environment")
    s = render_env_interface(cfg, n_assets=30)
    assert "def reward(weights, returns, prev_weights, port_ret, info)" in s
    assert "softmax" in s
    assert "numpy" in s.lower()
    assert "(31,)" in s  # n_act = 30 + 1 (weights/prev_weights shape)
    assert "(30,)" in s  # per-asset returns shape
    # Anonymised by construction (rendered from shapes/config, never from data): it
    # instructs the designer to use "no dates" and contains only numbers/shapes — there is
    # no asset identifier or calendar value to leak (N3).
    assert "ANONYMISED" in s


def test_build_prompt_set_fills_env_interface() -> None:
    cfg = load_config("environment")
    ps = build_prompt_set(cfg, n_assets=30)
    assert "{ENV_INTERFACE}" not in ps.initial  # placeholder filled
    assert "reward(weights, returns, prev_weights, port_ret, info)" in ps.initial
    assert "reward" in ps.system.lower()


def test_loop_sends_injected_prompts(tmp_path) -> None:
    """run_loop sends the injected env-interface-rich system/initial prompts (C3)."""
    from src.feedback.measurement import ReturnDistribution
    from src.llm import loop
    from src.selection.fitness import held_out_fitness

    class _RecLLM:
        def __init__(self) -> None:
            self.prompts: list[tuple[str, str]] = []
            self.cfg = {"model": "x"}

        def complete(self, system: str, user: str) -> str:
            self.prompts.append((system, user))
            return (
                "def reward(weights, returns, prev_weights, port_ret, info):\n"
                "    return float(port_ret), {'r': float(port_ret)}, None\n"
            )

    class _FakeEnv:
        def __init__(self, reward_fn) -> None:
            self.reward_fn = reward_fn

        def train_env(self):
            return self

        def train_returns(self, _policy):
            return -0.001 + 0.01 * np.random.default_rng(1).standard_t(5, size=256)

        def val_returns(self, _policy):
            return 0.01 * np.random.default_rng(2).standard_normal(256)

    env_cfg = load_config("environment")
    ps = build_prompt_set(env_cfg, n_assets=8)
    rec = _RecLLM()
    loop_cfg = {
        "generations": 1,
        "candidates_per_gen": 1,
        "budget": 1,
        "n_trials": 2,
        "prompts": {"system": ps.system, "initial": ps.initial},
    }
    loop.run_loop(
        "scalar",
        _FakeEnv,
        rec,
        lambda _env: object(),  # fake policy (FakeEnv ignores it)
        ReturnDistribution,
        held_out_fitness,
        loop_cfg,
        tmp_path,
    )
    system_sent, initial_sent = rec.prompts[0]
    assert system_sent == ps.system
    assert "reward(weights, returns, prev_weights, port_ret, info)" in initial_sent


def test_ARM_SHARED_prompts_are_TAIL_NEUTRAL() -> None:
    """The construct-validity hinge: nothing shown to EVERY arm may pre-load the tail concepts.

    H2's effect is the MARGINAL value of tail-specificity. That reading only holds if the text every
    arm sees is tail-NEUTRAL — if the base prompt already said "minimise CVaR", the distributional
    arm's block would be redundant rather than informative, and a null would be uninterpretable.
    The claim was verified once by hand (construct-validity audit) but never pinned, so a later
    prompt edit could silently dissolve the identification (deep review).

    Tail-neutral does NOT mean risk-neutral: the shared text deliberately says "optimise
    RISK-ADJUSTED performance" and names an online Sharpe, which is exactly why the SCALAR arm is the
    right control — it supplies the scalar the base prompt already gestures at. What must be absent
    is the tail-SPECIFIC vocabulary that only the distributional block supplies."""
    import re

    from src.llm.loop import _REFLECTION_PREAMBLE

    ps = build_prompt_set(load_config("environment"), 30)
    shared = {
        "system.txt": ps.system,
        "initial_generation.txt (rendered)": ps.initial,
        "_REFLECTION_PREAMBLE": _REFLECTION_PREAMBLE,
    }
    # the six fed statistics' vocabulary + its close synonyms
    tail_terms = (
        "cvar", "conditional value at risk", "expected shortfall", "shortfall",
        "tail", "left tail", "downside", "drawdown", "skew", "kurtosis",
        "quantile", "percentile", "worst case", "worst-case", "var at", "sortino",
    )
    offenders: list[str] = []
    for name, txt in shared.items():
        low = " ".join(txt.lower().split())
        for term in tail_terms:
            if re.search(r"\b" + re.escape(term), low):
                offenders.append(f"{name}: {term!r}")
    assert not offenders, (
        "ARM-SHARED prompt text contains tail-SPECIFIC vocabulary, which would pre-load the "
        f"manipulated variable and dissolve H2's identification: {offenders}"
    )

    # and the deliberate risk-adjusted framing IS present (the scalar arm's control validity)
    assert re.search(r"risk[- ]adjusted", ps.system, re.I), (
        "the shared system prompt must still ask for RISK-ADJUSTED performance — that is what makes "
        "the scalar arm a fair control rather than a straw man"
    )
    # the anonymisation contract: no dates anywhere in what the model sees
    assert not re.search(r"\b(19|20)\d{2}\b|\d{4}-\d{2}-\d{2}", ps.system + ps.initial), (
        "a date leaked into the prompts (N3 anonymisation contract)"
    )
