"""Prompt assembly for the LLM reward-designer (C3 fix; FINAL_PLAN F.8).

The discovery loop must hand the LLM the ENVIRONMENT INTERFACE — the observation/action
shapes, the step semantics, the reward contract, and the cost model — or it is prompted
with a contentless one-liner and writes blind (adversarial-review finding **C3**). Before
this module, ``src/llm/loop.py`` used hardcoded minimal prompts and never read ``prompts/``.

This module loads the ``system.txt`` and ``initial_generation.txt`` TEMPLATES from ``prompts/``
(single source of truth, no hardcoding) and fills the ``{ENV_INTERFACE}`` placeholder in the
initial prompt from the *anonymised* env spec (no tickers, no dates — N3). The reflection turn is
NOT templated here: the loop composes it from its built-in ``_REFLECTION_PREAMBLE`` plus the
per-arm ``{ARM_BLOCK}`` rendered at runtime by ``src/feedback/schema.build_block`` — so
``reflection.txt`` is intentionally not loaded (it was dead: no path ever substituted its markers).

The rendered :class:`PromptSet` (``system`` + ``initial``) is passed to ``run_loop`` via
``cfg["prompts"]``; when absent the loop falls back to its built-in minimal prompts, so unit
tests need no prompt files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import cfg_get

__all__ = ["PromptSet", "render_env_interface", "build_prompt_set", "DEFAULT_PROMPTS_DIR"]

#: Repo-root ``prompts/`` directory (the single source of truth for templates).
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class PromptSet:
    """The two rendered prompts the loop actually sends (``system`` + gen-0 ``initial``).

    There is deliberately NO ``reflection`` field: both run paths (``src/llm/loop.py`` and
    ``src/orchestration/parallel.py``) compose every reflection turn from the built-in
    ``_REFLECTION_PREAMBLE`` plus the per-arm feedback block rendered by
    ``src/feedback/schema.build_block`` — that block is the only thing that differs across the
    five LLM arms (same structure/length, so the contrast isolates information, not token-count).
    A previously-loaded ``reflection.txt`` template was carried here but never consumed by any
    path (its ``{SCALAR_METRIC}``/``{ARM_BLOCK}`` markers were never substituted), so it is
    no longer loaded — keeping this class an honest description of behavior (audit medium-finding).
    """

    system: str
    initial: str


def render_env_interface(env_cfg: Any, n_assets: int) -> str:
    """Render the anonymised ``{ENV_INTERFACE}`` block from the env config.

    Describes the observation (for context), the bounded-logit → softmax-simplex action,
    the proportional turnover cost, and the exact reward contract — using shapes/numbers
    only (no tickers, no dates; N3). Mirrors ``src/env/portfolio_env.py`` exactly.
    """
    state = cfg_get(env_cfg, "state", {})
    action = cfg_get(env_cfg, "action", {})
    costs = cfg_get(env_cfg, "costs", {})
    lookback = int(cfg_get(state, "lookback_days", 60))
    vol_windows = list(cfg_get(state, "realized_vol_windows", [20, 60]))
    include_vix = bool(cfg_get(state, "include_vix", True))
    include_prev = bool(cfg_get(state, "include_prev_weights", True))
    bound = float(cfg_get(action, "bound", 10.0))
    cost_bps = float(cfg_get(costs, "headline_bps", 10))
    # Per-session cash money-market rate (portfolio_env.py:279, pre-reg R20). DEFAULT 0.0 ->
    # the cash sleeve earns nothing and the two port_ret forms coincide; a NON-ZERO rate credits
    # the cash weight, so the rendered contract must surface it or it understates the agent's
    # reward and the "mirrors src/env/portfolio_env.py exactly" claim is false (audit low-finding).
    cash_daily_rate = float(cfg_get(state, "cash_daily_rate", 0.0))
    cash_term = " + weights[cash] * cash_daily_rate" if cash_daily_rate != 0.0 else ""

    n_act = n_assets + 1
    obs_dim = lookback * n_assets + len(vol_windows) * n_assets
    obs_dim += 1 if include_vix else 0
    obs_dim += 1  # cash marker
    obs_dim += n_act if include_prev else 0

    return (
        f"The agent allocates a LONG-ONLY portfolio over {n_assets} risky assets plus cash "
        f"({n_act} weights that sum to 1).\n\n"
        f"OBSERVATION (the agent's input — listed for context; the reward does NOT receive it), "
        f"dimension {obs_dim}: per asset, the last {lookback} daily returns; per-asset realized "
        f"volatility over windows {vol_windows}; "
        f"{'the lagged volatility-index level; ' if include_vix else ''}a cash marker"
        f"{'; and the previous weights' if include_prev else ''}.\n\n"
        f"ACTION: the agent outputs {n_act} real logits in [-{bound:g}, {bound:g}]; the environment "
        f"applies softmax to get long-only simplex weights, realizes the next-step asset returns, and "
        f"charges a proportional turnover cost of {cost_bps:g} bps on the one-way turnover, "
        f"0.5 * L1(weights - prev_weights_drifted) (previous weights drifted by the realized returns). "
        # Mirror portfolio_env.py:279 exactly: gross = risky leg + cash sleeve's money-market return;
        # the cash term renders only when cash_daily_rate != 0 (default 0.0 -> the sleeve earns nothing).
        f"port_ret = (weights[:risky] . returns){cash_term} - cost"
        f"{' (cash earns a per-session money-market rate cash_daily_rate)' if cash_term else ''}.\n\n"
        f"REWARD CONTRACT (what YOU write), called once per step with ANONYMISED numpy arrays only:\n"
        f"    def reward(weights, returns, prev_weights, port_ret, info):\n"
        f"        # weights:      np.ndarray ({n_act},)  current simplex weights (assets + cash)\n"
        f"        # returns:      np.ndarray ({n_assets},)  per-asset returns realized THIS step\n"
        f"        # prev_weights: np.ndarray ({n_act},)  previous weights (for turnover)\n"
        f"        # port_ret:     float                  realized portfolio return this step (gross - cost)\n"
        f'        # info:         dict; info["reward_state"] carries YOUR state across steps (None at reset)\n'
        f"        # return (total: float, components: dict[str, float], reward_state)\n"
        f"Use numpy (`np`) only — no other imports, no file/network/OS access, no dates. The agent "
        f"maximizes `total`. Build a STATEFUL reward (via reward_state) so you can accumulate "
        f"rolling statistics across steps — a single call sees only one step."
    )


def build_prompt_set(
    env_cfg: Any,
    n_assets: int,
    prompts_dir: Path | str | None = None,
) -> PromptSet:
    """Load the prompt templates and fill ``{ENV_INTERFACE}`` (C3).

    Parameters
    ----------
    env_cfg : Any
        The environment config (``config/environment.yaml``).
    n_assets : int
        Number of risky assets in the panel (the env uses ``panel.N``, not the config's
        ``universe.n_assets``, so this is passed explicitly for an exact interface).
    prompts_dir : Path | str | None
        Directory of templates; defaults to the repo ``prompts/``.

    Returns
    -------
    PromptSet
        ``system`` (the contract + rules) and ``initial`` (with ``{ENV_INTERFACE}`` filled).
        The reflection turn is composed by the loop from ``_REFLECTION_PREAMBLE`` +
        ``schema.build_block``, so ``reflection.txt`` is not loaded here (audit medium-finding).
    """
    d = Path(prompts_dir) if prompts_dir is not None else DEFAULT_PROMPTS_DIR
    system = (d / "system.txt").read_text(encoding="utf-8")
    initial = (d / "initial_generation.txt").read_text(encoding="utf-8")
    interface = render_env_interface(env_cfg, n_assets)
    initial = initial.replace("{ENV_INTERFACE}", interface)
    return PromptSet(system=system, initial=initial)
