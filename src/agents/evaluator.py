"""Reward evaluator — the C1 adapter that lets the SEARCH arms consume matched compute.

The discovery-loop arms (LLM-distributional / scalar / placebo / scalar+CVaR5) train and
score *inside* ``run_loop``. The two **search** arms (H4: random-search-over-code, and
Bayesian-optimization-over-template; ``src/search/*``) were written against an injected
``fitness_fn`` that maps a *reward* (or a template coefficient vector) to a scalar — but
the repo never built the adapter that turns "a reward function" into "the fixed SAC
agent's validation Deflated Sharpe" (adversarial-review finding **C1**). Without it the
search arms would either run on toy closures (scientifically void) or crash, and the
"matched compute across all NINE arms" (R108 7 -> 9) property that licenses the H4 comparison
(PREREGISTRATION §3) would not actually hold.

This module is that adapter. ``evaluate_reward`` runs the IDENTICAL
train -> rollout -> select pipeline as the LLM arms::

    bundle = env_builder(reward_fn)             # train/val envs on the shared PIT panel
    policy = agent_trainer(bundle.train_env())  # FIXED SAC, fixed step budget (audit A-1/B-5)
    val    = bundle.val_returns(policy)         # realized validation returns
    return fitness_fn(val, n_trials)            # validation Deflated Sharpe (Bailey-LdP 2014)

so a random-search reward and a BO-template reward are evaluated exactly as an
LLM-designed reward is — the only thing that differs across H1/H4 is *how the reward was
produced*, which is the whole point of the comparison.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = [
    "evaluate_reward",
    "evaluate_reward_with_returns",
    "make_reward_evaluator",
    "make_template_evaluator",
]


def evaluate_reward_with_returns(
    reward_fn: Any,
    env_builder: Callable[[Any], Any],
    agent_trainer: Callable[[Any], Any],
    fitness_fn: Callable[[np.ndarray, int], float],
    n_trials: int,
) -> tuple[float, np.ndarray]:
    """Train the fixed SAC on ``reward_fn``; return ``(fitness, val_returns)``.

    Identical train -> rollout -> select pipeline as :func:`evaluate_reward`, but ALSO
    surfaces the realized per-period VALIDATION return vector (the input to ``fitness_fn``).
    Rank 3 (PBO/CSCV) consumes that per-candidate per-period vector for the search arms,
    exactly as the LLM loop already archives ``metrics['val_returns']`` per candidate;
    :func:`evaluate_reward` delegates here so the scalar-only contract is unchanged.

    Returns
    -------
    tuple[float, numpy.ndarray]
        The candidate reward's validation fitness and the realized validation returns
        (a 1-D array on the validation split).
    """
    bundle = env_builder(reward_fn)
    policy = agent_trainer(bundle.train_env())
    val_returns = np.asarray(bundle.val_returns(policy), dtype=float)
    fitness = float(fitness_fn(val_returns, n_trials))
    return fitness, val_returns


def evaluate_reward(
    reward_fn: Any,
    env_builder: Callable[[Any], Any],
    agent_trainer: Callable[[Any], Any],
    fitness_fn: Callable[[np.ndarray, int], float],
    n_trials: int,
) -> float:
    """Train the fixed SAC on ``reward_fn`` and return its validation fitness.

    Parameters
    ----------
    reward_fn : Any
        A reward callable conforming to the contract (already sandbox-validated for the
        code-search arm; constructed from the template for the BO arm).
    env_builder : Callable
        ``(reward_fn) -> EnvBundle`` (``src.env.runner.make_env_builder``).
    agent_trainer : Callable
        ``(train_env) -> policy`` (``src.agents.trainer.make_agent_trainer``).
    fitness_fn : Callable
        ``(val_returns, n_trials) -> float`` (``src.selection.fitness.held_out_fitness``).
    n_trials : int
        Trial count for the Deflated-Sharpe expected-max correction.

    Returns
    -------
    float
        The candidate reward's validation fitness (validation Deflated Sharpe).
    """
    fitness, _val_returns = evaluate_reward_with_returns(
        reward_fn, env_builder, agent_trainer, fitness_fn, n_trials
    )
    return fitness


def make_reward_evaluator(
    env_builder: Callable[[Any], Any],
    agent_trainer: Callable[[Any], Any],
    fitness_fn: Callable[[np.ndarray, int], float],
    n_trials: int,
) -> Callable[[Any], float]:
    """Return ``(reward_fn) -> fitness`` for random-search-over-code (H4a).

    This is the ``fitness_fn`` that ``src.search.random_search.random_search_over_code``
    expects (it samples reward callables and scores each with one argument).
    """

    def _eval(reward_fn: Any) -> float:
        return evaluate_reward(reward_fn, env_builder, agent_trainer, fitness_fn, n_trials)

    return _eval


def make_template_evaluator(
    params_to_reward: Callable[[Any], Any],
    env_builder: Callable[[Any], Any],
    agent_trainer: Callable[[Any], Any],
    fitness_fn: Callable[[np.ndarray, int], float],
    n_trials: int,
) -> Callable[[Any], float]:
    """Return ``(coeffs) -> fitness`` for Bayesian-optimization-over-template (H4b).

    ``params_to_reward`` instantiates the fixed parametric reward family
    (``src.baselines`` / the H4 reward-family) from a coefficient vector; the resulting
    reward is then evaluated through the identical pipeline. This is the
    ``template_eval_fn`` that ``src.search.bayes_opt.bayes_opt_over_template`` expects.
    """

    def _eval(coeffs: Any) -> float:
        reward_fn = params_to_reward(coeffs)
        return evaluate_reward(reward_fn, env_builder, agent_trainer, fitness_fn, n_trials)

    return _eval
