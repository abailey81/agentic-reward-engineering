"""Held-out fitness — reward-independent candidate selection (F.4, audit B-3).

The winning reward candidate is selected on the **validation** Deflated Sharpe of
its realized validation returns. This is deliberately INDEPENDENT of the candidate
reward's own units/value (so selection cannot be reward-hacked) and is computed on
a DIFFERENT split from the fed-back signal (training returns), so the loop cannot
tune-then-select on the same data (audit B-3).

Algorithm (FINAL_PLAN F.4 / B.7):
  - Compute the validation Deflated Sharpe (F.11 / src/inference/deflated_sharpe.py)
    on the realized validation returns, using `n_trials` for the expected-max
    Sharpe correction under guided search.
  - Optionally subtract lam * |validation-CVaR(alpha)| per the pre-registered fitness,
    where alpha comes from config (``inference.yaml: fitness.alpha``), NOT a literal.
    The registered lam is 0.0 (R22), so this penalty branch is INERT in the campaign.
  - Never touch the candidate reward's own scalar value -- the fitness depends ONLY
    on realized returns.

Guard: training-split input must be rejected (the measurement/feedback signal lives
on the training split; fitness lives on validation -- mixing them re-introduces the
overfitting B-2/B-3 are designed to prevent).

Audit refs: B-3 (loop-once, validation selection independent of reward units).
"""

from __future__ import annotations

import numpy as np


class NoEligibleWinnerError(RuntimeError):
    """Every candidate in an arm failed the R115 winner-eligibility execution floor.

    Distinct from "the arm produced no candidates" (which the selector reports by returning
    ``None``): here candidates EXIST but not one of them ran its authored reward for enough of its
    training to represent the arm. The two demand different responses, so they must not collapse
    into one signal.

    Subclasses ``RuntimeError`` so existing broad handlers keep working. Lives here rather than in
    ``scripts/run_campaign.py`` so the cluster orchestrator can catch it by TYPE without importing
    the scripts module — matching on an error message would be brittle.
    """


def held_out_fitness(
    returns: np.ndarray,
    n_trials: int,
    split: str = "val",
    lam: float = 0.0,
    rng: np.random.Generator | None = None,
    var_sr: float | None = None,
    cvar_alpha: float | None = None,
) -> float:
    """Validation Deflated Sharpe (optionally penalized by validation CVaR).

    Parameters
    ----------
    returns:
        Realized portfolio returns on the VALIDATION split only (anonymized
        array; no tickers/dates). Depends ONLY on realized returns -- never on
        any reward value.
    n_trials:
        Number of trials for the Deflated-Sharpe expected-max correction.
    split:
        Must be ``"val"``; any other value raises ``ValueError`` (selection
        lives on the validation split, audit B-2/B-3).
    lam:
        Optional weight on ``|validation-CVaR(alpha)|`` (pre-registered fitness);
        ``0`` — the REGISTERED value (R22) — disables the penalty entirely.
    rng:
        Accepted for interface symmetry; the fitness is deterministic given the
        returns and is not used.
    var_sr:
        Optional cross-trial Sharpe variance forwarded verbatim to
        ``deflated_sharpe_ratio`` (Rank 16). ``None`` (the default — so the
        per-candidate WIRED selection path is unchanged) keeps the within-series
        sampling-variance PROXY; an explicit empirical cross-candidate Sharpe
        dispersion ``np.var(per_candidate_val_sharpes, ddof=1)`` selects the
        canonical Bailey-Lopez de Prado DSR. Threaded at ANALYSIS time (the
        population variance over ALL an arm's candidates is only knowable once
        the arm finishes -- see ``scripts/analyze_campaign.py: winner_dsr``),
        not inside the per-candidate loop.
    cvar_alpha:
        Tail level for the optional CVaR penalty. ``None`` (the default) reads
        ``inference.yaml: fitness.alpha`` — and ONLY when ``lam != 0``, so the
        registered ``lam == 0`` hot path never touches config.

    Returns
    -------
    float
        ``deflated_sharpe_ratio(returns, n_trials, var_sr=var_sr)
        - lam * |cvar(returns, cvar_alpha)|`` — the penalty term is dropped when
        ``lam == 0`` (the registered value) and treated as 0 when the CVaR is
        non-finite.

    Raises
    ------
    ValueError
        If ``split != "val"``.
    """
    if split != "val":
        raise ValueError(
            f"held_out_fitness must be computed on the validation split; "
            f"got split={split!r} (selection lives on validation, audit B-2/B-3)"
        )

    # Lazy imports to avoid import-order issues within the inference package.
    from src.inference.bootstrap import cvar
    from src.inference.deflated_sharpe import deflated_sharpe_ratio

    r = np.asarray(returns, dtype=float)
    dsr = deflated_sharpe_ratio(r, n_trials, var_sr=var_sr)
    if lam == 0.0:
        return float(dsr)
    # Guard the CVaR penalty against a non-finite tail (Rank 18): an empty or all-non-finite
    # validation series yields cvar(r) == nan, and lam*abs(nan) would POISON the fitness with NaN
    # (silently ranking the candidate above finite ones under argmax / breaking sorts). Treat a
    # non-finite CVaR as a zero penalty -- the (already non-finite) DSR alone then governs selection.
    # CVaR penalty level from config (inference.yaml: fitness.alpha), NOT a hardcoded 0.05; caller may
    # override via cvar_alpha. Read here (only when lam != 0, so the lam==0 hot path stays config-free).
    if cvar_alpha is None:
        from src.utils.config import load_config

        cvar_alpha = float(load_config("inference").get("fitness", {}).get("alpha", 0.05))
    c = cvar(r, cvar_alpha)
    penalty = lam * abs(c) if np.isfinite(c) else 0.0
    return float(dsr - penalty)
