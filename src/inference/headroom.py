"""Validation-headroom (oracle-selection) bound — report-only (PREREGISTRATION §2a micro-anchor (e)).

A predicted null invites the objection *"perhaps no better reward existed to find."* This module answers
it directly from the archive: over ALL authored candidates' **validation** returns, the
**oracle-selection frontier** — the best achievable validation CVaR-5% (and validation Deflated Sharpe)
had selection been clairvoyant — against what each arm's actual frozen selection (λ = 0: max
``val_fitness``) achieved. A material frontier-minus-achieved gap establishes that headroom EXISTED in
the authored search space; combined with the information-gap instrument (micro-anchor (d)) and SQ1 it
completes the quantitative localisation of the break (information present → headroom present → designer
unresponsive).

Estimands, per arm and pooled:

* **CVaR leg** — per candidate the empirical ``cvar(val_returns, level)`` (REUSING
  :func:`src.inference.bootstrap.cvar`); frontier = the max (least-negative tail), achieved = the
  selected winner's value, ``gap = frontier − achieved ≥ 0``.
* **DSR leg** — per candidate the Deflated Sharpe over its own validation returns (REUSING
  :func:`src.inference.deflated_sharpe.deflated_sharpe_ratio`), deflated by the arm's FULL searched
  multiplicity ``n_trials`` and the cross-candidate per-period-Sharpe dispersion ``var_sr`` — the exact
  canonical convention of ``analyze_campaign.winner_dsr`` (per-period Sharpes via ``_sample_moments``,
  ``ddof=1`` dispersion), so the achieved value coincides with the reported winner DSR.

The bootstrap CI on each gap resamples CANDIDATES (percentile, seeded): within each resample the
selection rule is re-applied (winner = max ``val_fitness`` among the resampled candidates) and the
frontier re-taken, so the CI reflects the joint sampling variability of frontier AND selection. The DSR
deflation constants (``n_trials``, ``var_sr``) are FIXED at their full-sample values inside the bootstrap
— the multiplicity the real search faced is a design constant, not a resampled quantity. The ``pooled``
block treats the pool of all analysed arms' candidates as one selection population (what a pooled λ = 0
selection would have achieved vs the pooled frontier).

VALIDATION data only: records carrying test-leg fields (per-seed ``test_returns`` re-runs, the frozen
marker) are EXCLUDED up front — the sealed test leg is never touched (a winners-only test-leg analogue is
registered as an appendix caveat, not computed here). Report-only, DISJOINT from the frozen ``m = 6``
family; degrades honestly (``skipped`` arms, ``no_data`` overall) when validation vectors are absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.inference.bootstrap import cvar
from src.inference.deflated_sharpe import _sample_moments, deflated_sharpe_ratio

__all__ = ["validation_headroom"]


def _is_validation_search_record(record: dict[str, Any]) -> bool:
    """True iff ``record`` is a SEARCH-leg candidate (mirrors ``analyze_campaign._is_search_candidate``).

    Excludes the frozen-winner marker (``frozen: True``) and any record carrying test-leg returns —
    the fail-safe that keeps this instrument on VALIDATION data even if a caller passes the full
    campaign tree.
    """
    if record.get("frozen"):
        return False
    metrics = record.get("metrics") or {}
    if metrics.get("test_returns") is not None or record.get("test_returns") is not None:
        return False
    return True


def _candidates(records: list[dict[str, Any]], arm: str) -> tuple[list[dict[str, Any]], int, int]:
    """One arm's usable candidates in deterministic (generation, candidate_id) order.

    Returns ``(usable, n_trials, n_excluded)``: ``usable`` are the search candidates with a finite
    non-empty ``val_returns`` vector AND a finite ``val_fitness`` (the selection statistic); ``n_trials``
    is the FULL search-candidate count (vectorless candidates still consumed a budget slot, so they
    count toward the DSR expected-max multiplicity — the winner_dsr #32 convention); ``n_excluded``
    counts the search candidates dropped for a missing/degenerate vector or fitness.
    """
    search = sorted(
        [r for r in records if r.get("arm") == arm and _is_validation_search_record(r)],
        key=lambda r: (int(r.get("generation", 0) or 0), str(r.get("candidate_id", r.get("run_id", "")))),
    )
    usable: list[dict[str, Any]] = []
    for r in search:
        metrics = r.get("metrics") or {}
        vr = metrics.get("val_returns")
        fitness = metrics.get("val_fitness")
        if vr is None or fitness is None:
            continue
        arr = np.asarray(vr, dtype=float).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)) or not np.isfinite(float(fitness)):
            continue
        usable.append(r)
    return usable, len(search), len(search) - len(usable)


def _gap_leg(
    values: np.ndarray,
    fitness: np.ndarray,
    ids: list[str],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Frontier / achieved / gap (+ percentile bootstrap CI, re-applying selection per resample).

    ``values`` is the leg's per-candidate metric (CVaR or DSR; HIGHER is better for both), ``fitness``
    the selection statistic, ``ids`` the candidate ids (ties break to the first in the deterministic
    input order, matching ``np.argmax`` / the selection loop).
    """
    n = int(values.size)
    winner_i = int(np.argmax(fitness))
    oracle_i = int(np.argmax(values))
    frontier = float(values[oracle_i])
    achieved = float(values[winner_i])
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sel = idx[int(np.argmax(fitness[idx]))]
        boots[b] = float(np.max(values[idx]) - values[sel])
    ci_low, ci_high = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {
        "frontier": frontier,
        "achieved": achieved,
        "gap": float(frontier - achieved),
        "gap_ci_low": ci_low,
        "gap_ci_high": ci_high,
        "oracle_id": ids[oracle_i],
        "selected_id": ids[winner_i],
        "oracle_is_selected": bool(oracle_i == winner_i),
    }


def _dsr_values(
    vectors: list[np.ndarray], n_trials: int, periods_per_year: int
) -> tuple[np.ndarray, float]:
    """Per-candidate canonical DSRs (shared ``n_trials`` + cross-candidate ``var_sr``) and that ``var_sr``.

    Per-period Sharpes come from the SAME ``_sample_moments`` the DSR uses internally (ddof=1), so the
    dispersion and the deflated statistic share one convention — exactly ``winner_dsr``'s recompute.
    """
    sharpes = np.asarray([_sample_moments(v)[0] for v in vectors], dtype=float)
    var_sr = float(np.var(sharpes, ddof=1))
    dsrs = np.asarray(
        [
            deflated_sharpe_ratio(v, n_trials, var_sr=var_sr, periods_per_year=periods_per_year)
            for v in vectors
        ],
        dtype=float,
    )
    return dsrs, var_sr


def validation_headroom(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] | list[str] | None = None,
    cvar_level: float = 0.05,
    n_boot: int = 2000,
    min_candidates: int = 3,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The validation-headroom (oracle-selection) bound over an archive (micro-anchor (e)).

    Parameters
    ----------
    records : list of dict
        Per-candidate run records; only ``metrics.val_returns`` / ``metrics.val_fitness`` are read.
        Test-leg records are excluded by the internal fail-safe (never touched).
    arms : sequence of str or None
        Arms to report (default: every arm present among the search records, sorted). Arms below
        ``min_candidates`` usable candidates are reported ``skipped``, never fabricated.
    cvar_level : float
        Tail level for the CVaR leg (the frozen headline level is passed by the caller from config).
    n_boot, min_candidates, periods_per_year, rng
        Bootstrap replicates; the per-arm floor; DSR annualisation convention (forwarded, invariant);
        the seeded generator (default ``default_rng(0)`` — byte-deterministic).

    Returns
    -------
    dict
        ``{"status", "executed", "cvar_level", "selection_rule", "per_arm": {arm: {...}}, "pooled",
        "n_boot"}`` — per arm the CVaR and DSR legs each carry ``frontier / achieved / gap /
        gap_ci_low / gap_ci_high / oracle_id / selected_id``; the DSR leg is ``skipped`` below two
        usable candidates (no cross-trial dispersion). ``{"status": "no_data", ...}`` when no arm
        reaches the floor.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if arms is None:
        arms = tuple(
            sorted({str(r.get("arm")) for r in records if r.get("arm") and _is_validation_search_record(r)})
        )

    out: dict[str, Any] = {
        "status": "ok",
        "executed": True,
        "cvar_level": float(cvar_level),
        "selection_rule": "max metrics.val_fitness (the frozen lambda=0 selection)",
        "n_boot": int(n_boot),
        "per_arm": {},
    }
    pooled_vals: list[float] = []
    pooled_vecs: list[np.ndarray] = []
    pooled_fitness: list[float] = []
    pooled_ids: list[str] = []
    pooled_n_trials = 0

    for arm in arms:
        usable, n_trials, n_excluded = _candidates(records, arm)
        entry: dict[str, Any] = {
            "n_candidates": len(usable),
            "n_trials": n_trials,
            "n_excluded": n_excluded,
        }
        if len(usable) < min_candidates:
            entry.update(
                status="skipped",
                reason=(
                    f"need >= {min_candidates} candidates with validation vectors + fitness; "
                    f"got {len(usable)}"
                ),
            )
            out["per_arm"][arm] = entry
            continue

        vectors = [np.asarray(r["metrics"]["val_returns"], dtype=float).ravel() for r in usable]
        fitness = np.asarray([float(r["metrics"]["val_fitness"]) for r in usable])
        ids = [str(r.get("candidate_id", r.get("run_id", ""))) for r in usable]
        cvar_vals = np.asarray([cvar(v, cvar_level) for v in vectors], dtype=float)

        entry["status"] = "ok"
        entry["cvar"] = _gap_leg(cvar_vals, fitness, ids, n_boot=n_boot, rng=rng)
        if len(usable) >= 2:
            dsr_vals, var_sr = _dsr_values(vectors, n_trials, periods_per_year)
            entry["dsr"] = _gap_leg(dsr_vals, fitness, ids, n_boot=n_boot, rng=rng)
            entry["dsr"]["var_sr"] = var_sr
        else:  # pragma: no cover - unreachable while min_candidates >= 2; kept as a guard
            entry["dsr"] = {"status": "skipped", "reason": "need >= 2 candidates for var_sr"}
        out["per_arm"][arm] = entry

        pooled_vals.extend(float(x) for x in cvar_vals)
        pooled_vecs.extend(vectors)
        pooled_fitness.extend(float(x) for x in fitness)
        pooled_ids.extend(f"{arm}:{cid}" for cid in ids)
        pooled_n_trials += n_trials

    if not pooled_ids:
        return {
            "status": "no_data",
            "executed": False,
            "reason": (
                f"no arm reached the {min_candidates}-candidate floor with usable "
                "val_returns + val_fitness"
            ),
            "per_arm": out["per_arm"],
            "cvar_level": float(cvar_level),
        }

    # Pooled block: the pool of all analysed arms' candidates as ONE selection population — what a
    # pooled lambda=0 selection would have achieved against the pooled oracle frontier.
    pooled_fit = np.asarray(pooled_fitness)
    out["pooled"] = {
        "n_candidates": len(pooled_ids),
        "n_trials": pooled_n_trials,
        "cvar": _gap_leg(
            np.asarray(pooled_vals), pooled_fit, pooled_ids, n_boot=n_boot, rng=rng
        ),
    }
    if len(pooled_ids) >= 2:
        pooled_dsr, pooled_var_sr = _dsr_values(pooled_vecs, pooled_n_trials, periods_per_year)
        out["pooled"]["dsr"] = _gap_leg(pooled_dsr, pooled_fit, pooled_ids, n_boot=n_boot, rng=rng)
        out["pooled"]["dsr"]["var_sr"] = pooled_var_sr
    return out
