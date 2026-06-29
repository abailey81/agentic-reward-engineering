"""Campaign overfitting analysis — PBO/CSCV per arm (PREREGISTRATION §10; FINAL_PLAN B.9).

This is the CAMPAIGN analogue of ``scripts/analyze_results.py`` (which is correctly scoped
to the 1-seed DIRECTIONAL go/no-go and stays that way). Here we compute the project's
**primary overfitting guard**: the Probability of Backtest Overfitting via Combinatorially
Symmetric Cross-Validation (PBO/CSCV; Bailey et al. 2017), implemented in
``src.inference.overfitting.pbo``.

What PBO is computed over (the design, matching the spec)
--------------------------------------------------------
PBO is computed **PER ARM over that arm's search CANDIDATES**. For one arm we stack its
candidates' per-period **validation** return vectors (``metrics['val_returns']``) into a
performance matrix of shape ``(T_val, N_candidates)`` and ask, across all CSCV in-sample /
out-of-sample block splits, how often the IS-best candidate ranks below the OOS median —
i.e. whether the within-arm "pick the best candidate by validation fitness" selection
**overfits** (FINAL_PLAN B.9 line 100: "partition the performance matrix ... best-in-sample
config's OOS relative rank ... PBO = fraction with logit < 0"; ``overfitting.py`` docstring:
"a family of candidate strategies", trial count ill-defined under guided search).

This is distinct from the CPCV-on-winners evaluation-fold scheme (PREREGISTRATION §6: "CPCV
is applied to the fixed winners afterward for inference") which feeds the difference tests,
NOT this overfitting metric. ``config/inference.yaml`` keeps them separate: ``pbo.n_blocks``
(this metric) vs ``splits.cpcv`` (the evaluation folds).

Inputs
------
Results are read ONLY through ``src.io.results.load_all`` (audit C-1). All nine arms persist
the per-candidate validation vector ``metrics['val_returns']`` during search (LLM arms via
``src/llm/loop.py``; search arms via ``scripts/run_prototype.py`` /
``src/orchestration/parallel.py``). ``n_blocks`` is read from ``config/inference.yaml``
``pbo.n_blocks`` (= 16).

DIRECTIONAL caveat: like the prototype, a single-seed development-split campaign is a
go/no-go on the mechanism. The PBO table is a campaign tool; it is meaningful only on the
real multi-candidate per-arm archives.

Campaign-inference additions (Rank 8 — operate on the TEST leg)
--------------------------------------------------------------
The functions below consume the per-(arm, seed) **frozen-winner TEST records** that
``scripts/run_campaign.py`` writes (each carrying ``metrics['test_returns']`` — the
realized per-step held-out 2020-2026 portfolio returns). They wire the FROZEN
pre-registration's selection-aware machinery into the campaign:

- :func:`collect_family_pvalues` — enumerate the pre-registered **arm-contrast × held-out
  metric** family (Sharpe + CVaR at the pre-reg levels) via the rliable PER-SEED difference
  test (per-seed scores → IQM → paired stratified bootstrap over the shared training seeds;
  ``src.inference.bootstrap.paired_seed_difference_test``), then Benjamini-Hochberg the
  p-values at ``config/inference.yaml: multiplicity.q = 0.05`` (PREREGISTRATION §10 + R16;
  FINAL_PLAN B.9 line 104, "Multiple-testing correction across the arm × metric family").
- :func:`romano_wolf_joint` — a JOINT Romano-Wolf stepdown that draws ONE shared
  **SEED-index resample** per replication and evaluates every hypothesis's per-seed IQM
  difference on it (so the cross-hypothesis dependence the stepdown relies on is preserved
  AND the across-seed variance is carried), feeding the existing
  ``src.inference.multiple_testing.romano_wolf`` stepdown. See that module's note in
  :func:`romano_wolf_joint` for why the shared resample is required (R16, 2026-06-20:
  the per-seed analogue of the prior series block-bootstrap).
- :func:`h2_conjunction` — the pre-registered **HEADLINE** test. H2 (PREREGISTRATION §1;
  FINAL_PLAN B.6 line 83) holds only if distributional beats scalar **and** survives
  beyond the **placebo** (information ≠ token-count) **and** beyond **scalar+CVaR-5%**
  (tail-shape ≠ any-downside-number): ``H2_supported`` iff ALL THREE legs reject in the
  predicted direction *after* the family-wise/FDR correction.
- :func:`benchmark_floor` — the DeMiguel 1/N floor (PREREGISTRATION §9 benchmark suite;
  §10). Rolls the EIGHT distinct benchmark weight policies (equal_weight=1/N, mean_variance,
  risk_parity, hrp, minimum_variance, maximum_diversification, inverse_volatility,
  cross_sectional_momentum; R19) through the IDENTICAL ``PortfolioEnv`` + cost via a
  minimal :class:`WeightPolicy` shim and ``src.env.runner.rollout_port_returns``, then
  gates the frozen winner's test Deflated Sharpe **>** the best benchmark's. This is a
  POST-FREEZE report-only gate — it NEVER re-selects the winner.

All four MATCH the frozen pre-registration; they do not invent a new family or conjunction
(STOP-AND-FLAG notes are inlined where the spec leaves a choice).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pandas is imported lazily at runtime (inside delisting_band); only the type hints need it
    import pandas as pd

__all__ = [
    "ARMS",
    "H2_CONTRASTS",
    "H4_CONTRASTS",
    "WeightPolicy",
    "assert_realized_family_matches_frozen",
    "beat_human_baseline",
    "benchmark_floor",
    "benchmark_floor_markdown",
    "build_perf_matrix",
    "campaign_pbo",
    "campaign_pbo_dsr",
    "collect_family_pvalues",
    "compute_accounting",
    "compute_accounting_markdown",
    "cross_hypothesis_multiplicity",
    "cross_hypothesis_multiplicity_markdown",
    "delisting_band",
    "delisting_band_markdown",
    "divergence_markdown",
    "divergence_report",
    "dsr_effective_n",
    "dsr_effective_n_markdown",
    "evt_consistency_guard",
    "evt_consistency_markdown",
    "h1_beat_human_markdown",
    "h2_conjunction",
    "h2_markdown",
    "h2_sharpe_rf_robustness",
    "h2_rf_robustness_markdown",
    "h2_structure_control",
    "h2_tost",
    "h2_tost_dsr",
    "h2_tost_dsr_markdown",
    "h2_tost_markdown",
    "h3_iterative_vs_singleshot",
    "h3_markdown",
    "h4_markdown",
    "h4_search_controls",
    "information_gap_markdown",
    "legible_format_responsiveness_markdown",
    "load_campaign_records",
    "mediation_markdown",
    "named_vs_blinded_structural_markdown",
    "pbo_dsr_markdown",
    "pbo_markdown",
    "responsiveness_markdown",
    "reward_taxonomy_markdown",
    "romano_wolf_joint",
    "validation_headroom_markdown",
    "winner_dsr",
    "winner_dsr_markdown",
]

_LOG = logging.getLogger(__name__)

#: The NINE pre-registered arms (PREREGISTRATION §3 "The nine arms"; `config/arms.yaml`). PBO is
#: reported for each that carries enough candidates with validation vectors; others are reported as
#: skipped.
#:
#: NOTE: 2026-07-27 (deep review, finding #102): this listed SEVEN — `cma_es` and `tpe` were never added
#: when the roster grew 7 -> 9 on 2026-07-26 (the same expansion as #100's H4 2 -> 4) — while the
#: comment cited "PREREGISTRATION §3" as its authority, and §3's own heading reads "The nine arms".
#: This constant is the DEFAULT for five analyses: `campaign_pbo` (the PRIMARY overfitting
#: diagnostic), `campaign_pbo_dsr`, `winner_dsr`, `compute_accounting` and
#: `model_confidence_set_report` (the MCS, whose COMPOSITION depends on which arms are entered). An
#: arm missing from this tuple produces no row AT ALL — not even a "skipped" row — so two registered
#: arms were silently invisible to all five, leaving no trace in the output.
#: Guarded by `test_ARMS_matches_the_frozen_arms_yaml_roster`, mirroring the config-vs-code assertion
#: that already keeps `_BENCHMARK_NAMES` honest.
ARMS: tuple[str, ...] = (
    "distributional",
    "scalar",
    "placebo",
    "scalar_cvar5",
    "placebo_shuffled",
    "random_search",
    "bayes_opt",
    "cma_es",
    "tpe",
)


def _val_returns(record: dict[str, Any]) -> np.ndarray | None:
    """Extract a candidate's per-period validation return vector, or ``None``.

    The vector rides inside ``metrics['val_returns']`` (all nine arms write it during
    search). Returns ``None`` when the field is absent, ``None``, or not a 1-D vector of
    length >= 1 — those candidates are skipped (with a logged warning) by
    :func:`build_perf_matrix`.
    """
    metrics = record.get("metrics") or {}
    vr = metrics.get("val_returns")
    if vr is None:
        return None
    arr = np.asarray(vr, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _power_analysis() -> Any:
    """Lazy import of the sibling ``scripts/power_analysis.py`` module (the DSR-units TOST + Sharpe→DSR map).

    Imported lazily (not at module top) so a records-only analyze run that never touches the DSR-units TOST
    pays no import cost, and so ``scripts/`` need not already be on ``sys.path`` (e.g. ``pytest`` importing
    ``analyze_campaign`` directly). Adds this file's own directory to ``sys.path`` once, then imports.
    """
    import importlib
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("power_analysis")


def _frozen_equiv_margin(fallback: float = 0.05) -> float:
    """READ the FROZEN equivalence margin / SESOI (validation-DSR units) from config — single source of
    truth: code reads config and never hardcodes. ``inference.equivalence_margin`` (= SESOI = 0.05;
    PREREGISTRATION §10 R12). Falls back to the literal only if config is unreadable."""
    try:
        from src.utils.config import load_config

        inf = load_config("preregistration").get("inference", {})
        return float(inf.get("equivalence_margin", inf.get("sesoi", fallback)))
    except Exception:  # noqa: BLE001 - config unreadable (isolated import) -> documented fallback
        return float(fallback)


def _ni_margins() -> dict[str, Any]:
    """The NON-INFERIORITY margins for validity-tier node N2, in the H2-RA legs' OWN units.

    A16, pre-specified 2026-08-01T13:01:15Z while VERIFIABLY BLIND (0 of 3 H2-RA legs computable at
    HEAD 57c5ecc4) and recorded before any code was written.

    WHAT IS REGISTERED, AND WHY THE PRIMARY IS THE PERMISSIVE ONE. Node ``N2_h2_ra`` registers
    ``equivalence: tost_0.05_dsr``, which NAMES the function :func:`h2_tost_dsr`. What that function
    DOES is therefore the registered conversion, and it defaults to
    ``power_analysis.VALIDATION_TRACK_LENGTH`` (694). So the registered margin is the SESOI divided
    by ``k(694) = 0.661571`` — about 0.0756 annualised Sharpe. Three independent routes agree:
    the frozen config records ``sesoi_derivation.sesoi_ann_sharpe_equiv = 0.0756`` and
    ``dsr_per_ann_sharpe = 0.6616``; ``src/selection/fitness.py`` refuses to compute the DSR on
    anything but the validation split, so 694 is the only track length a validation-DSR is defined
    at; and re-deriving the executed windows through the campaign's own code path gives
    val (3081, 3775) = 694 sessions and test (3835, 5406) = 1571.

    NOTE: STATED PLAINLY BECAUSE IT CUTS AGAINST THE ANALYST: 0.0756 is the WIDER margin, and for
    ``H0: theta <= -delta`` a wider margin makes the node EASIER to reject. Here "use the registered
    value" and "use the conservative value" point in OPPOSITE directions. That is exactly why the
    CONSERVATIVE margin — the same SESOI mapped at the executed TEST track length,
    ``k(1571) = 0.995771`` giving about 0.0502 — is computed alongside it and reported as a
    pre-specified sensitivity, and why both k values and both track lengths are returned rather
    than just the numbers. A margin chosen for its conservativeness rather than its registration
    would be a researcher degree of freedom in the other direction; reporting both removes the
    choice instead of making it.
    """
    pa = _power_analysis()
    sesoi = _frozen_equiv_margin()
    k_val = float(pa.sharpe_mde_to_dsr(1.0, track_length=int(pa.VALIDATION_TRACK_LENGTH)))
    k_test = float(pa.sharpe_mde_to_dsr(1.0, track_length=int(pa.TEST_TRACK_LENGTH)))
    return {
        "sesoi_dsr": float(sesoi),
        "primary": sesoi / k_val,
        "primary_track_length": int(pa.VALIDATION_TRACK_LENGTH),
        "primary_sharpe_to_dsr_factor": k_val,
        "primary_basis": "registered: tost_0.05_dsr at its own default validation track length",
        "conservative": sesoi / k_test,
        "conservative_track_length": int(pa.TEST_TRACK_LENGTH),
        "conservative_sharpe_to_dsr_factor": k_test,
        "conservative_basis": "sensitivity: the same SESOI at the EXECUTED test track length",
        "units": "annualised Sharpe (the H2-RA legs' own statistic)",
    }


def _is_search_candidate(record: dict[str, Any]) -> bool:
    """True iff ``record`` is a SEARCH-leg candidate (the per-candidate population PBO/DSR range over).

    Since :func:`load_campaign_records` now walks the WHOLE campaign tree from one root (so the H2/floor
    test leg is reachable alongside the search leg — leg-disjoint defect fix), the per-arm record list
    mixes three record kinds. The PBO column population and the canonical-DSR expected-max multiplicity
    (#32) must count ONLY the search candidates the selection actually faced, never the frozen-winner TEST
    records (carry ``metrics['test_returns']``, one shared ``<arm>-winner`` candidate_id per seed) nor the
    frozen marker (``frozen: True``, only ``val_fitness``). A search candidate is any record that is
    NEITHER: it carries ``val_fitness`` and (usually) ``val_returns`` and never sets ``frozen``/
    ``test_returns`` (``scripts/run_prototype.py`` ``_archive_record``; ``src/llm/loop.py``). A vectorless
    search candidate (``val_returns`` omitted) still counts toward the multiplicity, per #32.
    """
    if record.get("frozen"):
        return False
    metrics = record.get("metrics") or {}
    if metrics.get("test_returns") is not None or record.get("test_returns") is not None:
        return False
    return True


def build_perf_matrix(records: list[dict[str, Any]], arm: str) -> np.ndarray:
    """Stack one arm's candidates' validation vectors into a ``(T_val, N)`` matrix.

    Parameters
    ----------
    records : list of dict
        Per-candidate run records (as returned by ``src.io.results.load_all``). Records
        for OTHER arms are ignored; only those with ``record['arm'] == arm`` are used.
    arm : str
        The arm whose candidate population forms the columns.

    Returns
    -------
    numpy.ndarray
        A ``(T_val, N_candidates)`` performance matrix: column ``j`` is candidate ``j``'s
        per-period validation returns. Candidates lacking a usable
        ``metrics['val_returns']`` vector are SKIPPED with a logged warning. When the
        candidates' vectors differ in length they are aligned to the common (minimum)
        length by taking the leading ``T_min`` rows (all validation vectors start at the
        same validation-window session, so the leading rows are the shared calendar
        periods); a length mismatch is logged. Returns an empty ``(0, 0)`` array when no
        candidate carries a usable vector.

    Notes
    -----
    Candidates are ordered by ``candidate_id`` (then ``run_id``) for determinism, so the
    column order is stable across runs. The matrix is the input to
    ``src.inference.overfitting.pbo`` via :func:`campaign_pbo`.
    """
    # SEARCH candidates only: load_campaign_records walks the whole campaign tree, so the per-arm list now
    # also holds the frozen-winner TEST records + the frozen marker (no val_returns). Those are NOT search
    # candidates and must not be counted/warned as "skipped" columns (leg-disjoint loader fix); _val_returns
    # would drop them anyway, but excluding them here keeps n_candidates and the skip log honest.
    arm_records = [r for r in records if r.get("arm") == arm and _is_search_candidate(r)]
    arm_records.sort(key=lambda r: (str(r.get("candidate_id", "")), str(r.get("run_id", ""))))

    vectors: list[np.ndarray] = []
    n_skipped = 0
    for r in arm_records:
        vec = _val_returns(r)
        if vec is None:
            n_skipped += 1
            _LOG.warning(
                "arm %r: candidate %r has no usable metrics['val_returns'] — skipped",
                arm,
                r.get("candidate_id", r.get("run_id", "?")),
            )
            continue
        vectors.append(vec)

    if n_skipped:
        _LOG.warning("arm %r: skipped %d candidate(s) lacking a validation vector", arm, n_skipped)

    if not vectors:
        return np.empty((0, 0), dtype=float)

    lengths = {v.size for v in vectors}
    t_min = min(lengths)
    if len(lengths) > 1:
        _LOG.warning(
            "arm %r: candidate validation vectors differ in length %s — aligning to "
            "the common leading %d periods",
            arm,
            sorted(lengths),
            t_min,
        )
    aligned = [v[:t_min] for v in vectors]
    # Columns = candidates, rows = periods -> shape (T_val, N_candidates).
    return np.column_stack(aligned)


def campaign_pbo(
    records: list[dict[str, Any]],
    *,
    n_blocks: int,
    arms: tuple[str, ...] | list[str] = ARMS,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute PBO/CSCV per arm over each arm's candidate performance matrix.

    Parameters
    ----------
    records : list of dict
        All per-candidate run records (``src.io.results.load_all``); split by ``arm``.
    n_blocks : int
        Number of CSCV blocks ``S`` (``config/inference.yaml: pbo.n_blocks`` = 16). Passed
        straight to ``src.inference.overfitting.pbo``.
    arms : sequence of str, optional
        Arms to report (default: the six pre-registered arms). An arm absent from
        ``records`` is reported as skipped.
    rng : numpy.random.Generator, optional
        Forwarded to ``pbo`` (used only if the CSCV split count exceeds its internal cap).

    Returns
    -------
    dict[str, dict]
        ``{arm: {"pbo": float | None, "n_candidates": int, "t_val": int,
        "status": str, ["reason": str]}}``. ``status`` is ``"ok"`` when PBO was computed;
        otherwise ``"skipped"`` with a ``reason`` (too few candidates, validation window
        shorter than ``n_blocks``, or no validation vectors). PBO is never fabricated for
        an arm that cannot support the CSCV partition — it is reported as skipped so a
        small/degenerate arm degrades gracefully rather than raising.
    """
    from src.inference.overfitting import pbo as _pbo

    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        matrix = build_perf_matrix(records, arm)
        t_val, n_cfg = (int(matrix.shape[0]), int(matrix.shape[1])) if matrix.size else (0, 0)
        entry: dict[str, Any] = {"pbo": None, "n_candidates": n_cfg, "t_val": t_val}

        if n_cfg < 2:
            entry.update(status="skipped", reason=f"need >= 2 candidates with validation vectors; got {n_cfg}")
            out[arm] = entry
            continue
        if t_val < n_blocks:
            entry.update(
                status="skipped",
                reason=f"validation window T={t_val} shorter than n_blocks={n_blocks}",
            )
            out[arm] = entry
            continue

        try:
            # FULL CSCV enumeration (deterministic): pass the exact split count so the PRIMARY
            # overfitting guard never falls back to a random ``max_combinations`` subsample (which
            # would make the headline PBO seed-dependent). For the frozen n_blocks=16 this is
            # C(16,8)=12,870 splits (milliseconds; ``rng`` is then unused). See overfitting.pbo.
            n_full = math.comb(n_blocks, n_blocks // 2)
            value = float(_pbo(matrix, n_blocks, rng=rng, max_combinations=n_full))
        except ValueError as exc:  # defensive: any residual CSCV precondition failure
            entry.update(status="skipped", reason=str(exc))
        else:
            entry.update(pbo=value, status="ok")
        out[arm] = entry

    return out


def _pbo_ranked_on_sharpe(
    perf_matrix: np.ndarray,
    n_blocks: int = 16,
    *,
    rng: np.random.Generator | None = None,
    max_combinations: int = 4000,
    periods_per_year: int = 252,
) -> float:
    """PBO/CSCV ranked on per-block ANNUALISED SHARPE (the DSR-proxy selection statistic; R36, M3).

    This is the byte-for-byte CSCV of :func:`src.inference.overfitting.pbo` with ONE change: the per-split
    IS / OOS performance of each candidate is its **annualised Sharpe over the concatenated IS (resp. OOS)
    rows**, NOT the mean return ``block.mean(axis=0)``. The frozen primary PBO ranks on the mean validation
    return; winner SELECTION used the validation **DSR** (``src.selection.fitness.held_out_fitness``), and
    with the frozen ``lambda_cvar = 0`` (R22) the DSR is a MONOTONE transform of the per-series Sharpe — so
    ranking on the per-block Sharpe is the rank-equivalent of "pick the highest-DSR candidate". This SECOND
    PBO therefore guards the rule the campaign actually USED (DEEP_STATS A3 point 2), ALONGSIDE the frozen
    mean-return PBO (which is unchanged; this is purely additive — ``overfitting.pbo`` is never modified).

    Returns the same statistic as ``overfitting.pbo``: the fraction of IS/OOS splits whose IS-best candidate
    lands STRICTLY below the OOS median (logit ``lambda < 0``). The argument validation, contiguous
    equal-block partition, full-enumeration-or-capped-sample of ``C(S, S/2)`` splits, average-rank tie
    handling, and strict ``lambda < 0`` count all match ``overfitting.pbo`` exactly.
    """
    import itertools as _it

    from src.inference.bootstrap import sharpe_ratio as _sr

    perf = np.asarray(perf_matrix, dtype=float)
    if perf.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T_obs, N_configs)")
    t_obs, n_cfg = perf.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 2")
    if t_obs < n_blocks:
        raise ValueError("need at least n_blocks rows")

    s = n_blocks
    block_size = t_obs // s
    blocks = [perf[k * block_size : (k + 1) * block_size, :] for k in range(s)]
    half = s // 2
    all_block_ids = range(s)
    n_total = math.comb(s, half)

    if n_total <= max_combinations:
        combos = list(_it.combinations(all_block_ids, half))
    else:
        if rng is None:
            # Seeded fallback (deep review 2026-07-26, loop 6): this was an UNSEEDED
            # ``default_rng()`` in a repo whose contract is byte-identical replay, and it disagreed
            # with ``src/inference/overfitting.py`` which uses ``default_rng(0)``. Dead on the
            # production path (the caller passes ``max_combinations=n_full``, taking the exhaustive
            # branch) but a latent determinism hole; matched to the reference implementation.
            rng = np.random.default_rng(0)
        seen: set[tuple[int, ...]] = set()
        block_arr = np.arange(s)
        while len(seen) < max_combinations:
            pick = tuple(sorted(rng.choice(block_arr, size=half, replace=False).tolist()))
            seen.add(pick)
        combos = list(seen)

    def _col_sharpes(data: np.ndarray) -> np.ndarray:
        # Per-CANDIDATE annualised Sharpe over the concatenated rows (the DSR-proxy ranking statistic).
        return np.asarray([_sr(data[:, j], periods_per_year) for j in range(data.shape[1])], dtype=float)

    logits_negative = 0
    n_splits = 0
    for is_ids in combos:
        is_set = set(is_ids)
        oos_ids = [k for k in all_block_ids if k not in is_set]
        is_data = np.concatenate([blocks[k] for k in is_ids], axis=0)
        oos_data = np.concatenate([blocks[k] for k in oos_ids], axis=0)

        is_perf = _col_sharpes(is_data)
        oos_perf = _col_sharpes(oos_data)
        best = int(np.argmax(is_perf))

        order = np.argsort(oos_perf, kind="mergesort")
        ranks = np.empty(n_cfg, dtype=float)
        ranks[order] = np.arange(1, n_cfg + 1)
        sorted_vals = oos_perf[order]
        i = 0
        while i < n_cfg:
            j = i
            while j + 1 < n_cfg and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg = (i + 1 + j + 1) / 2.0
                ranks[order[i : j + 1]] = avg
            i = j + 1

        omega = ranks[best] / (n_cfg + 1.0)
        lam = math.log(omega / (1.0 - omega))
        if lam < 0.0:
            logits_negative += 1
        n_splits += 1

    return float(logits_negative / n_splits)


def campaign_pbo_dsr(
    records: list[dict[str, Any]],
    *,
    n_blocks: int,
    arms: tuple[str, ...] | list[str] = ARMS,
    rng: np.random.Generator | None = None,
    periods_per_year: int = 252,
) -> dict[str, dict[str, Any]]:
    """SECOND per-arm PBO ranked on per-block annualised Sharpe (the DSR-proxy SELECTION rule; R36, M3).

    Mirrors :func:`campaign_pbo` exactly (same per-arm matrix, same skip rules, same full-enumeration of
    ``C(n_blocks, n_blocks/2)`` splits) but ranks IS/OOS on the per-block annualised SHARPE
    (:func:`_pbo_ranked_on_sharpe`) — the statistic winner SELECTION used (validation DSR, a monotone
    transform of Sharpe at ``lambda = 0``) — rather than the mean validation return the frozen PRIMARY PBO
    ranks on. This CLOSES the DEEP_STATS A3 "you didn't guard the rule you used" concern: if the two PBO
    columns AGREE (with ``lambda = 0`` they should agree closely), the mean-return proxy is empirically
    validated for the realised selection rule. The frozen primary guard (:func:`campaign_pbo` →
    ``src.inference.overfitting.pbo``) is UNCHANGED — this is additive and report-only.

    Returns the same per-arm dict shape as :func:`campaign_pbo`
    (``{arm: {"pbo", "n_candidates", "t_val", "status", ["reason"]}}``).
    """
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        matrix = build_perf_matrix(records, arm)
        t_val, n_cfg = (int(matrix.shape[0]), int(matrix.shape[1])) if matrix.size else (0, 0)
        entry: dict[str, Any] = {"pbo": None, "n_candidates": n_cfg, "t_val": t_val}
        if n_cfg < 2:
            entry.update(status="skipped", reason=f"need >= 2 candidates with validation vectors; got {n_cfg}")
            out[arm] = entry
            continue
        if t_val < n_blocks:
            entry.update(status="skipped", reason=f"validation window T={t_val} shorter than n_blocks={n_blocks}")
            out[arm] = entry
            continue
        try:
            n_full = math.comb(n_blocks, n_blocks // 2)
            value = _pbo_ranked_on_sharpe(
                matrix, n_blocks, rng=rng, max_combinations=n_full, periods_per_year=periods_per_year
            )
        except ValueError as exc:
            entry.update(status="skipped", reason=str(exc))
        else:
            entry.update(pbo=float(value), status="ok")
        out[arm] = entry
    return out


def winner_dsr(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] | list[str] = ARMS,
    periods_per_year: int = 252,
) -> dict[str, dict[str, Any]]:
    """Recompute each arm's HEADLINE winner Deflated Sharpe with the CANONICAL cross-trial var (Rank 16).

    Fixes the cross-trial-variance defect in the WIRED selection path: during search,
    ``src.selection.fitness.held_out_fitness`` calls ``deflated_sharpe_ratio`` with
    ``var_sr=None`` (the within-series SAMPLING-variance proxy), because the population
    variance over ALL an arm's candidates is not knowable inside the per-candidate loop. The
    canonical Bailey-Lopez de Prado DSR instead deflates by the cross-TRIAL Sharpe
    DISPERSION; on a heterogeneous candidate population the two diverge (the proxy mis-states
    the DSR). This is the clean analysis-time recompute: per arm it

      1. reconstructs the candidate population's per-period **validation** Sharpes from the
         recorded ``metrics['val_returns']`` (the SAME columns :func:`build_perf_matrix`
         stacks for PBO -- one Sharpe per candidate);
      2. forms the empirical cross-candidate dispersion ``var_sr = Var(sharpes, ddof=1)``;
      3. identifies the WINNER (the candidate selected during search -- max
         ``metrics['val_fitness']``, EXACTLY ``run_campaign.select_winner`` /
         ``analyze_results._winner``);
      4. recomputes the winner's DSR over its OWN validation returns deflated by that
         population ``var_sr``, alongside the proxy DSR (``var_sr=None``) for comparison.

    DSR is the SECONDARY overfitting diagnostic (PBO/CSCV is primary, ``var``-free); this
    corrects the secondary number that may be reported.

    Parameters
    ----------
    records : list of dict
        All per-candidate run records (``src.io.results.load_all``); split by ``arm``.
    arms : sequence of str, optional
        Arms to report (default: the NINE pre-registered arms — said "six" until 2026-07-27, #102).
        An arm with fewer than two
        candidates carrying a usable validation vector is reported as skipped (a single
        candidate has no cross-trial dispersion -- ``ddof=1`` variance is undefined).
    periods_per_year : int
        Forwarded to ``deflated_sharpe_ratio`` (annualization-invariant; kept for symmetry).

    Returns
    -------
    dict[str, dict]
        ``{arm: {"dsr_canonical": float | None, "dsr_proxy": float | None,
        "var_sr": float | None, "winner_sharpe": float | None, "n_candidates": int,
        "winner_id": str | None, "status": str, ["reason": str],
        ["winner_scan_note": str]}}``. ``dsr_canonical`` is the winner DSR deflated by the
        empirical cross-trial ``var_sr``; ``dsr_proxy`` is the legacy ``var_sr=None`` value the
        search path recorded. ``winner_scan_note`` appears only when the arm's TRUE max-fitness
        candidate carries no usable ``val_returns`` (the DSR winner is then a disclosed
        substitute). Ties in ``val_fitness`` break deterministically by (generation,
        candidate_id) — the ``src.inference.headroom`` convention. Never fabricated for an arm
        that cannot support the cross-trial dispersion -- reported as skipped.
    """
    from src.inference.bootstrap import sharpe_ratio
    from src.inference.deflated_sharpe import _sample_moments, deflated_sharpe_ratio

    def _fitness(r: dict[str, Any]) -> float:
        """The selection statistic as a float; missing/non-numeric -> -inf (never a comparison TypeError)."""
        try:
            return float(r.get("metrics", {}).get("val_fitness", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        # SEARCH candidates only (leg-disjoint loader fix): load_campaign_records now also returns the
        # frozen-winner TEST records + the frozen marker for this arm, which are NOT search candidates and
        # must not enter the dispersion or inflate the expected-max multiplicity (#32). _is_search_candidate
        # excludes them so n_trials = the true searched-candidate count, exactly as before the loader change.
        # SORTED by (generation, candidate_id) — the deterministic order src.inference.headroom._candidates
        # uses — so the max-fitness winner scan below breaks a val_fitness TIE by that key (max() keeps the
        # FIRST maximum), not by archive load order; the two winner scans must name the same candidate.
        arm_records = sorted(
            [r for r in records if r.get("arm") == arm and _is_search_candidate(r)],
            key=lambda r: (int(r.get("generation", 0) or 0), str(r.get("candidate_id", r.get("run_id", "")))),
        )
        cands = [(r, vec) for r in arm_records if (vec := _val_returns(r)) is not None]
        n_cfg = len(cands)
        entry: dict[str, Any] = {
            "dsr_canonical": None,
            "dsr_proxy": None,
            "var_sr": None,
            "winner_sharpe": None,
            "n_candidates": n_cfg,
            "winner_id": None,
        }
        if n_cfg < 2:
            entry.update(
                status="skipped",
                reason=f"need >= 2 candidates with validation vectors for a cross-trial "
                f"variance; got {n_cfg}",
            )
            out[arm] = entry
            continue

        # Per-candidate validation Sharpes -> empirical cross-trial dispersion (the canonical
        # DSR input). One Sharpe per candidate, exactly build_perf_matrix's columns.
        # PER-PERIOD (periods_per_year=1): deflated_sharpe_ratio compares the winner's PER-PERIOD
        # Sharpe (_sample_moments) against sr_star = sqrt(var_sr)*term, so var_sr MUST be the
        # variance of PER-PERIOD Sharpes. Annualizing here (the sharpe_ratio default 252) inflates
        # var_sr ~252x -> sr_star ~15.87x -> the canonical DSR collapses spuriously to ~0
        # (final-acceptance-audit P1, 2026-06-19).
        # ddof CONSISTENCY (audit): build each candidate's per-period Sharpe with the SAME _sample_moments
        # the DSR uses for the WINNER Sharpe (sample std ddof=1) — NOT bootstrap.sharpe_ratio, whose
        # population std (ddof=0) put var_sr's dispersion and the winner Sharpe it is compared against on
        # mismatched conventions and contaminated the canonical-vs-proxy gap with a sqrt(T/(T-1)) artefact.
        sharpes = np.asarray([_sample_moments(vec)[0] for _, vec in cands], dtype=float)
        var_sr = float(np.var(sharpes, ddof=1))

        # WINNER = the candidate selected during search (max validation fitness), matching
        # run_campaign.select_winner / analyze_results._winner. NaN-safe key: a NaN val_fitness
        # compares False both ways, so a NaN FIRST element could poison python's max() (it would
        # never be displaced) — map non-finite fitness to -inf so it can never win the scan.
        winner_rec, winner_vec = max(
            cands, key=lambda rv: (f if np.isfinite(f := _fitness(rv[0])) else float("-inf"))
        )
        # DISCLOSE a vectorless true winner: the scan above ranks only candidates CARRYING val_returns
        # (the DSR needs the vector). If the arm's TRUE max-val_fitness candidate archived no usable
        # vector, the reported DSR belongs to a SUBSTITUTE winner — flag it rather than silently re-rank.
        finite_fit = [r for r in arm_records if np.isfinite(_fitness(r))]
        true_best = max(finite_fit, key=_fitness) if finite_fit else None
        winner_scan_note = None
        if true_best is not None and true_best is not winner_rec and _val_returns(true_best) is None:
            winner_scan_note = (
                "true max-val_fitness candidate "
                f"{true_best.get('candidate_id', true_best.get('run_id', '?'))!s} carries no usable "
                "val_returns; winner DSR is reported for the best VECTOR-CARRYING candidate instead"
            )
        # Expected-max trial count = the FULL per-arm candidate count the search actually faced, NOT
        # just the candidates that happen to carry a usable validation vector (final-audit #32: using
        # n_cfg deflated by a SMALLER N than selection used, biasing the canonical DSR UP). var_sr
        # above is still the dispersion of the AVAILABLE Sharpes (best estimate); only the multiplicity
        # uses the full count, matching the selection-time DSR.
        n_trials = len(arm_records)

        out[arm] = {
            "dsr_canonical": float(
                deflated_sharpe_ratio(
                    winner_vec, n_trials, var_sr=var_sr, periods_per_year=periods_per_year
                )
            ),
            "dsr_proxy": float(
                deflated_sharpe_ratio(
                    winner_vec, n_trials, var_sr=None, periods_per_year=periods_per_year
                )
            ),
            "var_sr": var_sr,
            "winner_sharpe": float(sharpe_ratio(winner_vec)),  # annualized, for display; dsr_* above use the per-period convention
            "n_candidates": n_cfg,         # candidates carrying a validation vector (var_sr's N)
            "n_trials": n_trials,          # full per-arm candidate count = the expected-max multiplicity (#32)
            "winner_id": str(winner_rec.get("candidate_id", winner_rec.get("run_id", "?"))),
            "status": "ok",
        }
        if winner_scan_note:
            out[arm]["winner_scan_note"] = winner_scan_note
    return out


def _cluster_anomalies_into_runs(anomalies: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a flat ``anomalies.jsonl`` event stream into per-RUN clusters by STEP-RESETS (R34).

    Monitoring (``src.utils.monitoring.RunMonitor.anomaly``) appends every ``critic_explosion`` (and other)
    event to ONE append-only ``anomalies.jsonl`` for the whole run, so a single training that diverges at
    steps 1000, 2000, ... contributes MANY lines — counting lines over-states how many distinct RUNS
    diverged. Each training re-starts its step counter at ~0, so the boundary between two runs is a
    DECREASE in ``step`` (a reset) relative to the previous event. We split the stream there: a new cluster
    begins whenever the current event's step is strictly LESS than the previous event's step. Events with no
    integer ``step`` (e.g. resource-pressure anomalies) are kept inside the current cluster without forcing a
    boundary (they carry no step to reset). Order is preserved (the file is already chronological).

    This is the deliberately SIMPLE, robust clustering the prototype fact was verified against (64
    ``critic_explosion`` lines → 6 diverged RUNS); it makes no claim beyond "a step that goes backwards is a
    new training" and is report-only (it never re-reads or alters the trainer).
    """
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_step: int | None = None
    for ev in anomalies:
        raw = ev.get("step")
        step = int(raw) if isinstance(raw, (int, float)) else None
        # A strict step DECREASE (a counter reset) starts a new run; the first event opens the first run.
        if step is not None and prev_step is not None and step < prev_step and current:
            clusters.append(current)
            current = []
        current.append(ev)
        if step is not None:
            prev_step = step
    if current:
        clusters.append(current)
    return clusters


def _load_anomalies_jsonl(root: str | Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Read every ``anomalies.jsonl`` under ``root`` (any depth) into one ordered event list (R34).

    The monitor writes ``anomalies.jsonl`` at the RUN directory root; a campaign output_dir may carry one
    (the whole-campaign monitor) and/or per-leg ones. We collect them ALL (sorted by path for determinism),
    parsing one JSON object per non-blank line and skipping malformed lines (best-effort — a report-only
    diagnostic must never raise on a partial file). Returns ``(events, files_read)``.
    """
    root = Path(root)
    files: list[Path] = []
    if root.is_dir():
        files = sorted(root.rglob("anomalies.jsonl"))
    elif root.is_file() and root.name == "anomalies.jsonl":
        files = [root]
    events: list[dict[str, Any]] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events, files


def divergence_report(
    root: str | Path,
    *,
    winner_ids: tuple[str, ...] | list[str] = (),
    kinds: tuple[str, ...] = ("critic_explosion",),
) -> dict[str, Any]:
    """Cluster training-divergence anomalies into the TRUE diverged-RUN count + rate (R34; report-only).

    Reads the existing append-only ``anomalies.jsonl`` (it does NOT modify the trainer) and answers the
    confound an examiner raises about the 64 ``critic_explosion`` LINES the prototype logged: those are NOT
    64 diverged runs. Each training re-starts its step counter at ~0, so the lines cluster into RUNS by
    step-reset (:func:`_cluster_anomalies_into_runs`); the prototype's 64 lines collapse to **6 diverged
    RUNS** (≈2.5% of the candidate budget), mostly TRANSIENT single-step spikes that recover.

    Why this is not an H2 confound (the disclosure this encodes). The reward is UNBOUNDED on purpose
    (``norm_reward=False`` is DELIBERATE — the reward *is* the object of study, so its scale is left as the
    LLM/search wrote it), so a poorly-scaled candidate can transiently blow the critic loss up. But such a
    candidate scores POORLY on the held-out validation fitness and is DROPPED by selection, so divergence
    biases toward NOISE in the dropped tail, NOT toward the H2 headline (a diverged candidate cannot become
    a winner unless it ALSO posted a strong sealed validation Sharpe — which the ``winner_diverged`` flag
    checks explicitly).

    Parameters
    ----------
    root : str or Path
        Campaign/prototype archive root (or a direct ``anomalies.jsonl`` path). Every ``anomalies.jsonl``
        beneath it is read.
    winner_ids : sequence of str, optional
        The frozen winners' ``candidate_id``s (e.g. ``"<arm>-g6-c3"``). When an anomaly event carries a
        ``candidate_id``/``cand`` field, a run whose events name a winner id is flagged in
        ``winner_diverged``. (The prototype anomaly schema has no candidate id, so this is best-effort and
        reports ``winner_diverged=[]`` / ``winner_attribution="unavailable"`` there — never a false claim.)
    kinds : tuple of str, optional
        The anomaly ``kind`` values that count as a training DIVERGENCE (default ``("critic_explosion",)``;
        the NaN/inf ``nonfinite_metric`` kind can be added by the caller). Non-divergence kinds (resource
        pressure, FPS) are excluded from the run/line tallies but still parsed for clustering context.

    Returns
    -------
    dict
        ``{"status", "n_anomaly_lines", "n_diverged_runs", "divergence_rate", "n_candidates_budget",
        "winner_diverged", "winner_attribution", "transient_runs", "files_read", "note"}`` or
        ``{"status": "skipped", "reason": ...}`` when no ``anomalies.jsonl`` is present (a clean run that
        never logged an anomaly — the common, healthy case).
    """
    kind_set = {str(k) for k in kinds}
    events, files = _load_anomalies_jsonl(root)
    if not files:
        return {
            "status": "skipped",
            "reason": "no anomalies.jsonl under root (no anomaly was logged — a clean run)",
        }

    # Divergence events only (keep the chronological order the file already has).
    div_events = [e for e in events if str(e.get("kind")) in kind_set]
    n_lines = len(div_events)
    if not div_events:
        return {
            "status": "ok",
            "n_anomaly_lines": 0,
            "n_diverged_runs": 0,
            "divergence_rate": None,
            "n_candidates_budget": None,
            "winner_diverged": [],
            "winner_attribution": "n/a (no divergence events)",
            "transient_runs": 0,
            "files_read": [str(f) for f in files],
            "note": "no divergence-kind anomalies in the log.",
        }

    clusters = _cluster_anomalies_into_runs(div_events)
    n_runs = len(clusters)

    # Matched-budget denominator for the rate: the per-arm candidate count × the number of arms that could
    # diverge (LLM/search arms train a model per candidate). Read from config (single source of truth); fall
    # back to None (rate omitted) when config is unreadable. The candidate budget is candidates_per_arm × the
    # number of TRAINED arms (every arm trains a model per candidate), so this is the full training count.
    n_budget: int | None = None
    try:
        from src.utils.config import load_config

        campaign = load_config("campaign")
        cpa = int(campaign.get("candidates_per_arm", 0) or 0)
        n_arms = len(list(campaign.get("arms", []) or []))
        if cpa and n_arms:
            n_budget = cpa * n_arms
    except Exception:  # noqa: BLE001 - config unreadable -> rate omitted, never raised
        n_budget = None

    rate = (n_runs / float(n_budget)) if n_budget else None

    # A run is TRANSIENT when its diverged steps are sparse relative to a sustained blow-up; we report the
    # count of runs whose divergence is a SINGLE logged step (the dominant prototype pattern — a spike that
    # recovers). Sustained runs (many consecutive logged steps) are the complement.
    transient = sum(1 for c in clusters if len({e.get("step") for e in c if e.get("step") is not None}) <= 1)

    # Winner attribution (best-effort): does any diverged RUN name a frozen-winner candidate id? The
    # prototype anomaly schema (detail/kind/step/ts) carries NO candidate id, so this is reported as
    # "unavailable" there rather than fabricated; the campaign monitor may add cand/candidate_id.
    win_set = {str(w) for w in winner_ids}
    have_ids = any(("candidate_id" in e or "cand" in e) for e in div_events)
    winner_diverged: list[str] = []
    if win_set and have_ids:
        for c in clusters:
            named = {str(e.get("candidate_id", e.get("cand", ""))) for e in c}
            hit = named & win_set
            winner_diverged.extend(sorted(h for h in hit if h))
        attribution = "ok"
    elif not win_set:
        attribution = "no winner_ids supplied"
    else:  # winner ids known but the anomaly schema has no candidate id (prototype)
        attribution = "unavailable (anomaly schema carries no candidate_id)"

    return {
        "status": "ok",
        "n_anomaly_lines": int(n_lines),
        "n_diverged_runs": int(n_runs),
        "divergence_rate": rate,
        "n_candidates_budget": n_budget,
        "winner_diverged": sorted(set(winner_diverged)),
        "winner_attribution": attribution,
        "transient_runs": int(transient),
        "files_read": [str(f) for f in files],
        "note": (
            f"{n_lines} anomaly LINES cluster (by step-reset) into {n_runs} diverged RUN(s); "
            "unbounded reward (norm_reward=False, DELIBERATE — the reward is the object of study) lets a "
            "mis-scaled candidate transiently blow the critic up, but a diverged candidate scores poorly on "
            "held-out validation and LOSES selection, so divergence biases toward NOISE (the dropped tail), "
            "not toward the H2 headline."
        ),
    }


def _count_jsonl_lines(path: Path) -> int:
    """Number of non-blank lines in a JSONL file (0 if absent/unreadable). Report-only, never raises."""
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _sum_prompt_tokens(path: Path) -> tuple[int, int]:
    """Sum ``usage.input_tokens`` (prompt) + ``usage.output_tokens`` over an ``llm_calls.jsonl`` (R35).

    Returns ``(prompt_tokens, completion_tokens)``; a record with no usage / non-int field contributes 0.
    Best-effort: a malformed line is skipped (a report-only token tally must never raise on a partial file).
    """
    prompt = 0
    completion = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        usage = rec.get("usage") if isinstance(rec, dict) else None
        if isinstance(usage, dict):
            it = usage.get("input_tokens")
            ot = usage.get("output_tokens")
            if isinstance(it, (int, float)):
                prompt += int(it)
            if isinstance(ot, (int, float)):
                completion += int(ot)
    return prompt, completion


def _find_arm_provenance_dir(root: Path, arm: str) -> Path | None:
    """Locate the directory that holds ``arm``'s ``llm_calls.jsonl`` / ``failures.jsonl`` (R35).

    The prototype writes them at ``<root>/<arm>/`` (``scripts/run_prototype.py`` / ``parallel.py``); the
    campaign writes the search leg at ``<root>/search/<arm>/``. Probe the known locations and return the
    first that exists (carrying either provenance file), else ``None``.
    """
    candidates = [root / arm, root / "search" / arm]
    for d in candidates:
        if (d / "llm_calls.jsonl").is_file() or (d / "failures.jsonl").is_file():
            return d
    return None


#: The LLM-authored arms (open-ended code from the designer): each candidate is ONE LLM call, and a
#: gate-failing candidate BURNS its budget slot (``src/llm/loop.py`` ~338,380 archives the failure + moves
#: on). The search arms (``random_search``/``bayes_opt``) instead RESAMPLE on a gate failure to a full slate
#: (``src/search/random_search.py`` ~259: a ``SandboxError`` ``continue``s WITHOUT consuming a budget unit),
#: so they obtain strictly MORE valid candidates per matched budget — the compute-accounting asymmetry (i).
_LLM_AUTHORED_ARMS: tuple[str, ...] = ("distributional", "scalar", "placebo", "scalar_cvar5", "placebo_shuffled")
_SEARCH_ARMS: tuple[str, ...] = ("random_search", "bayes_opt", "cma_es", "tpe")
#: The arms that feed the LLM EXTRA distributional feedback lines (the token-count asymmetry (ii)): the
#: tail-aware blocks carry ~8 feedback lines vs the scalar arm's 1. This token gap is CONTROLLED for the
#: information-vs-token-count contrast by the inert ``placebo`` leg (matched block length, zero content).
_TAIL_FED_ARMS: tuple[str, ...] = ("distributional", "scalar_cvar5", "placebo_shuffled")


def compute_accounting(
    records: list[dict[str, Any]],
    root: str | Path,
    *,
    arms: tuple[str, ...] | list[str] = ARMS,
) -> dict[str, Any]:
    """Per-arm compute-accounting: candidates attempted/accepted/failed + total prompt-tokens (R35).

    Tabulates, from the archived ``failures.jsonl`` + ``llm_calls.jsonl`` (token usage) and the loaded
    search-candidate records, how many candidates each arm ATTEMPTED, how many were ACCEPTED (passed the
    gate + were evaluated), how many FAILED the gate, and the arm's total prompt (input) tokens. REPORT-ONLY
    and DISJOINT (``out["compute_accounting"]``; no family-tuple keys).

    Two asymmetries it discloses (both CONSERVATIVE for the LLM headline / controlled):

    (i) **Failure asymmetry → favours search (against the LLM headline).** An LLM arm BURNS a budget slot on
        a gate failure (``src/llm/loop.py`` ~338,380: the failure is archived and the loop moves on), so it
        evaluates ``budget − n_failed`` valid candidates. A search arm RESAMPLES on a gate failure
        (``src/search/random_search.py`` ~259: ``SandboxError`` ``continue``s WITHOUT spending a budget
        unit), so it always reaches a FULL slate of valid candidates. Search therefore gets strictly MORE
        valid candidates per matched budget — a HANDICAP on the LLM arms, i.e. conservative for H2.

    (ii) **Token asymmetry → controlled by the placebo leg.** The tail-aware feedback block carries ~8
        feedback lines (the six tail stats + headers) vs the scalar arm's 1, so the distributional arm sends
        more prompt tokens. That token-count difference is the EXACT thing the inert ``placebo`` arm (matched
        block length, zero information) controls for in the H2 placebo leg — so the headline already nets the
        token count out of the distributional-vs-scalar contrast.

    Parameters
    ----------
    records : list of dict
        All loaded run records; the per-arm ACCEPTED count is the number of SEARCH candidates
        (:func:`_is_search_candidate`) for that arm.
    root : str or Path
        Archive root holding ``<arm>/`` (prototype) or ``search/<arm>/`` (campaign) provenance files.
    arms : sequence of str, optional
        Arms to tabulate (default the six pre-registered arms).

    Returns
    -------
    dict
        ``{"status": "ok", "rows": [ {arm, kind, n_llm_calls, n_accepted, n_failed, n_attempted,
        prompt_tokens, completion_tokens, resamples_to_full_slate}, ... ], "totals": {...},
        "asymmetry_note": str}``. ``n_attempted`` is ``n_accepted + n_failed`` (the slots the arm consumed);
        for an LLM arm this equals ``n_llm_calls`` when every call is accounted. Arms with no provenance
        files report zeros (records-only / search arms have no ``llm_calls.jsonl``).
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    tot_prompt = 0
    tot_completion = 0
    tot_accepted = 0
    tot_failed = 0
    for arm in arms:
        n_accepted = sum(1 for r in records if r.get("arm") == arm and _is_search_candidate(r))
        prov = _find_arm_provenance_dir(root, arm)
        n_calls = 0
        n_failed = 0
        prompt_tok = 0
        completion_tok = 0
        if prov is not None:
            n_calls = _count_jsonl_lines(prov / "llm_calls.jsonl")
            n_failed = _count_jsonl_lines(prov / "failures.jsonl")
            prompt_tok, completion_tok = _sum_prompt_tokens(prov / "llm_calls.jsonl")
        is_search = arm in _SEARCH_ARMS
        rows.append({
            "arm": arm,
            "kind": "search" if is_search else ("llm" if arm in _LLM_AUTHORED_ARMS else "other"),
            "n_llm_calls": int(n_calls),
            "n_accepted": int(n_accepted),
            "n_failed": int(n_failed),
            "n_attempted": int(n_accepted + n_failed),
            "prompt_tokens": int(prompt_tok),
            "completion_tokens": int(completion_tok),
            # Search arms resample on a gate failure to a FULL valid slate; LLM arms burn the slot.
            "resamples_to_full_slate": bool(is_search),
            "tail_fed": bool(arm in _TAIL_FED_ARMS),
        })
        tot_prompt += prompt_tok
        tot_completion += completion_tok
        tot_accepted += n_accepted
        tot_failed += n_failed

    return {
        "status": "ok",
        "rows": rows,
        "totals": {
            "n_accepted": int(tot_accepted),
            "n_failed": int(tot_failed),
            "prompt_tokens": int(tot_prompt),
            "completion_tokens": int(tot_completion),
        },
        "asymmetry_note": (
            "(i) LLM arms BURN a budget slot on a gate failure (src/llm/loop.py ~338,380) while search arms "
            "RESAMPLE to a full valid slate (src/search/random_search.py ~259), so search gets strictly MORE "
            "valid candidates per matched budget — a handicap on the LLM arms, conservative for the H2 "
            "headline. (ii) Tail-aware blocks send ~8 feedback lines vs scalar's 1; that token-count "
            "difference is controlled by the inert placebo leg in the H2 placebo contrast."
        ),
    }


#: Bounded depth for the campaign archive walk: the deepest real layout is the campaign's
#: ``output_dir/<leg>/<arm>/<candidate>/record.json`` (leg in {search, test, frozen}), i.e. the
#: ``record.json``-bearing leaf sits at most 3 directory levels under ``output_dir``. We walk to
#: that depth so a SINGLE ``--root`` (the campaign output_dir) reaches BOTH legs at once.
_MAX_ARCHIVE_DEPTH = 3


def load_campaign_records(root: str | Path) -> list[dict[str, Any]]:
    """Load all per-record archives under a campaign ``root``, to ANY layout depth (audit C-1).

    Auto-detects the three layouts the project writes and reads them ALL from one ``root``,
    de-duplicating by ``run_id`` (reads ONLY via ``src.io.results.load_all``):

    - flat ``<root>/<run>/record.json`` (e.g. the unit-test / single-leg archives);
    - prototype ``<root>/<arm>/<candidate>/record.json``
      (``scripts/run_prototype.py`` / ``src/orchestration/parallel.py``);
    - **campaign** ``<root>/<leg>/<arm>/<candidate>/record.json`` with ``leg`` in
      ``{search, test, frozen}`` (``scripts/run_campaign.py`` L526-528).

    Fix (leg-disjoint defect, audit): the PBO/DSR search candidates (carry
    ``metrics['val_returns']``) live under ``search/<arm>/<cand>`` while the H2/floor frozen-winner
    TEST records (carry ``metrics['test_returns']``) live under ``test/<arm>/<arm>-s{seed}`` — two
    levels deeper than the old loader's single ``load_all(<root>/<sub>)`` descent could reach, so
    pointing this at the campaign ``output_dir`` previously returned ONLY the frozen winners (one
    level under ``frozen/``) and BOTH the PBO/DSR table and the H2 family came back empty. Walking
    to ``_MAX_ARCHIVE_DEPTH`` makes ``analyze()``/``main()`` produce the full headline report
    (PBO + DSR + H2 + floor) from the campaign output_dir in a single call — matching the contract
    ``scripts/cost_sweep.py`` already relies on when it points the loader at the ``test`` leg.

    A directory is loaded via ``load_all`` (which reads the immediate ``record.json``-bearing
    children) iff it HAS such children; intermediate directories (``output_dir``, the ``<leg>`` and
    ``<arm>`` dirs) carry no ``record.json`` of their own and are merely traversed.
    """
    from src.io.results import _RECORD_NAME, load_all

    root = Path(root)
    # NOTE: KEYED BY (DIRECTORY, run_id), NOT run_id ALONE (A79, 2026-08-01, found during the analysis).
    # `run_id` is unique WITHIN a line and COLLIDES ACROSS them: every line writes
    # `distributional-s0`, `placebo-s0`, ... because the id encodes arm+seed and carries NO line.
    # A global run_id key therefore de-duplicated ACROSS lines and silently dropped every line but
    # the first one walked. MEASURED on the live RUN 4 archive before this fix:
    #   load_campaign_records -> 732 records, against 2,260 record.json on disk (ex-h3)
    #   => 1,528 records, 68% of the archive, discarded SILENTLY.
    # Every statistic downstream of this loader was therefore computed on a third of the data with
    # no error, no warning and no count mismatch to notice. The directory is the only line-bearing
    # discriminator available (the record itself has none), and `_walk` visits each directory exactly
    # once, so (directory, run_id) preserves the ORIGINAL intra-directory de-dup intent while making
    # a cross-line merge structurally impossible.
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    def _walk(directory: Path, depth: int) -> None:
        if not directory.is_dir():
            return
        children = sorted(p for p in directory.iterdir() if p.is_dir())
        # Load this directory's own run subdirs (those with a record.json) once, in run_id order.
        if any((c / _RECORD_NAME).is_file() for c in children):
            for rec in load_all(directory):
                seen.setdefault((str(directory), str(rec.get("run_id"))), rec)
        # Recurse into the remaining (intermediate) subdirs up to the bounded archive depth, so the
        # campaign's <leg>/<arm>/<cand> leaves are reached without an unbounded filesystem walk.
        # M15 (2026-07-05): the H3 single-shot control writes its own SEARCH/FROZEN/TEST subtrees
        # (``*_h3_singleshot/``) under the SAME campaign output_dir, with arm='distributional' and a
        # run_id pattern that COLLIDES with the headline serial search ids — walking into them would
        # silently pool (or, via the run_id de-dup, silently drop) single-shot candidates into the
        # HEADLINE distributional arm's records. They are a DISJOINT condition by design (DEEP_H3 §1),
        # loaded explicitly by the H3 analysis from its own roots — never by the default walk.
        if depth < _MAX_ARCHIVE_DEPTH:
            for child in children:
                # Skip DOT-PREFIXED dirs (`.pull_tmp.<pid>`, staging). Raised by coord as M267 and
                # made URGENT by the A79 fix, which INVERTED this defect's failure mode: under the
                # old global run_id key a temp copy sorted FIRST ('.' is 0x2E) and DISPLACED the real
                # record; under (directory, run_id) both keys are distinct so BOTH load and the record
                # is DUPLICATED. Duplication is the harder one to notice because the totals look
                # BETTER, not worse (measured: `random_search-c11` returned twice, byte-identical,
                # 1 spurious duplicate in 2,290 -- and random_search is an H4 comparator, so it would
                # enter that arm's PBO population and DSR multiplicity pool twice).
                # No legitimate archive subtree is dot-prefixed, so this removes the class permanently.
                #
                # Skip `_leg_` dirs (D49/D50):
                # the ten replication lines write `search_leg_<model>/` and `test_leg_<model>/`
                # under the SAME campaign root, and every line reuses the core run_id vocabulary
                # (`distributional-s0`, ...), so walking them pools ten lines into the core arms and
                # `_seed_scores` aborts on the cross-line seed conflict before write_report() runs.
                # They are per-line subtrees by design, loaded from their own roots exactly as
                # `*_h3_singleshot` is -- never by the default walk. The core stage dirs
                # (`search/`, `test/`, `frozen/`) do not match the pattern.
                if (
                    child.name.startswith(".")
                    or "_leg_" in child.name
                    or child.name.endswith("_h3_singleshot")
                ):
                    continue
                # A record-bearing dir is a LEAF by this loader's own layout contract (docstring
                # above: intermediate dirs carry no record.json of their own). It was already
                # loaded via load_all(directory), so descending into it can only re-load stray
                # nested copies. Run 4 holds exactly two (`<cand>/<cand>/record.json`,
                # byte-identical to their outer copy, verified 2026-08-18): under the
                # (directory, run_id) key both copies enter `seen` and one candidate counts twice
                # in the PBO population and the DSR multiplicity denominator.
                # Per D51 the stray files stay on disk and the skip is LOUD, never silent.
                if (child / _RECORD_NAME).is_file():
                    nested = [
                        p for p in child.iterdir()
                        if p.is_dir() and (p / _RECORD_NAME).is_file()
                    ]
                    if nested:
                        print(
                            f"[load_campaign_records] WARNING: {len(nested)} nested "
                            f"record-bearing dir(s) under leaf {child} ignored "
                            f"(leaves are terminal by the archive layout contract)",
                            file=sys.stderr,
                        )
                    continue
                _walk(child, depth + 1)

    _walk(root, 0)
    return list(seen.values())


# =========================================================================== #
# Rank 8 — the multiple-testing family + the H2 conjunction (TEST leg)         #
# =========================================================================== #
#: The pre-registered HEADLINE contrasts (PREREGISTRATION §1 H2; FINAL_PLAN B.6
#: line 83): the distributional arm must beat the scalar arm AND survive beyond the
#: placebo (information != token-count) AND beyond scalar+CVaR-5% (tail-shape !=
#: any-downside-number). Each entry is ``(arm_a, arm_b)`` read "a is predicted BETTER
#: than b" (one-sided direction: higher Sharpe / higher — less negative — CVaR).
H2_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("distributional", "scalar"),
    ("distributional", "placebo"),
    ("distributional", "scalar_cvar5"),
)


def headline_cvar_level(cvar_levels):
    """The FROZEN headline tail level (PREREGISTRATION S10: CVaR-5%), resolved consistently.

    AUDIT 2026-07-22 (row 30a): the gate previously took ``max(cvar_levels)`` while every report
    site took ``cvar_levels[0]`` -- two conventions that agree only under the current config
    ordering. This helper is the single resolution rule: the headline is the MAXIMUM of the
    FROZEN family levels (sanctioned expansions only add DEEPER/smaller alphas, so the frozen
    headline 0.05 is invariant), asserted present in the realized levels when any are given.
    """
    from src.utils.config import load_config

    fam = (load_config("preregistration").get("inference", {}) or {}).get("testing_family") or {}
    frozen = [float(x) for x in (fam.get("cvar_levels") or [0.05])]
    headline = max(frozen)
    realized = [float(x) for x in (cvar_levels or [])]
    if realized:
        assert headline in realized, (
            f"frozen headline CVaR level {headline} missing from realized levels {realized} "
            "(PREREGISTRATION S10 -- family drift)")
    return float(headline)


def assert_realized_family_matches_frozen(
    tests: list[dict[str, Any]],
    *,
    cvar_levels: tuple[float, ...],
) -> None:
    """# fail-loud: the REALIZED testing family must equal the FROZEN pre-registration family (R13).

    The frozen multiple-testing family is enumerated in ``config/preregistration.yaml``
    (``inference.testing_family``: ``m`` + the ``members`` list) and mirrored in PREREGISTRATION §10
    (Amendment 2026-06-19, R13). This guard re-derives the realized family from the campaign's own
    ``collect_family_pvalues`` output and asserts it is byte-for-byte the frozen one — so the campaign
    can NEVER silently test a family that drifts from the pre-registered, hashed design. It is a no-op
    ONLY when the realized ``cvar_levels`` are a STRICT SUPERSET of the frozen level set — the one
    sanctioned, prose-flagged expansion (the opt-in high-variance ``cvar_01`` leg, which grows ``m`` to
    9). A realized level set MISSING any frozen level is drift, never an extension, and falls through
    to the assert (fail-loud).

    Raises ``AssertionError`` (fail-loud) on any drift in the contrast set, the metric set, the CVaR
    levels, or the integer ``m``; raises nothing and asserts nothing if the frozen YAML lacks a
    ``testing_family`` block (older config) — the campaign still runs, but the freeze check
    (``scripts/freeze.py``) will catch a missing mirror.
    """
    from src.utils.config import load_config

    prereg = load_config("preregistration")
    fam = prereg.get("inference", {}).get("testing_family") if isinstance(prereg, dict) else None
    if not fam:
        return  # no frozen mirror to check against (pre-amendment config) — freeze.py is the backstop

    # A STRICT SUPERSET of the frozen levels is the one sanctioned, prose-flagged expansion (the opt-in
    # cvar_01 leg grows m to 9) -> no-op. ANYTHING else — in particular a realized set MISSING a frozen
    # level — must fall through to the assert below and fail LOUD: dropping a frozen level is family
    # drift, and the old any-difference early-return silently waved it past the guard.
    frozen_levels = {float(x) for x in fam.get("cvar_levels", [0.05])}
    realized_levels = {float(x) for x in cvar_levels}
    if realized_levels > frozen_levels:
        # AUDIT 2026-07-22 (row 30a): the old early-return skipped ALL checks (membership, m, the
        # co-primary partition), not just the level comparison. The sanctioned expansion only ADDS
        # levels, so restrict the realized tests to the FROZEN levels and verify the frozen core
        # exactly — the guard survives the expansion instead of being waved off by it.
        tests = [t for t in tests
                 if t["level"] is None or float(t["level"]) in frozen_levels]

    # Realized family, as the campaign actually built it.
    realized = {
        (
            str(t["arm_a"]),
            str(t["arm_b"]),
            str(t["metric"]),
            (None if t["level"] is None else float(t["level"])),
        )
        for t in tests
    }
    # Frozen family, from the hashed pre-registration mirror.
    frozen = {
        (
            str(mem["arm_a"]),
            str(mem["arm_b"]),
            str(mem["metric"]),
            (None if mem.get("level") is None else float(mem["level"])),
        )
        for mem in fam.get("members", [])
    }
    m_frozen = int(fam.get("m", len(frozen)))
    assert realized == frozen, (
        "REALIZED testing family != FROZEN pre-registration family (PREREGISTRATION §10 / "
        "config/preregistration.yaml: inference.testing_family, Amendment R13).\n"
        f"  realized only: {sorted(map(str, realized - frozen))}\n"
        f"  frozen only:   {sorted(map(str, frozen - realized))}"
    )
    assert len(frozen) == m_frozen, (
        f"FROZEN family size mismatch: members list has {len(frozen)} entries but "
        f"inference.testing_family.m = {m_frozen} (config/preregistration.yaml)."
    )
    assert len(realized) == m_frozen, (
        f"REALIZED family size {len(realized)} != frozen m = {m_frozen} "
        "(PREREGISTRATION §10, Amendment R13)."
    )

    # # fail-loud (R25): the two CO-PRIMARY IUT sub-families must PARTITION the m=6 union exactly — H2-RA
    # (3 Sharpe legs) ∪ H2-Tail (3 CVaR-5% legs) == the frozen union, disjoint, each of size its own `m`.
    # This guards the headline two-tier decision against family drift just as the union check guards the
    # enumerated family (DEEP_H2 §3.3 option A / §7.1).
    families = fam.get("families") if isinstance(fam.get("families"), dict) else None
    if families:
        union_of_subs: set[tuple[str, str, str, float | None]] = set()
        for sub_name, sub in families.items():
            sub_members = {
                (
                    str(mem["arm_a"]),
                    str(mem["arm_b"]),
                    str(mem["metric"]),
                    (None if mem.get("level") is None else float(mem["level"])),
                )
                for mem in (sub.get("members") or [])
            }
            sub_m = int(sub.get("m", len(sub_members)))
            assert len(sub_members) == sub_m, (
                f"FROZEN sub-family '{sub_name}' size mismatch: {len(sub_members)} members but "
                f"m = {sub_m} (config/preregistration.yaml: inference.testing_family.families.{sub_name})."
            )
            assert union_of_subs.isdisjoint(sub_members), (
                f"FROZEN sub-families overlap at {sorted(map(str, union_of_subs & sub_members))} "
                "(the two co-primary IUTs must be DISJOINT; R25)."
            )
            union_of_subs |= sub_members
        assert union_of_subs == frozen, (
            "FROZEN co-primary IUT sub-families do NOT partition the m=6 union "
            "(config/preregistration.yaml: inference.testing_family.families, Amendment R25).\n"
            f"  sub-families only: {sorted(map(str, union_of_subs - frozen))}\n"
            f"  union only:        {sorted(map(str, frozen - union_of_subs))}"
        )


def _test_returns(record: dict[str, Any]) -> np.ndarray | None:
    """Extract a frozen-winner record's realized per-step TEST return vector, or ``None``.

    The vector rides inside ``metrics['test_returns']`` — the per-(arm, seed) test-leg
    records ``scripts/run_campaign.py`` writes (it also mirrors it as the optional
    top-level ``test_returns`` field; we read ``metrics`` first, then fall back). Returns
    ``None`` when absent / non-finite / empty so a missing arm degrades gracefully.
    """
    metrics = record.get("metrics") or {}
    tr = metrics.get("test_returns")
    if tr is None:
        tr = record.get("test_returns")
    if tr is None:
        return None
    arr = np.asarray(tr, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _arm_test_returns(records: list[dict[str, Any]], arm: str) -> np.ndarray | None:
    """Pool one arm's per-seed TEST return vectors into a single seed-AVERAGED series.

    NB (#9/#14, 2026-06-20): this seed-average is **no longer the headline inference path** — feeding
    a per-period mean-over-seeds to a single-strategy bootstrap collapses the across-seed variance
    ~N× and is anti-conservative (the over-rejection scales WITH the across-seed variance — measured
    ~21% true-null rejection at the 5% level in a representative 30-seed calibration with
    training-RNG-scale seed variance; ``tests/test_audit_regressions.py``). The difference tests now
    consume PER-SEED SCORES via :func:`_seed_scores` +
    :func:`src.inference.bootstrap.paired_seed_difference_test` (rliable; Agarwal et al. 2021). This
    helper is retained only for an optional per-arm averaged-series DISPLAY / the cost-sweep parallel;
    it must not drive a significance test. Returns ``None`` when the arm has no usable test record.
    """
    vecs = [v for r in records if r.get("arm") == arm and (v := _test_returns(r)) is not None]
    if not vecs:
        return None
    t_min = min(v.size for v in vecs)
    if t_min == 0:
        return None
    stacked = np.vstack([v[:t_min] for v in vecs])
    return stacked.mean(axis=0)


def _seed_scores(
    records: list[dict[str, Any]], arm: str, score_fn: Callable[[np.ndarray], float]
) -> dict[int, float]:
    """Per-SEED test score for an arm: ``{seed: score_fn(that seed's test series)}``.

    The campaign writes one frozen-winner TEST record per (arm, seed); this reduces each seed's
    realized test series to a single score (e.g. that seed's annualized Sharpe or CVaR). These
    per-seed scores — NOT a seed-averaged series — are the unit of the headline inference (rliable;
    Agarwal et al. 2021), so the across-seed (training-RNG) variance is carried rather than collapsed.
    Records with a missing/empty test vector or no ``seed`` are skipped; an arm with no usable record
    returns ``{}``.
    """
    out: dict[int, float] = {}
    for r in records:
        if r.get("arm") != arm or r.get("seed") is None:
            continue
        v = _test_returns(r)
        if v is None:
            continue
        _seed = int(r["seed"])
        _score = float(score_fn(v))
        # Duplicate (arm, seed) handling (deep review 2026-07-26; loop 6 introduced the guard, loop 13
        # refined it). This ORIGINALLY assigned unconditionally, so two records for one arm+seed silently
        # kept the LAST one — and every headline estimator is PAIRED on the seed, so a silently-dropped
        # record shifts a paired difference with nothing in the output to say it happened.
        #
        # But an UNCONDITIONAL raise is too brittle to ship: `src/io/results.py::load_all` de-duplicates
        # nothing, and `--resume` / winner re-runs can legitimately write a SECOND run directory for a
        # seed that already has a record. Hard-failing the whole analysis at the last step on a benign
        # re-write would be a worse trade than the bug it prevents.
        #
        # So: AGREEING duplicates (idempotent re-write, archive replay) pass silently; DISAGREEING
        # duplicates raise, because that is exactly the case where last-wins picked one number
        # arbitrarily and changed a paired estimate.
        _prior = out.get(_seed)
        if _prior is not None and not (
            _prior == _score or (math.isnan(_prior) and math.isnan(_score))
        ):
            raise ValueError(
                f"CONFLICTING duplicate test records for arm={arm!r} seed={_seed}: scores "
                f"{_prior!r} and {_score!r}. The per-seed score map can only keep one, so a "
                "seed-paired estimator would silently use whichever was written last. Deduplicate the "
                "run archive (or fix the writer) before analysing; identical duplicates are fine."
            )
        out[_seed] = _score
    return out


def collect_family_pvalues(
    records: list[dict[str, Any]],
    *,
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    cvar_levels: tuple[float, ...] = (0.05,),
    sharpe_p: float = 0.1,
    n_boot: int = 2000,
    q: float = 0.05,
    alpha_one_sided: float = 0.05,
    rng: np.random.Generator | None = None,
    risk_free: np.ndarray | None = None,
) -> dict[str, Any]:
    """Enumerate the pre-registered arm-contrast × held-out-metric family; record one-sided + BH decisions.

    For every ``(arm_a, arm_b)`` contrast and every held-out metric — the Sharpe ratio and the CVaR
    at each pre-registered ``cvar_levels`` — runs the rliable per-seed difference test (Agarwal et al.
    2021): each arm's PER-SEED scores (Sharpe / CVaR of every seed's frozen-winner test series) are
    reduced to an IQM point estimate, and a PAIRED stratified bootstrap over the shared training SEEDS
    (``src.inference.bootstrap.paired_seed_difference_test``) tests the IQM difference — carrying the
    across-seed (training-RNG) variance.

    R25 (2026-06-25, PREREGISTRATION §1/§10): the HEADLINE decision is now per-test ONE-SIDED in the
    predicted direction (``a`` better than ``b``). R64 (2026-06-28): ``p_one`` is the bootstrap's DIRECT
    upper-tail one-sided p (``res["pvalue_one_sided_greater"]`` = ``P(boot - obs >= obs)``), used when
    ``direction_ok`` (the effect is in-direction), else the one-sided test does NOT reject. This replaces
    the earlier ``p_two / 2`` (DEEP_H2 A5), which equals the true one-sided tail only under a symmetric
    bootstrap and mis-states it under any skew of the CVaR-5% difference. Each test therefore carries a
    ``pvalue_one_sided`` and a ``reject_one_sided`` (``direction_ok AND p_one <= alpha_one_sided``);
    :func:`h2_conjunction` partitions these into the two co-primary intersection-union tests (H2-RA
    on the 3 Sharpe legs, H2-Tail on the 3 CVaR-5% legs), each gated by its own 3-leg IUT with NO
    further leg correction (the conjunction IS the correction — Berger 1982; DEEP_H2 §3.3 option A).

    The Benjamini-Hochberg rejection set over the WHOLE m=6 union (level ``q``;
    ``config/inference.yaml: multiplicity.q = 0.05``) is STILL computed and returned as ``reject_bh``,
    but it is now a REPORTED SENSITIVITY (DEEP_H2 §3.3 option B), NOT the headline gate — the old
    ``(conjunction o BH-over-6)`` double-corrected (Berger 1982) and was under-powered against H2.

    NB (#9/#14, 2026-06-20): this replaces the prior construction, which AVERAGED the per-seed return
    series per arm and fed that single denoised series to a single-strategy stationary block-bootstrap.
    Averaging N i.i.d.-seed paths shrinks the tested object's variance ~N×, so that test was
    anti-conservative (the inflation grows with the across-seed variance — ~21% true-null rejection at
    the 5% level in a representative 30-seed calibration vs the correctly-sized ~5% here). The valid
    series-level tests (``sharpe_difference_test`` /
    ``cvar_difference_test``) remain for single-realization use; the campaign family uses the per-seed
    test. ``sharpe_p`` is retained for signature stability but no longer applies (seeds are i.i.d.).

    The bootstrap difference tests are two-sided; the rejection set returned here is the raw
    two-sided BH set. The DIRECTIONAL post-correction decision (reject AND in the predicted
    direction ``a`` better than ``b``) is layered on by :func:`h2_conjunction`; the per-test
    ``direction_ok`` flag and the signed effect are recorded here so the caller need not
    re-run the bootstrap.

    Parameters
    ----------
    records : list of dict
        Per-(arm, seed) frozen-winner TEST records (carry ``metrics['test_returns']``).
    contrasts : tuple of (str, str), optional
        Arm pairs ``(a, b)`` read "a predicted better than b". Default: :data:`H2_CONTRASTS`.
    cvar_levels : tuple of float, optional
        CVaR tail levels for the CVaR-difference leg of the family (default the headline
        fitness level ``0.05``; PREREGISTRATION §4 also freezes ``cvar_01`` — pass
        ``(0.05, 0.01)`` to include it, flagged high-variance there).
    sharpe_p, n_boot : float, int
        Stationary-bootstrap block-restart probability and replication count forwarded to
        the difference tests.
    q : float, optional
        Benjamini-Hochberg FDR level (``config/inference.yaml: multiplicity.q``).
    rng : numpy.random.Generator, optional
        Seeded for reproducibility.

    Returns
    -------
    dict
        ``{"labels": [...], "pvals": np.ndarray, "reject_bh": np.ndarray(bool),
        "tests": [ {arm_a, arm_b, metric, level, pvalue, pvalue_one_sided, stat, effect,
        direction_ok, reject_one_sided, reject_bh}, ... ], "q": q, "alpha_one_sided": float,
        "n_family": int, "skipped": [...]}``. ``effect`` is the signed ``stat_a - stat_b``
        (positive = ``a`` better: higher Sharpe / higher CVaR); ``direction_ok`` is ``effect > 0``;
        ``pvalue_one_sided`` is the direct upper-tail bootstrap p (R64) and ``reject_one_sided`` is
        ``direction_ok AND pvalue_one_sided <= alpha_one_sided`` (the R25 headline decision);
        ``reject_bh`` is the BH-over-the-union FDR set (a reported SENSITIVITY, not the gate).
        Contrasts whose arms lack a usable test record are reported in ``skipped`` (never fabricated).
    """
    from src.inference.bootstrap import cvar, iqm, paired_seed_difference_test, sharpe_ratio
    from src.inference.multiple_testing import benjamini_hochberg

    if rng is None:
        rng = np.random.default_rng(0)

    # Per-seed SCORES per (arm, metric), cached. The headline difference test is rliable-style
    # (Agarwal et al. 2021): per-seed scores -> IQM point estimate -> PAIRED stratified bootstrap over
    # SEEDS (paired_seed_difference_test), carrying the across-seed (training-RNG) variance. This
    # replaces the prior seed-AVERAGED single series fed to a single-strategy bootstrap, which
    # collapsed that variance ~N× and was anti-conservative (#9/#14, 2026-06-20). `sharpe_p` (block-
    # restart prob) no longer applies — seeds are i.i.d., so the seed bootstrap is i.i.d. — and is
    # accepted-but-ignored for signature stability.
    _sharpe_cache: dict[str, dict[int, float]] = {}
    _cvar_cache: dict[tuple[str, float], dict[int, float]] = {}

    # R20 (additive): when a per-period risk-free vector is passed, the SHARPE leg uses EXCESS returns
    # (r - rf). risk_free=None (the default) is BYTE-IDENTICAL to the frozen rf=0 headline, so
    # h2_conjunction is unaffected; the excess version is a reported robustness sensitivity only. The rf
    # vector is the test-window rf (aligned to each per-seed test series, which starts at test_start), so
    # rf[:v.size] pairs element-wise. CVaR (a RAW-loss tail measure) is intentionally left on raw returns.
    _rf_vec = None if risk_free is None else np.asarray(risk_free, dtype=float).ravel()

    def _sharpe_score(v: np.ndarray) -> float:
        if _rf_vec is None or _rf_vec.size == 0:
            return sharpe_ratio(v)
        x = np.asarray(v, dtype=float)
        m = min(x.size, _rf_vec.size)
        return sharpe_ratio(x[:m] - _rf_vec[:m])

    def _sharpe_seed(arm: str) -> dict[int, float]:
        if arm not in _sharpe_cache:
            _sharpe_cache[arm] = _seed_scores(records, arm, _sharpe_score)
        return _sharpe_cache[arm]

    def _cvar_seed(arm: str, level: float) -> dict[int, float]:
        key = (arm, float(level))
        if key not in _cvar_cache:
            _cvar_cache[key] = _seed_scores(records, arm, lambda v: cvar(v, float(level)))
        return _cvar_cache[key]

    def _paired(sa: dict[int, float], sb: dict[int, float]) -> tuple[np.ndarray, np.ndarray, int]:
        common = sorted(set(sa) & set(sb))  # shared training seeds -> element-wise paired by seed
        a = np.array([sa[s] for s in common], dtype=float)
        b = np.array([sb[s] for s in common], dtype=float)
        return a, b, len(common)

    def _one_sided(res: dict[str, float]) -> tuple[bool, float, bool]:
        """(direction_ok, p_one, reject_one) from a paired-bootstrap result (R25; R64).

        The DIRECT one-sided p in the predicted direction: ``p_one = res["pvalue_one_sided_greater"]``,
        the upper-tail bootstrap probability ``P(boot - obs >= obs)`` for H1: effect>0 (``a`` better).
        This REPLACES the old ``p_two / 2`` (DEEP_H2 A5), which equals the true tail only under a
        symmetric bootstrap and mis-states it under any skew of the CVaR-5% difference — it could
        flip the co-primary tail leg at alpha (R64 / DEEP_AUDIT 2026-06-28). The leg still REQUIRES the
        effect in-direction; ``reject_one`` is the genuinely one-sided leg decision at ``alpha_one_sided``
        — the unit the two co-primary IUTs gate on (NO BH on the legs).
        """
        direction_ok = bool(res["effect"] > 0.0)
        p_one = float(res["pvalue_one_sided_greater"])
        reject_one = bool(direction_ok and p_one <= alpha_one_sided)
        return direction_ok, p_one, reject_one

    tests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for arm_a, arm_b in contrasts:
        a, b, n_seeds = _paired(_sharpe_seed(arm_a), _sharpe_seed(arm_b))
        if n_seeds < 2:
            reason = (
                "no shared test seeds"
                if n_seeds == 0
                else "only 1 shared test seed (need >= 2 for the across-seed bootstrap)"
            )
            skipped.append({"arm_a": arm_a, "arm_b": arm_b, "reason": reason})
            continue

        # Sharpe leg of the family (higher IQM-over-per-seed-Sharpe = better).
        sr = paired_seed_difference_test(a, b, statistic=iqm, n_boot=n_boot, rng=rng)
        sr_dir, sr_p1, sr_rej1 = _one_sided(sr)
        # ── A16 (2026-08-01): the registered NON-INFERIORITY p, for validity-tier node N2 ────────
        # The node registers `test: h2_ra_iut_or_tost` — superiority OR equivalence — and
        # {theta > 0} UNION {-d < theta < d} IS {theta > -d}, a one-sided non-inferiority
        # hypothesis. So the registered disjunction is ONE test, not two, and this is it.
        # It is an IDENTITY, not an approximation: IQM is translation-equivariant, so shifting arm
        # A by +d shifts the estimate AND every bootstrap replicate by exactly d, making the NI
        # test at boundary -d algebraically the superiority test with the null moved. Same rng,
        # same n_boot, same statistic — deliberately, so no second inferential basis enters a
        # confirmatory node.
        # NOTE: THE PRIMARY MARGIN IS THE PERMISSIVE ONE (see _ni_margins). The conservative margin is
        # computed here beside it so the sensitivity cannot be selected after the fact.
        _m = _ni_margins()
        _ni = paired_seed_difference_test(a + _m["primary"], b, statistic=iqm,
                                          n_boot=n_boot, rng=rng)
        _ni_cons = paired_seed_difference_test(a + _m["conservative"], b, statistic=iqm,
                                               n_boot=n_boot, rng=rng)
        tests.append(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "metric": "sharpe",
                "level": None,
                "pvalue": float(sr["pvalue"]),
                "pvalue_one_sided": sr_p1,
                "pvalue_non_inferiority": float(_ni["pvalue_one_sided_greater"]),
                "pvalue_non_inferiority_conservative": float(_ni_cons["pvalue_one_sided_greater"]),
                "ni_margins": _m,
                "stat": float(sr["stat"]),
                "effect": float(sr["effect"]),
                "direction_ok": sr_dir,
                "reject_one_sided": sr_rej1,
                "n_seeds": n_seeds,
            }
        )

        # CVaR leg(s) of the family (higher / less-negative tail IQM = better).
        for level in cvar_levels:
            ca, cb, nc = _paired(_cvar_seed(arm_a, level), _cvar_seed(arm_b, level))
            if nc < 2:
                continue
            cv = paired_seed_difference_test(ca, cb, statistic=iqm, n_boot=n_boot, rng=rng)
            cv_dir, cv_p1, cv_rej1 = _one_sided(cv)
            tests.append(
                {
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "metric": "cvar",
                    "level": float(level),
                    "pvalue": float(cv["pvalue"]),
                    "pvalue_one_sided": cv_p1,
                    "stat": float(cv["stat"]),
                    "effect": float(cv["effect"]),
                    "direction_ok": cv_dir,
                    "reject_one_sided": cv_rej1,
                    "n_seeds": nc,
                }
            )

    pvals = np.asarray([t["pvalue"] for t in tests], dtype=float)
    labels = [
        f"{t['arm_a']}>{t['arm_b']}:{t['metric']}"
        + (f"@{t['level']:g}" if t["level"] is not None else "")
        for t in tests
    ]
    reject_bh = benjamini_hochberg(pvals, q=q) if pvals.size else np.zeros(0, dtype=bool)
    for t, rj in zip(tests, reject_bh):
        t["reject_bh"] = bool(rj)

    # # fail-loud (R13): when the FULL pre-registered contrast set ran (every arm present, the frozen
    # contrasts), the realized {contrast x metric x level} family MUST equal the frozen one. A run with a
    # missing comparator arm legitimately yields a SUBSET (reported in `skipped`) — we do not assert there
    # (a null is a credible finding, never a crash); the freeze check guards the mirror in that case.
    if contrasts == H2_CONTRASTS and not skipped:
        assert_realized_family_matches_frozen(tests, cvar_levels=cvar_levels)

    return {
        "labels": labels,
        "pvals": pvals,
        "reject_bh": reject_bh,
        "tests": tests,
        "q": float(q),
        "alpha_one_sided": float(alpha_one_sided),
        "n_family": int(pvals.size),
        "skipped": skipped,
    }


def romano_wolf_joint(
    scores: dict[str, np.ndarray],
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """JOINT Romano-Wolf stepdown over the contrast family on PER-SEED SCORES, ONE shared SEED resample/rep.

    Why a joint draw (the Rank-8 STOP-AND-FLAG on ``multiple_testing.romano_wolf``)
    --------------------------------------------------------------------------------
    ``src.inference.multiple_testing.romano_wolf`` is a *pure stepdown* over a precomputed
    ``(n_boot, n_hypotheses)`` ``boot_stats`` whose joint max is valid **iff** each ROW is one joint
    draw of all hypotheses under a SHARED resample. This routine builds that: it draws ONE SEED-index
    resample per replication and evaluates EVERY contrast's IQM-difference on that single shared draw,
    so the columns of each row share a resample (joint draw) AND the across-seed (training-RNG)
    variance is carried. It is the rliable (Agarwal et al. 2021) per-seed analogue of the prior
    series block-bootstrap, which fed a seed-AVERAGED series to the stepdown and was anti-conservative
    (#9/#14, 2026-06-20).

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        ``{arm: per-seed score array}`` for the chosen metric (e.g. each arm's per-seed Sharpes),
        ALIGNED across arms by seed (element ``i`` is the same training seed for every arm). All arms
        named in ``contrasts`` must be present.
    contrasts : tuple of (str, str), optional
        Arm pairs ``(a, b)`` read "a predicted better than b". Default :data:`H2_CONTRASTS`.
    statistic : callable, optional
        Per-arm central tendency over the per-seed scores (default the rliable IQM).
    n_boot, alpha : int, float
        Bootstrap replications and the family-wise error level.
    rng : numpy.random.Generator, optional
        Seeded for reproducibility.

    Returns
    -------
    dict
        ``{"labels", "stats", "reject_rw", "direction_ok"}``. ``reject_rw`` is the one-sided FWER-
        controlled stepdown rejection (``a`` better than ``b``); ``direction_ok`` flags ``effect > 0``
        per contrast. Raises ``KeyError`` if a named arm is missing.
    """
    from src.inference.bootstrap import iqm
    from src.inference.multiple_testing import romano_wolf

    if statistic is None:
        statistic = iqm
    if rng is None:
        rng = np.random.default_rng(0)

    # Align every involved arm's per-seed score array to the common leading length (the caller builds
    # them in a shared seed order; this `[:n]` is a safety no-op when they are already equal-length).
    arms = sorted({a for pair in contrasts for a in pair})
    missing = [a for a in arms if a not in scores]
    if missing:
        raise KeyError(f"romano_wolf_joint: scores missing arm(s) {missing}")
    n = min(int(np.asarray(scores[a]).size) for a in arms)  # common seed count
    aligned = {a: np.asarray(scores[a], dtype=float)[:n] for a in arms}

    m = len(contrasts)
    # One-sided observed IQM-difference per contrast: positive when a is better than b.
    obs = np.empty(m, dtype=float)
    direction_ok = np.empty(m, dtype=bool)
    for k, (arm_a, arm_b) in enumerate(contrasts):
        diff = statistic(aligned[arm_a]) - statistic(aligned[arm_b])
        obs[k] = diff
        direction_ok[k] = diff > 0.0

    # ONE shared SEED resample per replication; every contrast evaluated on that single draw -> the
    # columns of each row share a resample (joint draw) and carry the across-seed variance.
    boot = np.empty((n_boot, m), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, (arm_a, arm_b) in enumerate(contrasts):
            boot[i, k] = statistic(aligned[arm_a][idx]) - statistic(aligned[arm_b][idx])
    # Recentre at the observed difference -> null draws of the one-sided statistic.
    boot_centred = boot - obs[None, :]

    reject = romano_wolf(obs, boot_centred, alpha=alpha)
    labels = [f"{a}>{b}" for a, b in contrasts]
    return {
        "labels": labels,
        "stats": obs,
        "reject_rw": reject,
        "direction_ok": direction_ok,
    }


def h2_conjunction(
    records: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    q: float = 0.05,
    method: str = "bh",
    cvar_levels: tuple[float, ...] = (0.05,),
    sharpe_p: float = 0.1,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The pre-registered HEADLINE H2 test: TWO co-primary intersection-union tests (R25).

    H2 (PREREGISTRATION §1/§10; DEEP_H2 §7.1) is decided as TWO co-primary intersection-union tests
    (IUTs), each over three legs of the SAME (distributional) arm vs the three comparators::

        H2-RA   (risk-adjusted):  Sharpe  of  distributional > {scalar, placebo, scalar_cvar5}
        H2-Tail (tail outcome):   CVaR-5% of  distributional > {scalar, placebo, scalar_cvar5}

    on the held-out TEST leg, at matched compute. Each IUT is supported iff **all three** of its legs
    reject ONE-SIDED at ``alpha`` in the predicted direction (distributional strictly better) — with
    NO further leg correction, because a conjunction IS an intersection-union test and is already the
    multiplicity correction (joint size <= max leg size = ``alpha``; Berger 1982). This replaces the
    prior ``(conjunction o BH-over-6)``, which double-corrected and was under-powered against H2
    (DEEP_H2 §3.2-3.3). The BH rejection set over the m=6 union is still computed (in ``family``) and
    REPORTED as a sensitivity, never the gate.

    Two-tier verdict. ``H2_RA`` is the risk-adjusted-performance claim (the Sharpe IUT); ``H2_Tail``
    is the tail-outcome claim (the CVaR-5% IUT), corroborated — not gated — by the FZ0/(VaR,ES)
    comparative ES backtest where available (``src.inference.es_backtest.comparative_es_backtest``).
    For BACK-COMPAT, ``H2_supported``/``legs`` mirror the Sharpe IUT (H2-RA); the tail legs are in
    ``tail_legs`` and the structured verdicts in ``H2_RA`` / ``H2_Tail``.

    Parameters
    ----------
    records : list of dict
        Per-(arm, seed) frozen-winner TEST records (carry ``metrics['test_returns']``).
    alpha : float
        One-sided per-leg level for BOTH IUTs (``method='bh'``) and the family-wise level for the
        Romano-Wolf stepdown alternative (``method='rw'``). Default ``0.05``.
    q : float
        Benjamini-Hochberg FDR level for the REPORTED BH-over-the-union sensitivity in ``family``
        (``config/inference.yaml: multiplicity.q``); it does NOT gate the verdict (R25).
    method : {"bh", "rw"}
        IUT leg decision — the genuinely one-sided per-leg bootstrap (default) or the JOINT Romano-Wolf
        stepdown (:func:`romano_wolf_joint`, FWER, one-sided). Both are coherent IUTs (no extra leg
        correction); BH is the default (``config/inference.yaml: multiplicity.method``).
    cvar_levels, sharpe_p, n_boot, rng
        Forwarded to :func:`collect_family_pvalues` / :func:`romano_wolf_joint`. The tail IUT gates at
        the HEADLINE level = ``max(cvar_levels)`` — the LEAST-extreme level, the frozen 0.05. (NOT
        ``min``: under the sanctioned ``[0.05, 0.01]`` expansion, min would silently flip the
        co-primary gate onto the high-variance 1% tail; extra opt-in levels stay in ``family``.)

    Returns
    -------
    dict
        ``{"H2_supported": bool, "H2_RA": {...}, "H2_Tail": {...}, "verdict": str, "method": str,
        "legs": [...sharpe...], "tail_legs": [...cvar...], "family": <collect_family_pvalues output>,
        "missing": [...]}``. Each verdict is ``False`` (not an error) when a leg's arm has no test
        record — a null is a credible finding, never fabricated.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    family = collect_family_pvalues(
        records,
        contrasts=H2_CONTRASTS,
        cvar_levels=cvar_levels,
        sharpe_p=sharpe_p,
        n_boot=n_boot,
        q=q,
        alpha_one_sided=alpha,
        rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
    )

    missing = [f"{s['arm_a']}>{s['arm_b']}" for s in family["skipped"]]
    # The tail IUT is gated at the HEADLINE CVaR level (the frozen 0.05) — the LEAST-extreme (largest-
    # alpha) level. The opt-in cvar_01 is MORE extreme (smaller alpha, high-variance per Bauer 2025) and
    # must NEVER gate, so take max(cvar_levels): when only [0.05] is frozen this is 0.05; when cvar_01 is
    # added for reporting it stays in `family` but does not become the tail IUT's gating level.
    tail_level = headline_cvar_level(cvar_levels)  # row 30a: one shared resolution rule

    if method == "rw":
        from src.inference.bootstrap import cvar, sharpe_ratio

        # Per-seed Sharpe + CVaR scores per arm, each aligned to the COMMON seed set across all H2 arms
        # (the joint stepdown resamples SEEDS, so every arm must be indexed by the SAME seed order).
        # rliable per-seed analogue of the prior series block-bootstrap (#9/#14).
        h2_arms = {a for pair in H2_CONTRASTS for a in pair}

        def _rw_legs(score_fn: Callable[[np.ndarray], float]) -> dict[str, tuple[bool, bool]]:
            seed_scores = {arm: _seed_scores(records, arm, score_fn) for arm in h2_arms}
            common = (
                set.intersection(*(set(s) for s in seed_scores.values()))
                if all(seed_scores.values())
                else set()
            )
            if len(common) < 2:  # need >= 2 shared seeds for the across-seed bootstrap
                return {}
            cs = sorted(common)
            aligned = {arm: np.array([seed_scores[arm][s] for s in cs], dtype=float) for arm in seed_scores}
            rw = romano_wolf_joint(
                aligned, H2_CONTRASTS, n_boot=n_boot, alpha=alpha,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            return {
                lbl: (bool(rj), bool(dok))
                for lbl, rj, dok in zip(rw["labels"], rw["reject_rw"], rw["direction_ok"])
            }

        sharpe_by_label = _rw_legs(sharpe_ratio)
        tail_by_label = _rw_legs(lambda v: cvar(v, tail_level))

        def _rw_leg_list(by_label: dict[str, tuple[bool, bool]]) -> list[dict[str, Any]]:
            out_legs: list[dict[str, Any]] = []
            for arm_a, arm_b in H2_CONTRASTS:
                label = f"{arm_a}>{arm_b}"
                reject, dir_ok = by_label.get(label, (False, False))
                out_legs.append(
                    {
                        "contrast": label,
                        "reject": bool(reject and dir_ok),
                        "direction_ok": dir_ok,
                        "leg_supported": bool(reject and dir_ok),
                        # back-compat aliases (the Sharpe leg historically used `sharpe_*` names).
                        "sharpe_reject": reject,
                        "sharpe_direction_ok": dir_ok,
                    }
                )
            return out_legs

        legs = _rw_leg_list(sharpe_by_label)
        tail_legs = _rw_leg_list(tail_by_label)
    else:  # "bh" — the genuinely one-sided per-leg IUT (no leg correction; R25)
        def _one_sided_legs(metric: str, level: float | None) -> list[dict[str, Any]]:
            by_pair = {
                (t["arm_a"], t["arm_b"]): t
                for t in family["tests"]
                if t["metric"] == metric and (level is None or t.get("level") == level)
            }
            out_legs: list[dict[str, Any]] = []
            for arm_a, arm_b in H2_CONTRASTS:
                t = by_pair.get((arm_a, arm_b))
                if t is None:
                    out_legs.append(
                        {
                            "contrast": f"{arm_a}>{arm_b}",
                            "reject": False,
                            "direction_ok": False,
                            "leg_supported": False,
                            "sharpe_reject": False,
                            "sharpe_direction_ok": False,
                        }
                    )
                    continue
                reject = bool(t["reject_one_sided"])  # one-sided in the predicted direction, NO BH (R25)
                dir_ok = bool(t["direction_ok"])
                out_legs.append(
                    {
                        "contrast": f"{arm_a}>{arm_b}",
                        "reject": reject,
                        "direction_ok": dir_ok,
                        "leg_supported": reject,  # reject_one_sided already embeds the direction gate
                        "sharpe_reject": reject,
                        "sharpe_direction_ok": dir_ok,
                        "pvalue": float(t["pvalue"]),
                        "pvalue_one_sided": float(t["pvalue_one_sided"]),
                        "effect": float(t["effect"]),
                        # A16: present on the SHARPE (H2-RA) legs only — the tail family N1 is
                        # unchanged and must stay a pure superiority IUT.
                        **({"pvalue_non_inferiority": float(t["pvalue_non_inferiority"]),
                            "pvalue_non_inferiority_conservative":
                                float(t["pvalue_non_inferiority_conservative"]),
                            "ni_margins": t["ni_margins"]}
                           if "pvalue_non_inferiority" in t else {}),
                    }
                )
            return out_legs

        legs = _one_sided_legs("sharpe", None)
        tail_legs = _one_sided_legs("cvar", tail_level)

    def _iut_supported(leg_list: list[dict[str, Any]]) -> bool:
        return bool(leg_list) and len(leg_list) == len(H2_CONTRASTS) and all(
            leg["leg_supported"] for leg in leg_list
        )

    ra_supported = _iut_supported(legs)
    tail_supported = _iut_supported(tail_legs)

    if ra_supported and tail_supported:
        verdict = "H2-RA + H2-Tail supported"
    elif ra_supported:
        verdict = "H2-RA supported (tail not)"
    elif tail_supported:
        verdict = "H2-Tail supported (risk-adjusted not)"
    else:
        verdict = "neither (null)"

    return {
        # Back-compat: H2_supported / legs mirror the Sharpe IUT (H2-RA) — the historical headline gate.
        "H2_supported": ra_supported,
        "H2_RA": {"supported": ra_supported, "metric": "sharpe", "legs": legs},
        "H2_Tail": {
            "supported": tail_supported,
            "metric": "cvar",
            "level": tail_level,
            "legs": tail_legs,
            "corroborated_by": "fz0_var_es_comparative_backtest",
        },
        "verdict": verdict,
        "method": method,
        "alpha": float(alpha),
        "q": float(q),
        "legs": legs,
        "tail_legs": tail_legs,
        "family": family,
        "missing": missing,
    }


def h2_sharpe_rf_robustness(
    records: list[dict[str, Any]],
    risk_free: np.ndarray,
    *,
    cvar_levels: tuple[float, ...] = (0.05,),
    n_boot: int = 2000,
    q: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """ADDITIVE R20 sensitivity check: does the H2 SHARPE conjunction survive on EXCESS returns (r - rf)?

    The frozen headline (:func:`h2_conjunction` -> :func:`collect_family_pvalues`, rf=0) is UNCHANGED.
    This recomputes the SAME family with the per-period FRED ``DGS3MO`` rate subtracted from the SHARPE leg
    (the CVaR leg, a RAW-loss tail measure, stays on raw returns) and reports — per contrast — the rf=0 vs
    excess effect / p-value / direction / one-sided rejection, and whether the H2-RA Sharpe IUT (the
    distributional arm beats all three comparators one-sided at α; R25) holds BOTH ways.

    Why this matters (PREREGISTRATION R20). The per-seed Sharpe rf penalty is ``mean(rf)*sqrt(252)/sigma``
    — LARGER for LOWER-volatility arms. If the distributional (tail-aware) arm wins partly by producing
    lower realised volatility, the excess-return effect is SMALLER than the rf=0 effect. So rf is NOT a
    harmless constant for the arm contrast; this quantifies the shrinkage and certifies whether the
    headline is robust to the risk-free convention.

    Parameters
    ----------
    records : list of dict
        Per-(arm, seed) frozen-winner TEST records (carry ``metrics['test_returns']``).
    risk_free : np.ndarray
        Per-period rf vector for the TEST window (e.g.
        ``market_reference.load_risk_free_daily(panel.dates[test_start:test_end]).daily``), aligned to
        each per-seed test series (which starts at ``test_start``), so ``rf[:v.size]`` pairs element-wise.

    Returns
    -------
    dict
        ``{"survives": bool, "contrasts": [ {contrast, effect_rf0, effect_excess, pvalue_*, reject_*,
        direction_ok_*, leg_survives, effect_shrinkage}, ... ], "rf_annualised_pct": float, "note": str}``.
        ``effect_shrinkage = effect_rf0 - effect_excess`` (positive => rf shrank the distributional edge).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rf = np.asarray(risk_free, dtype=float).ravel()
    # SAME bootstrap seed for both runs so the ONLY difference is the rf (isolates the convention effect).
    seed = int(rng.integers(0, 2**32 - 1))
    fam0 = collect_family_pvalues(
        records, contrasts=H2_CONTRASTS, cvar_levels=cvar_levels, n_boot=n_boot, q=q,
        rng=np.random.default_rng(seed), risk_free=None,
    )
    famx = collect_family_pvalues(
        records, contrasts=H2_CONTRASTS, cvar_levels=cvar_levels, n_boot=n_boot, q=q,
        rng=np.random.default_rng(seed), risk_free=rf,
    )

    def _sharpe_legs(fam: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {(t["arm_a"], t["arm_b"]): t for t in fam["tests"] if t["metric"] == "sharpe"}

    s0, sx = _sharpe_legs(fam0), _sharpe_legs(famx)
    rows: list[dict[str, Any]] = []
    for a, b in H2_CONTRASTS:
        t0, tx = s0.get((a, b)), sx.get((a, b))
        if t0 is None or tx is None:
            continue
        # R25: the headline H2-RA leg decision is the ONE-SIDED IUT reject (reject_one_sided already
        # embeds the direction gate), NOT the BH-over-6 set — so this rf sensitivity mirrors the gate.
        leg_survives = bool(t0["reject_one_sided"] and tx["reject_one_sided"])
        rows.append({
            "contrast": f"{a}>{b}",
            "effect_rf0": float(t0["effect"]), "effect_excess": float(tx["effect"]),
            "pvalue_rf0": float(t0["pvalue"]), "pvalue_excess": float(tx["pvalue"]),
            "reject_rf0": bool(t0["reject_one_sided"]), "reject_excess": bool(tx["reject_one_sided"]),
            "direction_ok_rf0": bool(t0["direction_ok"]), "direction_ok_excess": bool(tx["direction_ok"]),
            "leg_survives": leg_survives,
            "effect_shrinkage": float(t0["effect"] - tx["effect"]),
        })
    survives = bool(rows) and len(rows) == len(H2_CONTRASTS) and all(r["leg_survives"] for r in rows)
    return {
        "survives": survives,
        "contrasts": rows,
        "rf_annualised_pct": float(np.nanmean(rf) * 252 * 100.0) if rf.size else 0.0,
        "note": "Sharpe leg on excess returns (r - DGS3MO); CVaR raw; frozen headline (rf=0) unchanged (R20).",
    }


# =========================================================================== #
# DEEP_H4 / DEEP_H3 / DEEP_STATS / DEEP_H2 — ADDITIVE report-only secondaries    #
# (DISJOINT out[...] keys; NEVER the frozen m=6 H2 family; mirror the H2-RA      #
#  per-seed IQM paired-bootstrap one-sided IUT pattern.)                         #
# =========================================================================== #
#: The H4 search controls the LLM is contrasted against (DEEP_H4 §0; PREREGISTRATION §1/§3):
#: H4a = vs random-search-over-code; H4b = vs GP-EI; H4c = vs CMA-ES; H4d = vs TPE -- the 4-optimiser
#: portfolio whose all-supported conjunction is the CONFIRMATORY N4 beat-the-MAX IUT (2026-07-26).
#: Each entry is ``(arm_a, arm_b)`` read "a predicted BETTER than b" (one-sided, higher Sharpe).
H4_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("h4a", "distributional", "random_search"),
    ("h4b", "distributional", "bayes_opt"),
    ("h4c", "distributional", "cma_es"),
    ("h4d", "distributional", "tpe"),
)


def _one_sided_from_two(
    effect: float,
    pvalue_two_sided: float,
    alpha: float,
    *,
    pvalue_one_sided: float | None = None,
) -> tuple[bool, float, bool]:
    """(direction_ok, p_one, reject_one) from a paired-bootstrap result.

    The SAME one-sided convention the headline H2 leg uses (``collect_family_pvalues._one_sided``; R25/R64):
    prefer the DIRECT upper-tail one-sided p (``pvalue_one_sided`` = ``res["pvalue_one_sided_greater"]``),
    which is valid under a skewed bootstrap; fall back to the symmetric ``p_two / 2`` ONLY when a direct
    one-sided p is unavailable (the /2 mis-states the one-sided tail under any skew of the CVaR difference and can flip a leg
    at alpha — R64 / DEEP_AUDIT 2026-06-28). ``reject_one`` requires the effect in the predicted direction
    (``effect > 0`` = ``a`` better). Factored out so H3/H4 reuse the EXACT headline convention.
    """
    direction_ok = bool(effect > 0.0)
    p_one = float(pvalue_one_sided) if pvalue_one_sided is not None else float(pvalue_two_sided) / 2.0
    reject_one = bool(direction_ok and p_one <= alpha)
    return direction_ok, p_one, reject_one


def _iqm_tost(
    a: np.ndarray,
    b: np.ndarray,
    margin: float,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Symmetric-margin TOST equivalence on the IQM difference (the H2 test-statistic's units).

    The headline difference test reduces each arm's per-seed scores to an rliable IQM and bootstraps the
    paired IQM difference (``collect_family_pvalues`` / ``paired_seed_difference_test`` with
    ``statistic=iqm``). This is the equivalence COMPLEMENT in the SAME statistic + units (DEEP_H2 §5.3;
    Lakens 2017): the two arms are practically equivalent within ``±margin`` iff the two-sided 90%
    (``1 - 2·0.05``) percentile bootstrap CI for ``IQM(a) - IQM(b)`` lies entirely inside ``(-margin,
    +margin)``. PAIRED by seed (shared draw on both arms), so it carries the across-seed variance exactly
    like the difference test. ``margin`` is in the units of ``a``/``b`` (per-seed Sharpe or CVaR), NOT
    validation-DSR units — the bound is "±0.05 in the test-statistic's units" as the prompt specifies
    (distinct from the frozen validation-DSR SESOI used by ``scripts/power_analysis.tost_equivalence``).
    """
    from src.inference.bootstrap import iqm

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    if a.shape != b.shape:
        raise ValueError("a and b must be the same shape (paired per-seed scores)")
    n = a.size
    estimate = float(iqm(a) - iqm(b))
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)  # paired: SAME seed-index draw for both arms
        boot[i] = iqm(a[idx]) - iqm(b[idx])
    ci_low = float(np.quantile(boot, 0.05))
    ci_high = float(np.quantile(boot, 0.95))
    # §2a(c) registered severity presentation (Mayo–Spanos, 2026-07-06): with what post-data
    # severity did the claim |mu| <= m pass, per margin m? EMPIRICAL from the same paired bootstrap:
    # recenter the draws at the claim's boundary and measure how often a result LESS in accord with
    # the claim than the observed estimate would arise there; min(upper, lower) is the binding side
    # of the symmetric claim. Reported at 0.5x/1x/1.5x/2x the frozen margin (the severity CURVE).
    centered = boot - float(np.mean(boot))
    severity: dict[str, float] = {}
    for mult in (0.5, 1.0, 1.5, 2.0):
        m = float(margin) * mult
        sev_upper = float(np.mean(centered + m >= estimate))
        sev_lower = float(np.mean(centered - m <= estimate))
        severity[f"{mult:g}x_margin"] = round(min(sev_upper, sev_lower), 4)
    return {
        "equivalent": bool(ci_low > -margin and ci_high < margin),
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "margin": float(margin),
        "n_seeds": int(n),
        "severity": severity,
    }


#: How each H4 control should READ in the write-up (T3.4 (b); DEEP_H4 §1.2). H4a (random-search-over-code)
#: is the IN-FAMILY reference: it samples the SAME six-term reward family the LLM authors over (R28), so a
#: positive H4a is "procedure (open-ended reflection) over the SAME richness". H4b (Bayes-opt-over-template)
#: is the fixed-parametric-family reference. Promoting these to NAMED references is what turns H4 from a bare
#: nested horse-race into a procedure-vs-richness reading.
_H4_REFERENCE_FRAMING: dict[str, str] = {
    "h4a": "in-family random-search reference (same 6-term family, R28) — isolates PROCEDURE at matched richness",
    "h4b": "fixed-parametric-template reference (Bayes-opt over the BO family) — open-ended language vs fixed family",
    "h4c": "CMA-ES-over-template reference (evolution-strategy DFO) -- best-in-class numerical search of the family",
    "h4d": "TPE-over-template reference (density-ratio DFO) -- low-budget-friendly numerical search of the family",
}


def h4_search_controls(
    records: list[dict[str, Any]],
    *,
    winner_arm: str = "distributional",
    contrasts: tuple[tuple[str, str, str], ...] = H4_CONTRASTS,
    n_boot: int = 2000,
    alpha: float = 0.05,
    equiv_margin: float = _frozen_equiv_margin(),  # P14-F2: read the frozen SESOI from config, not a literal 0.05
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """H4 — the LLM winner vs the non-LLM search controls (DEEP_H4; PREREGISTRATION §1/§3).

    Two pre-registered difference tests on the SEALED test leg, each mirroring the H2-RA per-seed IQM
    paired bootstrap exactly (per-seed Sharpe -> IQM -> PAIRED stratified bootstrap over the shared
    training seeds; ``src.inference.bootstrap.paired_seed_difference_test``), one-sided in the predicted
    direction (the LLM is predicted to beat each control):

      * **H4a** — ``distributional`` vs ``random_search`` (random-search-over-code);
      * **H4b** — ``distributional`` vs ``bayes_opt`` (GP-EI-over-template);
      * **H4c** -- ``distributional`` vs ``cma_es`` (CMA-ES-over-template);
      * **H4d** -- ``distributional`` vs ``tpe`` (TPE-over-template).
    The all-supported conjunction is the confirmatory **N4 beat-the-MAX IUT** (LLM beats the best optimiser).

    SCOPE (DEEP_H4 §1.2 — state precisely; this function only computes, the write-up scopes): H4a is the
    closest to a *procedure-at-comparable-richness* comparison (both emit code); H4b is an *open-ended
    language vs fixed parametric family* comparison. Neither asserts "the LLM is a better optimiser over
    an identical space" — the LLM's hypothesis space is deliberately richer.

    REFERENCE FRAMING (T3.4 (b); DEEP_H4 §1.2): each control is reported with its :data:`_H4_REFERENCE_FRAMING`
    label — H4a as the IN-FAMILY random-search REFERENCE (the same six-term family the LLM authors over, R28)
    and H4b as the fixed-parametric-template reference — so H4 reads as **procedure-vs-richness**, not a bare
    nested horse-race. The label rides on each test row + a top-level ``reference_framing`` map.

    EQUIVALENCE (T3.4 (a); DEEP_H4 §1.2): each leg ALSO carries a symmetric ``±equiv_margin`` TOST on the
    per-seed Sharpe-IQM difference (:func:`_iqm_tost`), exactly as H3 does — so a non-rejection of H4 against
    a control can be reported as a BOUNDED equivalence ("the LLM and the in-family search are equivalent
    within ±0.05 in Sharpe-IQM units") rather than mere absence of evidence (Lakens 2017). This closes the
    H4-vs-H3 asymmetry (H3 had a TOST bound; H4 did not).

    MULTIPLICITY (DEEP_H4 §4 A5; +cma_es/tpe 2026-07-26): H4 is its OWN family of 4 tests {H4a, H4b, H4c,
    H4d} -- and its ALL-SUPPORTED conjunction is exactly the CONFIRMATORY N4 beat-the-MAX IUT (the LLM
    beats the best of the {random, GP-EI, CMA-ES, TPE} portfolio; size<=alpha for any portfolio size,
    berger1982iut). NOT in the frozen m=6 H2 family. A Bonferroni-over-4 reported decision
    (``reject_one_sided_bonferroni`` at ``alpha/4``) is the separate-estimands mirror carried alongside
    the per-test one-sided decision so the multiplicity is explicit; the
    per-test one-sided p is also returned for transparency.

    DISJOINT KEY: writes ``out["h4"]`` with NO ``arm_a/arm_b/metric/level`` family-tuple keys (it uses
    ``test`` + ``a``/``b`` + ``contrast``), so ``assert_realized_family_matches_frozen`` never sees it and
    the frozen H2 family is untouched.

    Returns
    -------
    dict
        ``{"status", "winner_arm", "alpha", "n_tests", "bonferroni_alpha", "equiv_margin",
        "reference_framing", "tests": [ {test, a, b, reference, effect, pvalue_two_sided, pvalue_one_sided,
        direction_ok, reject_one_sided, reject_one_sided_bonferroni, equivalence, verdict, n_seeds}, ... ],
        "skipped": [...], "all_supported": bool, "all_supported_bonferroni": bool}``. Gracefully
        ``status="skipped"`` when the winner arm has no test records; per-contrast skips (a control arm
        absent) are reported in ``skipped`` (never fabricated).
    """
    from src.inference.bootstrap import iqm, paired_seed_difference_test, sharpe_ratio

    if rng is None:
        rng = np.random.default_rng(0)
    n_tests = len(contrasts)
    bonf_alpha = float(alpha) / float(n_tests) if n_tests else float(alpha)

    win_scores = _seed_scores(records, winner_arm, sharpe_ratio)
    if len(win_scores) < 2:
        return {
            "status": "skipped",
            "reason": f"LLM winner arm {winner_arm!r} has < 2 test seeds (test/baseline stage not run?)",
            "winner_arm": winner_arm,
        }

    tests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for test_id, arm_a, arm_b in contrasts:
        sa = win_scores if arm_a == winner_arm else _seed_scores(records, arm_a, sharpe_ratio)
        sb = _seed_scores(records, arm_b, sharpe_ratio)
        common = sorted(set(sa) & set(sb))  # paired element-wise by the shared training seed
        if len(common) < 2:
            skipped.append({
                "test": test_id, "a": arm_a, "b": arm_b,
                "reason": "no shared test seeds" if not common else "only 1 shared test seed (need >= 2)",
            })
            continue
        a = np.array([sa[s] for s in common], dtype=float)
        b = np.array([sb[s] for s in common], dtype=float)
        res = paired_seed_difference_test(a, b, statistic=iqm, n_boot=n_boot, rng=rng)
        _p1s = float(res["pvalue_one_sided_greater"])  # direct one-sided p (R64; not p_two/2)
        dir_ok, p1, rej1 = _one_sided_from_two(
            float(res["effect"]), float(res["pvalue"]), alpha, pvalue_one_sided=_p1s
        )
        _, _, rej1_bonf = _one_sided_from_two(
            float(res["effect"]), float(res["pvalue"]), bonf_alpha, pvalue_one_sided=_p1s
        )
        # T3.4 (a): symmetric-margin TOST equivalence on the SAME per-seed Sharpe-IQM difference (mirrors H3),
        # so a non-rejection reads as a BOUNDED equivalence rather than absence of evidence (Lakens 2017).
        equiv = _iqm_tost(a, b, float(equiv_margin), n_boot=n_boot, rng=rng)
        if rej1:
            verdict = f"LLM > {arm_b} (difference)"
        elif equiv["equivalent"]:
            verdict = f"equivalent within ±{equiv_margin:g} (bounded null)"
        else:
            verdict = "inconclusive (neither difference nor equivalence)"
        tests.append({
            "test": test_id,
            "a": arm_a,
            "b": arm_b,
            "metric": "sharpe",
            # T3.4 (b): the procedure-vs-richness reference label for this control (DEEP_H4 §1.2).
            "reference": _H4_REFERENCE_FRAMING.get(test_id, ""),
            "effect": float(res["effect"]),
            "pvalue_two_sided": float(res["pvalue"]),
            "pvalue_one_sided": p1,
            "direction_ok": dir_ok,
            "reject_one_sided": rej1,
            "reject_one_sided_bonferroni": rej1_bonf,
            "equivalence": equiv,
            "verdict": verdict,
            "n_seeds": len(common),
        })

    ran = [t["test"] for t in tests]
    all_ran = len(tests) == n_tests
    return {
        "status": "ok" if tests else "skipped",
        "reason": None if tests else "no H4 control arm has >= 2 shared test seeds",
        "winner_arm": winner_arm,
        "alpha": float(alpha),
        "n_tests": n_tests,
        "bonferroni_alpha": bonf_alpha,
        "equiv_margin": float(equiv_margin),
        # T3.4 (b): the reference-framing map, so the write-up reads H4 as procedure-vs-richness.
        "reference_framing": dict(_H4_REFERENCE_FRAMING),
        "tests": tests,
        "skipped": skipped,
        # "supported" = ALL pre-registered H4 tests ran AND rejected one-sided (the LLM beat BOTH controls).
        "all_supported": bool(all_ran and all(t["reject_one_sided"] for t in tests)),
        "all_supported_bonferroni": bool(all_ran and all(t["reject_one_sided_bonferroni"] for t in tests)),
        # T3.4 (a): ALL ran legs are equivalence-bounded (a bankable null when not all_supported).
        "all_equivalent": bool(all_ran and all(t["equivalence"]["equivalent"] for t in tests)),
        "tests_ran": ran,
    }


def _h3_placebo_relative_uplift(
    iterative_records: list[dict[str, Any]],
    single_shot_records: list[dict[str, Any]],
    *,
    arm: str,
    placebo_arm: str,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The PAIRED placebo-relative uplift-difference (T3.4 c) — does reflection leave an INFORMATION signature?

    The bare H3 contrast (iterative − single-shot for the distributional arm) confounds two things reflection
    can do: track the FED information, and merely resample more code paths. The ``placebo`` arm reflects on a
    CONTENT-FREE feedback block (matched length, zero tail information), so it isolates the resample-only
    component. The DIFFERENCE-OF-UPLIFTS

        d(seed) = [iter_dist(seed) − ss_dist(seed)] − [iter_plac(seed) − ss_plac(seed)]

    is the reflection uplift SPECIFIC to real tail information: a paired-by-seed bootstrap of ``mean(d)``
    that does NOT reject (``d ≈ 0``) says reflection on the real distributional feedback produced NO
    information-tracking signature beyond what content-free reflection already produces — the qualitative
    counterpart to a null H3 (DEEP_H3 §8.2; the placebo is the same control the H2 headline uses).

    Reported TWO-SIDED (the sign is not pre-committed — reflection could help OR hurt relative to placebo).
    Gracefully returns ``{"status": "skipped", "reason": ...}`` when the placebo arm has no single-shot OR no
    iterative test condition sharing >= 2 seeds with the distributional pair (the prototype/headline H3
    single-shot stage runs only the distributional arm, so this activates ONLY if a placebo single-shot
    archive is ALSO present — never fabricated).
    """
    from src.inference.bootstrap import paired_seed_difference_test, sharpe_ratio

    iter_d = _seed_scores(iterative_records, arm, sharpe_ratio)
    ss_d = _seed_scores(single_shot_records, arm, sharpe_ratio)
    iter_p = _seed_scores(iterative_records, placebo_arm, sharpe_ratio)
    ss_p = _seed_scores(single_shot_records, placebo_arm, sharpe_ratio)
    common = sorted(set(iter_d) & set(ss_d) & set(iter_p) & set(ss_p))
    if len(common) < 2:
        return {
            "status": "skipped",
            "reason": (
                f"placebo-relative uplift needs >= 2 seeds shared across BOTH conditions of {arm!r} AND "
                f"{placebo_arm!r}; got {len(common)} (the single-shot stage runs only the distributional "
                "arm unless a placebo single-shot archive is also supplied)"
            ),
            "placebo_arm": placebo_arm,
            "n_shared_seeds": len(common),
        }
    uplift_dist = np.array([iter_d[s] - ss_d[s] for s in common], dtype=float)
    uplift_plac = np.array([iter_p[s] - ss_p[s] for s in common], dtype=float)
    # Paired difference-of-uplifts, tested two-sided on the per-seed MEAN (uplift is already a per-seed signed
    # delta, so the mean is the natural statistic — no second IQM is layered on).
    res = paired_seed_difference_test(uplift_dist, uplift_plac, statistic=np.mean, n_boot=n_boot, rng=rng)
    p_two = float(res["pvalue"])
    # The caller's alpha (the H3 level), not a hardcoded 0.05 — so an alpha override propagates here too.
    reject = bool(p_two <= alpha)
    return {
        "status": "ok",
        "placebo_arm": placebo_arm,
        "effect": float(res["effect"]),         # mean(uplift_dist − uplift_plac), Sharpe units
        "pvalue_two_sided": p_two,
        "reject_two_sided": reject,
        "mean_uplift_distributional": float(np.mean(uplift_dist)),
        "mean_uplift_placebo": float(np.mean(uplift_plac)),
        "n_seeds": len(common),
        "interpretation": (
            "reflection on the real distributional feedback produced an uplift DISTINGUISHABLE from "
            "content-free (placebo) reflection — an information-tracking signature"
            if reject
            else "reflection left NO information-tracking signature beyond content-free reflection "
            "(distributional and placebo uplifts indistinguishable) — the qualitative null"
        ),
    }


def h3_iterative_vs_singleshot(
    iterative_records: list[dict[str, Any]],
    single_shot_records: list[dict[str, Any]] | None,
    *,
    arm: str = "distributional",
    placebo_arm: str = "placebo",
    n_boot: int = 2000,
    alpha: float = 0.05,
    equiv_margin: float = _frozen_equiv_margin(),  # P14-F2: read the frozen SESOI from config, not a literal 0.05
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """H3 — iterative reflection vs single-shot best-of-N, at matched budget (DEEP_H3; PREREGISTRATION §1/§6).

    The canonical contrast (DEEP_H3 §1.1, §2.1-i), WITHIN the ``arm`` (default distributional): the
    ITERATIVE condition (the headline reflect-on-best search winner, in ``iterative_records``) vs the
    SINGLE-SHOT condition (1 generation, best-of-(G·M), in ``single_shot_records`` — D-2 archives it under
    ``test_h3_singleshot/distributional``), winner selection identical (best validation DSR), both re-run
    at the same campaign seeds.

    THREE decisions are reported (DEEP_H3 §8.1-8.2):
      * a **difference test** — per-seed Sharpe -> IQM -> PAIRED stratified bootstrap over shared seeds
        (mirroring H2-RA, one-sided "iterative better than single-shot"); and
      * a **TOST equivalence** at ``±equiv_margin`` in the test-statistic's (per-seed Sharpe IQM) units —
        the bankable-null bound ("iterative ≈ single-shot within ±0.05"), the DEEP_H3 §8.2 move that turns
        a non-rejection into a bounded finding; and
      * a **paired placebo-relative uplift difference** (T3.4 c; :func:`_h3_placebo_relative_uplift`) — the
        difference between the distributional arm's (iterative − single-shot) uplift and the ``placebo_arm``'s,
        so a null reads as "reflection left no information-tracking signature beyond content-free reflection".
        It rides in ``out["placebo_relative_uplift"]`` and GRACEFULLY SKIPS unless a placebo single-shot
        condition is also present (the headline single-shot stage runs only the distributional arm), so it is
        never fabricated.

    DEEP_H3's pre-registered prediction is the NULL/equivalence (reflection is expected to be weak in this
    sparse, noisy-verifier regime); this frames the likely null as a bounded finding, not a defect.

    GRACEFUL SKIP: when ``single_shot_records`` is ``None``/empty (the single-shot archive is absent — it
    is a SEPARATE, manually-launched run, DEEP_H3 §2.3/§2.4), returns ``status="skipped"`` so H3 simply is
    not reported rather than fabricated.

    DISJOINT KEY: writes ``out["h3"]`` with NO family-tuple keys, so the frozen m=6 H2 family is untouched.

    Returns
    -------
    dict
        ``{"status", "arm", "alpha", "difference": {effect, pvalue_two_sided, pvalue_one_sided,
        direction_ok, reject_one_sided, n_seeds}, "equivalence": {equivalent, estimate, ci_low, ci_high,
        margin, n_seeds}, "placebo_relative_uplift": {...}, "verdict": str}`` or
        ``{"status": "skipped", "reason": ...}``.
    """
    from src.inference.bootstrap import iqm, paired_seed_difference_test, sharpe_ratio

    if not single_shot_records:
        return {
            "status": "skipped",
            "reason": "single-shot archive absent (test_h3_singleshot/<arm> not provided — a separate "
            "manually-launched run, DEEP_H3 §2.3/§2.4)",
            "arm": arm,
        }
    if rng is None:
        rng = np.random.default_rng(0)

    iter_scores = _seed_scores(iterative_records, arm, sharpe_ratio)
    ss_scores = _seed_scores(single_shot_records, arm, sharpe_ratio)
    common = sorted(set(iter_scores) & set(ss_scores))
    if len(common) < 2:
        return {
            "status": "skipped",
            "reason": (
                f"need >= 2 shared seeds across the iterative + single-shot {arm!r} winners; "
                f"got {len(common)}"
            ),
            "arm": arm,
            "n_iter_seeds": len(iter_scores),
            "n_single_shot_seeds": len(ss_scores),
        }
    a = np.array([iter_scores[s] for s in common], dtype=float)  # iterative
    b = np.array([ss_scores[s] for s in common], dtype=float)    # single-shot

    diff = paired_seed_difference_test(a, b, statistic=iqm, n_boot=n_boot, rng=rng)
    dir_ok, p1, rej1 = _one_sided_from_two(
        float(diff["effect"]), float(diff["pvalue"]), alpha,
        pvalue_one_sided=float(diff["pvalue_one_sided_greater"]),
    )
    equiv = _iqm_tost(a, b, float(equiv_margin), n_boot=n_boot, rng=rng)

    # T3.4 (c): the paired placebo-relative uplift difference (its own seeded rng so it is order-independent).
    placebo_uplift = _h3_placebo_relative_uplift(
        iterative_records, single_shot_records, arm=arm, placebo_arm=placebo_arm,
        n_boot=n_boot, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)), alpha=alpha,
    )

    if rej1:
        verdict = "iterative > single-shot (difference)"
    elif equiv["equivalent"]:
        verdict = f"equivalent within ±{equiv_margin:g} (bounded null)"
    else:
        verdict = "inconclusive (neither difference nor equivalence)"

    return {
        "status": "ok",
        "arm": arm,
        "alpha": float(alpha),
        "difference": {
            "effect": float(diff["effect"]),
            "pvalue_two_sided": float(diff["pvalue"]),
            "pvalue_one_sided": p1,
            "direction_ok": dir_ok,
            "reject_one_sided": rej1,
            "n_seeds": len(common),
        },
        "equivalence": equiv,
        "placebo_relative_uplift": placebo_uplift,
        "verdict": verdict,
    }


def h2_tost(
    records: list[dict[str, Any]],
    *,
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    cvar_level: float = 0.05,
    equiv_margin: float = _frozen_equiv_margin(),  # P14-F2: read the frozen SESOI from config, not a literal 0.05
    tail_margin_fraction: float = 0.25,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Headline TOST — equivalence for the H2-RA + H2-Tail per-seed IQM differences (DEEP_H2 §5.3).

    The bankable-null complement to :func:`h2_conjunction`. For EACH H2 contrast (distributional vs
    {scalar, placebo, scalar_cvar5}) and EACH co-primary metric — Sharpe (H2-RA) and CVaR-5% (H2-Tail) —
    runs the symmetric ``±equiv_margin`` TOST on the per-seed IQM difference (:func:`_iqm_tost`), in the
    test-statistic's OWN units (per-seed Sharpe / CVaR, NOT validation-DSR). A leg is "equivalent" iff its
    90% paired bootstrap CI for ``IQM(a) - IQM(b)`` lies inside ``(-margin, +margin)``.

    So a non-rejection of an H2 leg can be reported as a BOUNDED effect ("distributional and scalar are
    equivalent within ±0.05 in Sharpe-IQM units") rather than mere absence of evidence (Lakens 2017;
    PREREGISTRATION §10 R12 frames the SESOI in validation-DSR units — this is the test-statistic-units
    companion the prompt specifies, reported alongside).

    P6-code (2026-07-01). The raw ±0.05 CVaR-Tail band is LARGE relative to the CVaR magnitude (a daily
    CVaR-5% is O(0.01–0.06)), so a ±0.05 band can trivially contain the CI and over-claim equivalence.
    ALONGSIDE the raw band, each tail leg therefore ALSO reports a RELATIVE band expressed as a FRACTION
    (``tail_margin_fraction``, default 25%) of the |baseline CVaR| — the comparator arm's IQM CVaR — so the
    equivalence is stated in a scale-appropriate, interpretable unit ("within 25% of the baseline tail
    loss"). Both verdicts are reported; NEITHER gates. The RA (Sharpe) leg is unchanged (Sharpe is unitless,
    so a fractional restatement is not meaningful there).

    DISJOINT KEY: ``out["h2_tost"]`` carries NO family-tuple keys, so the frozen m=6 assert is untouched.

    Returns
    -------
    dict
        ``{"status", "margin", "tail_margin_fraction", "ra": [ {contrast, ...tost...}, ... ],
        "tail": {level, legs:[ {..., baseline_cvar, margin_fraction, equivalent_fraction,
        margin_fraction_abs}, ...]}, "skipped": [...]}``; ``status="skipped"`` only if NO contrast had >= 2
        shared seeds on either leg.
    """
    from src.inference.bootstrap import cvar, iqm, sharpe_ratio

    if rng is None:
        rng = np.random.default_rng(0)
    margin = float(equiv_margin)
    frac = float(tail_margin_fraction)

    def _legs(
        score_fn: Callable[[np.ndarray], float], *, relative: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scores = {arm: _seed_scores(records, arm, score_fn) for arm in {a for pair in contrasts for a in pair}}
        legs: list[dict[str, Any]] = []
        skipped_legs: list[dict[str, Any]] = []
        for arm_a, arm_b in contrasts:
            sa, sb = scores.get(arm_a, {}), scores.get(arm_b, {})
            common = sorted(set(sa) & set(sb))
            if len(common) < 2:
                skipped_legs.append({"contrast": f"{arm_a}>{arm_b}", "reason": "< 2 shared seeds"})
                continue
            a = np.array([sa[s] for s in common], dtype=float)
            b = np.array([sb[s] for s in common], dtype=float)
            t = _iqm_tost(a, b, margin, n_boot=n_boot, rng=rng)
            t["contrast"] = f"{arm_a}>{arm_b}"
            if relative:
                # RELATIVE band: margin = fraction · |baseline CVaR| (the comparator arm_b's IQM CVaR). Report
                # BOTH bands. Only the THRESHOLD changes (not the estimate or CI), so we re-derive equivalence
                # from the ALREADY-computed absolute-CVaR-unit CI against the fractional band — no second
                # bootstrap needed (the CI is unchanged; we just compare it to a different, scale-relative margin).
                baseline_cvar = float(iqm(b))
                frac_margin = frac * abs(baseline_cvar)
                t["baseline_cvar"] = baseline_cvar
                t["margin_fraction"] = frac
                t["margin_fraction_abs"] = frac_margin
                t["equivalent_fraction"] = bool(
                    frac_margin > 0.0 and t["ci_low"] > -frac_margin and t["ci_high"] < frac_margin
                )
            legs.append(t)
        return legs, skipped_legs

    ra_legs, ra_skip = _legs(sharpe_ratio)
    tail_legs, tail_skip = _legs(lambda v: cvar(v, float(cvar_level)), relative=True)
    skipped = [{"metric": "sharpe", **s} for s in ra_skip] + [
        {"metric": "cvar", "level": float(cvar_level), **s} for s in tail_skip
    ]
    return {
        "status": "ok" if (ra_legs or tail_legs) else "skipped",
        "reason": None if (ra_legs or tail_legs) else "no H2 contrast has >= 2 shared test seeds",
        "margin": margin,
        "tail_margin_fraction": frac,
        "units": "test-statistic units (per-seed Sharpe / CVaR), NOT validation-DSR",
        "ra": ra_legs,
        "tail": {"level": float(cvar_level), "legs": tail_legs},
        "skipped": skipped,
    }


def h2_tost_dsr(
    records: list[dict[str, Any]],
    *,
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    sesoi_dsr: float | None = None,
    track_length: int | None = None,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The bankable-null TOST in **validation-DSR units** (the SESOI's units; outputs/CAMPAIGN_power.md, T2.5).

    Companion to :func:`h2_tost` (which bounds the difference in the test-statistic's OWN per-seed
    Sharpe/CVaR units, ±0.05). The FROZEN SESOI / equivalence margin (PREREGISTRATION §10 R12) is specified
    in **validation-DSR** units — the SELECTION metric — and ``outputs/CAMPAIGN_power.md`` is explicit that a
    campaign non-rejection only licenses a *practical-equivalence* claim if the TOST CI lies inside ±0.05 in
    **DSR units** (otherwise the result is INCONCLUSIVE, not equivalence). The Sharpe-units leg alone is at
    most inconclusive for that claim. This function computes the missing DSR-units leg so a non-rejection
    actually evaluates the equivalence the power doc requires.

    Method (report-only, DISJOINT). For each H2 RA contrast (distributional vs {scalar, placebo,
    scalar_cvar5}) it takes each arm's PER-SEED annualised Sharpe (:func:`_seed_scores` + ``sharpe_ratio`` —
    the SAME per-seed scores the headline difference test uses, carrying the across-seed variance) and maps
    each into a CONSERVATIVE (upper-bound) validation-DSR shift via
    ``power_analysis.sharpe_mde_to_dsr`` — the documented linear ceiling
    ``ΔDSR_max = φ(0)·√(T−1)/√252 · ΔSR_ann`` (T2.5). Because the map is linear and positive, the per-seed
    DSR-unit scores are an order-preserving rescaling of the Sharpe scores, so the PAIRED bootstrap CI for
    ``mean(a) − mean(b)`` is computed honestly in DSR units; equivalence holds iff that 90% CI lies inside
    ``±sesoi_dsr`` (``power_analysis.tost_equivalence`` with ``paired=True`` — CRN seed pairing, matching the
    headline difference test's across-seed covariance; Lakens 2017). The ceiling is the HONEST direction:
    it OVER-states ΔDSR (φ(z)≤φ(0), D≥1 off-the-money / fat-tailed), so a DSR-equivalence verdict here is a
    fortiori true under the exact map; a NON-equivalent verdict is reported as INCONCLUSIVE (the conservative
    map could not certify a bound that small), never as evidence of a difference.

    Tail (CVaR) has NO validation-DSR analogue (the DSR is a Sharpe/PSR selection statistic), so this leg is
    RA-only by construction; the CVaR equivalence stays in :func:`h2_tost` (its own units). ``sesoi_dsr``
    defaults to the FROZEN ``inference.equivalence_margin`` (0.05) read from config; ``track_length`` to the
    power module's ``VALIDATION_TRACK_LENGTH``.

    DISJOINT KEY: writes ``out["h2_tost_dsr"]`` with NO ``arm_a/arm_b/metric/level`` family-tuple keys, so
    the frozen m=6 union + :func:`assert_realized_family_matches_frozen` are untouched (the equivalence
    secondary already declared in ``config/preregistration.yaml: inference.secondary_families`` /
    ``h2_tost_equivalence``; this adds the DSR-units companion the doc specifies, never a gate).

    Returns
    -------
    dict
        ``{"status", "units": "validation-DSR (conservative ceiling)", "margin", "sharpe_to_dsr_factor",
        "track_length", "ra": [ {contrast, estimate, ci_low, ci_high, equivalent, inconclusive, n_seeds,
        estimate_sharpe}, ... ], "skipped": [...]}``; ``status="skipped"`` only if NO RA contrast had >= 2
        shared seeds.
    """
    from src.inference.bootstrap import sharpe_ratio

    sharpe_mde_to_dsr = _power_analysis().sharpe_mde_to_dsr
    tost_equivalence = _power_analysis().tost_equivalence
    if track_length is None:
        track_length = int(_power_analysis().VALIDATION_TRACK_LENGTH)
    if sesoi_dsr is None:
        sesoi_dsr = _frozen_equiv_margin()
    margin = float(sesoi_dsr)
    if rng is None:
        rng = np.random.default_rng(0)

    # The Sharpe->DSR map is linear: ΔDSR = k·ΔSR_ann with k = φ(0)·√(T−1)/√252. Recover k from a unit MDE
    # (sharpe_mde_to_dsr(1.0)) so each per-seed Sharpe score s_i -> k·s_i lands in validation-DSR units.
    k = float(sharpe_mde_to_dsr(1.0, track_length=int(track_length)))

    scores = {arm: _seed_scores(records, arm, sharpe_ratio) for arm in {a for pair in contrasts for a in pair}}
    ra_legs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for arm_a, arm_b in contrasts:
        sa, sb = scores.get(arm_a, {}), scores.get(arm_b, {})
        common = sorted(set(sa) & set(sb))
        if len(common) < 2:
            skipped.append({"contrast": f"{arm_a}>{arm_b}", "reason": "< 2 shared seeds"})
            continue
        a_dsr = np.array([sa[s] for s in common], dtype=float) * k  # per-seed Sharpe -> per-seed DSR shift
        b_dsr = np.array([sb[s] for s in common], dtype=float) * k
        # PAIRED (CRN seed pairing) to match the headline paired difference test — a_dsr[i] and b_dsr[i] are
        # the SAME training seed (P2 reconciliation, 2026-07-01); an independent resample here would discard
        # the across-seed covariance the headline carries and mis-state the equivalence CI width.
        res = tost_equivalence(
            a_dsr, b_dsr, margin, n_boot=n_boot, paired=True,
            rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
        )
        ra_legs.append({
            "contrast": f"{arm_a}>{arm_b}",
            "estimate": float(res.estimate),                 # DSR units
            "ci_low": float(res.ci_low),
            "ci_high": float(res.ci_high),
            "equivalent": bool(res.equivalent),
            # A NON-equivalent verdict under the CONSERVATIVE ceiling = INCONCLUSIVE (not a difference).
            "inconclusive": (not bool(res.equivalent)),
            "n_seeds": int(len(common)),
            "estimate_sharpe": float(np.mean([sa[s] for s in common]) - np.mean([sb[s] for s in common])),
        })
    return {
        "status": "ok" if ra_legs else "skipped",
        "reason": None if ra_legs else "no H2 RA contrast has >= 2 shared test seeds",
        "margin": margin,
        "units": "validation-DSR (conservative upper-bound ceiling, T2.5)",
        "sharpe_to_dsr_factor": k,
        "track_length": int(track_length),
        "ra": ra_legs,
        "skipped": skipped,
    }


def _arm_pooled_val_returns(records: list[dict[str, Any]], arm: str) -> np.ndarray | None:
    """Pool one arm's per-candidate VALIDATION return vectors into a single concatenated series, or ``None``.

    The (VaR, ES) tail forecast a comparative ES backtest scores is estimated on an arm's realized VALIDATION
    returns (fully archived for every search candidate). Concatenating them gives a larger, more stable tail
    sample than any single candidate. Returns ``None`` when the arm has no usable validation record.
    """
    vecs = [v for r in records if r.get("arm") == arm and (v := _val_returns(r)) is not None]
    if not vecs:
        return None
    return np.concatenate(vecs)


def _arm_median_tail_seed_test_returns(
    records: list[dict[str, Any]], arm: str, alpha: float
) -> tuple[np.ndarray | None, int | None]:
    """One arm's SINGLE realized TEST path whose empirical CVaR_alpha is the MEDIAN across seed paths.

    WHY a single genuine path and not the seed-average: a genuine single realized path preserves the
    tail structure that seed-averaging destroys — averaging ~30 seed paths shrinks the tail, and
    :func:`_arm_test_returns`'s own docstring forbids the averaged series driving a significance test
    (on it an ES backtest's VaR hit indicator rarely fires). The MEDIAN-tail seed is the representative
    path: each per-(record) test vector is kept WHOLE (one vector per record, with its ``seed``), its
    empirical ``cvar(v, alpha)`` is computed (:func:`src.inference.bootstrap.cvar`), and the path whose
    CVaR sits at the median across paths is returned — even count -> the LOWER of the two middle
    values; CVaR ties -> the smallest seed id (records lacking a seed id order last) — fully
    deterministic. Returns ``(path, seed)``, or ``(None, None)`` when the arm has no usable test record.
    """
    from src.inference.bootstrap import cvar

    pairs: list[tuple[int | None, np.ndarray]] = []
    for r in records:
        if r.get("arm") != arm:
            continue
        v = _test_returns(r)
        if v is None:
            continue
        try:
            seed = int(r["seed"]) if r.get("seed") is not None else None  # the _seed_scores coercion
        except (TypeError, ValueError):
            seed = None
        pairs.append((seed, v))
    if not pairs:
        return None, None

    def _seed_key(seed: int | None) -> tuple[int, int]:
        # None-last total order, so a record lacking a seed id still breaks a CVaR tie deterministically.
        return (1, 0) if seed is None else (0, int(seed))

    # Explicit sort key ONLY (never compare the raw triples — the ndarray member has no total order).
    scored = sorted(
        ((float(cvar(v, float(alpha))), seed, v) for seed, v in pairs),
        key=lambda t: (t[0], _seed_key(t[1])),
    )
    median_cv = scored[(len(scored) - 1) // 2][0]  # even count -> the LOWER of the two middle CVaRs
    _, seed, path = min((t for t in scored if t[0] == median_cv), key=lambda t: _seed_key(t[1]))
    return path, seed


def comparative_es_backtest_report(
    records: list[dict[str, Any]],
    *,
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    cvar_level: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Report-only FZ0 + two-sided Diebold-Mariano equal-accuracy ES backtest CORROBORATING H2-Tail (DEEP_H2; CH4 §4.7).

    CH4 §4.7 CLAIMS a (VaR, ES) Expected-Shortfall scoring comparison corroborates the H2-Tail CVaR result, but
    the analysis never invoked it — this wires it. For each H2 contrast (distributional vs {scalar, placebo,
    scalar_cvar5}) it runs the two-sided Diebold-Mariano equal-accuracy test on the FZ0 score differential
    (``src.inference.es_backtest.comparative_es_backtest``) — the Fissler-Ziegel jointly-consistent scoring with
    the Nolde-Ziegel (2017) DM apparatus, run TWO-SIDED, NOT their one-sided comparative/dominance form (matching
    CH4 §4.7):
    on a COMMON realized series — the distributional arm's MEDIAN-TAIL-SEED single realized TEST path
    (:func:`_arm_median_tail_seed_test_returns`; a genuine single path preserves the tail structure that
    seed-averaging destroys, and :func:`_arm_test_returns`'s docstring forbids the averaged series driving a
    significance test) — it scores two tail forecasts under the jointly-elicitable FZ0 loss
    (Fissler-Ziegel 2016) —

      * ``forecast1`` = the (VaR_alpha, ES_alpha) estimated from the DISTRIBUTIONAL arm's pooled validation
        returns (the tail-fed arm's own forecast);
      * ``forecast2`` = the (VaR_alpha, ES_alpha) estimated from the COMPARATOR arm's pooled validation returns.

    ``mean_score_diff < 0`` (``better == "model1"``) means the distributional arm's tail forecast scores
    strictly better on the realized tail — the direction that CORROBORATES H2-Tail. This is a forecast-scoring
    backtest on ONE realized series (the use the module supports), NOT a two-sample CVaR comparison
    of two different realizations (which stays in ``h2_conjunction``'s ``cvar_difference`` legs). Report-only,
    NEVER gates: it writes a DISJOINT block with NO ``arm_a/arm_b/metric/level`` family-tuple keys.

    Returns ``{"status", "level", "realized_series": {kind, seed, alpha}, "legs": [{contrast,
    mean_score_diff, pvalue, pvalue_dm_hln, better, forecast_calibration_favours_dist}, ...], "skipped": [...]}``;
    ``status="skipped"`` when NO contrast had a usable common realized series + two forecasts.
    """
    from src.inference.es_backtest import comparative_es_backtest, var_es_estimates

    if rng is None:
        rng = np.random.default_rng(0)
    alpha = float(cvar_level)
    # ONE genuine realized path, NOT the seed-average: the FZ0/DM scoring reads realized tail HITS, and
    # averaging the seed paths shrinks the tail until the hit indicator rarely fires (the seed-average is
    # display-only by its own docstring). The median-tail seed is the representative single realization.
    realized, realized_seed = _arm_median_tail_seed_test_returns(records, "distributional", alpha)
    legs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if realized is None or realized.size < 5:
        return {
            "status": "skipped",
            "level": alpha,
            "reason": "distributional arm has no usable TEST realized series",
            "legs": [],
            "skipped": [],
        }

    dist_val = _arm_pooled_val_returns(records, "distributional")
    f1 = var_es_estimates(dist_val, alpha) if dist_val is not None else (float("nan"), float("nan"))
    for arm_a, arm_b in contrasts:
        comp_val = _arm_pooled_val_returns(records, arm_b)
        f2 = var_es_estimates(comp_val, alpha) if comp_val is not None else (float("nan"), float("nan"))
        # need finite, strictly-negative ES for both forecasts (fz0 requires es < 0)
        if not (np.isfinite(f1[1]) and np.isfinite(f2[1]) and f1[1] < 0.0 and f2[1] < 0.0):
            skipped.append({"contrast": f"{arm_a}>{arm_b}", "reason": "missing / non-negative-ES forecast"})
            continue
        res = comparative_es_backtest(
            realized, f1, f2, alpha=alpha, n_boot=n_boot,
            rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
        )
        legs.append({
            "contrast": f"{arm_a}>{arm_b}",
            "mean_score_diff": float(res["mean_score_diff"]),
            "pvalue": float(res["pvalue"]),
            "pvalue_dm_hln": float(res["pvalue_dm_hln"]),
            "better": str(res["better"]),
            # NOTE: RENAMED 2026-07-26 (deep review row 35): this was `corroborates_h2_tail`, which it is NOT.
            # Both forecasts are FZ0-scored against the SAME series -- the distributional arm's OWN test
            # path -- while forecast 1 is fitted on that same arm's validation returns, so a strictly
            # consistent rule favours it near-automatically. This flags which forecast CALIBRATES that
            # path better (self-prediction across val->test); it says NOTHING about the DIRECTION of the
            # tail contrast, which is what H2-Tail asserts and what the 3-leg cvar_difference_test IUT
            # decides. es_backtest.py's own scope note warns against reading it the old way.
            "forecast_calibration_favours_dist": bool(res["better"] == "model1"),
            # B.5.2 (2026-07-06): Hill tail-index of the FZ0 loss differential — flags a
            # heavy-tailed d_t whose DM-HLN companion p should be read with extra caution.
            "loss_diff_tail": res.get("loss_diff_tail"),
        })
    return {
        "status": "ok" if legs else "skipped",
        "level": alpha,
        "reason": None if legs else "no contrast had a usable common realized series + two (VaR,ES) forecasts",
        # Which single realized path was scored (median-tail seed) — auditable, never silent.
        "realized_series": {"kind": "median_tail_seed", "seed": realized_seed, "alpha": alpha},
        "note": (
            "Two-sided equal-accuracy FZ0 forecast-scoring backtest on ONE realized series (the distributional "
            "arm's median-tail-seed TEST path); report-only corroboration of H2-Tail, DISJOINT from the frozen "
            "m=6 family."
        ),
        "legs": legs,
        "skipped": skipped,
    }


def bayesian_null_report_block(
    records: list[dict[str, Any]],
    *,
    contrasts: tuple[tuple[str, str], ...] = H2_CONTRASTS,
    cvar_level: float = 0.05,
    sesoi: float | None = None,
    tail_margin_fraction: float = 0.25,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Report-only Bayesian evidence-for-the-null complement to the frozen TOST (R67; DEEP_H2 §5.4).

    For each H2 contrast and each co-primary metric (Sharpe = H2-RA, CVaR-5% = H2-Tail), reduces the two
    arms' per-seed scores over the shared training seeds to PAIRED per-seed difference scores (the SAME
    object the headline paired IUT consumes) and runs ``src.inference.bayes_null.bayesian_null_report`` with
    the ROPE half-width = the frozen SESOI. This adds a Bayes-factor + posterior-in-ROPE lens on the null
    ALONGSIDE the frequentist TOST — a mature evidence-FOR-equivalence statement, never a gate.

    ROPE UNITS (mirrors :func:`h2_tost`'s P6 band): the raw ±SESOI (0.05) ROPE is in RAW CVaR units and is
    LARGE relative to a daily CVaR magnitude O(0.01–0.06), so on the TAIL legs it can near-trivially
    contain the posterior and over-claim null evidence. Each tail leg therefore ALSO reports a RELATIVE
    ROPE = ``tail_margin_fraction`` (default 25%) × |baseline CVaR| (the comparator arm's IQM CVaR), i.e.
    the SAME scale-appropriate band the TOST reports. BOTH ROPEs are reported; NEITHER gates. Only the
    ROPE-dependent fields (``rope_mass`` / ``hdi_in_rope`` / ``verdict``) move — the BF01 is ROPE-free.

    DISJOINT KEY: writes NO ``arm_a/arm_b/metric/level`` family-tuple keys; a declared secondary. Gracefully
    degrades to ``status="skipped"`` when no contrast has >= 2 shared seeds.

    Returns ``{"status", "sesoi", "tail_margin_fraction", "level", "ra": [...], "tail": {level,
    legs:[...]}, "skipped": [...]}`` where each leg is ``{contrast, verdict, bf01, effect, hdi_in_rope,
    robust_for_null, n}``; tail legs additionally carry ``{baseline_cvar, margin_fraction,
    relative: {rope_mass, hdi_in_rope, verdict} | {status}}``.
    """
    from src.inference.bayes_null import bayesian_null_report
    from src.inference.bootstrap import cvar, iqm, sharpe_ratio

    # ``rng`` is accepted for call-shape symmetry with the sibling report blocks and is NOT used: this
    # estimator is ANALYTIC end to end (the JZS Bayes factor is closed-form and the HDI is taken on the
    # supplied draws — ``src.inference.bayes_null`` contains no sampler), so there is no randomness to
    # steer. Stated explicitly, mirroring ``src.selection.fitness.held_out_fitness``; the twin case in
    # ``model_confidence_set_report`` was NOT benign (its bootstrap is real) and was fixed in #108.
    del rng
    margin = float(sesoi) if sesoi is not None else _frozen_equiv_margin()
    frac = float(tail_margin_fraction)

    def _legs(
        score_fn: Callable[[np.ndarray], float], *, relative: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scores = {arm: _seed_scores(records, arm, score_fn) for arm in {a for pair in contrasts for a in pair}}
        legs: list[dict[str, Any]] = []
        skipped_legs: list[dict[str, Any]] = []
        for arm_a, arm_b in contrasts:
            sa, sb = scores.get(arm_a, {}), scores.get(arm_b, {})
            common = sorted(set(sa) & set(sb))
            if len(common) < 2:
                skipped_legs.append({"contrast": f"{arm_a}>{arm_b}", "reason": "< 2 shared seeds"})
                continue
            diffs = np.array([sa[s] - sb[s] for s in common], dtype=float)  # PAIRED per-seed differences
            res = bayesian_null_report(diffs, margin)
            if res.get("status") != "ok":
                skipped_legs.append({"contrast": f"{arm_a}>{arm_b}", "reason": res.get("status", "insufficient")})
                continue
            leg = {
                "contrast": f"{arm_a}>{arm_b}",
                "verdict": str(res["verdict"]),
                "bf01": float(res["bf01"]),
                "effect": float(res["effect"]),
                "hdi_in_rope": bool(res["hdi_in_rope"]),
                "robust_for_null": bool(res["robust_for_null"]),
                "n": int(res["n"]),
            }
            if relative:
                # RELATIVE ROPE for the TAIL legs (the h2_tost P6 pattern): re-run the SAME report at
                # ROPE = fraction · |comparator arm_b's IQM CVaR|. Same diffs, same prior — only the
                # ROPE-dependent fields differ, so just those three are pulled out. Neither band gates.
                baseline = float(iqm(np.array([sb[s] for s in common], dtype=float)))
                rel_margin = frac * abs(baseline)
                leg["baseline_cvar"] = baseline
                leg["margin_fraction"] = frac
                if np.isfinite(rel_margin) and rel_margin > 0.0:
                    rel = bayesian_null_report(diffs, rel_margin)
                    if rel.get("status") == "ok":
                        leg["relative"] = {
                            "rope_mass": float(rel["posterior"]["rope_mass"]),
                            "hdi_in_rope": bool(rel["hdi_in_rope"]),
                            "verdict": str(rel["verdict"]),
                        }
                    else:
                        leg["relative"] = {"status": str(rel.get("status", "insufficient"))}
                else:
                    leg["relative"] = {"status": "degenerate baseline CVaR (relative margin not positive/finite)"}
            legs.append(leg)
        return legs, skipped_legs

    ra_legs, ra_skip = _legs(sharpe_ratio)
    tail_legs, tail_skip = _legs(lambda v: cvar(v, float(cvar_level)), relative=True)
    skipped = [{"metric": "sharpe", **s} for s in ra_skip] + [
        {"metric": "cvar", "level": float(cvar_level), **s} for s in tail_skip
    ]
    return {
        "status": "ok" if (ra_legs or tail_legs) else "skipped",
        "reason": None if (ra_legs or tail_legs) else "no H2 contrast has >= 2 shared test seeds",
        "sesoi": margin,
        "tail_margin_fraction": frac,
        "level": float(cvar_level),
        "ra": ra_legs,
        "tail": {"level": float(cvar_level), "legs": tail_legs},
        "skipped": skipped,
    }


def model_confidence_set_report(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = ARMS,
    cvar_level: float = 0.05,
    size: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """Report-only Model Confidence Set over the arms on their per-seed Sharpe + CVaR (Hansen et al. 2011).

    The multiplicity-honest "which arms are statistically INDISTINGUISHABLE" statement: over the arms that
    share a common set of training seeds, build ``{arm: per-seed score}`` and pass it to
    ``src.inference.model_confidence_set.model_confidence_set``. Under the predicted null the set contains
    (almost) all arms — the honest counterpart to the pairwise IUTs. Computed for BOTH co-primary metrics.

    Requires the optional ``arch`` dependency (MCS lives in ``arch.bootstrap``); when it is absent the block
    degrades to ``status="error"`` (via the caller's guard) rather than breaking the headline. DISJOINT block
    (no family-tuple keys). ``status="skipped"`` when < 2 arms share >= 2 common seeds.

    ``seed`` pins the MCS bootstrap (R69 registers the level and the seed as PINNED; ``0`` IS that
    registered pin, so the default reproduces the campaign number exactly). It is a REAL knob —
    changing it re-resamples — which is the point: a bootstrap-sensitivity re-run must actually move
    the p-values. This parameter was an ``rng: np.random.Generator`` until 2026-07-27 (deep review
    #108) that the body accepted, defaulted and then NEVER USED: the MCS ran at a hardcoded
    ``seed=0`` whatever the caller passed, so re-running with a different generator returned
    BIT-IDENTICAL p-values (measured) and would have invited the false conclusion that this
    1,000-rep bootstrap is insensitive to resampling. Contrast
    ``src.selection.fitness.held_out_fitness``, which also ignores its ``rng`` but says so in its
    docstring AND is genuinely deterministic — the honest form of the same shape.

    Returns ``{"status", "sharpe": {...mcs...}, "tail": {level, ...mcs...}}``.
    """
    from src.inference.bootstrap import cvar, sharpe_ratio
    from src.inference.model_confidence_set import model_confidence_set

    def _mcs(score_fn: Callable[[np.ndarray], float]) -> dict[str, Any]:
        per_arm = {arm: _seed_scores(records, arm, score_fn) for arm in arms}
        per_arm = {arm: d for arm, d in per_arm.items() if d}  # keep only arms with >= 1 seed
        if len(per_arm) < 2:
            return {"status": "skipped", "reason": "< 2 arms with test records"}
        common = sorted(set.intersection(*[set(d) for d in per_arm.values()]))
        if len(common) < 2:
            return {"status": "skipped", "reason": "< 2 shared seeds across the arms"}
        arm_scores = {arm: np.array([per_arm[arm][s] for s in common], dtype=float) for arm in per_arm}
        return model_confidence_set(arm_scores, size=size, seed=int(seed))

    sharpe_mcs = _mcs(sharpe_ratio)
    tail_mcs = _mcs(lambda v: cvar(v, float(cvar_level)))
    status = "ok" if (sharpe_mcs.get("status") == "ok" or tail_mcs.get("status") == "ok") else "skipped"
    return {
        "status": status,
        "size": float(size),
        "sharpe": sharpe_mcs,
        "tail": {"level": float(cvar_level), **tail_mcs},
    }


def h2_structure_control(
    records: list[dict[str, Any]],
    *,
    cvar_level: float = 0.05,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The PRE-REGISTERED structure-vs-content control (R32): distributional vs ``placebo_shuffled``.

    ``placebo_shuffled`` feeds a feedback block byte-structurally IDENTICAL to the distributional block
    (same header, intro, the six labels, the CVaR-1% high-variance annotation) but with the six real tail
    VALUES candidate-seeded-DERANGED across their labels — matching the FORMAT and the MARGINAL set of
    numbers while breaking the coherent label->value mapping (the tail SHAPE). A one-sided
    ``distributional > placebo_shuffled`` rejection on BOTH co-primary metrics (Sharpe AND CVaR-5%) means
    the distributional advantage reflects the COHERENT tail information (content), not the mere presence of
    a plausible-looking numeric table (format) — the Gupta-Hartford format-vs-content threat (the
    DEEP_SYSTEM red-team's HIGH-severity item). Reported alongside H2, NEVER a gate.

    DISJOINT KEY: writes ``out["h2_structure"]`` with NO ``arm_a/arm_b/metric/level`` family-tuple keys, so
    the frozen m=6 union + :func:`assert_realized_family_matches_frozen` are untouched (a declared secondary,
    ``config/preregistration.yaml: inference.secondary_families.h2_structure``).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    fam = collect_family_pvalues(
        records,
        contrasts=(("distributional", "placebo_shuffled"),),
        cvar_levels=(float(cvar_level),),
        n_boot=n_boot,
        alpha_one_sided=alpha,
        rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
    )

    def _leg(metric: str, level: float | None) -> dict[str, Any] | None:
        for t in fam["tests"]:
            if t["metric"] == metric and (level is None or t.get("level") == level):
                return {
                    "effect": float(t["effect"]),
                    "pvalue_one_sided": float(t["pvalue_one_sided"]),
                    "reject_one_sided": bool(t["reject_one_sided"]),
                    "direction_ok": bool(t["direction_ok"]),
                }
        return None

    sharpe = _leg("sharpe", None)
    tail = _leg("cvar", float(cvar_level))
    if sharpe is None and tail is None:
        return {
            "status": "skipped",
            "reason": "placebo_shuffled has no test record sharing >= 2 seeds with distributional",
        }
    content_over_format = bool(
        sharpe and tail and sharpe["reject_one_sided"] and tail["reject_one_sided"]
    )
    return {
        "status": "ok",
        "contrast": "distributional>placebo_shuffled",
        "cvar_level": float(cvar_level),
        "sharpe": sharpe,
        "cvar": tail,
        "content_over_format": content_over_format,
        "interpretation": (
            "distributional beats the format+marginal-matched structure control on BOTH Sharpe and "
            "CVaR-5%: the advantage reflects the coherent tail SHAPE (content), not table FORMAT"
            if content_over_format
            else "distributional does NOT beat the structure control on both metrics: report as a bound "
            "on the content claim (a format/anchoring component cannot be ruled out)"
        ),
    }


def dsr_effective_n(
    records: list[dict[str, Any]],
    *,
    winner_arm: str = "distributional",
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """DSR under raw N vs an effective-N proxy for correlated sequential-reflective trials (DEEP_STATS A1).

    The headline winner DSR (:func:`winner_dsr`) deflates by ``n_trials = N`` (the full per-arm candidate
    count). But the campaign's reflect-on-best search makes those N candidates POSITIVELY CORRELATED and
    sequential, not the i.i.d. draws the expected-max-Sharpe formula assumes (Bailey & López de Prado 2014;
    the paper's own independence caveat). This report-only sensitivity recomputes the ``winner_arm``
    winner's DSR at an EFFECTIVE N proxy

        ``N_eff = N / (1 + (N - 1) * rho_bar)``   (clamped to ``[1, N]``)

    where ``rho_bar`` is the mean off-diagonal pairwise correlation of the arm's per-candidate VALIDATION
    return vectors (the same columns :func:`build_perf_matrix` stacks). A positive ``rho_bar`` shrinks
    ``N_eff`` below ``N``, which LOWERS the expected-max benchmark and RAISES the DSR — so the naïve N is
    the CONSERVATIVE (larger-deflation) choice and any N error here is benign-direction (DEEP_STATS A8: a
    floor/H1 PASS at naïve N is robust). REPORT-ONLY: never re-selects, never changes the frozen gate.

    NB this is an APPROXIMATE proxy (the López de Prado 2018 ONC clustering count is the canonical
    effective-N; ``N / (1 + (N-1)·rho_bar)`` is the cheaper mean-correlation surrogate, flagged as such).

    Returns
    -------
    dict
        ``{"status", "arm", "n_trials", "rho_bar", "n_eff", "dsr_raw_n", "dsr_eff_n", "var_sr",
        "winner_id", "note"}`` or ``{"status": "skipped", "reason": ...}`` when the arm has < 2 candidates
        with usable validation vectors (no cross-trial dispersion / correlation).
    """
    from src.inference.deflated_sharpe import _sample_moments, deflated_sharpe_ratio

    arm_records = [r for r in records if r.get("arm") == winner_arm and _is_search_candidate(r)]
    cands = [(r, vec) for r in arm_records if (vec := _val_returns(r)) is not None]
    n_cfg = len(cands)
    if n_cfg < 2:
        return {
            "status": "skipped",
            "reason": f"need >= 2 candidates with validation vectors for a cross-trial correlation; got {n_cfg}",
            "arm": winner_arm,
            "n_candidates": n_cfg,
        }

    # Cross-trial Sharpe dispersion (canonical DSR var_sr), exactly as winner_dsr builds it (per-period
    # _sample_moments, ddof=1) so the two tables' DSR inputs are consistent.
    sharpes = np.asarray([_sample_moments(vec)[0] for _, vec in cands], dtype=float)
    var_sr = float(np.var(sharpes, ddof=1))
    # Pick the winner exactly as the canonical winner_dsr does (2026-07-05): sort by
    # (generation, candidate_id) for a deterministic tie-break, and map a non-finite val_fitness to
    # -inf so a NaN-fitness candidate can never win the scan (the old unsorted `max` could name a
    # different winner than the canonical DSR table on identical records).
    def _fit(rec: dict) -> float:
        v = (rec.get("metrics") or {}).get("val_fitness", float("-inf"))
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        return fv if np.isfinite(fv) else float("-inf")

    cands_sorted = sorted(
        cands,
        key=lambda rv: (int(rv[0].get("generation", 0) or 0),
                        str(rv[0].get("candidate_id", rv[0].get("run_id", "")))),
    )
    winner_rec, winner_vec = max(cands_sorted, key=lambda rv: _fit(rv[0]))
    n_trials = len(arm_records)  # full per-arm candidate count = the naïve multiplicity (#32, winner_dsr)

    # Mean off-diagonal pairwise correlation of the per-candidate validation return vectors (aligned to the
    # common leading length, as build_perf_matrix does). A degenerate (constant) column is dropped from the
    # correlation (np.corrcoef would NaN it); rho_bar falls back to 0 (=> N_eff=N) if < 2 usable columns.
    t_min = min(int(vec.size) for _, vec in cands)
    cols = [np.asarray(vec, dtype=float)[:t_min] for _, vec in cands]
    usable = [c for c in cols if np.std(c) > 1e-12 and np.all(np.isfinite(c))]
    rho_bar = 0.0
    if len(usable) >= 2:
        corr = np.corrcoef(np.column_stack(usable), rowvar=False)
        m = corr.shape[0]
        off = corr[~np.eye(m, dtype=bool)]
        off = off[np.isfinite(off)]
        rho_bar = float(off.mean()) if off.size else 0.0

    denom = 1.0 + (n_trials - 1) * rho_bar
    n_eff_raw = n_trials / denom if denom > 0.0 else float(n_trials)
    n_eff = int(round(min(float(n_trials), max(1.0, n_eff_raw))))

    dsr_raw = float(deflated_sharpe_ratio(winner_vec, n_trials, var_sr=var_sr, periods_per_year=periods_per_year))
    dsr_eff = float(deflated_sharpe_ratio(winner_vec, n_eff, var_sr=var_sr, periods_per_year=periods_per_year))
    return {
        "status": "ok",
        "arm": winner_arm,
        "n_trials": int(n_trials),
        "rho_bar": rho_bar,
        "n_eff": int(n_eff),
        "n_eff_raw": float(n_eff_raw),
        "dsr_raw_n": dsr_raw,
        "dsr_eff_n": dsr_eff,
        "var_sr": var_sr,
        "winner_id": str(winner_rec.get("candidate_id", winner_rec.get("run_id", "?"))),
        "note": (
            "N_eff = N/(1+(N-1)·rho_bar) is the mean-correlation surrogate for the López de Prado 2018 ONC "
            "effective-trial count; rho_bar>0 => N_eff<N => higher DSR, so naïve N is the conservative "
            "(larger-deflation) choice and any N error is benign-direction (DEEP_STATS A1/A8)."
        ),
    }


def cross_hypothesis_multiplicity(
    *,
    h1: dict[str, Any] | None,
    h2: dict[str, Any] | None,
    h3: dict[str, Any] | None,
    h4: dict[str, Any] | None,
    n_hypotheses: int = 4,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Report-only Bonferroni-across-4 SENSITIVITY for the H1–H4 headline decisions (DEEP_STATS A4).

    The per-hypothesis families stay PRIMARY (H2's frozen m=6 conjunction; H1/H3/H4 their own declared
    families). This is the explicitly-stated cross-hypothesis stance (DEEP_STATS C4): H1–H4 are distinct
    pre-registered questions tested in separate declared families, so NO global correction is imposed — but
    a reader preferring a programme-wide Bonferroni would multiply each per-hypothesis hurdle by
    ``n_hypotheses`` (≈4). This surfaces that stricter hurdle for each hypothesis's headline p so the
    examiner sees it was CONSIDERED, without changing any primary decision.

    For each hypothesis a single representative HEADLINE p is extracted (the leg/test most relevant to its
    decision) and compared against the Bonferroni level ``alpha / n_hypotheses``:
      * H1 — the beat-the-human-canon INTERSECTION-UNION TEST: headline p = ``h1_beat_human.iut.
        iut_pvalue`` (already the MAX one-sided leg p over the 11-name canon), gated on
        ``all_baselines_present`` exactly as ``validity_tier``'s N6_h1 gates it.
        NOTE: THIS LINE PREVIOUSLY READ "descriptive panel (no inferential p)". That was true until the
        2026-07-26 upgrade made H1 confirmatory, and false — but reassuring — for every reader after
        it, which is why the stale EXTRACTION beneath it survived unexamined until 2026-08-01
        (§100.34). Corrected with the code, not after it.
      * H2 — the WORST (max) one-sided p over the two co-primary IUTs' legs (the conjunction's binding leg);
        an IUT rejects only if its MAX leg p clears alpha, so the conjunction p IS that max.
      * H3 — the one-sided difference p (if computed).
      * H4 — the WORST (max) one-sided p over {H4a, H4b} (both must hold for "LLM beats search").

    Returns
    -------
    dict
        ``{"alpha", "n_hypotheses", "bonferroni_alpha", "rows": [ {hypothesis, headline_p, decision_primary,
        survives_bonferroni, note}, ... ]}``.
    """
    bonf = float(alpha) / float(n_hypotheses) if n_hypotheses else float(alpha)
    rows: list[dict[str, Any]] = []

    # H1 — the beat-the-human-canon IUT (2026-07-26 upgrade), NOT the retired descriptive panel.
    #
    # 2026-08-01 FIX (record §100.34) — STALE CONSUMER. This row read
    # ``h1["beats_best_baseline_dsr"]`` and hardcoded ``headline_p: None`` /
    # ``"descriptive panel, no inferential p — Bonferroni n/a"``. That was correct when H1 WAS a
    # descriptive panel. On 2026-07-26 H1 became a confirmatory INTERSECTION-UNION TEST over the full
    # 11-name hand-reward canon (``h1_beat_human``), with a real one-sided ``iut_pvalue`` — and the
    # CALLER was updated to pass ``h1=out.get("h1_beat_human")`` while THIS extraction was not.
    #
    # THE CONSEQUENCE WAS NOT COSMETIC. ``src/inference/validity_tier.py`` gives N6_h1 an INITIAL GRAPH
    # WEIGHT OF 0.0, so under the design's own PRE-REGISTERED PREDICTION (H2 null on both co-primaries)
    # nothing propagates and H1 is tested at local alpha EXACTLY 0.0 (§100.33). With this row also
    # reporting "no inferential p", **H1 WAS DECIDABLE NOWHERE** — its IUT p-value was computed on every
    # run and consumed by no rule that could act on it. A headline claim of the dissertation had no
    # decision path at all under the outcome the design predicts.
    #
    # NOTE: THIS IS NOT A DESIGN CHANGE, AND THE DISTINCTION IS THE WHOLE POINT. The R31 rule is untouched:
    # each hypothesis's headline p against alpha/n_hypotheses. What changes is WHICH QUANTITY IS H1's
    # headline p — and that was already decided and registered by the 2026-07-26 upgrade (whose endpoint
    # correction, DSR -> annualised Sharpe, is itself recorded in the frozen config's N6_h1 note). The
    # two consumers of ``h1_beat_human`` now AGREE; before, one read the IUT and the other read a
    # retired field, which is executed-vs-registered drift by any definition.
    #
    # ``all_baselines_present`` is honoured EXACTLY as validity_tier's N6_h1 does — dominance is
    # certifiable only when every canon member is testable, so an under-seeded member yields "not
    # certifiable" rather than a cheaper claim. Reading it the same way in both places is deliberate:
    # two consumers of one quantity disagreeing about its gate is how this defect arose.
    h1_status = (h1 or {}).get("status")
    _h1_iut = (h1 or {}).get("iut") or {}
    _h1_p = _h1_iut.get("iut_pvalue")
    _h1_all_present = bool(_h1_iut.get("all_baselines_present"))
    _h1_usable = isinstance(_h1_p, (int, float)) and np.isfinite(_h1_p) and _h1_all_present
    rows.append({
        "hypothesis": "H1",
        "headline_p": float(_h1_p) if _h1_usable else None,
        "decision_primary": (
            f"dominates_canon={_h1_iut.get('dominates_canon')} "
            f"(beaten {_h1_iut.get('n_significantly_beaten')}/{_h1_iut.get('n_baselines')})"
            if h1_status == "ok" else f"({h1_status or 'not run'})"
        ),
        "survives_bonferroni": (float(_h1_p) <= bonf) if _h1_usable else None,
        "note": (
            "beat-the-human-canon IUT (berger1982iut): headline p = MAX one-sided leg p over the "
            "11-name canon; rejecting means the LLM reward beats EVERY hand-crafted member"
            if _h1_usable else
            f"IUT p not usable for Bonferroni (all_baselines_present={_h1_all_present}, p={_h1_p!r}) "
            f"— dominance is certifiable only when every canon member has >= 2 shared test seeds"
        ),
    })

    # H2 — the conjunction's binding leg = the MAX one-sided p over BOTH IUTs' legs.
    def _max_leg_p(h2d: dict[str, Any]) -> float | None:
        ps: list[float] = []
        for key in ("legs", "tail_legs"):
            for leg in (h2d or {}).get(key, []) or []:
                p = leg.get("pvalue_one_sided")
                if isinstance(p, (int, float)):
                    ps.append(float(p))
        return max(ps) if ps else None

    h2_p = _max_leg_p(h2 or {})
    rows.append({
        "hypothesis": "H2",
        "headline_p": h2_p,
        "decision_primary": (h2 or {}).get("verdict", "n/a"),
        "survives_bonferroni": (bool(h2_p <= bonf) if h2_p is not None else None),
        "note": "max one-sided p over the two co-primary IUT leg sets (the conjunction's binding leg)",
    })

    # H3 — the one-sided difference p (if H3 ran).
    h3_p = ((h3 or {}).get("difference") or {}).get("pvalue_one_sided") if (h3 or {}).get("status") == "ok" else None
    rows.append({
        "hypothesis": "H3",
        "headline_p": (float(h3_p) if isinstance(h3_p, (int, float)) else None),
        "decision_primary": (h3 or {}).get("verdict", f"({(h3 or {}).get('status', 'not run')})"),
        "survives_bonferroni": (bool(h3_p <= bonf) if isinstance(h3_p, (int, float)) else None),
        "note": "one-sided iterative>single-shot difference p (if the single-shot archive was present)",
    })

    # H4 — the MAX one-sided p over EVERY leg in H4_CONTRASTS (all must hold for "LLM beats
    # search"): the intersection-union test against the four-optimiser portfolio
    # {random_search, bayes_opt, cma_es, tpe}, i.e. beat-the-BEST-black-box-optimiser.
    # NOTE: 2026-08-01: this comment said "over {H4a, H4b}" — two legs — and had been stale since the
    # 2->4 optimiser upgrade. The CODE below was already correct (it maxes over whatever
    # `tests` holds); only the comment was wrong, which is the more dangerous artefact of the two
    # because it stops the next reader from checking. Reported by WRITEUP (M158).
    h4_ps = [
        float(t["pvalue_one_sided"]) for t in (h4 or {}).get("tests", []) or []
        if isinstance(t.get("pvalue_one_sided"), (int, float))
    ]
    h4_p = max(h4_ps) if h4_ps else None
    rows.append({
        "hypothesis": "H4",
        "headline_p": h4_p,
        "decision_primary": (
            f"all_supported={(h4 or {}).get('all_supported')}" if (h4 or {}).get("status") == "ok"
            else f"({(h4 or {}).get('status', 'not run')})"
        ),
        "survives_bonferroni": (bool(h4_p <= bonf) if h4_p is not None else None),
        "note": "max one-sided p over {H4a, H4b}",
    })

    return {
        "alpha": float(alpha),
        "n_hypotheses": int(n_hypotheses),
        "bonferroni_alpha": bonf,
        "rows": rows,
        "stance": (
            "Per-hypothesis families are PRIMARY (pre-registered separate estimands); this programme-wide "
            "Bonferroni-across-4 is a REPORTED sensitivity only (DEEP_STATS A4/C4), not the headline gate."
        ),
    }


def mechanism_multiplicity(
    *,
    responsiveness: dict[str, Any] | None = None,
    mediation: dict[str, Any] | None = None,
    named_vs_blinded_structural: dict[str, Any] | None = None,
    legible_format_responsiveness: dict[str, Any] | None = None,
    regime_stratified: dict[str, Any] | None = None,
    alpha: float = 0.05,
    q: float = 0.05,
) -> dict[str, Any]:
    """Report-only Bonferroni / BH SENSITIVITY across the report-only mechanism (SQ1/SQ2/SQ3) legs.

    Forking-paths insurance for the mechanism kernel, mirroring :func:`cross_hypothesis_multiplicity` (the
    cross-HYPOTHESIS Bonferroni-across-4) but ACROSS the report-only mechanism legs: SQ1 responsiveness,
    SQ2 mediation, the AST label-permutation, the structural McNemar, the coefficient Mahalanobis-permutation,
    the legible-format differential, and the regime split. A reader worried that surveying several mechanism
    diagnostics inflates the family-wise error can read the STRICTER hurdle here. It is REPORT-ONLY, DISJOINT
    from the frozen m=6 family, and NEVER changes any mechanism leg's own reported decision (each mechanism
    leg stays a descriptive report-only object). It writes NO ``arm_a/arm_b/metric/level`` family-tuple keys.

    Each leg contributes a row. Legs that emit a genuine p-value (the AST permutation ``p_value``, the
    McNemar ``pvalue``, the Mahalanobis ``pvalue``, the named-vs-blinded random-pairing ``p_random_pairing_
    matches``) enter the Bonferroni (level ``alpha / n_p``) and BH (level ``q``) correction over the p-bearing
    set. CI-based legs (responsiveness, mediation, legible-format — decided by a bootstrap CI, no p) and the
    descriptive regime split are LISTED with their own decision but marked ``has_p=False`` (Bonferroni/BH
    n/a), exactly as :func:`cross_hypothesis_multiplicity` lists H1 as "descriptive, no p".

    Returns
    -------
    dict
        ``{"alpha", "q", "n_p_tests", "bonferroni_alpha", "rows": [ {leg, p, has_p, decision,
        survives_bonferroni, reject_bh, note}, ... ], "stance"}``.
    """
    from src.inference.multiple_testing import benjamini_hochberg

    rows: list[dict[str, Any]] = []

    def _ci_row(leg: str, d: dict[str, Any] | None, decides: str, note: str) -> None:
        d = d or {}
        status = d.get("status")
        decision = str(d.get(decides)) if status == "ok" else f"({status or 'not run'})"
        rows.append({
            "leg": leg, "p": None, "has_p": False,
            "decision": decision, "survives_bonferroni": None, "reject_bh": None, "note": note,
        })

    def _p_row(leg: str, d: dict[str, Any] | None, p_key: str, note: str) -> None:
        d = d or {}
        p = d.get(p_key)
        rows.append({
            "leg": leg,
            "p": (float(p) if isinstance(p, (int, float)) else None),
            "has_p": isinstance(p, (int, float)),
            "decision": (str(d.get("status")) if d.get("status") not in (None, "ok") else None),
            "survives_bonferroni": None, "reject_bh": None, "note": note,
        })

    # SQ1 responsiveness (CI-based).
    _ci_row("SQ1 responsiveness", responsiveness, "responsive",
            "Spearman fed→construct-count; decided by bootstrap CI-excludes-0 (no p)")
    # SQ2 mediation (CI-based on the indirect effect).
    _ci_row("SQ2 mediation (a·b)", mediation, "mediated",
            "indirect effect a·b; decided by bootstrap CI-excludes-0 (no p)")
    # SQ3 legible-format differential (CI-based).
    _ci_row("SQ3 legible-format differential", legible_format_responsiveness, "legibility_helps",
            "legible−raw responsiveness; decided by bootstrap CI (no p); often executed=False")
    # Regime split (descriptive).
    _ci_row("Regime split (T3′)", regime_stratified, "status",
            "descriptive per-regime tail/return metrics (no inferential p)")
    # The structural legs need the NAMED authoring pass. TODAY only ``named_vs_blinded_structural`` itself is
    # wired into analyze(), and it returns a FLAT dict whose sole p is ``p_random_pairing_matches`` (the
    # AST-permutation / McNemar / Mahalanobis p-values come from SEPARATE contamination.py functions that a
    # NAMED-pass aggregator would nest under ``structural_ast`` / ``mcnemar`` / ``mahalanobis`` — absent that
    # pass, those rows degrade to has_p=False, never a phantom key). The extraction is written to read the
    # nested layout IF a future aggregator supplies it, and to degrade honestly when it does not.
    nvb = named_vs_blinded_structural or {}
    _p_row("AST label-permutation", nvb.get("structural_ast"), "p_value",
           "within-vs-across structural clustering; label-permutation p (needs the NAMED pass; else n/a)")
    _p_row("Structural McNemar", nvb.get("mcnemar"), "pvalue",
           "per-motif discordant-pair McNemar p (needs the NAMED pass; else n/a)")
    _p_row("Coefficient Mahalanobis-perm", nvb.get("mahalanobis"), "pvalue",
           "centroid Mahalanobis-distance permutation p (needs the NAMED pass; else n/a)")
    _p_row("Named-vs-blinded random-pairing", nvb if nvb.get("status") == "ok" else None,
           "p_random_pairing_matches", "random named→blinded bijection null p (the wired structural leg)")

    # Apply Bonferroni + BH over the p-BEARING legs only.
    p_rows = [r for r in rows if r["has_p"] and r["p"] is not None]
    n_p = len(p_rows)
    bonf = float(alpha) / n_p if n_p else float(alpha)
    if n_p:
        pvals = np.array([r["p"] for r in p_rows], dtype=float)
        bh = benjamini_hochberg(pvals, q=float(q))
        for r, pv, rej in zip(p_rows, pvals, bh):
            r["survives_bonferroni"] = bool(pv <= bonf)
            r["reject_bh"] = bool(rej)

    return {
        "alpha": float(alpha),
        "q": float(q),
        "n_p_tests": int(n_p),
        "bonferroni_alpha": bonf,
        "rows": rows,
        "stance": (
            "Mechanism legs are REPORT-ONLY and DISJOINT from the frozen m=6 family; this Bonferroni/BH-across-"
            "mechanism-legs pass is a forking-paths SENSITIVITY only (mirrors the cross-hypothesis Bonferroni-"
            "across-4), never a gate. CI-based legs (responsiveness, mediation, legible-format) and the "
            "descriptive regime split carry no p and are listed for completeness."
        ),
    }


def evt_consistency_guard(records: list[dict[str, Any]], *, levels: tuple[float, ...] = (0.05, 0.01)) -> dict[str, Any]:
    """Assert/log the fed CVaR-5%/1% used a CONSISTENT EVT estimator across arms (DEEP_H2 §6.3).

    The fed tail block carries ``cvar_05``/``cvar_01`` measured per candidate by
    ``src.feedback.measurement.ReturnDistribution`` (archived in ``metrics['tail_stats']``). At each level
    that estimator routes EVT/GPD vs empirical, with a fallback inside ``_evt_cvar``: when the requested
    ``alpha`` exceeds the tail-exceedance fraction ``F_u`` (``alpha > fu``), the GPD fit is degenerate
    (non-finite beta), or the shape leaves the regular MLE region (xi >= 1 infinite-mean / xi <= -0.5
    non-regular, Smith 1985 — T2.8a), it FALLS BACK to the empirical estimator. If that fallback fires for SOME
    arms' winners but not others, the distributional vs scalar_cvar5 tail comparison would mix EVT and
    empirical CVaR — an estimator inconsistency that DEEP_H2 §6.3 flags.

    This re-derives, per arm winner (the most-recent / highest-fitness candidate carrying ``val_returns``),
    which estimator the fed CVaR at each level WOULD have used (re-fitting ``ReturnDistribution`` on that
    candidate's validation returns as a proxy for the training distribution it was fed on) and reports
    whether the estimator path is consistent across the arms that were FED the tail (distributional,
    scalar_cvar5; ``inspect_rewards._was_fed_tail``). REPORT-ONLY: logs a warning on inconsistency, never
    raises — a null/flag is a finding, not a crash.

    Returns
    -------
    dict
        ``{"status", "levels", "per_arm": {arm: {level: "evt"|"empirical"|"empirical(fallback)"}},
        "consistent": {level: bool}, "fed_arms": [...], "note"}`` or ``{"status": "skipped", "reason": ...}``.
    """
    # Which arms are FED the tail (so an estimator mismatch is meaningful): the distributional + scalar_cvar5
    # designers SEE the CVaR block; scalar/placebo/search do not. ONE source of truth (2026-07-06): the
    # corrected inspect_rewards._was_fed_tail (prompt-first, generation-gated) — the old local closure
    # kept the pre-M13 own-block-first convention and could classify a gen-0 archive as fed.
    _ir = _inspect_rewards()

    def _fed_tail(rec: dict[str, Any]) -> bool:
        return bool(_ir._was_fed_tail(rec))

    def _estimator_path(rd: Any, alpha: float) -> str:
        """Which estimator ReturnDistribution.cvar(alpha) routes to (mirrors measurement.py routing).

        Delegates the EVT-vs-empirical decision to the estimator's own ``_evt_falls_back`` (the single
        source of truth in measurement.py) so this audit-time mirror cannot drift from the real routing —
        it now covers the degenerate-fit, ``alpha > exceed_frac``, ``xi >= 1`` AND ``xi <= -0.5`` (Smith
        1985 non-regular GPD MLE, T2.8a) fallbacks alike.
        """
        from src.feedback.measurement import EVT_ALPHA_CUTOFF

        if alpha > EVT_ALPHA_CUTOFF:
            return "empirical"
        # alpha <= cutoff -> EVT branch, UNLESS _evt_cvar's fallback fires (degenerate fit, alpha > fu,
        # xi >= 1 infinite-mean, or xi <= -0.5 non-regular). _evt_falls_back returns the reason or None.
        return "evt" if rd._evt_falls_back(alpha) is None else "empirical(fallback)"

    try:
        from src.feedback.measurement import ReturnDistribution
    except Exception as exc:  # noqa: BLE001 - a report-only guard must never break the analysis
        return {"status": "skipped", "reason": f"measurement module unavailable: {exc}"}

    # Per arm: the winner search candidate (max val_fitness) that carries a usable validation vector + was
    # fed the tail. Re-fit ReturnDistribution on its val_returns as a proxy for the fed training distribution.
    per_arm: dict[str, dict[str, str]] = {}
    fed_arms: list[str] = []
    for arm in ARMS:
        arm_recs = [
            (r, vec) for r in records
            if r.get("arm") == arm and _is_search_candidate(r) and (vec := _val_returns(r)) is not None
        ]
        if not arm_recs:
            continue
        # Only arms actually FED the tail are relevant to a tail-estimator consistency claim.
        if not any(_fed_tail(r) for r, _ in arm_recs):
            continue
        fed_arms.append(arm)
        winner_rec, winner_vec = max(
            arm_recs, key=lambda rv: rv[0].get("metrics", {}).get("val_fitness", float("-inf"))
        )
        try:
            rd = ReturnDistribution().fit(winner_vec)
            per_arm[arm] = {f"{lvl:g}": _estimator_path(rd, float(lvl)) for lvl in levels}
        except Exception:  # noqa: BLE001 - a degenerate val vector must not crash the guard
            per_arm[arm] = {f"{lvl:g}": "skipped" for lvl in levels}

    if len(per_arm) < 2:
        return {
            "status": "skipped",
            "reason": (
                "fewer than 2 tail-fed arms have a usable validation vector — no cross-arm estimator "
                "consistency to check (records-only / search stage not archived with prompts?)"
            ),
            "fed_arms": fed_arms,
        }

    consistent: dict[str, bool] = {}
    for lvl in levels:
        key = f"{lvl:g}"
        paths = {a[key] for a in per_arm.values() if a.get(key) not in (None, "skipped")}
        consistent[key] = len(paths) <= 1
        if not consistent[key]:
            _LOG.warning(
                "EVT-consistency guard (DEEP_H2 §6.3): fed CVaR at alpha=%s uses INCONSISTENT estimators "
                "across tail-fed arms %s -> %s. The distributional-vs-scalar_cvar5 tail comparison mixes "
                "EVT and empirical CVaR; flag in the write-up.",
                key, fed_arms, {a: per_arm[a].get(key) for a in per_arm},
            )

    return {
        "status": "ok",
        "levels": [float(x) for x in levels],
        "fed_arms": fed_arms,
        "per_arm": per_arm,
        "consistent": consistent,
        "all_consistent": bool(all(consistent.values())),
        "note": (
            "Per-arm estimator path for the FED CVaR levels, re-derived from each arm winner's validation "
            "returns (a proxy for the fed training distribution). 'empirical(fallback)' = _evt_cvar's "
            "alpha>fu / degenerate-fit fallback fired. Inconsistency across tail-fed arms is logged + "
            "reported (DEEP_H2 §6.3), never raised."
        ),
    }


# =========================================================================== #
# Rank 8 — the costed 1/N benchmark floor (DeMiguel; PREREGISTRATION §9/§10)    #
# =========================================================================== #
class WeightPolicy:
    """A ``predict``-compatible shim that rolls a benchmark WEIGHT policy through the env.

    The benchmark allocators in ``src.baselines.strategies`` map a returns window to a
    target weight vector on the simplex. ``rollout_port_returns`` / ``PortfolioEnv`` instead
    drive a *policy* that emits a raw ACTION which the env projects onto the simplex via its
    FROZEN ``action.projection``. This shim bridges the two so a benchmark rolls through the
    IDENTICAL ``PortfolioEnv`` + transaction cost as the learned agent:

      1. reconstruct the per-asset lookback returns window from the observation — the env's
         ``_obs`` packs ``returns[t-lookback:t].ravel()`` as the LEADING ``lookback*N``
         block, so ``obs[:lookback*N].reshape(lookback, N)`` recovers exactly the window the
         strategy functions consume (no look-ahead — it is strictly-past data);
      2. call the benchmark ``strategy(window, cfg) -> w_risky`` (N risky weights on the
         simplex; the strategies are long-only, fully-invested in the risky sleeve);
      3. return an ACTION that the env's projection maps back to ``[w_risky, cash=0]``
         EXACTLY: for ``softmax`` return ``log(w)`` (softmax(log w) == w, up to the additive
         constant softmax is invariant to); for ``l1_normalize_of_clipped`` return ``w``
         itself (already non-negative and L1-normalised). A tiny floor avoids ``log(0)``.

    So the realized per-step net return the env reports for a 1/N policy is the genuine
    cost-charged equal-weight return — the DeMiguel floor, measured on the SAME leg.
    """

    def __init__(
        self,
        strategy: "Callable[..., np.ndarray]",
        *,
        lookback: int,
        n_assets: int,
        projection: str,
        cfg: Any = None,
    ) -> None:
        self.strategy = strategy
        self.lookback = int(lookback)
        self.n_assets = int(n_assets)
        self.projection = str(projection)
        self.cfg = cfg

    def _window(self, obs: np.ndarray) -> np.ndarray:
        flat = np.asarray(obs, dtype=float).ravel()
        block = self.lookback * self.n_assets
        return flat[:block].reshape(self.lookback, self.n_assets)

    def _action_for(self, w_risky: np.ndarray) -> np.ndarray:
        # Full action over N risky + 1 cash; benchmark holds no cash (fully invested).
        w = np.zeros(self.n_assets + 1, dtype=np.float64)
        wr = np.asarray(w_risky, dtype=np.float64).ravel()
        # A frozen benchmark can emit non-finite weights on a degenerate (very short)
        # window (e.g. risk_parity's iterative update divides by a zero marginal-risk
        # contribution). We must NOT edit the frozen strategy, so the shim falls back to
        # uniform-risky 1/N there — keeping the costed rollout finite and meaningful.
        if wr.size != self.n_assets or not np.all(np.isfinite(wr)) or wr.sum() <= 0:
            wr = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)
        w[: self.n_assets] = wr
        s = w.sum()
        if s > 0:
            w = w / s
        if self.projection == "softmax":
            # Invert softmax: logits = log(w); softmax(log w) == w (shift-invariant).
            return np.log(np.clip(w, 1e-12, None))
        # l1_normalize_of_clipped: the clipped-then-L1 projection returns w unchanged.
        return w

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Return ``(action, None)`` so the env projects back to the benchmark weights."""
        window = self._window(obs)
        try:
            w_risky = np.asarray(self.strategy(window, self.cfg), dtype=np.float64).ravel()
        except Exception:  # noqa: BLE001 - a degenerate window must degrade to 1/N, never crash the gate
            # ``_action_for`` already maps a non-finite/short vector to uniform 1/N; mirror that contract
            # for a RAISING strategy (e.g. a clustering allocator on a pathological window).
            w_risky = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)
        return self._action_for(w_risky), None


#: The frozen benchmark suite (PREREGISTRATION §9; FINAL_PLAN F.6). Names map to the
#: callables in ``src.baselines.strategies``; ``equal_weight`` is the DeMiguel 1/N floor.
# The frozen benchmark GATE (PREREGISTRATION §9; FINAL_PLAN F.6). ``spy_buy_and_hold`` was an EXACT 1/N
# duplicate of ``equal_weight`` mislabelled as the S&P 500 (no index/caps in the anonymised panel) — it is
# REMOVED to de-duplicate the DeMiguel floor, and the suite is EXPANDED with four further published,
# distinct long-only allocators (R19, 2026-06-20 leakage/rigor audit). All are forecast-free risk/structure
# allocators except momentum; a genuine market-cap or SPX-TR benchmark remains a gated data addition.
_BENCHMARK_NAMES: tuple[str, ...] = (
    "equal_weight",              # DeMiguel, Garlappi & Uppal (2009) 1/N floor
    "mean_variance",             # Markowitz (1952) + Ledoit-Wolf (2004) shrinkage
    "risk_parity",               # equal risk contribution, convex Spinu (2013) / Maillard et al. (2010)
    "hrp",                       # López de Prado (2016) hierarchical risk parity
    "minimum_variance",          # global minimum-variance (Clarke, de Silva & Thorley 2011)
    "maximum_diversification",   # Choueifaty & Coignard (2008) most-diversified portfolio
    "inverse_volatility",        # naive risk parity, 1/sigma
    "cross_sectional_momentum",  # Jegadeesh & Titman (1993) top-tertile, long-only
    "min_cvar",                  # CVaR-optimal / TAIL-optimal (Rockafellar & Uryasev 2000) — the beats-the-tail benchmark (2026-07-26)
)


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of a per-step simple-return series (a non-negative magnitude)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]  # a degenerate benchmark step must not poison the drawdown
    if r.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(-dd.min())  # report as a positive magnitude


#: The pre-registered delisting-return sensitivity band (R33; PREREGISTRATION §7 / config
#: ``inference.secondary_families.delisting_band.grid``). Each ``d`` is the delisting return imposed on
#: the 333 Shumway delisting cells; ``univ4`` (the −30/−55 M&A-contaminated heavy END of the delisting band
#: (R44)) sits at the band's structural extreme
#: because Refinitiv carries no vendor delisting terminal (the fixed surcharge hits 100% of delistings).
DELISTING_BAND_GRID: tuple[float, ...] = (0.0, -0.30, -0.55, -1.00)


def _pooled_cvar(panel_test: "pd.DataFrame", level: float) -> float:
    """Pooled lower-tail CVaR over ALL (name, session) return cells of a test-window panel.

    The non-delisting NaNs (a name not yet listed / already dead) are zero-filled — the headline
    ``liquidate_to_cash`` finiteness policy (loaders.OnMissing) — so the pool is the SAME finite return
    population the env trades, and only the band overwrite moves the tail. Reuses the A-line canonical
    ``src.inference.bootstrap.cvar`` (mean of the worst ⌈level·T⌉ cells), never a re-derivation.
    """
    from src.inference.bootstrap import cvar as _cvar

    x = np.asarray(panel_test.to_numpy(dtype=float)).ravel()
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size == 0:
        return float("nan")
    return float(_cvar(x, float(level)))


#: The panel+audit suffix the band ALWAYS reads — the delisting CELLS live in ``univ4`` (the surcharged
#: build), regardless of the headline default. The headline panel is a ZERO-FILL build (post-R44
#: ``univ3``; ``univ5`` post-Split-C) that carries NO Shumway audit log; if the band deferred to
#: ``gold_suffix()`` it would look for ``shumway_audit_log_univ5.parquet`` (which does NOT exist on
#: disk — only the ``_univ4`` audit + the finite ``returns_panel_univ4`` cells the surcharge
#: writes), so the LOAD-BEARING band would silently ``skip`` under a default analyze run. Pinning to
#: ``univ4`` lets the band LOCATE the 333 cells and BRACKET back to univ3 at ``d=0`` (the zero-fill end),
#: which is exactly the data-integrity claim it instruments (R44/R33; verified 2026-06-26). Override for
#: a test or a future re-pull via the ``audit_suffix`` kwarg.
DELISTING_BAND_AUDIT_SUFFIX: str = "univ4"


def delisting_band(
    *,
    panel_df: "pd.DataFrame | None" = None,
    audit_df: "pd.DataFrame | None" = None,
    test_window_dates: "tuple[Any, Any] | None" = None,
    grid: tuple[float, ...] = DELISTING_BAND_GRID,
    levels: tuple[float, ...] = (0.05, 0.01),
    gold_dir: "Path | str | None" = None,
    audit_suffix: str = DELISTING_BAND_AUDIT_SUFFIX,
) -> dict[str, Any]:
    """The delisting-return SENSITIVITY band (R33; PREREGISTRATION §7) — DATA-level, report-only, DISJOINT.

    Reprices ONLY the data: for each delisting return ``d`` in ``grid`` (``{0.0, −0.30, −0.55, −1.00}``)
    the 333 Shumway delisting cells that fall in the TEST window are overwritten with ``d`` and the
    POOLED test-window CVaR-5% (+ CVaR-1%) is recomputed. There is NO policy re-run — the frozen winners
    are untouched; this maps how sensitive the realized left tail is to the (unobservable) delisting
    recovery assumption.

    The band BRACKETS the tail — and ADR-051 located the truth AT the 0% end (2026-07-02)
    --------------------------------------------------------------------------------------
    ``d = 0.0`` is the ``univ3``/``liquidate_to_cash`` zero-fill end — and the EXECUTED observed-terminal
    recovery (ADR-051, superseding the planned ``univ4r`` re-pull) showed the corrected panel EQUALS it:
    all 333 dead names' realised terminal returns are already booked by the vendor series
    (``vendor_terminal_kept: 333``, ZERO surcharges), so the corrected Shumway panel (``univ5s``) is
    byte-identical to the zero-fill headline and the truth sits AT the band's 0% pole, not inside it.
    ``univ4`` (−30/−55) is the heavy end and is DOUBLY wrong: Refinitiv's frozen ``rf_meta_*`` pull
    carries NO delisting REASON (only ``TR.InstrumentDelistedDate`` — empty for all
    333 — ``TR.ExchangeName``, ``TR.TRBCEconomicSector``), so the surcharge was applied UNCONDITIONALLY to
    100% of delistings, INCLUDING premium M&A/mergers whose true terminal return was POSITIVE or neutral
    (verified test-window examples booked at a fabricated loss: ABMD→J&J, ALTR→Intel, CELG→BMS, RHT→IBM,
    TWX→AT&T, ATVI→Microsoft, XLNX→AMD, ALXN→AstraZeneca, ...) — and it DOUBLE-COUNTS terminals the
    vendor series already carries (ADR-051), so even for the genuine-failure minority (SHLD, FTR, JCP,
    SIVB, SBNY, ...) the flat −30/−55% lands ON TOP of the realised crash. The band is retained as the
    disclosed sensitivity BRACKET around that measured answer.
    Empirically the whole ``d∈{0,−1}`` sweep moves the pooled test CVaR-5% by only ~2% (105 test-window
    delisting cells — measured pre-Split-C on the 2018–2025 window; re-derived at analyze time on the
    executed window), a tiny fraction of the pooled return population, so the headline tail
    ORDERING is shown INVARIANT across the band — the M&A contamination is bounded and immaterial to the
    H2 conclusion, even though it would badly bias a per-name delisting study. The superseded reason-gated
    re-pull procedure remains documented, NOT fabricated here (R4).

    Cell location (verified, not the audit ``date``)
    ------------------------------------------------
    The audit log's ``date`` is the delisting EVENT/booking date (~13 sessions AFTER the corrected
    session); the Shumway value is booked onto each dead name's **last valid return session**. So each
    audit ``ric`` is mapped to its last non-NaN session in ``panel_df`` (which is exactly the one cell
    where ``univ4`` differs from ``univ3``, carrying the audit ``value`` — confirmed 333/333), and only
    those whose session lands in the test window are overwritten.

    Parameters
    ----------
    panel_df : pandas.DataFrame, optional
        The returns panel (sessions × RICs). Defaults to the band's PINNED panel
        (``returns_panel_{DELISTING_BAND_AUDIT_SUFFIX}`` = ``univ4``), independent of ``gold_suffix()`` —
        the headline default (post-R44 ``univ3``; ``univ5`` post-Split-C) carries no Shumway audit log.
        Injectable for the unit test.
    audit_df : pandas.DataFrame, optional
        The Shumway audit log (columns ``ric``, ``delisting_return``, ``value``; 333 rows for univ4).
        Defaults to ``data/clean/shumway_audit_log_<DELISTING_BAND_AUDIT_SUFFIX>.parquet`` (= ``univ4``). Injectable.
    test_window_dates : (start, end), optional
        Inclusive session bounds. Defaults to the frozen evaluation span
        (``config/inference.yaml: splits.evaluation.span`` = 2020-01-01 … 2026-06-30, SPLIT C).
        The window is CLAMPED to the band panel's available span (the univ4-era panel ends 2025-12,
        so the Split-C 2026H1 overhang is outside the band audit) and any clamp is RECORDED in the
        output (``window_requested`` / ``window_clamped_to`` / ``window_clamp_note``).
    grid, levels : tuple of float
        The delisting-return grid and the CVaR tail levels (default the co-primary 0.05 + the 0.01 EVT).
    gold_dir : Path or str, optional
        Override the gold/clean parent dir (defaults to ``data/gold`` resolved from the repo root).

    Returns
    -------
    dict
        ``{"status", "test_window", "n_delisting_cells_in_test", "grid": [...],
        "cvar": {"0.05": {d: cvar, ...}, "0.01": {...}}, "rows": [ {d, cvar_05, cvar_01,
        is_headline_extreme}, ... ], "headline_panel", "note"}`` — NO ``arm_a/arm_b/metric/level`` keys, so
        :func:`assert_realized_family_matches_frozen` is untouched (a declared report-only secondary,
        ``config/preregistration.yaml: inference.secondary_families.delisting_band``). ``status="skipped"``
        with a ``reason`` if the panel/audit cannot be loaded or no delisting cell lands in the test window.
    """
    import pandas as pd

    # --- resolve the panel + audit log. The band is PINNED to the suffix that carries the delisting cells
    # (``univ4`` — DELISTING_BAND_AUDIT_SUFFIX), NOT gold_suffix(): the headline default (post-R44 univ3;
    # univ5 post-Split-C) carries no shumway_audit_log on disk (only the _univ4 audit + the finite
    # returns_panel_univ4 cells the surcharge writes). The univ4 panel's last-valid cells are EXACTLY the
    # 333 the surcharge filled; overwriting them with each ``d`` gives d=0 ≈ the univ3 (liquidate_to_cash)
    # end (≈ to ~0.04%, NOT byte-identical: the band zeroes the final-session return that univ3 RETAINS)
    # and d∈{−0.30,−0.55} == univ4. Injectable for the test (panel_df/audit_df/audit_suffix). --- #
    suffix = audit_suffix
    if gold_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        gold_dir = repo_root / "data" / "gold"
    gold_dir = Path(gold_dir)
    try:
        if panel_df is None:
            panel_df = pd.read_parquet(gold_dir / f"returns_panel_{suffix}.parquet")
        if audit_df is None:
            # The audit log lives in data/clean/ (the CLEAN layer), sibling to data/gold/.
            audit_path = gold_dir.parent / "clean" / f"shumway_audit_log_{suffix}.parquet"
            audit_df = pd.read_parquet(audit_path)
    except (FileNotFoundError, OSError) as exc:
        return {
            "status": "skipped",
            "reason": f"delisting band needs the {suffix} panel + audit log: {exc}",
            "headline_panel": suffix,
        }
    if "ric" not in audit_df.columns:
        return {"status": "skipped", "reason": "audit log lacks a 'ric' column", "headline_panel": suffix}

    # --- test window (frozen evaluation span by default) --- #
    if test_window_dates is None:
        from src.utils.config import load_config

        span = load_config("inference").get("splits", {}).get("evaluation", {}).get(
            "span", ["2020-01-01", "2026-06-30"]
        )
        test_window_dates = (span[0], span[1])
    t0, t1 = pd.Timestamp(test_window_dates[0]), pd.Timestamp(test_window_dates[1])
    idx = pd.to_datetime(panel_df.index)
    # --- CLAMP the window to the band panel's available span: the band is PINNED to the univ4-era panel
    # (5,283 sessions, ending 2025-12) while the Split-C default span runs to 2026-06-30 — the overhang is
    # OUTSIDE the band audit, so it is clamped and RECORDED (never silently truncated); an EMPTY overlap
    # fails loud below via the skip reason naming both spans. univ5 ≡ univ3 byte-identically on the
    # band-panel overlap (ADR-044/051), so the clamp costs the audit nothing. --- #
    panel_start, panel_end = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    requested_window = (t0, t1)
    t0, t1 = max(t0, panel_start), min(t1, panel_end)
    window_clamped = (t0, t1) != requested_window
    panel_test = panel_df.loc[(idx >= t0) & (idx <= t1)].copy()
    if panel_test.empty:
        return {
            "status": "skipped",
            "reason": (
                f"no panel sessions in the test window {requested_window[0].date()}.."
                f"{requested_window[1].date()} (the {suffix} band panel spans "
                f"{panel_start.date()}..{panel_end.date()}; empty overlap)"
            ),
            "headline_panel": suffix,
        }

    # --- locate the delisting cells: per audit RIC, its last valid session in the FULL panel (= the cell
    # where univ4 differs from univ3 and carries the audit `value`), kept iff it lands in the test window. --- #
    cells: list[tuple[Any, str]] = []
    for ric in pd.unique(audit_df["ric"]):
        if ric not in panel_df.columns:
            continue
        lv = panel_df[ric].last_valid_index()
        if lv is not None and t0 <= pd.Timestamp(lv) <= t1:
            cells.append((lv, str(ric)))
    if not cells:
        return {
            "status": "skipped",
            "reason": f"no Shumway delisting cell falls in the test window {t0.date()}..{t1.date()}",
            "headline_panel": suffix,
            "test_window": [str(t0.date()), str(t1.date())],
        }

    # --- the band: overwrite the cells with each d, recompute the pooled tail (DATA-level, no policy re-run) --- #
    cvar_by_level: dict[str, dict[str, float]] = {f"{lvl:g}": {} for lvl in levels}
    rows: list[dict[str, Any]] = []
    headline_d = {-0.30, -0.55}  # the Shumway headline mixture; flagged as the band's structural extreme
    for d in grid:
        m = panel_test.copy()
        for dt, ric in cells:
            m.loc[dt, ric] = float(d)
        row: dict[str, Any] = {"d": float(d), "is_headline_extreme": float(d) in headline_d}
        for lvl in levels:
            c = _pooled_cvar(m, float(lvl))
            cvar_by_level[f"{lvl:g}"][f"{float(d):g}"] = c
            row[f"cvar_{int(round(lvl * 100)):02d}"] = c
        rows.append(row)

    return {
        "status": "ok",
        "headline_panel": suffix,
        "cells_source": suffix,  # the panel/audit the 333 delisting CELLS are read from (always univ4)
        "brackets_to": "≈univ3 at d=0 (liquidation; ≈ not byte-identical) … univ4 at d∈{−0.30,−0.55}",
        "test_window": [str(t0.date()), str(t1.date())],  # the EFFECTIVE (clamped) window
        "window_requested": [str(requested_window[0].date()), str(requested_window[1].date())],
        "window_clamped": bool(window_clamped),
        **(
            {
                "window_clamped_to": [str(t0.date()), str(t1.date())],
                "window_clamp_note": (
                    f"the {suffix}-era band panel spans {panel_start.date()}..{panel_end.date()} "
                    "(univ4 ends 2025-12), so the requested span's overhang (2026H1 under Split C) is "
                    "outside the band audit — immaterial: univ5 ≡ univ3 byte-identically on the "
                    "band-panel overlap (ADR-044/051)"
                ),
            }
            if window_clamped
            else {}
        ),
        "n_delisting_cells_in_test": len(cells),
        "grid": [float(d) for d in grid],
        "levels": [float(lvl) for lvl in levels],
        "cvar": cvar_by_level,
        "rows": rows,
        "note": (
            "DATA-level sensitivity (no policy re-run): the 333 Shumway delisting cells that fall in the "
            "test window are overwritten with each d and the POOLED test-window CVaR recomputed. The band "
            "BRACKETS the tail, and ADR-051 (2026-07-02) located the truth AT the 0% end: the executed "
            "observed-terminal recovery (univ5s) kept the vendor terminal for all 333 dead names with ZERO "
            "surcharges booked, so the corrected panel is byte-identical to the zero-fill headline — d=0.0 "
            "≈ the univ3 / liquidate_to_cash end (≈, NOT byte-identical — the band zeroes the "
            "final-session return univ3 retains). univ4 (−30/−55) is the heavy end and is doubly wrong: "
            "M&A-CONTAMINATED (Refinitiv's vault carries no delisting reason, so the surcharge hits 100% "
            "of delistings including premium M&A — ABMD→J&J, ALTR→Intel, CELG→BMS, RHT→IBM, ... — whose "
            "true terminal was positive) AND a terminal DOUBLE-COUNT on top of returns the vendor series "
            "already books. The whole sweep moves the pooled CVaR-5% ~2% (measured pre-Split-C on the "
            "2018–2025 window; re-derived here on the executed window), so the headline tail ordering is "
            "INVARIANT across it. The superseded reason-gated re-pull is documented in "
            "the data pipeline. Reported, DISJOINT from the frozen m=6 union, never a gate."
        ),
        "ma_contamination": (
            "univ4 surcharges 100% of the 333 delistings unconditionally (no vendor reason in the frozen "
            "rf_meta_* pull); a large share of the test-window cells are premium M&A booked at a "
            "fabricated −30/−55% loss, and ADR-051's observed-terminal recovery showed the flat surcharge "
            "also DOUBLE-COUNTS terminals the vendor series already books (333/333 vendor_terminal_kept, "
            "zero surcharges). The band's heavy end therefore OVER-states the tail and is an upper "
            "bracket only, not the tail — the corrected panel (univ5s ≡ univ5) sits at the 0% end."
        ),
    }


def benchmark_floor(
    panel: Any,
    cfg: Any,
    test_window: tuple[int, int],
    strategies: "dict[str, Callable[..., np.ndarray]] | None" = None,
    *,
    winner_test_returns: np.ndarray | None = None,
    winner_test_returns_per_seed: "list[np.ndarray] | None" = None,
    n_trials: int = 1,
    winner_n_trials: int | None = None,
    cvar_alpha: float = 0.05,
) -> dict[str, Any]:
    """Roll the benchmark suite through the SAME costed test env; gate the winner's DSR.

    Implements the DeMiguel 1/N floor (PREREGISTRATION §9 benchmark suite; §10) as a
    POST-FREEZE, report-only gate. Each benchmark weight policy is rolled through the
    IDENTICAL ``PortfolioEnv`` over ``test_window`` via :class:`WeightPolicy` +
    ``src.env.runner.rollout_port_returns`` (so every benchmark pays the same transaction
    cost as the learned winner), and its test Sharpe / CVaR / MaxDD / Deflated Sharpe are
    reported. The gate requires the FROZEN winner's test-DSR to strictly exceed the best
    benchmark's test-DSR — it NEVER re-selects the winner (the winner was frozen on
    validation; this only reports pass/fail against the floor).

    Parameters
    ----------
    panel : Panel
        The shared panel (the test window indexes into it).
    cfg : Any
        The environment config (``config/environment.yaml``) — the same object the campaign
        builds the env from, so lookback / projection / cost match exactly.
    test_window : (int, int)
        Half-open ``[start, end)`` test window (the campaign's resolved 2020-2026 leg).
    strategies : dict[str, callable], optional
        ``{name: strategy_fn}`` override (default: the eight frozen benchmarks from
        ``src.baselines.strategies``).
    winner_test_returns : np.ndarray, optional
        The frozen winner's realized per-step TEST returns. When given, the floor gate is
        evaluated (winner DSR vs best-benchmark DSR); otherwise only the benchmark table is
        returned (``gate`` is ``None``).
    n_trials : int
        Trial count for the Deflated Sharpe (the winner is ONE frozen strategy -> 1; the
        benchmarks are likewise un-searched). Forwarded to ``deflated_sharpe_ratio``.
    cvar_alpha : float
        CVaR tail level for the reported benchmark CVaR (default headline ``0.05``).

    Returns
    -------
    dict
        ``{"benchmarks": {name: {sharpe, cvar, max_drawdown, dsr, n_steps}}, "gate":
        {winner_dsr, best_benchmark, best_benchmark_dsr, passed} | None}``.
    """
    from src.env.portfolio_env import PortfolioEnv
    from src.env.runner import rollout_port_series
    from src.inference.bootstrap import cvar, sharpe_ratio
    from src.inference.deflated_sharpe import deflated_sharpe_ratio

    if strategies is None:
        from src.baselines import strategies as _strat

        strategies = {name: getattr(_strat, name) for name in _BENCHMARK_NAMES}

    state_cfg = cfg["state"] if "state" in cfg else cfg.state
    lookback = int(state_cfg["lookback_days"])
    action_cfg = cfg["action"] if "action" in cfg else cfg.action
    projection = str(action_cfg["projection"])
    n_assets = int(panel.N)
    # Headline cost (bps) for the per-benchmark annualised turnover-cost column (T0 DEEP_BENCH_T0 #1). Read
    # from the env cfg (single source); default to the documented 10 bps if a synthetic cfg omits it.
    cost_cfg = (cfg["costs"] if "costs" in cfg else getattr(cfg, "costs", {})) or {}
    try:
        headline_bps = float(cost_cfg["headline_bps"] if "headline_bps" in cost_cfg else cost_cfg.get("headline_bps", 10.0))
    except Exception:  # noqa: BLE001 - a missing cost block degrades to the documented default
        headline_bps = 10.0

    # A passthrough reward: the env's port_ret (gross - cost) is what we measure; the reward
    # VALUE is irrelevant to a benchmark rollout (rollout reads info['port_ret'], not reward).
    def _passthrough(w, r_t, w_prev, port_ret, info):  # noqa: ANN001
        return float(port_ret), {}, None

    start, end = int(test_window[0]), int(test_window[1])

    bench: dict[str, dict[str, Any]] = {}
    for name, strat in strategies.items():
        policy = WeightPolicy(
            strat, lookback=lookback, n_assets=n_assets, projection=projection, cfg=cfg
        )
        env = PortfolioEnv(panel, cfg, _passthrough, start=start, end=end)
        # rollout_port_series exposes the per-step gross/turnover/net decomposition (Rank 4), so we get the
        # realised turnover for the T0 cost table (DEEP_BENCH_T0 #1) alongside the net returns. `net` is
        # byte-identical to rollout_port_returns' output, so the Sharpe/CVaR/DSR are unchanged.
        series = rollout_port_series(env, policy)
        rets = series["net"]
        turn = series["turnover"]
        turn_f = turn[np.isfinite(turn)]
        mean_turnover = float(turn_f.mean()) if turn_f.size else 0.0
        # Annualised cost @ headline bps: per-step cost = bps*1e-4*turnover; *252 trading days/yr; *100 -> %.
        ann_cost_pct = float(mean_turnover * headline_bps * 1e-4 * 252 * 100.0)
        bench[name] = {
            "sharpe": float(sharpe_ratio(rets)),
            "cvar": float(cvar(rets, cvar_alpha)),
            "max_drawdown": _max_drawdown(rets),
            "dsr": float(deflated_sharpe_ratio(rets, n_trials)),
            "n_steps": int(rets.size),
            # T0 cost table (DEEP_BENCH_T0 #1): proves the BINDING benchmark is a fairly-costed diversified
            # allocator, not a daily-re-estimation cost artefact (the "you taxed the benchmarks" defence).
            "mean_turnover": mean_turnover,
            "ann_cost_pct": ann_cost_pct,
        }

    gate: dict[str, Any] | None = None
    wr = None
    if winner_test_returns is not None or winner_test_returns_per_seed:
        if winner_test_returns is not None:
            wr = np.asarray(winner_test_returns, dtype=float).ravel()  # representative path (market_reference)
        # The WINNER was SEARCHED over its candidate budget, so its DSR is deflated by that search
        # multiplicity (winner_n_trials); the benchmarks are UN-searched (n_trials, default 1) — #17.
        wnt = int(winner_n_trials) if winner_n_trials is not None else int(n_trials)
        # ROBUST winner DSR for the gate: the winner has STOCHASTIC per-seed test paths. Gate the MEDIAN of
        # the per-seed DSRs (each deflated by wnt) against the best SINGLE-PATH benchmark DSR — a like-for-
        # like single-realisation comparison. Averaging the seed paths FIRST shrinks the variance ~sqrt(S)
        # and INFLATES the DSR (the seed-averaging anti-conservatism the H2 #9/#14 fix removed; this gate
        # re-introduced it via the seed-mean path — critical-review 2026-06-20). Single-path fallback.
        # UNDEFLATED (N=1) winner DSR alongside the deflated gate (DEEP_BENCH_T0 #2): makes the asymmetric
        # deflation transparent — a reader can separate "won/lost on performance" (N=1 vs N=1) from "the
        # multiplicity penalty" (the winner pays N=wnt, benchmarks N=1). Same median-per-seed / single-path
        # construction, but deflated by N=1 to match the benchmarks' un-searched treatment.
        if winner_test_returns_per_seed:
            per = [
                float(deflated_sharpe_ratio(np.asarray(v, dtype=float).ravel(), wnt))
                for v in winner_test_returns_per_seed
                if np.asarray(v).size > 1
            ]
            per_n1 = [
                float(deflated_sharpe_ratio(np.asarray(v, dtype=float).ravel(), 1))
                for v in winner_test_returns_per_seed
                if np.asarray(v).size > 1
            ]
            winner_dsr = float(np.median(per)) if per else float("-inf")
            winner_dsr_undeflated = float(np.median(per_n1)) if per_n1 else float("-inf")
            method = "median_per_seed"
        else:
            winner_dsr = float(deflated_sharpe_ratio(wr, wnt)) if wr is not None else float("-inf")
            winner_dsr_undeflated = float(deflated_sharpe_ratio(wr, 1)) if wr is not None else float("-inf")
            method = "single_path"
        best_name = max(bench, key=lambda k: bench[k]["dsr"]) if bench else None
        best_dsr = bench[best_name]["dsr"] if best_name is not None else float("-inf")
        gate = {
            "winner_dsr": winner_dsr,
            "winner_dsr_undeflated_n1": winner_dsr_undeflated,  # T0 DEEP_BENCH_T0 #2 (transparency, not the gate)
            "winner_dsr_method": method,
            "winner_n_trials": wnt,
            "best_benchmark": best_name,
            "best_benchmark_dsr": float(best_dsr),
            # DeMiguel floor: the winner must STRICTLY beat the best benchmark's DSR.
            "passed": bool(winner_dsr > best_dsr),
        }

    # REAL market reference (R20 / data-enrichment research 2026-06-20): the full-universe equal-weight
    # market line (`data/gold/market_proxy_*.parquet`) priced with the FRED DGS3MO risk-free rate, plus
    # the winner's market-relative stats (beta / annualised alpha / information ratio). ADDITIVE reporting
    # only — NOT part of the same-universe DeMiguel gate above (the agent trades the 30-asset sleeve, the
    # market spans the full universe). Degrades to None on a synthetic-only install (files absent).
    market_reference: dict[str, Any] | None = None
    try:
        from src.backtest.metrics import compute_metrics
        from src.data.market_reference import load_market_proxy_returns, load_risk_free_daily

        win_dates = np.asarray(panel.dates)[start:end]
        mkt = load_market_proxy_returns(win_dates)
        rf = load_risk_free_daily(win_dates)
        if mkt.available and mkt.returns.size:
            m = mkt.returns
            rfd: Any = rf.daily if rf.available else 0.0
            # Route the market Sharpe through the SAME rf convention as winner_vs_market below (excess
            # returns), so the side-by-side Sharpe pair is comparable — else one is raw, one rf-adjusted
            # (critical-review 2026-06-20, #14). mean-rf is exact for the Sharpe numerator.
            rf_mean = float(np.mean(rfd)) if rf.available else 0.0
            market_reference = {
                "market": {
                    "sharpe": float(sharpe_ratio(np.asarray(m) - rf_mean)),
                    "cvar": float(cvar(m, cvar_alpha)),
                    "max_drawdown": _max_drawdown(m),
                    "dsr": float(deflated_sharpe_ratio(m, 1)),  # the market is UN-searched -> N=1
                    "ann_return_pct": float(np.asarray(m).mean() * 252 * 100.0),
                    "n_steps": int(m.size),
                },
                "rf_source": rf.source if rf.available else "none",
                "rf_annual_pct_mean": float(rf.annual_pct_mean) if rf.available else 0.0,
            }
            if wr is not None and wr.size == m.size:
                full = compute_metrics(wr, benchmark=m, risk_free=rfd, n_trials=n_trials)
                market_reference["winner_vs_market"] = {
                    k: float(full[k])
                    for k in ("beta", "alpha_ann", "information_ratio", "tracking_error_ann", "sharpe")
                    if k in full
                }
    except Exception:  # noqa: BLE001 - a reporting reference must NEVER break the benchmark gate
        market_reference = None

    return {"benchmarks": bench, "gate": gate, "market_reference": market_reference}


def pbo_markdown(results: dict[str, dict[str, Any]], *, n_blocks: int) -> str:
    """Render a small per-arm PBO table as markdown.

    PBO near/above 0.5 indicates in-sample candidate selection carries no OOS information
    (severe overfitting); near 0 indicates the IS-best candidate tends to stay good OOS.
    """
    lines = [
        "# Campaign overfitting — PBO / CSCV (PREREGISTRATION §10; primary guard)",
        "",
        f"CSCV blocks S = {n_blocks} (`config/inference.yaml: pbo.n_blocks`). PBO is computed "
        "PER ARM over that arm's candidates' per-period validation returns "
        "(`src.inference.overfitting.pbo`). PBO near/above 0.5 = severe overfitting; near 0 = "
        "in-sample-best stays good out-of-sample.",
        "",
        "| arm | n candidates | T_val | PBO | status |",
        "|---|---|---|---|---|",
    ]
    for arm, e in results.items():
        pbo_str = "n/a" if e.get("pbo") is None else f"{e['pbo']:.3f}"
        status = e.get("status", "?")
        if status == "skipped" and e.get("reason"):
            status = f"skipped ({e['reason']})"
        lines.append(f"| {arm} | {e.get('n_candidates', 0)} | {e.get('t_val', 0)} | {pbo_str} | {status} |")
    lines.append("")
    return "\n".join(lines)


def pbo_dsr_markdown(
    primary: dict[str, dict[str, Any]],
    dsr: dict[str, dict[str, Any]] | dict[str, Any],
    *,
    n_blocks: int,
) -> str:
    """Render the SECOND PBO ranked on the DSR-proxy (per-block Sharpe) alongside the frozen primary (R36).

    Shows, per arm, the frozen mean-return PBO (the PRIMARY guard) next to the per-block-Sharpe PBO (the
    DSR-proxy selection rule) and their absolute gap. Close agreement empirically closes the DEEP_STATS A3
    "you didn't guard the rule you used" concern (with λ=0 the DSR is monotone in Sharpe, so they should
    agree closely). The PRIMARY column here is the SAME `campaign_pbo` output reported above — unchanged.
    """
    if not isinstance(dsr, dict) or dsr.get("status") == "error":
        reason = dsr.get("reason", "not computed") if isinstance(dsr, dict) else "not computed"
        return f"## Second PBO ranked on DSR-proxy (R36) — n/a\n\n{reason}\n"
    lines = [
        "## Second PBO ranked on the DSR-proxy (per-block Sharpe) — guards the SELECTION rule (R36; M3)",
        "",
        f"CSCV blocks S = {n_blocks}. The frozen PRIMARY PBO (`src.inference.overfitting.pbo`, UNCHANGED) "
        "ranks IS/OOS on the MEAN validation return; winner SELECTION used the validation **DSR** "
        "(`src.selection.fitness`, monotone in per-series Sharpe at the frozen λ=0). This SECOND column ranks "
        "on the per-block annualised SHARPE — the DSR-proxy. Close agreement ⇒ the mean-return proxy "
        "empirically guards the rule the campaign actually USED (DEEP_STATS A3 point 2). Report-only, "
        "additive; the frozen guard is the mean-return column.",
        "",
        "| arm | n candidates | PBO (mean-return, PRIMARY) | PBO (per-block Sharpe / DSR-proxy) | |Δ| | status |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        ep = primary.get(arm, {}) if isinstance(primary, dict) else {}
        ed = dsr.get(arm, {}) if isinstance(dsr, dict) else {}
        p_primary = ep.get("pbo")
        p_dsr = ed.get("pbo")
        p_str = "n/a" if p_primary is None else f"{p_primary:.3f}"
        d_str = "n/a" if p_dsr is None else f"{p_dsr:.3f}"
        gap = "n/a" if (p_primary is None or p_dsr is None) else f"{abs(float(p_primary) - float(p_dsr)):.3f}"
        status = ed.get("status", "?")
        if status == "skipped" and ed.get("reason"):
            status = f"skipped ({ed['reason']})"
        lines.append(
            f"| {arm} | {ed.get('n_candidates', ep.get('n_candidates', 0))} | {p_str} | {d_str} | {gap} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def winner_dsr_markdown(results: dict[str, dict[str, Any]]) -> str:
    """Render the per-arm headline winner-DSR table (canonical cross-trial var vs proxy).

    ``DSR (canonical)`` deflates the winner's Sharpe by the empirical cross-CANDIDATE Sharpe
    dispersion (the Bailey-Lopez de Prado input); ``DSR (proxy)`` is the within-series
    ``var_sr=None`` value the search path recorded. They coincide only under a homogeneous
    zero-skill null; a gap is the cross-trial-variance correction (Rank 16).
    """
    lines = [
        "# Campaign headline Deflated Sharpe — canonical cross-trial variance (Rank 16; secondary)",
        "",
        "Per arm: the WINNER's validation Deflated Sharpe recomputed with the empirical "
        "cross-candidate Sharpe dispersion `var_sr = Var(per-candidate val Sharpes, ddof=1)` "
        "(canonical Bailey-Lopez de Prado) versus the within-series `var_sr=None` proxy the "
        "WIRED selection path records. DSR is SECONDARY (PBO/CSCV is primary).",
        "",
        "| arm | n candidates | winner | winner Sharpe | var_sr | DSR (canonical) | DSR (proxy) | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, e in results.items():
        def _f(key: str) -> str:
            v = e.get(key)
            return "n/a" if v is None else f"{v:.4f}"

        status = e.get("status", "?")
        if status == "skipped" and e.get("reason"):
            status = f"skipped ({e['reason']})"
        lines.append(
            f"| {arm} | {e.get('n_candidates', 0)} | {e.get('winner_id') or 'n/a'} | "
            f"{_f('winner_sharpe')} | {_f('var_sr')} | {_f('dsr_canonical')} | "
            f"{_f('dsr_proxy')} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def divergence_markdown(d: dict[str, Any]) -> str:
    """Render the training-divergence diagnostic (R34) — diverged-RUN count + rate, report-only."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Training-divergence diagnostic (R34) — n/a\n\n{reason}\n"
    rate = d.get("divergence_rate")
    rate_s = "n/a" if rate is None else f"{rate:.4f} ({rate * 100:.2f}%)"
    budget = d.get("n_candidates_budget")
    wd = d.get("winner_diverged") or []
    winner_line = (
        "NO winner's training diverged"
        if not wd
        else f"WINNERS whose training diverged: {', '.join(wd)} — INVESTIGATE"
    )
    out = [
        "## Training-divergence diagnostic — diverged-RUN count + rate (R34; report-only, DISJOINT)",
        "",
        "Anomaly monitoring writes every `critic_explosion` event to one append-only `anomalies.jsonl` per "
        "run, so the LINE count over-states how many distinct RUNS diverged. Events are clustered into RUNS "
        "by step-reset (a step that goes backwards = a new training). Report-only: the trainer is unchanged.",
        "",
        f"- Anomaly LINES (`critic_explosion`): **{d.get('n_anomaly_lines', 0)}**",
        f"- Diverged RUNS (clustered by step-reset): **{d.get('n_diverged_runs', 0)}**"
        + ("" if not d.get("transient_runs") else f"  (of which {d.get('transient_runs')} single-step/transient)"),
        f"- Divergence rate (runs / {budget if budget else '?'} candidate-trainings): **{rate_s}**",
        f"- {winner_line}  *(attribution: {d.get('winner_attribution', '?')})*",
        "",
        "**Disclosure.** The reward is UNBOUNDED on purpose (`norm_reward=False` is DELIBERATE — the reward "
        "is the object of study, so its scale is left as authored), so a mis-scaled candidate can transiently "
        "blow the critic loss up. But a diverged candidate scores POORLY on the held-out validation fitness "
        "and LOSES selection, so divergence biases toward NOISE in the dropped tail, NOT toward the H2 "
        "headline (a diverged candidate becomes a winner only if it ALSO posted a strong sealed validation "
        "Sharpe).",
        "",
    ]
    return "\n".join(out)


def compute_accounting_markdown(c: dict[str, Any]) -> str:
    """Render the per-arm compute-accounting table (R35) — attempted/accepted/failed + prompt-tokens."""
    if not c or c.get("status") != "ok" or not c.get("rows"):
        reason = (c or {}).get("reason", "not computed")
        return f"## Compute-accounting (R35) — n/a\n\n{reason}\n"
    out = [
        "## Compute-accounting — candidates + token usage per arm (R35; report-only, DISJOINT)",
        "",
        "Per arm, from the archived `failures.jsonl` + `llm_calls.jsonl`: candidates ATTEMPTED / ACCEPTED "
        "(passed the gate + evaluated) / FAILED the gate, and total prompt (input) tokens. Report-only; "
        "DISJOINT from the frozen m=6 family.",
        "",
        "| arm | kind | LLM calls | accepted | failed | attempted | resamples? | prompt tok | completion tok | tail-fed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in c.get("rows", []):
        out.append(
            f"| {r['arm']} | {r.get('kind', '?')} | {r.get('n_llm_calls', 0)} | {r.get('n_accepted', 0)} | "
            f"{r.get('n_failed', 0)} | {r.get('n_attempted', 0)} | "
            f"{'yes' if r.get('resamples_to_full_slate') else 'no'} | {r.get('prompt_tokens', 0)} | "
            f"{r.get('completion_tokens', 0)} | {'yes' if r.get('tail_fed') else 'no'} |"
        )
    t = c.get("totals", {})
    out += [
        "",
        f"Totals: accepted **{t.get('n_accepted', 0)}**, failed **{t.get('n_failed', 0)}**, prompt-tokens "
        f"**{t.get('prompt_tokens', 0):,}**, completion-tokens **{t.get('completion_tokens', 0):,}**.",
        "",
        "**Disclosure.** " + c.get("asymmetry_note", ""),
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Report-only mechanism kernel (ADR-039) — responsiveness / mediation / regime / named-vs-blinded
# (DISJOINT from the frozen m=6 family; each degrades gracefully and NEVER gates the headline).
# --------------------------------------------------------------------------- #
def _inspect_rewards():  # type: ignore[no-untyped-def]
    """Import the qualitative-inspection helpers (path-insert plumbing, mirroring inspect_rewards itself)."""
    try:  # pragma: no cover - import plumbing
        import inspect_rewards as _ir
    except ImportError:  # pragma: no cover - standalone invocation
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import inspect_rewards as _ir
    return _ir


def _mechanism_pairs(
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Paired mechanism arrays over the TAIL-FED candidates, in (generation, candidate_id) order.

    2026-07-05 (M13 construct fix): ``x`` is now the tail summary the DESIGNER WAS FED — parsed from
    the archived prompt via ``_fed_tail_vector`` (= the previous generation's best block) — NOT the
    candidate's own post-training measured tail, which the old wiring used and which reversed the
    registered estimand (PREREGISTRATION §2a SQ1: *does the FED tail signal change the authored
    code?*). Generation-0 candidates (fed nothing) are excluded by the ``_was_fed_tail`` gate.

    * ``x`` — the FED **CVaR-5%** level (from the prompt's feedback block).
    * ``m`` — an authored-code feature: the count of **tail-shaped constructs** the program references
      (``_construct_prevalence`` restricted to ``_TAIL_CONSTRUCTS``).
    * ``y`` — the realised tail OUTCOME: empirical ``cvar(metrics['val_returns'], 0.05)`` per candidate
      (the VAL-RETURNS proxy path — fully archived and better powered than the sparse winner path).
    * ``dx`` / ``dm`` — the REGISTERED §2a form: per-(arm, generation) deltas. Within a generation
      every candidate sees the SAME fed block (x is generation-constant ⇒ the per-candidate rows are
      CLUSTERED); the registered "Spearman of Δ(fed tail) vs Δ(authored-reward feature)" therefore
      aggregates m to the generation mean and differences consecutive generations WITHIN each arm,
      pooling the deltas across the COHERENT tail-fed arms (distributional, scalar_cvar5 — S7,
      2026-07-06: placebo_shuffled also carries the labels but with DERANGED values; it is the
      calibration floor, reported as its own §2a(f) fingerprint row, never pooled into the signal).
      The (X, M) delta cells are populated INDEPENDENTLY of outcome availability; only the
      levels/mediation arrays condition on a finite ``y`` (they need it).

    Returns ``{"x", "m", "y", "dx", "dm"}`` (float arrays; possibly empty).
    """
    from src.inference.bootstrap import cvar

    ir = _inspect_rewards()
    # S7 (2026-07-06): the pooled PRIMARY is restricted to the COHERENT-tail arms. placebo_shuffled
    # also passes _was_fed_tail (its prompt carries the six labels), but its values are DERANGED by
    # construction — the number under "CVaR 5%" is some OTHER statistic — so pooling it feeds the
    # primary an incoherent x (scale-mixed, attenuating rho toward the predicted null verdict:
    # anti-conservative). It is the design's calibration FLOOR (information_gap's floor_arm) and is
    # reported as its own row in the §2a(f) fingerprint table, never inside the signal pool.
    _COHERENT_TAIL_ARMS = ("distributional", "scalar_cvar5")
    fed = [r for r in records
           if str(r.get("arm") or "") in _COHERENT_TAIL_ARMS and ir._was_fed_tail(r)]
    fed = ir._by_generation(fed)
    xs: list[float] = []
    ms: list[float] = []
    ys: list[float] = []
    by_arm_gen: dict[tuple[str, int], dict[str, list[float]]] = {}
    for r in fed:
        vec = ir._fed_tail_vector(r)
        if vec is None or vec.size == 0 or not np.isfinite(vec[0]):
            continue
        prevalence = ir._construct_prevalence(ir._reward_source(r))
        m_count = float(sum(1 for name in ir._TAIL_CONSTRUCTS if prevalence.get(name)))
        x_fed = float(vec[0])  # cvar_05 — the headline FED tail level
        # SQ1 needs only (X, M): populate the delta cells BEFORE the y gate (2026-07-06) — a fed
        # candidate whose code is archived but whose val_returns are missing/non-finite belongs in
        # the code-responsiveness estimand; conditioning the X->M sample on OUTCOME availability was
        # a silent sample-selection narrowing (it also let a fully-y-missing generation truncate the
        # delta chain). The y gate below still scopes the LEVELS/mediation arrays, which need y.
        cell = by_arm_gen.setdefault(
            (str(r.get("arm") or ""), int(r.get("generation") or 0)), {"x": [], "m": []}
        )
        cell["x"].append(x_fed)
        cell["m"].append(m_count)
        vr = _val_returns(r)
        y = float(cvar(vr, 0.05)) if vr is not None else float("nan")
        if not np.isfinite(y):
            continue
        xs.append(x_fed)
        ms.append(m_count)
        ys.append(y)
    dxs: list[float] = []
    dms: list[float] = []
    for arm in sorted({a for a, _g in by_arm_gen}):
        gens = sorted(g for a, g in by_arm_gen if a == arm)
        for g_prev, g_next in zip(gens, gens[1:]):
            if g_next != g_prev + 1:
                # Registered §2a form: deltas over STRICTLY consecutive generations only. If an
                # entire generation is absent for an arm (every candidate filtered non-finite),
                # differencing across the gap would silently off-spec the estimand — skip it.
                continue
            prev, nxt = by_arm_gen[(arm, g_prev)], by_arm_gen[(arm, g_next)]
            dxs.append(float(np.mean(nxt["x"]) - np.mean(prev["x"])))
            dms.append(float(np.mean(nxt["m"]) - np.mean(prev["m"])))
    return {
        "x": np.asarray(xs, dtype=float),
        "m": np.asarray(ms, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "dx": np.asarray(dxs, dtype=float),
        "dm": np.asarray(dms, dtype=float),
    }


def _condition_seed_cells(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """One ``(x, mean-m)`` row PER SEED for a sub-experiment condition (S8, 2026-07-06).

    The redesigned SQ3b injects ONE per-seed reference tail block, so ``x`` is CONSTANT within a
    ``(condition, seed)`` cell and the K authored candidates are clustered; i.i.d. candidate-row
    resampling would understate the differential CI by up to ``~sqrt(K)`` — anti-conservative for
    the REGISTERED ``legibility_helps = ci_low > 0`` decision rule. Collapsing to per-seed cells
    makes the bootstrap's exchangeable unit the SEED, the true independent unit."""
    ir = _inspect_rewards()
    by_seed: dict[int, list[tuple[float, float]]] = {}
    for r in records:
        if not ir._was_fed_tail(r):
            continue
        vec = ir._fed_tail_vector(r)
        if vec is None or vec.size == 0 or not np.isfinite(vec[0]):
            continue
        prevalence = ir._construct_prevalence(ir._reward_source(r))
        m = float(sum(1 for name in ir._TAIL_CONSTRUCTS if prevalence.get(name)))
        by_seed.setdefault(int(r.get("seed") or 0), []).append((float(vec[0]), m))
    xs: list[float] = []
    ms: list[float] = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        xs.append(rows[0][0])  # x is cell-constant by design (one injected block per seed)
        ms.append(float(np.mean([mm for _xx, mm in rows])))
    return np.asarray(xs, dtype=float), np.asarray(ms, dtype=float)


#: The §2a(f) fingerprint's per-arm X definition: which fed quantity each arm's designers SAW.
_FINGERPRINT_X_KIND: dict[str, str] = {
    "distributional": "fed cvar_05 (six-scalar tail block)",
    "scalar_cvar5": "fed cvar_05 (single-line CVaR block)",
    "scalar": "fed scalar (validation score line)",
    "placebo_shuffled": "value under the 'CVaR 5%' label (DERANGED by construction — the floor row)",
}

_FED_SCALAR_RE = re.compile(r"Your previous reward scored:\s*([+-]?\d+(?:\.\d+)?)")


def _arm_generation_deltas(
    records: list[dict[str, Any]], arm: str, x_of: Any
) -> tuple[np.ndarray, np.ndarray, int]:
    """Strictly-consecutive per-generation ``(dx, dm)`` deltas for ONE arm (the SQ1 kernel's form).

    ``x_of(record) -> float | None`` extracts the arm-appropriate FED quantity; ``None``/non-finite
    rows are skipped. Returns ``(dx, dm, n_levels)``."""
    ir = _inspect_rewards()
    by_gen: dict[int, dict[str, list[float]]] = {}
    n_levels = 0
    for r in records:
        if str(r.get("arm") or "") != arm:
            continue
        x = x_of(r)
        if x is None or not np.isfinite(float(x)):
            continue
        prevalence = ir._construct_prevalence(ir._reward_source(r))
        m = float(sum(1 for name in ir._TAIL_CONSTRUCTS if prevalence.get(name)))
        cell = by_gen.setdefault(int(r.get("generation") or 0), {"x": [], "m": []})
        cell["x"].append(float(x))
        cell["m"].append(m)
        n_levels += 1
    gens = sorted(by_gen)
    dxs: list[float] = []
    dms: list[float] = []
    for g_prev, g_next in zip(gens, gens[1:]):
        if g_next != g_prev + 1:
            continue  # registered §2a form: strictly consecutive generations only
        dxs.append(float(np.mean(by_gen[g_next]["x"]) - np.mean(by_gen[g_prev]["x"])))
        dms.append(float(np.mean(by_gen[g_next]["m"]) - np.mean(by_gen[g_prev]["m"])))
    return np.asarray(dxs, dtype=float), np.asarray(dms, dtype=float), n_levels


def fingerprint_responsiveness_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The §2a(f) MECHANISM-FINGERPRINT responsiveness rows, per arm (S9, 2026-07-06; report-only).

    The hash-frozen fingerprint matrix registers per-arm rows the pooled SQ1 block cannot fill —
    most critically the **scalar arm's own-scalar responsiveness** (the same delta-form statistic
    with X = the fed scalar score line), which UNIQUELY discriminates the A4 prior-dominance
    account (A4 predicts ≈0 there; A2 readout predicts >0 because the scalar is legible), and the
    ``placebo_shuffled`` row as the calibration FLOOR (its 'CVaR 5%' slot carries a deranged
    value). Registered as "computable from the same archived records with the same SQ1 statistic";
    this makes the registered adjudication matrix mechanically producible instead of improvised
    post-data. Never gates anything."""
    from src.inference.responsiveness import responsiveness

    ir = _inspect_rewards()

    def _x_tail(r: dict[str, Any]) -> float | None:
        if not ir._was_fed_tail(r):
            return None
        vec = ir._fed_tail_vector(r)
        return None if vec is None or vec.size == 0 else float(vec[0])

    def _x_scalar(r: dict[str, Any]) -> float | None:
        if int(r.get("generation") or 0) < 1:
            return None  # gen-0 authors from the tail/score-neutral base prompt
        m = _FED_SCALAR_RE.search(str(r.get("prompt") or ""))
        return None if m is None else float(m.group(1))

    x_fns = {
        "distributional": _x_tail,
        "scalar_cvar5": _x_tail,
        "scalar": _x_scalar,
        "placebo_shuffled": _x_tail,  # parses the deranged value under the CVaR-5% label — the floor
    }
    rows: dict[str, Any] = {}
    for arm, x_of in x_fns.items():
        dx, dm, n_levels = _arm_generation_deltas(records, arm, x_of)
        if dx.size >= 3:
            r = responsiveness(dx, dm, rng=np.random.default_rng(0))
        else:
            r = {"status": "no_data", "reason": f"{dx.size} consecutive-generation deltas (need >= 3)"}
        r["x_kind"] = _FINGERPRINT_X_KIND[arm]
        r["n_deltas"] = int(dx.size)
        r["n_levels"] = int(n_levels)
        r["is_floor"] = arm == "placebo_shuffled"
        rows[arm] = r
    any_ok = any(v.get("status") == "ok" for v in rows.values())
    return {"status": "ok" if any_ok else "no_data", "rows": rows,
            "note": "§2a(f) fingerprint rows — the scalar-arm row is the A4 discriminator; "
                    "placebo_shuffled is the calibration floor, never pooled into the SQ1 primary"}


def distance_moderator_exploratory(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The §2a(b) DECLARED-EXPLORATORY distance moderator (S9, 2026-07-06; report-only, no gate).

    Registered form: within the CVaR-fed arms only, per-generation designer responsiveness (Δm)
    regressed on the NUMERIC DISTANCE between consecutively fed CVaR values, controlling magnitude,
    iteration index, and fitness-plateau state. The numeracy account predicts a POSITIVE distance
    coefficient (a psychophysics-style distance effect). n is small by design (one row per
    consecutive-generation pair per arm) — reported with a case-bootstrap CI and zero confirmatory
    weight, exactly as registered."""
    ir = _inspect_rewards()
    rows: list[tuple[float, float, float, float, float]] = []  # (dm, dist, mag, iter, plateau)
    for arm in ("distributional", "scalar_cvar5"):
        by_gen: dict[int, dict[str, list[float]]] = {}
        best_fit: dict[int, float] = {}
        for r in records:
            if str(r.get("arm") or "") != arm or not ir._was_fed_tail(r):
                continue
            vec = ir._fed_tail_vector(r)
            if vec is None or vec.size == 0 or not np.isfinite(vec[0]):
                continue
            prevalence = ir._construct_prevalence(ir._reward_source(r))
            m = float(sum(1 for name in ir._TAIL_CONSTRUCTS if prevalence.get(name)))
            g = int(r.get("generation") or 0)
            cell = by_gen.setdefault(g, {"x": [], "m": []})
            cell["x"].append(float(vec[0]))
            cell["m"].append(m)
            fit = (r.get("metrics") or {}).get("val_fitness")
            if fit is not None and np.isfinite(float(fit)):
                best_fit[g] = max(best_fit.get(g, float("-inf")), float(fit))
        gens = sorted(by_gen)
        for g_prev, g_next in zip(gens, gens[1:]):
            if g_next != g_prev + 1:
                continue
            x_prev = float(np.mean(by_gen[g_prev]["x"]))
            x_next = float(np.mean(by_gen[g_next]["x"]))
            dm = float(np.mean(by_gen[g_next]["m"]) - np.mean(by_gen[g_prev]["m"]))
            plateau = 1.0 if (g_prev in best_fit and g_next in best_fit
                              and best_fit[g_next] <= best_fit[g_prev] + 1e-9) else 0.0
            rows.append((dm, abs(x_next - x_prev), abs(x_prev), float(g_prev), plateau))
    if len(rows) < 5:
        return {"status": "no_data", "reason": f"{len(rows)} consecutive-pair rows (need >= 5)",
                "exploratory": True}
    arr = np.asarray(rows, dtype=float)
    y = arr[:, 0]
    X = arr[:, 1:5]
    # Standardise predictors (guarding constant columns) so betas are comparable; add an intercept.
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - X.mean(axis=0)) / sd
    design = np.column_stack([np.ones(len(y)), Xz])

    def _fit(design_m: np.ndarray, y_v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design_m, y_v, rcond=None)
        return beta

    beta = _fit(design, y)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        try:
            boots.append(_fit(design[idx], y[idx])[1])  # the DISTANCE beta
        except np.linalg.LinAlgError:
            continue
    boots_a = np.asarray([b for b in boots if np.isfinite(b)], dtype=float)
    ci_low, ci_high = ((float(np.percentile(boots_a, 2.5)), float(np.percentile(boots_a, 97.5)))
                       if boots_a.size else (float("nan"), float("nan")))
    return {
        "status": "ok",
        "exploratory": True,
        "n": int(len(y)),
        "beta_distance": float(beta[1]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "betas": {"distance": float(beta[1]), "magnitude": float(beta[2]),
                  "iteration": float(beta[3]), "plateau": float(beta[4])},
        "note": "declared EXPLORATORY (§2a(b)) — standardised OLS, case-bootstrap CI, no "
                "confirmatory weight; positive distance beta = the numeracy account's prediction",
    }


def responsiveness_markdown(d: dict[str, Any]) -> str:
    """Render the SQ1 responsiveness coefficient (fed tail -> authored tail-construct count); report-only."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Responsiveness (SQ1; ADR-039) — n/a\n\n{reason}\n"
    verdict = "RESPONSIVE (CI excludes 0)" if d.get("responsive") else "NOT responsive (CI spans 0)"
    out = [
        f"## Responsiveness — does the FED tail move the authored CODE? (SQ1; ADR-039, report-only) — {verdict}",
        "",
        "PRIMARY = the REGISTERED §2a form: Spearman of Δ(fed tail) vs Δ(authored tail-construct count) "
        "over consecutive generations within each tail-FED arm (X = the CVaR-5% the designer was actually "
        "shown, parsed from the archived prompt — the previous generation's best block; 2026-07-05 M13 "
        "construct fix: the old wiring used each candidate's OWN post-training tail, reversing the "
        "estimand). Generation-0 candidates (fed nothing) are excluded. Seeded bootstrap CI. DISJOINT "
        "from the frozen m=6 family — never gates the headline.",
        "",
        f"- n generation-deltas: **{d.get('n', 0)}** ({d.get('method', '?')}; "
        f"{d.get('n_boot_valid', 0)} valid boots)",
        f"- coefficient: **{d.get('coef', float('nan')):+.4f}**  "
        f"(95% CI [{d.get('ci_low', float('nan')):+.4f}, {d.get('ci_high', float('nan')):+.4f}])",
        "",
    ]
    lc = d.get("levels_companion") or {}
    if lc.get("status") == "ok":
        out += [
            f"- per-candidate LEVELS companion (descriptive; x is generation-constant, rows CLUSTERED "
            f"within generation): coef **{lc.get('coef', float('nan')):+.4f}** "
            f"(95% CI [{lc.get('ci_low', float('nan')):+.4f}, {lc.get('ci_high', float('nan')):+.4f}]; "
            f"n={lc.get('n', 0)})",
            "",
        ]
    return "\n".join(out)


def mediation_markdown(d: dict[str, Any]) -> str:
    """Render the fed -> code -> outcome mediation decomposition (report-only; states the endogeneity caveat)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Mediation (fed -> code -> outcome; ADR-039) — n/a\n\n{reason}\n"
    verdict = "indirect effect a·b CI excludes 0" if d.get("mediated") else "indirect effect a·b CI spans 0"
    out = [
        f"## Mediation — fed tail -> authored CODE -> realised tail (ADR-039, report-only) — {verdict}",
        "",
        "Single-mediator linear mediation per candidate: X = fed CVaR-5% summary, M = authored tail-construct "
        "count, Y = realised `cvar(val_returns, 0.05)`. Path **a** (X->M, responsiveness), path **b** (M->Y "
        "given X, transmission), and the bootstrap-CI'd **indirect effect a·b** (Preacher–Hayes). A severed "
        "FIRST link (a≈0 -> a·b≈0) would LOCATE the equivalence at responsiveness. DISJOINT from the frozen "
        "m=6 family.",
        "",
        f"- n candidates: **{d.get('n', 0)}** (standardised={d.get('standardized')}; "
        f"{d.get('n_boot_valid', 0)} valid boots)",
        f"- path a (X->M): **{d.get('a', float('nan')):+.4f}** | path b (M->Y|X): **{d.get('b', float('nan')):+.4f}**",
        f"- total c: **{d.get('c_total', float('nan')):+.4f}** | direct c': **{d.get('c_direct', float('nan')):+.4f}** "
        f"| indirect a·b: **{d.get('indirect', float('nan')):+.4f}** "
        f"(95% CI [{d.get('ci_low', float('nan')):+.4f}, {d.get('ci_high', float('nan')):+.4f}])",
        "",
        "**Honesty (caveat).** Observational mediation is ASSOCIATIONAL: X, M, Y are all read off the SAME "
        "trained candidate, so M is endogenous to the agent it steers (the fed tail is the trained policy's "
        "own realised returns). This is a DESCRIPTIVE decomposition of the mechanism under sequential "
        "ignorability — it never gates a hypothesis (see `src.inference.mediation` docstring).",
        "",
    ]
    return "\n".join(out)


def named_vs_blinded_structural_markdown(d: dict[str, Any]) -> str:
    """One-line disclosure for the named-vs-blinded structural A/B (honest degrade when not run)."""
    if d and d.get("status") == "ok":
        verdict = "data-locked (structure driven by shared data)" if d.get("data_locked") else "identity-driven structure FLAGGED"
        return (
            "## Named-vs-blinded structural A/B (ADR-039; report-only) — ran\n\n"
            f"paired structural sim **{d.get('paired_mean', float('nan')):+.4f}** vs noise floor "
            f"**{d.get('within_blinded_mean', float('nan')):+.4f}** (p_random_pairing="
            f"{d.get('p_random_pairing_matches', float('nan')):.4f}) — {verdict}.\n"
        )
    reason = (d or {}).get("reason", "not run")
    return (
        "## Named-vs-blinded structural A/B (ADR-039; report-only) — NOT RUN\n\n"
        f"_executed={bool((d or {}).get('executed'))}_: {reason}\n"
    )


def legible_format_responsiveness_markdown(d: dict[str, Any]) -> str:
    """One-line disclosure for the legible-format responsiveness differential (honest degrade when not run)."""
    if d and d.get("status") == "ok":
        verdict = "legibility RAISES responsiveness" if d.get("legibility_helps") else "no legibility gain"
        return (
            "## Legible-format responsiveness differential (numeracy bottleneck; ADR-039; report-only) — ran\n\n"
            f"differential (legible − raw) **{d.get('differential', float('nan')):+.4f}** "
            f"(95% CI [{d.get('ci_low', float('nan')):+.4f}, {d.get('ci_high', float('nan')):+.4f}]) — {verdict}.\n"
        )
    reason = (d or {}).get("reason", "not run")
    return (
        "## Legible-format responsiveness differential (numeracy bottleneck; ADR-039; report-only) — NOT RUN\n\n"
        f"_executed={bool((d or {}).get('executed'))}_: {reason}\n"
    )


def information_gap_markdown(d: dict[str, Any]) -> str:
    """Render the information-utilization gap (§2a micro-anchor (d); report-only; honest degrade)."""
    if not d or d.get("status") not in {"ok"}:
        reason = (d or {}).get("reason", "not computed")
        return (
            "## Information-utilization gap (§2a micro-anchor (d); report-only) — n/a\n\n"
            f"_executed={bool((d or {}).get('executed'))}_: {reason}\n"
        )
    out = [
        "## Information-utilization gap (§2a micro-anchor (d); report-only)",
        "",
        "Redundancy of the fed six-component tail vector given the concurrently fed scalar summary, on "
        "the ACTUAL archived fed feedback sequences (one observation per distinct per-generation block). "
        "`fed_rendered` = the values exactly as the LLM saw them (render-precision quantization is part "
        "of the estimand); `fed_underlying` = the same fed observations at full archived precision "
        "(parent-matched). The complement 1 − redundancy is the fed-but-unusable-by-a-scalar "
        "information. DISJOINT from the frozen m=6 family — never gates the headline.",
        "",
        "| arm | channel | n | mean R² [95% CI] | mean rank ρ² [95% CI] | non-redundant (1−ρ²) |",
        "|---|---|---|---|---|---|",
    ]
    for arm, entry in d.get("arms", {}).items():
        for channel, ch in entry.get("channels", {}).items():
            if ch.get("status") != "ok":
                out.append(f"| {arm} | {channel} | — | {ch.get('reason', 'n/a')} | — | — |")
                continue
            p = ch["pooled"]
            flag = " (scalar degenerate at render precision)" if ch.get("scalar_degenerate") else ""
            out.append(
                f"| {arm} | {channel} | {ch['n']} | "
                f"{p['mean_r2']:.3f} [{p['mean_r2_ci_low']:.3f}, {p['mean_r2_ci_high']:.3f}] | "
                f"{p['mean_rank_rho2']:.3f} [{p['mean_rank_ci_low']:.3f}, {p['mean_rank_ci_high']:.3f}] | "
                f"**{p['non_redundant_rank']:.3f}**{flag} |"
            )
    out.append("")
    floor = d.get("floor", {})
    if floor.get("executed"):
        fp = floor["fed_rendered"]["pooled"]
        out.append(
            f"Calibration floor (`{floor.get('arm')}`, destroyed linkage by construction): "
            f"mean R² **{fp['mean_r2']:.3f}**, mean rank ρ² **{fp['mean_rank_rho2']:.3f}** "
            f"(n={floor['fed_rendered']['n']})."
        )
        fc = d.get("floor_comparison", {})
        if fc.get("executed"):
            out.append(
                f"Linkage-attributable redundancy (arm − floor, R²): "
                f"**{fc['linkage_attributable_r2']:+.3f}**."
            )
    else:
        out.append(
            f"Calibration floor — NOT AVAILABLE (_executed=False_): {floor.get('reason', 'n/a')}."
        )
    out.append("")
    ug = d.get("utilization_gap", {})
    if ug.get("executed"):
        ci = ug.get("responsiveness_ci") or [float("nan"), float("nan")]
        out += [
            f"**Utilization gap** ({ug.get('arm')}, `{ug.get('channel')}` channel): GIVEN "
            f"(non-redundant fed fraction) **{ug['non_redundant_fed']:.3f}** vs USED "
            f"(|SQ1 responsiveness|) **{ug['responsiveness_abs_coef']:.3f}** "
            f"(SQ1 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]) → gap **{ug['gap']:+.3f}**. "
            "Descriptive index (two different [0,1]-bounded estimands), never a parameter estimate.",
        ]
    else:
        out.append(f"Utilization gap — NOT COMPUTED (_executed=False_): {ug.get('reason', 'n/a')}.")
    out.append("")
    return "\n".join(out)


def validation_headroom_markdown(d: dict[str, Any]) -> str:
    """Render the validation-headroom (oracle-selection) bound (§2a micro-anchor (e); report-only)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return (
            "## Validation-headroom (oracle-selection) bound (§2a micro-anchor (e); report-only) — n/a\n\n"
            f"_executed={bool((d or {}).get('executed'))}_: {reason}\n"
        )

    def _leg(leg: dict[str, Any] | None) -> str:
        if not leg or "gap" not in leg:
            return "— | — | —"
        return (
            f"{leg['frontier']:+.4f} | {leg['achieved']:+.4f} | "
            f"{leg['gap']:+.4f} [{leg['gap_ci_low']:+.4f}, {leg['gap_ci_high']:+.4f}]"
        )

    out = [
        "## Validation-headroom (oracle-selection) bound (§2a micro-anchor (e); report-only)",
        "",
        f"Oracle frontier = best achievable validation CVaR-{d.get('cvar_level', 0.05):g} / DSR over ALL "
        f"archived candidates vs what the frozen selection — {d.get('selection_rule')} — achieved; a "
        "material gap establishes headroom EXISTED in the authored search space. Bootstrap CI resamples "
        "candidates and re-applies the selection rule. VALIDATION data only — the sealed test leg is "
        "never touched. DISJOINT from the frozen m=6 family — never gates the headline.",
        "",
        "CI is one-sided by construction (gap >= 0 in every resample — a frontier max cannot fall below "
        "the selected candidate); a low bound near 0 means NO headroom, not significance.",
        "",
        "| arm | n | CVaR frontier | achieved | gap [95% CI] | DSR frontier | achieved | gap [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, e in d.get("per_arm", {}).items():
        if e.get("status") != "ok":
            out.append(
                f"| {arm} | {e.get('n_candidates', 0)} | skipped: {e.get('reason', 'n/a')} |  |  |  |  |  |"
            )
            continue
        out.append(f"| {arm} | {e['n_candidates']} | {_leg(e.get('cvar'))} | {_leg(e.get('dsr'))} |")
    pooled = d.get("pooled")
    if pooled:
        out.append(
            f"| **pooled** | {pooled['n_candidates']} | {_leg(pooled.get('cvar'))} | "
            f"{_leg(pooled.get('dsr'))} |"
        )
    out.append("")
    return "\n".join(out)


#: EVERY key ``analyze()`` can place in its result. Two routes reach ``out``, and counting only one
#: of them is the wrong-denominator error in its purest form: 32 ``out["..."] = `` subscript
#: assignments PLUS 7 keys in the annotated ``out: dict[str, Any] = {...}`` initialiser = 39.
#: (An earlier scope note said "the 35 out[...] keys"; 35 is a remembered number and matches
#: neither route nor their union. Flagged 2026-08-01, RUN 11.)
#: This literal is pinned to the source by ``tests/test_analyze_key_registry.py``, which re-derives
#: the set from the AST and fails if the two disagree — so adding a result WITHOUT registering it
#: breaks the build rather than escaping the enumeration.
REGISTERED_OUTPUT_KEYS = frozenset({
    "archive_integrity", "attribution", "bayesian_null_report", "benchmark_floor",
    "comparative_es_backtest", "compute_accounting", "cross_hypothesis_multiplicity",
    "cross_model", "delisting_band", "distance_moderator", "divergence", "dsr_effective_n",
    "evt_consistency", "h1_beat_human", "h2", "h2_rf_robustness", "h2_structure", "h2_tost",
    "h2_tost_dsr", "h3", "h4", "information_gap", "legible_format_responsiveness",
    "mechanism_multiplicity", "mediation", "model_confidence_set", "n_blocks", "n_records",
    "named_vs_blinded_structural", "pbo", "pbo_dsr", "regime_stratified", "responsiveness",
    "responsiveness_by_arm", "reward_taxonomy", "validation_headroom", "validity_tier",
    "variance", "winner_dsr",
})

#: Registered keys that are LEGITIMATELY absent when a stated precondition is not supplied, mapped
#: to that precondition. Everything else missing is a DEFECT, not a configuration.
#: NOTE: The four ``campaign_summary.json`` dependants are the reason this registry exists. They share
#: one cause — ``main()`` populates panel/cfg/test_window/winner_n_trials only from that file — and
#: until 2026-08-01 the skip message named only the benchmark floor, so a reader who SAW the warning
#: still did not learn that three further registered outputs had vanished with it.
CONDITIONAL_OUTPUT_KEYS = {
    "variance": "--variance-runs",
    "benchmark_floor": "campaign_summary.json at the archive root (supplies test_window + panel)",
    "attribution": "campaign_summary.json at the archive root (supplies test_window + panel)",
    "h2_rf_robustness": "campaign_summary.json at the archive root (supplies test_window + panel)",
    "regime_stratified": "campaign_summary.json at the archive root (supplies test_window + panel)",
}


def missing_output_keys(result: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Split the registered keys absent from ``result`` into (explained, UNEXPLAINED).

    Returns ``({key: precondition}, [key, ...])``. The second list is the one that matters: a
    registered output missing with no stated precondition is a defect in the analysis, and the
    caller exits non-zero on it. Reporting the first list is not optional either — an explained
    absence is still an absence, and four of them shipped unnoticed once already.
    """
    absent = sorted(REGISTERED_OUTPUT_KEYS - set(result))
    explained = {k: CONDITIONAL_OUTPUT_KEYS[k] for k in absent if k in CONDITIONAL_OUTPUT_KEYS}
    unexplained = [k for k in absent if k not in CONDITIONAL_OUTPUT_KEYS]
    return explained, unexplained


def analyze(
    root: str | Path,
    *,
    n_blocks: int | None = None,
    panel: Any = None,
    cfg: Any = None,
    test_window: tuple[int, int] | None = None,
    winner_n_trials: int | None = None,
    variance_run_roots: list[str] | None = None,
    single_shot_root: str | Path | None = None,
    named_blinded_root: str | Path | None = None,
    legible_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a campaign archive and compute the per-arm PBO table + headline winner DSR + H2.

    When ``panel``/``cfg``/``test_window`` are supplied (the production analysis path), ALSO runs the
    panel-dependent **DeMiguel benchmark floor** + market reference (PREREGISTRATION §9/§10). Records-only
    when they are absent (the default, e.g. unit tests). The floor was previously implemented + unit-tested
    but invoked by NO entry point — the same dead-component defect already fixed for H2 (critical-review #2).

    ADDITIVE report-only secondaries (DISJOINT ``out[...]`` keys; NEVER the frozen m=6 H2 family): ``h4``
    (DEEP_H4 LLM-vs-search difference tests), ``h3`` (DEEP_H3 iterative-vs-single-shot — needs
    ``single_shot_root``, gracefully skipped when absent), ``h2_tost`` (DEEP_H2 §5.3 equivalence bounds),
    ``comparative_es_backtest`` (FZ0 + Diebold-Mariano; corroborates H2-Tail, CH4 §4.7),
    ``bayesian_null_report`` (BF01 + posterior-in-ROPE complement to the TOST, R67),
    ``model_confidence_set`` (Hansen et al. 2011 indistinguishable-arm set; needs the optional ``arch`` dep,
    degrades to ``status="error"`` when absent), ``dsr_effective_n`` (DEEP_STATS A1), ``evt_consistency``
    (DEEP_H2 §6.3), ``mechanism_multiplicity`` (Bonferroni/BH-across-mechanism-legs sensitivity),
    ``reward_taxonomy`` (program KINDS induced from the pooled authored sources + per-arm composition;
    ``src.inference.reward_taxonomy``), and ``cross_hypothesis_multiplicity`` (DEEP_STATS A4
    Bonferroni-across-4 sensitivity). All panel-independent except where noted; each degrades to a
    ``status="skipped"`` block when its data is absent.
    """
    from src.utils.config import load_config

    inf = load_config("inference")
    if n_blocks is None:
        n_blocks = int(inf.get("pbo", {}).get("n_blocks", 16))
    # READ the headline FDR level + CVaR tail level from config (the single source of truth) rather than
    # relying on the hardcoded function defaults — the wired analysis path previously never passed them, so
    # the frozen q/level lived only as a literal (no-hardcoding audit 2026-06-20).
    mult = inf.get("multiplicity", {}) if isinstance(inf.get("multiplicity"), dict) else {}
    q_level = float(mult.get("q", 0.05))  # BH FDR level (config/inference.yaml: multiplicity.q)
    # The FROZEN testing family (incl. its headline CVaR tail levels) lives in preregistration.yaml, NOT
    # inference.yaml — read it from THERE so cvar_levels is the frozen value, not a silent (0.05,) fallback.
    _prereg_inf = load_config("preregistration").get("inference", {})
    fam = _prereg_inf.get("testing_family", {}) if isinstance(_prereg_inf.get("testing_family"), dict) else {}
    cvar_levels = tuple(float(x) for x in fam.get("cvar_levels", [0.05])) or (0.05,)
    # R25: the headline two co-primary IUTs decide each leg ONE-SIDED at this frozen alpha (no leg
    # correction). Read it from the frozen testing_family (single source of truth), not a literal.
    alpha_one_sided = float(fam.get("alpha_one_sided", 0.05))

    # VERIFY-BEFORE-TRUST (2026-07-05): if the run sealed an archive-integrity manifest, re-verify the
    # live archive against it BEFORE trusting any number — a modified/dropped/added record between the
    # run and analysis is caught here, not silently averaged into a result. Report-only + best-effort
    # (an absent manifest, e.g. a unit-test archive, is simply skipped); a genuine MISMATCH is surfaced
    # loudly under out["archive_integrity"] so it cannot pass unnoticed.
    archive_integrity = {"status": "not_sealed", "reason": "no archive_integrity.json manifest present"}
    try:
        try:
            from scripts.archive_integrity import verify_manifest
        except ModuleNotFoundError:  # direct `python scripts/...` puts scripts/ on sys.path, NOT the
            import sys as _sys        # repo root, so `scripts` is not an importable package (the SAME
            _sys.path.insert(0, str(Path(__file__).resolve().parent))  # class as the run_campaign seal
            from archive_integrity import verify_manifest              # bug, 2026-07-24) -> without this
            # fallback the seal is written but NEVER VERIFIED at analysis time (the provenance chain is
            # broken at the read end for BOTH the local and the pulled-cluster archive).

        manifest = Path(root) / "archive_integrity.json"
        if manifest.is_file():
            vr = verify_manifest(root)
            archive_integrity = {
                "status": "ok" if vr.ok else "MISMATCH", "root": vr.actual_root,
                "sealed_root": vr.expected_root, "n_verified": vr.n_verified,
                "changed": vr.changed[:20], "removed": vr.removed[:20], "added": vr.added[:20],
            }
            if not vr.ok:
                print(f"[analyze_campaign] ARCHIVE INTEGRITY MISMATCH: {vr.summary()}", flush=True)
    except Exception as exc:  # noqa: BLE001 — the integrity check must never itself break analysis
        archive_integrity = {"status": "error", "reason": str(exc)[:200]}

    records = load_campaign_records(root)
    results = campaign_pbo(records, n_blocks=n_blocks, rng=np.random.default_rng(0))
    # M3 (R36): a SECOND PBO ranked on per-block annualised Sharpe — the DSR-proxy statistic winner
    # SELECTION used (validation DSR; monotone in Sharpe at lambda=0) — reported ALONGSIDE the frozen
    # mean-return PBO (`results`, UNCHANGED). Agreement empirically closes the DEEP_STATS A3 "guard the rule
    # you used" concern. Report-only; a failure must never break the headline PBO table.
    try:
        results_dsr = campaign_pbo_dsr(records, n_blocks=n_blocks, rng=np.random.default_rng(0))
    except Exception as exc:  # noqa: BLE001 - a report-only second PBO must never break the primary
        results_dsr = {"status": "error", "reason": str(exc)[:200]}  # type: ignore[dict-item]
    dsr = winner_dsr(records)  # Rank 16: canonical cross-trial-variance headline DSR
    # The pre-registered HEADLINE H2 — TWO co-primary IUTs (H2-RA on Sharpe + H2-Tail on CVaR-5%),
    # each one-sided at alpha with NO leg correction (R25; DEEP_H2 §7.1) — on the held-out TEST leg,
    # per-seed rliable inference, with the fail-loud family-equals-frozen assert fired inside
    # collect_family_pvalues (#18, 2026-06-20). The BH-over-6 set is reported as a sensitivity only.
    try:
        h2 = h2_conjunction(
            records, alpha=alpha_one_sided, q=q_level, cvar_levels=cvar_levels,
            rng=np.random.default_rng(0),
        )
    except AssertionError as exc:  # realized family != frozen (R13/R25) -> surface, don't bury
        h2 = {"H2_supported": None, "error": f"frozen-family assertion failed: {exc}"}

    out: dict[str, Any] = {
        "n_blocks": int(n_blocks),
        "n_records": len(records),
        "archive_integrity": archive_integrity,  # verify-before-trust seal check (2026-07-05)
        "pbo": results,
        "pbo_dsr": results_dsr,  # M3 (R36): second PBO ranked on the DSR-proxy (per-block Sharpe) statistic
        "winner_dsr": dsr,
        "h2": h2,
    }

    _head = H2_CONTRASTS[0][0] if H2_CONTRASTS else "distributional"

    # H4 (DEEP_H4; PREREGISTRATION §1/§3) — the LLM winner vs random_search + bayes_opt on the sealed leg,
    # per-seed IQM paired bootstrap mirroring H2-RA, as TWO one-sided tests with the 2-test multiplicity.
    # DISJOINT out["h4"] (no arm_a/arm_b/metric/level), panel-independent, graceful skip when search winners
    # have no test records.
    try:
        out["h4"] = h4_search_controls(
            records, winner_arm=_head, alpha=alpha_one_sided, rng=np.random.default_rng(0)
        )
    except Exception as exc:  # noqa: BLE001 - a report-only secondary must never break the headline
        out["h4"] = {"status": "error", "reason": str(exc)[:200]}

    # CROSS-MODEL SYNTHESIS (R86/R101) — the registered "across every model we tested, the effect is
    # at most X" statement. WIRED 2026-07-26: `cross_model` and `leg_aggregate` were both built and
    # unit-tested but had NO production caller, so the claim had no executable route to a number and
    # would have had to be withdrawn AFTER the campaign's compute was spent (the R16 failure
    # repeating). The helper reports `no_leg_archives` / `no_core_baseline` / `no_legs_passed_floor`
    # as DISTINCT states, so an empty synthesis can never be mistaken for a null effect.
    try:
        from src.inference.leg_aggregate import cross_model_synthesis

        out["cross_model"] = cross_model_synthesis(root)
    except Exception as exc:  # noqa: BLE001 - never break the headline analysis
        out["cross_model"] = {"status": "error", "reason": str(exc)[:200]}

    # H3 (DEEP_H3; PREREGISTRATION §1/§6) — iterative reflection vs single-shot best-of-N, per-seed IQM
    # paired bootstrap + a TOST equivalence (±0.05). The single-shot condition is a SEPARATE run archived
    # under single_shot_root (D-2: test_h3_singleshot/<arm>); graceful skip when absent. DISJOINT out["h3"].
    ss_records: list[dict[str, Any]] | None = None
    if single_shot_root is not None and Path(single_shot_root).is_dir():
        try:
            ss_records = load_campaign_records(single_shot_root)
        except Exception:  # noqa: BLE001 - a missing/garbled single-shot archive => H3 skips, never breaks
            ss_records = None
    try:
        out["h3"] = h3_iterative_vs_singleshot(
            records, ss_records, arm=_head, alpha=alpha_one_sided, rng=np.random.default_rng(0)
        )
    except Exception as exc:  # noqa: BLE001
        out["h3"] = {"status": "error", "reason": str(exc)[:200]}

    # Headline TOST (DEEP_H2 §5.3) — equivalence bounds for the H2-RA + H2-Tail per-seed IQM differences,
    # ±0.05 in the test-statistic's units. The bankable-null complement to h2_conjunction. DISJOINT.
    try:
        out["h2_tost"] = h2_tost(
            records, cvar_level=headline_cvar_level(cvar_levels),
            rng=np.random.default_rng(0),
        )
    except Exception as exc:  # noqa: BLE001
        out["h2_tost"] = {"status": "error", "reason": str(exc)[:200]}

    # DSR-units companion TOST — the SAME bankable-null equivalence evaluated
    # in the SESOI's OWN validation-DSR units (±0.05), via the conservative Sharpe->DSR ceiling. Without it a
    # campaign non-rejection is at most INCONCLUSIVE for the DSR-equivalence claim the power doc requires;
    # this computes that leg. RA-only (DSR has no CVaR analogue), report-only, DISJOINT.
    try:
        out["h2_tost_dsr"] = h2_tost_dsr(records, rng=np.random.default_rng(0))
    except Exception as exc:  # noqa: BLE001
        out["h2_tost_dsr"] = {"status": "error", "reason": str(exc)[:200]}

    # Comparative ES backtest (FZ0 + Diebold-Mariano; DEEP_H2; CH4 §4.7) — CORROBORATES H2-Tail. Built +
    # unit-tested (src.inference.es_backtest.comparative_es_backtest) but previously invoked by NO entry
    # point, though CH4 §4.7 CLAIMS it corroborates the tail result. Panel-INDEPENDENT (distributional TEST
    # realized series + pooled per-arm val (VaR,ES) forecasts). Report-only, DISJOINT; graceful skip absent data.
    try:
        out["comparative_es_backtest"] = comparative_es_backtest_report(
            records, cvar_level=headline_cvar_level(cvar_levels), rng=np.random.default_rng(0)
        )
    except Exception as exc:  # noqa: BLE001 - a report-only corroboration must never break the headline
        out["comparative_es_backtest"] = {"status": "error", "reason": str(exc)[:200]}

    # Bayesian evidence-for-the-null (R67; DEEP_H2 §5.4) — BF01 + posterior-in-ROPE on the PAIRED per-seed
    # differences (ROPE = frozen SESOI), for BOTH co-primary metrics. The Bayes complement to the frozen TOST.
    # Built + unit-tested (src.inference.bayes_null) but never wired. Report-only, DISJOINT; graceful skip.
    try:
        out["bayesian_null_report"] = bayesian_null_report_block(
            records, cvar_level=headline_cvar_level(cvar_levels), rng=np.random.default_rng(0)
        )
    except Exception as exc:  # noqa: BLE001 - a report-only Bayesian lens must never break the headline
        out["bayesian_null_report"] = {"status": "error", "reason": str(exc)[:200]}

    # Model Confidence Set (Hansen et al. 2011; DEEP_STATS) — the multiplicity-honest "which arms are
    # INDISTINGUISHABLE" set over the arms, for both co-primary metrics. Built + unit-tested
    # (src.inference.model_confidence_set) but never wired; needs the optional `arch` dependency, so it
    # degrades to status="error" (this guard) when absent. Report-only, DISJOINT; never gates.
    try:
        out["model_confidence_set"] = model_confidence_set_report(
            records, cvar_level=headline_cvar_level(cvar_levels), seed=0
        )
    except Exception as exc:  # noqa: BLE001 - a report-only MCS (arch-optional) must never break the headline
        out["model_confidence_set"] = {"status": "error", "reason": str(exc)[:200], "executed": False}

    # Structure-vs-content control (R32) — distributional vs placebo_shuffled (FORMAT + MARGINALS matched,
    # the coherent tail SHAPE deranged). Directly answers the Gupta-Hartford format-vs-content threat.
    # Reported, DISJOINT from the frozen m=6 union, NEVER a gate.
    try:
        out["h2_structure"] = h2_structure_control(
            records, cvar_level=headline_cvar_level(cvar_levels),
            rng=np.random.default_rng(0),
        )
    except Exception as exc:  # noqa: BLE001
        out["h2_structure"] = {"status": "error", "reason": str(exc)[:200]}

    # Delisting-return sensitivity band (R33; PREREGISTRATION §7) — DATA-level (loads the panel + the Shumway
    # audit log directly; pinned to DELISTING_BAND_AUDIT_SUFFIX=univ4, independent of gold_suffix() —
    # the headline is a zero-fill build: post-R44 univ3, univ5 post-Split-C), reprices
    # ONLY the 333 test-window delisting cells across d∈{0,−0.30,−0.55,−1.00} and recomputes the POOLED
    # CVaR-5%/1%. No policy re-run; report-only; DISJOINT (no family-tuple keys). Gracefully skips when the
    # gold/clean parquets are absent (records-only / synthetic install).
    try:
        out["delisting_band"] = delisting_band()
    except Exception as exc:  # noqa: BLE001 - a report-only data band must never break the headline
        out["delisting_band"] = {"status": "error", "reason": str(exc)[:200]}

    # DSR effective-N sensitivity (DEEP_STATS A1) — winner DSR at raw N vs an effective-N proxy for the
    # sequential-reflective candidate correlation; report-only, benign-direction. DISJOINT.
    try:
        out["dsr_effective_n"] = dsr_effective_n(records, winner_arm=_head)
    except Exception as exc:  # noqa: BLE001
        out["dsr_effective_n"] = {"status": "error", "reason": str(exc)[:200]}

    # EVT-consistency guard (DEEP_H2 §6.3) — assert/log the fed CVaR-5%/1% used a CONSISTENT estimator
    # (the alpha>fu / degenerate-fit fallback in _evt_cvar) across the tail-fed arms. Report-only. DISJOINT.
    try:
        out["evt_consistency"] = evt_consistency_guard(records)
    except Exception as exc:  # noqa: BLE001
        out["evt_consistency"] = {"status": "error", "reason": str(exc)[:200]}

    # Training-divergence diagnostic (R34) — READ the existing anomalies.jsonl (the trainer is NOT touched),
    # cluster critic_explosion LINES into the true diverged-RUN count + rate, and flag whether any frozen
    # WINNER's training diverged. Report-only; DISJOINT (no family-tuple keys). The winner ids come from the
    # canonical-DSR table (per-arm winner candidate_id). Gracefully skips when no anomalies.jsonl exists.
    try:
        _winner_ids = tuple(
            str(e["winner_id"]) for e in dsr.values()
            if isinstance(e, dict) and e.get("winner_id")
        ) if isinstance(dsr, dict) else ()
        out["divergence"] = divergence_report(root, winner_ids=_winner_ids)
    except Exception as exc:  # noqa: BLE001 - a report-only divergence read must never break the headline
        out["divergence"] = {"status": "error", "reason": str(exc)[:200]}

    # Compute-accounting (R35) — per-arm candidates attempted/accepted/failed + prompt-tokens from the
    # archived failures.jsonl + llm_calls.jsonl. Discloses the failure asymmetry (search resamples to a full
    # slate; LLM arms burn the slot → conservative for H2) + the token asymmetry (controlled by placebo).
    # Report-only; DISJOINT. Gracefully reports zeros when the provenance files are absent.
    try:
        out["compute_accounting"] = compute_accounting(records, root)
    except Exception as exc:  # noqa: BLE001 - a report-only accounting read must never break the headline
        out["compute_accounting"] = {"status": "error", "reason": str(exc)[:200]}

    # MECHANISM kernel (ADR-039) — report-only, DISJOINT from the frozen m=6 family; NEVER gates the
    # headline. Panel-INDEPENDENT (reads the archived per-candidate tail_stats + reward_source + val_returns
    # of the TAIL-FED arms). Each wrapped so a failure degrades to status="error" and never breaks analyze().
    #
    # SQ1 responsiveness — does the FED CVaR-5% summary (X, parsed from the archived prompt) move the
    # count of tail-shaped constructs the program writes (M)? 2026-07-05 (M13): the REGISTERED §2a form
    # ("Spearman of Δ(fed tail) vs Δ(authored-reward feature)", per-generation deltas within arm) is the
    # PRIMARY statistic; the per-candidate LEVEL association is reported alongside as a descriptive
    # companion (x is generation-constant, so its rows are clustered — disclosed in the markdown).
    try:
        from src.inference.responsiveness import responsiveness

        _pairs = _mechanism_pairs(records)
        _resp = responsiveness(_pairs["dx"], _pairs["dm"], rng=np.random.default_rng(0))
        _resp["form"] = "registered_generation_deltas"
        _resp["levels_companion"] = responsiveness(
            _pairs["x"], _pairs["m"], rng=np.random.default_rng(0)
        )
        _resp["levels_companion"]["form"] = "per_candidate_levels_clustered"
        out["responsiveness"] = _resp
    except Exception as exc:  # noqa: BLE001 - a report-only mechanism leg must never break the headline
        out["responsiveness"] = {"status": "error", "reason": str(exc)[:200]}

    # §2a(f) fingerprint rows (S9, 2026-07-06) — the per-arm responsiveness table the frozen
    # adjudication matrix registers (scalar-arm own-scalar row = the A4 discriminator;
    # placebo_shuffled = the floor row) + the §2a(b) declared-exploratory distance moderator.
    # Report-only, DISJOINT, never gate.
    try:
        out["responsiveness_by_arm"] = fingerprint_responsiveness_rows(records)
    except Exception as exc:  # noqa: BLE001
        out["responsiveness_by_arm"] = {"status": "error", "reason": str(exc)[:200]}
    try:
        out["distance_moderator"] = distance_moderator_exploratory(records)
    except Exception as exc:  # noqa: BLE001
        out["distance_moderator"] = {"status": "error", "reason": str(exc)[:200]}

    # Mediation — fed tail (X) -> authored CODE (M) -> realised tail outcome Y = cvar(val_returns, 0.05).
    # The VAL-RETURNS proxy path (fully archived, better powered than the sparse test-winner path). A severed
    # FIRST link (a≈0 -> a·b≈0) LOCATES the equivalence at responsiveness. Endogeneity caveat in the markdown.
    try:
        from src.inference.mediation import mediation_analysis

        _pairs = _mechanism_pairs(records)
        out["mediation"] = mediation_analysis(
            _pairs["x"], _pairs["m"], _pairs["y"], rng=np.random.default_rng(0)
        )
    except Exception as exc:  # noqa: BLE001 - a report-only mechanism leg must never break the headline
        out["mediation"] = {"status": "error", "reason": str(exc)[:200]}

    # Named-vs-blinded structural A/B + legible-format responsiveness differential (ADR-039; PREREGISTRATION
    # §2a). These require a SEPARATE re-authoring pass (NAMED labelling / legible rendering) that produces
    # PAIRED reward sources — code that DOES NOT EXIST in the confirmatory run. We DISCLOSE the omission
    # honestly (mirroring contamination.cross_model_disagreement's executed=False pattern); we NEVER fabricate
    # by pairing blinded-vs-blinded. If a caller supplies a re-authoring archive root, it would feed the real
    # function; absent it (the default), the leg degrades.
    out["named_vs_blinded_structural"] = {
        "status": "no_data",
        "executed": False,
        "reason": (
            "named-vs-blinded A/B not run (no NAMED authoring pass produces paired sources); registered in "
            "PREREGISTRATION §2a as a planned sub-experiment"
        ),
    }
    if named_blinded_root is not None and Path(named_blinded_root).is_dir():
        try:
            from src.inference.contamination import named_vs_blinded_structural

            ir = _inspect_rewards()
            ab = load_campaign_records(named_blinded_root)
            named = [ir._reward_source(r) for r in ir._by_generation(ab) if r.get("label") == "named"]
            blinded = [ir._reward_source(r) for r in ir._by_generation(ab) if r.get("label") == "blinded"]
            if named and blinded and len(named) == len(blinded):
                out["named_vs_blinded_structural"] = named_vs_blinded_structural(
                    named, blinded, rng=np.random.default_rng(0)
                )
        except Exception as exc:  # noqa: BLE001 - never break the headline; degrade to the disclosure above
            out["named_vs_blinded_structural"] = {
                "status": "error", "executed": False, "reason": str(exc)[:200]
            }

    out["legible_format_responsiveness"] = {
        "status": "no_data",
        "executed": False,
        "reason": (
            "legible-format responsiveness differential not run (no legible re-rendering authoring pass "
            "produces paired sources); registered in PREREGISTRATION §2a as a planned sub-experiment"
        ),
    }
    if legible_root is not None and Path(legible_root).is_dir():
        try:
            from src.inference.responsiveness import legible_format_responsiveness_differential

            leg = load_campaign_records(legible_root)
            # S8 (2026-07-06): aggregate to ONE row PER (condition, seed) CELL before the
            # differential. The redesigned sub-experiment injects ONE per-seed reference block, so x
            # is CONSTANT within a cell and the K authored candidates are CLUSTERED — i.i.d. row
            # resampling of candidate rows understates the CI width by up to ~sqrt(K), letting the
            # registered `legibility_helps = ci_low > 0` decision fire spuriously. Collapsing to
            # per-seed cells (x, mean m) makes the bootstrap's exchangeable unit the SEED — the true
            # independent unit — at the honest cost of a smaller n (disclosed in the output).
            lx, lm = _condition_seed_cells([r for r in leg if r.get("condition") == "legible"])
            rx, rm = _condition_seed_cells([r for r in leg if r.get("condition") == "raw"])
            if lx.size and rx.size:
                res = legible_format_responsiveness_differential(
                    lx, lm, rx, rm, rng=np.random.default_rng(0)
                )
                res["clustering"] = (
                    "one row per (condition, seed) cell — x is cell-constant by design; candidate "
                    "rows within a cell are clustered, so the seed is the bootstrap unit (S8)"
                )
                out["legible_format_responsiveness"] = res
        except Exception as exc:  # noqa: BLE001 - never break the headline; degrade to the disclosure above
            out["legible_format_responsiveness"] = {
                "status": "error", "executed": False, "reason": str(exc)[:200]
            }

    # Reward-program TAXONOMY (the CH7 "left to future work" instrument, delivered) — induce program
    # KINDS over the pooled authored reward sources (identifier-invariant AST shape-sets -> Jaccard graph
    # -> connected components; src.inference.reward_taxonomy), then the per-arm KIND composition: do
    # different feedback arms author different KINDS of programs, or the same kinds reshaped? Report-only,
    # DISJOINT from the frozen m=6 family (no arm_a/arm_b/metric/level keys); NEVER gates H1-H4. Sources
    # are deduped per (arm, candidate_id) in (generation, candidate_id) order, so a frozen winner whose
    # source is re-archived by per-seed TEST records cannot inflate its kind. Unparseable/empty sources
    # are excluded + counted inside the module (P7c); records with NO archived source (e.g. the per-seed
    # test re-runs) are counted here as n_missing_source. Graceful no_data on a source-free archive.
    try:
        from src.inference.reward_taxonomy import (
            pool_sources,
            taxonomy_by_arm,
            taxonomy_threshold_sensitivity,
        )

        ir = _inspect_rewards()
        sources_by_arm: dict[str, dict[str, str]] = {}
        n_missing_source = 0
        for arm in ARMS:
            per_arm: dict[str, str] = {}
            for r in ir._by_generation([r for r in records if r.get("arm") == arm]):
                cid = str(r.get("candidate_id") or r.get("run_id") or "")
                src_text = ir._reward_source(r)
                if not cid or not src_text.strip():
                    n_missing_source += 1
                    continue
                per_arm.setdefault(cid, src_text)  # first occurrence wins (search leg precedes re-tests)
            if per_arm:
                sources_by_arm[arm] = per_arm
        if not sources_by_arm:
            out["reward_taxonomy"] = {
                "status": "no_data",
                "reason": "no record carries a non-empty reward_source (records-only archive)",
                "n_missing_source": n_missing_source,
            }
        else:
            tax = taxonomy_by_arm(sources_by_arm)
            tax["sensitivity"] = taxonomy_threshold_sensitivity(pool_sources(sources_by_arm))
            tax["n_missing_source"] = n_missing_source
            out["reward_taxonomy"] = tax
    except Exception as exc:  # noqa: BLE001 - a report-only taxonomy must never break the headline
        out["reward_taxonomy"] = {"status": "error", "reason": str(exc)[:200]}

    # Information-utilization gap (PREREGISTRATION §2a micro-anchor (d)) — the redundancy of the fed
    # six-component tail vector given the concurrently fed scalar summary, on the ACTUAL archived fed
    # feedback sequences (deduped per generation; placebo_shuffled = the destroyed-linkage calibration
    # floor), plus the GIVEN-vs-USED gap against the SQ1 responsiveness estimate computed above (never
    # recomputed inside the module). Report-only, DISJOINT (no family-tuple keys); NEVER gates H1-H4.
    try:
        from src.inference.information_gap import information_gap

        out["information_gap"] = information_gap(
            records,
            responsiveness=out.get("responsiveness"),
            rng=np.random.default_rng(0),
        )
    except Exception as exc:  # noqa: BLE001 - a report-only mechanism leg must never break the headline
        out["information_gap"] = {"status": "error", "reason": str(exc)[:200]}

    # Validation-headroom (oracle-selection) bound (PREREGISTRATION §2a micro-anchor (e)) — the oracle
    # frontier (best achievable validation CVaR-5% + DSR over ALL archived candidates, reusing the
    # canonical winner_dsr conventions) vs what each arm's frozen lambda=0 selection achieved, with a
    # candidate-resampling bootstrap CI on the gap. VALIDATION data only (test-leg records are excluded
    # by the module's fail-safe). Report-only, DISJOINT; NEVER gates H1-H4.
    try:
        from src.inference.headroom import validation_headroom

        out["validation_headroom"] = validation_headroom(
            [r for r in records if _is_search_candidate(r)],
            arms=ARMS,
            cvar_level=headline_cvar_level(cvar_levels),
            rng=np.random.default_rng(0),
        )
    except Exception as exc:  # noqa: BLE001 - a report-only mechanism leg must never break the headline
        out["validation_headroom"] = {"status": "error", "reason": str(exc)[:200]}

    # H1 — the Eureka-style "beat-the-human" metric (PREREGISTRATION §1 H1 / §9 hand-reward panel; Ma et al.
    # 2024). The "§18-19" cite was WRONG ("18-19" were line numbers given in error; the prereg has only 12
    # sections — DEEP_H1 C-1). PANEL-INDEPENDENT
    # (reads the realized per-seed TEST returns of the LLM winner arm + each baseline_<name> arm, NOT the
    # panel), so it runs in records-only analysis too — degrading to status="skipped" when the baseline
    # stage was not run. POST-FREEZE, report-only: it writes the DISJOINT out["h1_beat_human"] key (no
    # arm_a/arm_b/metric/level), so the frozen m=6 family + assert_realized_family_matches_frozen are
    # untouched. The baseline NAMES are READ from config (h1_baselines; no hardcoding), and the
    # LLM winner DSR is deflated by its SEARCHED multiplicity (the headline arm's n_trials), baselines N=1.
    _head_h1 = H2_CONTRASTS[0][0] if H2_CONTRASTS else "distributional"
    _h1_names = [str(b) for b in load_config("campaign").get("h1_baselines", [])]
    _h1_wnt = dsr.get(_head_h1, {}).get("n_trials") if isinstance(dsr, dict) else None
    if _h1_names:
        try:
            out["h1_beat_human"] = beat_human_baseline(
                records,
                baseline_names=_h1_names,
                winner_arm=_head_h1,
                winner_n_trials=int(_h1_wnt) if _h1_wnt else winner_n_trials,
            )
        except Exception as exc:  # noqa: BLE001 - a report-only H1 panel must never break the analysis
            out["h1_beat_human"] = {"status": "error", "reason": str(exc)[:200]}
    else:
        out["h1_beat_human"] = {"status": "skipped", "reason": "config/campaign.yaml: h1_baselines is empty"}

    # DeMiguel benchmark floor + market reference (panel-dependent). The floor gate uses the HEADLINE
    # (distributional) arm's seed-MEAN test path as the representative winner (a secondary descriptive
    # gate, distinct from the per-seed H2 inference), deflated by the search budget (winner_n_trials).
    if panel is not None and cfg is not None and test_window is not None:
        head = H2_CONTRASTS[0][0] if H2_CONTRASTS else "distributional"
        vecs = [v for r in records if r.get("arm") == head and (v := _test_returns(r)) is not None]
        winner_ret = None
        if vecs:
            mlen = min(int(v.size) for v in vecs)
            winner_ret = np.mean(np.stack([np.asarray(v)[:mlen] for v in vecs]), axis=0)
        # Deflate the floor's winner DSR by the SAME searched-candidate multiplicity the headline winner
        # DSR uses (the per-arm count derived from the records by `winner_dsr`), so the two gates are
        # consistent regardless of which archive (prototype 40 / campaign 30) is analysed; fall back to
        # the caller-supplied value when the headline arm has no DSR entry.
        head_n = dsr.get(head, {}).get("n_trials") if isinstance(dsr, dict) else None
        wnt = int(head_n) if head_n else winner_n_trials
        try:
            out["benchmark_floor"] = benchmark_floor(
                panel, cfg, (int(test_window[0]), int(test_window[1])),
                winner_test_returns=winner_ret,        # seed-mean: representative path for beta/alpha only
                winner_test_returns_per_seed=vecs,     # per-seed: the GATE uses median-per-seed DSR (#1)
                n_trials=1, winner_n_trials=wnt,
            )
        except Exception as exc:  # noqa: BLE001 - a reporting floor must not break records-only analysis
            out["benchmark_floor"] = {"error": str(exc)}

        # R20 (additive): does the H2 SHARPE conjunction SURVIVE on EXCESS returns (r - DGS3MO)? The frozen
        # rf=0 headline (out["h2"]) is UNCHANGED; this is the reported sensitivity. The per-period rf for
        # the test window aligns to each per-seed test series (both start at test_start).
        try:
            from src.data.market_reference import load_risk_free_daily

            win_dates = np.asarray(panel.dates)[int(test_window[0]):int(test_window[1])]
            rf = load_risk_free_daily(win_dates)
            if rf.available:
                out["h2_rf_robustness"] = h2_sharpe_rf_robustness(
                    records, rf.daily, rng=np.random.default_rng(0)
                )
        except Exception as exc:  # noqa: BLE001 - robustness panel must not break the analysis
            out["h2_rf_robustness"] = {"error": str(exc)}

        # Door-C secondary: factor attribution / difference-in-alpha (SEPARATE declared family; NOT m=6).
        # Reads each (arm, seed) winner's metrics['test_returns']; aligns the on-disk FF3+Mom factors to
        # the test-leg dates. Disjoint keys (rung/factor/alpha_diff/seed) -> the frozen-family assert is
        # untouched; the BH runs INSIDE its own family. Skipped (records-only) when no factor data aligns.
        try:
            from src.data.market_reference import load_risk_free_daily
            from src.inference.attribution import campaign_attribution, load_factor_panel

            attr_dates = np.asarray(panel.dates)[int(test_window[0]):int(test_window[1])]
            fp = load_factor_panel(attr_dates)
            rf_attr = load_risk_free_daily(attr_dates)
            out["attribution"] = campaign_attribution(
                records,
                fp["factors"] if fp.get("available") else None,
                risk_free=(rf_attr.daily if rf_attr.available else None),
                q=q_level,
                n_boot=int(inf.get("sharpe_test", {}).get("n_boot", 2000)),
                rng=np.random.default_rng(0),
            )
        except Exception as exc:  # noqa: BLE001 - a reporting secondary must never break the analysis
            out["attribution"] = {"status": "error", "error": str(exc)}

        # Regime-stratified tail/return metrics (T3'; ADR-039) — per-arm seed-averaged TEST returns
        # conditioned on VIX-threshold regime (calm/normal/stress). Report-only, DISJOINT; descriptive
        # (the independent-EPISODE count, not the per-date count, bounds regime-conditional power). The test
        # series is an UNINDEXED contiguous-from-test_start vector, so we slice the regime labels to the same
        # [test_window[0]:test_window[1]] window and ASSERT length alignment; on mismatch we SKIP (never
        # stratify a misaligned series). DISJOINT out["regime_stratified"].
        try:
            from src.inference.regime_analysis import regime_stratified_metrics
            from src.regimes.definition import label_regimes

            returns_by_arm: dict[str, np.ndarray] = {}
            for arm in ARMS:
                # 2026-07-06: stratify the MEDIAN-TAIL-SEED realized path, NOT the seed-averaged
                # series — averaging ~30 seed paths shrinks the tail ~sqrt(n), so per-regime CVaR-5%
                # on the mean path is systematically thin (the exact aggregation the repo's own ES-
                # backtest doctrine forbids for tail statistics). A genuine single realized path
                # preserves the tail structure; cross-arm comparison stays common-mode.
                realized, _seed = _arm_median_tail_seed_test_returns(records, arm, 0.05)
                if realized is not None and np.asarray(realized).size:
                    returns_by_arm[arm] = np.asarray(realized, dtype=float)
            if not returns_by_arm:
                out["regime_stratified"] = {
                    "status": "skipped", "reason": "no arm has usable TEST returns"
                }
            else:
                t_common = min(int(v.size) for v in returns_by_arm.values())
                returns_by_arm = {a: v[:t_common] for a, v in returns_by_arm.items()}
                labels = label_regimes(panel, load_config("regimes"))[
                    int(test_window[0]):int(test_window[1])
                ]
                # CRITICAL: the test series is a contiguous-from-test_start vector; its regime labels MUST be
                # the same window. A mismatch (e.g. an embargo offset) would silently misalign the strata.
                if int(np.asarray(labels).size) != int(t_common):
                    out["regime_stratified"] = {
                        "status": "skipped",
                        "reason": "length/embargo misalignment",
                        "n_labels": int(np.asarray(labels).size),
                        "t_common": int(t_common),
                    }
                else:
                    out["regime_stratified"] = regime_stratified_metrics(returns_by_arm, labels)
        except Exception as exc:  # noqa: BLE001 - a report-only regime split must never break the analysis
            out["regime_stratified"] = {"status": "error", "reason": str(exc)[:200]}

    # Variance decomposition (reviewer attack #10) -- the one-lucky-reward defence. ADDITIVE + report-only:
    # runs ONLY when >= 2 independent search re-run roots are supplied (else omitted), writing a DISJOINT
    # `variance` key (component family) so the frozen arm x metric family assert is untouched.
    if variance_run_roots and len(variance_run_roots) >= 2:
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import variance_decomposition as _vd  # scripts/variance_decomposition.py

            run_records = _vd._load_runs(list(variance_run_roots))
            out["variance"] = _vd.decompose_from_campaign(
                run_records, n_boot=2000, rng=np.random.default_rng(0)
            )
        except Exception as exc:  # noqa: BLE001 - a report-only appendix must never break the headline
            out["variance"] = {"status": "skipped", "reason": str(exc)}

    # Cross-hypothesis multiplicity (DEEP_STATS A4) — the H1–H4 headline decisions under a Bonferroni-
    # across-4 SENSITIVITY. Report-only (the per-hypothesis families stay primary); computed LAST so it
    # sees the final h1/h2/h3/h4 blocks. DISJOINT out["cross_hypothesis_multiplicity"].
    try:
        out["cross_hypothesis_multiplicity"] = cross_hypothesis_multiplicity(
            h1=out.get("h1_beat_human"), h2=out.get("h2"),
            h3=out.get("h3"), h4=out.get("h4"), alpha=alpha_one_sided,
        )
    except Exception as exc:  # noqa: BLE001 - a report-only sensitivity must never break the headline
        out["cross_hypothesis_multiplicity"] = {"status": "error", "reason": str(exc)[:200]}

    # THE RATIFIED PRIMARY DECISION RULE (R108, 2026-07-26): the graphical (Bretz-Maurer-Brannath-Posch
    # 2009) alpha-propagation over the six confirmatory nodes. This SUPERSEDES R31's separate-estimands
    # stance — the Bonferroni-over-4 block ABOVE is retained as the disclosed SENSITIVITY, not the gate.
    # Placed here because it needs every node's producer to have run (h2, h3, h4, h2_structure,
    # h1_beat_human). The graph is READ from config/preregistration.yaml, never hardcoded, so the
    # executed rule cannot drift from the frozen one (forking_path_guard). Fail-safe: a node whose
    # p-value cannot be located is reported `untestable` and can never reject — a shape mismatch shows
    # up in the artifact instead of producing a confidently wrong confirmatory verdict.
    try:
        from src.inference.validity_tier import tier_verdict

        out["validity_tier"] = tier_verdict(out)
    except Exception as exc:  # noqa: BLE001 - never let the primary rule's plumbing kill the whole report
        out["validity_tier"] = {"status": "error", "reason": str(exc)[:200]}

    # Mechanism multiplicity (forking-paths) — a Bonferroni/BH-across-mechanism-legs SENSITIVITY over the
    # report-only SQ1/SQ2/SQ3 diagnostics (responsiveness, mediation, AST permutation, McNemar, Mahalanobis,
    # legible-format, regime), mirroring the cross-hypothesis Bonferroni-across-4. Report-only, DISJOINT;
    # computed LAST so it sees the final mechanism blocks. Never changes any mechanism leg's own decision.
    try:
        out["mechanism_multiplicity"] = mechanism_multiplicity(
            responsiveness=out.get("responsiveness"),
            mediation=out.get("mediation"),
            named_vs_blinded_structural=out.get("named_vs_blinded_structural"),
            legible_format_responsiveness=out.get("legible_format_responsiveness"),
            regime_stratified=out.get("regime_stratified"),
            alpha=alpha_one_sided, q=q_level,
        )
    except Exception as exc:  # noqa: BLE001 - a report-only sensitivity must never break the headline
        out["mechanism_multiplicity"] = {"status": "error", "reason": str(exc)[:200]}
    return out


def _h2_iut_table(title: str, legs: list[dict[str, Any]]) -> list[str]:
    """Render one co-primary IUT's per-leg table (one-sided reject / direction / leg_supported)."""
    rows = [
        title,
        "",
        "| contrast | reject (1-sided) | direction_ok | leg_supported |",
        "|---|---|---|---|",
    ]
    for leg in legs:
        rows.append(
            f"| {leg['contrast']} | {leg.get('reject', leg.get('sharpe_reject'))} | "
            f"{leg.get('direction_ok', leg.get('sharpe_direction_ok'))} | {leg.get('leg_supported')} |"
        )
    return rows


def h2_markdown(h2: dict[str, Any]) -> str:
    """Render the headline two-tier H2 verdict (H2-RA + H2-Tail IUTs) + the BH-over-6 sensitivity (R25)."""
    if h2.get("error"):
        return f"## H2 (distributional feedback) — ERROR\n\n{h2['error']}\n"

    def _flag(supported: Any) -> str:
        return "SUPPORTED" if supported else ("NOT supported" if supported is not None else "n/a")

    ra = h2.get("H2_RA", {"supported": h2.get("H2_supported"), "legs": h2.get("legs", [])})
    tail = h2.get("H2_Tail", {"supported": None, "legs": h2.get("tail_legs", []), "level": 0.05})
    method = str(h2.get("method", "bh")).upper()
    verdict = h2.get("verdict", _flag(h2.get("H2_supported")))
    out = [
        f"## H2 (distributional feedback) — {verdict} ({method})",
        "",
        "Two CO-PRIMARY intersection-union tests (R25; DEEP_H2 §7.1), each decided ONE-SIDED at "
        f"α={float(h2.get('alpha', 0.05)):.2f} in the predicted direction with NO leg correction (the "
        "conjunction IS the correction — Berger 1982). Per-seed rliable inference (IQM + paired "
        "across-seed bootstrap; Agarwal et al. 2021).",
        "",
        f"- **H2-RA (risk-adjusted, Sharpe IUT):** {_flag(ra.get('supported'))}",
        f"- **H2-Tail (tail outcome, CVaR-{float(tail.get('level', 0.05)):g} IUT):** "
        f"{_flag(tail.get('supported'))} — corroborated (not gated) by the FZ0/(VaR,ES) comparative "
        "ES backtest where available.",
        "",
    ]
    out += _h2_iut_table("### H2-RA — risk-adjusted (Sharpe) legs", ra.get("legs", []))
    out += [""]
    out += _h2_iut_table("### H2-Tail — tail-outcome (CVaR-5%) legs", tail.get("legs", []))

    fam = h2.get("family", {})
    if fam.get("tests"):
        out += [
            "",
            # NOTE: 2026-07-27 (#106): this heading hardcoded "m=6" ONE LINE above the DERIVED
            # `m={n_family}` below it — the same quantity, one written twice. Currently consistent
            # (m IS 6), but a family change would make the report contradict itself in adjacent
            # lines. Derived now, so it cannot.
            f"### Sensitivity — Benjamini-Hochberg over the m={int(fam.get('n_family', 0))} union "
            "(REPORTED, NOT the gate; R25)",
            "",
            f"BH q={float(fam.get('q', 0.05)):.2f}, m={int(fam.get('n_family', 0))}. The headline gate is "
            "the two IUTs above; this FDR set is reported as a robustness sensitivity (DEEP_H2 §3.3-B).",
            "",
            "| label | p (2-sided) | p (1-sided) | effect | n_seeds | reject_1sided | reject_bh |",
            "|---|---|---|---|---|---|---|",
        ]
        for t, lbl in zip(fam["tests"], fam.get("labels", [])):
            p1 = t.get("pvalue_one_sided")
            p1_s = f"{float(p1):.4f}" if isinstance(p1, (int, float)) else "?"
            out.append(
                f"| {lbl} | {float(t['pvalue']):.4f} | {p1_s} | {float(t['effect']):+.4f} | "
                f"{t.get('n_seeds', '?')} | {t.get('reject_one_sided')} | {t.get('reject_bh')} |"
            )
    if h2.get("missing"):
        out += ["", f"Missing contrasts (unsupported, not fabricated): {', '.join(h2['missing'])}"]
    return "\n".join(out) + "\n"


def benchmark_floor_markdown(floor: dict[str, Any]) -> str:
    """Render the DeMiguel benchmark floor + market reference as markdown (PREREGISTRATION §9/§10)."""
    if not floor or floor.get("error"):
        reason = (floor or {}).get("error", "not computed")
        return f"## Benchmark floor (DeMiguel §9/§10) — n/a\n\n{reason}\n"
    bench = floor.get("benchmarks", {})
    out = [
        "## Benchmark floor — DeMiguel 1/N + 7 published allocators (PREREGISTRATION §9/§10; R19)",
        "",
        "Each allocator rolled through the IDENTICAL costed test env; benchmark DSR is un-searched (N=1). The "
        "turnover/cost columns (DEEP_BENCH_T0 #1) show the BINDING benchmark is a fairly-costed diversified "
        "allocator, not a daily-re-estimation cost artefact.",
        "",
        "| benchmark | Sharpe | CVaR 5% | max DD | DSR | mean turnover | ann cost % |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(bench.items(), key=lambda kv: -kv[1].get("dsr", 0.0)):
        out.append(
            f"| {name} | {m.get('sharpe', 0):.3f} | {m.get('cvar', 0):.4f} | "
            f"{m.get('max_drawdown', 0):.3f} | {m.get('dsr', 0):.3f} | "
            f"{m.get('mean_turnover', 0):.4f} | {m.get('ann_cost_pct', 0):.2f} |"
        )
    gate = floor.get("gate")
    if gate:
        passed = "PASS" if gate.get("passed") else "FAIL"
        out += [
            "",
            f"**Winner-vs-floor gate:** winner DSR {gate.get('winner_dsr', 0):.3f} "
            f"({gate.get('winner_dsr_method', 'single_path')}, deflated by N={gate.get('winner_n_trials', '?')} "
            f"searched candidates) vs best benchmark `{gate.get('best_benchmark')}` DSR "
            f"{gate.get('best_benchmark_dsr', 0):.3f} → **{passed}**",
            "",
            f"*Transparency (DEEP_BENCH_T0 #2):* the winner's UNDEFLATED (N=1) DSR is "
            f"{gate.get('winner_dsr_undeflated_n1', 0):.3f} — the like-for-like (N=1 vs N=1) comparison "
            f"separating realised performance from the {gate.get('winner_n_trials', '?')}× search-multiplicity "
            "penalty the winner alone pays (the benchmarks are un-searched).",
        ]
    mr = floor.get("market_reference")
    if mr and mr.get("market"):
        mk = mr["market"]
        out += [
            "",
            f"**Market reference** (EW universe; rf={mr.get('rf_source')} "
            f"{mr.get('rf_annual_pct_mean', 0):.2f}%/yr): Sharpe {mk.get('sharpe', 0):.3f}, ann return "
            f"{mk.get('ann_return_pct', 0):.1f}%, max DD {mk.get('max_drawdown', 0):.3f}.",
        ]
        wvm = mr.get("winner_vs_market")
        if wvm:
            out.append(
                f"Winner vs market: beta {wvm.get('beta', 0):.3f}, alpha "
                f"{wvm.get('alpha_ann', 0) * 100:.1f}%/yr, IR {wvm.get('information_ratio', 0):.3f}."
            )
    out.append("")
    return "\n".join(out)


# =========================================================================== #
# H1 — the Eureka-style "beat-the-human" metric (Ma et al. 2024; PREREGISTRATION §1 / §9)  #
# =========================================================================== #
#: The Eureka reference bars (Ma et al., ICLR 2024): Eureka beat the human-engineered reward on 83% of
#: 29 tasks at an average +52% normalised improvement. POST-FREEZE, REPORT-ONLY context — NOT a
#: pass/fail threshold the campaign must clear (the headline is the comparative H2, not H1).
EUREKA_BEAT_FRACTION: float = 0.83
EUREKA_NORM_IMPROVEMENT: float = 0.52


def _arm_seed_test_sharpes(records: list[dict[str, Any]], arm: str) -> list[float]:
    """Per-seed annualised test Sharpe for every (arm, seed) frozen-winner record of ``arm``.

    Reads each record's realized per-step ``metrics['test_returns']`` (via :func:`_test_returns`) and
    maps it through the inference ``sharpe_ratio`` — the SAME per-seed score the H2 leg and the floor gate
    use. Records without a usable test vector are skipped. Returns one Sharpe per seed (unaggregated).
    """
    from src.inference.bootstrap import sharpe_ratio

    out: list[float] = []
    for r in records:
        if r.get("arm") != arm:
            continue
        vec = _test_returns(r)
        if vec is None or vec.size < 2:
            continue
        out.append(float(sharpe_ratio(vec)))
    return out


def beat_human_baseline(
    records: list[dict[str, Any]],
    *,
    baseline_names: list[str] | tuple[str, ...],
    winner_arm: str | None = None,
    winner_n_trials: int | None = None,
    n_boot: int = 2000,
    alpha_one_sided: float = 0.05,
    rng: Any | None = None,
) -> dict[str, Any]:
    """H1 — the Eureka-STYLE "did the LLM-designed reward beat the BEST hand-designed reward?" metric.

    Implements PREREGISTRATION §1 (H1) / §9 (hand-reward panel) (Eureka-style, Ma et al. 2024) as a
    POST-FREEZE, REPORT-ONLY panel that is DISJOINT from the frozen H2 family — it writes
    ``out["h1_beat_human"]`` and carries NO ``arm_a/arm_b/metric/level`` keys, so
    ``assert_realized_family_matches_frozen`` never sees it and the frozen ``m=6`` testing family is
    untouched. It does NOT re-select the LLM winner (frozen on validation); it only reports the LLM
    winner's realized TEST performance against the hand baselines. (The "§18-19" cite previously here was
    WRONG: "18-19" were line numbers given in error; the pre-registration has only 12 numbered sections —
    H1 is §1, the hand-reward panel §9. DEEP_H1 C-1.)

    The comparison (matched budget, the FAIR Eureka asymmetry):
      * the LLM ``winner_arm`` was SEARCHED, so its Deflated Sharpe is deflated by its OWN search
        multiplicity (``winner_n_trials``, e.g. 30 candidates) — the same N the headline winner-DSR uses;
      * each ``baseline_<name>`` is UN-SEARCHED (a fixed hand reward), so its DSR is deflated by N=1 — like
        the 9 allocators in :func:`benchmark_floor` (8 until `min_cvar` was added; corrected 2026-07-27,
        #106). This asymmetry FAVOURS the baselines (a higher human
        bar), i.e. it is CONSERVATIVE for H1.

    Best-baseline IDENTITY selected on VALIDATION (DEEP_H1 T-REF data-snoop fix): the "best of the
    canon" (FOUR until the R105/R108 expansion to the ELEVEN-name canon; corrected 2026-07-27, #106)
    is an order statistic, so choosing it on the SAME sealed-test data the win is reported on data-snoops
    the comparator's identity (White 2000). When the baseline records carry a validation signal, the best
    baseline is picked by its VALIDATION score — in ONE unit chosen all-or-nothing for the whole baseline
    set (``selection_metric``: the annualized ``val_returns`` Sharpe if ANY baseline archives it, else the
    ``val_fitness`` DSR-like scalar; the two are incommensurate units and are never mixed per-record) —
    and the LLM-vs-that-FIXED-baseline gap is then reported on the sealed test leg (a pre-committed,
    Dunnett-valid comparator). If NO baseline archives a validation
    signal (the campaign's baseline TEST stage writes ``val_fitness=NaN`` and no ``val_returns`` — DEEP_H1
    §3.1), this FALLS BACK to test-median selection and FLAGS it (``best_selected_on`` records which, and
    ``val_snoop_caveat`` is set) so the data-snoop is disclosed, never hidden.

    The metrics — the single-task fraction-beaten + normalised-improvement, cited as "Eureka-STYLE" (NOT
    Eureka's Human Normalized Score ``(Method − Sparse)/|Human − Sparse|``, which needs a sparse
    ground-truth anchor finance lacks and is therefore NOT computable here — DEEP_H1 C-2; the repo metric
    is a relative-Sharpe improvement, an inspired analogue, not the identical Eureka formula):
      * ``best_baseline`` — the hand reward selected as above (the human bar);
      * ``beat_fraction`` — the SINGLE-TASK fraction-beaten: the fraction of the LLM winner's per-seed test
        Sharpes that exceed the best baseline's central-tendency Sharpe (the single-task, OOS analogue of
        Eureka's cross-task "fraction of tasks beaten"; a Bernoulli over SEEDS, not tasks);
      * ``norm_improvement`` — ``(median(LLM Sharpe) - best_baseline_central) / |best_baseline_central|``,
        a relative-Sharpe (Eureka-STYLE) normalised improvement on the central-tendency Sharpe;
      * ``beat_fraction_paired`` (SECONDARY, flagged) — the fraction of seeds where, AT THE SAME SEED
        INDEX, the LLM Sharpe beats the best-per-seed baseline Sharpe. The seed index is NOT a paired draw
        across two DIFFERENT rewards (each reward induces its own trajectory), so this is reported only as
        a sensitivity, not the headline;
      * ``winner_dsr`` / ``best_baseline_dsr`` — the median-per-seed DSR comparison (consistent with the
        floor gate's robust winner-DSR), deflated by ``winner_n_trials`` (LLM) vs N=1 (baseline).
      * ``iut`` — N6 (``validity_tier``, RATIFIED R108 2026-07-26): the SNOOP-FREE intersection-union test of
        whether the LLM reward DOMINATES the hand-reward canon — does it beat EVERY member one-sided at
        ``alpha_one_sided`` (== beat the *best* human, since best = max and beat-max ⟺ beat-all; berger1982iut)?
        Carries ``iut_pvalue`` (= the MAX over the per-baseline one-sided leg p-values), ``dominates_canon``
        (the verdict; requires every member testable), and the per-baseline ``dominance_profile``
        (effect / one-sided p / verdict ∈ {dominates, ahead_ns, behind, not_testable}). NO comparator is ever
        selected, so there is nothing to snoop — this dissolves the White-2000 snoop that keeps the
        max-selection above REPORT-ONLY, and needs no baseline validation-roll (the campaign archives
        ``val_fitness=NaN``). Reuses the H2 paired seed bootstrap so H1 and the headline share one inference.

    Degrades gracefully to ``{"status": "skipped", "reason": ...}`` when the winner arm or all configured
    baselines are absent from ``records`` (e.g. the baseline stage was not run, or a records-only window).
    """
    from src.inference.deflated_sharpe import deflated_sharpe_ratio

    head = winner_arm or (H2_CONTRASTS[0][0] if H2_CONTRASTS else "distributional")
    names = [str(b) for b in baseline_names]

    # The LLM winner's per-seed test Sharpes + per-seed test vectors (for the DSR side).
    winner_sharpes = _arm_seed_test_sharpes(records, head)
    winner_vecs = [
        v for r in records if r.get("arm") == head and (v := _test_returns(r)) is not None and v.size > 1
    ]
    if not winner_sharpes:
        return {
            "status": "skipped",
            "reason": f"no test records for the LLM winner arm {head!r} (baseline stage / records-only?)",
            "winner_arm": head,
            "baselines_configured": names,
        }

    from src.inference.bootstrap import sharpe_ratio as _sr

    baseline_arms = {f"baseline_{n}" for n in names}

    def _usable_val_returns(m: dict[str, Any]) -> np.ndarray | None:
        """The record's usable per-period validation vector (finite, length > 1), or None."""
        vr = m.get("val_returns")
        if vr is None:
            return None
        arr = np.asarray(vr, dtype=float).ravel()
        return arr if arr.size > 1 and np.all(np.isfinite(arr)) else None

    # ALL-OR-NOTHING selection-metric choice (unit-pooling fix): an annualized val_returns Sharpe is O(1)
    # while val_fitness is a DSR probability in [0, 1] — pooling the two into one median (or comparing
    # baselines scored on different ones) mixes incommensurate units and can flip the best-baseline
    # argmax. The metric is therefore chosen ONCE for the WHOLE baseline set: val_returns-derived Sharpe
    # if ANY baseline record archives a usable val_returns; val_fitness ONLY when NONE does. The choice
    # is disclosed as ``selection_metric`` in the output.
    any_val_returns = any(
        r.get("arm") in baseline_arms and _usable_val_returns(r.get("metrics") or {}) is not None
        for r in records
    )
    selection_metric = "val_sharpe" if any_val_returns else "val_fitness"

    def _baseline_val_sharpe(name: str) -> float | None:
        """A baseline's VALIDATION selection score (median over its records), or None when not archived.

        Scores are taken ONLY in the set-wide ``selection_metric`` unit (never a per-record mix): under
        ``val_sharpe``, the median annualized Sharpe of this baseline's records carrying a usable
        ``metrics['val_returns']`` (a baseline with none returns None and drops out of the validation
        scan); under ``val_fitness`` (NO baseline archived val_returns), the median finite
        ``metrics['val_fitness']``. The campaign's baseline TEST stage writes ``val_fitness=NaN`` and no
        ``val_returns`` (DEEP_H1 §3.1), so this returns None there -> the caller falls back to
        test-selection WITH a flag.
        """
        arm = f"baseline_{name}"
        vals: list[float] = []
        for r in records:
            if r.get("arm") != arm:
                continue
            m = r.get("metrics") or {}
            if selection_metric == "val_sharpe":
                arr = _usable_val_returns(m)
                if arr is not None:
                    vals.append(float(_sr(arr)))
            else:
                vf = m.get("val_fitness")
                if vf is not None and np.isfinite(float(vf)):
                    vals.append(float(vf))
        return float(np.median(vals)) if vals else None

    # Per-baseline per-seed test Sharpes; a baseline with no test records is recorded as absent. Also capture
    # a VALIDATION Sharpe per baseline (DEEP_H1 T-REF) so the best-baseline IDENTITY can be picked on val.
    per_baseline: dict[str, dict[str, Any]] = {}
    for name in names:
        arm = f"baseline_{name}"
        shp = _arm_seed_test_sharpes(records, arm)
        if not shp:
            per_baseline[name] = {"present": False, "n_seeds": 0, "median_sharpe": None, "val_sharpe": None}
            continue
        per_baseline[name] = {
            "present": True,
            "n_seeds": len(shp),
            "median_sharpe": float(np.median(shp)),
            "mean_sharpe": float(np.mean(shp)),
            "sharpes": [float(x) for x in shp],
            "val_sharpe": _baseline_val_sharpe(name),
        }

    present = {n: e for n, e in per_baseline.items() if e.get("present")}
    if not present:
        return {
            "status": "skipped",
            "reason": "none of the configured H1 baselines have test records (baseline stage not run?)",
            "winner_arm": head,
            "baselines_configured": names,
            "baselines": per_baseline,
        }

    # The HUMAN BAR identity. PREFER VALIDATION (DEEP_H1 T-REF data-snoop fix): pick the best baseline by its
    # VALIDATION Sharpe (a pre-committed comparator), then report the gap on the sealed TEST leg. Only when
    # NO present baseline archives a validation signal do we FALL BACK to test-median selection — and FLAG
    # it (val_snoop_caveat) so the data-snoop on the comparator identity is disclosed, never hidden.
    val_present = {n: e for n, e in present.items() if e.get("val_sharpe") is not None}
    if val_present:
        best_name = max(val_present, key=lambda n: val_present[n]["val_sharpe"])
        best_selected_on = "validation"
        val_snoop_caveat = False
        # The validation-selected human bar is a pre-committed comparator (Dunnett-valid; White 2000) -> the
        # H1 comparison carries a defensible inferential reading.
        inference_status = "val_selected"
        caveat_text = ""
    else:
        best_name = max(present, key=lambda n: present[n]["median_sharpe"])
        best_selected_on = "test (median Sharpe) — validation not archived"
        val_snoop_caveat = True
        # T2.2 (unmissable disclosure): NO baseline archived a validation signal (the campaign baseline TEST
        # stage writes val_fitness=NaN and no val_returns — _baseline_winner_record), so the best-of-4 human
        # bar IDENTITY is an order statistic chosen on the SAME sealed-test data the gap is reported on. That
        # data-snoops the comparator (White 2000), so the H1 number is DESCRIPTIVE-ONLY and carries no
        # inferential "beat the human" claim. This is surfaced as a structured status + a prominent warning
        # block at the TOP of the H1 markdown (not a buried bullet) and echoed to stdout.
        inference_status = "test_snooped_descriptive_only"
        caveat_text = (
            "H1 best-baseline identity was selected on the SEALED TEST leg (no baseline archived a "
            "validation signal: the campaign baseline stage writes val_fitness=NaN). The 'best of 4' is an "
            "order statistic, so choosing it on the same test data the gap is reported on DATA-SNOOPS the "
            "comparator (White 2000). H1 is therefore DESCRIPTIVE-ONLY here — it carries NO inferential "
            "beat-the-human claim. FIX to restore a valid bar: archive a validation signal for the baselines "
            "(roll each baseline on the validation leg so it carries val_returns) — see PREREGISTRATION §1 / "
            "§10 proposed amendment."
        )
    best_median = float(present[best_name]["median_sharpe"])
    best_sharpes = np.asarray(present[best_name]["sharpes"], dtype=float)

    w = np.asarray(winner_sharpes, dtype=float)
    winner_median = float(np.median(w))

    # (1) Single-task fraction-beaten (Eureka-STYLE): how often the LLM winner's per-seed Sharpe clears the
    # human bar (best baseline's central Sharpe) — a Bernoulli over SEEDS, the single-task OOS analogue of
    # Eureka's cross-task "fraction of tasks beaten".
    beat_fraction = float(np.mean(w > best_median))
    # (2) Relative-Sharpe (Eureka-STYLE) normalised improvement on the central-tendency Sharpe. This is NOT
    # Eureka's Human Normalized Score (M−Sparse)/|H−Sparse| (no sparse ground-truth anchor in finance —
    # DEEP_H1 C-2); it is a relative-Sharpe improvement cited as Eureka-style. Guard a ~0 denominator (a
    # best baseline whose median Sharpe is ~0 makes the ratio meaningless) -> report None, not a blow-up.
    denom = abs(best_median)
    norm_improvement = float((winner_median - best_median) / denom) if denom > 1e-9 else None

    # (SECONDARY, flagged) paired-by-seed fraction: LLM seed s vs the best-per-seed baseline at seed s.
    # The pairing is across DIFFERENT rewards (not a common-noise paired draw), so this is a sensitivity
    # only. Aligns to the common seed count; uses the per-seed MAX over present baselines as the bar.
    paired_fraction: float | None = None
    aligned = {n: e["sharpes"] for n, e in present.items()}
    common = min([len(w)] + [len(v) for v in aligned.values()])
    if common >= 1:
        bar_per_seed = np.max(
            np.stack([np.asarray(v, dtype=float)[:common] for v in aligned.values()]), axis=0
        )
        paired_fraction = float(np.mean(w[:common] > bar_per_seed))

    # (DSR side) median-per-seed winner DSR (deflated by the SEARCH multiplicity) vs best-baseline DSR
    # (deflated by N=1) — the robust, no-seed-averaging comparison the floor gate uses.
    wnt = int(winner_n_trials) if winner_n_trials else 1
    winner_dsr = float(
        np.median([deflated_sharpe_ratio(np.asarray(v, dtype=float).ravel(), wnt) for v in winner_vecs])
    ) if winner_vecs else None
    best_baseline_dsr = None
    best_base_vecs = [
        v for r in records if r.get("arm") == f"baseline_{best_name}"
        and (v := _test_returns(r)) is not None and v.size > 1
    ]
    if best_base_vecs:
        best_baseline_dsr = float(
            np.median([deflated_sharpe_ratio(np.asarray(v, dtype=float).ravel(), 1) for v in best_base_vecs])
        )

    # --- N6 (validity_tier, RATIFIED R108 2026-07-26): does the LLM reward DOMINATE the hand-reward canon? ---
    # The intuitive claim "the LLM beats the BEST human-crafted reward" made PRECISE and SNOOP-FREE. The best
    # human is the pointwise MAX over the canon, and (LLM > max) <=> (LLM beats EVERY member); so requiring the
    # LLM to beat all N baselines — each one-sided at alpha — is an INTERSECTION-UNION TEST (Berger 1982): the
    # IUT p-value is the MAX over the per-baseline one-sided p-values, and the LLM "dominates the canon" iff
    # every leg rejects. NO comparator is ever selected, so there is nothing to snoop — this DISSOLVES the
    # White-2000 data-snoop that keeps the max-selection above report-only, WITHOUT the fragile baseline
    # val-roll the val-select framing needed (the campaign archives val_fitness=NaN). Reuses the SAME paired
    # per-seed bootstrap as the H2 legs (paired_seed_difference_test over shared seeds, iqm statistic) so H1
    # and the headline share one inference tool.
    # NOTE: 2026-08-01 (A15, raised by the ANALYSIS lane, verified first-hand here): this comment used to read
    # "Conservative by design (the searched LLM's DSR is deflated by N=winner_n_trials, each hand reward by
    # N=1 — the human bar is conservatively HIGH)". BOTH HALVES WERE WRONG. (a) There is no DSR anywhere in
    # this leg computation — `grep -niE "dsr|deflat"` over the block below returns nothing; the endpoint is
    # ANNUALISED PER-SEED SHARPE (endpoint CORRECTED 2026-07-26 precisely because the registered
    # `deflated_sharpe` was never computed). It was DSR-era prose that outlived its own retraction.
    # (b) It stated the bias BACKWARDS. The frozen config's own corrected note says the genuine residual
    # asymmetry — the LLM winner being the best of N VALIDATION candidates against one fixed specification
    # per hand reward — **FAVOURS THE LLM**, and is disclosed as the un-tuned-baseline bias. Deflation would
    # correct for taking a max over N on the SAME data; selection here happens on VALIDATION and the test leg
    # is SEALED, so there is no test-set max to deflate. The error ran in the direction that flatters our own
    # hypothesis, inside CONFIRMATORY output. CONFIRMATORY since
    # R108 (2026-07-26) ratified the validity tier; the per-baseline dominance PROFILE is the honest
    # structure beneath the binary.
    from src.inference.bootstrap import iqm as _iqm, paired_seed_difference_test as _paired_test
    _iut_rng = rng if rng is not None else np.random.default_rng(0)
    _w_seed = _seed_scores(records, head, _sr)                # winner arm: {seed -> annualized test Sharpe}
    iut_legs: list[dict[str, Any]] = []
    for name in names:
        _b_seed = _seed_scores(records, f"baseline_{name}", _sr)
        _common = sorted(set(_w_seed) & set(_b_seed))
        if len(_common) < 2:
            iut_legs.append({
                "baseline": name, "present": bool(_b_seed), "n_seeds": len(_common),
                "effect": None, "pvalue_one_sided": None, "beaten": None,
                "verdict": "not_testable",
                "reason": "no shared test seed" if not _b_seed else "only 1 shared test seed (need >= 2)",
            })
            continue
        _a = np.asarray([_w_seed[s] for s in _common], dtype=float)
        _bb = np.asarray([_b_seed[s] for s in _common], dtype=float)
        _res = _paired_test(_a, _bb, statistic=_iqm, n_boot=n_boot, rng=_iut_rng)
        _eff = float(_res["effect"])
        _p1 = float(_res["pvalue_one_sided_greater"])
        _beaten = bool(_eff > 0.0 and _p1 <= alpha_one_sided)
        iut_legs.append({
            "baseline": name, "present": True, "n_seeds": len(_common),
            "effect": _eff, "pvalue_one_sided": _p1, "beaten": _beaten,
            # 3-way honest profile from the single LLM-vs-baseline leg: dominated / ahead-but-n.s. / behind.
            "verdict": ("dominates" if _beaten else ("ahead_ns" if _eff > 0.0 else "behind")),
        })
    _testable = [lg for lg in iut_legs if lg.get("pvalue_one_sided") is not None]
    _all_present = bool(len(_testable) == len(names) and len(names) > 0)
    _iut_p = max((lg["pvalue_one_sided"] for lg in _testable), default=None)   # IUT p = MAX leg p (Berger 1982)
    _dominates = bool(_all_present and _testable and all(lg.get("beaten") for lg in _testable))
    iut_block = {
        "test": "llm_dominates_hand_reward_canon",
        "claim": "the LLM-designed reward beats the BEST human-crafted reward (dominates the full hand-reward canon)",
        "method": "intersection_union_over_canon",   # berger1982iut: beat every member <=> beat the max (the best)
        "snoop_free": True,                           # no comparator selected -> nothing to snoop (cf. white2000reality)
        "ref": ["berger1982iut", "white2000reality"],
        "alpha_one_sided": float(alpha_one_sided),
        "n_baselines": len(names),
        "n_testable": len(_testable),
        "all_baselines_present": _all_present,        # dominance is certifiable ONLY when every member is testable
        "iut_pvalue": _iut_p,                         # the MAX one-sided leg p (None if none testable)
        "dominates_canon": _dominates,                # True iff EVERY member beaten one-sided at alpha AND all present
        "n_significantly_beaten": int(sum(1 for lg in _testable if lg.get("beaten"))),
        "n_ahead_not_significant": int(sum(1 for lg in _testable if lg.get("verdict") == "ahead_ns")),
        "n_behind_on_point_estimate": int(sum(1 for lg in _testable if lg.get("verdict") == "behind")),
        "dominance_profile": iut_legs,                # per-baseline effect + one-sided p + verdict (the honest structure)
        "note": ("dominance is certifiable ONLY when every canon member has >= 2 shared test seeds; a missing or "
                 "under-seeded member sets all_baselines_present=false -> report-only, never a dominance claim"),
        # NOTE: 2026-07-27 (deep review, finding #99): this field said
        # "registered_pending_supervisor_ratification" — STALE since R108 (2026-07-26) recorded
        # `inference.validity_tier.status: ratified` with `ratification_pending: []` and
        # `n6_h1_confirmatory_node` in `ratification_completed`, signed off in supervision.
        # It is write-only metadata (nothing branches on it), but it lands in the ARCHIVED result
        # JSON that feeds the write-up, so a replay-only campaign would have reported its own
        # confirmatory node as un-ratified. Same stale-fact class as #84.
        "status": "ratified (validity_tier N6; R108 2026-07-26, supervisor sign-off)",
    }

    return {
        "status": "ok",
        "winner_arm": head,
        "winner_n_trials": wnt,
        "n_winner_seeds": int(w.size),
        "winner_median_sharpe": winner_median,
        "winner_mean_sharpe": float(np.mean(w)),
        "best_baseline": best_name,
        "best_baseline_median_sharpe": best_median,
        "best_baseline_n_seeds": int(best_sharpes.size),
        # DEEP_H1 T-REF: which leg picked the best-baseline IDENTITY (validation = data-snoop-free; test
        # fallback = flagged). `val_snoop_caveat` True => identity chosen on the same test data the gap is
        # reported on (the data-snoop the write-up must disclose).
        "best_selected_on": best_selected_on,
        # The UNIT of the validation selection score (chosen all-or-nothing across the whole baseline
        # set): "val_sharpe" = annualized Sharpe from val_returns; "val_fitness" = the DSR-like scalar,
        # used ONLY when no baseline archived val_returns — never a per-record mix of the two.
        "selection_metric": selection_metric,
        "val_snoop_caveat": val_snoop_caveat,
        # T2.2: a STRUCTURED inference status downstream can key on ("val_selected" = defensible bar;
        # "test_snooped_descriptive_only" = the data-snoop fallback, H1 is descriptive-only) + the unmissable
        # caveat text the markdown surfaces as a top-of-section warning block.
        "inference_status": inference_status,
        "caveat": caveat_text,
        "best_baseline_val_sharpe": present[best_name].get("val_sharpe"),
        # The headline metrics — single-task fraction-beaten + normalised-improvement, "Eureka-STYLE" (NOT
        # Eureka's HNS formula, which is not computable here — DEEP_H1 C-2). Refs kept for CONTEXT only.
        "beat_fraction": beat_fraction,
        "norm_improvement": norm_improvement,
        "beat_fraction_paired": paired_fraction,
        "eureka_beat_fraction_ref": EUREKA_BEAT_FRACTION,
        "eureka_norm_improvement_ref": EUREKA_NORM_IMPROVEMENT,
        # Verdicts (descriptive, NOT a frozen test): LLM clears the human bar on central tendency / DSR.
        "beats_best_baseline_median": bool(winner_median > best_median),
        "winner_dsr": winner_dsr,
        "best_baseline_dsr": best_baseline_dsr,
        "beats_best_baseline_dsr": (
            bool(winner_dsr > best_baseline_dsr)
            if (winner_dsr is not None and best_baseline_dsr is not None) else None
        ),
        "baselines": per_baseline,
        # N6 (validity_tier, RATIFIED R108): the snoop-free IUT — does the LLM reward DOMINATE the
        # entire hand-reward canon (== beat the best human, made precise)? + the per-baseline dominance profile.
        "iut": iut_block,
    }


def h1_beat_human_markdown(h1: dict[str, Any]) -> str:
    """Render the H1 Eureka-STYLE "beat-the-human" panel as markdown (PREREGISTRATION §1 / §9; Ma et al. 2024)."""
    if not h1 or h1.get("status") != "ok":
        reason = (h1 or {}).get("reason", "not computed")
        return f"## H1 — beat-the-human (Eureka-style; §1 / §9) — n/a\n\n{reason}\n"

    def _f(v: Any, p: int = 3) -> str:
        return "n/a" if v is None else f"{v:.{p}f}"

    ni = h1.get("norm_improvement")
    ni_s = "n/a" if ni is None else f"{ni * 100:+.1f}%"
    pf = h1.get("beat_fraction_paired")
    selected_on = h1.get("best_selected_on", "test")
    snoop = h1.get("val_snoop_caveat")
    out = [
        "## H1 — did the LLM-designed reward beat the BEST hand reward? (Eureka-style, Ma et al. 2024; §1 / §9)",
        "",
    ]
    # T2.2 — the UNMISSABLE data-snoop disclosure: a prominent blockquote WARNING at the very TOP of the H1
    # section (not a buried bullet) whenever the human-bar identity was selected on the sealed test leg.
    if snoop:
        out += [
            "> **DATA-SNOOP — H1 IS DESCRIPTIVE-ONLY (no inferential beat-the-human claim).** "
            + (h1.get("caveat") or "")
            .replace(" — see PREREGISTRATION", "\n>\n> *Remedy:* see PREREGISTRATION"),
            "",
        ]
    out += [
        f"POST-FREEZE, report-only. LLM winner arm `{h1.get('winner_arm')}` ({h1.get('n_winner_seeds')} "
        f"seeds) vs the best of {len(h1.get('baselines', {}))} hand baselines. The LLM DSR is deflated by "
        f"its searched N={h1.get('winner_n_trials')}; each baseline by N=1 (un-searched) — the fair, "
        "CONSERVATIVE asymmetry (a higher human bar). The metrics are single-task fraction-beaten + "
        "relative-Sharpe normalised-improvement, cited as *Eureka-STYLE* (NOT Eureka's "
        "(Method−Sparse)/|Human−Sparse| HNS, which needs a sparse ground-truth anchor finance lacks — "
        "the Eureka bars below are CONTEXT, not a target).",
        "",
        f"- **Best-baseline identity selected on:** {selected_on} "
        f"(selection metric: `{h1.get('selection_metric', 'val_sharpe')}`)"
        + ("  *(data-snoop: identity chosen on the same test data the gap is reported on — DEEP_H1 T-REF)*"
           if snoop else "  *(validation — data-snoop-free comparator, DEEP_H1 T-REF)*"),
        f"- **Best hand reward (human bar):** `{h1.get('best_baseline')}` "
        f"(median test Sharpe {_f(h1.get('best_baseline_median_sharpe'))}; "
        f"val Sharpe {_f(h1.get('best_baseline_val_sharpe'))}).",
        f"- **LLM winner:** median test Sharpe {_f(h1.get('winner_median_sharpe'))}.",
        f"- **Fraction beaten** (LLM per-seed Sharpe > human bar): "
        f"**{_f(h1.get('beat_fraction'))}** vs Eureka {h1.get('eureka_beat_fraction_ref')} (context).",
        f"- **Normalised improvement** (relative Sharpe, median): **{ni_s}** vs Eureka "
        f"+{int(h1.get('eureka_norm_improvement_ref', 0.52) * 100)}% (context).",
        f"- **DSR:** LLM {_f(h1.get('winner_dsr'))} vs best-baseline {_f(h1.get('best_baseline_dsr'))} → "
        f"beats-on-DSR = {h1.get('beats_best_baseline_dsr')}.",
    ]
    if pf is not None:
        out.append(
            f"- *(secondary, flagged)* paired-by-seed beat fraction {_f(pf)} — the seed index is NOT a "
            "common-noise paired draw across two different rewards, so this is a sensitivity only."
        )
    out += [
        "",
        "| baseline | present | n seeds | median test Sharpe | val Sharpe |",
        "|---|---|---|---|---|",
    ]
    for name, e in h1.get("baselines", {}).items():
        marker = " (best)" if name == h1.get("best_baseline") else ""
        out.append(
            f"| {name}{marker} | {e.get('present')} | {e.get('n_seeds', 0)} | "
            f"{_f(e.get('median_sharpe'))} | {_f(e.get('val_sharpe'))} |"
        )

    # N6 (validity_tier, RATIFIED R108): the snoop-free IUT — does the LLM reward DOMINATE the canon?
    iut = h1.get("iut") or {}
    if iut.get("n_testable"):
        dom = iut.get("dominates_canon")
        verdict = "DOMINATES the hand-reward canon" if dom else "does NOT dominate the canon"
        cert = "" if iut.get("all_baselines_present") else (
            "  *(not every canon member is testable — dominance is NOT certifiable here; report-only)*"
        )
        out += [
            "",
            f"### N6 — does the LLM reward beat the BEST human? (snoop-free IUT over the canon; Berger 1982) "
            f"— **{verdict}**",
            "",
            f"The best hand reward is the pointwise max over the canon, and beating the max ⟺ beating *every* "
            f"member; so the LLM must beat all {iut.get('n_baselines')} hand rewards, each one-sided at "
            f"α={_f(iut.get('alpha_one_sided'), 3)} — an intersection-union test that selects **no** comparator "
            f"(nothing to snoop; this is the snoop-free reading of “beat the best human”). "
            f"**IUT p = max leg p = {_f(iut.get('iut_pvalue'), 4)}.**{cert}",
            f"- **Dominance profile:** {iut.get('n_significantly_beaten')}/{iut.get('n_testable')} "
            f"significantly beaten · {iut.get('n_ahead_not_significant')} ahead (n.s.) · "
            f"{iut.get('n_behind_on_point_estimate')} behind (on the point estimate).",
            "",
            f"**Direction of the residual selection asymmetry — stated against us, because that is the way "
            f"it runs.** The LLM winner is the best of {h1.get('winner_n_trials')} *validation* candidates, "
            f"while each hand reward is a single fixed specification carried into the sealed test unchanged. "
            f"That asymmetry **FAVOURS THE LLM**, and it is disclosed here as the un-tuned-baseline bias "
            f"rather than argued away. There is no deflation correcting for it: the legs are annualised "
            f"per-seed Sharpe (the registered `deflated_sharpe` endpoint was corrected on 2026-07-26 because "
            f"the code never computed it), and deflation would in any case address a max taken over N on the "
            f"*same* data — whereas selection here happens on validation and the test leg is sealed.",
            "",
            "| hand reward | Δ Sharpe (LLM − human) | one-sided p | verdict |",
            "|---|---|---|---|",
        ]
        for lg in iut.get("dominance_profile", []):
            out.append(
                f"| {lg.get('baseline')} | {_f(lg.get('effect'), 4)} | {_f(lg.get('pvalue_one_sided'), 4)} | "
                f"{lg.get('verdict')} |"
            )
    out.append("")
    return "\n".join(out)


def h2_rf_robustness_markdown(rob: dict[str, Any]) -> str:
    """Render the R20 risk-free robustness of the H2 Sharpe conjunction (additive sensitivity)."""
    if not rob or rob.get("error") or not rob.get("contrasts"):
        return ""  # silent when not computed (records-only / no rf series)
    verdict = "SURVIVES" if rob.get("survives") else "does NOT survive"
    out = [
        f"## R20 — H2 Sharpe robustness to the risk-free convention — {verdict}",
        "",
        f"Sharpe leg recomputed on EXCESS returns (r − DGS3MO, {rob.get('rf_annualised_pct', 0):.2f}%/yr); "
        "CVaR raw; the frozen rf=0 headline above is UNCHANGED. The per-seed rf penalty mean(rf)·√252/σ is "
        "larger for lower-vol arms, so positive shrinkage = the distributional edge narrows under rf.",
        "",
        "| contrast | effect rf=0 | effect excess | shrinkage | reject rf=0 | reject excess | survives |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rob.get("contrasts", []):
        out.append(
            f"| {r['contrast']} | {r['effect_rf0']:+.4f} | {r['effect_excess']:+.4f} | "
            f"{r['effect_shrinkage']:+.4f} | {r['reject_rf0']} | {r['reject_excess']} | {r['leg_survives']} |"
        )
    out.append("")
    return "\n".join(out)


def h4_markdown(h4: dict[str, Any]) -> str:
    """Render the H4 (LLM winner vs random-search / Bayes-opt) difference tests (DEEP_H4)."""
    if not h4 or h4.get("status") != "ok":
        reason = (h4 or {}).get("reason", "not computed")
        return f"## H4 — LLM vs search controls (DEEP_H4) — n/a\n\n{reason}\n"
    verdict = "SUPPORTED" if h4.get("all_supported") else "NOT supported (one or more legs)"
    # NOTE: 2026-07-27 (deep review, finding #100): every count below is now DERIVED from the realised
    # test set. The H4 family grew 2 -> 4 on 2026-07-26 (+cma_es/tpe) and this emitter was never
    # updated, so the REPORTED table described a 2-test family with an alpha/2 Bonferroni while the
    # code had already been correcting over 4. Deriving stops it re-staling on the next expansion.
    _n_h4 = int(h4.get("n_tests") or len(H4_CONTRASTS))
    _bonf = float(h4.get("bonferroni_alpha", float(h4.get("alpha", 0.05)) / max(_n_h4, 1)))
    margin = h4.get("equiv_margin", 0.05)
    out = [
        f"## H4 — did the LLM reward-designer beat the search controls? — {verdict}",
        "",
        f"{_n_h4} pre-registered difference tests on the sealed test leg (per-seed Sharpe → IQM → paired "
        f"bootstrap, one-sided), mirroring H2-RA, EACH with a ±{margin:g} TOST equivalence bound (T3.4 a). "
        "Read as **procedure-vs-richness**, not a nested horse-race: H4a = the IN-FAMILY random-search "
        "REFERENCE (same 6-term family the LLM authors over, R28) → isolates PROCEDURE at matched richness; "
        "H4b = the fixed-parametric-template reference → open-ended language vs fixed family (DEEP_H4 §1.2). "
        f"NEITHER asserts 'better optimiser over an identical space'. Own {_n_h4}-test family (NOT the "
        f"m=6 H2 family); a Bonferroni-over-{_n_h4} (α={_bonf:.4f}) sensitivity is shown.",
        "",
        f"| test | contrast | reference | effect (Sharpe IQM) | p (1-sided) | reject | reject (Bonf-{_n_h4}) | "
        f"equiv ±{margin:g} | verdict | n_seeds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in h4.get("tests", []):
        eq = t.get("equivalence", {})
        # Use the AUTHORITATIVE per-test label (_H4_REFERENCE_FRAMING, attached at the test row) rather
        # than a hardcoded 2-way branch, which mislabelled BOTH h4c (CMA-ES) and h4d (TPE) as the
        # h4b Bayes-opt framing. Keep only the leading clause so the table cell stays readable.
        ref = (t.get("reference") or t["test"]).split(" — ")[0].split(" -- ")[0]
        out.append(
            f"| {t['test'].upper()} | {t['a']}>{t['b']} | {ref} | {t['effect']:+.4f} | "
            f"{t['pvalue_one_sided']:.4f} | {t['reject_one_sided']} | "
            f"{t['reject_one_sided_bonferroni']} | {eq.get('equivalent')} | "
            f"{t.get('verdict', '?')} | {t['n_seeds']} |"
        )
    for s in h4.get("skipped", []):
        out.append(
            f"| {s['test'].upper()} | {s['a']}>{s['b']} | — | — | — | skipped | — | — | — | ({s['reason']}) |"
        )
    out += [
        "",
        f"All H4 tests reject one-sided: **{h4.get('all_supported')}** "
        f"(Bonferroni-over-{_n_h4}: {h4.get('all_supported_bonferroni')}); all ran legs equivalence-bounded "
        f"within ±{margin:g}: **{h4.get('all_equivalent')}** (the bankable H4 null).",
        "",
    ]
    return "\n".join(out)


def h3_markdown(h3: dict[str, Any]) -> str:
    """Render H3 (iterative reflection vs single-shot) difference + TOST equivalence (DEEP_H3)."""
    if not h3 or h3.get("status") != "ok":
        reason = (h3 or {}).get("reason", "not computed")
        return (
            "## H3 — iterative reflection vs single-shot (DEEP_H3) — n/a\n\n"
            f"{reason}\n\n*(H3 is a separate, manually-launched single-shot run; absent ⇒ skipped, not "
            "fabricated.)*\n"
        )
    diff = h3.get("difference", {})
    eq = h3.get("equivalence", {})
    out = [
        f"## H3 — iterative reflection vs single-shot best-of-N — {h3.get('verdict', '?')}",
        "",
        f"Within the `{h3.get('arm')}` arm, matched budget, identical selection (DEEP_H3 §1.1). Difference "
        "test (per-seed Sharpe → IQM → paired bootstrap, one-sided 'iterative > single-shot') + a TOST "
        f"equivalence at ±{eq.get('margin', 0.05):g} in the test-statistic's (Sharpe IQM) units (the "
        "bankable-null bound, DEEP_H3 §8.2). NOT in the m=6 H2 family.",
        "",
        f"- **Difference:** effect {diff.get('effect', 0):+.4f} (Sharpe IQM), p(1-sided) "
        f"{diff.get('pvalue_one_sided', 1.0):.4f} → reject = **{diff.get('reject_one_sided')}** "
        f"(n_seeds {diff.get('n_seeds', '?')}).",
        f"- **Equivalence (TOST ±{eq.get('margin', 0.05):g}):** estimate {eq.get('estimate', 0):+.4f}, "
        f"90% CI [{eq.get('ci_low', 0):+.4f}, {eq.get('ci_high', 0):+.4f}] → equivalent = "
        f"**{eq.get('equivalent')}**.",
    ]
    # T3.4 (c): the paired placebo-relative uplift difference (an information-tracking-signature probe).
    pru = h3.get("placebo_relative_uplift") or {}
    if pru.get("status") == "ok":
        out.append(
            f"- **Placebo-relative uplift (T3.4):** distributional uplift "
            f"{pru.get('mean_uplift_distributional', 0):+.4f} − placebo uplift "
            f"{pru.get('mean_uplift_placebo', 0):+.4f} = **{pru.get('effect', 0):+.4f}** (Sharpe), p(2-sided) "
            f"{pru.get('pvalue_two_sided', 1.0):.4f} → information-tracking signature = "
            f"**{pru.get('reject_two_sided')}**. {pru.get('interpretation', '')}"
        )
    else:
        out.append(
            f"- **Placebo-relative uplift (T3.4):** n/a — {pru.get('reason', 'not computed')} "
            "(needs a placebo single-shot condition; the headline single-shot stage runs only the "
            "distributional arm)."
        )
    out.append("")
    return "\n".join(out)


def h2_structure_markdown(sc: dict[str, Any]) -> str:
    """Render the structure-vs-content control (R32) — distributional vs placebo_shuffled."""
    if not sc or sc.get("status") != "ok":
        reason = (sc or {}).get("reason", "not computed")
        return f"## Structure-vs-content control (R32) — n/a\n\n{reason}\n"

    def _row(name: str, leg: dict[str, Any] | None) -> str:
        if not leg:
            return f"| {name} | n/a | n/a | n/a |"
        return (
            f"| {name} | {leg['effect']:+.4f} | {leg['pvalue_one_sided']:.4f} | "
            f"{leg['reject_one_sided']} |"
        )

    verdict = "CONTENT over format" if sc.get("content_over_format") else "NOT established (bound only)"
    return "\n".join(
        [
            "## Structure-vs-content control (R32) — distributional vs `placebo_shuffled`",
            "",
            "`placebo_shuffled` is byte-structurally identical to the distributional block (same format + "
            "marginal numbers) with the tail VALUES candidate-seeded-DERANGED — so a one-sided "
            "`distributional > placebo_shuffled` win on BOTH metrics isolates the coherent tail SHAPE "
            "(content) from a plausible-looking numeric table (format): the Gupta-Hartford threat. Reported, "
            "DISJOINT from the m=6 union, never a gate.",
            "",
            f"**Verdict: {verdict}.** {sc.get('interpretation', '')}",
            "",
            "| metric | effect (distributional − placebo_shuffled) | one-sided p | reject |",
            "|---|---|---|---|",
            _row("Sharpe", sc.get("sharpe")),
            _row(f"CVaR-{sc.get('cvar_level', 0.05):g}", sc.get("cvar")),
            "",
        ]
    )


def h2_tost_markdown(tost: dict[str, Any]) -> str:
    """Render the headline H2-RA + H2-Tail TOST equivalence bounds (DEEP_H2 §5.3)."""
    if not tost or tost.get("status") != "ok":
        reason = (tost or {}).get("reason", "not computed")
        return f"## H2 TOST equivalence (DEEP_H2 §5.3) — n/a\n\n{reason}\n"
    margin = tost.get("margin", 0.05)
    out = [
        f"## H2 TOST equivalence — the bankable-null bound (±{margin:g}, test-statistic units)",
        "",
        f"Equivalence complement to the H2 conjunction (Lakens 2017): a leg is 'equivalent within ±{margin:g}' "
        "iff its 90% paired-bootstrap CI for the per-seed IQM difference lies inside (−margin, +margin), in "
        f"the test-statistic's OWN units ({tost.get('units', 'per-seed Sharpe / CVaR')}). Lets a non-rejected "
        "leg be reported as a BOUNDED effect, not absence of evidence.",
        "",
        "### H2-RA (Sharpe IQM) equivalence legs",
        "",
        "| contrast | estimate | 90% CI low | 90% CI high | equivalent | n_seeds |",
        "|---|---|---|---|---|---|",
    ]
    for leg in tost.get("ra", []):
        out.append(
            f"| {leg['contrast']} | {leg['estimate']:+.4f} | {leg['ci_low']:+.4f} | "
            f"{leg['ci_high']:+.4f} | {leg['equivalent']} | {leg['n_seeds']} |"
        )
    tail = tost.get("tail", {})
    frac = tost.get("tail_margin_fraction", 0.25)
    out += [
        "",
        f"### H2-Tail (CVaR-{float(tail.get('level', 0.05)):g} IQM) equivalence legs",
        "",
        f"The raw ±{margin:g} band is LARGE vs a daily CVaR magnitude, so a RELATIVE band = {frac:.0%} of the "
        "|baseline (comparator) CVaR| is reported ALONGSIDE it (P6-code) — a scale-appropriate "
        "'within X% of the baseline tail loss' statement. BOTH verdicts shown; neither gates.",
        "",
        "| contrast | estimate | 90% CI low | 90% CI high | equiv (±raw) | baseline CVaR | "
        f"±{frac:.0%}·|base| | equiv (frac) | n_seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for leg in tail.get("legs", []):
        bc = leg.get("baseline_cvar")
        fm = leg.get("margin_fraction_abs")
        out.append(
            f"| {leg['contrast']} | {leg['estimate']:+.4f} | {leg['ci_low']:+.4f} | "
            f"{leg['ci_high']:+.4f} | {leg['equivalent']} | "
            f"{bc:+.4f} | {fm:.4f} | {leg.get('equivalent_fraction')} | {leg['n_seeds']} |"
            if bc is not None and fm is not None else
            f"| {leg['contrast']} | {leg['estimate']:+.4f} | {leg['ci_low']:+.4f} | "
            f"{leg['ci_high']:+.4f} | {leg['equivalent']} | n/a | n/a | n/a | {leg['n_seeds']} |"
        )
    out.append("")
    return "\n".join(out)


def h2_tost_dsr_markdown(t: dict[str, Any]) -> str:
    """Render the DSR-units TOST — the SESOI-units bankable-null bound."""
    if not t or t.get("status") != "ok":
        reason = (t or {}).get("reason", "not computed")
        return f"## H2 TOST equivalence — validation-DSR units (CAMPAIGN_power.md T2.5) — n/a\n\n{reason}\n"
    margin = t.get("margin", 0.05)
    k = t.get("sharpe_to_dsr_factor")
    out = [
        f"## H2 TOST equivalence — validation-DSR units (the SESOI's units; ±{margin:g})",
        "",
        f"The equivalence the power doc requires (CAMPAIGN_power.md T2.5): the FROZEN ±{margin:g} SESOI is in "
        "**validation-DSR** units (the selection metric), so a campaign non-rejection only licenses a "
        "*practical-equivalence* claim if the 90% CI lies inside ±SESOI **in DSR units** — otherwise it is "
        "INCONCLUSIVE, not equivalence (Lakens 2017). Each arm's per-seed Sharpe is mapped to a CONSERVATIVE "
        f"(upper-bound) DSR shift via the documented ceiling ΔDSR_max = φ(0)·√(T−1)/√252·ΔSR_ann "
        f"(factor k = {('%.4f' % k) if k is not None else '?'}, T = {t.get('track_length', '?')}); the bound "
        "is a fortiori true under the exact map. RA-only (the DSR is a Sharpe/PSR statistic — CVaR has no DSR "
        "analogue; that leg stays in the Sharpe-units TOST). Report-only, DISJOINT from the frozen m=6 union.",
        "",
        "| contrast | ΔDSR estimate | 90% CI low | 90% CI high | equivalent (±SESOI) | inconclusive | n_seeds | (ΔSharpe) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for leg in t.get("ra", []):
        out.append(
            f"| {leg['contrast']} | {leg['estimate']:+.4f} | {leg['ci_low']:+.4f} | "
            f"{leg['ci_high']:+.4f} | {leg['equivalent']} | {leg['inconclusive']} | {leg['n_seeds']} | "
            f"{leg['estimate_sharpe']:+.4f} |"
        )
    out.append("")
    return "\n".join(out)


def comparative_es_backtest_markdown(d: dict[str, Any]) -> str:
    """Render the FZ0/(VaR,ES) comparative ES backtest corroborating H2-Tail (DEEP_H2; CH4 §4.7)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Comparative ES backtest (FZ0 + DM; corroborates H2-Tail) — n/a\n\n{reason}\n"
    rs_seed = (d.get("realized_series") or {}).get("seed")
    out = [
        f"## Comparative ES backtest — FZ0 + Diebold-Mariano (level {d.get('level', 0.05):g}; corroborates H2-Tail)",
        "",
        "Nolde-Ziegel (2017) comparative backtest on the jointly-elicitable (VaR, ES) pair (Fissler-Ziegel "
        "2016) over the distributional arm's realized TEST series: model 1 = the distributional arm's own "
        "(VaR, ES) tail forecast, model 2 = the comparator's. `mean_score_diff < 0` (better=model1) = the "
        "distributional tail forecast scores strictly better — the direction that CORROBORATES H2-Tail. "
        "`pvalue_dm_hln` is the conservative small-sample DM p (report-only; DISJOINT, never a gate).",
        "",
        f"Realized series = the MEDIAN-tail seed path (seed {'?' if rs_seed is None else rs_seed}; the seed "
        f"whose empirical CVaR-{d.get('level', 0.05):g} is the median across seeds) — a genuine single "
        "realization, NOT the seed-average (averaging shrinks the tail); forecasts are VALIDATION-estimated "
        "ex-ante (by design).",
        "",
        "| contrast | mean score diff | boot p | DM-HLN p | better | corroborates H2-Tail |",
        "|---|---|---|---|---|---|",
    ]
    for leg in d.get("legs", []):
        out.append(
            f"| {leg['contrast']} | {leg['mean_score_diff']:+.4f} | {leg['pvalue']:.4f} | "
            f"{leg['pvalue_dm_hln']:.4f} | {leg['better']} | {leg['forecast_calibration_favours_dist']} |"
        )
    if d.get("skipped"):
        out += ["", f"Skipped: {', '.join(s['contrast'] for s in d['skipped'])} (missing/degenerate forecast)."]
    out.append("")
    return "\n".join(out)


def bayesian_null_markdown(d: dict[str, Any]) -> str:
    """Render the Bayesian evidence-for-the-null complement to the TOST (R67; DEEP_H2 §5.4)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Bayesian evidence-for-the-null (BF01 + ROPE) — n/a\n\n{reason}\n"
    sesoi = d.get("sesoi", 0.05)
    frac = float(d.get("tail_margin_fraction", 0.25))
    out = [
        f"## Bayesian evidence-for-the-null — BF01 + posterior-in-ROPE (SESOI ±{sesoi:g})",
        "",
        "The Bayesian complement to the frozen TOST (R67): a JZS Bayes factor (BF01 = evidence FOR practical "
        "equivalence) + the 90% HDI-in-ROPE check on the PAIRED per-seed difference scores (ROPE = ±SESOI). "
        "`robust_for_null` requires BF01 ≥ 3 across the whole prior grid. Report-only, DISJOINT — never gates.",
        "",
        "### H2-RA (Sharpe)",
        "",
        "| contrast | verdict | BF01 | effect | HDI⊂ROPE | robust_for_null | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for leg in d.get("ra", []):
        out.append(
            f"| {leg['contrast']} | {leg['verdict']} | {leg['bf01']:.2f} | {leg['effect']:+.4f} | "
            f"{leg['hdi_in_rope']} | {leg['robust_for_null']} | {leg['n']} |"
        )
    out += [
        "",
        f"### H2-Tail (CVaR-{d.get('level', 0.05):g})",
        "",
        f"The raw ±{sesoi:g} ROPE is in RAW CVaR units — LARGE vs a daily CVaR magnitude O(0.01–0.06), so it "
        f"can near-trivially contain the posterior (the h2_tost P6 concern). A RELATIVE ROPE = "
        f"{frac:.0%}·|baseline (comparator) CVaR| is therefore reported ALONGSIDE — only the ROPE-dependent "
        "fields move (the BF01 is ROPE-free). BOTH shown; neither gates.",
        "",
        "| contrast | verdict (raw ROPE) | BF01 | effect | HDI⊂ROPE (raw) | robust_for_null | baseline CVaR "
        "| rel ROPE mass | HDI⊂rel-ROPE | verdict (rel ROPE) | n |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for leg in d.get("tail", {}).get("legs", []):
        rel = leg.get("relative") or {}
        bc = leg.get("baseline_cvar")
        bc_s = "n/a" if bc is None else f"{bc:+.4f}"
        if "verdict" in rel:
            rel_cells = f"{bc_s} | {rel['rope_mass']:.3f} | {rel['hdi_in_rope']} | {rel['verdict']}"
        else:
            rel_cells = f"{bc_s} | n/a | n/a | {rel.get('status', 'n/a')}"
        out.append(
            f"| {leg['contrast']} | {leg['verdict']} | {leg['bf01']:.2f} | {leg['effect']:+.4f} | "
            f"{leg['hdi_in_rope']} | {leg['robust_for_null']} | {rel_cells} | {leg['n']} |"
        )
    out.append("")
    return "\n".join(out)


def model_confidence_set_markdown(d: dict[str, Any]) -> str:
    """Render the Model Confidence Set over the arms (Hansen et al. 2011; report-only)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Model Confidence Set (Hansen et al. 2011) — n/a\n\n{reason}\n"
    out = [
        "## Model Confidence Set — which arms are statistically INDISTINGUISHABLE (Hansen et al. 2011)",
        "",
        f"The multiplicity-honest complement to the pairwise IUTs (MCS level {d.get('size', 0.1):g}): the set of "
        "arms not eliminated at the level. Under the predicted null the set contains (almost) all arms. "
        "Report-only, DISJOINT — never gates the frozen m=6.",
        "",
    ]
    for title, mcs in (("Sharpe", d.get("sharpe", {})), (f"CVaR-{d.get('tail', {}).get('level', 0.05):g}", d.get("tail", {}))):
        if mcs.get("status") == "ok":
            out += [
                f"- **{title}:** included = {{{', '.join(mcs.get('included', []))}}}; "
                f"best = `{mcs.get('best_arm')}` (in set: {mcs.get('best_in_set')}); n_seeds = {mcs.get('n_seeds')}",
            ]
        else:
            out += [f"- **{title}:** {mcs.get('status', 'n/a')} ({mcs.get('reason', '')})"]
    out.append("")
    return "\n".join(out)


def dsr_effective_n_markdown(d: dict[str, Any]) -> str:
    """Render the DSR raw-N vs effective-N sensitivity (DEEP_STATS A1)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## DSR effective-N sensitivity (DEEP_STATS A1) — n/a\n\n{reason}\n"
    out = [
        "## DSR sensitivity — raw N vs effective N under sequential-reflective correlation (DEEP_STATS A1)",
        "",
        f"Winner arm `{d.get('arm')}`. The reflect-on-best search makes the N={d.get('n_trials')} candidates "
        "correlated (mean pairwise validation-return correlation ρ̄), so the i.i.d. expected-max-Sharpe "
        "assumption is violated. N_eff = N/(1+(N−1)·ρ̄) is the mean-correlation surrogate (ONC is canonical).",
        "",
        f"- ρ̄ (mean off-diagonal candidate correlation): **{d.get('rho_bar', 0):.4f}**",
        f"- N (naïve) = {d.get('n_trials')} → DSR = **{d.get('dsr_raw_n', 0):.4f}**",
        f"- N_eff = {d.get('n_eff')} → DSR = **{d.get('dsr_eff_n', 0):.4f}**",
        "",
        "Direction is benign: ρ̄>0 ⇒ N_eff<N ⇒ smaller deflation ⇒ higher DSR, so the naïve N is the "
        "CONSERVATIVE choice and any floor/H1 PASS at naïve N is robust to the trial-count dispute "
        "(DEEP_STATS A8).",
        "",
    ]
    return "\n".join(out)


def delisting_band_markdown(b: dict[str, Any]) -> str:
    """Render the delisting-return sensitivity band (R33; PREREGISTRATION §7)."""
    if not b or b.get("status") != "ok":
        reason = (b or {}).get("reason", "not computed")
        return f"## Delisting-return sensitivity band (R33; §7) — n/a\n\n{reason}\n"
    tw = b.get("test_window", ["?", "?"])
    out = [
        "## Delisting-return sensitivity band (R33; PREREGISTRATION §7) — report-only, DISJOINT",
        "",
        f"Panel `{b.get('headline_panel', '?')}`; test window {tw[0]}..{tw[1]}; "
        f"{b.get('n_delisting_cells_in_test', 0)} delisting cells fall in the test window. The POOLED "
        "test-window CVaR is recomputed with those cells set to each delisting return `d` (DATA-level; no "
        "policy re-run). The band **BRACKETS** the tail, and ADR-051 located the truth AT the 0% end: the "
        "executed observed-terminal recovery (univ5s) kept the vendor terminal for all 333 dead names with "
        "ZERO surcharges, so the corrected panel is byte-identical to the zero-fill headline (`d=0.0` ≈ the "
        "univ3 / liquidate_to_cash end); `univ4` (−30/−55) is the **heavy end and doubly wrong** — "
        "M&A-contaminated (Refinitiv's vault carries no delisting reason → the surcharge hits 100% of "
        "delistings, INCLUDING premium M&A — ABMD→J&J, ALTR→Intel, CELG→BMS, RHT→IBM — booked at a "
        "fabricated −30/−55% loss) AND a terminal double-count on top of returns the vendor series already "
        "books. The whole sweep typically moves the "
        "pooled CVaR-5% by only ~2% (measured pre-Split-C; re-derived on the executed window), so the "
        "headline tail **ordering is invariant** across it. DISJOINT "
        "from the frozen m=6 union; never a gate.",
        "",
        "| delisting return d | pooled CVaR-5% | pooled CVaR-1% | note |",
        "|---|---|---|---|",
    ]
    if b.get("window_clamped"):
        wr = b.get("window_requested", ["?", "?"])
        out.insert(3, f"NOTE: window clamped from the requested {wr[0]}..{wr[1]}: {b.get('window_clamp_note', '')}")
        out.insert(4, "")
    for row in b.get("rows", []):
        d = float(row.get("d", 0.0))
        c5 = row.get("cvar_05")
        c1 = row.get("cvar_01")
        c5_s = "n/a" if c5 is None else f"{float(c5):.4f}"
        c1_s = "n/a" if c1 is None else f"{float(c1):.4f}"
        note = "univ4 (−30/−55; heavy END, M&A-contaminated upper bracket — NOT the tail)" \
            if row.get("is_headline_extreme") else (
            "univ3 / liquidate_to_cash (0% zero-fill END — ≡ the ADR-051 corrected panel univ5s)" if d == 0.0
            else ("total-loss floor (d=−100%)" if d == -1.0 else "")
        )
        out.append(f"| {d:+.2f} | {c5_s} | {c1_s} | {note} |")
    out.append("")
    return "\n".join(out)


def cross_hypothesis_multiplicity_markdown(m: dict[str, Any]) -> str:
    """Render the Bonferroni-across-4 cross-hypothesis sensitivity (DEEP_STATS A4)."""
    if not m or not m.get("rows"):
        return ""
    out = [
        "## Cross-hypothesis multiplicity — Bonferroni-across-4 SENSITIVITY (DEEP_STATS A4; report-only)",
        "",
        m.get("stance", ""),
        "",
        f"Programme-wide Bonferroni level α/{m.get('n_hypotheses', 4)} = {m.get('bonferroni_alpha', 0.0125):.4f} "
        f"(per-hypothesis α = {m.get('alpha', 0.05):.2f}).",
        "",
        "| hypothesis | headline p | primary decision | survives Bonferroni-4 | note |",
        "|---|---|---|---|---|",
    ]
    for r in m.get("rows", []):
        hp = r.get("headline_p")
        hp_s = "—" if hp is None else f"{float(hp):.4f}"
        sb = r.get("survives_bonferroni")
        sb_s = "n/a" if sb is None else str(sb)
        out.append(
            f"| {r['hypothesis']} | {hp_s} | {r.get('decision_primary', '?')} | {sb_s} | {r.get('note', '')} |"
        )
    out.append("")
    return "\n".join(out)


def mechanism_multiplicity_markdown(m: dict[str, Any]) -> str:
    """Render the Bonferroni/BH-across-mechanism-legs forking-paths sensitivity (report-only)."""
    if not m or not m.get("rows"):
        return ""
    n_p = m.get("n_p_tests", 0)
    out = [
        "## Mechanism multiplicity — Bonferroni/BH-across-mechanism-legs SENSITIVITY (forking-paths; report-only)",
        "",
        m.get("stance", ""),
        "",
        (
            f"Over the {n_p} p-bearing mechanism leg(s): Bonferroni level α/{n_p} = "
            f"{m.get('bonferroni_alpha', m.get('alpha', 0.05)):.4f} (per-leg α = {m.get('alpha', 0.05):.2f}); "
            f"BH at q = {m.get('q', 0.05):.2f}."
            if n_p else
            "No mechanism leg emitted an inferential p in this run (the CI-based legs are decided by "
            "bootstrap CIs; the p-bearing structural legs need the NAMED authoring pass, executed=False here)."
        ),
        "",
        "| mechanism leg | p | has_p | decision | survives Bonferroni | reject BH | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in m.get("rows", []):
        p = r.get("p")
        p_s = "—" if p is None else f"{float(p):.4f}"
        sb = r.get("survives_bonferroni")
        rb = r.get("reject_bh")
        out.append(
            f"| {r['leg']} | {p_s} | {r.get('has_p')} | {r.get('decision') or '—'} | "
            f"{'n/a' if sb is None else sb} | {'n/a' if rb is None else rb} | {r.get('note', '')} |"
        )
    out.append("")
    return "\n".join(out)


def reward_taxonomy_markdown(d: dict[str, Any]) -> str:
    """Render the induced reward-program taxonomy + per-arm kind composition (report-only, DISJOINT)."""
    if not d or d.get("status") != "ok":
        reason = (d or {}).get("reason", "not computed")
        return f"## Reward-program taxonomy (induced kinds; report-only) — n/a\n\n{reason}\n"
    pooled = d.get("pooled", {})
    kind_arms = d.get("kind_arms", {})
    out = [
        f"## Reward-program taxonomy — {pooled.get('n_kinds', 0)} kind(s) over "
        f"{pooled.get('n_programs', 0)} authored program(s) (report-only, DISJOINT)",
        "",
        "The CH7 'left to future work' instrument, delivered: kinds are connected components of the "
        "identifier-invariant AST shape-set Jaccard graph at similarity ≥ "
        f"{pooled.get('sim_threshold', 0.6):g} (depth-{pooled.get('depth', 4)} canonical subtree shapes, "
        "the reward-code-distance signature; unparseable/empty sources EXCLUDED and counted, never "
        "clustered). Labels = the majority construct combination within the kind; 'unlabelled' is the "
        "honest fallback. DISJOINT from the frozen m=6 family — never gates H1-H4.",
        "",
        "| kind | size | label | medoid exemplar | mean within-sim | arms |",
        "|---|---|---|---|---|---|",
    ]
    for k in pooled.get("kinds", []):
        mw = k.get("mean_within_similarity")
        out.append(
            f"| {k['kind_id']} | {k['size']} | {k['label']} | `{k['medoid']}` | "
            f"{'—' if mw is None else f'{mw:.3f}'} | {', '.join(kind_arms.get(k['kind_id'], [])) or '—'} |"
        )
    out += [
        "",
        f"Unparseable/empty sources excluded: **{pooled.get('n_unparseable', 0)}** "
        f"(records with no archived source: {d.get('n_missing_source', 0)}); "
        f"singleton kinds: {pooled.get('n_singletons', 0)}.",
        "",
        "### Per-arm kind composition (do arms author different KINDS, or the same kinds reshaped?)",
        "",
        "| arm | programs | unparseable | kinds present | entropy (bits) | composition |",
        "|---|---|---|---|---|---|",
    ]
    for arm, e in (d.get("per_arm") or {}).items():
        comp = ", ".join(
            f"{kid}×{cnt}" for kid, cnt in
            sorted(e.get("kind_counts", {}).items(), key=lambda kv: (-kv[1], kv[0]))
        ) or "—"
        ent = e.get("entropy_bits")
        out.append(
            f"| {arm} | {e.get('n_programs', 0)} | {e.get('n_unparseable', 0)} | "
            f"{e.get('n_kinds_present', 0)} | {'—' if ent is None else f'{ent:.3f}'} | {comp} |"
        )
    overlap = d.get("kind_overlap") or {}
    if overlap:
        out += ["", "Kind-set overlap between arm pairs (Jaccard over each arm's set of kinds):", ""]
        out += ["| arm pair | overlap |", "|---|---|"]
        out += [f"| {pair} | {val:.3f} |" for pair, val in overlap.items()]
    sens = d.get("sensitivity") or {}
    if sens.get("status") == "ok":
        n_kinds_s = ", ".join(
            f"τ={row['threshold']:g}: {row['n_kinds']}" for row in sens.get("by_threshold", [])
        )
        stab_parts = []
        for row in sens.get("adjacent_stability", []):
            ri = row.get("rand_index")
            stab_parts.append(f"{row['pair']}: {'—' if ri is None else f'{ri:.3f}'}")
        out += [
            "",
            f"Threshold sensitivity — n_kinds by threshold: {n_kinds_s}; adjacent-threshold stability "
            f"(pair-counting Rand index): {', '.join(stab_parts) or '—'}.",
            "",
        ]
    else:
        out.append("")
    return "\n".join(out)


def evt_consistency_markdown(g: dict[str, Any]) -> str:
    """Render the EVT-estimator-consistency guard for the fed CVaR levels (DEEP_H2 §6.3)."""
    if not g or g.get("status") != "ok":
        reason = (g or {}).get("reason", "not computed")
        return f"## EVT-consistency guard (DEEP_H2 §6.3) — n/a\n\n{reason}\n"
    verdict = "CONSISTENT" if g.get("all_consistent") else "INCONSISTENT (flagged — see log)"
    out = [
        f"## EVT-consistency guard — fed CVaR estimator across tail-fed arms — {verdict}",
        "",
        "Re-derived per arm winner's validation distribution: which estimator the FED CVaR level routes to "
        "('evt' / 'empirical' / 'empirical(fallback)' = the `alpha>fu`/degenerate-fit fallback in "
        "`_evt_cvar`). Inconsistency across tail-fed arms means the distributional-vs-scalar_cvar5 tail "
        "comparison mixes estimators (DEEP_H2 §6.3). Report-only — logged, never raised.",
        "",
        "| arm | " + " | ".join(f"CVaR {lvl:g} path" for lvl in g.get("levels", [])) + " |",
        "|---|" + "---|" * len(g.get("levels", [])),
    ]
    for arm, paths in g.get("per_arm", {}).items():
        cells = " | ".join(paths.get(f"{lvl:g}", "—") for lvl in g.get("levels", []))
        out.append(f"| {arm} | {cells} |")
    cons = g.get("consistent", {})
    out += [
        "",
        "Per-level consistency: " + ", ".join(f"CVaR {k} = {v}" for k, v in cons.items()) + ".",
        "",
    ]
    return "\n".join(out)


def write_report(result: dict[str, Any], root: str | Path) -> Path:
    """Write ``campaign_overfitting.md`` + ``campaign_overfitting.json`` next to the archive."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    md = pbo_markdown(result["pbo"], n_blocks=int(result["n_blocks"]))
    if result.get("pbo_dsr"):
        md = md + "\n" + pbo_dsr_markdown(result["pbo"], result["pbo_dsr"], n_blocks=int(result["n_blocks"]))
    if result.get("winner_dsr"):
        md = md + "\n" + winner_dsr_markdown(result["winner_dsr"])
    if result.get("h2"):
        md = md + "\n" + h2_markdown(result["h2"])
    if result.get("h2_tost"):
        md = md + "\n" + h2_tost_markdown(result["h2_tost"])
    if result.get("h2_tost_dsr"):
        md = md + "\n" + h2_tost_dsr_markdown(result["h2_tost_dsr"])
    if result.get("comparative_es_backtest"):
        md = md + "\n" + comparative_es_backtest_markdown(result["comparative_es_backtest"])
    if result.get("bayesian_null_report"):
        md = md + "\n" + bayesian_null_markdown(result["bayesian_null_report"])
    if result.get("model_confidence_set"):
        md = md + "\n" + model_confidence_set_markdown(result["model_confidence_set"])
    if result.get("h2_structure"):
        md = md + "\n" + h2_structure_markdown(result["h2_structure"])
    if result.get("h2_rf_robustness"):
        md = md + "\n" + h2_rf_robustness_markdown(result["h2_rf_robustness"])
    if result.get("h3"):
        md = md + "\n" + h3_markdown(result["h3"])
    if result.get("h4"):
        md = md + "\n" + h4_markdown(result["h4"])
    if result.get("dsr_effective_n"):
        md = md + "\n" + dsr_effective_n_markdown(result["dsr_effective_n"])
    if result.get("delisting_band"):
        md = md + "\n" + delisting_band_markdown(result["delisting_band"])
    if result.get("evt_consistency"):
        md = md + "\n" + evt_consistency_markdown(result["evt_consistency"])
    if result.get("divergence"):
        md = md + "\n" + divergence_markdown(result["divergence"])
    if result.get("compute_accounting"):
        md = md + "\n" + compute_accounting_markdown(result["compute_accounting"])
    if result.get("responsiveness"):
        md = md + "\n" + responsiveness_markdown(result["responsiveness"])
    if result.get("mediation"):
        md = md + "\n" + mediation_markdown(result["mediation"])
    if result.get("regime_stratified"):
        from src.inference.regime_analysis import render_regime_table

        md = md + "\n## Regime-stratified tail/return metrics (T3'; ADR-039; report-only, DISJOINT)\n\n" \
            + render_regime_table(result["regime_stratified"]) + "\n"
    if result.get("named_vs_blinded_structural"):
        md = md + "\n" + named_vs_blinded_structural_markdown(result["named_vs_blinded_structural"])
    if result.get("legible_format_responsiveness"):
        md = md + "\n" + legible_format_responsiveness_markdown(result["legible_format_responsiveness"])
    if result.get("reward_taxonomy"):
        md = md + "\n" + reward_taxonomy_markdown(result["reward_taxonomy"])
    if result.get("information_gap"):
        md = md + "\n" + information_gap_markdown(result["information_gap"])
    if result.get("validation_headroom"):
        md = md + "\n" + validation_headroom_markdown(result["validation_headroom"])
    if result.get("cross_hypothesis_multiplicity"):
        md = md + "\n" + cross_hypothesis_multiplicity_markdown(result["cross_hypothesis_multiplicity"])
    if result.get("mechanism_multiplicity"):
        md = md + "\n" + mechanism_multiplicity_markdown(result["mechanism_multiplicity"])
    if result.get("benchmark_floor"):
        md = md + "\n" + benchmark_floor_markdown(result["benchmark_floor"])
    if result.get("h1_beat_human"):
        md = md + "\n" + h1_beat_human_markdown(result["h1_beat_human"])
    if result.get("attribution"):
        from src.inference.attribution import attribution_markdown

        md = md + "\n" + attribution_markdown(result["attribution"])
    if result.get("variance"):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import variance_decomposition as _vd

        md = md + "\n" + _vd.verdict_markdown(result["variance"])
    report = root / "campaign_overfitting.md"
    report.write_text(md, encoding="utf-8")
    (root / "campaign_overfitting.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> None:
    # The report (and argparse --help) print Unicode arrows/>= glyphs; on a non-UTF-8 console
    # (e.g. Windows cp1251) that raises UnicodeEncodeError and crashed even `--help`. Reconfigure
    # stdout to UTF-8 with replacement BEFORE parse_args so both the help and the analysis output
    # render everywhere (same guard build_paper.py uses). No-op where stdout is already UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream (pytest capsys)
        pass
    p = argparse.ArgumentParser(description="Campaign PBO/CSCV per-arm overfitting analysis.")
    p.add_argument(
        "--root",
        default=None,
        help=(
            "Campaign archive root holding search/, test/, frozen/ + campaign_summary.json "
            "(default: outputs/campaign; falls back to the prototype output_dir, PBO/DSR-only)."
        ),
    )
    p.add_argument("--n-blocks", type=int, default=None, help="CSCV blocks S (default: inference.yaml pbo.n_blocks).")
    p.add_argument(
        "--single-shot-root",
        default=None,
        help=(
            "Optional H3 single-shot archive root (D-2: test_h3_singleshot/<arm> with the single-shot "
            "winner's test records). When present, H3 (iterative vs single-shot) is computed; absent ⇒ "
            "H3 is skipped (DEEP_H3)."
        ),
    )
    p.add_argument(
        "--variance-runs",
        nargs="+",
        default=None,
        help=(
            ">=2 independent search re-run roots to identify the reward-draw variance sigma^2_search "
            "(reviewer attack #10). Omit to skip the variance appendix."
        ),
    )
    args = p.parse_args()

    if args.root is None:
        from src.utils.config import load_config

        # Default to the CAMPAIGN output_dir (the run_campaign.py archive that carries search/<arm>/<cand>
        # with val_returns AND test/<arm>/<seed> with test_returns AND campaign_summary.json) so a single
        # --root yields the FULL headline report (PBO + DSR + H2 + floor) — the loader now walks both legs.
        # The campaign config has no output_dir key (run_campaign.py pins "outputs/campaign"); fall back to
        # the prototype single-tree (PBO/DSR-only — it has no test leg, so H2/floor stay empty there) when
        # the campaign archive is absent. (leg-disjoint single-root defect fix.)
        campaign_root = str(load_config("campaign").get("output_dir", "outputs/campaign"))
        if Path(campaign_root).is_dir():
            args.root = campaign_root
        else:
            args.root = str(load_config("prototype").get("output_dir", "outputs/prototype"))

    # Panel-dependent DeMiguel benchmark floor: read the resolved test_window from the campaign summary
    # and load the panel the SAME way the campaign's test stage does (development-phase PIT universe to the
    # 2025 eval end). GUARDED — a records-only archive (no summary / no gold panel) runs without the floor.
    panel = cfg = test_window = winner_n_trials = None
    try:
        from src.data.loaders import load_gold_panel
        from src.utils.config import load_config

        summary = json.loads((Path(args.root) / "campaign_summary.json").read_text(encoding="utf-8"))
        _tw = summary["test_window"]
        test_window = (int(_tw[0]), int(_tw[1]))
        cfg = load_config("environment")
        # FALLBACK only: analyze() derives the winner search multiplicity from the records (the count
        # winner_dsr uses) so the floor stays consistent. main() reads campaign_summary.json (written by
        # run_campaign), so the matching budget is the CAMPAIGN's candidates_per_arm (30) — NOT the
        # prototype's n_trials (40); using the latter mis-deflated the campaign gate (#3, 2026-06-20).
        winner_n_trials = int(load_config("campaign").get("candidates_per_arm", 1) or 1)
        # Eval-span END from config (the frozen single source), NOT a hardcoded date (no-hardcoding audit).
        _span = load_config("inference").get("splits", {}).get("evaluation", {}).get("span", [None, "2026-06-30"])
        # Checksum-verified against the frozen manifest (2026-07-26 review): this panel feeds the
        # BENCHMARK-FLOOR exhibit (H1/T0), a confirmatory-bearing quantity, and every other path that
        # reads real gold for a confirmatory purpose already verifies (run_campaign.py:1474,
        # run_campaign_cluster.py:113, parallel.py:291, test_leg.py:202). Analysis must not be the one
        # place that scores the campaign against an UNVERIFIED panel. A mismatch raises and is reported
        # by the handler below as a loud, named SKIP rather than a silently-wrong floor.
        panel = load_gold_panel(phase="development", end=str(_span[1]), verify_checksum=True).panel
    except Exception as _floor_exc:  # noqa: BLE001 - floor is best-effort; records-only analysis always runs
        # P7 (2026-07-13 audit): was a SILENT blanket except — the benchmark-floor exhibit vanished
        # without a word when campaign_summary.json was absent (the cluster mirror never had one).
        # NOTE: 2026-08-01 (RUN 11, analysis M166): this message named ONLY the benchmark floor, but
        # FOUR registered outputs live inside the `panel is not None` block and all four vanish
        # together. A reader who saw the warning still did not learn what else had gone.
        print(f"[analyze] PANEL-DEPENDENT OUTPUTS SKIPPED ({type(_floor_exc).__name__}: "
              f"{_floor_exc}) — records-only analysis proceeds, but FOUR registered outputs will "
              f"be ABSENT from the report: benchmark_floor (the DeMiguel 1/N floor — the "
              f"'nine published allocators' table wired into the PDF), attribution, "
              f"h2_rf_robustness, regime_stratified. To enable them, write campaign_summary.json "
              f"at the archive root — at teardown, via the campaign-summary writer, which "
              f"derives the windows through the campaign's own code path. NEVER hand-author it: a "
              f"wrong test_window scores the floor on the WRONG SLICE, silently, which is worse "
              f"than having no floor at all.", flush=True)
        panel = cfg = test_window = winner_n_trials = None

    result = analyze(
        args.root, n_blocks=args.n_blocks, panel=panel, cfg=cfg,
        test_window=test_window, winner_n_trials=winner_n_trials,
        variance_run_roots=args.variance_runs,
        single_shot_root=args.single_shot_root,
    )
    # ---- REGISTERED-KEY COMPLETENESS (2026-08-01, RUN 11) -----------------------------------
    # Until today the only signal that a registered output had vanished was a print, and the
    # analysis exited 0 regardless. Four of them went missing at once and nothing downstream
    # noticed. Name every absence, and make an UNEXPLAINED one fail.
    explained, unexplained = missing_output_keys(result)
    print(f"[analyze] registered outputs: {len(REGISTERED_OUTPUT_KEYS) - len(explained) - len(unexplained)}"
          f"/{len(REGISTERED_OUTPUT_KEYS)} present")
    for key, precondition in sorted(explained.items()):
        print(f"[analyze]   ABSENT (explained): {key} — requires {precondition}")
    if unexplained:
        print(f"[analyze] REGISTERED-OUTPUT DEFECT: {len(unexplained)} output(s) absent with NO "
              f"stated precondition — a defect in the analysis, not a configuration: "
              f"{', '.join(unexplained)}. The report is still written below so it can be "
              f"inspected; this run exits NON-ZERO.", file=sys.stderr, flush=True)

    report = write_report(result, args.root)
    print(f"[analyze_campaign] PBO/CSCV (S={result['n_blocks']}) over {result['n_records']} records -> {report}")
    for arm, e in result["pbo"].items():
        pbo_str = "n/a" if e.get("pbo") is None else f"{e['pbo']:.3f}"
        print(f"  {arm:>14}: PBO={pbo_str} (n={e.get('n_candidates', 0)}, T={e.get('t_val', 0)}, {e.get('status')})")
    for arm, e in result.get("winner_dsr", {}).items():
        if e.get("status") == "ok":
            print(
                f"  {arm:>14}: DSR canonical={e['dsr_canonical']:.4f} vs proxy={e['dsr_proxy']:.4f} "
                f"(var_sr={e['var_sr']:.4f}, n={e['n_candidates']})"
            )
    h2 = result.get("h2") or {}
    if h2.get("error"):
        print(f"  H2: ERROR — {h2['error']}")
    elif h2.get("H2_supported") is not None:
        # R25: two co-primary IUTs. Print the two-tier verdict, then per-leg one-sided decisions.
        method = str(h2.get("method", "bh")).upper()
        print(f"  H2 (distributional feedback, {method}): {h2.get('verdict', '?')}")
        ra_ok = (h2.get("H2_RA") or {}).get("supported", h2.get("H2_supported"))
        tail = h2.get("H2_Tail") or {}
        print(f"      H2-RA   (risk-adjusted, Sharpe IUT): {'SUPPORTED' if ra_ok else 'NOT supported'}")
        print(
            f"      H2-Tail (tail outcome, CVaR-{float(tail.get('level', 0.05)):g} IUT): "
            f"{'SUPPORTED' if tail.get('supported') else 'NOT supported'}"
        )
        for tier, legs in (("RA", h2.get("legs", [])), ("Tail", h2.get("tail_legs", []))):
            for leg in legs:
                pv = leg.get("pvalue_one_sided")
                pv_s = f" p1={pv:.4f}" if isinstance(pv, float) else ""
                print(f"      [{tier:>4}] {leg['contrast']:>34}: leg_supported={leg.get('leg_supported')}{pv_s}")
        if h2.get("missing"):
            print(f"      missing (unsupported): {', '.join(h2['missing'])}")

    # Deferred to the very end so the report and every diagnostic above are produced first: a run
    # that fails this gate is exactly the run whose output someone needs to read.
    if unexplained:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
