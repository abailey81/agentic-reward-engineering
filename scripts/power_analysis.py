"""Statistical power analysis for the HEADLINE H2 test (FINAL_PLAN B-5; viva Q21; PREREGISTRATION §10 R12/R16).

Purpose
-------
Compute the design-time power numbers that justify the campaign size and write them into
``outputs/CAMPAIGN_power.md``. This runs BEFORE the Phase-1 freeze so the pre-registered design is
defensible: if the headline H2 returns a non-rejection, we report it as a **bounded effect** (an
effect smaller than the minimum detectable effect, MDE, AND — if the TOST CI is inside the SESOI —
practically equivalent), rather than an underpowered failure.

The REALIZED headline test this powers (the fix — R16)
------------------------------------------------------
The frozen inference (``config/preregistration.yaml: inference.difference_test_unit =
per_seed_iqm_paired_seed_bootstrap``; PREREGISTRATION §10 amendment R16) is, EXACTLY:

  * Each per-arm winner is re-run at ``n_seeds`` training seeds (Amendment D2: ``n_seeds = 30``).
  * Each seed's frozen-winner TEST return series is reduced to ONE per-seed score — the annualised
    Sharpe (the headline H2-conjunction leg, R13) or the CVaR at the tail level (the tail leg).
  * The per-seed scores are reduced to an **rliable IQM** point estimate per arm (Agarwal et al.
    2021) and compared by ``src.inference.bootstrap.paired_seed_difference_test(a, b,
    statistic=iqm, ...)`` — a re-centred basic empirical bootstrap that **resamples the shared
    training-seed indices** (paired across arms; the across-seed/training-RNG variance is carried,
    the common variance cancels) — consuming **exactly ``n_seeds`` paired per-seed scores**.

So the realized resampling unit is **seeds-on-per-seed-scores** and the sample size is ``n_seeds``
(30), **NOT** ``seeds x folds x N_regimes``. This module's power model is now faithful to that test:
:func:`simulate_power` draws ``n`` correlated per-seed score PAIRS and rejects via the REAL
``paired_seed_difference_test`` (no re-implementation of the decision rule), so the size and the
small-``n`` conservativeness of the actual test are reproduced rather than approximated.

(The SUPERSEDED model this file used to implement — an unpaired, seed-AVERAGED-series block bootstrap
on ``n_eff = seeds x folds x N`` iid cells — overstated the realized degrees of freedom ~6x at N=6
and modelled a test that is no longer run. That is gone; the regime count ``N`` is still computed and
reported because it bounds the SECONDARY regime-stratified analyses, but it does **not** enter the
headline paired ``n``.)

The honest noise input σ_seed (DIRECTIONAL upper bound)
------------------------------------------------------
The relevant noise is the **seed-to-seed standard deviation of the per-seed score within an arm**
(``sigma_seed``). The clean campaign pilot (winners re-run at 30 seeds with the reward FIXED) does not
exist yet, so we anchor σ_seed on the only realized dispersion available: the prototype archive
(``outputs/prototype``). There, each SEARCH candidate is trained at 1 seed, so the **per-candidate
within-arm dispersion of the per-seed annualised Sharpe is ≈ 0.36** (pooled across arms; the headline
``distributional``/``scalar`` arms are 0.36/0.37; ``scripts/power_analysis`` derivation, reproduced in
``outputs/CAMPAIGN_power.md``). This is a **DIRECTIONAL UPPER BOUND** on the true σ_seed for two reasons:
(1) it mixes reward-DESIGN variance (a different reward per candidate) WITH the 1-seed sampling
variance, whereas the campaign's σ_seed holds the reward fixed and varies only the seed; (2) it was
measured at 25k steps on the prototype's shorter validation window, not the campaign's 50k-step sealed
test leg. A larger σ_seed inflates the MDE, so the reported MDE is a **pessimistic (upper-bound) MDE**
on the σ axis — the realized campaign should detect at least as small an effect. ``DIRECTIONAL_SIGMA_SEED``
records this proxy; ``--sigma-seed`` overrides it with the clean pilot value when it exists.

External cross-check (Colas et al. 2019, "A Hitchhiker's Guide to Statistical Comparisons of RL
Algorithms", arXiv:1904.06979). For a two-sample comparison of central RL performance across seeds at
**80% power**, they report (their guidelines): standardized effect δ = |Δμ|/σ_pool of **δ=0.5 → N≈100,
δ=1 → N≈20, δ=2 → N≈5–10** (and a real SAC-vs-TD3 case needed N≈10–15 at δ≈0.93). Our paired test on
**N=30** seeds therefore lands its 80%-power MDE near **δ≈0.8–1.0 σ**, consistent with the simulated
table below. Colas also flags that the *two-sample* bootstrap is anti-conservative below N≈50; our test
is the **paired/one-sample** variant (it cancels the common across-seed variance), and the simulated
null rejection here is conservative (≤ the selection-aware α), not inflated — the small-``n``
conservativeness is reported, not hidden.

The pairing correlation ρ (swept; default conservative)
------------------------------------------------------
The paired test gains power when the two arms' per-seed scores are POSITIVELY correlated across the
shared seed (a "lucky" seed lifts both arms, so the paired difference has lower variance). The true ρ
is unknown pre-campaign, so :func:`simulate_power` takes ``rho`` and the design sweeps
ρ∈{-0.141, 0, 0.3, 0.5, 0.7} (the negative point is the MEASURED pilot value). NOTE: CORRECTED
2026-07-26 (deep review): ρ=0 is NOT the conservative worst case for THIS design. Since
σ_D = σ_seed·√(2(1-ρ)), a NEGATIVE ρ gives MORE paired variance than ρ=0, and the pilot measured
ρ = -0.141 (σ_seed 0.244 → σ_D 0.3686, matching the registered 0.369, vs only 0.3451 at ρ=0). So
ρ=0 UNDERSTATES σ_D and hence the MDE by ~7% here — it is ANTI-conservative, and the previously
shipped sweep never exercised a negative ρ, so no reported row covered the actual design point.
The honest statement: any real ρ>0 shrinks the MDE, any ρ<0 inflates it, and the MEASURED ρ is
the row to read.

Null framing (pre-committed; PREREGISTRATION §10 R12)
----------------------------------------------------
A non-rejection of H2 is reported as **non-rejection = (the effect is bounded below the MDE at the
target power and Šidák-α) AND (the TOST 90% bootstrap CI for the per-seed-score difference lies inside
±SESOI, i.e. the arms are practically equivalent within ±0.05)**. Neither alone; both together turn a
null into a credible bounded-effect finding rather than an underpowered shrug. (Units caveat: the
frozen SESOI/TOST margin ±0.05 is specified in *validation-DSR* units — the selection metric — whereas
the difference TEST runs on the per-seed annualised Sharpe and on CVaR; the TOST helper here applies
the ±margin in the SAME units as the score arrays it is handed, so feed it the metric the equivalence
claim is about. This unit distinction is stated, not silently conflated.)

Sharpe ↔ validation-DSR reconciliation (T2.5)
--------------------------------------------
Because the MDE is in annualised-Sharpe but the SESOI is in validation-DSR, :func:`sharpe_mde_to_dsr`
maps the Sharpe-MDE into validation-DSR units via a documented CONSERVATIVE (upper-bound) Sharpe→DSR
ceiling (delta method on the PSR `Φ(z)` at its maximum-sensitivity point `z=0`, normal returns,
``ΔDSR_max = φ(0)·√(T−1)/√252·ΔSR_ann``; T = the validation track length). The doc reports the MDE in BOTH
units. The reconciliation makes the unit gap quantitative and pins the **INCONCLUSIVE branch**: at the
campaign's T≈756 the ceiling EXCEEDS 0.05, so a Sharpe non-rejection alone does NOT license a DSR
equivalence claim — only a TOST CI evaluated **in validation-DSR units** inside ±SESOI does; otherwise the
result is reported INCONCLUSIVE (not equivalence). See ``outputs/CAMPAIGN_power.md``.

Flags
-----
  --target-power   Desired power (default 0.80; the doc table also reports 0.90).
  --alpha          Family significance level before the selection penalty (0.05).
  --sigma-seed     Seed-to-seed σ of the per-seed score within an arm (CLEAN PILOT INPUT). Omit to use
                   the flagged directional upper-bound proxy from the prototype (``DIRECTIONAL_SIGMA_SEED``).
  --rho            Pairing correlation across the shared seed (default 0.0; NOT the worst case —
                   the measured pilot value is -0.141, which inflates sigma_D; see the module note).
  --sesoi          Smallest effect size of interest (FROZEN: read from config; PREREGISTRATION §10 R12).
  --equiv-margin   Symmetric TOST equivalence margin (FROZEN: read from config).
  --n-seeds        Override the per-arm winner seed count n_seeds (default: from config; headline 30).
  --n-regimes      Override the independent-regime count N (REPORTED only; bounds the secondary
                   regime-stratified analyses — it does NOT enter the headline paired n).
  --n-comparisons  Size of the selection-aware testing family (default: frozen m=6).
  --n-sims         Monte-Carlo simulations per effect-size grid point (default 2000).
  --n-boot         Bootstrap replications inside each simulated paired test (default 2000, the code
                   default of ``paired_seed_difference_test``).
  --effect-max     Upper end of the effect grid in σ_seed units (default 5.0 — at N≤5 the conservative
                   paired test needs δ≈3–4σ, Colas 2019, so the grid must extend there to locate an MDE).
  --effect-points  Number of grid points on the effect-size search (default 41).
  --val-track-length  Validation track length T for the Sharpe->validation-DSR MDE reconciliation (T2.5;
                   default 694 = the SCORED Split-C validation window 2017-03-30..2019-12-31 after the
                   60-session purge — the old "756 / [2015,2017]" figures were the superseded
                   pre-Split-C window). Maps the Sharpe-unit MDE into the SESOI's DSR units
                   (conservative ceiling) so the null framing is single-unit.
  --seed           RNG seed for reproducibility (default 0).
  --out            Output markdown path (default outputs/CAMPAIGN_power.md).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Any, TypedDict

import numpy as np
from scipy.stats import norm

# Fixed import (the stub did `from src.regimes import detect`, which does not exist).
from src.regimes.definition import independent_regime_count, label_regimes

__all__ = [
    "PLACEHOLDER_SIGMA_DSR",
    "PLACEHOLDER_SESOI",
    "PLACEHOLDER_EQUIV_MARGIN",
    "DIRECTIONAL_SIGMA_SEED",
    "VALIDATION_TRACK_LENGTH",
    "TEST_TRACK_LENGTH",
    "PERIODS_PER_YEAR",
    "PowerConfig",
    "regime_count_on_panel",
    "selection_aware_alpha",
    "simulate_power",
    "minimum_detectable_effect",
    "sharpe_mde_to_dsr",
    "tost_equivalence",
    "build_parser",
    "main",
]

# --------------------------------------------------------------------------- #
# Noise input σ_seed — a DIRECTIONAL upper bound from the prototype, replaced by
# the clean pilot value (winners re-run at 30 seeds, reward fixed) when it exists.
# --------------------------------------------------------------------------- #

#: DIRECTIONAL seed-to-seed σ of the per-seed score (annualised Sharpe) within an arm, anchored on the
#: prototype archive's per-candidate within-arm Sharpe dispersion ≈ 0.36 (pooled; headline arms 0.36/0.37
#: — see ``outputs/CAMPAIGN_power.md`` and the in-repo derivation). It is an UPPER BOUND on the campaign's
#: σ_seed (it folds reward-design variance and a 1-seed window into the dispersion), so the MDE computed
#: against it is pessimistic on the σ axis. The clean number is whatever the seeds-on-winners pilot reports.
DIRECTIONAL_SIGMA_SEED: float = 0.36

#: BACK-COMPAT alias. The old name modelled a σ in *validation-DSR* (probability-like) units for the
#: SUPERSEDED seed-averaged-series test; the realized headline test runs on the per-seed annualised Sharpe,
#: so the headline σ is :data:`DIRECTIONAL_SIGMA_SEED` (Sharpe units). Retained as a public symbol (and the
#: ``PowerConfig.sigma_dsr`` default) so importers/tests do not break; it now equals the directional proxy.
PLACEHOLDER_SIGMA_DSR: float = DIRECTIONAL_SIGMA_SEED

# --------------------------------------------------------------------------- #
# Sharpe -> validation-DSR mapping (T2.5): reconcile the Sharpe-unit MDE with the
# DSR-unit SESOI so a non-rejection's "bounded effect" claim is stated in the SAME
# units as the frozen SESOI (0.05 validation-DSR), not silently across two scales.
# --------------------------------------------------------------------------- #

#: Annualisation factor for the headline Sharpe (``src.backtest.metrics.PERIODS_PER_YEAR`` = 252 daily).
#: The MDE is in ANNUALISED Sharpe; the DSR's PSR uses the PER-PERIOD Sharpe, so the mapping divides the
#: annualised Sharpe difference by ``sqrt(PERIODS_PER_YEAR)`` to recover the per-period shift.
PERIODS_PER_YEAR: int = 252

#: EXECUTED validation track length T (the ``n`` in the DSR's ``sqrt(n-1)`` factor). Split C (R73): the
#: executed, purged validation window is ``expected_windows.univ5.val = [3081, 3775)`` in
#: config/inference.yaml — 2017-03-30..2019-12-31 — i.e. **694** sessions (3775−3081), NOT the nominal
#: 3×252=756 the pre-Split-C ``[2015, 2017]`` docstring assumed (corrected 2026-07-02: the stale 756
#: inflated k=0.3989·sqrt(T−1)/sqrt(252) to 0.6905 vs the executed 0.6616 and shifted every DSR↔Sharpe
#: equivalent). A DESIGN constant of the Sharpe->DSR mapping (NOT a frozen prereg field); overridable via
#: ``--val-track-length`` so the reconciliation can be re-stated against the realized session count.
VALIDATION_TRACK_LENGTH: int = 694

#: EXECUTED TEST track length T, the sibling of the constant above and derived the same way: Split C's
#: purged test window is ``expected_windows.univ5.test = [3835, 5406)`` in config/inference.yaml —
#: **1571** sessions (5406−3835). Named here (2026-08-01, RUN 11) because the A16 non-inferiority
#: margin must be reported at BOTH track lengths and the alternative was a literal 1571 at the call
#: site, which is exactly the drift this constant's neighbour exists to prevent.
#:
#: NOTE: IT IS NOT INTERCHANGEABLE WITH THE ONE ABOVE, and confusing them moves a confirmatory margin by
#: 50 %. The SESOI is registered "in validation-DSR units", and ``src/selection/fitness.py`` refuses
#: to compute the DSR off the validation split — so ``sharpe_mde_to_dsr``'s DEFAULT (694) is the
#: registered conversion, and 1571 answers a different question. k(694)=0.661571 → 0.05 DSR =
#: 0.0756 ann-Sharpe; k(1571)=0.995771 → 0.0502. This constant exists to make the CONSERVATIVE
#: sensitivity computable beside the registered value, never to replace it.
#:
#: VERIFIED 2026-08-01 by re-deriving the executed windows through the campaign's own code path
#: (``run_campaign_cluster.assemble_cluster_inputs`` → ``resolve_windows``): val (3081, 3775) = 694,
#: test (3835, 5406) = 1571. Config, code and the executed windows all agree.
TEST_TRACK_LENGTH: int = 1571


def sharpe_mde_to_dsr(
    sharpe_mde: float,
    *,
    track_length: int = VALIDATION_TRACK_LENGTH,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Map an ANNUALISED-Sharpe MDE to a CONSERVATIVE (upper-bound) shift in validation-DSR units (T2.5).

    Why this exists. The power analysis reports the MDE in annualised-Sharpe units (the headline difference
    test's scale), but the FROZEN SESOI / TOST margin is **0.05 validation-DSR** units (the selection
    metric; PREREGISTRATION §10 R12). A non-rejection is only a *bankable bounded-effect* finding if the
    effect is small in the SESOI's units — so the two must be reconciled rather than compared across scales.

    The mapping (documented, defensible). The validation DSR is a Probabilistic Sharpe Ratio,
    ``DSR = Phi(z)`` with ``z = SR_pp * sqrt(T - 1) / sqrt(D)`` (Bailey & Lopez de Prado;
    ``src.inference.deflated_sharpe``), where ``SR_pp`` is the **per-period** Sharpe and
    ``D = 1 - skew*SR_pp + (kurt - 1)/4 * SR_pp^2`` the non-normality denominator. A small ANNUALISED Sharpe
    gap ``ΔSR_ann`` is ``ΔSR_pp = ΔSR_ann / sqrt(periods_per_year)`` per period, and by the delta method
    induces a DSR shift ``ΔDSR ≈ phi(z) * sqrt(T - 1) / sqrt(D) * ΔSR_pp``. The sensitivity ``phi(z)/sqrt(D)``
    is **maximised** at the benchmark (``z = 0`` ⇒ ``phi(0) = 0.3989``, DSR ≈ 0.5) and under normal returns
    (``skew = 0, kurt = 3`` ⇒ ``D = 1``), so the CONSERVATIVE CEILING — the largest DSR shift ``ΔSR_ann``
    can produce — is::

        ΔDSR_max = 0.3989 * sqrt(T - 1) / sqrt(periods_per_year) * ΔSR_ann

    This is an UPPER bound: it is exact at-the-money/normal and over-states ΔDSR elsewhere (where ``z != 0``
    or the returns are fat-tailed, ``phi(z) < 0.3989`` and ``D > 1`` both shrink it). Using the ceiling is the
    honest direction for the null framing — if even the *largest* DSR shift the Sharpe-MDE could cause is
    below the 0.05 SESOI, the Sharpe-MDE is unambiguously sub-SESOI in DSR units; if it EXCEEDS 0.05 (it does
    at T≈756: ΔDSR_max ≈ 0.18 ≫ 0.05), the two metrics probe DIFFERENT scales and a Sharpe non-rejection
    does NOT by itself license a DSR-equivalence claim — only the TOST CI in DSR units can (the INCONCLUSIVE
    branch).

    Parameters
    ----------
    sharpe_mde:
        The minimum detectable effect in ANNUALISED-Sharpe units (e.g. ``minimum_detectable_effect(...)['mde']``).
    track_length:
        Validation track length ``T`` (the ``sqrt(T - 1)`` factor); default :data:`VALIDATION_TRACK_LENGTH`.
    periods_per_year:
        Annualisation factor; default :data:`PERIODS_PER_YEAR` (252).

    Returns
    -------
    float
        ``ΔDSR_max``, the conservative (upper-bound) validation-DSR shift the Sharpe MDE could induce.
        ``0.0`` for a non-positive / non-finite ``sharpe_mde`` (an unreached MDE maps to no DSR shift).
    """
    if not np.isfinite(sharpe_mde) or sharpe_mde <= 0.0:
        return 0.0
    t = int(track_length)
    ppy = int(periods_per_year)
    if t < 2 or ppy < 1:
        raise ValueError("track_length must be >= 2 and periods_per_year >= 1")
    phi0 = float(norm.pdf(0.0))  # 0.39894...
    return float(phi0 * math.sqrt(t - 1) / math.sqrt(ppy) * float(sharpe_mde))


def _frozen_inference(key: str, fallback: float) -> float:
    """READ a frozen inference parameter from config/preregistration.yaml — the single source of truth
    (code reads config and never hardcodes). Falls back to the literal only if config is unreadable."""
    try:
        from src.utils.config import load_config

        return float(load_config("preregistration").get("inference", {}).get(key, fallback))
    except Exception:  # noqa: BLE001 - config unreadable (e.g. isolated import) -> documented fallback
        return float(fallback)


#: Smallest effect size of interest — READ from the FROZEN pre-registration (config/preregistration.yaml:
#: inference.sesoi; PREREGISTRATION §10 R12), NOT hardcoded. Specified in *validation-DSR* units (see the
#: units caveat in the module docstring / :func:`tost_equivalence`).
PLACEHOLDER_SESOI: float = _frozen_inference("sesoi", 0.05)

#: Symmetric TOST equivalence margin = the SESOI — READ from the frozen config (inference.equivalence_margin).
PLACEHOLDER_EQUIV_MARGIN: float = _frozen_inference("equivalence_margin", 0.05)


def _frozen_family_m(fallback: int = 6) -> int:
    """READ the frozen multiple-testing family size m (config/preregistration.yaml:
    inference.testing_family.m) — the m driving the Šidák/Bonferroni selection-power adjustment — NOT a
    hardcoded 6. Falls back to the literal only if config is unreadable."""
    try:
        from src.utils.config import load_config

        fam = load_config("preregistration").get("inference", {}).get("testing_family", {})
        return int(fam.get("m", fallback)) if isinstance(fam, dict) else int(fallback)
    except Exception:  # noqa: BLE001 - config unreadable (e.g. isolated import) -> documented fallback
        return int(fallback)


#: Pre-registered testing-family size m for the multiplicity adjustment — READ from frozen config (the
#: SAME m that analyze_campaign/run_campaign enumerate + assert), not a hardcoded 6.
FROZEN_FAMILY_M: int = _frozen_family_m(6)


def _frozen_n_seeds(fallback: int = 30) -> int:
    """READ the per-arm WINNER seed count n_seeds — the realized paired sample size of the headline test —
    from the frozen config (config/preregistration.yaml: ``seeds`` list; Amendment D2 raised it 5→30),
    NOT a hardcoded literal. Falls back to the literal only if config is unreadable."""
    try:
        from src.utils.config import load_config
        from src.utils.seeds import resolve_seeds as _resolve_seeds

        # 'seeds' is a bare list OR a schema ({mode: tiered/uniform, …}); resolve to the flat set so
        # the realized paired n is correct for both forms (a raw len() of the dict would read 2).
        seeds = load_config("preregistration").get("seeds", None)
        return int(len(_resolve_seeds(seeds))) if seeds else int(fallback)
    except Exception:  # noqa: BLE001 - config unreadable -> documented fallback
        return int(fallback)


#: Headline per-arm winner seed count = the realized PAIRED sample size of the H2 test (Amendment D2: 30).
DESIGN_N_SEEDS: int = _frozen_n_seeds(30)

#: Design-time independent-regime count for 2005–2025 (GFC, the 2010s low-vol bull, 2018 vol spikes,
#: COVID, the 2022 drawdown — ~6 independent regimes). REPORTED to bound the SECONDARY regime-stratified
#: analyses; it does NOT enter the headline paired n (the per-seed IQM test does not stratify by regime).
#: NB the vix-units issue (raw FRED VIXCLS is stored as a FRACTION ≈0.10–0.81, not points) is FIXED at load:
#: ``src/data/loaders.load_gold_panel`` head-detects the unit and rescales vix ×100 → conventional points
#: (loaders.py ~L455-458; 2026-06-19), so ``config/regimes.yaml``'s point thresholds (calm=15/stress=25) apply
#: directly and the independent-regime count no longer collapses to 1. DESIGN_REGIME_COUNT is a hardcoded
#: design-time constant regardless (NOT auto-detected here), so it is report-only and this note is
#: informational — it does NOT enter the headline paired n.
DESIGN_REGIME_COUNT: int = 6


def _max_plausible_regimes(fallback: int = 12) -> int:
    """READ the upper sanity bound on the INDEPENDENT-regime count from config (config/regimes.yaml:
    ``max_plausible_independent_regimes``) — NOT a hardcoded literal. Falls back if config/key is absent."""
    try:
        from src.utils.config import load_config

        val = load_config("regimes").get("max_plausible_independent_regimes", fallback)
        return int(val) if val is not None else int(fallback)
    except Exception:  # noqa: BLE001 - config unreadable -> documented fallback
        return int(fallback)


#: Upper sanity bound on a TRUSTWORTHY auto-detected regime count (above it, ``independent_regime_count``
#: has over-segmented a noisy points-scale vix into many short run-length blocks, not the ~6 macro regimes
#: the design assumes — so the design value is used). Symmetric to the collapse-to-1 case. Sourced from config.
MAX_PLAUSIBLE_REGIMES: int = _max_plausible_regimes(12)


@dataclass(frozen=True)
class PowerConfig:
    """The planned design + statistical knobs for the headline-H2 power analysis.

    The headline test is the rliable per-seed PAIRED bootstrap (R16): its realized sample size is
    ``n_seeds`` (the per-arm winner seed count; Amendment D2 = 30), NOT ``seeds x folds x N_regimes``.
    Effect/noise quantities are in **per-seed-score units** — for the headline H2-conjunction leg that is
    the annualised Sharpe, so ``effect`` is a difference in IQM-over-per-seed-Sharpe between arms and
    ``sigma_dsr`` (kept as the field name for back-compat) is the seed-to-seed σ of that per-seed score.

    NB on the legacy fields. ``seeds``/``folds``/``n_regimes`` and the :attr:`n_eff` product are retained
    for API/back-compat and to drive the SECONDARY regime-stratified bound; the realized paired sample
    size used by :func:`simulate_power` is :attr:`n_paired`. For the HEADLINE design,
    :func:`main` sets ``n_regimes=1, folds=1`` so ``n_paired == seeds == n_seeds`` (regimes do not inflate
    the paired n). ``rho`` is the pairing correlation across the shared seed (default 0.0 — NOT
    conservative: the measured pilot rho is -0.141, and rho<0 INFLATES sigma_D; see the module note).
    """

    n_regimes: int
    seeds: int = DESIGN_N_SEEDS
    folds: int = 1
    sigma_dsr: float = DIRECTIONAL_SIGMA_SEED  # seed-to-seed σ of the per-seed score (Sharpe units)
    rho: float = 0.0  # pairing correlation across the shared seed (0 = no cancellation; NOT the worst
                      # case — measured pilot rho = -0.141 inflates sigma_D; module note, loop 5)
    target_power: float = 0.80
    alpha: float = 0.05
    n_comparisons: int = FROZEN_FAMILY_M  # frozen testing-family size (config), not a hardcoded 6
    sesoi: float = PLACEHOLDER_SESOI
    equiv_margin: float = PLACEHOLDER_EQUIV_MARGIN
    n_sims: int = 2000
    n_boot: int = 2000  # paired_seed_difference_test code default
    seed: int = 0
    sigma_is_placeholder: bool = True  # True while σ is the directional proxy (no clean pilot yet)
    #: R37 (M4): the LIVE per-leg decision rule. ``True`` = the headline ONE-SIDED IUT leg test at ``alpha``
    #: (the conjunction/IUT IS the correction within H2-RA/H2-Tail; multiplicity across the m=6 union is the
    #: LIVE BH/Romano-Wolf, NOT a fixed Šidák-α_eff). ``False`` (the back-compat default) = the conservative
    #: TWO-SIDED test at the Šidák-over-m α_eff — retained as the REPORTED BH-over-6 sensitivity, never deleted.
    iut_one_sided: bool = False

    @property
    def n_eff(self) -> int:
        """LEGACY product ``seeds x folds x independent regimes``.

        Retained for back-compat and as the SECONDARY regime-stratified cell count. It is **NOT** the
        realized headline paired sample size — the per-seed IQM PAIRED test (R16) consumes ``n_seeds``
        paired scores (:attr:`n_paired`), and the regime count does not multiply into it. See the module
        docstring; use :attr:`n_paired` for the headline.
        """
        return int(self.seeds) * int(self.folds) * int(self.n_regimes)

    @property
    def n_paired(self) -> int:
        """Realized PAIRED sample size of the headline test = the number of per-seed score pairs.

        This is what :func:`simulate_power` resamples. It equals :attr:`n_eff` so that the design knobs
        (seeds / folds / regimes) all move it monotonically; for the HEADLINE config (``n_regimes=1,
        folds=1``) it collapses to ``seeds == n_seeds`` (30) — the honest per-seed paired count, with the
        regime count excluded from the headline n (it bounds only the secondary stratified analyses).
        """
        return self.n_eff


def regime_count_on_panel(panel: Any, cfg: Any) -> int:
    """Independent-regime count ``N`` for ``panel`` under regime config ``cfg``.

    Thin wrapper over :func:`src.regimes.definition.label_regimes` + :func:`independent_regime_count`;
    isolated so the campaign-time call site and the tests share one code path.
    """
    return independent_regime_count(label_regimes(panel, cfg))


def selection_aware_alpha(alpha: float, n_comparisons: int) -> float:
    """Per-test alpha after a selection-aware (Šidák) penalty.

    The H2 contrast is one of a family of arm x metric comparisons (the frozen m=6, R13); testing it at
    the nominal ``alpha`` would inflate the family-wise error. We apply the Šidák correction
    ``1 - (1 - alpha)**(1 / m)`` (slightly less conservative than, numerically close to, Bonferroni
    ``alpha / m``) to obtain the per-test alpha the MDE is computed against. This is the *power-analysis*
    counterpart of the BH-FDR / Romano-Wolf control applied at analysis time
    (``src.inference.multiple_testing``); here we need one fixed per-test alpha to size the design, so we
    take the conservative FWER route. With ``m == 1`` this is the identity (no penalty).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    m = int(n_comparisons)
    if m < 1:
        raise ValueError("n_comparisons must be >= 1")
    if m == 1:
        return float(alpha)
    return float(1.0 - (1.0 - alpha) ** (1.0 / m))


def _draw_paired_scores(
    effect: float,
    sigma: float,
    n: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` per-seed score PAIRS ``(a_i, b_i)`` for the two arms under a true ``effect``.

    Gaussian per-seed scores with per-arm SD ``sigma`` and a shared-seed (pairing) correlation ``rho``.
    For ``rho >= 0``: a common per-seed component ``sqrt(rho)*z_common`` (a "lucky/unlucky" seed lifting
    BOTH arms — the variance the paired test cancels) plus an arm-private ``sqrt(1-rho)*z_arm``. For
    ``rho < 0`` (batch-5 M3, 2026-07-03): the shared-component trick cannot represent anti-correlation
    (``sqrt`` of a negative), and the old code silently CLAMPED to 0 — understating σ_D = σ√(2(1−ρ)) > σ√2
    exactly in the branch where the pilot's own report tells the operator to trust this Monte-Carlo. Now a
    Cholesky pair on ``(z_common, za)`` draws the measured negative ρ exactly; all three normals are drawn
    on BOTH branches so the ρ >= 0 stream stays byte-identical to the pre-fix code. The distributional
    arm's mean is shifted up by ``effect`` (a one-sided H2 alternative; ``effect == 0`` recovers the null).
    Element ``i`` is the SAME training seed for both arms, matching the paired test's by-seed alignment.
    """
    rho = float(min(max(rho, -0.999), 0.999))  # keep the 2x2 covariance PSD; |rho|=1 degenerates the pair
    z_common = rng.standard_normal(n)
    za = rng.standard_normal(n)
    zb = rng.standard_normal(n)
    if rho >= 0.0:
        root_shared = np.sqrt(rho)
        root_priv = np.sqrt(1.0 - rho)
        a_noise = root_shared * z_common + root_priv * za  # distributional arm
        b_noise = root_shared * z_common + root_priv * zb  # scalar arm
    else:
        # corr(a_noise, b_noise) = rho exactly for rho in (-1, 0); both marginals stay unit-SD:
        # Var(b_noise) = rho² + (1 − rho²) = 1. zb is drawn (above) but unused on this branch, keeping
        # the RNG stream aligned across branches.
        a_noise = z_common
        b_noise = rho * z_common + np.sqrt(1.0 - rho * rho) * za
    a = effect + sigma * a_noise
    b = 0.0 + sigma * b_noise
    return a, b


def simulate_power(effect: float, cfg: PowerConfig) -> float:
    """Monte-Carlo power of the REALIZED headline H2 paired test at a true ``effect``.

    For each of ``cfg.n_sims`` simulations we draw ``cfg.n_paired`` correlated per-seed score PAIRS
    (per-arm SD ``cfg.sigma_dsr``, pairing correlation ``cfg.rho``, distributional mean shifted by
    ``effect``) and run the ACTUAL frozen test
    :func:`src.inference.bootstrap.paired_seed_difference_test` with the rliable IQM statistic — the same
    function ``scripts/analyze_campaign.collect_family_pvalues`` calls — rejecting when its two-sided
    p-value is below the selection-aware per-test alpha (Šidák over the frozen m). Power is the rejection
    fraction. Because the real test is invoked (not a re-implementation), the simulated size and the
    small-``n`` conservativeness of the paired bootstrap are reproduced faithfully; ``effect == 0`` returns
    ~the selection-aware alpha (in practice ≤ it — the paired test is conservative at small n).

    Parameters
    ----------
    effect:
        True per-seed-score gap (distributional minus scalar IQM), in the score's units (annualised
        Sharpe for the headline leg).
    cfg:
        The planned design + knobs (``n_paired`` is the realized paired sample size).

    Returns
    -------
    float
        Estimated power in ``[0, 1]``.
    """
    if cfg.sigma_dsr <= 0.0:
        raise ValueError("sigma_dsr (seed-to-seed sigma) must be positive")
    # Lazy import keeps the module importable in isolation; this is the SAME test the campaign runs.
    from src.inference.bootstrap import iqm, paired_seed_difference_test

    rng = np.random.default_rng(cfg.seed)
    # Two decision rules, both on the REAL paired bootstrap (no re-implemented stat):
    #  - iut_one_sided=True  (R37/M4, the LIVE rule): a per-leg ONE-SIDED test at `alpha` in the predicted
    #    direction. The two-sided bootstrap p is converted in-direction (p_one = p_two/2 when the observed
    #    effect > 0, else the one-sided leg does NOT reject) — a SYMMETRIC-DGP approximation to
    #    `collect_family_pvalues._one_sided`, which post-R64 uses the direct upper-tail `pvalue_one_sided_greater`;
    #    the two coincide here because the power DGP draws symmetric (normal) pairs.
    #    The IUT/conjunction IS the within-H2 correction; the across-union multiplicity is the LIVE BH/RW, so
    #    NO fixed Šidák-α_eff is applied here. This is the PRIMARY MDE.
    #  - iut_one_sided=False (back-compat): the TWO-SIDED test at the Šidák-over-m α_eff — the conservative
    #    BH-over-6 SENSITIVITY, retained (not deleted) and reported alongside.
    a_alpha = float(cfg.alpha) if cfg.iut_one_sided else selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    n = int(cfg.n_paired)
    if n < 2:
        # The paired test needs >= 2 seeds; with fewer, power is undefined -> report 0.0 so the MDE
        # search treats it as unreachable (it raises ValueError below 2, so we never call it).
        return 0.0
    rejections = 0
    for _ in range(int(cfg.n_sims)):
        a, b = _draw_paired_scores(float(effect), cfg.sigma_dsr, n, cfg.rho, rng)
        res = paired_seed_difference_test(a, b, statistic=iqm, n_boot=cfg.n_boot, rng=rng)
        if cfg.iut_one_sided:
            # In-direction one-sided conversion (DEEP_H2 stats note A5): p_one = p_two/2 when the observed
            # effect is in the predicted direction (a better than b), else the one-sided test cannot reject.
            direction_ok = float(res["effect"]) > 0.0
            reject = direction_ok and (float(res["pvalue"]) / 2.0 <= a_alpha)
        else:
            reject = float(res["pvalue"]) < a_alpha
        if reject:
            rejections += 1
    return rejections / float(cfg.n_sims)


def minimum_detectable_effect(
    cfg: PowerConfig,
    effect_max: float = 5.0,
    effect_points: int = 41,
) -> dict[str, object]:
    """Locate the MDE: the smallest effect reaching ``cfg.target_power``.

    Sweeps a grid of true effects in ``[0, effect_max * sigma_seed]`` (so the grid auto-scales with the
    noise and is expressed in σ_seed multiples — the standardized δ = effect/σ), estimates power at each
    point via :func:`simulate_power`, and returns the first grid effect whose power ≥ ``target_power``
    (linearly interpolated against the previous point for a sub-grid estimate).

    ``effect_max`` defaults to **5.0 σ** because the PAIRED bootstrap is conservative at small ``n``: at
    ``n ≤ 5`` an 80%-power MDE sits near δ≈3–4σ (Colas et al. 2019), so a narrower grid would fail to
    bracket it. At the headline ``n_seeds = 30`` the MDE is ≈1σ and is found well inside the grid.

    Returns
    -------
    dict
        ``{"mde", "mde_sigma_units", "grid", "power", "reached", "alpha_eff"}``. ``mde`` is ``None``
        (``reached`` False) if ``target_power`` is not hit anywhere on the grid.
    """
    grid = np.linspace(0.0, effect_max * cfg.sigma_dsr, int(effect_points))
    # Sequential sweep with EARLY EXIT at the target-power crossing (batch-5 M2, 2026-07-03): the
    # crossing + interpolation only need points up to the first i >= 1 with power >= target, but the
    # old full-grid comprehension burned ~80% of every sweep past it (power ~= 1 out to 5σ; 11 sweeps
    # per doc build). simulate_power re-seeds default_rng(cfg.seed) per call, so every KEPT point is
    # byte-identical to the full sweep; grid/power are truncated TOGETHER (shape parity holds; the doc
    # table samples whatever length it gets, and a NOT-reached curve still computes the full grid so
    # its "up to grid.max()" message is unchanged).
    powers_list: list[float] = []
    for j, e in enumerate(grid):
        powers_list.append(simulate_power(float(e), cfg))
        if j >= 1 and powers_list[-1] >= cfg.target_power:
            break
    powers = np.array(powers_list, dtype=float)
    grid = grid[: powers.size]

    mde: float | None = None
    reached = False
    for i in range(1, grid.size):
        if powers[i] >= cfg.target_power:
            reached = True
            e0, e1 = grid[i - 1], grid[i]
            p0, p1 = powers[i - 1], powers[i]
            if p1 > p0:
                frac = (cfg.target_power - p0) / (p1 - p0)
                mde = float(e0 + frac * (e1 - e0))
            else:
                mde = float(e1)
            break

    # The per-test alpha the curve was computed against: the live ONE-SIDED IUT uses `alpha` straight (the
    # IUT is the correction; multiplicity is the live BH/RW), whereas the back-compat mode uses the Šidák-α_eff.
    alpha_eff = float(cfg.alpha) if cfg.iut_one_sided else selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    return {
        "mde": mde,
        "mde_sigma_units": None if mde is None else float(mde / cfg.sigma_dsr),
        "grid": grid,
        "power": powers,
        "reached": reached,
        "alpha_eff": alpha_eff,
    }


@dataclass(frozen=True)
class TostResult:
    """Outcome of a two-one-sided-tests (TOST) equivalence assessment."""

    equivalent: bool
    estimate: float
    ci_low: float
    ci_high: float
    margin: float


def tost_equivalence(
    a: np.ndarray,
    b: np.ndarray,
    margin: float,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    *,
    paired: bool = False,
) -> TostResult:
    """Symmetric-margin **MEAN** TOST equivalence test for two arms' per-seed scores (Lakens 2017 primer).

    NOTE: STATISTIC / PAIRING (P2 reconciliation, 2026-07-01). This is the MEAN-difference TOST. The HEADLINE
    campaign equivalence is a DIFFERENT statistic: it is PAIRED (common-random-number seed pairing) on the
    **IQM** difference — ``scripts/analyze_campaign._iqm_tost`` — matching the headline difference test
    (``paired_seed_difference_test`` with ``statistic=iqm``). The two must not be silently swapped, so:

      * ``paired=False`` (DEFAULT) resamples ``a`` and ``b`` INDEPENDENTLY. Because it discards the
        across-arm seed pairing, its difference CI is (for positively CRN-correlated arms) WIDER than the
        paired CI, so it is a **CONSERVATIVE, POWER-ANALYSIS-ONLY helper** (the sensitivity/planning TOST in
        this module's power tables) — it should NOT decide the headline equivalence.
      * ``paired=True`` resamples a SHARED seed index for both arms (CRN pairing), so the mean-difference CI
        carries the across-seed covariance exactly like the headline. Use this whenever ``a``/``b`` are
        genuinely paired per-seed scores (e.g. the order-preserving Sharpe→DSR rescale in
        ``analyze_campaign.h2_tost_dsr``). Requires ``a.shape == b.shape``.

    The headline equivalence VERDICT always routes through the paired+IQM ``_iqm_tost``; this MEAN TOST is
    the conservative complement (mean vs IQM) and the power-analysis planning tool.

    Declares the two arms *practically equivalent* when a two-sided ``1 - 2*alpha`` (90%) percentile
    bootstrap CI for the mean difference falls **entirely within** ``(-margin, +margin)`` — the TOST rule
    (the 90% CI inside ±SESOI ⇔ both one-sided tests reject; Lakens 2017). A non-rejection of H2 *plus* a
    TOST equivalence verdict lets the dissertation report "distributional and scalar are equivalent within
    ±``margin``" — a bounded effect, not "we found nothing" (PREREGISTRATION §10 R12; viva Q21).

    UNITS. ``margin`` is applied in the SAME units as ``a``/``b``. The FROZEN SESOI/TOST margin ±0.05 is
    specified in *validation-DSR* units (the selection metric), so for a DSR-space equivalence claim feed
    per-seed DSR; the headline difference TEST runs on per-seed Sharpe/CVaR, which are a different scale —
    do not conflate them (see the module docstring). Lakens 2017: equivalence bounds should be set on the
    RAW scale, not a standardized (Cohen-d) scale, to avoid bias.

    Parameters
    ----------
    a, b:
        Per-seed scores for the two arms (one value per training seed).
    margin:
        Symmetric equivalence margin (the SESOI), in the units of ``a``/``b``. Must be positive.
    n_boot:
        Bootstrap replications for the CI.
    rng:
        Optional NumPy generator.
    paired:
        If True, resample a SHARED seed index for both arms (CRN pairing; requires equal shapes). If False
        (default), resample independently — the conservative, power-analysis-only mode.

    Returns
    -------
    TostResult
        ``equivalent`` is True iff the 90% CI lies inside ``(-margin, +margin)``.
    """
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rng is None:
        # 2026-07-06 (repro audit): the omitted-rng default is now DETERMINISTIC (seed 0), not OS
        # entropy — the exact footgun that once made the parallel BO winner non-reproducible
        # (final-audit #2). Every production caller passes a run-seeded generator explicitly.
        rng = np.random.default_rng(0)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        raise ValueError("a and b must be non-empty")

    estimate = float(a.mean() - b.mean())
    nb_boot = int(n_boot)
    if paired:
        if a.shape != b.shape:
            raise ValueError("paired=True requires a and b to be the same shape (paired per-seed scores)")
        idx = rng.integers(0, na, size=(nb_boot, na))  # SAME seed-index draw for both arms (CRN pairing)
        boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        ia = rng.integers(0, na, size=(nb_boot, na))
        ib = rng.integers(0, nb, size=(nb_boot, nb))
        boot = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    # 90% CI (alpha = 0.05 each side) is the standard TOST equivalence CI (1 - 2*alpha).
    ci_low = float(np.quantile(boot, 0.05))
    ci_high = float(np.quantile(boot, 0.95))
    equivalent = bool(ci_low > -margin and ci_high < margin)
    return TostResult(
        equivalent=equivalent,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        margin=float(margin),
    )


# --------------------------------------------------------------------------- #
# Doc rendering + CLI
# --------------------------------------------------------------------------- #


def _fmt(x: Any, nd: int = 3) -> str:
    """Format a numeric (or ``None``) for the doc. Accepts ``Any`` because the MDE record values are
    typed ``object`` (a heterogeneous dict); a non-finite/None renders as an em dash."""
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def _mde_at(cfg: PowerConfig, target_power: float, effect_max: float, effect_points: int) -> dict[str, object]:
    """MDE record at a specific target power (re-uses :func:`minimum_detectable_effect` with an override)."""
    return minimum_detectable_effect(replace(cfg, target_power=target_power), effect_max=effect_max, effect_points=effect_points)


def _rho_sweep_table(
    cfg: PowerConfig,
    rhos: tuple[float, ...],
    effect_max: float,
    effect_points: int,
) -> str:
    """MDE@80% / MDE@90% for each pairing correlation ρ — the headline sensitivity table."""
    lines = [
        "| ρ (pairing corr.) | Δ_MDE @80% (Sharpe) | (σ_seed units) | Δ_MDE @90% (Sharpe) | (σ_seed units) |",
        "|---|---|---|---|---|",
    ]
    for rho in rhos:
        print(f"    rho = {rho:.3f} ...")
        c = replace(cfg, rho=float(rho))
        m80 = minimum_detectable_effect(replace(c, target_power=0.80), effect_max=effect_max, effect_points=effect_points)
        m90 = minimum_detectable_effect(replace(c, target_power=0.90), effect_max=effect_max, effect_points=effect_points)
        flag = " *(default, conservative)*" if rho == 0.0 else ""
        lines.append(
            f"| {rho:.3f}{flag} | {_fmt(m80['mde'])} | {_fmt(m80['mde_sigma_units'], 2)} | "
            f"{_fmt(m90['mde'])} | {_fmt(m90['mde_sigma_units'], 2)} |"
        )
    return "\n".join(lines)


def render_markdown(
    cfg: PowerConfig,
    mde: dict[str, object],
    n_trials: int,
    *,
    n_detected: int | None = None,
    detect_unreliable: bool = False,
    n_source: str = "design_default",
    unreliable_reason: str = "collapsed",
    rho_table: str = "",
    mde90: dict[str, object] | None = None,
    mde_sidak: dict[str, object] | None = None,
    sidak_alpha: float | None = None,
    val_track_length: int = VALIDATION_TRACK_LENGTH,
) -> str:
    """Render ``outputs/CAMPAIGN_power.md``: the method, the regenerated MDE table, the σ/ρ assumptions.

    Flags the pilot-dependent fields (σ_seed and therefore the MDE) as a DIRECTIONAL upper bound until the
    clean seeds-on-winners pilot replaces σ, and the regime-units caveat when the auto-detected N is
    unreliable. ``unreliable_reason`` distinguishes ``"collapsed"`` (gold vix-units mismatch -> 1 regime)
    vs ``"too_many"`` (implausibly large run-length count, e.g. the offline/synthetic panel).
    """
    a_eff = float(mde["alpha_eff"])  # type: ignore[arg-type]
    mde_val = mde["mde"]  # type: ignore[assignment]
    mde_sig = mde["mde_sigma_units"]  # type: ignore[assignment]
    sigma_flag = (
        " **(DIRECTIONAL upper-bound proxy — prototype dispersion; replace with the seeds-on-winners pilot σ)**"
        if cfg.sigma_is_placeholder
        else ""
    )
    mde_flag = " *(at the directional σ_seed; re-run with the clean pilot σ)*" if cfg.sigma_is_placeholder else ""
    # NOTE: 2026-07-27 (deep review, finding #95): the provenance strings below were HARDCODED to the
    # placeholder case, so the generated doc kept asserting "computed at the directional σ_seed — pilot
    # field flagged" and "re-run … to finalise the TBD fields" even on a run invoked with the MEASURED
    # pilot σ — i.e. it went stale the moment a reader did what it instructs. The script already tracks
    # `sigma_is_placeholder` (it drives the checkbox and `sigma_flag`), so the heading contradicted a
    # flag the same function was computing. This matters because CAMPAIGN_power.md is the document that
    # justifies the seed count: a reader who trusted the heading would read the design's real, measured
    # power as provisional. Same defect class as #87 — a rendered claim that does not track its input.
    result_heading = (
        "## Result (computed at the directional σ_seed — pilot field flagged)"
        if cfg.sigma_is_placeholder
        else "## Result (computed at the MEASURED clean-pilot σ_seed and ρ)"
    )
    purpose_tail = (
        "Committed\nbefore the Phase-1 freeze; re-run with the clean seeds-on-winners pilot σ to finalise the TBD fields."
        if cfg.sigma_is_placeholder
        else "Committed\nbefore the Phase-1 freeze and RE-RUN at the measured seeds-on-winners pilot σ — no TBD fields remain."
    )
    pilot_caveat = (
        "> Pilot-dependent fields (σ_seed and therefore Δ_MDE) are **directional** until the seeds-on-winners "
        "pilot runs; re-run `scripts/power_analysis.py --sigma-seed <pilot value>` to finalise this file "
        "before the freeze."
        if cfg.sigma_is_placeholder
        else "> Pilot-dependent fields (σ_seed and therefore Δ_MDE) are **FINAL**: this file was generated at "
        "the MEASURED seeds-on-winners pilot σ, so nothing here is directional. Re-run only if the pilot "
        "is redone."
    )
    # ρ's caveat must describe the ρ actually used, not the CLI default.
    rho_note = (
        "default 0.0; NOT the worst case — the"
        if abs(cfg.rho) < 1e-12
        else "MEASURED, not the 0.0 default; ρ<0 is NOT the benign case — the"
    )

    grid = np.asarray(mde["grid"], dtype=float)
    powers = np.asarray(mde["power"], dtype=float)
    rows = list(range(0, grid.size, max(1, grid.size // 10)))
    if rows[-1] != grid.size - 1:
        rows.append(grid.size - 1)
    table_lines = ["| effect (Sharpe) | effect (σ_seed units / δ) | power |", "|---|---|---|"]
    for i in rows:
        table_lines.append(f"| {grid[i]:.3f} | {grid[i] / cfg.sigma_dsr:.2f} | {powers[i]:.3f} |")
    power_table = "\n".join(table_lines)

    if n_source == "cli":
        n_note = f" *(set via --n-regimes; auto-detected = {n_detected})*"
    elif detect_unreliable and unreliable_reason == "too_many":
        n_note = (
            f" *(design value; auto-detected = {n_detected} is UNRELIABLE — it exceeds the single-digit-"
            f"regimes premise (> {MAX_PLAUSIBLE_REGIMES} plausible-max). `independent_regime_count` over-"
            "segments the noisy vix into many short run-length blocks, not the ~6 macro regimes the design "
            "assumes, so the design value is used)*"
        )
    elif detect_unreliable:
        n_note = (
            f" *(design value; auto-detected = {n_detected} is UNRELIABLE — the gold `vix` field is a "
            "DECIMAL, not VIX points, so `config/regimes.yaml`'s 15/25 point thresholds collapse every date "
            "to one regime. UNITS MISMATCH — flagged for the user; not fixed here)*"
        )
    else:
        n_note = " *(auto-detected on the panel)*"

    reached = bool(mde["reached"])
    mde90_clause = ""
    if mde90 is not None and mde90.get("reached"):
        mde90_clause = f" · **Δ_MDE@90% = {_fmt(mde90['mde'])} Sharpe** ({_fmt(mde90['mde_sigma_units'], 2)} σ_seed)"
    mde_clause = (
        f"**Δ_MDE@{cfg.target_power:.0%} = {_fmt(mde_val)} Sharpe** ({_fmt(mde_sig, 2)} σ_seed){mde_flag}{mde90_clause}"
        if reached
        else f"**NOT reached on the grid** (target power {cfg.target_power:.2f} not attained up to "
        f"{grid[-1]:.3f} Sharpe){mde_flag}"
    )

    # T2.5: reconcile the Sharpe-unit MDE with the validation-DSR-unit SESOI via the documented conservative
    # (upper-bound) Sharpe->DSR mapping, so the null framing is stated in the SESOI's OWN units.
    dsr_mde = sharpe_mde_to_dsr(float(mde_val), track_length=val_track_length) if reached else 0.0  # type: ignore[arg-type]
    dsr_mde90 = (
        sharpe_mde_to_dsr(float(mde90["mde"]), track_length=val_track_length)  # type: ignore[arg-type]
        if (mde90 is not None and mde90.get("reached"))
        else 0.0
    )
    sesoi = float(cfg.sesoi)
    # Does the CEILING DSR shift the Sharpe-MDE could induce sit within the SESOI?
    dsr_within_sesoi = reached and dsr_mde <= sesoi
    if not reached:
        dsr_mde_clause = "**not computed** (the Sharpe MDE was not reached on the grid)"
    elif dsr_within_sesoi:
        dsr_mde_clause = (
            f"**Δ_MDE@{cfg.target_power:.0%} ≈ {_fmt(dsr_mde)} validation-DSR (conservative ceiling)** — "
            f"this is **≤ the {_fmt(sesoi)} SESOI**, so a Sharpe non-rejection at the MDE is *already* "
            "sub-SESOI in DSR units (the bound and the SESOI agree)."
        )
    else:
        dsr_mde_clause = (
            f"**Δ_MDE@{cfg.target_power:.0%} ≈ {_fmt(dsr_mde)} validation-DSR (conservative ceiling)** — "
            f"this **EXCEEDS the {_fmt(sesoi)} SESOI**, so the Sharpe-MDE and the DSR-SESOI probe DIFFERENT "
            "scales: a Sharpe non-rejection alone does NOT establish DSR-equivalence (the INCONCLUSIVE "
            "branch below — only the TOST CI *in DSR units* can)."
        )

    # R37 (M4): name the LIVE decision rule the PRIMARY MDE used, and ALWAYS report the Šidák-m=6 figure as
    # the conservative BH-over-6 sensitivity (it is never deleted).
    if cfg.iut_one_sided:
        mode_clause = (
            f"**ONE-SIDED IUT leg at α = {cfg.alpha:.2f}** (the live per-leg rule): each H2-RA / H2-Tail leg "
            "is tested one-sided in the predicted direction; the IUT/conjunction IS the within-H2 correction "
            "(Berger 1982), and multiplicity across the m = 6 union is the LIVE Benjamini-Hochberg / "
            "Romano-Wolf — NOT a fixed Šidák-α_eff. This is the PRIMARY MDE."
        )
    else:
        mode_clause = (
            f"**TWO-SIDED Šidák-over-m rule at α_eff = {a_eff:.4f}** (m = {cfg.n_comparisons}) — the "
            "conservative BH-over-6 figure, here selected as the primary."
        )
    sidak_a = sidak_alpha if sidak_alpha is not None else selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    if mde_sidak is not None:
        s_reached = bool(mde_sidak.get("reached"))
        sidak_clause = (
            f"**Δ_MDE@{cfg.target_power:.0%} (Šidák-m={cfg.n_comparisons}, two-sided, α_eff = {sidak_a:.4f}) = "
            f"{_fmt(mde_sidak.get('mde'))} Sharpe** ({_fmt(mde_sidak.get('mde_sigma_units'), 2)} σ_seed)"
            if s_reached
            else f"**NOT reached on the grid** at α_eff = {sidak_a:.4f}"
        )
        sidak_section = (
            "\n## Conservative BH-over-6 sensitivity (Šidák-m, NOT deleted)\n"
            "The pre-R25 design tested the family at a fixed Šidák-over-m α_eff (a conjunction ∘ BH-over-6, "
            "which double-corrected — Berger 1982). R25 made H2 two co-primary IUTs decided one-sided at α "
            "(the primary above), but the Šidák-m figure is RETAINED here as the conservative sensitivity so a "
            "reader sees both:\n\n"
            f"> {sidak_clause}\n\n"
            "As expected this conservative MDE is **≥** the primary one-sided IUT MDE (a smaller per-test α and "
            "a two-sided rule both raise the detectable effect); the design detects at least as small an effect "
            "under the live rule.\n"
        )
    else:
        sidak_section = (
            "\n## Multiplicity note\n"
            "This run's primary rule is already the conservative two-sided Šidák-over-m figure "
            f"(α_eff = {sidak_a:.4f}); the live one-sided IUT rule (`--mode iut_one_sided`) detects an effect "
            "no larger than this.\n"
        )

    # 2026-07-05 (M02): the σ_seed narrative is CONDITIONAL on the sigma source — the old renderer
    # printed the "clean pilot does not exist yet / directional upper bound" story UNCONDITIONALLY,
    # so a --sigma-seed run (the measured pilot value) produced a doc contradicting its own inputs and
    # the committed CAMPAIGN_power.md had to be hand-edited beyond what this generator could reproduce.
    if cfg.sigma_is_placeholder:
        sigma_seed_section = f"""## σ_seed — the noise input (DIRECTIONAL upper bound)
σ_seed is the seed-to-seed SD of the per-seed score within an arm. The clean pilot (winners re-run at
{cfg.seeds} seeds, reward FIXED) does not exist yet, so σ_seed = **{_fmt(cfg.sigma_dsr)}** is anchored on
the prototype archive's per-candidate within-arm dispersion of the per-seed **annualised Sharpe** ≈ 0.36
(pooled; headline `distributional`/`scalar` arms 0.36/0.37). This is an **UPPER BOUND** on the campaign
σ_seed — it folds reward-DESIGN variance (a different reward per candidate) and a 1-seed, shorter window
into the dispersion — so the MDE here is **pessimistic on the σ axis**; the clean pilot σ should be ≤ it,
yielding an MDE ≤ what is tabled. Replace via `--sigma-seed <pilot value>` before the freeze."""
    else:
        sigma_seed_section = f"""## σ_seed — the noise input (MEASURED clean-pilot value)
σ_seed is the seed-to-seed SD of the per-seed score within an arm. This run was invoked with the
MEASURED pilot value (`--sigma-seed {_fmt(cfg.sigma_dsr)}`, annualised-Sharpe units — the 2026-07-03
σ_D farm: two fixed hand-written rewards × 15 CRN seeds at B\\*=200k on the Split-C univ5 panel), so
every MDE above is computed at the pilot-measured noise, not the prototype-era directional proxy.
Disclosed caveat: the pilot's two hand-written rewards PROXY the (unknown until run) LLM-authored
campaign winners; the seed-count sizing therefore evaluates σ at its χ² upper confidence bound, which
buffers exactly that proxy error."""

    return f"""# Campaign power analysis (H2 headline) — generated by `scripts/power_analysis.py`

**Purpose.** Decide whether the planned design can detect a realistic H2 effect before spending compute,
and be honest in the dissertation about how much power the test has. A non-rejection of H2 is reported
as a **bounded effect**, NOT an underpowered failure — see "Pre-committed null framing" below. {purpose_tail}

## The test this powers (the realized headline — PREREGISTRATION §10 R16)
The frozen inference unit is `per_seed_iqm_paired_seed_bootstrap`. Concretely: each per-arm winner is
re-run at **n_seeds = {cfg.seeds}** training seeds; each seed's frozen-winner TEST series is reduced to
one per-seed score (the **annualised Sharpe** for the H2-conjunction leg, R13; CVaR for the tail leg);
the per-seed scores are reduced to an **rliable IQM** (Agarwal et al. 2021) and compared by
`src.inference.bootstrap.paired_seed_difference_test(a, b, statistic=iqm)` — a re-centred basic empirical
bootstrap that resamples the **shared training-seed indices** (paired across arms), consuming exactly
**n_seeds = {cfg.seeds}** paired per-seed scores. The realized sample size is therefore **n_seeds, NOT
seeds × folds × N_regimes** — this analysis was re-derived against that test (the superseded
seed-averaged-series model that inflated the effective n ~6× is retired). `simulate_power` rejects via
the SAME `paired_seed_difference_test`, so the simulated size + small-n behaviour are the real test's.

## Decision rule (R37; M4)
- [x] PRIMARY rule: {mode_clause}
- [x] Per-test alpha the PRIMARY MDE used: **α = {a_eff:.4f}**.
- [x] Conservative BH-over-6 sensitivity (Šidák-m={cfg.n_comparisons}, two-sided): **α_eff = {sidak_a:.4f}** *(reported below, never deleted)*.

{result_heading}
- [x] Realized PAIRED sample size **n_seeds = {cfg.seeds}** (per-arm winners; Amendment D2 5→30).
- [{'x' if not cfg.sigma_is_placeholder else ' '}] Seed-to-seed σ of the per-seed score (Sharpe units): **σ_seed = {_fmt(cfg.sigma_dsr)}**{sigma_flag}
- [x] Pairing correlation across the shared seed: **ρ = {cfg.rho:.3f}** *({rho_note}
      MEASURED pilot ρ is -0.141 and ρ<0 INFLATES σ_D; swept below, including the measured point)*
- [{'x' if reached else ' '}] Minimum detectable effect at {cfg.target_power:.0%} power: {mde_clause}
- [x] Per-test alpha (PRIMARY rule above): **α = {a_eff:.4f}** *(one-sided IUT leg at α; multiplicity is the live BH/RW, not a fixed Šidák-α_eff — R37)* {'' if cfg.iut_one_sided else '— this run uses the Šidák-m two-sided rule as primary'}
- [x] Independent-regime count from `src/regimes/definition.py`: **N = {cfg.n_regimes}**{n_note} — REPORTED for the SECONDARY regime-stratified analyses; it does **not** enter the headline paired n.
- [x] **Trial count for the campaign, stated explicitly: {n_trials}** (candidates generated across all arms)
- [x] SESOI (smallest effect size of interest): **{_fmt(cfg.sesoi)} validation-DSR** *(FROZEN; PREREGISTRATION §10 R12)*
- [x] TOST symmetric equivalence margin: **±{_fmt(cfg.equiv_margin)} validation-DSR**
- [{'x' if reached else ' '}] **Δ_MDE in the SESOI's units (T2.5 reconciliation):** {dsr_mde_clause} *(Sharpe→DSR conservative ceiling, T={val_track_length}; see "Sharpe ↔ validation-DSR reconciliation")*

## MDE definition
Δ_MDE is the smallest true per-seed-score gap Δ (distributional − scalar, in annualised-Sharpe units)
at which the realized rliable paired test (`paired_seed_difference_test`, IQM statistic) rejects H0 with
probability ≥ **target power = {cfg.target_power:.2f}**, at the per-test alpha of the PRIMARY rule (α = {a_eff:.4f}):

    Δ_MDE = min {{ Δ ≥ 0 : Power(Δ | n_seeds = {cfg.seeds}, σ_seed = {_fmt(cfg.sigma_dsr)}, ρ = {cfg.rho:.3f}, α = {a_eff:.4f}) ≥ {cfg.target_power:.2f} }}

estimated by Monte-Carlo ({cfg.n_sims} sims/grid-point, {cfg.n_boot} bootstrap reps/test) over the REAL test.
For the power simulation the bootstrap two-sided p is converted in-direction (p_one = p_two/2 when the
effect is in the predicted direction) — a symmetric-DGP approximation to the headline R64 direct upper-tail
rule (`pvalue_one_sided_greater`), exact here because the simulation draws symmetric (normal) pairs. The conservative Šidák-over-m figure
(`alpha_eff = 1 − (1 − {cfg.alpha:.2f})^(1/{cfg.n_comparisons}) = {sidak_a:.4f}`) is the BH-over-6 SENSITIVITY below.
{sidak_section}
## Pairing-correlation (ρ) sensitivity — headline MDE table
The paired test gains power as the two arms' per-seed scores correlate across the shared seed (ρ>0
cancels the common "lucky-seed" variance). NOTE: **ρ=0 is NOT the conservative worst case here** (corrected
2026-07-26): σ_D = σ_seed·√(2(1-ρ)), so a NEGATIVE ρ inflates the paired variance, and the pilot
MEASURED ρ = -0.141 (σ_D 0.3686 vs 0.3451 at ρ=0 — ρ=0 understates the MDE by ~7%). The sweep now
includes the measured negative point; read THAT row as the design's operating point. Any real ρ>0
shrinks the MDE; any ρ<0 inflates it. The ρ-sweep
table uses the PRIMARY decision rule above.

{rho_table}

{sigma_seed_section}

**External cross-check (Colas et al. 2019, arXiv:1904.06979).** For comparing central RL performance
across seeds at 80% power they report δ=0.5 → N≈100, δ=1 → N≈20, δ=2 → N≈5–10 (δ = |Δμ|/σ_pool; a real
SAC-vs-TD3 case needed N≈10–15 at δ≈0.93). At our **N={cfg.seeds}** the simulated 80%-power MDE is ≈ {_fmt(mde_sig, 2)} σ_seed
(δ≈{_fmt(mde_sig, 2)}), squarely in that envelope. Colas flag the *two-sample* bootstrap as anti-conservative
below N≈50; ours is the **paired/one-sample** variant (it cancels the common variance) and the simulated
null rejection is close to alpha_eff — see the power curve's effect=0 row. NOTE: CORRECTED 2026-07-26
(deep review): this previously asserted the null rejection is "≤ alpha_eff (conservative), not
inflated". Measured at production settings (statistic=iqm, n_boot=2000, n=30, 6,000 reps, MC SE 0.0028)
the real size is **0.0573 two-sided / 0.0613 one-sided at α=0.05** — mildly ABOVE nominal, so the
"conservative" direction was wrong. Small in absolute terms, but each IUT leg is decided at α=0.05 and
the joint IUT inherits the same drift; report the measured value rather than claiming conservatism.

## Computed power curve (σ_seed = {_fmt(cfg.sigma_dsr)}, ρ = {cfg.rho:.3f}, n_seeds = {cfg.seeds})
{power_table}

## Sharpe ↔ validation-DSR reconciliation (T2.5)
The MDE above is in **annualised-Sharpe** units (the difference test's scale), but the FROZEN SESOI / TOST
margin is **{_fmt(cfg.sesoi)} validation-DSR** units (the selection metric). These were never reconciled — a
Sharpe-MDE of {_fmt(mde_val)} and a DSR-SESOI of {_fmt(cfg.sesoi)} are not directly comparable. We close that
gap with a **documented, conservative (upper-bound) Sharpe→DSR map** (`sharpe_mde_to_dsr`).

The validation DSR is a Probabilistic Sharpe Ratio `DSR = Φ(z)`, `z = SR_pp·√(T−1)/√D`, with `SR_pp` the
**per-period** Sharpe and `D = 1 − skew·SR_pp + (kurt−1)/4·SR_pp²` (Bailey & López de Prado;
`src/inference/deflated_sharpe.py`). An annualised gap `ΔSR_ann` is `ΔSR_pp = ΔSR_ann/√{PERIODS_PER_YEAR}`
per period and, by the delta method, shifts the DSR by `ΔDSR ≈ φ(z)·√(T−1)/√D·ΔSR_pp`. The sensitivity
`φ(z)/√D` is **maximised** at the benchmark (`z=0 ⇒ φ(0)=0.3989`, DSR≈0.5) and under normal returns
(`D=1`), giving the conservative **ceiling**

    ΔDSR_max = 0.3989 · √(T−1) / √{PERIODS_PER_YEAR} · ΔSR_ann      (T = {val_track_length} scored sessions; the frozen Split-C validation window 2017–2019, first scored day ~2017-03-30 after the 60-session purge — the old "[2015,2017]" label was the superseded pre-Split-C window, 2026-07-05).

Plugging the headline MDE: {dsr_mde_clause}{(" The 90%-power Sharpe MDE maps to ≈ " + _fmt(dsr_mde90) + " validation-DSR (ceiling).") if dsr_mde90 > 0.0 else ""}

This ceiling is an UPPER bound (exact at-the-money/normal, smaller elsewhere: off-benchmark `z≠0` or fat
tails both reduce it). It is the honest direction for a null: if even the *largest* DSR shift the Sharpe-MDE
could cause is ≤ the {_fmt(cfg.sesoi)} SESOI, the Sharpe-MDE is unambiguously sub-SESOI in DSR units. At the
campaign's T≈{val_track_length} that ceiling **exceeds** the SESOI, so the Sharpe difference test and the
DSR SESOI **probe different scales** — which is exactly why the INCONCLUSIVE branch below is load-bearing and
the TOST equivalence must be evaluated **in DSR units** to license a bounded-effect (equivalence) claim.

## Pre-committed null framing (PREREGISTRATION §10 R12; viva Q21)
A non-rejection of H2 is reported as a **bounded effect**, defined as the conjunction:

> **non-rejection ⇔ (the H2 effect is bounded BELOW the Δ_MDE above, at {cfg.target_power:.0%} power and the
> Šidák per-test alpha) AND (the TOST 90% bootstrap CI for the arm difference lies INSIDE ±{_fmt(cfg.equiv_margin)}
> *in validation-DSR units*, i.e. the arms are practically equivalent within the SESOI).**

Both conditions, not either alone: the MDE bound says "any real effect is smaller than Δ_MDE"; the TOST
verdict says "and it is small enough to be practically equivalent".

**The INCONCLUSIVE branch (explicit, T2.5).** Because the Sharpe-unit MDE maps to a DSR shift that EXCEEDS the
{_fmt(cfg.sesoi)} SESOI (the reconciliation above), a Sharpe non-rejection **does not by itself** license an
equivalence/bounded-effect claim in the SESOI's units. Concretely:

> If H2 does not reject on the Sharpe difference test **and** the TOST 90% bootstrap CI (computed **in
> validation-DSR units**) lies INSIDE ±{_fmt(cfg.equiv_margin)} → report **bounded effect / practical
> equivalence**. If H2 does not reject but the DSR-unit TOST CI is **NOT** inside ±{_fmt(cfg.equiv_margin)}
> (or the DSR-unit TOST is not run) → report **INCONCLUSIVE** (underpowered for an effect this small in DSR
> terms), NOT equivalence (Lakens 2017). A Sharpe-only non-rejection is at most INCONCLUSIVE for the DSR
> equivalence claim.

*Units:* the frozen ±{_fmt(cfg.equiv_margin)} is in validation-DSR (the selection metric); `tost_equivalence`
applies the margin in whatever units it is handed, so feed it **per-seed validation-DSR** for the equivalence
verdict (do not conflate the DSR-scale SESOI with the Sharpe-scale difference test — stated, not silently
merged). The Sharpe→DSR ceiling above is what makes that unit gap quantitative rather than a caveat.

## Trial count (explicit)
Total candidates generated across all arms = **{n_trials}** (arms × candidates_per_arm, from
`config/campaign.yaml`). CPCV is applied to the *winners* afterward for inference (PREREGISTRATION §6),
not inside each candidate. This is the count stated for the Deflated-Sharpe deflation, with the
documented caveat that the *effective* trial count is ill-defined under guided search (DSR secondary).

## Method
1. Label regimes; count independent blocks (`independent_regime_count`). → N (reported; secondary only).
2. From the **seeds-on-winners pilot**{' (TBD)' if cfg.sigma_is_placeholder else ' (RUN)'}, estimate σ_seed = seed-to-seed SD of the per-seed score
   within an arm.{' Until then use the prototype directional upper bound.' if cfg.sigma_is_placeholder else ''} → σ_seed.
3. Monte-Carlo: under a grid of true effects, draw n_seeds correlated per-seed score pairs (per-arm SD
   σ_seed, pairing ρ) and reject via the REAL `paired_seed_difference_test` at alpha_eff. → power curve.
4. Report Δ_MDE at {cfg.target_power:.0%} (and 90%) power across ρ∈{{0,0.3,0.5,0.7}}; if the budget does not
   reach it, say so and record that the contribution may rest on a rigorous null + a TOST equivalence bound.
5. TOST: a 90% bootstrap CI for the arm score difference inside ±margin ⇒ practical equivalence.

## Acceptance boxes
- [x] A committed analysis re-derived against the REAL paired test, with MDE@80%/90% + an explicit trial count (this file).
- [x] PRIMARY decision rule = {'one-sided IUT leg at α = ' + f'{cfg.alpha:.2f}' + ' (multiplicity = live BH/RW, R37)' if cfg.iut_one_sided else 'two-sided Šidák-m = ' + str(cfg.n_comparisons)}; conservative Šidák-m={cfg.n_comparisons} sensitivity reported (α_eff = {sidak_a:.4f}).
- [x] Pairing correlation ρ swept {{-0.141, 0, 0.3, 0.5, 0.7}} — including the MEASURED pilot ρ;
      ρ=0 is NOT the worst case (ρ<0 inflates σ_D). Corrected 2026-07-26.
- [{'x' if not cfg.sigma_is_placeholder else ' '}] σ_seed filled from the clean pilot{' (currently the **directional upper-bound proxy**)' if cfg.sigma_is_placeholder else ' (**done** — measured pilot σ)'}.
- [{'x' if reached else ' '}] Target power {cfg.target_power:.0%} reached at Δ_MDE{' (currently at the directional σ_seed)' if cfg.sigma_is_placeholder else ' (at the measured pilot σ)'}.
- [x] SESOI + TOST margin recorded (FROZEN; PREREGISTRATION §10 R12); pre-committed null framing stated.

{pilot_caveat}
"""


#: The SESOI-crossing seed count at the POINT pilot σ̂_D — the authoritative MC-tool value
#: (simulate_power on the σ_D pilot; seed-decision doc 2026-07-05, "n ≈ 189"). Re-run the MC seed
#: sweep to refresh it if the pilot σ̂_D changes; the assurance ladder scales it analytically (n ∝ σ²).
ASSURANCE_N_POINT: int = 189
ASSURANCE_SIGMA_HAT_D: float = 0.369  # the 15-CRN-pair pilot σ_D (2026-07-03 farm)
ASSURANCE_PILOT_N: int = 15           # -> ν = 14
#: The grade-securing TIER ladder (cumulative seed bounds): distinction-core → 90% → 95% → 99%.
ASSURANCE_TIER_BOUNDS: tuple[int, ...] = (30, 100, 189, 279, 340, 403, 568)


def assurance_seed_count(
    confidence: float,
    *,
    sigma_hat_d: float = ASSURANCE_SIGMA_HAT_D,
    pilot_n: int = ASSURANCE_PILOT_N,
    n_point: int = ASSURANCE_N_POINT,
) -> dict[str, float]:
    """Seeds to power the SESOI at the χ² UPPER confidence bound of the pilot σ_D (seed-decision
    methodology). ν = ``pilot_n - 1``; σ_up(C) = σ̂_D·√(ν/χ²_{1-C,ν}); n(C) = round(n_point·ν/χ²_{1-C,ν})
    since n ∝ σ² at a fixed SESOI. REPRODUCES the seed-decision doc EXACTLY: 80%→279, 90%→340,
    95%→403, 99%→568 (verified in tests)."""
    from scipy.stats import chi2

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"assurance confidence must be in (0,1), got {confidence}")
    nu = int(pilot_n) - 1
    q = float(chi2.ppf(1.0 - confidence, nu))
    factor2 = nu / q
    return {
        "confidence": float(confidence),
        "sigma_up": float(sigma_hat_d) * factor2**0.5,
        "factor": factor2**0.5,
        "n": int(round(n_point * factor2)),
    }


def assurance_sweep_units() -> int:
    """DERIVE the uniform-n sweep unit count from config — never hardcode a roster size.

    Semantics (PLAN §3 catalogue), unchanged: every unit carried from the n=30 floor to the target n
    = the EXECUTED ARMS + the H1 baseline canon + the single H3 winner. The C4 block in
    ``campaign.run_campaign_tiered`` sweeps all but H3 (which runs as a separate ``--generations 1``
    invocation), but the TIME to a completed uniform design must include H3. D1 (levels tested at a
    fixed n=100) is NOT at the uniform n, so it is excluded.

    WHY DERIVED (deep review 2026-07-26). This was the literal ``12``, written as "7 arms + 4 H1 +
    H3". BOTH inputs later changed — R108 took the roster 7 -> 9, and the H1 canon expanded 4 -> 11 —
    leaving a PLANNING constant at 12 against a true 21, i.e. **43% low, with no signal**. Reading the
    roster from config is this module's own stated discipline (see ``_frozen_inference``: "code reads
    config, never hardcodes"), and it closes the staleness permanently rather than for one value.

    Raises
    ------
    ValueError
        If either roster is missing/empty — a silent fallback would reinstate exactly the
        stale-constant failure this replaces.
    """
    from src.utils.config import load_config

    arms = load_config("campaign").get("arms") or []
    h1 = load_config("preregistration").get("h1_baselines") or []
    if not arms or not h1:
        raise ValueError(
            "cannot derive the assurance sweep units: "
            f"campaign.yaml arms={len(arms)}, preregistration.yaml h1_baselines={len(h1)} "
            "(both must be non-empty; refusing to guess a planning constant)"
        )
    return len(arms) + len(h1) + 1  # + the H3 winner


def recommend_assurance_target(
    trainings_per_hour: float,
    days_available: float,
    *,
    ladder: dict[float, int] | None = None,
    n_floor: int = ASSURANCE_TIER_BOUNDS[0],
    sweep_units: int | None = None,
    safety_buffer_frac: float = 0.25,
) -> dict[str, Any]:
    """Throughput-aware, deadline-safe assurance target — GRADE SECURITY operationalised.

    The tier ladder (30 floor / 100 σ-precision / 189 point-estimate power / 279=80% / 340=90% /
    403=95% / 568=99%) says how many seeds each rung needs; this says how FAR up it we can safely go
    given the run's MEASURED speed and the calendar.
    Given the effective throughput ``trainings_per_hour`` (already reflecting sustained concurrency AND
    GPU packing — a number MEASURED at G1, never guessed) and ``days_available`` before the submission
    buffer, return the HIGHEST assurance tier whose uniform-n sweep (``sweep_units`` units from
    ``n_floor`` to that tier's n) finishes within ``days_available`` × (1 − ``safety_buffer_frac``).
    ``sweep_units`` defaults to ``assurance_sweep_units()`` — DERIVED from the live rosters in config,
    never a hardcoded count — and is echoed in the returned dict so the planning input is auditable.

    The n=30 floor — the complete distinction-grade study — is the recommendation of last resort (if
    even 90% will not fit, we still bank the floor). The stop is EXOGENOUS (throughput + calendar,
    NEVER the observed effect), so whichever tier we land on is a valid pre-committed single-look design
    — no optional stopping. This is the mechanism that makes "secure the grade safely" a property of the
    plan rather than a hope: at G1 we measure speed, run this, and adopt the safest target that fits.
    """
    if trainings_per_hour <= 0 or days_available <= 0:
        raise ValueError("trainings_per_hour and days_available must both be positive")
    if sweep_units is None:
        sweep_units = assurance_sweep_units()
    if int(sweep_units) <= 0:
        raise ValueError(f"sweep_units must be positive; got {sweep_units!r}")
    if ladder is None:
        ladder = {c: assurance_seed_count(c)["n"] for c in (0.90, 0.95, 0.99)}
    budget_h = float(days_available) * 24.0 * (1.0 - float(safety_buffer_frac))

    def sweep_hours(n_target: int) -> float:
        return sweep_units * max(0, int(n_target) - int(n_floor)) / float(trainings_per_hour)

    tiers = [
        {"confidence": float(c), "n": int(n), "sweep_hours": sweep_hours(n),
         "sweep_days": sweep_hours(n) / 24.0, "fits": sweep_hours(n) <= budget_h}
        for c, n in sorted(ladder.items())
    ]
    reachable = [t for t in tiers if t["fits"]]
    best = max(reachable, key=lambda t: t["confidence"]) if reachable else None
    return {
        "trainings_per_hour": float(trainings_per_hour),
        "days_available": float(days_available),
        "budget_hours": budget_h,
        "n_floor": int(n_floor),
        "sweep_units": int(sweep_units),
        "recommended_confidence": (best["confidence"] if best else None),
        "recommended_n": (best["n"] if best else int(n_floor)),
        "floor_only": best is None,
        "tiers": tiers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Power analysis for the headline H2 campaign test (FINAL_PLAN B-5; viva Q21; R16).",
    )
    parser.add_argument(
        "--assurance", action="store_true",
        help="Print the χ² upper-CI ASSURANCE seed ladder (80/90/95/99%%) + the grade-securing tier "
        "bounds, then exit. Reproduces the seed-decision sizing (279/340/403/568).",
    )
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--sigma-seed",
        type=float,
        default=None,
        help="Seed-to-seed sigma of the per-seed score within an arm (CLEAN PILOT INPUT). Omit to use the "
        "flagged directional upper-bound proxy from the prototype.",
    )
    # Back-compat alias for the old flag name; --sigma-seed wins if both are given.
    parser.add_argument("--sigma-dsr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rho", type=float, default=0.0, help="Pairing correlation across the shared seed (default 0.0, conservative).")
    parser.add_argument("--sesoi", type=float, default=PLACEHOLDER_SESOI)
    parser.add_argument("--equiv-margin", type=float, default=PLACEHOLDER_EQUIV_MARGIN)
    parser.add_argument("--n-comparisons", type=int, default=FROZEN_FAMILY_M)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--effect-max", type=float, default=5.0)
    parser.add_argument("--effect-points", type=int, default=41)
    parser.add_argument(
        "--val-track-length",
        type=int,
        default=VALIDATION_TRACK_LENGTH,
        help="Validation track length T (the sqrt(T-1) factor in the Sharpe->validation-DSR MDE mapping; "
        f"T2.5). Default {VALIDATION_TRACK_LENGTH} = the executed Split-C validation window [3081,3775) "
        "(2017-03-30..2019-12-31).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help=f"Override the per-arm winner seed count n_seeds = the realized paired sample size "
        f"(default: from config, {DESIGN_N_SEEDS}).",
    )
    parser.add_argument(
        "--n-regimes",
        type=int,
        default=None,
        help="Override the independent-regime count N (REPORTED only; bounds the secondary regime-"
        f"stratified analyses). Default: the design value ({DESIGN_REGIME_COUNT}); the auto-detected "
        "panel count is reported as a cross-check (see units caveat in the docstring).",
    )
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["iut_one_sided", "sidak_two_sided"],
        default="iut_one_sided",
        help="Primary decision rule for the MDE (R37/M4). 'iut_one_sided' (default, the LIVE rule): per-leg "
        "one-sided at alpha inside an IUT; multiplicity is the live BH/Romano-Wolf, NOT a fixed Šidák-α_eff. "
        "'sidak_two_sided': the conservative two-sided Šidák-over-m rule, reported as the BH-over-6 sensitivity.",
    )
    parser.add_argument("--out", default="outputs/CAMPAIGN_power.md")
    return parser


class _DesignInfo(TypedDict):
    """Typed result of :func:`_load_design_from_config` (so the consumers need no casts)."""

    n_regimes_detected: int
    detect_unreliable: bool
    unreliable_reason: str
    max_plausible_regimes: int
    vix_max: float
    n_seeds: int
    n_trials: int


def _load_design_from_config() -> _DesignInfo:
    """Read N (regimes on the gold/synthetic panel), n_seeds, and the trial count.

    Falls back to a synthetic panel when the licensed gold panel is unavailable (so the script and its
    tests run offline). The trial count is ``len(arms) * candidates_per_arm`` from ``config/campaign.yaml``.
    """
    from src.utils.config import load_config

    regimes_cfg = load_config("regimes")
    panel: Any  # duck-typed Panel (gold loader returns a wrapper; synthetic returns the Panel directly)
    try:
        from src.data.loaders import load_gold_panel

        loaded = load_gold_panel()
        panel = getattr(loaded, "panel", loaded)
    except Exception as _gold_exc:  # pragma: no cover - gold is licensed / may be absent
        from src.data.synthetic import make_synthetic_panel

        # LOUD, not silent (2026-07-26 review). The fallback itself is correct and deliberate — gold is
        # licensed, so an external reproducer legitimately has none — but it was SILENT, and the regime
        # count derived here is a DESIGN INPUT to the power analysis (hence the seed ladder). Substituting
        # an 8-asset synthetic panel for the 963-name gold without a word means an operator could read a
        # regime count off synthetic data and never know. Same fix P7 applied to analyze_campaign's
        # benchmark floor: name the exception, keep going.
        print(
            f"[power_analysis] gold panel UNAVAILABLE ({type(_gold_exc).__name__}: {_gold_exc}) — "
            "falling back to a SYNTHETIC panel for the regime count; this is a stand-in, not the "
            "executed design input. Provide the licensed gold to compute it on real data.",
            flush=True,
        )
        panel = make_synthetic_panel(n_assets=8, n_days=2_000, seed=0)

    n_regimes_detected = regime_count_on_panel(panel, regimes_cfg)
    vix = np.asarray(panel.vix, dtype=float)
    vix_looks_fractional = bool(np.nanmax(vix) < 5.0)
    too_many = bool(n_regimes_detected > MAX_PLAUSIBLE_REGIMES)
    detect_unreliable = vix_looks_fractional or n_regimes_detected <= 1 or too_many
    if detect_unreliable:
        unreliable_reason = (
            "too_many" if (too_many and not vix_looks_fractional and n_regimes_detected > 1) else "collapsed"
        )
    else:
        unreliable_reason = "none"

    campaign = load_config("campaign")
    arms = list(campaign.get("arms", []))
    cpa = int(campaign.get("candidates_per_arm", 0))
    # n_seeds is the per-arm winner seed count (the realized paired n). Prefer the frozen prereg list.
    prereg = load_config("preregistration")
    # Prefer the frozen prereg seeds, then the campaign's; resolve either form (list OR schema) to
    # the flat set. A naive list(dict) on the tiered schema would yield its keys, not the seeds.
    from src.utils.seeds import resolve_seeds as _resolve_seeds

    _sc = prereg.get("seeds") or campaign.get("seeds")
    try:
        seeds_list = _resolve_seeds(_sc) if _sc else list(range(DESIGN_N_SEEDS))
    except Exception:  # noqa: BLE001 - malformed schema -> documented design default
        seeds_list = list(range(DESIGN_N_SEEDS))
    n_trials = len(arms) * cpa
    return {
        "n_regimes_detected": int(n_regimes_detected),
        "detect_unreliable": detect_unreliable,
        "unreliable_reason": unreliable_reason,
        "max_plausible_regimes": int(MAX_PLAUSIBLE_REGIMES),
        "vix_max": float(np.nanmax(vix)),
        "n_seeds": int(len(seeds_list)),
        "n_trials": int(n_trials),
    }


def main() -> None:
    # Line-buffer the streams FIRST (batch-5 M2, 2026-07-03): a redirected/scheduled run block-buffers
    # stdout, so the old multi-hour sweep emitted ZERO bytes and was mis-diagnosed as a hang (three
    # orphaned runs killed). Line-buffering + the per-sweep progress prints below make a long run
    # observably alive from its first line. (Prints stay ASCII-only per the console note below.)
    import sys as _sys

    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args()

    if args.assurance:
        # ASCII-only console prints (the Windows cp1251 console cannot encode chi2/sigma glyphs).
        nu = ASSURANCE_PILOT_N - 1
        print("[power_analysis] chi2 upper-CI ASSURANCE seed ladder (secure-the-grade sizing)")
        print(f"  pilot sigma_hat_D={ASSURANCE_SIGMA_HAT_D}  nu={nu}  point SESOI-crossing n_point={ASSURANCE_N_POINT}")
        print("  level   sigma_up   factor   n")
        for c in (0.80, 0.90, 0.95, 0.99):
            r = assurance_seed_count(c)
            print(f"   {c:.2f}    {r['sigma_up']:.3f}    {r['factor']:.3f}   {int(r['n'])}")
        print(f"  grade-securing TIER ladder (cumulative): {list(ASSURANCE_TIER_BOUNDS)} = "
              "distinction-core(30) / 90% / 95% / 99%")
        return

    print("[power_analysis] loading design/config + panel (regime count) ...")
    design = _load_design_from_config()
    # --sigma-seed is the canonical flag; --sigma-dsr is the deprecated alias.
    sigma_arg = args.sigma_seed if args.sigma_seed is not None else args.sigma_dsr
    sigma_is_placeholder = sigma_arg is None
    sigma_seed = DIRECTIONAL_SIGMA_SEED if sigma_is_placeholder else float(sigma_arg)
    n_seeds = int(args.n_seeds) if args.n_seeds is not None else int(design["n_seeds"])

    detected = int(design["n_regimes_detected"])
    detect_unreliable = bool(design["detect_unreliable"])
    unreliable_reason = str(design["unreliable_reason"])
    # N is REPORTED (secondary). For the HEADLINE paired n we deliberately set n_regimes=1 so that
    # n_paired == seeds == n_seeds (regimes do NOT inflate the per-seed paired test). The reported N for
    # the doc note is chosen as: explicit --n-regimes wins; else the design value when auto-detect is
    # unreliable; else the detected count.
    if args.n_regimes is not None:
        n_regimes_reported = int(args.n_regimes)
        n_source = "cli"
    elif detect_unreliable:
        n_regimes_reported = DESIGN_REGIME_COUNT
        n_source = "design_default"
    else:
        n_regimes_reported = detected
        n_source = "auto_detected"

    # R37 (M4): the PRIMARY MDE is the LIVE one-sided IUT rule unless --mode overrides it.
    iut_one_sided = str(args.mode) == "iut_one_sided"

    # HEADLINE config: n_regimes=1, folds=1 => n_paired = seeds = n_seeds (the realized paired count).
    cfg = PowerConfig(
        n_regimes=1,
        seeds=n_seeds,
        folds=1,
        sigma_dsr=sigma_seed,
        rho=float(args.rho),
        target_power=float(args.target_power),
        alpha=float(args.alpha),
        n_comparisons=int(args.n_comparisons),
        sesoi=float(args.sesoi),
        equiv_margin=float(args.equiv_margin),
        n_sims=int(args.n_sims),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        sigma_is_placeholder=sigma_is_placeholder,
        iut_one_sided=iut_one_sided,
    )

    print("[power_analysis] B-5 (R16/R37) — power for the REALIZED paired per-seed IQM H2 test")
    print(f"  mode = {args.mode}  (primary: {'one-sided IUT leg at alpha' if iut_one_sided else 'two-sided Šidák-α_eff'})")
    print(f"  n_seeds (realized PAIRED sample size) = {cfg.n_paired}")
    print(f"  N (independent regimes, REPORTED/secondary) = {n_regimes_reported}  [source: {n_source}]")
    if detect_unreliable and unreliable_reason == "too_many":
        print(
            f"  NOTE: auto-detected N={detected} is UNRELIABLE (> {design['max_plausible_regimes']} "
            "plausible-max; independent_regime_count over-segments the noisy vix; using the design value)."
        )
    elif detect_unreliable:
        print(
            f"  NOTE: auto-detected N={detected} is UNRELIABLE (panel vix max={design['vix_max']:.3f}; the "
            "gold vix is a DECIMAL but regimes.yaml thresholds are VIX POINTS 15/25 — units mismatch FLAGGED)."
        )
    print(
        f"  sigma_seed = {cfg.sigma_dsr:.3f}"
        + ("  (DIRECTIONAL upper-bound proxy — prototype)" if sigma_is_placeholder else "  (from --sigma-seed)")
    )
    print(f"  rho = {cfg.rho:.3f}  (default 0.0 is NOT the worst case; sweeping {{-0.141,0,0.3,0.5,0.7}})")
    a_eff = float(cfg.alpha) if iut_one_sided else selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    a_sidak = selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    if iut_one_sided:
        # ASCII-only console prints (the Windows cp1251 console cannot encode the doc's proper Unicode; the
        # markdown file itself is written UTF-8 with the correct glyphs).
        print(f"  PRIMARY per-test alpha = {a_eff:.4f} (one-sided IUT leg; NO Sidak penalty - the IUT is the correction)")
        print(f"  SENSITIVITY Sidak-m={cfg.n_comparisons} alpha_eff = {a_sidak:.4f} (reported alongside, not the gate)")
    else:
        print(f"  alpha={cfg.alpha}, selection-aware alpha_eff={a_eff:.4f} (m={cfg.n_comparisons})")

    print(f"  [sweep 1/4] primary MDE @ {cfg.target_power:.2f} (up to {args.effect_points} grid points x {cfg.n_sims} sims) ...")
    mde = minimum_detectable_effect(cfg, effect_max=float(args.effect_max), effect_points=int(args.effect_points))
    print("  [sweep 2/4] MDE @ 0.90 ...")
    mde90 = _mde_at(cfg, 0.90, float(args.effect_max), int(args.effect_points))
    if mde["reached"]:
        print(f"  MDE @ power {cfg.target_power:.2f}: {_fmt(mde['mde'], 4)} Sharpe ({_fmt(mde['mde_sigma_units'], 2)} sigma_seed)")
        # T2.5: the same MDE expressed in the SESOI's validation-DSR units (conservative ceiling).
        dsr_ceiling = sharpe_mde_to_dsr(float(mde["mde"]), track_length=int(args.val_track_length))  # type: ignore[arg-type]
        verdict = "<= SESOI (agree)" if dsr_ceiling <= float(cfg.sesoi) else "> SESOI (DIFFERENT scales -> INCONCLUSIVE unless DSR-unit TOST passes)"
        print(f"  MDE in DSR units (T={int(args.val_track_length)}, conservative ceiling): {dsr_ceiling:.3f} validation-DSR  [{verdict}; SESOI={cfg.sesoi:.3f}]")
    else:
        print(f"  MDE @ power {cfg.target_power:.2f}: NOT reached on the grid")

    # The Šidák-m=6 figure is NEVER deleted: when the primary IS the one-sided IUT, compute the conservative
    # two-sided Šidák-over-6 MDE as the REPORTED BH-over-6 SENSITIVITY alongside; when the primary already is
    # the Šidák rule, the sensitivity is the same record (no second pass).
    mde_sidak: dict[str, object] | None = None
    if iut_one_sided:
        print("  [sweep 3/4] Sidak-m sensitivity MDE ...")
        mde_sidak = minimum_detectable_effect(
            replace(cfg, iut_one_sided=False), effect_max=float(args.effect_max), effect_points=int(args.effect_points)
        )
        if mde_sidak["reached"]:
            print(f"  SENSITIVITY MDE @ {cfg.target_power:.2f} (Sidak-m={cfg.n_comparisons}): "
                  f"{_fmt(mde_sidak['mde'], 4)} Sharpe ({_fmt(mde_sidak['mde_sigma_units'], 2)} sigma_seed)")

    print("  [sweep 4/4] rho sensitivity table (4 rho x 2 target powers) ...")
    rho_table = _rho_sweep_table(cfg, (-0.141, 0.0, 0.3, 0.5, 0.7), float(args.effect_max), int(args.effect_points))
    n_trials = int(design["n_trials"])
    print(f"  trial count = {n_trials}")

    # The doc reports N (secondary) in its note; the headline cfg keeps n_regimes=1 (n_paired == n_seeds),
    # so we render with n_regimes overridden to the REPORTED N for that note only — the simulated power
    # already used the headline cfg, so this display override does not change any computed number.
    md = render_markdown(
        replace(cfg, n_regimes=n_regimes_reported),
        mde,
        n_trials,
        n_detected=detected,
        detect_unreliable=detect_unreliable,
        n_source=n_source,
        unreliable_reason=unreliable_reason,
        rho_table=rho_table,
        mde90=mde90,
        mde_sidak=mde_sidak,
        sidak_alpha=a_sidak,
        val_track_length=int(args.val_track_length),
    )
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[power_analysis] wrote {out_path}")


if __name__ == "__main__":
    main()
