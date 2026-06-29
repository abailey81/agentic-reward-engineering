"""Tests for src/feedback/measurement.py — empirical body + EVT tails (F.2, B.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.feedback.measurement import (
    CVAR_01_HIGH_VARIANCE_NOTE,
    EVT_ESTIMATOR_NOTE,
    EVT_NONREGULAR_XI_NOTE,
    FED_HEADLINE_CVAR_LEVELS,
    ReturnDistribution,
    fed_estimator_log,
    reset_fed_estimator_log,
)

FROZEN_KEYS = {"cvar_01", "cvar_05", "cvar_10", "cvar_25", "left_tail_mass", "robust_skew"}

# Hypothesis is an OPTIONAL dev dependency. Guard the import so this module (and all the
# example tests above) still collects in a minimal env; the property-based coherence block
# below is skipped per-test when it is absent, and seeded-numpy fallbacks cover the same
# axioms unconditionally so the laws are never left untested.
try:  # pragma: no cover - trivial import guard
    from hypothesis import given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    _HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    # No-op shims so the @settings/@given decorators AND their strategy arguments (st.floats(...),
    # arrays(...)) evaluate harmlessly when hypothesis is absent — otherwise they raise NameError/TypeError
    # at COLLECTION time, defeating the _needs_hypothesis skip below (the file must degrade gracefully on a
    # minimal core install, which is exactly what the skip marker promises).
    _HAS_HYPOTHESIS = False

    def _noop_decorator(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        def _wrap(fn):  # type: ignore[no-untyped-def]
            return fn
        return _wrap

    given = settings = _noop_decorator  # type: ignore[assignment]

    class _DummyStrategies:  # any attribute access returns a callable that tolerates any args
        def __getattr__(self, _name: str):  # type: ignore[no-untyped-def]
            return lambda *_a, **_k: None

    st = _DummyStrategies()  # type: ignore[assignment]

    def arrays(*_a: object, **_k: object) -> None:  # type: ignore[misc]
        return None

_needs_hypothesis = pytest.mark.skipif(not _HAS_HYPOTHESIS, reason="hypothesis not installed")


def _normal_es(alpha: float) -> float:
    """Closed-form standard-normal Expected Shortfall (signed, left tail)."""
    z = stats.norm.ppf(alpha)
    return -stats.norm.pdf(z) / alpha


def test_empirical_cvar_and_quantiles_match_closed_form_on_normal_fixture(
    normal_returns: np.ndarray,
) -> None:
    """Empirical CVaR/quantiles match the closed-form normal values within tol (B-1)."""
    rd = ReturnDistribution().fit(normal_returns)

    # Empirical CVaR(5%) vs closed-form normal ES within ~3%.
    emp_cvar5 = rd.cvar(0.05, method="empirical")
    closed = _normal_es(0.05)
    assert abs(emp_cvar5 - closed) / abs(closed) < 0.03

    # Quantiles match scipy norm.ppf within tolerance.
    qs = rd.quantiles([0.05, 0.25, 0.50, 0.95])
    for tau, val in qs.items():
        assert abs(val - stats.norm.ppf(tau)) < 0.02


def test_evt_cvar_recovers_known_gpd_tail_within_tolerance() -> None:
    """GPD/EVT tail fit recovers CVaR of a synthetic known-GPD tail (B-1, EX-DRL)."""
    rng = np.random.default_rng(7)
    xi_true, beta_true = 0.2, 0.5
    # Build returns whose losses have a GPD upper tail: body N(0,1), tail GPD above u.
    u = 2.0
    fu = 0.10  # tail fraction
    n = 200_000
    n_tail = int(n * fu)
    body = stats.truncnorm.rvs(-np.inf, u, size=n - n_tail, random_state=rng)
    tail = u + stats.genpareto.rvs(xi_true, loc=0.0, scale=beta_true, size=n_tail,
                                   random_state=rng)
    losses = np.concatenate([body, tail])
    returns = -losses

    rd = ReturnDistribution(threshold_q=fu).fit(returns)

    # Closed-form GPD CVaR for losses at alpha=0.01 (within tail), signed return-space.
    alpha = 0.01
    var_loss = u + (beta_true / xi_true) * ((alpha / fu) ** (-xi_true) - 1.0)
    cvar_loss = (var_loss + beta_true - xi_true * u) / (1.0 - xi_true)
    expected = -cvar_loss

    evt = rd.cvar(alpha, method="evt")
    assert abs(evt - expected) / abs(expected) < 0.10

    # Fitted shape/scale recovered reasonably.
    assert abs(rd.xi - xi_true) < 0.1
    assert abs(rd.beta - beta_true) < 0.15


def test_cvar_is_monotone_in_alpha(heavy_tail_returns: np.ndarray) -> None:
    """cvar(alpha) is non-increasing as alpha shrinks (deeper tail <= shallower)."""
    rd = ReturnDistribution().fit(heavy_tail_returns)
    c01 = rd.cvar(0.01)
    c05 = rd.cvar(0.05)
    c10 = rd.cvar(0.10)
    c25 = rd.cvar(0.25)
    assert c01 <= c05 <= c10 <= c25


def test_cvar_monotone_on_normal(normal_returns: np.ndarray) -> None:
    """Monotonicity also holds on the light-tailed normal fixture."""
    rd = ReturnDistribution().fit(normal_returns)
    assert rd.cvar(0.01) <= rd.cvar(0.05) <= rd.cvar(0.10) <= rd.cvar(0.25)


def test_tail_stats_returns_exactly_frozen_fields(normal_returns: np.ndarray) -> None:
    """tail_stats keys are EXACTLY the six frozen fields."""
    rd = ReturnDistribution().fit(normal_returns)
    ts = rd.tail_stats()
    assert set(ts.keys()) == FROZEN_KEYS
    assert all(math.isfinite(v) for v in ts.values())


def test_robust_skew_negative_on_left_skewed_fixture() -> None:
    """robust_skew is negative when the left tail is longer (B.10)."""
    rng = np.random.default_rng(11)
    # Negative-shifted lognormal -> long left tail.
    left_skewed = -rng.lognormal(mean=0.0, sigma=0.5, size=100_000)
    rd = ReturnDistribution().fit(left_skewed)
    assert rd.tail_stats()["robust_skew"] < 0.0


def test_empirical_evt_crosscheck_path(heavy_tail_returns: np.ndarray) -> None:
    """The method flag lets a test compare EVT vs empirical CVaR in the tail."""
    rd = ReturnDistribution().fit(heavy_tail_returns)
    emp = rd.cvar(0.05, method="empirical")
    evt = rd.cvar(0.05, method="evt")
    # Both are valid downside estimates of the same order of magnitude.
    assert emp < 0 and evt < 0
    assert abs(evt - emp) / abs(emp) < 0.5


def test_cvar_01_documented_high_variance() -> None:
    """cvar_01 is EVT-estimated and documented high-variance (audit B-7)."""
    assert "high-variance" in CVAR_01_HIGH_VARIANCE_NOTE.lower()
    assert "B-7" in CVAR_01_HIGH_VARIANCE_NOTE


def test_evt_estimator_disclosed_as_plain_mle_troop_future_work() -> None:
    """The EVT estimator is honestly disclosed as plain GPD MLE; Troop POT is future work.

    Guards the DEEP_H2 sec.6.2 disclosure so the docstring/note cannot silently drift
    back to promising a bias-corrected estimator the code does not perform.
    """
    note = EVT_ESTIMATOR_NOTE.lower()
    assert "plain gpd maximum-likelihood" in note
    assert "no bias correction" in note
    assert "troop" in note
    assert "future work" in note


def _signed_gpd_cvar(u: float, fu: float, xi: float, beta: float, alpha: float) -> float:
    """Closed-form signed (return-space) GPD-POT CVaR at level alpha within the tail."""
    var_loss = u + (beta / xi) * ((alpha / fu) ** (-xi) - 1.0)
    cvar_loss = (var_loss + beta - xi * u) / (1.0 - xi)
    return -cvar_loss


def test_plain_mle_cvar_bias_is_small_and_variance_dominated_in_regime() -> None:
    """Honest in-regime exhibit: the plain GPD-MLE CVaR error at n~750 is dominated by
    VARIANCE, not bias (DEEP_H2 sec.6.4 / L15).

    This is the quantitative justification for shipping the plain MLE and deferring the
    Troop (2021) bias-corrected POT: a bias correction targets the ~few-% bias term,
    which is negligible here relative to the estimator's standard deviation, so it
    cannot improve RMSE in this regime. Few exceedances by construction (~75 of ~750).
    """
    rng = np.random.default_rng(20240625)
    xi_true, beta_true, u, fu, scale = 0.15, 0.012, 0.018, 0.10, 0.01
    n, reps = 750, 400

    def one_sample() -> np.ndarray:
        n_tail = int(round(n * fu))
        body = (
            stats.truncnorm.rvs(-np.inf, u / scale, size=n - n_tail, random_state=rng)
            * scale
        )
        tail = u + stats.genpareto.rvs(
            xi_true, loc=0.0, scale=beta_true, size=n_tail, random_state=rng
        )
        return -np.concatenate([body, tail])  # returns = -losses

    for alpha in (0.05, 0.01):
        est = np.array(
            [ReturnDistribution(threshold_q=fu).fit(one_sample()).cvar(alpha, method="evt")
             for _ in range(reps)]
        )
        true = _signed_gpd_cvar(u, fu, xi_true, beta_true, alpha)
        bias = float(est.mean() - true)
        sd = float(est.std())
        # Few exceedances (the regime the EVT fit is for).
        assert int(round(n * fu)) <= 100
        # Bias is small in absolute terms relative to the truth (<~5% at 5%, <~12% at 1%).
        rel_bias_cap = 0.05 if alpha == 0.05 else 0.12
        assert abs(bias) / abs(true) < rel_bias_cap
        # And crucially VARIANCE-dominated: |bias| is a small fraction of the std, so a
        # bias-only correction (Troop UPOT) cannot materially reduce RMSE here.
        assert abs(bias) < 0.5 * sd


def test_threshold_sensitivity_diagnostic(heavy_tail_returns: np.ndarray) -> None:
    """threshold_sensitivity returns per-threshold CVaRs + a finite, non-negative spread,
    and does not mutate the fitted estimator (side-effect-free)."""
    rd = ReturnDistribution(threshold_q=0.10).fit(heavy_tail_returns)
    before = rd.tail_stats()
    sens = rd.threshold_sensitivity(alpha=0.01, threshold_qs=(0.05, 0.10, 0.20))
    assert {"0.05", "0.10", "0.20", "spread", "cv", "n_empirical_fallback"} <= set(sens)
    assert np.isfinite(sens["spread"]) and sens["spread"] >= 0.0
    assert 0.0 <= sens["n_empirical_fallback"] <= 3.0  # 0..len(threshold_qs) routed to empirical fallback
    assert all(v <= 0.0 for k, v in sens.items() if k not in {"spread", "cv", "n_empirical_fallback"})  # signed losses
    # estimator state unchanged by the diagnostic
    assert rd.threshold_q == 0.10
    assert rd.tail_stats() == before


def test_threshold_sensitivity_refits_on_time_ordered_returns() -> None:
    """Internal refits must receive the TIME-ORDERED series, never ``sorted_returns`` (2026-07-26).

    The POT/GPD path is order-invariant, so the returned values are byte-identical either way — this
    pins the INVARIANT, not the numbers. It matters because each internal replicate is a fully-formed
    ReturnDistribution whose ``_raw`` is what the stationary block bootstrap resamples: seeded from a
    SORTED series it would resample monotone runs, destroy the serial dependence the block scheme
    exists to preserve, and silently report a materially too-wide CI (~3.5x on an AR(1) fixture).
    """
    rng = np.random.default_rng(7)
    eps = rng.standard_t(4, 1200) * 0.008
    r = np.empty(1200)
    r[0] = eps[0]
    for i in range(1, 1200):
        r[i] = 0.15 * r[i - 1] + eps[i]  # serial dependence sorting would destroy
    assert not np.all(np.diff(r) >= 0.0), "fixture precondition: the series is not already sorted"

    seen: list[np.ndarray] = []
    original_fit = ReturnDistribution.fit

    def _spy(self, train_realized_returns):  # noqa: ANN001, ANN202 - local test spy
        seen.append(np.asarray(train_realized_returns, dtype=float).ravel())
        return original_fit(self, train_realized_returns)

    rd = ReturnDistribution(threshold_q=0.10).fit(r)
    ReturnDistribution.fit = _spy  # type: ignore[method-assign]
    try:
        rd.threshold_sensitivity(alpha=0.01, threshold_qs=(0.05, 0.10, 0.20))
    finally:
        ReturnDistribution.fit = original_fit  # type: ignore[method-assign]

    assert len(seen) == 3, f"expected one refit per probed threshold, saw {len(seen)}"
    for got in seen:
        assert not np.all(np.diff(got) >= 0.0), "a refit received a SORTED series (the _raw trap)"
        np.testing.assert_allclose(got, r)


# --------------------------------------------------------------------------- #
# T2.8(a) — the xi <= -0.5 non-regular-GPD-MLE guard (mirrors the xi >= 1 guard)
# --------------------------------------------------------------------------- #


def _uniform_loss_returns(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Returns whose LOSSES (= -returns) are uniform on [0, 1].

    A hard-bounded (finite right endpoint) loss tail forces a NEGATIVE GPD shape; the
    GPD MLE on its exceedances lands at xi ~= -1 (well inside the non-regular xi <= -0.5
    region), which is the branch T2.8(a) added a guard for.
    """
    rng = np.random.default_rng(seed)
    return -rng.uniform(0.0, 1.0, size=n)


def test_evt_falls_back_to_empirical_when_xi_le_minus_half() -> None:
    """A synthetic xi <= -0.5 sample makes _evt_cvar fall back to the EMPIRICAL CVaR (T2.8a).

    The non-regular GPD MLE region (Smith 1985) is now guarded the same way as the xi >= 1
    infinite-mean region: the EVT estimator must RETURN the empirical value there, exactly.
    """
    rd = ReturnDistribution(threshold_q=0.10).fit(_uniform_loss_returns())
    # The fitted shape is in the non-regular region and the guard recognises it.
    assert rd.xi <= -0.5
    assert rd._evt_falls_back(0.05) == "xi_le_-0.5"
    # EVT routing returns EXACTLY the empirical CVaR (the fallback fired, no extrapolation).
    evt = rd.cvar(0.05, method="evt")
    emp = rd.cvar(0.05, method="empirical")
    assert evt == pytest.approx(emp, abs=0.0)
    # And it is a finite, signed loss (not NaN/inf from a degenerate closed form).
    assert math.isfinite(evt) and evt < 0.0


def test_evt_used_when_xi_in_regular_region_control() -> None:
    """CONTROL: a NEGATIVE-but-regular shape (-0.5 < xi < 0) still uses EVT (guard is not over-eager).

    A Beta(2,5) loss tail fits xi ~= -0.25 (inside the regular region), so the guard must NOT
    fire and the EVT estimate must differ from the empirical one — proving the xi <= -0.5 branch
    is a tight boundary, not a blanket negative-shape fallback.
    """
    rng = np.random.default_rng(0)
    returns = -rng.beta(2.0, 5.0, size=5000)
    rd = ReturnDistribution(threshold_q=0.10).fit(returns)
    assert -0.5 < rd.xi < 0.0
    assert rd._evt_falls_back(0.05) is None
    assert rd.cvar(0.05, method="evt") != rd.cvar(0.05, method="empirical")


def test_evt_nonregular_xi_note_documents_both_guards() -> None:
    """The disclosure note names the regular region and both fallback boundaries (auditability)."""
    note = EVT_NONREGULAR_XI_NOTE.lower()
    assert "-0.5 < xi < 1" in note
    assert "xi <= -0.5" in note
    assert "xi >= 1" in note
    assert "empirical" in note


# --------------------------------------------------------------------------- #
# T2.8(b) — the FED-headline (CVaR-5%) cross-candidate estimator-consistency log
# --------------------------------------------------------------------------- #


def test_fed_estimator_log_fires_on_cross_candidate_switch(caplog) -> None:
    """The FED CVaR-5% estimator-consistency check LOGS when candidates route differently (T2.8b).

    Candidate A (normal losses) routes EVT; candidate B (uniform-bounded losses, xi<=-0.5) falls
    back to empirical. Evaluating the FED level cvar(0.05) on both must (a) record both paths in the
    registry and (b) emit a WARNING naming the EVT<->empirical switch — making the otherwise SILENT
    per-candidate switch explicit, without changing either returned value.
    """
    reset_fed_estimator_log()
    rng = np.random.default_rng(1)
    rd_a = ReturnDistribution(threshold_q=0.10).fit(0.01 * rng.standard_normal(5000))
    rd_b = ReturnDistribution(threshold_q=0.10).fit(_uniform_loss_returns(seed=2))
    # Sanity: the two candidates really do route to different estimators at the FED level.
    assert rd_a._evt_falls_back(0.05) is None  # EVT
    assert rd_b._evt_falls_back(0.05) == "xi_le_-0.5"  # empirical fallback

    with caplog.at_level("WARNING", logger="src.feedback.measurement"):
        c_a = rd_a.cvar(0.05)  # auto routing -> FED level -> records "evt"
        c_b = rd_b.cvar(0.05)  # auto routing -> FED level -> records "empirical" + WARNS

    # The switch is recorded in first-seen order ...
    assert fed_estimator_log()[0.05] == ["evt", "empirical"]
    # ... and a single inconsistency WARNING fired, naming the fed level + the switch.
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("INCONSISTENT" in m and "CVaR-5%" in m for m in msgs)
    # The logging did NOT alter the returned CVaR values (no science change).
    assert c_a == pytest.approx(rd_a.cvar(0.05, method="evt"))
    assert c_b == pytest.approx(rd_b.cvar(0.05, method="empirical"))


def test_fed_estimator_log_silent_when_consistent(caplog) -> None:
    """No warning when every candidate routes the FED CVaR-5% through the SAME estimator (T2.8b)."""
    reset_fed_estimator_log()
    rng = np.random.default_rng(3)
    with caplog.at_level("WARNING", logger="src.feedback.measurement"):
        for _ in range(3):
            ReturnDistribution(threshold_q=0.10).fit(0.01 * rng.standard_normal(5000)).cvar(0.05)
    # All three used EVT -> a single distinct path, no inconsistency warning.
    assert set(fed_estimator_log()[0.05]) == {"evt"}
    assert not [r for r in caplog.records if "INCONSISTENT" in r.getMessage()]


def test_fed_estimator_log_only_tracks_fed_levels() -> None:
    """Only the FED headline level (CVaR-5%) is tracked; non-fed levels (e.g. 1%) are not recorded."""
    reset_fed_estimator_log()
    assert FED_HEADLINE_CVAR_LEVELS == (0.05,)
    rng = np.random.default_rng(4)
    rd = ReturnDistribution(threshold_q=0.10).fit(0.01 * rng.standard_normal(5000))
    rd.cvar(0.01)  # NOT a fed headline level -> must not be recorded
    rd.cvar(0.25)  # empirical, non-fed -> not recorded
    assert 0.01 not in fed_estimator_log()
    assert 0.25 not in fed_estimator_log()
    rd.cvar(0.05)  # fed level -> recorded
    assert fed_estimator_log()[0.05] == ["evt"]
    # method="evt"/"empirical" (forced) bypass the auto fed-tracking (only "auto" is the fed path).
    reset_fed_estimator_log()
    rd.cvar(0.05, method="evt")
    assert fed_estimator_log() == {}


# --------------------------------------------------------------------------- #
# ADDED: CVaR COHERENCE AXIOMS on ReturnDistribution.cvar() (both methods)
#
# These prove the *headline* estimator (ReturnDistribution.cvar, the object actually fed to
# the LLM) realises the coherent-risk axioms the Chapter-3 theory assumes — distinct from the
# axioms already proved on src.inference.bootstrap.cvar in tests/test_properties.py. We test
# BOTH method="empirical" and method="evt" wherever the level is in-region for the EVT branch.
#
# Subtlety the tests respect: the EVT closed form is invariant under a SHIFT/SCALE of returns
# only up to the re-FIT it induces (u, beta move with the data, xi is shift/scale invariant for
# a GPD), so equivariance is checked on the re-fitted estimator at the SAME alpha — exactly how
# a caller uses it. atol is set explicitly (never the brittle default atol=0) for near-zero CVaR.
# --------------------------------------------------------------------------- #

#: Tail levels that route through the EVT/GPD branch (alpha <= EVT_ALPHA_CUTOFF) and the body
#: levels that route empirical; the axiom tests sweep both estimators where each is defined.
_EVT_LEVELS = (0.05, 0.01)
_EMP_LEVELS = (0.25, 0.10)
#: A return scale (~0.5% daily) and a shift kept SMALL so the closed-form re-fit stays numerically
#: clean; atol absorbs the few-bp float noise of two independent GPD MLE fits.
_CVAR_ATOL = 1e-4


def _seeded_returns(seed: int = 20240627, n: int = 40_000, scale: float = 0.01) -> np.ndarray:
    """A seeded heavy-tailed return sample (Student-t, df=4) for the fixed-seed axiom checks.

    Used by the non-Hypothesis fallbacks so the coherence laws are exercised even in a minimal
    env. Heavy enough that the EVT branch fits a regular xi in (-0.5, 1) at alpha in {5%, 1%}.
    """
    return scale * np.random.default_rng(seed).standard_t(4, size=n)


def _assert_translation_equivariance(r: np.ndarray, alpha: float, c: float, method: str) -> None:
    """Coherence axiom: cvar(r + c) == cvar(r) + c (a deterministic shift moves the tail mean by c).

    Re-fits on the shifted sample (the EVT threshold/scale move with the data) and compares at the
    SAME alpha — the way ``measurement`` is actually called.
    """
    base = ReturnDistribution().fit(r).cvar(alpha, method=method)
    shifted = ReturnDistribution().fit(r + c).cvar(alpha, method=method)
    np.testing.assert_allclose(shifted, base + c, rtol=1e-6, atol=_CVAR_ATOL)


def _assert_positive_homogeneity(r: np.ndarray, alpha: float, k: float, method: str) -> None:
    """Coherence axiom: cvar(k * r) == k * cvar(r) for k > 0 (positive rescaling of returns)."""
    assert k > 0.0
    base = ReturnDistribution().fit(r).cvar(alpha, method=method)
    scaled = ReturnDistribution().fit(k * r).cvar(alpha, method=method)
    np.testing.assert_allclose(scaled, k * base, rtol=1e-6, atol=_CVAR_ATOL)


def _assert_monotonicity_under_dominance(
    r: np.ndarray, alpha: float, d: np.ndarray, method: str
) -> None:
    """Coherence axiom (monotonicity): r + d (d >= 0 elementwise) dominates r, so its signed CVaR
    is NOT LOWER — a uniformly better return stream cannot have a worse (more negative) tail mean."""
    dom = np.abs(d)
    base = ReturnDistribution().fit(r).cvar(alpha, method=method)
    better = ReturnDistribution().fit(r + dom).cvar(alpha, method=method)
    assert better >= base - _CVAR_ATOL


def _assert_level_monotonicity(r: np.ndarray, method: str) -> None:
    """Level-monotonicity: a SMALLER alpha probes deeper into the loss tail, so signed CVaR is
    non-increasing (more negative / deeper) as alpha shrinks — within a single estimator branch."""
    levels = _EVT_LEVELS if method == "evt" else _EMP_LEVELS
    rd = ReturnDistribution().fit(r)
    vals = [rd.cvar(a, method=method) for a in sorted(levels)]  # ascending alpha
    # ascending alpha -> non-decreasing (shallower tail is >= deeper tail).
    for shallower, deeper in zip(vals[1:], vals[:-1]):
        assert deeper <= shallower + _CVAR_ATOL


class TestReturnDistributionCVaRCoherence:
    """Coherent-risk axioms on ReturnDistribution.cvar() itself (translation/homogeneity/monotone)."""

    @pytest.mark.parametrize("method", ["empirical", "evt"])
    def test_translation_equivariance_fixed_seed(self, method: str) -> None:
        """cvar(r + c) == cvar(r) + c on a seeded heavy-tailed sample (both estimator branches)."""
        r = _seeded_returns()
        for alpha in (_EVT_LEVELS if method == "evt" else _EMP_LEVELS):
            _assert_translation_equivariance(r, alpha, c=0.003, method=method)
            _assert_translation_equivariance(r, alpha, c=-0.004, method=method)

    @pytest.mark.parametrize("method", ["empirical", "evt"])
    def test_positive_homogeneity_fixed_seed(self, method: str) -> None:
        """cvar(k * r) == k * cvar(r), k > 0, on a seeded sample (both estimator branches)."""
        r = _seeded_returns()
        for alpha in (_EVT_LEVELS if method == "evt" else _EMP_LEVELS):
            for k in (0.5, 2.0, 3.3):
                _assert_positive_homogeneity(r, alpha, k=k, method=method)

    @pytest.mark.parametrize("method", ["empirical", "evt"])
    def test_monotonicity_under_dominance_fixed_seed(self, method: str) -> None:
        """Adding d >= 0 elementwise cannot lower the signed CVaR (both estimator branches)."""
        r = _seeded_returns()
        d = 0.002 * np.abs(np.random.default_rng(99).standard_normal(r.size))
        for alpha in (_EVT_LEVELS if method == "evt" else _EMP_LEVELS):
            _assert_monotonicity_under_dominance(r, alpha, d=d, method=method)

    @pytest.mark.parametrize("method", ["empirical", "evt"])
    def test_level_monotonicity_fixed_seed(self, method: str) -> None:
        """Smaller alpha -> deeper (more negative) CVaR, within each estimator branch."""
        _assert_level_monotonicity(_seeded_returns(), method=method)

    # --- Hypothesis-driven variants (derandomized via the 'deterministic' profile) --------- #

    @_needs_hypothesis
    @settings(max_examples=40, deadline=None)
    @given(
        seed=st.integers(0, 2**31 - 1),
        alpha=st.sampled_from(_EMP_LEVELS),
        c=st.floats(-5e-3, 5e-3, allow_nan=False, allow_infinity=False),
    )
    def test_empirical_translation_equivariance_property(
        self, seed: int, alpha: float, c: float
    ) -> None:
        """PROPERTY (coherence: translation equivariance) on the EMPIRICAL branch over many samples."""
        _assert_translation_equivariance(_seeded_returns(seed=seed, n=4_000), alpha, c, "empirical")

    @_needs_hypothesis
    @settings(max_examples=40, deadline=None)
    @given(
        seed=st.integers(0, 2**31 - 1),
        alpha=st.sampled_from(_EMP_LEVELS),
        k=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
    )
    def test_empirical_positive_homogeneity_property(
        self, seed: int, alpha: float, k: float
    ) -> None:
        """PROPERTY (coherence: positive homogeneity, k > 0) on the EMPIRICAL branch over many samples."""
        _assert_positive_homogeneity(_seeded_returns(seed=seed, n=4_000), alpha, k, "empirical")

    @_needs_hypothesis
    @settings(max_examples=40, deadline=None)
    @given(
        seed=st.integers(0, 2**31 - 1),
        delta=arrays(np.float64, 4_000, elements=st.floats(0.0, 1e-2, allow_nan=False)),
    )
    def test_empirical_monotonicity_under_dominance_property(
        self, seed: int, delta: np.ndarray
    ) -> None:
        """PROPERTY (coherence: monotonicity) on the EMPIRICAL branch: r + |delta| dominates r."""
        r = _seeded_returns(seed=seed, n=4_000)
        for alpha in _EMP_LEVELS:
            _assert_monotonicity_under_dominance(r, alpha, delta, "empirical")

    @_needs_hypothesis
    @settings(max_examples=40, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), method=st.sampled_from(["empirical", "evt"]))
    def test_level_monotonicity_property(self, seed: int, method: str) -> None:
        """PROPERTY (level-monotonicity): smaller alpha -> deeper signed CVaR, both branches."""
        _assert_level_monotonicity(_seeded_returns(seed=seed, n=8_000), method=method)


# --------------------------------------------------------------------------- #
# ADDED: EVT BOUNDARY / xi-guard — ALL regimes of _evt_falls_back
#
# The xi <= -0.5 fallback and an in-region control already exist above (T2.8a). This block
# completes the regime coverage: xi >= 1 (infinite-mean), alpha > exceed_frac (level shallower
# than the fitted tail mass), degenerate_fit (no finite beta/u / no exceedances), and an EXPLICIT
# in-region assertion that the EVT closed form is used. In every regime the returned CVaR must be
# FINITE (no NaN/inf from a degenerate closed form leaking to the LLM).
# --------------------------------------------------------------------------- #


def _heavy_gpd_loss_returns(xi_true: float, n: int = 200_000, seed: int = 3) -> np.ndarray:
    """Returns whose LOSSES have a GPD upper tail of shape ``xi_true`` above u=2.0 (tail frac 0.10).

    With ``xi_true`` >= ~1.3 the GPD MLE on the exceedances lands at xi >= 1 (the infinite-mean
    region the ``xi_ge_1`` guard is for).
    """
    rng = np.random.default_rng(seed)
    u, fu = 2.0, 0.10
    n_tail = int(n * fu)
    body = stats.truncnorm.rvs(-np.inf, u, size=n - n_tail, random_state=rng)
    tail = u + stats.genpareto.rvs(xi_true, loc=0.0, scale=0.5, size=n_tail, random_state=rng)
    return -np.concatenate([body, tail])


def test_evt_falls_back_when_xi_ge_1_infinite_mean() -> None:
    """xi >= 1 (infinite-mean tail) -> _evt_falls_back == 'xi_ge_1' and EVT returns the empirical CVaR.

    Mirror of the xi <= -0.5 guard: the CVaR closed form (VaR + beta - xi*u)/(1 - xi) is undefined /
    sign-flipped for xi >= 1, so the estimator MUST refuse the closed form and return empirical exactly.
    """
    rd = ReturnDistribution(threshold_q=0.10).fit(_heavy_gpd_loss_returns(1.5))
    assert rd.xi >= 1.0
    assert rd._evt_falls_back(0.01) == "xi_ge_1"
    evt = rd.cvar(0.01, method="evt")
    emp = rd.cvar(0.01, method="empirical")
    np.testing.assert_allclose(evt, emp, rtol=0.0, atol=0.0)
    assert math.isfinite(evt) and evt < 0.0


def test_evt_falls_back_when_alpha_gt_exceed_frac() -> None:
    """alpha > exceed_frac (level shallower than the fitted tail mass) -> 'alpha_gt_exceed_frac'.

    POT is only valid for alpha within F_u; a low threshold_q (0.02) gives exceed_frac=0.02, so
    cvar(0.05) is OUTSIDE the tail and must route to empirical, finite.
    """
    rng = np.random.default_rng(5)
    rd = ReturnDistribution(threshold_q=0.02).fit(0.01 * rng.standard_normal(5_000))
    assert rd.exceed_frac < 0.05
    assert rd._evt_falls_back(0.05) == "alpha_gt_exceed_frac"
    evt = rd.cvar(0.05, method="evt")
    np.testing.assert_allclose(evt, rd.cvar(0.05, method="empirical"), rtol=0.0, atol=0.0)
    assert math.isfinite(evt) and evt < 0.0


def test_evt_falls_back_on_degenerate_fit() -> None:
    """A sample with < 2 exceedances (no finite beta / exceed_frac == 0) -> 'degenerate_fit'.

    The estimator must not emit NaN to the LLM: it falls back to the empirical worst-k mean.
    """
    rd = ReturnDistribution(threshold_q=0.10).fit(np.array([-0.01, -0.01, -0.01, 0.0, 0.01]))
    assert not math.isfinite(rd.beta) or rd.exceed_frac <= 0.0
    assert rd._evt_falls_back(0.05) == "degenerate_fit"
    evt = rd.cvar(0.05, method="evt")
    np.testing.assert_allclose(evt, rd.cvar(0.05, method="empirical"), rtol=0.0, atol=0.0)
    assert math.isfinite(evt)


def test_evt_used_and_finite_when_xi_strictly_in_regular_region(
    heavy_tail_returns: np.ndarray,
) -> None:
    """In-region (-0.5 < xi < 1): _evt_falls_back is None, the EVT CLOSED FORM is used, result finite.

    Asserts the positive case of the guard (not merely the fallbacks) and that the closed form differs
    from empirical (so the EVT branch genuinely fired, not a silent fallback).
    """
    rd = ReturnDistribution().fit(heavy_tail_returns)
    assert -0.5 < rd.xi < 1.0
    for alpha in _EVT_LEVELS:
        assert rd._evt_falls_back(alpha) is None
        evt = rd.cvar(alpha, method="evt")
        assert math.isfinite(evt) and evt < 0.0
        assert evt != rd.cvar(alpha, method="empirical")


# --------------------------------------------------------------------------- #
# ADDED: FIT DETERMINISM — byte-identical refit + independence of global RNG state
# --------------------------------------------------------------------------- #


def test_fit_is_byte_identical_on_identical_returns(heavy_tail_returns: np.ndarray) -> None:
    """Fitting twice on the SAME returns yields byte-identical (xi, beta, u, exceed_frac) and CVaR.

    The estimator is a pure function of its input sample (no hidden RNG) — required for the
    replay-from-archive determinism contract.
    """
    a = ReturnDistribution().fit(heavy_tail_returns)
    b = ReturnDistribution().fit(heavy_tail_returns)
    for attr in ("xi", "beta", "u", "exceed_frac", "T"):
        assert getattr(a, attr) == getattr(b, attr)
    for alpha, method in [(0.01, "evt"), (0.05, "evt"), (0.10, "empirical"), (0.25, "empirical")]:
        assert a.cvar(alpha, method=method) == b.cvar(alpha, method=method)
    assert a.tail_stats() == b.tail_stats()


def test_fit_independent_of_global_rng_state(heavy_tail_returns: np.ndarray) -> None:
    """fit() does NOT consume / depend on the global NumPy RNG: churning np.random in between
    leaves (xi, beta, u) and every CVaR byte-identical (no leakage of process-global randomness)."""
    np.random.seed(0)
    a = ReturnDistribution().fit(heavy_tail_returns)
    a_stats = a.tail_stats()
    # Perturb the GLOBAL RNG between fits in several ways.
    np.random.seed(123456)
    _ = np.random.standard_normal(10_000)
    _ = np.random.random()
    b = ReturnDistribution().fit(heavy_tail_returns)
    for attr in ("xi", "beta", "u", "exceed_frac"):
        assert getattr(a, attr) == getattr(b, attr)
    assert a.cvar(0.05, method="evt") == b.cvar(0.05, method="evt")
    assert a_stats == b.tail_stats()


# --------------------------------------------------------------------------- #
# ADDED: SIMPLE-vs-LOG return boundary (the module fits SIMPLE/arithmetic returns)
# --------------------------------------------------------------------------- #


def test_cvar_simple_vs_log_agree_at_small_returns_diverge_at_large() -> None:
    """At small returns (|r| ~ <=1% daily) CVaR on simple r vs log1p(r) agree to <1%; at +-50%
    they diverge materially (documents the simple-vs-log choice the module makes — it fits SIMPLE,
    arithmetic, per-step returns; see the module docstring).

    Rationale: log1p(r) ~= r - r^2/2 + ..., so the O(r^2) gap is sub-percent at the ~0.5% daily
    scale that dominates portfolio returns, but is order-of-magnitude at +-50%.
    """
    rng = np.random.default_rng(42)
    # Small (~0.5% daily) returns: simple and log CVaR agree to <1% at every reported level.
    r_small = 0.005 * rng.standard_normal(200_000)
    rd_simple = ReturnDistribution().fit(r_small)
    rd_log = ReturnDistribution().fit(np.log1p(r_small))
    for alpha in (0.05, 0.10, 0.25):
        cs = rd_simple.cvar(alpha)
        cl = rd_log.cvar(alpha)
        assert abs(cs - cl) / abs(cs) < 0.01, f"alpha={alpha} simple-vs-log gap >=1%"

    # Large (~50%) returns: the simple-vs-log gap is LARGE (documented divergence, not a bug).
    r_big = 0.50 * rng.standard_normal(200_000)
    r_big = r_big[r_big > -0.99]  # log1p defined only for r > -1
    cs_big = ReturnDistribution().fit(r_big).cvar(0.05)
    cl_big = ReturnDistribution().fit(np.log1p(r_big)).cvar(0.05)
    assert abs(cs_big - cl_big) / abs(cs_big) > 0.20  # divergence at +-50%


# --------------------------------------------------------------------------- #
# ADDED: tail_stats() — left_tail_mass bound (frozen keys + robust_skew sign already covered)
# --------------------------------------------------------------------------- #


def test_left_tail_mass_is_a_probability_in_unit_interval(
    normal_returns: np.ndarray, heavy_tail_returns: np.ndarray
) -> None:
    """left_tail_mass = P(return < -k*std) is a probability in [0, 1] (B.10).

    Complements the existing frozen-keys / robust_skew-sign tests by pinning the documented range
    of the fed left_tail_mass scalar on both a light- and a heavy-tailed sample.
    """
    for sample in (normal_returns, heavy_tail_returns):
        ltm = ReturnDistribution().fit(sample).tail_stats()["left_tail_mass"]
        assert 0.0 <= ltm <= 1.0
        assert math.isfinite(ltm)
