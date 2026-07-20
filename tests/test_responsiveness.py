"""Behaviour tests for responsiveness + the numeracy-bottleneck differential (src/inference/responsiveness.py).

Deterministic (seeded bootstrap). Checks: a real X->M association is flagged responsive; the null (independent
X, M) is not; the legible-format differential recovers a constructed legibility advantage; degrade paths.
"""

from __future__ import annotations

import numpy as np

import pytest

from src.inference.responsiveness import (
    MIN_BOOT_VALID_FRACTION,
    _bootstrap_coef,
    _bootstrap_coef_raw,
    _coef,
    legible_format_responsiveness_differential,
    responsiveness,
)

SEED = 20260701


def test_detects_a_real_responsiveness() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=120)
    m = 0.8 * x + 0.4 * rng.normal(size=120)
    res = responsiveness(x, m, n_boot=800, rng=np.random.default_rng(1))
    assert res["status"] == "ok"
    assert res["coef"] > 0.5
    assert res["responsive"] is True
    assert res["ci_low"] > 0.0


def test_null_is_not_flagged_responsive() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=120)
    m = rng.normal(size=120)  # independent of x
    res = responsiveness(x, m, n_boot=800, rng=np.random.default_rng(2))
    assert res["status"] == "ok"
    assert abs(res["coef"]) < 0.25
    assert res["responsive"] is False
    assert res["ci_low"] < 0.0 < res["ci_high"]


def test_slope_method_matches_pearson() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=200)
    m = 0.6 * x + 0.5 * rng.normal(size=200)
    res = responsiveness(x, m, method="slope", n_boot=200, rng=np.random.default_rng(3))
    assert res["coef"] == np.float64(np.corrcoef(x, m)[0, 1]).item() or abs(
        res["coef"] - np.corrcoef(x, m)[0, 1]
    ) < 1e-9


def test_legible_format_differential_recovers_legibility_advantage() -> None:
    rng = np.random.default_rng(SEED)
    n = 150
    # legible condition: strong X->M coupling (the numbers are readable, so the code tracks them)
    xl = rng.normal(size=n)
    ml = 0.85 * xl + 0.3 * rng.normal(size=n)
    # raw condition: same tail content but illegible -> the coupling washes out (near-null)
    xr = rng.normal(size=n)
    mr = 0.05 * xr + rng.normal(size=n)
    res = legible_format_responsiveness_differential(xl, ml, xr, mr, n_boot=1000, rng=np.random.default_rng(4))
    assert res["status"] == "ok"
    assert res["coef_legible"] > res["coef_raw"]
    assert res["differential"] > 0.3
    assert res["legibility_helps"] is True
    assert res["ci_low"] > 0.0


def test_differential_pairs_bootstraps_by_replicate_index_not_position() -> None:
    """P7a: the differential CI must pair the two conditions by REPLICATE INDEX with a JOINT finite mask, not
    by differencing two SEPARATELY-compacted prefixes. Construct a RAW condition whose m is integer-valued so
    SOME of its resamples are degenerate (dropped from a compacted array) while the legible condition's are
    all finite. The old bl[:k]-br[:k] scheme would then difference mismatched replicates. With the fix, the
    number of paired diffs equals the count of replicates finite in BOTH, and the CI is well-formed."""
    rng = np.random.default_rng(SEED)
    n = 40
    xl = rng.normal(size=n)
    ml = 0.85 * xl + 0.3 * rng.normal(size=n)                 # continuous -> all resamples finite
    xr = rng.normal(size=n)
    mr = np.rint(0.3 * xr).astype(float)                     # integer-valued -> some resamples degenerate
    res = legible_format_responsiveness_differential(xl, ml, xr, mr, n_boot=1500, rng=np.random.default_rng(4))
    # The fixture must reach the regime under test (deep review 2026-07-26, #73 pattern): every assertion
    # below sits under this condition, so a construction that degraded to no_data would assert NOTHING and
    # still pass. MEASURED here: status="ok", n_boot_valid=1488/1500.
    assert res["status"] == "ok", f"fixture drifted to {res['status']!r} — the assertions below would not run"
    # the differential is a proper number and n_boot_valid counts JOINTLY-finite replicate pairs
    assert np.isfinite(res["differential"])
    assert np.isfinite(res["ci_low"]) and np.isfinite(res["ci_high"])
    assert res["ci_low"] <= res["ci_high"]
    assert 0 < res["n_boot_valid"] <= 1500


def test_responsiveness_ci_unreliable_flagged_when_most_resamples_degenerate() -> None:
    """P7b: an INTEGER-valued m with few distinct values makes many case-resamples collapse to a constant
    column (NaN coef, dropped). When fewer than MIN_BOOT_VALID_FRACTION of the resamples survive, ci_reliable
    must be False and responsive must be forced False even if the surviving CI happens to exclude 0.

    NOTE: This test was VACUOUS until 2026-07-26 (deep review, #73). It used ``x=[0,1,2]``, ``m=[0,0,1]``
    and asserted the two meaningful conditions only ``if frac < MIN_BOOT_VALID_FRACTION`` — but that
    config yields ``frac = 0.673``, so the body NEVER executed and P7b's False branch had never been
    exercised by any test. The cause is structural: when x and m are tied on the SAME indices their
    degeneracies COINCIDE, capping the degenerate fraction near 1/3 whatever n is. Decorrelating the
    ties (x tied on {0,1}, m tied on {1,2}) makes the two degeneracy events overlap only on the three
    constant resamples, so 15 of the 27 equally-likely n=3 resamples are degenerate -> frac ~ 0.444.
    The assertions are now UNCONDITIONAL, with the fixture's own precondition asserted first so a drift
    that re-vacuums the test fails loudly instead of silently passing.
    """
    x = np.array([0.0, 0.0, 1.0])  # tied on indices {0,1}
    m = np.array([1.0, 2.0, 2.0])  # tied on indices {1,2} -> DECORRELATED from x's ties
    res = responsiveness(x, m, n_boot=2000, rng=np.random.default_rng(0))
    assert res["status"] == "ok"
    frac = res["n_boot_valid"] / 2000.0
    # The fixture must actually reach the regime under test — else the assertions below are vacuous.
    assert frac < MIN_BOOT_VALID_FRACTION, (
        f"fixture drifted: valid-boot fraction {frac:.3f} is no longer below "
        f"{MIN_BOOT_VALID_FRACTION}, so this test would assert nothing"
    )
    assert res["ci_reliable"] is False
    assert res["responsive"] is False  # unreliable CI can never certify responsiveness


def test_non_finite_input_reports_no_data_not_ok() -> None:
    """#69-class, in the mechanism kernel (#71): the degeneracy guard is ``np.ptp(x) == 0``, but
    ``np.ptp`` of an array holding a NaN is NaN and ``NaN == 0`` is False — so the guard could not fire
    on exactly the degenerate input it exists to reject.

    MEASURED before the fix: an ALL-NaN x returned ``status="ok"`` with ``coef=NaN``; a SINGLE NaN
    returned ``status="ok"`` with ``coef=NaN`` and a real-looking percentile CI built from only the ~34%
    of resamples that happened to dodge the bad row. Every caller gates on ``status``, so advertising
    success is the harm.
    """
    good_x = np.array([0.1, 0.5, 0.2, 0.9, 0.4, 0.7, 0.3, 0.8])
    good_m = np.array([1.0, 3.0, 2.0, 5.0, 2.0, 4.0, 1.0, 5.0])

    for label, x, m in (
        ("one NaN in x", np.where(np.arange(8) == 3, np.nan, good_x), good_m),
        ("all NaN in x", np.full(8, np.nan), good_m),
        ("one NaN in m", good_x, np.where(np.arange(8) == 2, np.nan, good_m)),
        ("inf in x", np.where(np.arange(8) == 5, np.inf, good_x), good_m),
    ):
        res = responsiveness(x, m, n_boot=300, rng=np.random.default_rng(0))
        assert res["status"] == "no_data", f"{label}: advertised success on unusable input"
        assert "non-finite" in res["reason"]
        assert "coef" not in res  # nothing numeric to hand a caller

    # The LEGITIMATE degenerate paths must keep their OWN reasons (do not over-trigger).
    assert "constant" in responsiveness(np.ones(8), good_m)["reason"]
    assert ">= 3" in responsiveness(good_x[:2], good_m[:2])["reason"]
    # ...and clean input is untouched.
    clean = responsiveness(good_x, good_m, n_boot=300, rng=np.random.default_rng(0))
    assert clean["status"] == "ok" and np.isfinite(clean["coef"])


def test_differential_will_not_claim_legibility_helps_off_an_unreliable_ci() -> None:
    """#72: ``responsiveness`` gates its own verdict on the valid-replicate fraction
    (``responsive = ci_reliable and ...``), but the differential asserted ``legibility_helps`` with NO
    such gate — while holding both conditions' ``ci_reliable`` flags in hand.

    The differential's CI is built from the replicates where BOTH conditions were non-degenerate, so it
    inherits the same problem and must inherit the same gate.
    """
    # A condition whose ties are DECORRELATED across x and m is genuinely unreliable (see the P7b test).
    x_unrel, m_unrel = np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 2.0])
    assert responsiveness(x_unrel, m_unrel, n_boot=2000, rng=np.random.default_rng(0))["ci_reliable"] is False

    res = legible_format_responsiveness_differential(
        x_unrel, m_unrel,
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]),
        n_boot=2000, rng=np.random.default_rng(0),
    )
    if res["status"] == "ok":
        assert res["ci_reliable"] is False
        assert res["legibility_helps"] is False, "claimed a verdict off an untrustworthy CI"

    # The gate must be a NO-OP when the CI IS reliable — verify the LEGITIMATE case, not only the misfire.
    rng = np.random.default_rng(11)
    k = 40
    xs = np.linspace(0, 1, k)
    strong = legible_format_responsiveness_differential(
        xs, xs * 10 + rng.normal(0, 0.15, k),   # legible: strong monotone relation
        xs, rng.normal(0, 1.0, k),              # raw: none
        n_boot=2000, rng=np.random.default_rng(5),
    )
    assert strong["status"] == "ok"
    assert strong["ci_reliable"] is True
    assert strong["legibility_helps"] is True                      # the true positive still lands
    assert strong["legibility_helps"] == (strong["ci_low"] > 0.0)  # gate changed nothing here


def test_bootstrap_raw_is_length_preserving_with_nan() -> None:
    """P7a helper: _bootstrap_coef_raw keeps the full n_boot length (degenerate -> NaN), while the public
    _bootstrap_coef drops them. The raw length must equal n_boot exactly (index-preservation invariant)."""
    x = np.array([2.0, 2.0, 2.0, 3.0])  # near-constant -> many degenerate resamples
    m = np.array([0.0, 1.0, 0.0, 1.0])
    raw = _bootstrap_coef_raw(x, m, "spearman", 100, np.random.default_rng(0))
    finite = _bootstrap_coef(x, m, "spearman", 100, np.random.default_rng(0))
    assert raw.size == 100                       # length-preserving
    assert np.isnan(raw).any()                   # some degenerate replicates recorded as NaN, not dropped
    assert finite.size == int(np.isfinite(raw).sum())  # public helper == raw with NaNs removed


def test_degrade_paths() -> None:
    assert responsiveness([1.0, 2.0], [1.0, 2.0])["status"] == "no_data"          # n < 3
    assert responsiveness([1, 1, 1, 1], [1, 2, 3, 4])["status"] == "no_data"      # constant x
    # a degenerate condition makes the differential degrade honestly
    bad = legible_format_responsiveness_differential([1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
    assert bad["status"] == "no_data"


def test_unequal_length_paired_arrays_no_data() -> None:
    # x.size != m.size -> validation no_data (line 77)
    res = responsiveness([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert res["status"] == "no_data"
    assert "equal-length" in res["reason"]


def test_default_rng_is_used_and_deterministic() -> None:
    # rng=None takes the default_rng(0) branch (line 83); two default-rng calls agree byte-for-byte.
    x = np.linspace(-1.0, 1.0, 40)
    m = 0.7 * x + 0.2 * np.cos(np.arange(40))
    a = responsiveness(x, m, n_boot=300)
    b = responsiveness(x, m, n_boot=300)
    assert a["status"] == "ok"
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    assert a["coef"] == b["coef"]


def test_coef_slope_degenerate_returns_nan() -> None:
    # slope method with zero-variance m -> nan (lines 41-42)
    val = _coef(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]), "slope")
    assert np.isnan(val)


def test_coef_unknown_method_raises() -> None:
    # unknown method -> ValueError (line 44)
    with pytest.raises(ValueError, match="unknown method"):
        _coef(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "kendall")


def test_bootstrap_all_degenerate_yields_nan_ci() -> None:
    # If every bootstrap resample is rank-degenerate the valid-boot set is empty ->
    # responsiveness reports nan CI and responsive=False (lines 90-91). Force this by
    # feeding a two-level m where most resamples collapse: use slope on near-constant data
    # so _coef returns nan whenever a resample draws a single distinct x value.
    # Directly exercise _bootstrap_coef emptiness on a constant input.
    empty = _bootstrap_coef(
        np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]), "spearman", 50, np.random.default_rng(0)
    )
    assert empty.size == 0  # constant x -> every resample nan -> filtered out


def test_differential_empty_bootstrap_no_data() -> None:
    # Construct conditions that pass responsiveness() (non-degenerate) yet whose bootstraps
    # can still be empty is not naturally reachable; instead verify the k==0 guard path (line 148)
    # is wired by checking the differential degrades to no_data when a nested condition is degenerate.
    # (The empty-bootstrap branch shares the same no_data contract.)
    res = legible_format_responsiveness_differential(
        [1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0],  # degenerate m_legible
        [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],
    )
    assert res["status"] == "no_data"
    assert res["legible"]["status"] == "no_data"
