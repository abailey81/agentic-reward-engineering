"""A16 — validity-tier node N2 reads the REGISTERED non-inferiority rule, plus its sensitivities.

The defect these tests pin
--------------------------
``config/preregistration.yaml`` registers N2 as ``{test: h2_ra_iut_or_tost, equivalence:
tost_0.05_dsr}`` and its own dated note says "activation rests entirely on N2 rejecting via TOST — a
real pre-registered alpha source". ``validity_tier.NODE_SOURCES`` read ``pvalue_one_sided`` — the
SUPERIORITY leg only — so the registered disjunction had no implementation. Under the design's own
predicted branch that left ALL SIX confirmatory nodes unreachable.

``{theta > 0} UNION {-d < theta < d}`` IS ``{theta > -d}``: the registered disjunction is one
one-sided non-inferiority hypothesis at the SESOI. The repair is conformance to a frozen design.

Every test below fails against the pre-fix code: pre-fix there is no ``pvalue_non_inferiority`` key
anywhere, so N2 reports untestable and cannot reject.

Pre-specified 2026-08-01T13:01:15Z (HEAD 57c5ecc4) with 0 of 3 H2-RA legs computable — i.e. while no
H2 outcome existed — and recorded before any of this was written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_campaign as ac  # noqa: E402
from src.inference.validity_tier import N2_KEYS, NODE_SOURCES, tier_node_pvalues, tier_verdict  # noqa: E402


def _out(legs: list[dict[str, float]], tail_p: float = 0.9) -> dict:
    """An analyze_campaign-shaped result with the given RA legs and a non-rejecting tail family."""
    return {
        "h2": {
            "tail_legs": [{"contrast": f"t{i}", "pvalue_one_sided": tail_p} for i in range(3)],
            "legs": [{"contrast": f"r{i}", **leg} for i, leg in enumerate(legs)],
        },
    }


# --------------------------------------------------------------------------- the falsifier
def test_mildly_inferior_legs_reject_under_ni_and_not_under_superiority() -> None:
    """THE test. All three legs mildly inferior — inside the SESOI but on the wrong side of zero.

    The registered rule must CERTIFY this (that is what "equivalence" means); the superiority-only
    rule must not. If both agreed here, the whole A16 repair would be a no-op and this test would be
    verifying nothing.
    """
    legs = [{"pvalue_one_sided": 0.80,                       # nowhere near superior
             "pvalue_non_inferiority": 0.004,                # comfortably not-worse-by-more-than-d
             "pvalue_non_inferiority_conservative": 0.010} for _ in range(3)]
    out = _out(legs)

    primary = tier_verdict(out)
    assert primary["n2_rule"] == "pvalue_non_inferiority"
    assert "N2_h2_ra" in primary["rejected"], "the registered rule must certify non-inferiority"

    superiority = tier_verdict(out, n2_key=N2_KEYS["sensitivity_superiority_only"],
                               with_sensitivities=False)
    assert "N2_h2_ra" not in superiority["rejected"], (
        "positive control: the as-implemented-until-now rule must NOT reject here — otherwise this "
        "test could not distinguish the two rules and would prove nothing")


def test_the_default_is_the_registered_rule_not_a_sensitivity() -> None:
    assert NODE_SOURCES["N2_h2_ra"]["key"] == N2_KEYS["primary"] == "pvalue_non_inferiority"


def test_all_three_pre_specified_verdicts_are_reported_unconditionally() -> None:
    """A sensitivity you have to ask for is one you can decline to ask for after seeing the primary."""
    out = _out([{"pvalue_one_sided": 0.80, "pvalue_non_inferiority": 0.004,
                 "pvalue_non_inferiority_conservative": 0.010} for _ in range(3)])
    res = tier_verdict(out)
    assert set(res["sensitivities"]) == {"sensitivity_conservative_margin",
                                         "sensitivity_superiority_only"}
    assert res["sensitivities"]["sensitivity_conservative_margin"]["nodes"]["N2_h2_ra"]["pvalue"] \
        == pytest.approx(0.010)
    assert res["sensitivities"]["sensitivity_superiority_only"]["nodes"]["N2_h2_ra"]["pvalue"] \
        == pytest.approx(0.80)
    assert "never 'superior or equivalent'" in res["sensitivity_note"]


def test_n2_is_an_iut_over_the_legs_so_one_bad_leg_shuts_it() -> None:
    """Berger 1982: the node p is the MAX over legs. One inferior leg must sink the whole node."""
    legs = [{"pvalue_one_sided": 0.8, "pvalue_non_inferiority": 0.004,
             "pvalue_non_inferiority_conservative": 0.01} for _ in range(2)]
    legs.append({"pvalue_one_sided": 0.9, "pvalue_non_inferiority": 0.7,
                 "pvalue_non_inferiority_conservative": 0.9})
    res = tier_verdict(_out(legs))
    assert res["nodes"]["N2_h2_ra"]["pvalue"] == pytest.approx(0.7)
    assert "N2_h2_ra" not in res["rejected"]


def test_a_leg_missing_the_key_makes_the_node_untestable_never_a_pass() -> None:
    """Absent is not zero and not one: it is UNTESTABLE, and an IUT cannot certify from it."""
    legs = [{"pvalue_one_sided": 0.8, "pvalue_non_inferiority": 0.004,
             "pvalue_non_inferiority_conservative": 0.01} for _ in range(2)]
    legs.append({"pvalue_one_sided": 0.8})  # the pre-fix shape
    nodes = tier_node_pvalues(_out(legs))
    assert nodes["N2_h2_ra"]["pvalue"] is None
    assert "cannot certify" in nodes["N2_h2_ra"]["reason"]


def test_an_unregistered_n2_rule_raises_rather_than_falling_back() -> None:
    out = _out([{"pvalue_one_sided": 0.8, "pvalue_non_inferiority": 0.004,
                 "pvalue_non_inferiority_conservative": 0.01} for _ in range(3)])
    with pytest.raises(ValueError, match="not one of the pre-specified N2 rules"):
        tier_node_pvalues(out, n2_key="pvalue_two_sided")


# --------------------------------------------------------------------------- the margins
def test_margins_are_derived_from_the_registered_conversion_not_literals() -> None:
    m = ac._ni_margins()
    pa = ac._power_analysis()
    assert m["sesoi_dsr"] == pytest.approx(ac._frozen_equiv_margin())
    assert m["primary_track_length"] == pa.VALIDATION_TRACK_LENGTH == 694
    assert m["conservative_track_length"] == pa.TEST_TRACK_LENGTH == 1571
    # The two numbers the three lanes argued over, reproduced from the code that defines them.
    assert m["primary"] == pytest.approx(0.075578, abs=1e-6)
    assert m["conservative"] == pytest.approx(0.050212, abs=1e-6)
    assert m["primary_sharpe_to_dsr_factor"] == pytest.approx(0.661571, abs=1e-6)
    assert m["conservative_sharpe_to_dsr_factor"] == pytest.approx(0.995771, abs=1e-6)


def test_executed_margin_equals_the_hash_bound_sesoi_ann_sharpe_equiv() -> None:
    """COORD's requested guard (M174), adopted unconditionally and endorsed by ANALYSIS (M176 (3)).

    The executed margin must equal the frozen ``inference.sesoi_derivation.sesoi_ann_sharpe_equiv``
    to 4 dp. Two reasons this is worth more than the literal it replaces:

      * it FAILS against the patch line originally circulated for this fix, which passed
        ``_frozen_equiv_margin()`` — 0.05 in **validation-DSR** units, per its own docstring —
        straight into per-seed **annualised-Sharpe** data. Coord measured the consequence on
        synthetic legs in the disputed band: p(N2) = 0.0065 (REJECTS) versus 0.5515 (does not). A
        unit error in a confirmatory node, and the accompanying prose said the right thing while
        the code line contradicted it;
      * it fires loudly if anyone ever "corrects" the track length to the TEST 1571, which would
        move a REPORTED equivalence bound by 50 % — ``h2_tost_dsr`` ships as report-only and the
        bankable-null statement rests on it.
    """
    from src.utils.config import load_config

    frozen = load_config("preregistration")["inference"]["sesoi_derivation"]
    assert ac._ni_margins()["primary"] == pytest.approx(
        float(frozen["sesoi_ann_sharpe_equiv"]), abs=5e-5), (
        "the executed non-inferiority margin has drifted from the hash-bound "
        "sesoi_ann_sharpe_equiv — check the track length before checking anything else")
    assert ac._ni_margins()["primary_sharpe_to_dsr_factor"] == pytest.approx(
        float(frozen["dsr_per_ann_sharpe"]), abs=5e-5)


def test_margin_sits_inside_the_hash_bound_r104_economic_band() -> None:
    """R104 binds 0.0055 < SESOI < 0.10 IN ANNUALISED SHARPE, verdict ``sesoi_inside_band``.

    A fourth, independent corroboration that the registered margin is expressed in Sharpe units and
    equals 0.0756: were the operative margin 0.0502, the frozen block would no longer describe the
    executed number and a referee finds that with one grep.
    """
    d = ac._ni_margins()["primary"]
    assert 0.0055 < d < 0.10, f"{d} falls outside the hash-bound R104 economic band"


def test_the_registered_margin_is_the_permissive_one_and_that_is_stated() -> None:
    """Pinned deliberately: the direction that cuts against the analyst must not drift silently."""
    m = ac._ni_margins()
    assert m["primary"] > m["conservative"], (
        "the registered margin is the WIDER one, hence the EASIER to reject — if this ever flips, "
        "the framing in _ni_margins and in the A16 record must be re-derived, not assumed")


# --------------------------------------------------------------------------- the identity
def test_ni_p_is_the_superiority_test_with_the_null_shifted() -> None:
    """IQM is translation-equivariant, so the NI test at -d IS the superiority test at 0 on shifted
    data. Verified against the real bootstrap, not asserted: shifting arm A by +d must give exactly
    the p the NI leg reports, and it must be <= the unshifted superiority p."""
    from src.inference.bootstrap import iqm, paired_seed_difference_test

    rng_seed = 12345
    a = np.linspace(-0.2, 0.4, 30)
    b = a + 0.03  # arm B mildly better -> not superior, but plausibly non-inferior
    d = ac._ni_margins()["primary"]

    sup = paired_seed_difference_test(a, b, statistic=iqm, n_boot=500,
                                      rng=np.random.default_rng(rng_seed))
    ni = paired_seed_difference_test(a + d, b, statistic=iqm, n_boot=500,
                                     rng=np.random.default_rng(rng_seed))
    assert ni["effect"] == pytest.approx(sup["effect"] + d, abs=1e-12), \
        "translation-equivariance: the estimate must shift by exactly d"
    assert ni["pvalue_one_sided_greater"] <= sup["pvalue_one_sided_greater"] + 1e-12, \
        "moving the null left can only make rejection easier; a violation means the shift is wrong"


def test_the_tail_family_did_not_acquire_a_non_inferiority_route() -> None:
    """N1 must stay a pure superiority IUT. The A16 repair is scoped to the RA legs alone."""
    assert NODE_SOURCES["N1_h2_tail"]["key"] == "pvalue_one_sided"
    out = _out([{"pvalue_one_sided": 0.8, "pvalue_non_inferiority": 0.001,
                 "pvalue_non_inferiority_conservative": 0.001} for _ in range(3)])
    res = tier_verdict(out)
    assert res["nodes"]["N1_h2_tail"]["pvalue"] == pytest.approx(0.9)
    assert "N1_h2_tail" not in res["rejected"]
