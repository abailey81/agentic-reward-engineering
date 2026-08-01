"""H1's beat-the-canon IUT must reach the R31 cross-hypothesis sensitivity (record §100.34).

THE DEFECT THIS PINS. On 2026-07-26 H1 became a confirmatory INTERSECTION-UNION TEST over the 11-name
hand-reward canon, with a real one-sided ``iut_pvalue``. The CALLER of
``cross_hypothesis_multiplicity`` was updated to pass ``h1=out["h1_beat_human"]``; the EXTRACTION
inside it was not. It kept reading the retired ``beats_best_baseline_dsr`` and hardcoded
``headline_p: None`` with the note "descriptive panel, no inferential p — Bonferroni n/a".

WHY THAT MATTERED. ``validity_tier`` gives N6_h1 an initial graph weight of 0.0, so under the design's
own PRE-REGISTERED PREDICTION (H2 null on both co-primaries) H1 is tested at local alpha EXACTLY 0.0
(§100.33). With this row ALSO reporting no p, **H1 was decidable nowhere** — its IUT p-value computed
on every run and consumed by no rule able to act on it.

This is a STALE CONSUMER, not a design change: the R31 rule (headline p vs alpha/n) is untouched; only
WHICH quantity is H1's headline p changes, and that was settled by the registered 2026-07-26 upgrade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_campaign import cross_hypothesis_multiplicity as X  # noqa: E402

ALPHA = 0.05
BONF = ALPHA / 4          # 0.0125


def _h1(p, *, all_present=True, status="ok"):
    return {"status": status, "iut": {
        "iut_pvalue": p, "all_baselines_present": all_present,
        "dominates_canon": bool(p is not None and all_present and p <= ALPHA),
        "n_significantly_beaten": 11, "n_baselines": 11,
    }}


def _row(h1):
    r = X(h1=h1, h2=None, h3=None, h4=None, alpha=ALPHA)
    return next(x for x in r["rows"] if x["hypothesis"] == "H1")


class TestH1ReachesTheSensitivity:
    def test_iut_pvalue_becomes_the_headline_p(self):
        """Before the fix this was hardcoded None on every run."""
        assert _row(_h1(0.0001))["headline_p"] == pytest.approx(0.0001)

    def test_a_strong_H1_survives_the_bonferroni_hurdle(self):
        """p=0.0001 clears alpha/4=0.0125 — the decision that was previously unavailable anywhere."""
        assert _row(_h1(0.0001))["survives_bonferroni"] is True

    def test_a_weak_H1_does_NOT_survive(self):
        """The fix must not simply make H1 pass; the hurdle has to bite."""
        assert _row(_h1(0.02))["survives_bonferroni"] is False

    def test_the_boundary_is_inclusive_at_alpha_over_n(self):
        assert _row(_h1(BONF))["survives_bonferroni"] is True
        assert _row(_h1(BONF + 1e-9))["survives_bonferroni"] is False

    def test_the_note_names_the_actual_test(self):
        note = _row(_h1(0.0001))["note"]
        assert "IUT" in note and "canon" in note
        assert "descriptive panel" not in note, "the retired framing must not survive"


class TestTheGateIsHonoured:
    """``all_baselines_present`` is read EXACTLY as validity_tier's N6_h1 reads it.

    Two consumers of one quantity disagreeing about its gate is how this defect arose in the first
    place, so the agreement is itself the property under test.
    """

    def test_under_seeded_canon_yields_no_claim_even_with_a_tiny_p(self):
        row = _row(_h1(0.0001, all_present=False))
        assert row["headline_p"] is None
        assert row["survives_bonferroni"] is None

    def test_under_seeded_canon_states_the_reason(self):
        assert "all_baselines_present=False" in _row(_h1(0.0001, all_present=False))["note"]

    def test_missing_pvalue_yields_no_claim(self):
        row = _row(_h1(None, status="skipped"))
        assert row["headline_p"] is None and row["survives_bonferroni"] is None

    def test_non_finite_pvalue_yields_no_claim(self):
        assert _row(_h1(float("nan")))["headline_p"] is None


class TestNoCollateralChange:
    def test_the_other_hypotheses_are_untouched_when_absent(self):
        r = X(h1=_h1(0.0001), h2=None, h3=None, h4=None, alpha=ALPHA)
        assert {x["hypothesis"] for x in r["rows"]} == {"H1", "H2", "H3", "H4"}
        for h in ("H2", "H3", "H4"):
            assert next(x for x in r["rows"] if x["hypothesis"] == h)["headline_p"] is None

    def test_bonferroni_level_is_still_alpha_over_n(self):
        r = X(h1=_h1(0.0001), h2=None, h3=None, h4=None, alpha=ALPHA)
        assert r["bonferroni_alpha"] == pytest.approx(BONF)

    def test_a_malformed_h1_block_does_not_raise(self):
        """A shape change upstream must degrade to 'no claim', never crash the sensitivity."""
        for bad in ({}, {"status": "ok"}, {"status": "ok", "iut": {}}, None):
            assert _row(bad)["headline_p"] is None
