"""A node allotted ZERO local alpha must never be reported as a failed test (record §100.33).

THE DEFECT THIS PINS. Under the REGISTERED graph, ``initial_weights`` starts N3/N4/N5/N6 at 0.0, so
if no upstream node rejects they are tested at ``w * alpha == 0.0`` and cannot reject at any
attainable p-value. They were previously returned under ``not_rejected`` — indistinguishable from a
hypothesis that WAS tested and failed. Executed on the design's own PRE-REGISTERED PREDICTION (H2 null
on both co-primaries), that reported an H1 p-value of 0.0001 as "not rejected".

Every test below asserts a property that FAILED before the fix, plus the two that must NOT change:
``rejected`` is inference-identical, and a genuinely-tested non-rejection stays in ``not_rejected``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.multiple_testing import (  # noqa: E402
    graphical_alpha_propagation, registered_alpha_graph,
)

#: The design's own predicted branch: H2 nulls on BOTH co-primaries, with a spectacular H1.
PREDICTED = {"N1_h2_tail": 0.40, "N2_h2_ra": 0.60, "N3_h3": 0.20,
             "N4_h4": 0.02, "N5_structure": 0.30, "N6_h1": 0.0001}

#: The same data, but N2 rejects via the equivalence route the registration says activation rests on.
ACTIVATED = dict(PREDICTED, N2_h2_ra=0.01)


@pytest.fixture()
def graph():
    return registered_alpha_graph()


class TestRegisteredGraphPredictedBranch:
    def test_zero_alpha_nodes_are_structurally_untestable_not_failed(self, graph):
        w, e, alpha = graph
        out = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        assert set(out["structurally_untestable"]) == {"N3_h3", "N4_h4", "N5_structure", "N6_h1"}
        for n in ("N3_h3", "N4_h4", "N5_structure", "N6_h1"):
            assert n not in out["not_rejected"], f"{n} had alpha 0.0 and must not read as a failed test"

    def test_H1_at_p_0001_is_not_called_a_failed_test(self, graph):
        """The specific misstatement: a result that rejects under the SUPERSEDED rule, reported as failed."""
        w, e, alpha = graph
        out = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        assert out["local_alpha"]["N6_h1"] == 0.0
        assert "N6_h1" in out["structurally_untestable"]
        assert "N6_h1" not in out["not_rejected"]

    def test_genuinely_tested_nodes_remain_in_not_rejected(self, graph):
        """N1/N2 WERE tested, at 0.025 each, and failed. They must keep reading as failed tests."""
        w, e, alpha = graph
        out = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        assert out["not_rejected"] == ["N1_h2_tail", "N2_h2_ra"]
        assert out["local_alpha"]["N1_h2_tail"] == pytest.approx(0.025)

    def test_note_is_carried_in_the_artifact(self, graph):
        w, e, alpha = graph
        out = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        note = out.get("structurally_untestable_note", "")
        assert "never tested" in note and "N6_h1" in note

    def test_every_node_is_in_exactly_one_category(self, graph):
        """No node may be double-counted or silently dropped between the four buckets."""
        w, e, alpha = graph
        out = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        buckets = (out["rejected"], out["not_rejected"],
                   out["untestable"], out["structurally_untestable"])
        flat = [n for b in buckets for n in b]
        assert sorted(flat) == sorted(w), "the four categories must partition the node set"
        assert len(flat) == len(set(flat)), "a node appears in more than one category"


class TestInferenceIsUnchanged:
    """The fix must alter CATEGORISATION only — never which nodes reject."""

    def test_rejected_set_unchanged_on_the_predicted_branch(self, graph):
        w, e, alpha = graph
        assert graphical_alpha_propagation(PREDICTED, w, e, alpha)["rejected"] == []

    def test_rejected_set_unchanged_when_the_graph_activates(self, graph):
        w, e, alpha = graph
        out = graphical_alpha_propagation(ACTIVATED, w, e, alpha)
        assert out["rejected"] == ["N2_h2_ra", "N6_h1"]

    def test_activation_shrinks_the_untestable_set(self, graph):
        """With N2 rejecting, alpha flows and nodes become genuinely testable."""
        w, e, alpha = graph
        pred = graphical_alpha_propagation(PREDICTED, w, e, alpha)
        act = graphical_alpha_propagation(ACTIVATED, w, e, alpha)
        assert len(act["structurally_untestable"]) < len(pred["structurally_untestable"])
        assert act["local_alpha"]["N6_h1"] > 0.0


class TestEdgeCases:
    def test_a_zero_alpha_node_that_DID_reject_is_not_listed(self):
        """p <= 1e-15 can clear a zero threshold; it rejected, so it was testable in the only sense that counts."""
        w = {"A": 0.0, "B": 1.0}
        e = {"A": {"B": 0.0}, "B": {"A": 0.0}}
        out = graphical_alpha_propagation({"A": 0.0, "B": 0.9}, w, e, alpha=0.05)
        assert "A" in out["rejected"]
        assert "A" not in out["structurally_untestable"]

    def test_missing_pvalue_stays_in_untestable_not_the_new_bucket(self):
        """The two reasons are DIFFERENT facts and must not be merged."""
        w = {"A": 0.5, "B": 0.5}
        e = {"A": {"B": 1.0}, "B": {"A": 1.0}}
        out = graphical_alpha_propagation({"A": 0.9, "B": None}, w, e, alpha=0.05)
        assert out["untestable"] == ["B"]
        assert "B" not in out["structurally_untestable"]

    def test_note_absent_when_nothing_is_structurally_untestable(self):
        w = {"A": 0.5, "B": 0.5}
        e = {"A": {"B": 1.0}, "B": {"A": 1.0}}
        out = graphical_alpha_propagation({"A": 0.9, "B": 0.9}, w, e, alpha=0.05)
        assert out["structurally_untestable"] == []
        assert "structurally_untestable_note" not in out
