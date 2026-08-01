"""The RATIFIED primary decision rule: graphical alpha-propagation (Bretz et al. 2009).

These tests exist because the 2026-07-26 deep review found the rule REGISTERED AND RATIFIED (R108:
`primary_rule: bonferroni_weighted_graph`, R31 superseded) but with **no implementation anywhere** —
the analysis still computed only the superseded Bonferroni-over-4 mirror, so the confirmatory
inference could not be produced as registered (write-time registry row 36).

They are written to FAIL if the rule is removed, hardcoded, or drifts from the registered graph.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.multiple_testing import (  # noqa: E402
    graphical_alpha_propagation,
    registered_alpha_graph,
)

# A hand-workable two-node graph: equal weights, full recycling both ways.
W2 = {"A": 0.5, "B": 0.5}
G2 = {"A": {"B": 1.0}, "B": {"A": 1.0}}


def test_known_answer_cascade() -> None:
    """A rejection RELEASES its weight, which can then carry the second node.

    Hand-worked at alpha=0.05: local levels start at 0.025 each. p_A=0.02 <= 0.025 -> reject A; A's
    weight propagates (w_B = 0.5 + 0.5*1 = 1.0) so B is now tested at 0.05, and p_B=0.04 <= 0.05 ->
    reject B. BOTH reject, even though 0.04 > 0.025 — the cascade is the whole point of the graph.
    """
    out = graphical_alpha_propagation({"A": 0.02, "B": 0.04}, W2, G2, alpha=0.05)
    assert out["rejected"] == ["A", "B"]
    assert out["local_alpha"]["B"] == pytest.approx(0.05)


def test_known_answer_alpha_split_blocks_both() -> None:
    """The measured PRICE of the graph: at the split level neither node clears, though both would at
    the full alpha. This is the 0.80 -> 0.7007 headline cost the ratification pack quantified."""
    out = graphical_alpha_propagation({"A": 0.03, "B": 0.04}, W2, G2, alpha=0.05)
    assert out["rejected"] == []
    assert out["any_rejected"] is False


def test_rejected_set_is_order_invariant() -> None:
    """The closed-test shortcut's defining property: the rejected SET does not depend on the order in
    which simultaneously-rejectable nodes are taken. Exercised by permuting the node ordering."""
    p = {"A": 0.001, "B": 0.001, "C": 0.20}
    base = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    edges = {"A": {"B": 0.5, "C": 0.5}, "B": {"A": 0.5, "C": 0.5}, "C": {"A": 0.5, "B": 0.5}}
    sets = set()
    for order in itertools.permutations(["A", "B", "C"]):
        w = {k: base[k] for k in order}
        e = {k: edges[k] for k in order}
        sets.add(frozenset(graphical_alpha_propagation(p, w, e, alpha=0.05)["rejected"]))
    assert len(sets) == 1, f"rejected set depends on evaluation order: {sets}"


def test_untestable_node_never_rejects_and_is_reported() -> None:
    """A node with no p-value (too few shared seeds) must NOT be silently treated as passing."""
    out = graphical_alpha_propagation({"A": 0.001, "B": None}, W2, G2, alpha=0.05)
    assert out["untestable"] == ["B"]
    assert "B" not in out["rejected"]
    assert out["rejected"] == ["A"]


def test_nothing_rejects_under_the_global_null() -> None:
    out = graphical_alpha_propagation({"A": 1.0, "B": 1.0}, W2, G2, alpha=0.05)
    assert out["rejected"] == [] and out["not_rejected"] == ["A", "B"]


def test_malformed_graphs_fail_loud() -> None:
    with pytest.raises(ValueError, match="sum to <= 1"):
        graphical_alpha_propagation({"A": 0.01, "B": 0.01}, {"A": 0.8, "B": 0.8}, G2, alpha=0.05)
    with pytest.raises(ValueError, match="out-edges"):
        graphical_alpha_propagation(
            {"A": 0.01, "B": 0.01}, W2, {"A": {"B": 1.5}, "B": {"A": 1.0}}, alpha=0.05
        )
    with pytest.raises(ValueError, match="alpha must be in"):
        graphical_alpha_propagation({"A": 0.01, "B": 0.01}, W2, G2, alpha=1.5)


def test_executed_graph_IS_the_registered_graph() -> None:
    """Drift lock: the rule must run on the graph the pre-registration declares FROZEN.

    `registered_alpha_graph` READS `config/preregistration.yaml: inference.validity_tier`; a hardcoded
    copy would be exactly the executed-vs-registered drift the freeze gate's cross-file guards exist to
    catch (arm roster / h1_baselines / confirmatory_author).
    """
    w, e, alpha = registered_alpha_graph()
    assert alpha == pytest.approx(0.05)
    assert set(w) == {"N1_h2_tail", "N2_h2_ra", "N3_h3", "N4_h4", "N5_structure", "N6_h1"}
    assert w["N1_h2_tail"] == pytest.approx(0.5) and w["N2_h2_ra"] == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)
    for node, outs in e.items():
        assert sum(outs.values()) <= 1.0 + 1e-9, f"{node} out-edges exceed 1"
    # And it RUNS end-to-end on the registered graph.
    p = {n: 1.0 for n in w}
    assert graphical_alpha_propagation(p, w, e, alpha)["rejected"] == []


def test_registered_graph_reproduces_the_predicted_activation_path() -> None:
    """The design's predicted path: N1 does NOT reject (the registered §1a prediction is the NULL
    branch), but N2's equivalence route DOES — and that alone must open the tier, which is exactly
    why `bergerhsu1996equivalence` is load-bearing in the tier design.

    NOTE: RE-POINTED 2026-08-01 (A16 / analysis A24). This test used to build the node p-value dict BY
    HAND — ``p = {n: 1.0 ...}; p["N2_h2_ra"] = 0.001`` — and call the propagation directly. Its own
    docstring asserted "N2's TOST DOES [reject]" while the code path that would have SHOWN there was
    no TOST route was bypassed entirely. THAT BYPASS IS WHY NO TEST EVER CAUGHT THE MISSING ROUTE:
    the test documented the registered behaviour and was structurally unable to check it. It now
    goes through ``tier_node_pvalues``, so the extraction and the propagation are exercised together.
    """
    from src.inference.validity_tier import tier_verdict  # noqa: PLC0415 — local: exercises the real path

    w, e, alpha = registered_alpha_graph()
    # A result dict of the shape analyze_campaign actually produces, under the PREDICTED branch:
    # every tail leg fails to reject (N1 stays shut) while every RA leg clears non-inferiority.
    out = {
        "h2": {
            "tail_legs": [{"contrast": f"t{i}", "pvalue_one_sided": 0.9} for i in range(3)],
            "legs": [{"contrast": f"r{i}", "pvalue_one_sided": 0.9,
                      "pvalue_non_inferiority": 0.001,
                      "pvalue_non_inferiority_conservative": 0.002} for i in range(3)],
        },
    }
    res = tier_verdict(out)
    assert res["nodes"]["N2_h2_ra"]["pvalue"] == pytest.approx(0.001), \
        "N2 must read the registered non-inferiority p, not the superiority leg"
    assert "N2_h2_ra" in res["rejected"]
    assert "N1_h2_tail" not in res["rejected"], "the predicted branch keeps the tail node shut"
    # N2's weight must flow onward, raising at least one downstream node's local level above its start.
    assert any(res["local_alpha"][n] > w[n] * alpha + 1e-12 for n in w if n != "N2_h2_ra")
