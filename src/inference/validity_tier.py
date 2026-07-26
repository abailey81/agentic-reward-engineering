"""Assemble the six confirmatory node p-values and run the RATIFIED graphical decision rule.

Bridges what ``scripts/analyze_campaign.py`` already computes to
``multiple_testing.graphical_alpha_propagation`` — the rule ratified 2026-07-26 (R108:
``primary_rule: bonferroni_weighted_graph``, superseding R31).

WHY A SEPARATE MODULE. The rule must be executable and testable WITHOUT campaign data, and the graph
must be READ from the registered config rather than copied (``forking_path_guard`` declares it FROZEN).
So ``analyze_campaign`` needs exactly one call, and this logic is unit-tested today — before any results
exist, which is precisely when a decision rule is allowed to be written.

FAIL-SAFE BY CONSTRUCTION. A node whose p-value cannot be located is reported ``untestable`` and can
never reject; it is NEVER silently treated as a pass or a fail. Every path searched is recorded, so a
shape mismatch surfaces in the artifact instead of producing a confidently wrong confirmatory verdict —
the worst possible failure here.

NODE MAP — every path below was VERIFIED against the producing code, not inferred from docstrings
(the 2026-07-26 deep review found 4 of 6 first-guess paths wrong, which is why this is spelled out):

===============  ==========================================================  =======================
node             source in ``out``                                            rule
===============  ==========================================================  =======================
N1_h2_tail       ``out["h2"]["tail_legs"][*]["pvalue_one_sided"]``            MAX over legs (IUT)
N2_h2_ra         ``out["h2"]["legs"][*]["pvalue_non_inferiority"]``            MAX over legs (IUT)
N3_h3            ``out["h3"]["difference"]["pvalue_one_sided"]``              single test
N4_h4            ``out["h4"]["tests"][*]["pvalue_one_sided"]``                MAX over legs (IUT)
N5_structure     ``out["h2_structure"]["cvar"]["pvalue_one_sided"]``          single test (metric: cvar)
N6_h1            ``out["h1_beat_human"]["iut"]["iut_pvalue"]``                ALREADY the MAX over the
                                                                              canon, gated on
                                                                              ``all_baselines_present``
===============  ==========================================================  =======================

N5 reads the **cvar** sub-result because the registered node is
``{test: distributional_gt_placebo_shuffled, metric: cvar, level: 0.05}``. N6 reuses the pipeline's own
``iut_pvalue`` and honours ``all_baselines_present``: dominance is certifiable only when every canon
member is testable, so a missing member yields ``untestable`` rather than a cheaper claim.

N2 AND THE A16 CORRECTION (2026-08-01, RUN 11) — READ THIS BEFORE CHANGING N2's KEY.
The registered node is ``{test: h2_ra_iut_or_tost, equivalence: tost_0.05_dsr}``, and the same frozen
block records "alpha recycled on ANY rejection (superiority OR equivalence)" and, in its own dated
note, "activation rests entirely on N2 rejecting via TOST — a real pre-registered alpha source".
Until today this module read ``pvalue_one_sided`` — the SUPERIORITY leg only — so the registered
disjunction had **no implementation at all**. Under the design's OWN predicted branch (Sharpe tie AND
tail tie) N1 cannot reject, so with N2 unable to reject either, ALL SIX nodes were unreachable and
four hypotheses had no confirmatory decision path.

The repair is CONFORMANCE, not an amendment: ``{theta > 0} UNION {-d < theta < d}`` IS
``{theta > -d}``, so "superiority or equivalence" is ONE hypothesis and it is exactly non-inferiority
at the SESOI. Nothing frozen changes.

TWO THINGS THIS MODULE MUST NOT LET A LATER READER FORGET:
  1. Over THREE legs, ``AND_j {theta_j > -d}`` is STRICTLY WEAKER than
     ``(AND_j {theta_j > 0}) OR (AND_j {|theta_j| < d})`` — it also fires at "superior on two legs,
     mildly inferior on the third". So N2's claim is WRITTEN as one-sided NON-INFERIORITY at the
     SESOI, **never** as "superior or equivalent".
  2. The registered margin is the PERMISSIVE one. ``n2_key`` therefore exists so the caller can
     report the pre-specified sensitivities — the conservative margin and the as-implemented-until-
     now superiority rule — ALONGSIDE the primary. All three were fixed in advance, on the record,
     while 0 of 3 H2-RA legs were computable (2026-08-01T13:01:15Z, HEAD 57c5ecc4), so which one
     rejects cannot select the claim.
"""
from __future__ import annotations

import math
from typing import Any

from src.inference.multiple_testing import graphical_alpha_propagation, registered_alpha_graph

__all__ = ["NODE_SOURCES", "N2_KEYS", "tier_node_pvalues", "tier_verdict"]

#: The three N2 rules, PRE-SPECIFIED TOGETHER while blind (A16; see the module docstring). Keyed by
#: the label under which each verdict is reported, so the primary cannot be swapped for a
#: sensitivity by editing one string in one place.
N2_KEYS: dict[str, str] = {
    "primary": "pvalue_non_inferiority",
    "sensitivity_conservative_margin": "pvalue_non_inferiority_conservative",
    "sensitivity_superiority_only": "pvalue_one_sided",
}

#: node -> spec. ``legs`` = MAX over that list (intersection-union, Berger 1982); ``key`` = a single
#: value at ``path``; ``require`` = a boolean flag at ``path`` that must be True to certify.
NODE_SOURCES: dict[str, dict[str, Any]] = {
    "N1_h2_tail":   {"path": ("h2",), "legs": "tail_legs", "key": "pvalue_one_sided"},
    "N2_h2_ra":     {"path": ("h2",), "legs": "legs", "key": N2_KEYS["primary"]},
    "N3_h3":        {"path": ("h3", "difference"), "key": "pvalue_one_sided"},
    "N4_h4":        {"path": ("h4",), "legs": "tests", "key": "pvalue_one_sided"},
    "N5_structure": {"path": ("h2_structure", "cvar"), "key": "pvalue_one_sided"},
    "N6_h1":        {"path": ("h1_beat_human", "iut"), "key": "iut_pvalue",
                     "require": "all_baselines_present"},
}


def _dig(out: Any, path: tuple[str, ...]) -> Any:
    node = out
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def tier_node_pvalues(out: dict[str, Any], *, n2_key: str | None = None) -> dict[str, dict[str, Any]]:
    """Extract ONE p-value per confirmatory node from an ``analyze_campaign`` result dict.

    Returns ``{node: {"pvalue": float|None, "source": str, "rule": str, "n_legs": int|None,
    "reason": str|None}}``. ``pvalue is None`` means NOT TESTABLE — reported, never rejectable.

    ``n2_key`` selects which of the three pre-specified N2 rules to read; it must be one of
    :data:`N2_KEYS`' VALUES, and an unrecognised key raises rather than silently falling back to the
    primary. Defaulting to the primary is deliberate: a caller that forgets the argument gets the
    registered rule, never a sensitivity.
    """
    if n2_key is not None and n2_key not in set(N2_KEYS.values()):
        raise ValueError(
            f"n2_key={n2_key!r} is not one of the pre-specified N2 rules {sorted(set(N2_KEYS.values()))}. "
            f"A16 fixed these three in advance while blind; introducing a fourth after the fact is "
            f"the forking path this argument exists to prevent.")
    found: dict[str, dict[str, Any]] = {}
    for node, spec in NODE_SOURCES.items():
        path: tuple[str, ...] = spec["path"]
        pkey: str = n2_key if (node == "N2_h2_ra" and n2_key) else spec["key"]
        source = 'out["' + '"]["'.join(path) + '"]'
        rec: dict[str, Any] = {
            # `key` is recorded per node, not just `source`: under A16 the SAME path can be read
            # through three different keys, and a confirmatory record that says WHERE it looked but
            # not WHAT it read cannot be audited.
            "pvalue": None, "source": source, "key": pkey, "n_legs": None, "reason": None,
            "rule": "max over legs (intersection-union, Berger 1982)" if spec.get("legs") else "single test",
        }
        blob = _dig(out, path)
        if not isinstance(blob, dict):
            rec["reason"] = f"no dict at {source}"
            found[node] = rec
            continue
        if spec.get("require") and not blob.get(spec["require"]):
            # e.g. N6: dominance is certifiable ONLY when every canon member is testable.
            rec["reason"] = f"{spec['require']} is not True — the node cannot certify"
            found[node] = rec
            continue
        if spec.get("legs"):
            legs = blob.get(spec["legs"])
            if not isinstance(legs, list) or not legs:
                rec["reason"] = f"no non-empty {spec['legs']!r} list at {source}"
            else:
                ps = [_finite(leg.get(pkey)) if isinstance(leg, dict) else None for leg in legs]
                rec["n_legs"] = len(ps)
                if any(p is None for p in ps):
                    rec["reason"] = "at least one leg has no usable p-value (an IUT cannot certify)"
                else:
                    rec["pvalue"] = max(ps)
        else:
            rec["pvalue"] = _finite(blob.get(pkey))
            if rec["pvalue"] is None:
                rec["reason"] = f"{pkey!r} absent or non-finite at {source}"
        found[node] = rec
    return found


def tier_verdict(out: dict[str, Any], root: Any = None, *,
                 n2_key: str | None = None, with_sensitivities: bool = True) -> dict[str, Any]:
    """Run the ratified graphical rule over an ``analyze_campaign`` result dict.

    The graph (initial weights, edges, alpha) is READ from the registered config — never hardcoded, so
    the executed rule cannot drift from the frozen one. Returns the propagation result plus the
    per-node extraction record, so a shape mismatch is visible rather than silently decisive.

    With ``with_sensitivities`` (the default) the result also carries ``["sensitivities"]``: the SAME
    propagation re-run under each non-primary N2 rule from :data:`N2_KEYS`. They are computed
    UNCONDITIONALLY rather than on request, because a sensitivity you have to ask for is a
    sensitivity you can decline to ask for once you have seen the primary.
    """
    weights, edges, alpha = registered_alpha_graph(root)
    nodes = tier_node_pvalues(out, n2_key=n2_key)
    pvalues: dict[str, float | None] = {n: nodes.get(n, {}).get("pvalue") for n in weights}
    result = graphical_alpha_propagation(pvalues, weights, edges, alpha)
    result["nodes"] = nodes
    result["method"] = "graphical_bretz_maurer_brannath_posch_2009"
    result["registered_rule"] = "bonferroni_weighted_graph"
    result["supersedes"] = "R31 separate-estimands (ratified 2026-07-26, R108)"
    result["n2_rule"] = n2_key or NODE_SOURCES["N2_h2_ra"]["key"]
    if with_sensitivities:
        result["sensitivities"] = {
            label: tier_verdict(out, root, n2_key=key, with_sensitivities=False)
            for label, key in N2_KEYS.items()
            if key != result["n2_rule"]
        }
        result["sensitivity_note"] = (
            "A16: the three N2 rules were fixed in advance, on the record, at "
            "2026-08-01T13:01:15Z with 0 of 3 H2-RA legs computable. Which one rejects cannot "
            "select the claim because all three are reported. N2's claim is one-sided "
            "NON-INFERIORITY at the SESOI, never 'superior or equivalent'."
        )
    return result
