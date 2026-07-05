"""Structural similarity of LLM-authored reward CODE — report-only mechanism enrichment (R68).

Closes the placebo construct-validity gap the audits flagged. The inert `placebo` and the `placebo_shuffled`
controls already neutralise *cosmetic-token* leakage, but they cannot show whether the language model
conditions the **program structure** of its reward on the fed signal. This module answers that directly and
*identifier-invariantly*: it canonicalises every reward's abstract syntax tree (variable/argument names and
constant VALUES are discarded — only the node-TYPE structure survives), enumerates the set of depth-bounded
subtree shapes (the "AST Distinct-N" signature; canonicalised subtree enumeration), and measures pairwise
Jaccard similarity. The headline statistic is **mean within-condition vs mean across-condition** structural
similarity, with a deterministic label-permutation test: if the LLM routes the fed tail signal into the
*shape* of the code it writes, rewards authored under the same feedback condition are structurally more
similar to each other than to rewards authored under other conditions.

Scope: report-only, DISJOINT from the frozen m=6 testing family; never gates a hypothesis. It is a stronger,
identifier-invariant companion to the reward-program differential (R51) — that one counts declared tail
constructs; this one tests whole-program structural clustering by condition. Pure standard-library `ast`
(+ numpy, already a dependency, for the seeded permutation test); fully DETERMINISTIC (no model, no GPU);
does not touch the numpy<2 / pandas<3 pins.

References: TSED (Song et al., ACL 2024, arXiv:2404.08817); AST canonicalised-subtree "Distinct-N"
(arXiv:2504.12522, ICLR 2025); APTED/Zhang-Shasha tree-edit lineage (Pawlik & Augsten). The Jaccard of
canonicalised subtree sets is a deterministic, dependency-free structural similarity in the same family.
"""

from __future__ import annotations

import ast
from itertools import combinations
from typing import Any

import numpy as np

__all__ = ["canonical_shapes", "has_structure", "jaccard", "structural_similarity",
           "reward_code_structure_report"]


def _shape(node: ast.AST, depth: int) -> str:
    """Canonical depth-bounded subtree shape of ``node`` — NODE TYPES only.

    Variable/argument names and constant values never enter: ``ast.iter_child_nodes`` traverses only AST
    children, so a ``Name``'s ``id`` string and a ``Constant``'s value (both plain attributes, not child
    nodes) are excluded by construction. Two snippets that differ ONLY in identifiers/literals therefore
    produce identical shapes — exactly the cosmetic-invariance the placebo construct-validity check needs.
    """
    t = type(node).__name__
    if depth <= 1:
        return t
    kids = [_shape(c, depth - 1) for c in ast.iter_child_nodes(node)]
    return t + "(" + ",".join(kids) + ")" if kids else t


def canonical_shapes(source: str, depth: int = 4) -> frozenset[str]:
    """The set of canonicalised depth-``depth`` subtree shapes over every node of ``source``'s AST.

    Returns an empty set for unparseable source (a malformed candidate contributes nothing rather than
    crashing the report)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    return frozenset(_shape(n, depth) for n in ast.walk(tree))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity ``|a ∩ b| / |a ∪ b|`` in ``[0, 1]``; ``1.0`` for two empty sets (both trivial)."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


#: A bodiless ``Module`` is what ``ast.parse`` yields for an EMPTY, whitespace- or comment-only
#: source — it parses fine, so its shape-set is the TRUTHY singleton ``{"Module"}`` rather than the
#: empty set an unparseable source gives.
_MODULE_ONLY: frozenset[str] = frozenset({"Module"})


def has_structure(shapes: frozenset[str] | set[str]) -> bool:
    """Does this shape-set carry any program structure to compare? (the P7c predicate, ONE definition)

    False for BOTH degenerate cases: an unparseable source (empty set) and an empty / whitespace- /
    comment-only one (``{"Module"}``). Both matter because :func:`jaccard` returns **1.0** for two
    empty sets by the trivial-set convention and **1.0** for two ``{"Module"}`` singletons by ordinary
    set identity — so either pair reads as "maximally similar" and inflates whatever
    within-condition / paired mean it lands in.

    Introduced 2026-07-27 (deep review #117) because this predicate existed in THREE places and was
    wrong in two: :func:`reward_code_structure_report` below and
    ``contamination.named_vs_blinded_structural`` both tested only for the empty set, while
    ``reward_taxonomy._signature`` correctly tested ``shapes <= {"Module"}``. Same pattern, same
    consequence (a fabricated structural-similarity signal), fixed in one file and not the others —
    so the predicate is defined ONCE here, in the module that owns ``canonical_shapes``, and imported
    by the rest.
    """
    return bool(shapes) and not set(shapes) <= _MODULE_ONLY


def structural_similarity(src_a: str, src_b: str, depth: int = 4) -> float:
    """Identifier- and literal-invariant structural similarity of two reward sources (Jaccard of shapes).

    NOTE: A raw primitive: it does NOT screen structureless sources, so two empty/comment-only programs
    score 1.0. Callers that AGGREGATE similarities must filter with :func:`has_structure` first (see
    #117); the two in-repo aggregators do.
    """
    return jaccard(canonical_shapes(src_a, depth), canonical_shapes(src_b, depth))


def reward_code_structure_report(
    arm_sources: dict[str, list[str]],
    *,
    depth: int = 4,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Within- vs across-condition structural clustering of authored reward code, with a permutation test.

    Parameters
    ----------
    arm_sources : dict[str, list[str]]
        ``{feedback_condition: [reward_source, ...]}`` — the authored reward code grouped by arm. Needs at
        least two conditions and at least one within-condition pair to produce a within-similarity number.
    depth : int
        Subtree-shape depth (default 4; the AST Distinct-N height).
    n_perm : int
        Label-permutation replications for the deterministic significance test.
    seed : int
        RNG seed (pinned -> byte-deterministic).

    Returns
    -------
    dict
        ``{within_mean, across_mean, difference, p_value, n_within_pairs, n_across_pairs, n_unparseable,
        per_condition}``. ``n_unparseable`` counts sources EXCLUDED as STRUCTURELESS — an empty AST
        signature (unparseable) OR a bodiless ``Module`` (empty / whitespace / comment-only, widened
        #117) — they are dropped, not scored as similarity-1.0, so two non-programs cannot inflate the
        within-condition mean (P7c).
        ``difference = within_mean - across_mean`` (> 0 = the LLM clusters code structure by condition);
        ``p_value`` is the one-sided permutation probability that a random re-labelling reaches the observed
        difference. Report-only and directional — it never enters the inferential result.
    """
    # Precompute shape sets once per source; flatten with a condition label. EXCLUDE unparseable sources
    # (empty shape signature): a pair of two EMPTY sets has Jaccard 1.0 by the trivial-set convention, so two
    # malformed rewards under the SAME condition would spuriously read as maximally similar and INFLATE the
    # within-condition mean — a false "structural clustering by condition". Dropping them (with a logged
    # count) is the honest handling: an unparseable reward carries NO structure to cluster on (P7c).
    labels: list[str] = []
    shapes: list[frozenset[str]] = []
    n_unparseable = 0
    for cond, srcs in arm_sources.items():
        for s in srcs:
            sh = canonical_shapes(s, depth)
            # WIDENED 2026-07-27 (deep review #117). This was `if not sh` — NON-EMPTY only, which
            # catches an UNPARSEABLE source but NOT an EMPTY or comment-only one: `ast.parse("")`
            # succeeds and walks one bodiless `Module`, so its shape-set is the TRUTHY singleton
            # {"Module"} and `jaccard` between two of them is 1.0. Those pairs therefore survived the
            # filter and produced exactly the "false structural clustering by condition" the comment
            # above warns about. `reward_taxonomy._signature` already guarded `shapes <= {"Module"}`;
            # this site and `contamination.named_vs_blinded_structural` did not — one pattern, three
            # call sites, two of them wrong until swept.
            if not has_structure(sh):  # no AST body -> no structure to cluster on (P7c)
                n_unparseable += 1
                continue
            labels.append(cond)
            shapes.append(sh)
    m = len(shapes)
    if m < 2:
        return {"status": "insufficient", "n_sources": m, "n_unparseable": n_unparseable}

    # Full pairwise similarity matrix (deterministic, symmetric).
    sim = np.ones((m, m), dtype=float)
    for i, j in combinations(range(m), 2):
        sij = jaccard(shapes[i], shapes[j])
        sim[i, j] = sim[j, i] = sij
    lab = np.asarray(labels)

    def _within_across(label_arr: np.ndarray) -> tuple[float, float, int, int]:
        same = label_arr[:, None] == label_arr[None, :]
        iu = np.triu_indices(m, k=1)
        same_pairs = same[iu]
        vals = sim[iu]
        w = vals[same_pairs]
        a = vals[~same_pairs]
        return (
            float(w.mean()) if w.size else float("nan"),
            float(a.mean()) if a.size else float("nan"),
            int(w.size),
            int(a.size),
        )

    w_mean, a_mean, n_w, n_a = _within_across(lab)
    if n_w == 0 or n_a == 0:
        return {
            "status": "degenerate",
            "within_mean": w_mean,
            "across_mean": a_mean,
            "n_within_pairs": n_w,
            "n_across_pairs": n_a,
            "n_unparseable": n_unparseable,
        }
    obs = w_mean - a_mean

    rng = np.random.default_rng(seed)
    ge = 1  # +1 for the observed (standard permutation-p numerator/denominator guard)
    perm = lab.copy()
    for _ in range(n_perm):
        rng.shuffle(perm)
        pw, pa, _, _ = _within_across(perm)
        if (pw - pa) >= obs - 1e-12:
            ge += 1
    p_value = ge / (n_perm + 1)

    per_cond = {
        c: int(sum(1 for label_v in labels if label_v == c)) for c in dict.fromkeys(labels)
    }
    return {
        "status": "ok",
        "within_mean": w_mean,
        "across_mean": a_mean,
        "difference": obs,
        "p_value": float(p_value),
        "n_within_pairs": n_w,
        "n_across_pairs": n_a,
        "n_unparseable": n_unparseable,
        "per_condition": per_cond,
        "depth": depth,
        "seed": seed,
    }
