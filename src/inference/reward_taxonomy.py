"""Reward-program TAXONOMY — program KINDS induced from the authored-reward archive (report-only).

Delivers the instrument CH7 explicitly deferred ("a true taxonomy of program kinds, induced from the
campaign archive, is left to future work"): a categorical, identifier-invariant taxonomy of the reward
PROGRAMS the designer authored, plus the per-arm KIND composition. The construct-prevalence differential
(R51) counts declared tail constructs per program; the structural-similarity report (R68) tests
within-vs-across-condition clustering as one scalar. Neither names the *kinds*. This module does: it
canonicalises every program's AST shape-set (:func:`~src.inference.reward_code_distance.canonical_shapes`
— node TYPES only, so identifiers/literals never enter), builds the pairwise Jaccard similarity graph, and
takes the connected components at ``similarity >= sim_threshold`` as the program kinds. Each kind gets a
MEDOID exemplar (the member with maximal mean within-kind similarity) and a HUMAN-READABLE label derived
from the construct prevalence among its members (the same theory->code construct vocabulary the mechanism
kernel counts — single source of truth, hosted here and re-imported by ``scripts/inspect_rewards.py``).

The scientific payoff is :func:`taxonomy_by_arm`: induce ONE taxonomy over the pooled corpus, then read
off the per-arm composition — do different feedback arms author different KINDS of programs, or the same
kinds reshaped? A search arm sampling one parameterised template should collapse to one kind; an LLM arm
that routes the fed tail into program STRUCTURE should shift its kind mixture. Shannon entropy over kinds
summarises per-arm diversity; the Jaccard overlap of kind SETS summarises cross-arm sharing.

Scope and discipline
--------------------
* **Report-only, DISJOINT** from the frozen m=6 confirmatory family: no ``arm_a/arm_b/metric/level``
  family-tuple keys are ever emitted, and nothing here gates H1-H4.
* **P7c empty-AST rule, extended**: unparseable sources are EXCLUDED and counted (``n_unparseable``),
  never clustered — ``jaccard(empty, empty) == 1.0`` would glue every malformed program into one fake
  kind. The same exclusion covers EMPTY/comment-only sources: ``ast.parse("")`` succeeds and yields a
  bodiless ``Module`` whose shape-set is the singleton ``{"Module"}``, so two empty sources would read as
  identical structure; a program with no AST body carries NO structure to classify.
* **Deterministic**: no randomness anywhere; ids are sorted, components are ordered by (-size, first
  member), medoid ties break lexicographically — byte-identical output on re-run.
* **Threshold honesty**: :func:`taxonomy_threshold_sensitivity` re-clusters at several thresholds and
  reports the pair-counting Rand index (Rand 1971; the UNadjusted index, computed directly — no sklearn)
  between adjacent thresholds, so the write-up can show the taxonomy is not a threshold artifact.
  Connected components at a HIGHER threshold only ever split (the edge set shrinks), so ``n_kinds`` is
  provably non-decreasing in the threshold — a built-in sanity invariant the tests assert.

Pure standard-library ``ast``/``re`` + numpy (already a dependency); no new dependencies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np

from src.inference.reward_code_distance import canonical_shapes, has_structure, jaccard

__all__ = [
    "CONSTRUCTS",
    "TAIL_CONSTRUCTS",
    "construct_prevalence",
    "induce_taxonomy",
    "kind_label",
    "pool_sources",
    "rand_index",
    "taxonomy_by_arm",
    "taxonomy_threshold_sensitivity",
]


# --------------------------------------------------------------------------- #
# theory -> code construct vocabulary (single source of truth)                 #
# --------------------------------------------------------------------------- #
#: The risk-shaping CONSTRUCTS a reward program can encode, each a (name, regex, tail?) probe applied to
#: the comment-stripped source. These are the theory->code primitives the dissertation's mechanism story
#: turns on (cf. the six-term shared family ADR-010 + the prototype rewards' own vocabulary): does a
#: distributional arm's CODE reference tail/risk structure MORE than scalar/placebo? Patterns are
#: deliberately word-boundaried + permissive on the maths around them so a construct counts whether the
#: LLM wrote ``cvar``, ``CVaR``, ``np.percentile(..., 5)`` or ``np.quantile(..., 0.05)``. A construct is
#: TAIL-shaped (``tail=True``) when it touches the left tail specifically (CVaR / drawdown /
#: Sortino-downside / a low percentile) vs a generic risk control (rolling-vol / turnover /
#: concentration) — the tail set is what H2 predicts the distributional feedback should grow.
#:
#: MOVED here from ``scripts/inspect_rewards.py`` (R51 lineage) VERBATIM so the taxonomy labels, the
#: forensics report, and the mechanism kernel all count constructs against ONE vocabulary —
#: ``inspect_rewards`` re-imports these under its original private names, so every existing call site
#: (``ir._construct_prevalence`` in ``scripts/analyze_campaign.py``, the forensics tables, the tests)
#: reads the same table. A ``src`` module must not import from ``scripts/``, hence the move, not a copy.
CONSTRUCTS: tuple[tuple[str, str, bool], ...] = (
    # name                pattern (applied to comment-stripped, lower-cased source)         tail?
    ("cvar", r"\bcvar\b|conditional[_ ]?value[_ ]?at[_ ]?risk", True),
    ("quantile_tail", r"\b(?:np\.)?(?:percentile|quantile)\b", True),
    ("drawdown", r"\bdraw[_ ]?down\b|\bpeak\b|\bunderwater\b", True),
    ("sortino_downside", r"\bsortino\b|\bdownside\b|\bsemi[_ ]?dev|\bsemivariance\b", True),
    ("left_tail_mass", r"left[_ ]?tail|tail[_ ]?mass|tail[_ ]?loss", True),
    ("rolling_vol", r"\bstd\b|\bvol\b|\bvolatilit|\bvariance\b|\bsigma\b|\bewm|\bema_std\b|welford|m2\b", False),
    ("turnover", r"\bturnover\b|abs\(\s*weights\s*-\s*prev_weights|trade[_ ]?cost|transaction[_ ]?cost", False),
    ("online_sharpe", r"\bsharpe\b|risk[_ ]?adj|mean\s*/\s*std|\bmu\s*/\s*", False),
    ("herfindahl", r"\bherfindahl\b|\bhhi\b|concentration|sum\(\s*\w*\s*\*\*\s*2|diversif", False),
)

#: The TAIL-shaped subset (names) — the constructs H2 predicts distributional feedback should grow.
TAIL_CONSTRUCTS: tuple[str, ...] = tuple(name for name, _pat, is_tail in CONSTRUCTS if is_tail)

#: Human label fragment per construct, used to compose kind labels ("cvar-penalized + sharpe-ratio").
#: Keyed by construct name; a construct added to :data:`CONSTRUCTS` without a fragment falls back to its
#: own name, so the label can never silently drop a dominant construct.
_LABEL_FRAGMENTS: dict[str, str] = {
    "cvar": "cvar-penalized",
    "quantile_tail": "quantile-tail-shaped",
    "drawdown": "drawdown-penalized",
    "sortino_downside": "downside-semidev-penalized",
    "left_tail_mass": "tail-mass-penalized",
    "rolling_vol": "volatility-penalized",
    "turnover": "turnover-regularized",
    "online_sharpe": "sharpe-ratio",
    "herfindahl": "concentration-regularized",
}

#: At most this many construct fragments appear verbatim in a kind label; the rest are summarised as
#: "(+k more)". LLM-authored rewards routinely reference 5-6 constructs at once, and a 6-fragment label
#: is unreadable in a table — the full prevalence map is always emitted alongside.
_LABEL_MAX_FRAGMENTS = 3


def construct_prevalence(source: str) -> dict[str, bool]:
    """Which risk-shaping constructs does this reward SOURCE reference? (comment-stripped, case-folded).

    Returns ``{construct_name: present}`` over :data:`CONSTRUCTS`. Comments are stripped first (a
    construct named only in a ``# …`` note does not count as shaped code) and the source is lower-cased
    so ``CVaR`` and ``cvar`` match the same probe. This is the theory->code half of the mechanism loop:
    the named constructs the LLM actually WROTE, per program.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines()).lower()
    return {name: bool(re.search(pat, code)) for name, pat, _is_tail in CONSTRUCTS}


# --------------------------------------------------------------------------- #
# structural signatures + deterministic clustering                             #
# --------------------------------------------------------------------------- #
def _signature(source: str, depth: int) -> frozenset[str] | None:
    """The canonical shape-set of ``source``, or ``None`` when it carries no clusterable structure.

    ``None`` for (a) unparseable source (``canonical_shapes`` returns the empty set — P7c) and (b) an
    EMPTY/comment-only source: ``ast.parse("")`` succeeds and walks exactly one bodiless ``Module`` node,
    so its shape-set is the non-empty singleton ``{"Module"}`` — two such sources would score Jaccard 1.0
    and glue into a fake "empty program" kind. A program with no AST body has no structure to classify.
    """
    shapes = canonical_shapes(source, depth)
    if not has_structure(shapes):  # #117: ONE definition of the P7c predicate, in reward_code_distance
        return None
    return shapes


def _split_signatures(
    sources: Mapping[str, str], depth: int
) -> tuple[list[str], list[frozenset[str]], list[str]]:
    """Sorted (parseable ids, aligned shape-sets, excluded ids) for a ``{id: source}`` corpus."""
    ids: list[str] = []
    shapes: list[frozenset[str]] = []
    excluded: list[str] = []
    for cid in sorted(str(k) for k in sources):
        sig = _signature(str(sources[cid]), depth)
        if sig is None:
            excluded.append(cid)
        else:
            ids.append(cid)
            shapes.append(sig)
    return ids, shapes, excluded


def _pairwise_similarity(shapes: list[frozenset[str]]) -> np.ndarray:
    """Full symmetric Jaccard similarity matrix over the shape-sets (deterministic; diagonal 1)."""
    m = len(shapes)
    sim = np.ones((m, m), dtype=float)
    for i, j in combinations(range(m), 2):
        sim[i, j] = sim[j, i] = jaccard(shapes[i], shapes[j])
    return sim


def _components(ids: list[str], sim: np.ndarray, threshold: float) -> list[list[str]]:
    """Connected components of the ``sim >= threshold`` graph — the program kinds.

    Union-find over the SORTED id order, so the result is independent of input dict order. Components are
    returned as sorted member lists, ordered by (-size, first member id): "kind 1" is always the largest
    kind, and equal-size kinds order by their lexicographically first member — fully deterministic.
    """
    n = len(ids)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for i, j in combinations(range(n), 2):
        if sim[i, j] >= threshold:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)  # root at the smaller index -> id-order canonical
    groups: dict[int, list[str]] = {}
    for k in range(n):
        groups.setdefault(_find(k), []).append(ids[k])  # ids already sorted -> members sorted
    return sorted(groups.values(), key=lambda members: (-len(members), members[0]))


def _medoid(member_idx: list[int], sim: np.ndarray) -> tuple[int, float | None]:
    """The medoid index (max mean similarity to the OTHER members; ties -> first in id order).

    A singleton kind has no other member to be similar to — its medoid is the member itself with mean
    similarity ``None`` (never a fabricated 1.0).
    """
    if len(member_idx) == 1:
        return member_idx[0], None
    best_idx = member_idx[0]
    best_mean = -1.0
    for i in member_idx:  # ascending index == ascending id -> strict '>' keeps the lexicographic winner
        mean_sim = float(np.mean([sim[i, j] for j in member_idx if j != i]))
        if mean_sim > best_mean:
            best_idx, best_mean = i, mean_sim
    return best_idx, best_mean


def kind_label(member_sources: Iterable[str], kind_number: int) -> tuple[str, dict[str, float]]:
    """Human-readable kind label from construct prevalence over the kind's members, plus the prevalences.

    A construct is CHARACTERISTIC of the kind when a strict majority (> 0.5) of members reference it —
    strict, so a 2-member kind where each member has a different construct is not labelled with both.
    The label joins the characteristic constructs' fragments by descending prevalence (ties break in
    :data:`CONSTRUCTS` order), capped at :data:`_LABEL_MAX_FRAGMENTS`. Honest fallbacks: "plain return"
    when NO member references ANY construct; ``"unlabelled kind N"`` when constructs appear but none
    reaches a majority — never a fabricated family name.
    """
    prevalences = [construct_prevalence(s) for s in member_sources]
    n = len(prevalences)
    if n == 0:  # defensive: callers never pass an empty kind
        return f"unlabelled kind {kind_number}", {}
    fractions = {
        name: sum(1 for p in prevalences if p[name]) / n for name, _pat, _tail in CONSTRUCTS
    }
    dominant = [
        (name, frac)
        for name, frac in fractions.items()  # dict preserves CONSTRUCTS order -> deterministic ties
        if frac > 0.5
    ]
    if not dominant:
        if max(fractions.values()) == 0.0:
            return "plain return", fractions
        return f"unlabelled kind {kind_number}", fractions
    dominant.sort(key=lambda item: (-item[1], [c[0] for c in CONSTRUCTS].index(item[0])))
    fragments = [_LABEL_FRAGMENTS.get(name, name) for name, _frac in dominant]
    label = " + ".join(fragments[:_LABEL_MAX_FRAGMENTS])
    if len(fragments) > _LABEL_MAX_FRAGMENTS:
        label += f" (+{len(fragments) - _LABEL_MAX_FRAGMENTS} more)"
    return label, fractions


# --------------------------------------------------------------------------- #
# the taxonomy                                                                 #
# --------------------------------------------------------------------------- #
def induce_taxonomy(
    sources: Mapping[str, str], *, depth: int = 4, sim_threshold: float = 0.6
) -> dict[str, Any]:
    """Induce the program-KIND taxonomy over a ``{candidate_id: reward_source}`` corpus.

    Pipeline: canonical depth-``depth`` AST shape-sets per program (identifier/literal-invariant) ->
    pairwise Jaccard -> connected components at ``>= sim_threshold`` -> one kind per component, each with
    a medoid exemplar and a construct-prevalence label (:func:`kind_label`). Unparseable and empty
    sources are EXCLUDED and counted, never clustered (see the module docstring).

    Returns
    -------
    dict
        ``{"status", "depth", "sim_threshold", "n_sources", "n_programs", "n_unparseable",
        "unparseable_ids", "n_kinds", "n_singletons", "kinds": [{kind_id, label, size, members, medoid,
        medoid_mean_similarity, mean_within_similarity, construct_prevalence}, ...]}``; kinds are ordered
        largest-first. ``status="no_data"`` when no source is parseable. Fully deterministic.
    """
    ids, shapes, excluded = _split_signatures(sources, depth)
    base = {
        "depth": int(depth),
        "sim_threshold": float(sim_threshold),
        "n_sources": len(sources),
        "n_programs": len(ids),
        "n_unparseable": len(excluded),
        "unparseable_ids": excluded,
    }
    if not ids:
        return {
            "status": "no_data",
            "reason": "no parseable, non-empty reward source to classify",
            **base,
            "n_kinds": 0,
            "n_singletons": 0,
            "kinds": [],
        }

    sim = _pairwise_similarity(shapes)
    idx_of = {cid: i for i, cid in enumerate(ids)}
    components = _components(ids, sim, float(sim_threshold))

    # Zero-pad kind ids to the corpus's own kind count (>= 2 digits) so ids SORT in kind order —
    # with a fixed 2-digit pad, a 157-kind corpus would list "kind_100" before "kind_82" in every
    # sorted composition table.
    width = max(2, len(str(len(components))))
    kinds: list[dict[str, Any]] = []
    for number, members in enumerate(components, start=1):
        member_idx = [idx_of[m] for m in members]
        med_i, med_mean = _medoid(member_idx, sim)
        # Mean similarity over all unordered member pairs — how tight the kind is (None for singletons).
        pair_sims = [sim[i, j] for i, j in combinations(member_idx, 2)]
        label, fractions = kind_label([str(sources[m]) for m in members], number)
        kinds.append({
            "kind_id": f"kind_{number:0{width}d}",
            "label": label,
            "size": len(members),
            "members": members,
            "medoid": ids[med_i],
            "medoid_mean_similarity": med_mean,
            "mean_within_similarity": float(np.mean(pair_sims)) if pair_sims else None,
            "construct_prevalence": fractions,
        })
    return {
        "status": "ok",
        **base,
        "n_kinds": len(kinds),
        "n_singletons": sum(1 for k in kinds if k["size"] == 1),
        "kinds": kinds,
    }


def pool_sources(sources_by_arm: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Flatten ``{arm: {candidate_id: source}}`` into one ``{"arm/candidate_id": source}`` corpus.

    The ``arm/`` prefix makes pooled ids unambiguous even when two arms reuse a candidate id (e.g. both
    emit ``g0-c0``), and lets :func:`taxonomy_by_arm` recover each member's arm from the id alone. Arm
    names are directory/config tokens and never contain ``/``.
    """
    pooled: dict[str, str] = {}
    for arm in sorted(str(a) for a in sources_by_arm):
        for cid in sorted(str(c) for c in sources_by_arm[arm]):
            pooled[f"{arm}/{cid}"] = str(sources_by_arm[arm][cid])
    return pooled


def _shannon_bits(counts: Iterable[int]) -> float:
    """Shannon entropy (bits) of a count vector — per-arm kind diversity (0.0 for a single kind).

    ``+ 0.0`` normalises the IEEE negative zero a single-kind arm would otherwise produce
    (``-(1.0 * log2(1.0)) == -0.0``), so reports render ``0.000`` rather than ``-0.000`` and the JSON
    is byte-stable.
    """
    values = [int(c) for c in counts if int(c) > 0]
    total = sum(values)
    if not total:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in values) + 0.0)


def taxonomy_by_arm(
    sources_by_arm: Mapping[str, Mapping[str, str]], **kw: Any
) -> dict[str, Any]:
    """ONE pooled taxonomy + the per-arm KIND composition — the arm-level scientific readout.

    Induces :func:`induce_taxonomy` over the POOLED corpus (kinds must be comparable across arms, so
    they are defined once, on everything), then reports per arm: kind counts + fractions, kind-diversity
    (Shannon entropy in bits over the arm's kind distribution), the number of kinds present, and the
    arm's ``n_unparseable``. Cross-arm: ``kind_arms`` (which arms each kind appears in) and
    ``kind_overlap`` (Jaccard of the two arms' kind SETS, per unordered arm pair — reusing
    :func:`~src.inference.reward_code_distance.jaccard`; two arms with no kinds at all score the trivial
    1.0 of the empty-set convention, disclosed rather than special-cased).

    ``**kw`` forwards ``depth`` / ``sim_threshold`` to :func:`induce_taxonomy`. Report-only and
    DISJOINT: no ``arm_a/arm_b/metric/level`` keys anywhere in the result.

    Returns
    -------
    dict
        ``{"status", "pooled": <induce_taxonomy result over "arm/candidate" ids>,
        "per_arm": {arm: {n_sources, n_programs, n_unparseable, kind_counts, kind_fractions,
        n_kinds_present, entropy_bits}}, "kind_arms": {kind_id: [arms]},
        "kind_overlap": {"armA|armB": float}}``.
    """
    pooled_sources = pool_sources(sources_by_arm)
    pooled = induce_taxonomy(pooled_sources, **kw)
    arms = sorted(str(a) for a in sources_by_arm)

    def _arm_of(member_id: str) -> str:
        return member_id.split("/", 1)[0]

    # Per-arm kind counts from the pooled kinds' members (members carry the arm/ prefix).
    kind_counts: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    kind_arms: dict[str, list[str]] = {}
    for kind in pooled.get("kinds", []):
        present: set[str] = set()
        for member in kind["members"]:
            arm = _arm_of(member)
            kind_counts[arm][kind["kind_id"]] = kind_counts[arm].get(kind["kind_id"], 0) + 1
            present.add(arm)
        kind_arms[kind["kind_id"]] = sorted(present)

    unparseable_by_arm: dict[str, int] = {arm: 0 for arm in arms}
    for excluded_id in pooled.get("unparseable_ids", []):
        unparseable_by_arm[_arm_of(excluded_id)] += 1

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        counts = dict(sorted(kind_counts[arm].items()))
        n_programs = sum(counts.values())
        per_arm[arm] = {
            "n_sources": len(sources_by_arm[arm]),
            "n_programs": n_programs,
            "n_unparseable": unparseable_by_arm[arm],
            "kind_counts": counts,
            "kind_fractions": {
                k: c / n_programs for k, c in counts.items()
            } if n_programs else {},
            "n_kinds_present": len(counts),
            # entropy of an arm with NO classified program is undefined, not 0.0 -> None (honest).
            "entropy_bits": _shannon_bits(counts.values()) if n_programs else None,
        }

    kind_overlap = {
        f"{a}|{b}": jaccard(
            frozenset(per_arm[a]["kind_counts"]), frozenset(per_arm[b]["kind_counts"])
        )
        for a, b in combinations(arms, 2)
    }
    return {
        "status": pooled["status"],
        "pooled": pooled,
        "per_arm": per_arm,
        "kind_arms": kind_arms,
        "kind_overlap": kind_overlap,
    }


# --------------------------------------------------------------------------- #
# threshold sensitivity (is the taxonomy a threshold artifact?)                #
# --------------------------------------------------------------------------- #
def rand_index(assign_a: Mapping[str, str], assign_b: Mapping[str, str]) -> float:
    """Pair-counting Rand index between two partitions of the SAME id set, in ``[0, 1]``.

    ``(agreements) / C(n, 2)`` where a pair agrees when both partitions place it together or both place
    it apart (Rand 1971). Implemented directly (deterministic, O(n^2) over a few hundred programs — no
    sklearn). The UNadjusted index: we report it as raw pair-agreement between adjacent thresholds, not
    as a chance-corrected statistic. Raises on mismatched id sets or fewer than 2 ids (no pairs).
    """
    ids = sorted(assign_a)
    if sorted(assign_b) != ids:
        raise ValueError("rand_index: the two partitions must cover the identical id set")
    n_pairs = len(ids) * (len(ids) - 1) // 2
    if n_pairs == 0:
        raise ValueError("rand_index: need at least 2 ids to form a pair")
    agreements = 0
    for x, y in combinations(ids, 2):
        together_a = assign_a[x] == assign_a[y]
        together_b = assign_b[x] == assign_b[y]
        if together_a == together_b:
            agreements += 1
    return agreements / n_pairs


def taxonomy_threshold_sensitivity(
    sources: Mapping[str, str],
    *,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
    depth: int = 4,
) -> dict[str, Any]:
    """Re-cluster the corpus at several thresholds; report ``n_kinds`` + adjacent-threshold stability.

    Shape-sets and the similarity matrix are computed ONCE and re-thresholded, so the sweep is cheap and
    every threshold sees the identical parseable id set (making the partitions directly comparable).
    Because raising the threshold only removes edges, components only split: ``n_kinds`` is
    non-decreasing across the (sorted) thresholds by construction. Stability between adjacent thresholds
    is the pair-counting :func:`rand_index` — high values mean the induced kinds are not an artifact of
    the specific cut. Report-only, deterministic.

    Returns
    -------
    dict
        ``{"status", "depth", "n_programs", "n_unparseable", "thresholds",
        "by_threshold": [{threshold, n_kinds, n_singletons}], "adjacent_stability":
        [{pair, rand_index}]}``; ``rand_index`` is ``None`` when fewer than 2 programs exist (no pairs).
    """
    ids, shapes, excluded = _split_signatures(sources, depth)
    ordered = sorted(float(t) for t in thresholds)
    if not ids:
        return {
            "status": "no_data",
            "reason": "no parseable, non-empty reward source to classify",
            "depth": int(depth),
            "n_programs": 0,
            "n_unparseable": len(excluded),
            "thresholds": ordered,
            "by_threshold": [],
            "adjacent_stability": [],
        }
    sim = _pairwise_similarity(shapes)
    partitions: list[dict[str, str]] = []
    by_threshold: list[dict[str, Any]] = []
    for t in ordered:
        components = _components(ids, sim, t)
        partitions.append({
            member: f"k{number}" for number, comp in enumerate(components) for member in comp
        })
        by_threshold.append({
            "threshold": t,
            "n_kinds": len(components),
            "n_singletons": sum(1 for c in components if len(c) == 1),
        })
    adjacent = [
        {
            "pair": f"{lo['threshold']:g}->{hi['threshold']:g}",
            "rand_index": rand_index(pa, pb) if len(ids) >= 2 else None,
        }
        for (lo, pa), (hi, pb) in zip(
            zip(by_threshold, partitions), zip(by_threshold[1:], partitions[1:])
        )
    ]
    return {
        "status": "ok",
        "depth": int(depth),
        "n_programs": len(ids),
        "n_unparseable": len(excluded),
        "thresholds": ordered,
        "by_threshold": by_threshold,
        "adjacent_stability": adjacent,
    }
