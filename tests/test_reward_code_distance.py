"""Behaviour tests for the report-only reward-code structural-similarity module (R68).

Pins the load-bearing properties: identifier/literal INVARIANCE (the construct-validity point — cosmetic
token differences must not register), structural sensitivity, within-vs-across clustering with a
deterministic permutation test, determinism, and graceful handling of unparseable / insufficient input.
Pure stdlib `ast` + numpy; import-light (no torch).
"""

from __future__ import annotations

from src.inference.reward_code_distance import (
    canonical_shapes,
    jaccard,
    reward_code_structure_report,
    structural_similarity,
)

_HDR = "def reward(w, r, pw, pr, info):\n"


def test_identifier_invariance() -> None:
    """Two rewards differing ONLY in variable names are structurally identical (similarity 1.0)."""
    a = _HDR + "    x = r.mean()\n    return float(x), {}, None"
    b = _HDR + "    y = r.mean()\n    return float(y), {}, None"
    assert structural_similarity(a, b) == 1.0


def test_literal_value_invariance() -> None:
    """Two rewards differing ONLY in a constant value (0.05 vs 0.10) are structurally identical — the
    construct-validity guard: a cosmetic literal change must not look like structural conditioning."""
    a = _HDR + "    return float(r.mean() * 0.05), {}, None"
    b = _HDR + "    return float(r.mean() * 0.10), {}, None"
    assert structural_similarity(a, b) == 1.0


def test_structural_difference_is_detected() -> None:
    """A genuinely different program structure (an extra quantile computation) scores < 1.0."""
    a = _HDR + "    return float(r.mean()), {}, None"
    b = _HDR + "    q = quantile(r, 0.05)\n    return float(r.mean() - q), {}, None"
    assert structural_similarity(a, b) < 1.0


def test_unparseable_source_is_safe() -> None:
    """Malformed source contributes an empty signature and never crashes."""
    assert canonical_shapes("def reward(:::") == frozenset()
    assert structural_similarity("def reward(:::", _HDR + "    return float(0), {}, None") == 0.0


def test_jaccard_bounds() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3


def test_within_exceeds_across_when_conditions_cluster() -> None:
    """When two feedback conditions author structurally distinct code, within-condition similarity exceeds
    across-condition similarity (difference > 0) and the permutation p-value is small — the signature of
    the LLM routing the fed signal into program STRUCTURE."""
    tail = [
        _HDR + "    q = quantile(r, 0.05)\n    return float(q), {}, None",
        _HDR + "    z = quantile(r, 0.10)\n    return float(z), {}, None",
    ]
    mean = [
        _HDR + "    return float(r.mean()), {}, None",
        _HDR + "    return float(r.mean() + 1), {}, None",
    ]
    rep = reward_code_structure_report({"tail": tail, "mean": mean}, n_perm=500, seed=0)
    assert rep["status"] == "ok"
    assert rep["difference"] > 0.0
    assert 0.0 < rep["p_value"] <= 1.0
    assert rep["n_within_pairs"] == 2 and rep["n_across_pairs"] == 4


def test_report_is_deterministic() -> None:
    groups = {
        "a": [_HDR + "    return float(r.mean()), {}, None", _HDR + "    return float(r.sum()), {}, None"],
        "b": [_HDR + "    q = quantile(r, 0.05)\n    return float(q), {}, None"],
    }
    assert reward_code_structure_report(groups, n_perm=300, seed=1) == reward_code_structure_report(
        groups, n_perm=300, seed=1
    )


def test_insufficient_input() -> None:
    assert reward_code_structure_report({"a": [_HDR + "    return float(0), {}, None"]})["status"] == "insufficient"


def test_unparseable_sources_excluded_not_scored_as_similar() -> None:
    """P7c: two UNPARSEABLE rewards under the SAME condition must NOT inflate the within-condition mean via
    the empty-vs-empty Jaccard=1.0 convention. They are DROPPED with an n_unparseable count, leaving only the
    parseable sources to be scored."""
    good_tail = [
        _HDR + "    q = quantile(r, 0.05)\n    return float(q), {}, None",
        _HDR + "    z = quantile(r, 0.10)\n    return float(z), {}, None",
    ]
    # 'mean' condition mixes two malformed rewards (empty AST) with one valid one.
    mixed_mean = [
        "def reward(:::",                                    # unparseable
        "def reward(@@@",                                    # unparseable
        _HDR + "    return float(r.mean()), {}, None",       # valid
    ]
    rep = reward_code_structure_report({"tail": good_tail, "mean": mixed_mean}, n_perm=300, seed=0)
    assert rep["status"] == "ok"
    assert rep["n_unparseable"] == 2                          # both malformed 'mean' rewards excluded
    # only the 3 parseable sources survive: 2 tail (1 within pair) + 1 mean (0 within pairs) -> 1 within pair.
    assert rep["n_within_pairs"] == 1
    assert rep["per_condition"] == {"tail": 2, "mean": 1}     # the excluded pair does not appear


def test_all_unparseable_under_one_condition_does_not_fabricate_similarity() -> None:
    """If a whole condition is unparseable it contributes NO sources (n_unparseable counted); the report must
    fall through to insufficient/degenerate rather than reporting a spurious within-mean of 1.0."""
    rep = reward_code_structure_report(
        {"a": ["def reward(:::", "def reward(@@@"], "b": [_HDR + "    return float(r.mean()), {}, None"]},
        n_perm=100, seed=0,
    )
    # only 1 parseable source survives -> insufficient (< 2), with the 2 unparseable ones logged.
    assert rep["status"] == "insufficient"
    assert rep["n_unparseable"] == 2
    assert rep["n_sources"] == 1
