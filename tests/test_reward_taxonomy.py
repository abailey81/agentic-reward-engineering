"""Tests for the reward-program taxonomy (src/inference/reward_taxonomy.py) + its analyze() wiring.

The instrument CH7 deferred ("a true taxonomy of program kinds ... left to future work"), delivered as a
report-only, DISJOINT block. Covered here:

  1. determinism — identical input (any dict insertion order) -> byte-identical output;
  2. threshold monotonicity — raising the similarity threshold only removes graph edges, so components
     only split: ``n_kinds`` is non-decreasing (asserted through the sensitivity sweep AND direct calls);
  3. P7c exclusion — unparseable AND empty/comment-only sources are EXCLUDED + counted, never clustered
     (``jaccard(empty, empty) == 1.0`` would otherwise glue them into one fake kind — mirror of
     tests/test_contamination_ood.py::test_named_vs_blinded_structural_excludes_unparseable_pairs);
  4. per-arm composition arithmetic on hand-built sources (2 cvar + 1 sharpe across 2 arms -> known
     counts / fractions / entropy / overlap);
  5. medoid + label determinism (lexicographic tie-break; majority-construct labels with the honest
     "plain return" / "unlabelled kind N" fallbacks);
  6. the pair-counting Rand index (known small-case values);
  7. analyze() wiring — ``out["reward_taxonomy"]`` fires on a real disk archive, carries NO
     ``arm_a/arm_b/metric/level`` family-tuple keys, degrades to no_data on a source-free archive, and
     the renderer lands in write_report (mirrors tests/test_analyze_mechanism_wiring.py).

Every structural premise (which synthetic pair is/isn't similar at the tested threshold) is ASSERTED via
``structural_similarity`` rather than assumed, so the tests cannot rot silently if the shape signature
changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402
import build_taxonomy as BT  # noqa: E402

from src.inference.reward_code_distance import structural_similarity  # noqa: E402
from src.inference.reward_taxonomy import (  # noqa: E402
    induce_taxonomy,
    kind_label,
    pool_sources,
    rand_index,
    taxonomy_by_arm,
    taxonomy_threshold_sensitivity,
)
from src.io.results import write_run  # noqa: E402

# --------------------------------------------------------------------------- #
# synthetic corpus — construct content verified by hand against CONSTRUCTS     #
# --------------------------------------------------------------------------- #
#: A structurally rich cvar-shaped program: probes hit ``cvar`` (bare name) + ``quantile_tail``
#: (np.percentile) and NOTHING else (no std/vol/sharpe/drawdown/turnover tokens).
_CVAR_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    import numpy as np\n"
    "    window = info.get('window', [])\n"
    "    window.append(float(port_ret))\n"
    "    arr = np.asarray(window, dtype=float)\n"
    "    q = np.percentile(arr, 5)\n"
    "    cvar = arr[arr <= q].mean()\n"
    "    if arr.size > 60:\n"
    "        window.pop(0)\n"
    "    return float(port_ret) + 0.5 * cvar\n"
)
#: The same program with ONE extra statement — high (but < 1.0) structural similarity to _CVAR_SRC.
_CVAR_VARIANT_SRC = _CVAR_SRC.replace(
    "    q = np.percentile(arr, 5)\n",
    "    q = np.percentile(arr, 5)\n    q_upper = np.percentile(arr, 10)\n",
)
#: A tiny sharpe-shaped program: probes hit ``online_sharpe`` (bare "sharpe") and nothing else.
_SHARPE_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    sharpe = float(port_ret) * 2.0\n"
    "    return sharpe\n"
)
#: SyntaxError source — canonical_shapes() == frozenset() (the P7c case).
_BAD_SRC = "def reward(r:\n    return"


def _assert_premises() -> tuple[float, float]:
    """Assert the structural premises the clustering tests rely on; return the two key similarities."""
    sim_identical = structural_similarity(_CVAR_SRC, _CVAR_SRC)
    sim_cross = structural_similarity(_CVAR_SRC, _SHARPE_SRC)
    assert sim_identical == 1.0
    assert sim_cross < 0.6, f"premise broken: cvar-vs-sharpe similarity {sim_cross:.3f} >= 0.6"
    return sim_identical, sim_cross


# --------------------------------------------------------------------------- #
# 1 — determinism                                                              #
# --------------------------------------------------------------------------- #
def test_induce_taxonomy_deterministic_and_insertion_order_independent() -> None:
    forward = {"a": _CVAR_SRC, "b": _SHARPE_SRC, "c": _CVAR_SRC, "z": _BAD_SRC}
    backward = {"z": _BAD_SRC, "c": _CVAR_SRC, "b": _SHARPE_SRC, "a": _CVAR_SRC}
    out1 = induce_taxonomy(forward)
    out2 = induce_taxonomy(backward)
    assert out1 == out2
    # byte-identical serialisation on re-run (the report artifact contract)
    assert json.dumps(out1, sort_keys=True) == json.dumps(induce_taxonomy(forward), sort_keys=True)


def test_taxonomy_by_arm_deterministic() -> None:
    arms = {"beta": {"b1": _CVAR_SRC}, "alpha": {"a2": _SHARPE_SRC, "a1": _CVAR_SRC}}
    out1 = taxonomy_by_arm(arms)
    out2 = taxonomy_by_arm({"alpha": {"a1": _CVAR_SRC, "a2": _SHARPE_SRC}, "beta": {"b1": _CVAR_SRC}})
    assert out1 == out2
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


# --------------------------------------------------------------------------- #
# 2 — threshold monotonicity (higher threshold -> >= n_kinds)                  #
# --------------------------------------------------------------------------- #
def test_threshold_monotonicity_n_kinds_non_decreasing() -> None:
    _assert_premises()
    sources = {"c1": _CVAR_SRC, "c2": _CVAR_SRC, "c3": _CVAR_VARIANT_SRC, "s1": _SHARPE_SRC}
    # premise: every pair is similar above 0.1 (all are Python reward functions sharing shallow shapes),
    # so the lowest threshold yields ONE kind and the sweep is non-trivial end to end.
    ids = sorted(sources)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            assert structural_similarity(sources[x], sources[y]) >= 0.1

    thresholds = (0.1, 0.6, 0.8, 0.99)
    n_kinds = [induce_taxonomy(sources, sim_threshold=t)["n_kinds"] for t in thresholds]
    assert n_kinds == sorted(n_kinds), f"n_kinds must be non-decreasing in threshold: {n_kinds}"
    assert n_kinds[0] == 1  # everything joined at 0.1
    # at 0.99 only the byte-identical pair (similarity exactly 1.0) can stay joined -> 3 kinds
    assert n_kinds[-1] == 3

    sweep = taxonomy_threshold_sensitivity(sources, thresholds=thresholds)
    assert sweep["status"] == "ok"
    assert [row["n_kinds"] for row in sweep["by_threshold"]] == n_kinds
    for row in sweep["adjacent_stability"]:
        assert row["rand_index"] is not None and 0.0 <= row["rand_index"] <= 1.0


# --------------------------------------------------------------------------- #
# 3 — P7c: unparseable/empty sources are EXCLUDED + counted, never clustered   #
# --------------------------------------------------------------------------- #
def test_unparseable_and_empty_sources_excluded_never_clustered() -> None:
    # two unparseable + one empty + one comment-only source alongside two real programs: were the empty
    # signatures allowed in, jaccard(empty, empty) == 1.0 would weld them into one fake "kind".
    sources = {
        "good1": _CVAR_SRC,
        "good2": _SHARPE_SRC,
        "bad1": _BAD_SRC,
        "bad2": _BAD_SRC,
        "empty": "",
        "comment_only": "# just a comment\n",
    }
    out = induce_taxonomy(sources)
    assert out["status"] == "ok"
    assert out["n_unparseable"] == 4
    assert out["unparseable_ids"] == ["bad1", "bad2", "comment_only", "empty"]
    assert out["n_programs"] == 2
    clustered = {m for k in out["kinds"] for m in k["members"]}
    assert clustered == {"good1", "good2"}  # no excluded id ever appears in a kind

    # ALL sources unparseable/empty -> honest no_data, never a spuriously coherent taxonomy (P7c mirror).
    out_bad = induce_taxonomy({"bad1": _BAD_SRC, "bad2": _BAD_SRC, "empty": ""})
    assert out_bad["status"] == "no_data"
    assert out_bad["n_unparseable"] == 3
    assert out_bad["n_kinds"] == 0 and out_bad["kinds"] == []


# --------------------------------------------------------------------------- #
# 4 — per-arm composition arithmetic (2 cvar + 1 sharpe across 2 arms)         #
# --------------------------------------------------------------------------- #
def test_per_arm_composition_counts_entropy_overlap() -> None:
    _assert_premises()
    arms = {
        "alpha": {"a-c1": _CVAR_SRC, "a-c2": _SHARPE_SRC},
        "beta": {"b-c1": _CVAR_SRC},
    }
    out = taxonomy_by_arm(arms, sim_threshold=0.6)
    assert out["status"] == "ok"
    pooled = out["pooled"]
    assert pooled["n_programs"] == 3 and pooled["n_kinds"] == 2

    # kind_01 = the larger (cvar) kind spanning both arms; kind_02 = the sharpe singleton in alpha.
    k1, k2 = pooled["kinds"]
    assert k1["kind_id"] == "kind_01" and k1["size"] == 2
    assert sorted(k1["members"]) == ["alpha/a-c1", "beta/b-c1"]
    assert k2["kind_id"] == "kind_02" and k2["members"] == ["alpha/a-c2"]
    assert out["kind_arms"] == {"kind_01": ["alpha", "beta"], "kind_02": ["alpha"]}

    alpha = out["per_arm"]["alpha"]
    assert alpha["kind_counts"] == {"kind_01": 1, "kind_02": 1}
    assert alpha["kind_fractions"] == {"kind_01": 0.5, "kind_02": 0.5}
    assert alpha["entropy_bits"] == pytest.approx(1.0)  # two equiprobable kinds = exactly 1 bit
    assert alpha["n_kinds_present"] == 2 and alpha["n_unparseable"] == 0

    beta = out["per_arm"]["beta"]
    assert beta["kind_counts"] == {"kind_01": 1}
    assert beta["entropy_bits"] == pytest.approx(0.0)  # a single kind carries zero diversity

    # kind-set overlap alpha|beta = |{kind_01}| / |{kind_01, kind_02}| = 0.5
    assert out["kind_overlap"] == {"alpha|beta": pytest.approx(0.5)}


def test_per_arm_unparseable_counted_per_arm() -> None:
    arms = {
        "alpha": {"a-c1": _CVAR_SRC, "a-bad": _BAD_SRC},
        "beta": {"b-c1": _SHARPE_SRC},
    }
    out = taxonomy_by_arm(arms)
    assert out["per_arm"]["alpha"]["n_unparseable"] == 1
    assert out["per_arm"]["alpha"]["n_programs"] == 1
    assert out["per_arm"]["beta"]["n_unparseable"] == 0
    assert out["pooled"]["unparseable_ids"] == ["alpha/a-bad"]


# --------------------------------------------------------------------------- #
# 5 — medoid + label determinism                                               #
# --------------------------------------------------------------------------- #
def test_medoid_deterministic_lexicographic_tie_break() -> None:
    # m1 == m2 (identical text) and m3 a close variant: m1/m2 tie on mean within-kind similarity
    # ((1 + s)/2 each vs s for m3) -> the medoid must be the lexicographically FIRST of the tie, m1.
    s = structural_similarity(_CVAR_SRC, _CVAR_VARIANT_SRC)
    assert 0.2 <= s < 1.0, f"premise broken: variant similarity {s:.3f}"
    out = induce_taxonomy(
        {"m1": _CVAR_SRC, "m2": _CVAR_SRC, "m3": _CVAR_VARIANT_SRC}, sim_threshold=0.2
    )
    assert out["n_kinds"] == 1
    kind = out["kinds"][0]
    assert kind["medoid"] == "m1"
    assert kind["medoid_mean_similarity"] == pytest.approx((1.0 + s) / 2.0)
    assert kind["mean_within_similarity"] == pytest.approx((1.0 + s + s) / 3.0)


def test_kind_labels_majority_and_honest_fallbacks() -> None:
    # majority construct combination, fragments in descending-prevalence order
    label, fractions = kind_label([_CVAR_SRC, _CVAR_SRC], 1)
    assert label == "cvar-penalized + quantile-tail-shaped"
    assert fractions["cvar"] == 1.0 and fractions["quantile_tail"] == 1.0
    assert kind_label([_SHARPE_SRC], 1)[0] == "sharpe-ratio"

    # "plain return": NO member references ANY construct
    plain = "def reward(w, r, pw, port_ret, info):\n    return float(port_ret)\n"
    label_plain, fractions_plain = kind_label([plain, plain], 2)
    assert label_plain == "plain return"
    assert max(fractions_plain.values()) == 0.0

    # honest fallback: constructs present but none reaches a strict majority (0.5 each)
    label_mixed, _ = kind_label([_CVAR_SRC, plain], 3)
    assert label_mixed == "unlabelled kind 3"


def test_singleton_kind_has_no_fabricated_similarity() -> None:
    out = induce_taxonomy({"only": _CVAR_SRC})
    assert out["n_kinds"] == 1 and out["n_singletons"] == 1
    kind = out["kinds"][0]
    assert kind["medoid"] == "only"
    assert kind["medoid_mean_similarity"] is None  # never a fabricated 1.0
    assert kind["mean_within_similarity"] is None


# --------------------------------------------------------------------------- #
# 6 — pair-counting Rand index                                                 #
# --------------------------------------------------------------------------- #
def test_rand_index_known_values_and_guards() -> None:
    same = {"a": "k1", "b": "k1", "c": "k2"}
    assert rand_index(same, dict(same)) == 1.0
    # {a,b}{c} vs {a}{b}{c}: pair (a,b) disagrees, (a,c) and (b,c) agree -> 2/3
    split = {"a": "x", "b": "y", "c": "z"}
    assert rand_index(same, split) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        rand_index({"a": "k1"}, {"b": "k1"})  # mismatched id sets
    with pytest.raises(ValueError):
        rand_index({"a": "k1"}, {"a": "k1"})  # < 2 ids -> no pairs


# --------------------------------------------------------------------------- #
# 7 — renderer + analyze() wiring (disk-backed, mirrors test_analyze_mechanism_wiring)
# --------------------------------------------------------------------------- #
def _search_record(arm: str, i: int, source: str) -> dict:
    """A minimal valid search-candidate record carrying an authored reward source."""
    return {
        "run_id": f"{arm}-tax-c{i:02d}",
        "arm": arm,
        "seed": 0,
        "fold": 0,
        "candidate_id": f"c{i:02d}",
        "generation": i // 4,
        "reward_source_hash": f"taxhash{i:02d}",
        "reward_source": source,
        "feedback_block": "",
        "metrics": {"val_fitness": 0.1 + 0.001 * i, "val_returns": [0.001] * 30},
        "wall_clock": 1.0,
        "env_fingerprint": "test",
    }


def _sourceless_record(arm: str, seed: int) -> dict:
    """A frozen-winner-style TEST record with NO reward_source (the records-only degrade case)."""
    return {
        "run_id": f"{arm}-tax-s{seed}",
        "arm": arm,
        "seed": seed,
        "fold": 0,
        "candidate_id": f"{arm}-winner",
        "generation": 0,
        "reward_source_hash": "winnerhash",
        "feedback_block": "",
        "metrics": {"val_fitness": 0.2, "test_returns": [0.001] * 30},
        "wall_clock": 1.0,
        "env_fingerprint": "test",
    }


def test_analyze_emits_reward_taxonomy_disjoint_and_never_gates(tmp_path: Path) -> None:
    _assert_premises()
    for i, src in enumerate((_CVAR_SRC, _CVAR_VARIANT_SRC, _SHARPE_SRC, _BAD_SRC)):
        write_run(_search_record("distributional", i, src), tmp_path / "search" / "distributional")
    for i, src in enumerate((_CVAR_SRC, _SHARPE_SRC)):
        write_run(_search_record("scalar", i, src), tmp_path / "search" / "scalar")

    out = AC.analyze(tmp_path)
    tax = out["reward_taxonomy"]
    assert tax["status"] == "ok", tax
    assert tax["pooled"]["n_programs"] == 5
    assert tax["pooled"]["n_unparseable"] == 1  # the _BAD_SRC candidate, excluded + counted
    assert set(tax["per_arm"]) == {"distributional", "scalar"}
    assert tax["sensitivity"]["status"] == "ok"

    # DISJOINT: no family-tuple keys anywhere at the block's top level; the frozen headline untouched.
    assert not any(k in tax for k in ("arm_a", "arm_b", "metric", "level"))
    assert "h2" in out

    md = AC.write_report(out, tmp_path / "report").read_text(encoding="utf-8")
    assert "Reward-program taxonomy" in md
    assert "Per-arm kind composition" in md


def test_analyze_reward_taxonomy_degrades_no_data_without_sources(tmp_path: Path) -> None:
    for seed in range(2):
        write_run(_sourceless_record("distributional", seed), tmp_path / "test" / "distributional")
    out = AC.analyze(tmp_path)
    tax = out["reward_taxonomy"]
    assert tax["status"] == "no_data"
    assert tax["n_missing_source"] == 2
    # the n/a renderer branch still emits a section (never crashes the report)
    assert "Reward-program taxonomy" in AC.reward_taxonomy_markdown(tax)


def test_renderer_smoke_tables_and_na_branch() -> None:
    arms = {"alpha": {"a-c1": _CVAR_SRC, "a-c2": _SHARPE_SRC}, "beta": {"b-c1": _CVAR_SRC}}
    result = taxonomy_by_arm(arms)
    result["sensitivity"] = taxonomy_threshold_sensitivity(pool_sources(arms))
    md = AC.reward_taxonomy_markdown(result)
    assert "| kind_01 | 2 |" in md
    assert "cvar-penalized + quantile-tail-shaped" in md
    assert "entropy (bits)" in md and "alpha|beta" in md
    assert "Threshold sensitivity" in md and "Rand index" in md
    assert AC.reward_taxonomy_markdown({"status": "no_data", "reason": "x"}).startswith(
        "## Reward-program taxonomy"
    )


# --------------------------------------------------------------------------- #
# 8 — the standalone CLI collector/builder (prototype-shaped archive walk)     #
# --------------------------------------------------------------------------- #
def test_build_taxonomy_collects_arm_candidate_reward_files(tmp_path: Path) -> None:
    for arm, sources in (("alpha", (_CVAR_SRC, _SHARPE_SRC)), ("beta", (_CVAR_SRC,))):
        for i, src in enumerate(sources):
            cand = tmp_path / arm / f"{arm}-g0-c{i}"
            cand.mkdir(parents=True)
            (cand / "reward.py").write_text(src, encoding="utf-8")
    (tmp_path / "alpha" / "COMPLETE").write_text("", encoding="utf-8")  # loose file: ignored
    (tmp_path / "tables").mkdir()  # no reward.py-bearing children: not an arm

    collected = BT.collect_sources(tmp_path)
    assert set(collected) == {"alpha", "beta"}
    assert set(collected["alpha"]) == {"alpha-g0-c0", "alpha-g0-c1"}

    built = BT.build(tmp_path)
    assert built["status"] == "ok"
    assert built["pooled"]["n_programs"] == 3
    assert built["sensitivity"]["status"] == "ok"
    # an empty root degrades honestly
    assert BT.build(tmp_path / "does_not_exist")["status"] == "no_data"
