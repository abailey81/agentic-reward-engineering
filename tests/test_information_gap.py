"""Behaviour tests for the information-utilization gap (src/inference/information_gap.py) + its wiring.

PREREGISTRATION §2a micro-anchor (d). Deterministic (seeded bootstrap). Covered:

  1. KNOWN redundancy — fed tail vectors built as EXACT monotone functions of the fed scalar -> pooled
     redundancy ~ 1 (both R² and Spearman rank, on both the rendered and the parent-matched underlying
     channel); independent-noise vectors -> pooled redundancy near 0;
  2. the ``placebo_shuffled`` calibration floor — the candidate-seeded derangement destroys the
     label<->value linkage, so the floor's redundancy sits far below the intact arm's and the
     linkage-attributable difference is positive;
  3. the fed-block parser is validated AGAINST the real renderer (``src.feedback.schema.build_block``),
     including per-generation dedup (all siblings see the same block -> one observation) and the honest
     exclusion of scalar / scalar_cvar5 / placebo blocks;
  4. the utilization gap = GIVEN (non-redundant fed fraction) - USED (|SQ1 coefficient|), honest degrade
     when no responsiveness estimate is supplied;
  5. degradation paths (too few generations, absent floor) + determinism (seeded bootstrap);
  6. analyze() wiring — ``out["information_gap"]`` is a DISJOINT report-only block (no family-tuple
     keys) and its renderer lands in write_report (mirrors tests/test_analyze_mechanism_wiring.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402

from src.feedback.schema import build_block, shuffle_seed_from_id  # noqa: E402
from src.inference.information_gap import (  # noqa: E402
    _parse_fed_block,
    extract_fed_observations,
    information_gap,
)
from src.io.results import write_run  # noqa: E402

SEED = 20260702

#: Component slopes (distinct, mixed-sign) so a derangement of the values SCRAMBLES the per-label
#: association with the scalar (the floor mechanism) while the intact mapping stays perfectly monotone.
_SLOPES = {
    "cvar_05": -0.10,
    "cvar_10": -0.08,
    "cvar_25": 0.05,
    "cvar_01": -0.12,
    "left_tail_mass": 0.02,
    "robust_skew": -0.50,
}


def _tail_of(s: float) -> dict[str, float]:
    """The six-component tail vector as an EXACT linear function of the scalar s (redundancy == 1)."""
    return {fid: slope * s for fid, slope in _SLOPES.items()}


def _chain_records(
    arm: str,
    scalars: list[float],
    tails: list[dict[str, float]],
    *,
    n_siblings: int = 2,
) -> list[dict]:
    """A reflective chain: generation g's records are FED generation g-1's (scalar, tail) block.

    Sibling c0 of generation g ARCHIVES the (tail_stats, val_fitness) = (tails[g], scalars[g]) that the
    NEXT generation is fed, so the parent-matching underlying channel can recover the fed pair at full
    precision; later siblings archive PERTURBED metrics (as real sibling candidates do), keeping the
    parent match unique. ``n_siblings`` records per generation share one fed block (exercising the
    per-generation dedup).
    """
    records: list[dict] = []
    for g, (s, t) in enumerate(zip(scalars, tails)):
        for c in range(n_siblings):
            cid = f"{arm}-g{g}-c{c}"
            own_tail = {fid: v + 0.011 * c for fid, v in t.items()}  # c0 == the fed parent stats
            if g == 0:
                prompt = "Write a reward function."
            else:
                seed = shuffle_seed_from_id(cid) if arm == "placebo_shuffled" else None
                prompt = "Reflect on the previous candidate.\n" + build_block(
                    arm, scalars[g - 1], tails[g - 1], shuffle_seed=seed
                )
            records.append(
                {
                    "arm": arm,
                    "generation": g,
                    "candidate_id": cid,
                    "run_id": cid,
                    "prompt": prompt,
                    "feedback_block": "",
                    "metrics": {"val_fitness": s + 0.003 * c, "tail_stats": own_tail},
                }
            )
    return records


def _linear_chain(arm: str, n_gens: int = 10) -> list[dict]:
    scalars = [round(0.05 + 0.09 * g, 2) for g in range(n_gens)]  # distinct at 2-dp render precision
    return _chain_records(arm, scalars, [_tail_of(s) for s in scalars])


# --------------------------------------------------------------------------- #
# 1 — known redundancy: exact-function vectors ~ 1, independent noise ~ 0
# --------------------------------------------------------------------------- #
def test_exact_function_vectors_yield_redundancy_near_one() -> None:
    records = _linear_chain("distributional")
    out = information_gap(records, n_boot=500, rng=np.random.default_rng(1))
    assert out["status"] == "ok" and out["executed"] is True
    entry = out["arms"]["distributional"]
    assert entry["n_obs"] == 9  # gens 1..9, one distinct block each (siblings deduped)
    for channel in ("fed_rendered", "fed_underlying"):
        ch = entry["channels"][channel]
        assert ch["status"] == "ok", ch
        assert ch["scalar_degenerate"] is False
        assert ch["pooled"]["mean_r2"] > 0.95, channel
        assert ch["pooled"]["mean_rank_rho2"] > 0.95, channel
        assert ch["pooled"]["non_redundant_rank"] < 0.05, channel
    # underlying channel matched EVERY fed block to its unique parent at full precision
    assert entry["channels"]["fed_underlying"]["n_matched_parents"] == 9
    assert entry["n_parent_unmatched"] == 0 and entry["n_parent_ambiguous"] == 0
    assert entry["channels"]["fed_underlying"]["pooled"]["mean_r2"] > 0.999


def test_independent_noise_vectors_yield_redundancy_near_zero() -> None:
    rng = np.random.default_rng(SEED)
    n_gens = 14
    scalars = [round(float(x), 2) for x in rng.uniform(0.05, 0.95, n_gens)]
    tails = [
        {fid: float(v) for fid, v in zip(_SLOPES, rng.normal(-0.05, 0.03, 6))}
        for _ in range(n_gens)
    ]
    records = _chain_records("distributional", scalars, tails)
    out = information_gap(records, n_boot=500, rng=np.random.default_rng(2))
    ch = out["arms"]["distributional"]["channels"]["fed_underlying"]
    assert ch["status"] == "ok"
    assert ch["pooled"]["mean_r2"] < 0.35  # ~1/(n-1) sampling floor, not real redundancy
    assert ch["pooled"]["mean_rank_rho2"] < 0.35
    assert ch["pooled"]["non_redundant_rank"] > 0.65


# --------------------------------------------------------------------------- #
# 2 — the placebo_shuffled calibration floor
# --------------------------------------------------------------------------- #
def test_shuffled_floor_sits_below_intact_arm_and_comparison_fires() -> None:
    records = _linear_chain("distributional", n_gens=13) + _linear_chain(
        "placebo_shuffled", n_gens=13
    )
    out = information_gap(records, n_boot=500, rng=np.random.default_rng(3))
    dist = out["arms"]["distributional"]["channels"]["fed_rendered"]["pooled"]
    assert out["floor"]["executed"] is True
    floor = out["floor"]["fed_rendered"]["pooled"]
    # the derangement scrambles which slope each label carries -> redundancy collapses vs the intact 1.0
    assert floor["mean_rank_rho2"] < dist["mean_rank_rho2"] - 0.2
    fc = out["floor_comparison"]
    assert fc["executed"] is True
    assert fc["linkage_attributable_r2"] > 0.2
    assert fc["arm"] == "distributional" and fc["floor_arm"] == "placebo_shuffled"


def test_floor_absent_degrades_honestly() -> None:
    out = information_gap(_linear_chain("distributional"), n_boot=200, rng=np.random.default_rng(4))
    assert out["floor"]["executed"] is False
    assert out["floor_comparison"]["executed"] is False
    assert "placebo_shuffled" in out["floor_comparison"]["reason"]


# --------------------------------------------------------------------------- #
# 3 — parser vs the REAL renderer; dedup; non-vector arms excluded
# --------------------------------------------------------------------------- #
def test_parser_reads_back_exactly_what_build_block_rendered() -> None:
    tail = _tail_of(0.4)
    parsed = _parse_fed_block(build_block("distributional", 0.37, tail))
    assert parsed is not None
    assert parsed["scalar"] == 0.37
    # the parsed vector equals the tail values at the 3-dp render precision, in canonical field order
    expected = np.asarray([round(tail[fid], 3) for fid in _SLOPES])
    np.testing.assert_allclose(parsed["vector"], expected, atol=5e-4)


def test_parser_rejects_scalar_cvar5_and_placebo_blocks() -> None:
    tail = _tail_of(0.4)
    assert _parse_fed_block(build_block("scalar", 0.4, None)) is None
    assert _parse_fed_block(build_block("scalar_cvar5", 0.4, tail)) is None
    assert _parse_fed_block(build_block("placebo", 0.4, tail)) is None
    assert _parse_fed_block("") is None


def test_siblings_sharing_a_block_dedup_to_one_observation_per_generation() -> None:
    records = _linear_chain("distributional")  # n_siblings=2 share each generation's block
    observations, counters = extract_fed_observations(records, "distributional")
    assert len(observations) == 9
    assert counters["n_records_parsed"] == 18  # both siblings parsed, then deduped


def test_ambiguous_parent_is_counted_not_guessed() -> None:
    records = _linear_chain("distributional")
    # a SECOND g0 record with the SAME tail multiset -> the g1 fed block matches two parents
    twin = dict(records[0])
    twin["candidate_id"] = twin["run_id"] = "distributional-g0-twin"
    twin["metrics"] = {"val_fitness": 0.99, "tail_stats": dict(records[0]["metrics"]["tail_stats"])}
    observations, counters = extract_fed_observations(records + [twin], "distributional")
    assert counters["n_parent_ambiguous"] == 1
    gen1 = [o for o in observations if o["generation"] == 1]
    assert gen1 and gen1[0]["underlying_vector"] is None  # never guessed


# --------------------------------------------------------------------------- #
# 4 — the utilization gap
# --------------------------------------------------------------------------- #
def test_utilization_gap_given_minus_used() -> None:
    records = _linear_chain("distributional")
    resp = {"status": "ok", "coef": -0.13, "ci_low": -0.36, "ci_high": 0.10}
    out = information_gap(records, responsiveness=resp, n_boot=200, rng=np.random.default_rng(5))
    ug = out["utilization_gap"]
    assert ug["executed"] is True
    assert ug["channel"] == "fed_underlying"  # full precision preferred
    nr = out["arms"]["distributional"]["channels"]["fed_underlying"]["pooled"]["non_redundant_rank"]
    assert ug["non_redundant_fed"] == nr
    assert ug["responsiveness_abs_coef"] == 0.13
    assert abs(ug["gap"] - (nr - 0.13)) < 1e-12


def test_utilization_gap_degrades_without_responsiveness() -> None:
    records = _linear_chain("distributional")
    out = information_gap(records, n_boot=200, rng=np.random.default_rng(6))
    assert out["utilization_gap"]["executed"] is False
    bad = information_gap(
        records,
        responsiveness={"status": "no_data"},
        n_boot=200,
        rng=np.random.default_rng(6),
    )
    assert bad["utilization_gap"]["executed"] is False
    assert "no_data" in bad["utilization_gap"]["reason"]


# --------------------------------------------------------------------------- #
# 5 — degradation + determinism
# --------------------------------------------------------------------------- #
def test_too_few_generations_no_data() -> None:
    out = information_gap(_linear_chain("distributional", n_gens=3), n_boot=100)  # only 2 fed obs
    assert out["status"] == "no_data" and out["executed"] is False
    assert "fed six-vector observations" in out["reason"]


def test_degenerate_rendered_scalar_is_flagged_not_fabricated() -> None:
    # Every fed scalar quantises to the SAME rendered header value -> rendered redundancy 0 BY
    # CONSTRUCTION. Values updated for the `.6f` header (#87): under the old `.2f` this fixture used
    # 0.0004*(g+1), which is exactly the collapse #87 removed — those now render distinctly, and the
    # measured median archived fitness (0.000914) is no longer reported to the designer as "0.00".
    # Sub-resolution values keep the DEGENERATE condition this guard exists to detect.
    scalars = [1e-9 * (g + 1) for g in range(8)]  # all render as "0.000000"
    records = _chain_records("distributional", scalars, [_tail_of(0.1 + 0.1 * g) for g in range(8)])
    out = information_gap(records, n_boot=200, rng=np.random.default_rng(7))
    ch = out["arms"]["distributional"]["channels"]["fed_rendered"]
    assert ch["status"] == "ok"
    assert ch["scalar_degenerate"] is True
    assert ch["pooled"]["mean_r2"] == 0.0
    assert ch["pooled"]["non_redundant_r2"] == 1.0
    assert np.isnan(ch["pooled"]["mean_rank_rho2"])  # rank redundancy undefined, never invented
    # the UNDERLYING channel still resolves the full-precision scalar (varies across parents)
    assert out["arms"]["distributional"]["channels"]["fed_underlying"]["scalar_degenerate"] is False


def test_default_rng_is_deterministic() -> None:
    records = _linear_chain("distributional")
    a = information_gap(records, n_boot=300)
    b = information_gap(records, n_boot=300)
    pa = a["arms"]["distributional"]["channels"]["fed_rendered"]["pooled"]
    pb = b["arms"]["distributional"]["channels"]["fed_rendered"]["pooled"]
    assert pa == pb


# --------------------------------------------------------------------------- #
# 6 — analyze() wiring: DISJOINT report-only block + renderer in write_report
# --------------------------------------------------------------------------- #
def _archive(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    for rec in _linear_chain("distributional"):
        rec.update(
            seed=0,
            fold=0,
            reward_source_hash=f"h{rec['candidate_id']}",
            reward_source="def reward(r):\n    return cvar\n",
            wall_clock=1.0,
            env_fingerprint="test",
        )
        rec["metrics"]["val_returns"] = [float(x) for x in rng.normal(0.0005, 0.01, 120)]
        write_run(rec, tmp_path / "search" / "distributional")


def test_analyze_wires_information_gap_disjoint_and_renders(tmp_path: Path) -> None:
    _archive(tmp_path)
    out = AC.analyze(tmp_path)
    ig = out["information_gap"]
    assert ig["status"] in {"ok", "no_data", "error"}
    assert ig["status"] == "ok", ig
    # DISJOINT: no frozen-family tuple keys; the headline is untouched
    assert not any(k in ig for k in ("arm_a", "arm_b", "metric", "level"))
    assert "h2" in out
    # the SQ1 estimate is PASSED IN (never recomputed): with responsiveness ok the gap block fires
    if out["responsiveness"].get("status") == "ok":
        assert ig["utilization_gap"]["executed"] is True
    md = AC.write_report(out, tmp_path / "report").read_text(encoding="utf-8")
    assert "Information-utilization gap" in md
    assert "micro-anchor (d)" in md
