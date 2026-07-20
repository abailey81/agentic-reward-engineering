"""Behaviour tests for the validation-headroom (oracle-selection) bound (src/inference/headroom.py).

PREREGISTRATION §2a micro-anchor (e). Deterministic (seeded bootstrap). Covered:

  1. KNOWN headroom — a planted candidate with a strictly better validation CVaR than the val_fitness
     winner -> positive gap equal to the hand-computed frontier-minus-achieved, correct oracle/selected
     ids; an arm whose selection IS the oracle -> gap exactly 0;
  2. the DSR leg REUSES the canonical utilities (``deflated_sharpe_ratio`` + ``_sample_moments``): the
     achieved value equals the utility recomputed by hand with the same n_trials / var_sr convention;
  3. VALIDATION-only fail-safe — frozen markers and records carrying test_returns are excluded even
     when passed in, and never shift the result;
  4. degradation paths (missing vectors/fitness, below the candidate floor) + determinism;
  5. analyze() wiring — ``out["validation_headroom"]`` is a DISJOINT report-only block and its renderer
     lands in write_report (mirrors tests/test_analyze_mechanism_wiring.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402

from src.inference.bootstrap import cvar  # noqa: E402
from src.inference.deflated_sharpe import _sample_moments, deflated_sharpe_ratio  # noqa: E402
from src.inference.headroom import validation_headroom  # noqa: E402
from src.io.results import write_run  # noqa: E402

SEED = 20260702


def _candidate(
    arm: str,
    cid: str,
    fitness: float,
    returns: np.ndarray,
    *,
    generation: int = 0,
    extra_metrics: dict | None = None,
) -> dict:
    metrics = {"val_fitness": float(fitness), "val_returns": [float(x) for x in returns]}
    metrics.update(extra_metrics or {})
    return {
        "arm": arm,
        "generation": generation,
        "candidate_id": cid,
        "run_id": cid,
        "feedback_block": "",
        "metrics": metrics,
    }


def _base_returns(rng: np.random.Generator, n: int = 6) -> list[np.ndarray]:
    return [rng.normal(0.0005, 0.01, 250) for _ in range(n)]


def test_planted_better_candidate_yields_the_hand_computed_gap() -> None:
    rng = np.random.default_rng(SEED)
    vecs = _base_returns(rng)
    # the SELECTED winner (max fitness) has a fat left tail; the ORACLE candidate a thin one
    winner_vec = np.concatenate([vecs[0], [-0.08, -0.09, -0.10]])
    oracle_vec = np.clip(vecs[1], -0.005, None)  # strictly better CVaR by construction
    records = [
        _candidate("distributional", "c-winner", 0.9, winner_vec),
        _candidate("distributional", "c-oracle", 0.1, oracle_vec),
        _candidate("distributional", "c-mid", 0.5, vecs[2]),
    ]
    out = validation_headroom(records, n_boot=400, rng=np.random.default_rng(1))
    assert out["status"] == "ok" and out["executed"] is True
    leg = out["per_arm"]["distributional"]["cvar"]
    achieved = cvar(winner_vec, 0.05)
    frontier = max(cvar(v, 0.05) for v in (winner_vec, oracle_vec, vecs[2]))
    assert leg["selected_id"] == "c-winner"
    assert leg["oracle_id"] == "c-oracle"
    assert leg["oracle_is_selected"] is False
    assert abs(leg["achieved"] - achieved) < 1e-12
    assert abs(leg["frontier"] - frontier) < 1e-12
    assert leg["gap"] > 0.0
    assert abs(leg["gap"] - (frontier - achieved)) < 1e-12
    assert leg["gap_ci_low"] >= 0.0  # within-resample the gap is nonnegative by construction
    assert leg["gap_ci_low"] <= leg["gap"] <= leg["gap_ci_high"] + 1e-12


def test_selection_equal_to_oracle_gives_zero_gap() -> None:
    rng = np.random.default_rng(SEED + 1)
    vecs = _base_returns(rng, n=4)
    # fitness ORDER == CVaR order (best CVaR candidate also has max fitness) -> zero gap, CI [0, 0]
    ranked = sorted(vecs, key=lambda v: cvar(v, 0.05))
    records = [
        _candidate("scalar", f"c{i}", 0.1 * (i + 1), v) for i, v in enumerate(ranked)
    ]
    out = validation_headroom(records, n_boot=300, rng=np.random.default_rng(2))
    leg = out["per_arm"]["scalar"]["cvar"]
    assert leg["oracle_is_selected"] is True
    assert leg["gap"] == 0.0
    assert leg["gap_ci_low"] == 0.0 and leg["gap_ci_high"] == 0.0


def test_dsr_leg_reuses_the_canonical_convention() -> None:
    rng = np.random.default_rng(SEED + 2)
    vecs = _base_returns(rng, n=5)
    records = [_candidate("distributional", f"c{i}", 0.1 * i, v) for i, v in enumerate(vecs)]
    out = validation_headroom(records, n_boot=200, rng=np.random.default_rng(3))
    dsr = out["per_arm"]["distributional"]["dsr"]
    # hand-recompute with the SAME utilities + convention (winner_dsr: per-period sharpes, ddof=1)
    sharpes = np.asarray([_sample_moments(v)[0] for v in vecs])
    var_sr = float(np.var(sharpes, ddof=1))
    assert abs(dsr["var_sr"] - var_sr) < 1e-15
    achieved = deflated_sharpe_ratio(vecs[4], 5, var_sr=var_sr)  # winner = max fitness = c4
    frontier = max(deflated_sharpe_ratio(v, 5, var_sr=var_sr) for v in vecs)
    assert abs(dsr["achieved"] - achieved) < 1e-15
    assert abs(dsr["frontier"] - frontier) < 1e-15
    assert dsr["selected_id"] == "c4"
    assert abs(dsr["gap"] - (frontier - achieved)) < 1e-15


def test_validation_only_fail_safe_excludes_test_and_frozen_records() -> None:
    rng = np.random.default_rng(SEED + 3)
    vecs = _base_returns(rng, n=3)
    records = [_candidate("distributional", f"c{i}", 0.1 * i, v) for i, v in enumerate(vecs)]
    out_clean = validation_headroom(records, n_boot=200, rng=np.random.default_rng(4))
    # a frozen marker + a per-seed TEST record with a fabulous vector must NOT shift anything
    poisoned = records + [
        {**_candidate("distributional", "c-frozen", 9.9, np.full(250, 0.01)), "frozen": True},
        _candidate(
            "distributional",
            "c-test",
            9.9,
            np.full(250, 0.01),
            extra_metrics={"test_returns": [0.01] * 250},
        ),
    ]
    out_poisoned = validation_headroom(poisoned, n_boot=200, rng=np.random.default_rng(4))
    assert out_clean["per_arm"]["distributional"] == out_poisoned["per_arm"]["distributional"]


def test_vectorless_candidates_count_toward_n_trials_only() -> None:
    rng = np.random.default_rng(SEED + 4)
    vecs = _base_returns(rng, n=3)
    records = [_candidate("distributional", f"c{i}", 0.1 * i, v) for i, v in enumerate(vecs)]
    records.append(
        {
            "arm": "distributional",
            "generation": 1,
            "candidate_id": "c-novec",
            "run_id": "c-novec",
            "feedback_block": "",
            "metrics": {"val_fitness": 0.05},  # no val_returns -> excluded, still a searched slot
        }
    )
    out = validation_headroom(records, n_boot=100, rng=np.random.default_rng(5))
    entry = out["per_arm"]["distributional"]
    assert entry["n_candidates"] == 3
    assert entry["n_trials"] == 4  # the #32 expected-max multiplicity counts the burnt slot
    assert entry["n_excluded"] == 1


def test_degrades_below_floor_and_with_no_usable_data() -> None:
    rng = np.random.default_rng(SEED + 5)
    few = [_candidate("scalar", "c0", 0.1, rng.normal(0, 0.01, 100))]
    out = validation_headroom(few, n_boot=100, rng=np.random.default_rng(6))
    assert out["status"] == "no_data" and out["executed"] is False
    assert out["per_arm"]["scalar"]["status"] == "skipped"
    empty = validation_headroom([], n_boot=100)
    assert empty["status"] == "no_data"


def test_pooled_block_is_one_selection_population() -> None:
    rng = np.random.default_rng(SEED + 6)
    a = [_candidate("scalar", f"a{i}", 0.1 * i, v) for i, v in enumerate(_base_returns(rng, 3))]
    b = [
        _candidate("distributional", f"b{i}", 1.0 + 0.1 * i, v)
        for i, v in enumerate(_base_returns(rng, 3))
    ]
    out = validation_headroom(a + b, n_boot=200, rng=np.random.default_rng(7))
    pooled = out["pooled"]
    assert pooled["n_candidates"] == 6 and pooled["n_trials"] == 6
    # the pooled selected candidate is the GLOBAL max-fitness candidate (arm-qualified id)
    assert pooled["cvar"]["selected_id"] == "distributional:b2"
    all_cvars = {
        f"{r['arm']}:{r['candidate_id']}": cvar(np.asarray(r["metrics"]["val_returns"]), 0.05)
        for r in a + b
    }
    assert pooled["cvar"]["oracle_id"] == max(all_cvars, key=all_cvars.get)  # type: ignore[arg-type]
    assert abs(pooled["cvar"]["frontier"] - max(all_cvars.values())) < 1e-12


def test_default_rng_is_deterministic() -> None:
    rng = np.random.default_rng(SEED + 7)
    records = [
        _candidate("distributional", f"c{i}", 0.1 * i, v)
        for i, v in enumerate(_base_returns(rng, 4))
    ]
    a = validation_headroom(records, n_boot=300)
    b = validation_headroom(records, n_boot=300)
    assert a == b


# --------------------------------------------------------------------------- #
# analyze() wiring: DISJOINT report-only block + renderer in write_report
# --------------------------------------------------------------------------- #
def test_analyze_wires_validation_headroom_disjoint_and_renders(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED + 8)
    for i, v in enumerate(_base_returns(rng, 6)):
        rec = _candidate("distributional", f"c{i:02d}", 0.1 * i, v, generation=i // 3)
        rec.update(
            seed=0,
            fold=0,
            reward_source_hash=f"h{i}",
            wall_clock=1.0,
            env_fingerprint="test",
        )
        write_run(rec, tmp_path / "search" / "distributional")
    out = AC.analyze(tmp_path)
    vh = out["validation_headroom"]
    assert vh["status"] == "ok", vh
    assert vh["per_arm"]["distributional"]["status"] == "ok"
    # arms without records are reported skipped, never fabricated
    assert vh["per_arm"]["scalar"]["status"] == "skipped"
    # DISJOINT: no frozen-family tuple keys; the headline is untouched
    assert not any(k in vh for k in ("arm_a", "arm_b", "metric", "level"))
    assert "h2" in out
    md = AC.write_report(out, tmp_path / "report").read_text(encoding="utf-8")
    assert "Validation-headroom" in md
    assert "micro-anchor (e)" in md
