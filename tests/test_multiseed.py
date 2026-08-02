"""k-seed search aggregation (B-A2): IQM selection, mean PBO vector, concat fed-tail, all-3 validity."""
from __future__ import annotations

import numpy as np
import pytest

from src.search.multiseed import aggregate_k_seeds, candidate_seed_ids


def test_candidate_seed_ids_k1_byte_identical_and_k3_fanout():
    assert candidate_seed_ids("distributional-g0-c0", 1, 5) == [("distributional-g0-c0", 5)]
    assert candidate_seed_ids("distributional-g0-c0", 3, 0) == [
        ("distributional-g0-c0-s0", 0),
        ("distributional-g0-c0-s1", 1),
        ("distributional-g0-c0-s2", 2),
    ]


def _seed_result(fitness, val, train):
    return {"ok": True, "fitness": fitness, "val_returns": val, "train_returns": train,
            "reward_source": "src", "reward_hash": "h"}


def test_aggregate_iqm_mean_and_train_concat_fed_tail():
    from src.feedback.measurement import ReturnDistribution
    from src.inference.bootstrap import iqm

    trains = [[0.005, -0.02, 0.01, 0.03], [0.002, -0.01, 0.02, -0.005], [0.0, -0.03, 0.015, 0.01]]
    per_seed = [
        _seed_result(1.0, [0.01, -0.02, 0.03], trains[0]),
        _seed_result(2.0, [0.02, -0.01, 0.04], trains[1]),
        _seed_result(3.0, [0.00, -0.03, 0.02], trains[2]),
    ]
    r = aggregate_k_seeds("distributional-g0-c0", "distributional", per_seed)
    assert r["ok"] and r["k_seeds"] == 3
    # SELECTION = IQM of the per-seed fitnesses (exactly the bootstrap.iqm)
    assert r["fitness"] == pytest.approx(float(iqm(np.array([1.0, 2.0, 3.0]))))
    # PBO vector = per-period MEAN across seeds (VAL — the selection series)
    assert r["val_returns"] == pytest.approx([0.01, -0.02, 0.03])
    # FED tail = the 6 frozen stats on the TRAIN-window concatenation (fed in-sample construct):
    # byte-equal to fitting the concatenated train vectors directly
    expected = ReturnDistribution().fit(np.concatenate([np.asarray(t) for t in trains])).tail_stats()
    assert r["tail_stats"] == pytest.approx(expected)
    assert r["reward_source"] == "src" and r["reward_hash"] == "h"


def test_aggregate_is_all_or_nothing():
    # any failed seed -> the whole candidate is FAILED (no lucky-subset bias)
    r = aggregate_k_seeds("c0", "x", [_seed_result(1.0, [0.1], [0.1]), {"ok": False}])
    assert not r["ok"] and "not all" in r["error"]
    # a seed with no val_returns is a failure (cannot aggregate)
    r2 = aggregate_k_seeds("c0", "x", [_seed_result(1.0, [], [0.1])])
    assert not r2["ok"]
    # a seed with no TRAIN returns fails LOUD (the fed tail is train-window by design; a silent
    # val fallback would change the manipulated variable's window)
    r3 = aggregate_k_seeds("c0", "x", [_seed_result(1.0, [0.1], [])])
    assert not r3["ok"] and "train_returns" in r3["error"]
    # empty -> failed
    assert not aggregate_k_seeds("c0", "x", [])["ok"]
