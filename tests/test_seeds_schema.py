"""Seed-schema resolution + tier structure (the long-run crash-insurance foundation)."""
from __future__ import annotations

import pytest

from src.utils.seeds import SeedSchemaError, n_seeds, resolve_seeds, seed_tiers


def test_bare_list_backcompat():
    assert resolve_seeds([0, 1, 2, 3]) == [0, 1, 2, 3]
    assert seed_tiers([0, 1, 2]) == [[0, 1, 2]]
    with pytest.raises(SeedSchemaError, match="contiguous"):
        resolve_seeds([0, 2, 3])  # non-contiguous rejected (CRN + freeze bind on [0..N-1])
    with pytest.raises(SeedSchemaError, match="empty"):
        resolve_seeds([])


def test_uniform_schema():
    assert resolve_seeds({"mode": "uniform", "n": 403}) == list(range(403))
    assert n_seeds({"mode": "uniform", "n": 403}) == 403
    assert seed_tiers({"mode": "uniform", "n": 5}) == [[0, 1, 2, 3, 4]]  # one tier
    with pytest.raises(SeedSchemaError, match="positive n"):
        resolve_seeds({"mode": "uniform", "n": 0})


def test_tiered_schema_partitions_the_flat_set_with_no_overlap_or_gap():
    cfg = {"mode": "tiered", "tiers": [30, 403]}
    flat = resolve_seeds(cfg)
    tiers = seed_tiers(cfg)
    assert flat == list(range(403)) and n_seeds(cfg) == 403
    # tier 0 = the distinction-bankable n=30 core; tier 1 = the equivalence-power extension
    assert tiers[0] == list(range(30)) and tiers[1] == list(range(30, 403))
    # THE invariant: tiers partition the flat set exactly (crash-insurance ordering, no re-runs)
    assert [s for t in tiers for s in t] == flat
    assert sum(len(t) for t in tiers) == len(flat)
    # earlier tier's seeds are a strict SUBSET (CRN pairing preserved across the ladder)
    assert set(tiers[0]).issubset(set(flat)) and set(tiers[0]).isdisjoint(set(tiers[1]))


def test_tiered_schema_rejects_malformed_bounds():
    for bad in ([30, 30], [403, 30], [0, 403], [-1, 30]):
        with pytest.raises(SeedSchemaError, match="increasing positive"):
            resolve_seeds({"mode": "tiered", "tiers": bad})
    with pytest.raises(SeedSchemaError, match="non-empty"):
        resolve_seeds({"mode": "tiered", "tiers": []})


def test_unknown_mode_and_type_fail_loud():
    with pytest.raises(SeedSchemaError, match="unknown seeds mode"):
        resolve_seeds({"mode": "wobble", "n": 5})
    with pytest.raises(SeedSchemaError, match="list or a schema"):
        resolve_seeds("0-402")
