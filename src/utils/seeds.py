"""Seed-schema resolution + crash-insurance TIER structure for the long confirmatory campaign.

The confirmatory campaign is LONG (uniform-n seeds × the arm roster + H1 baselines, days-to-weeks on
Myriad), so the seed set is a SCHEMA that encodes BOTH the count AND the tier ordering — never a bare
list whose order a scheduler is free to shuffle. The invariant a long run must hold: **a crash at any
point leaves the best-possible COMPLETE partial result, not a ragged one.**

Three representations (``resolve_seeds`` accepts all; the freeze binds on the FLAT set it returns):

* ``{mode: uniform, n: N}``            → seeds ``0..N-1`` as ONE tier (the plan §3 Stage-1 uniform-n);
* ``{mode: tiered, tiers: [n1,n2,…]}`` → CUMULATIVE tiers: tier 0 = ``0..n1-1``, tier 1 = ``n1..n2-1``,
  … The tiers are the crash-insurance ladder — tier 0 is the distinction-bankable ``n=30`` core
  (H2 + mechanism + H1 + H3 all complete), later tiers scale the seeds for equivalence power. Tiers
  are **ORDER-ONLY**: an earlier tier's seeds are a strict SUBSET of the full set (CRN pairing
  preserved), so a later tier NEVER re-runs an earlier tier's seed — it only extends it;
* a bare ``list`` (e.g. ``[0..29]``) → used as-is (back-compat with the pre-schema config).

``resolve_seeds`` → the FLAT sorted ``[0..N-1]`` list (what the analysis + the freeze-hash bind on);
``seed_tiers`` → the list-of-tiers, highest-value first (what the tiered runner submits in order).
Both are pure + deterministic so the freeze mirror and every runner agree on the exact same seeds.
"""
from __future__ import annotations

from typing import Any

__all__ = ["resolve_seeds", "seed_tiers", "n_seeds", "SeedSchemaError"]


class SeedSchemaError(ValueError):
    """A malformed seeds schema — raised LOUD so a bad seed set can never silently mis-size a run."""


def _as_contiguous(seeds: list[int]) -> list[int]:
    ints = [int(s) for s in seeds]
    if ints != list(range(len(ints))):
        raise SeedSchemaError(
            f"a seed list must be the contiguous [0..N-1] block (CRN pairing + freeze bind on it), "
            f"got {seeds!r}"
        )
    return ints


def resolve_seeds(seeds_cfg: Any) -> list[int]:
    """The FLAT ``[0..N-1]`` seed list for ``seeds_cfg`` (list / uniform / tiered schema)."""
    if isinstance(seeds_cfg, list):
        if not seeds_cfg:
            raise SeedSchemaError("seeds list is empty")
        return _as_contiguous(seeds_cfg)
    if isinstance(seeds_cfg, dict):
        mode = seeds_cfg.get("mode")
        if mode == "uniform":
            n = int(seeds_cfg.get("n", 0))
            if n <= 0:
                raise SeedSchemaError(f"uniform seeds needs a positive n, got {seeds_cfg.get('n')!r}")
            return list(range(n))
        if mode == "tiered":
            tiers = seeds_cfg.get("tiers")
            if not isinstance(tiers, list) or not tiers:
                raise SeedSchemaError("tiered seeds needs a non-empty 'tiers' list of cumulative bounds")
            bounds = [int(t) for t in tiers]
            if bounds[0] <= 0 or bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
                raise SeedSchemaError(
                    f"tiered 'tiers' must be STRICTLY increasing positive cumulative bounds "
                    f"(e.g. [30, 403] = n=30 core then up to n=403), got {tiers!r}"
                )
            return list(range(bounds[-1]))
        raise SeedSchemaError(f"unknown seeds mode {mode!r} (want 'uniform' or 'tiered')")
    raise SeedSchemaError(f"seeds must be a list or a schema dict, got {type(seeds_cfg).__name__}")


def seed_tiers(seeds_cfg: Any) -> list[list[int]]:
    """The seed TIERS, highest-value (smallest, decisive) first. A single-tier config → one tier.

    For ``{mode: tiered, tiers: [n1, n2, …]}`` the tiers partition ``0..n_max-1`` at the cumulative
    bounds, so ``sum(len(t) for t in seed_tiers) == n_seeds`` and ``[s for t in seed_tiers for s in t]``
    is exactly ``resolve_seeds`` — a runner can execute tier-by-tier and be certain it covers the whole
    set with no overlap and no gap.
    """
    if isinstance(seeds_cfg, dict) and seeds_cfg.get("mode") == "tiered":
        resolve_seeds(seeds_cfg)  # validate first (raises on malformed)
        bounds = [0] + [int(t) for t in seeds_cfg["tiers"]]
        return [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]
    return [resolve_seeds(seeds_cfg)]


def n_seeds(seeds_cfg: Any) -> int:
    """The total seed count for ``seeds_cfg`` (what the power analysis + freeze report)."""
    return len(resolve_seeds(seeds_cfg))
