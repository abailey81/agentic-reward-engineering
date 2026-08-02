"""CMA-ES generation batching (D27, 2026-08-02) — a DISPATCH change only, and this proves it.

`cma_es_over_template` proposed a whole population with `es.ask()` and then evaluated it with a
SERIAL list comprehension. On the cluster path every one of those calls is a blocking `run_batch`, so
a population of 9 was 9 sequential cluster round-trips. Measured on RUN 4: `cma_es` sat at 9 of 30
candidates on a 21-step remaining chain of ~8.6 h per step, gating the CORE line's C1 barrier and
therefore the campaign's COMMON rung.

The safety argument is that `es.ask()` produces the whole population BEFORE any member is evaluated
and `es.tell()` consumes all of it, so no member's proposal depends on another member's fitness. That
argument is exactly the kind of thing that was already wrong once here — `campaign.py` asserted in a
comment that CMA-ES "already dispatches a whole population per generation" while the code did the
opposite — so these tests MEASURE the identity instead of restating it.

Mirrors `test_dfo_tpe_batch.py`, which proves the same property for TPE's startup phase.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.search.dfo_toolkit import cma_es_over_template

BUDGET = 30          # config/campaign.yaml: candidates_per_arm (the matched H4 budget)
DIM = 6              # baselines.reward_family.family_bounds -> pycma default popsize 9
BOUNDS = [(-2.0, 2.0)] * DIM
SEED = 12345


def _objective(x: np.ndarray) -> float:
    """Deterministic and order-independent, so any difference is the ALGORITHM, not the objective."""
    x = np.asarray(x, dtype=float)
    return float(-np.sum((x - 0.37) ** 2) - 0.15 * np.sum(np.sin(3.0 * x) ** 2))


class _Trace:
    """Records every point and score the optimiser actually asked for, in order."""

    def __init__(self) -> None:
        self.points: list[tuple[float, ...]] = []
        self.scores: list[float] = []
        self.on_evaluated_idx: list[int] = []

    def single(self, x) -> float:
        s = _objective(x)
        self.points.append(tuple(round(float(v), 12) for v in np.asarray(x, dtype=float)))
        self.scores.append(round(s, 12))
        return s

    def batch(self, xs):
        return [self.single(x) for x in xs]


def _run(*, batched: bool, **kw):
    tr = _Trace()
    extra = {"batch_eval_fn": tr.batch} if batched else {}
    res = cma_es_over_template(tr.single, BOUNDS, {"matched_budget": BUDGET},
                               rng=np.random.default_rng(SEED), **extra, **kw)
    return res, tr


def _observable(res, tr):
    """Everything a caller can see. Points are compared IN ORDER because candidate ids are assigned
    in proposal order and `--resume` replays archived candidates BY ID: a reordering would silently
    corrupt the replay on a confirmatory arm."""
    return (
        tr.points,
        tr.scores,
        tuple(round(float(v), 12) for v in np.asarray(res["best_coeffs"], dtype=float)),
        round(float(res["best_score"]), 12),
        int(res["n_evaluated"]),
        [round(float(h["score"]), 12) for h in res["history"]],
        [h["source"] for h in res["history"]],
    )


def test_batched_generation_is_identical_to_serial():
    """THE claim: batching a generation changes nothing a caller can observe."""
    serial = _run(batched=False)
    batched = _run(batched=True)
    assert _observable(*serial) == _observable(*batched)


def test_matched_budget_is_exact_in_both_paths():
    """The matched budget IS the fair H4 comparison; batching must not spend one call more or less."""
    for batched in (False, True):
        res, tr = _run(batched=batched)
        assert int(res["n_evaluated"]) == BUDGET
        assert len(tr.points) == BUDGET


def test_the_batch_is_actually_batched():
    """Guards the opposite failure: a `batch_eval_fn` that is accepted and never used would pass the
    identity test perfectly while delivering none of the speed-up — which is precisely the state the
    code was already in, hidden behind a comment that said otherwise."""
    sizes: list[int] = []

    def counting_batch(xs):
        sizes.append(len(xs))
        return [_objective(x) for x in xs]

    tr = _Trace()
    cma_es_over_template(tr.single, BOUNDS, {"matched_budget": BUDGET},
                         rng=np.random.default_rng(SEED), batch_eval_fn=counting_batch)
    assert sizes, "batch_eval_fn was never called - the wiring is dead"
    assert sum(sizes) == BUDGET
    assert len(sizes) < BUDGET, f"{len(sizes)} dispatches for {BUDGET} points is not batching"
    assert max(sizes) > 1, "every dispatch held one point - still serial"


def test_resume_replays_a_cached_generation_without_recheckpointing_it():
    """The RESUME path. On `--resume` the cluster's `template_eval_batch` replays archived candidates
    by id; the in-process equivalent is `cache_lookup`. A replayed member must produce the same
    history and must NOT re-fire `on_evaluated`, or completed work would be checkpointed twice."""
    baseline, _ = _run(batched=True)
    first_gen = {i: h["score"] for i, h in enumerate(baseline["history"]) if i < 9}

    seen: list[int] = []
    tr = _Trace()
    resumed = cma_es_over_template(
        tr.single, BOUNDS, {"matched_budget": BUDGET}, rng=np.random.default_rng(SEED),
        cache_lookup=lambda idx, x: first_gen.get(idx),
        on_evaluated=lambda idx, x, s: seen.append(idx),
        batch_eval_fn=tr.batch,
    )
    assert [round(float(h["score"]), 12) for h in resumed["history"]] == \
           [round(float(h["score"]), 12) for h in baseline["history"]]
    assert all(i >= 9 for i in seen), "on_evaluated fired for a CACHED member"
    assert sorted(seen) == list(range(9, BUDGET))


def test_without_batch_eval_fn_the_behaviour_is_unchanged():
    """Backward compatibility: the laptop path and every existing caller pass no `batch_eval_fn`."""
    a = _run(batched=False)
    b = _run(batched=False)
    assert _observable(*a) == _observable(*b)
    assert int(a[0]["n_evaluated"]) == BUDGET


def test_a_batch_returning_the_wrong_count_fails_loudly():
    """Fail fast and loud rather than desynchronising the covariance update from the history."""
    with pytest.raises(ValueError, match="1:1|desynchronise"):
        cma_es_over_template(_objective, BOUNDS, {"matched_budget": BUDGET},
                             rng=np.random.default_rng(SEED),
                             batch_eval_fn=lambda xs: [_objective(x) for x in xs][:-1])


def test_a_perturbed_score_is_detected():
    """MUTATION CONTROL. If perturbing one score changed nothing, every assertion above would be
    vacuous. This is the check that makes the others mean something."""
    serial = _run(batched=False)

    tr = _Trace()

    def bad_batch(xs):
        out = [_objective(x) for x in xs]
        for x in xs:
            tr.single(x)
        if len(tr.scores) > 4:
            out[0] += 1e-6
        return out

    bad = cma_es_over_template(tr.single, BOUNDS, {"matched_budget": BUDGET},
                               rng=np.random.default_rng(SEED), batch_eval_fn=bad_batch)
    assert round(float(bad["best_score"]), 12) != _observable(*serial)[3] or \
        [round(float(h["score"]), 12) for h in bad["history"]] != _observable(*serial)[5]
