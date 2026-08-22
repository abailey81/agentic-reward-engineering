"""Point 1: how much of the arm difference is authoring noise?

Dr Okhrati's first point, in his words:

    "The 102 seeds mostly measure randomness in training the reinforcement-learning agent. They do
    not adequately measure randomness in: what the LLM happens to write; which ideas it produces in
    a particular run; which program happens to become the winner ... Therefore, the experiment can
    confidently say: The selected tail-feedback program performed this way relative to the selected
    control program. But it cannot say: Tail feedback generally causes an LLM to produce programs
    that perform this way."

He is describing a variance the campaign never resampled. ``run_authoring_variance.py`` resamples it
-- two independent authoring chains per model, and the WHOLE final generation of each chain sent to
the sealed window rather than a winner -- and this script reads what that produced.

The design is balanced BY INTENT: for each (line, arm), 2 chains, 5 candidates within a chain,
30 seeds within a candidate. That is a three-level nested random-effects layout, so the variance
components come straight from the nested mean squares with no model fitting and no optimiser to
distrust:

    sigma2_seed  = MS_seed
    sigma2_cand  = (MS_cand  - MS_seed) / n_seeds
    sigma2_chain = (MS_chain - MS_cand) / (n_seeds * n_cands)

NOTE: THE ARCHIVE IS NOT BALANCED, so two cuts run before the estimator and BOTH report what they
removed. A generation-1 candidate whose program fails validation is permanently rejected, leaving
one chain with five candidates and the other with four (measured 2026-08-24: 63 rejects, 19 of the
55 pairs unbalanced), so ``balance_candidates_across_chains`` cuts each pair to the common count.
And the seeds have holes -- 185 of 440 candidates were missing a seed below their own maximum --
so ``truncate_to_seed_floor`` cuts to the seeds COMMON to a unit's candidates rather than to each
candidate's own lowest. Both cuts are outcome-blind, and neither is silent.

We report all three levels on one scale, because the comparison IS the argument: if the spread across
authoring runs is of the same order as the spread across arms, then a single authoring run cannot
identify an arm effect, which is exactly what Dr Okhrati said. Having measured it, we can now say
which it is instead of assuming.

A component estimated from few degrees of freedom can come out negative. We clamp at zero and say so
in the output rather than hiding it -- with two chains, sigma2_chain carries one degree of freedom
per (line, arm), so it is the pooled figure across lines that is worth reading, never a single cell.

Usage
-----
    python scripts/analyse_authoring_variance.py --output-dir outputs/authoring_variance
    python scripts/analyse_authoring_variance.py --selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("analyse_authoring_variance")

#: The sealed-window outcome. ``sharpe`` is the registered metric for node N2
#: (``config/preregistration.yaml``: ``metric: sharpe``), so it is the default here too.
DEFAULT_METRIC = "test_sharpe"

#: The treatment and its control, in the registered contrast direction.
TREATMENT = "distributional"
CONTROL = "scalar"

#: Every comparator the treatment is registered against, in the campaign's own order: the 3 of the
#: m=6 family plus the disjoint scrambled control. ``CONTROL`` leads because it carries the
#: registered contrast; the other 3 are here so the resample answers for the same family the main
#: campaign answers for, and so that reporting one of them can never be a choice among four.
COMPARATORS = (CONTROL, "scalar_cvar5", "placebo", "placebo_shuffled")

#: The eleven authoring lines, in ARCHIVE-DIRECTORY spelling. Taken from the driver rather than
#: retyped, because a preliminary read reports which lines are ABSENT and a list that drifted from
#: the run's own would silently mis-state that. ``run_authoring_variance`` imports only the standard
#: library at module level, so this costs nothing.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from run_authoring_variance import ALL_LINES as _DRIVER_LINES, _slug as _line_slug  # noqa: E402

ALL_LINES: tuple[str, ...] = tuple(_line_slug(x) for x in _DRIVER_LINES)

#: ``candidate_id`` is ``<arm>-g<generation>-c<index>``. The record's own ``arm`` field carries the
#: TEST LABEL (``<arm>__c<k>``), not the arm, so the arm must be read from here or every candidate
#: of an arm looks like a separate arm.
_CID = re.compile(r"^(?P<arm>[a-z_0-9]+)-g(?P<gen>\d+)-c(?P<k>\d+)$")


class Record:
    """One sealed-window training: which model wrote it, in which authoring run, and how it did."""

    __slots__ = ("line", "chain", "arm", "candidate", "seed", "score")

    def __init__(self, line: str, chain: int, arm: str, candidate: int, seed: int, score: float):
        self.line, self.chain, self.arm = line, chain, arm
        self.candidate, self.seed, self.score = candidate, seed, score

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"Record({self.line} r{self.chain} {self.arm} c{self.candidate} "
                f"s{self.seed} {self.score:.4f})")


def parse_candidate_id(cid: str) -> tuple[str, int, int] | None:
    """``(arm, generation, candidate_index)`` from a candidate id, or None if it is not one."""
    m = _CID.match(str(cid))
    if m is None:
        return None
    return m.group("arm"), int(m.group("gen")), int(m.group("k"))


def load_records(output_dir: str | Path, metric: str = DEFAULT_METRIC,
                 final_generation: int = 1) -> tuple[list[Record], dict[str, int]]:
    """Every sealed-window record under ``output_dir``, as flat rows, plus what was SKIPPED.

    The skip tally is RETURNED rather than only logged, so it reaches the report and the JSON. A
    record dropped for an unreadable file or a non-finite metric is missing data, and missing data
    that only appears on a log line nobody reads is indistinguishable from data that was never
    there.

    NOTE: THE CHAIN IS IN THE PATH, NOT IN THE RECORD. A record carries its line nowhere and its chain
    nowhere; both come from the archive layout ``<line>/r<chain>/test/...``. That is why this loader
    walks the tree rather than globbing ``record.json`` and reading fields — a flattened copy of
    these files would silently lose the very factor the analysis exists to estimate.
    """
    root = Path(output_dir)
    rows: list[Record] = []
    skipped: dict[str, int] = defaultdict(int)
    for path in sorted(root.glob("*/r*/test/*/*/record.json")):
        rel = path.relative_to(root).parts
        line, chain_dir = rel[0], rel[1]
        if not chain_dir.startswith("r") or not chain_dir[1:].isdigit():
            skipped["unparseable chain directory"] += 1
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped["unreadable record"] += 1
            continue
        parsed = parse_candidate_id(rec.get("candidate_id", ""))
        if parsed is None:
            skipped["candidate_id not in <arm>-g<n>-c<k> form"] += 1
            continue
        arm, gen, k = parsed
        if gen != int(final_generation):
            skipped[f"generation {gen} (only the final generation is sealed-window work)"] += 1
            continue
        value = (rec.get("metrics") or {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            skipped[f"missing or non-finite {metric}"] += 1
            continue
        seed = rec.get("seed")
        if not isinstance(seed, int):
            skipped["missing seed"] += 1
            continue
        rows.append(Record(line, int(chain_dir[1:]), arm, k, seed, float(value)))
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        _LOG.info("skipped %d record(s): %s", n, reason)
    return rows, dict(skipped)


def truncate_to_seed_floor(rows: list[Record], n_seeds: int) -> tuple[list[Record], dict[str, int]]:
    """Keep the ``n_seeds`` lowest seeds COMMON to every candidate of a (line, arm) unit.

    ``nested_variance_components`` refuses an unbalanced nest by design, so a part-finished run
    decomposes nothing at all: the seed counts run from 1 to 30 while the estimator needs one
    number. A preliminary read therefore has to be balanced BEFORE it is analysed.

    Cutting to the lowest seeds is the only rule that is independent of the scores, so it adds no
    selection on outcome the way a "best n" or "latest n" rule would.

    NOTE: The cut is COMMON-seed rather than per-candidate, and that correction matters. This function
    read the n lowest seeds of each candidate SEPARATELY until 2026-08-24, on the stated ground that
    the driver sorts each unit's taskfile seed-major (``campaign.run_test_leg``, ``interleave=True``)
    so every candidate reaches the same seeds first. **Measured on the live archive, that is false**:
    the low seeds have holes, because a unit that restarted skipped ahead. Of the 287 candidates
    holding at least five seeds, only 181 held seeds 0-4, and one deepseek cell ran candidates on
    [0, 1, 7, 8, 9] and [0, 6, 7, 8, 9]. Seeds are common random numbers here, so scoring two
    candidates on different draws leaks seed noise into ``sigma2_candidate`` — the quantity this
    whole script exists to estimate. Intersecting first costs a few records and removes that leak.
    """
    by_unit: dict[tuple[str, str], dict[tuple[int, int], list[Record]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        by_unit[(r.line, r.arm)][(r.chain, r.candidate)].append(r)

    kept: list[Record] = []
    units_dropped = short = discarded = 0
    n_cands = 0
    for cells in by_unit.values():
        n_cands += len(cells)
        common = set.intersection(*(set(r.seed for r in recs) for recs in cells.values()))
        if len(common) < n_seeds:
            units_dropped += 1
            short += len(cells)
            discarded += sum(len(v) for v in cells.values())
            continue
        floor = set(sorted(common)[:n_seeds])
        for recs in cells.values():
            take = [r for r in recs if r.seed in floor]
            kept.extend(take)
            discarded += len(recs) - len(take)
    return kept, {"candidates_kept": n_cands - short,
                  "candidates_dropped_as_short": short,
                  "units_dropped_for_no_common_floor": units_dropped,
                  "records_discarded": discarded}


def balance_candidates_across_chains(
    rows: list[Record], min_candidates: int = 2
) -> tuple[list[Record], dict[str, int]]:
    """Keep the same NUMBER of candidates in every chain of a ``(line, arm)`` unit.

    NOTE: THIS IS A PERMANENT IMBALANCE, NOT A PROGRESS PROBLEM, AND WITHOUT THIS STEP 19 OF THE 55
    PAIRS COULD NEVER DECOMPOSE — no quantity of extra seeds would fix them. A generation-1
    candidate whose authored program fails node-side validation is permanently rejected
    (``src/cluster/ledger.py``, ``MAX_RETRIES = 2``), leaving no search record and no test
    directory. Measured 2026-08-24: 63 g1 candidates were rejected that way, so one chain of a pair
    holds five candidates and the other holds four, and ``nested_variance_components`` refuses an
    unbalanced nest by design.

    Cutting to the LOWEST candidate indices is outcome-blind: ``cid = f"{arm}-g{gen}-c{ci}"`` with
    ``ci`` running over ``range(cpg)`` (``campaign.py:893``), so the index is the authoring
    enumeration order and carries no information about the score.

    A unit is dropped when a chain holds fewer than ``min_candidates``, because the estimator needs
    at least two candidates per chain to separate the candidate level from the seed level at all.
    """
    by_unit: dict[tuple[str, str], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        by_unit[(r.line, r.arm)][r.chain].add(r.candidate)

    keep: dict[tuple[str, str], dict[int, set[int]]] = {}
    units_dropped = trimmed = 0
    for unit, by_chain in by_unit.items():
        m = min(len(v) for v in by_chain.values())
        if len(by_chain) < 2 or m < min_candidates:
            units_dropped += 1
            continue
        sel = {ch: set(sorted(cands)[:m]) for ch, cands in by_chain.items()}
        trimmed += sum(len(by_chain[ch]) - len(sel[ch]) for ch in by_chain)
        keep[unit] = sel

    kept = [r for r in rows
            if (r.line, r.arm) in keep and r.candidate in keep[(r.line, r.arm)].get(r.chain, ())]
    return kept, {"units_kept": len(keep),
                  "units_dropped_for_too_few_candidates": units_dropped,
                  "candidates_trimmed_for_balance": trimmed,
                  "records_discarded": len(rows) - len(kept)}


def nested_variance_components(
    by_chain: dict[int, dict[int, list[float]]]
) -> dict[str, float] | None:
    """Seed / candidate / chain variance components from a balanced three-level nest.

    ``by_chain[chain][candidate] -> [score per seed]``. Returns None unless the layout is balanced
    and has at least two chains, two candidates and two seeds — an unbalanced nest needs a different
    estimator, and returning a number computed by the wrong one would be worse than returning
    nothing.
    """
    chains = sorted(by_chain)
    if len(chains) < 2:
        return None
    cand_counts = {len(by_chain[c]) for c in chains}
    if len(cand_counts) != 1 or cand_counts.pop() < 2:
        return None
    seed_counts = {len(v) for c in chains for v in by_chain[c].values()}
    if len(seed_counts) != 1:
        return None
    n_seeds = seed_counts.pop()
    if n_seeds < 2:
        return None
    n_cands = len(by_chain[chains[0]])
    n_chains = len(chains)

    cand_means = {c: {k: float(np.mean(v)) for k, v in by_chain[c].items()} for c in chains}
    chain_means = {c: float(np.mean(list(cand_means[c].values()))) for c in chains}
    grand = float(np.mean(list(chain_means.values())))

    ss_seed = sum((x - cand_means[c][k]) ** 2
                  for c in chains for k, v in by_chain[c].items() for x in v)
    ss_cand = sum(n_seeds * (cand_means[c][k] - chain_means[c]) ** 2
                  for c in chains for k in by_chain[c])
    ss_chain = sum(n_seeds * n_cands * (chain_means[c] - grand) ** 2 for c in chains)

    df_seed = n_chains * n_cands * (n_seeds - 1)
    df_cand = n_chains * (n_cands - 1)
    df_chain = n_chains - 1

    ms_seed = ss_seed / df_seed
    ms_cand = ss_cand / df_cand
    ms_chain = ss_chain / df_chain

    # Negative estimates happen at low df. Clamp, and report that we clamped.
    v_seed = ms_seed
    v_cand = (ms_cand - ms_seed) / n_seeds
    v_chain = (ms_chain - ms_cand) / (n_seeds * n_cands)
    return {
        "sigma2_seed": max(0.0, v_seed),
        "sigma2_candidate": max(0.0, v_cand),
        "sigma2_chain": max(0.0, v_chain),
        "clamped": float(v_cand < 0) + float(v_chain < 0),
        "n_chains": float(n_chains), "n_candidates": float(n_cands), "n_seeds": float(n_seeds),
    }


def arm_contrast_over_authoring(rows: list[Record], treatment: str = TREATMENT,
                                control: str = CONTROL, min_seeds: int = 2) -> dict[str, Any]:
    """The treatment-minus-control contrast, with AUTHORING resampled rather than held fixed.

    The campaign's estimate conditions on one authored program per arm. Here each model contributes
    one contrast per chain, computed from the IQM of that chain's whole final generation, so the
    spread across chains is authoring variance and it lands in the interval instead of being assumed
    away. That is the difference between "the selected tail program beat the selected control" and
    "tail feedback makes these models write better programs".

    NOTE: THE CONTRAST IS SEED-MATCHED, and it was not until 2026-08-24. The first version pooled every
    treatment score and every control score inside a chain and took one IQM of each. With the two
    arms holding different seed sets — which they do, because the arms fill at different rates and
    the archive has holes — that difference of IQMs is contaminated by which seeds each arm happened
    to reach. Seeds are common random numbers, so the fix is the one the campaign itself uses:
    ``analyze_campaign.py:1588`` intersects the seed sets and pairs element-wise, and
    ``paired_seed_difference_test`` then takes ``statistic(a) - statistic(b)`` over that common set.
    We mirror it exactly, one level up: the per-seed score of an ARM is the IQM over that arm's whole
    final generation at that seed, and the chain's contrast is the IQM of one minus the IQM of the
    other over the seeds both arms hold.

    A seed counts only when EVERY candidate of that arm has a record at it, so each per-seed
    generation score is an average over the same programs rather than over whichever finished first.
    """
    from src.inference.reporting import iqm

    # (line, chain) -> arm -> seed -> {candidate: score}
    by: dict[tuple[str, int], dict[str, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        by[(r.line, r.chain)][r.arm][r.seed][r.candidate] = r.score

    def complete_seeds(per_seed: dict[int, dict[int, float]]) -> tuple[set[int], int]:
        """Seeds at which EVERY candidate of this arm has a record, and that candidate count."""
        if not per_seed:
            return set(), 0
        cands = set().union(*(set(d) for d in per_seed.values()))
        return {s for s, d in per_seed.items() if set(d) == cands}, len(cands)

    contrasts: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for (line, chain), by_arm in sorted(by.items()):
        if treatment not in by_arm or control not in by_arm:
            dropped.append({"line": line, "chain": chain, "why": "one of the two arms is absent"})
            continue
        t_seeds, n_t = complete_seeds(by_arm[treatment])
        c_seeds, n_c = complete_seeds(by_arm[control])
        common = sorted(t_seeds & c_seeds)
        if len(common) < min_seeds:
            why = ("the two arms share no seed at which every candidate has a record"
                   if not common else
                   f"only {len(common)} matched seed(s), below the minimum of {min_seeds}")
            dropped.append({"line": line, "chain": chain, "why": why,
                            "treatment_complete_seeds": len(t_seeds),
                            "control_complete_seeds": len(c_seeds)})
            continue
        t_per_seed = np.asarray(
            [iqm(np.asarray(list(by_arm[treatment][s].values()), dtype=float)) for s in common],
            dtype=float)
        c_per_seed = np.asarray(
            [iqm(np.asarray(list(by_arm[control][s].values()), dtype=float)) for s in common],
            dtype=float)
        t, c = float(iqm(t_per_seed)), float(iqm(c_per_seed))
        contrasts.append({"line": line, "chain": chain, "treatment": t, "control": c,
                          "delta": t - c, "n_seeds": len(common),
                          "n_treatment_candidates": n_t, "n_control_candidates": n_c,
                          "seeds_dropped_treatment": len(by_arm[treatment]) - len(t_seeds),
                          "seeds_dropped_control": len(by_arm[control]) - len(c_seeds)})
    if not contrasts:
        # SAME SHAPE as the populated return, zeroed. A caller must never have to ask which keys
        # exist before reading a result.
        return {"contrasts": [], "dropped": dropped, "n_line_chains": 0,
                "mean_delta": float("nan"), "iqm_delta": float("nan"),
                "iqm_delta_ci95": [float("nan"), float("nan")], "ci_n_boot": 0,
                "ci_resampling_unit": "line (model), clustered", "n_lines_resampled": 0,
                "sd_delta": float("nan"), "treatment_wins": 0,
                "lines_with_both_chains": 0, "lines_where_chains_disagree_in_sign": 0,
                "sign_flip_lines": [],
                "note": f"no (line, chain) held {treatment} and {control} on a shared complete seed"}

    deltas = np.asarray([c["delta"] for c in contrasts], dtype=float)
    # The whole point of this sub-experiment is that authoring variance should land IN the interval
    # rather than be assumed away, and until 2026-08-24 no interval was computed at all.
    #
    # NOTE: THE RESAMPLING UNIT IS THE MODEL, NOT THE CONTRAST, and getting that wrong is a mistake this
    # project has already made once (a seed-level bootstrap that ran far too narrow because the unit
    # was wrong). A model contributes TWO contrasts, one per authoring chain, and they share
    # everything except the chain — so resampling contrasts independently would treat one model as
    # two observations and report an interval that is too tight. We resample LINES with replacement
    # and take every contrast a drawn line owns, which is the ordinary cluster bootstrap.
    n_boot = 2000
    by_line_deltas: dict[str, list[float]] = defaultdict(list)
    for x in contrasts:
        by_line_deltas[x["line"]].append(x["delta"])
    line_keys = sorted(by_line_deltas)
    rng = np.random.default_rng(0)          # fixed: the reported interval must be reproducible
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        drawn = rng.integers(0, len(line_keys), size=len(line_keys))
        pool = [d for j in drawn for d in by_line_deltas[line_keys[j]]]
        boot[i] = iqm(np.asarray(pool, dtype=float))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    out: dict[str, Any] = {
        "contrasts": contrasts,
        "dropped": dropped,
        "n_line_chains": int(deltas.size),
        "mean_delta": float(np.mean(deltas)),
        "iqm_delta": float(iqm(deltas)),
        "iqm_delta_ci95": [float(lo), float(hi)],
        "ci_n_boot": n_boot,
        "ci_resampling_unit": "line (model), clustered — a model's two chains move together",
        "n_lines_resampled": len(line_keys),
        "sd_delta": float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan"),
        "treatment_wins": int(np.sum(deltas > 0)),
    }
    # Where two chains of the SAME line disagree in SIGN, one authoring run would have reported the
    # opposite result. That count is the most direct answer to the point he raised.
    by_line: dict[str, list[float]] = defaultdict(list)
    for c in contrasts:
        by_line[c["line"]].append(c["delta"])
    flips = [ln for ln, d in by_line.items() if len(d) > 1 and min(d) < 0 < max(d)]
    out["lines_with_both_chains"] = sum(1 for d in by_line.values() if len(d) > 1)
    out["lines_where_chains_disagree_in_sign"] = len(flips)
    out["sign_flip_lines"] = sorted(flips)
    return out


def _cluster_bootstrap(by_line: dict[str, list[float]], stat, n_boot: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """A 95% percentile interval for ``stat``, resampling LINES with replacement.

    A drawn line brings every value it owns. This is the same unit ``arm_contrast_over_authoring``
    resamples and for the same reason: a model's programs and runs move together, so resampling the
    individual values would treat one model as many observations and report an interval too tight.
    """
    keys = sorted(by_line)
    rng = np.random.default_rng(seed)       # fixed: the reported interval must be reproducible
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        drawn = rng.integers(0, len(keys), size=len(keys))
        pool = np.asarray([v for j in drawn for v in by_line[keys[j]]], dtype=float)
        draws[i] = stat(pool)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def program_level_contrast(rows: list[Record], treatment: str = TREATMENT,
                           control: str = CONTROL, min_seeds: int = 2) -> dict[str, Any]:
    """How often does ONE treatment program beat ONE control program the same run wrote?

    ``arm_contrast_over_authoring`` collapses each arm's whole final generation to one IQM per seed
    before differencing the arms, so it reports a functional of the program distribution. Dr
    Okhrati's closing sentence is about that distribution itself -- the experiment "cannot say: Tail
    feedback GENERALLY causes an LLM to produce programs that perform this way" -- and an average
    over programs does not answer a question about programs. So this pairs every treatment program
    against every control program of the same authoring run, seed-matches each pair, and reports how
    often the treatment program is ahead.

    The estimator is not invented here. ``rliable`` (Agarwal et al., NeurIPS 2021) is already the
    authority the registered inference plan names, and its recommendation table asks for the
    "average probability of improvement" beside the interval estimate. This is that quantity,
    evaluated over programs rather than over runs.

    NOTE: THE AUTHORING RUN IS HELD FIXED INSIDE A PAIR. Crossing runs would fold the run draw back into
    a number whose whole job is to isolate the program draw, and the run draw is already reported on
    its own above. The crossed figure is computed anyway and returned beside it, because "did you
    check it the other way" is the first question a reader asks.

    NOTE: THE INTERVAL RESAMPLES MODELS, for the reason it does above. A model contributes up to 25
    pairs per run built from 10 programs, so treating those pairs as independent observations would
    report an interval far too tight -- the mistake this project has already made once at seed level.
    """
    from src.inference.reporting import iqm

    # (line, chain) -> arm -> candidate -> {seed: score}
    by: dict[tuple[str, int], dict[str, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        by[(r.line, r.chain)][r.arm][r.candidate][r.seed] = r.score

    def pair_delta(t: dict[int, float], c: dict[int, float]) -> tuple[float, int] | None:
        """One program against one program, on the seeds BOTH hold, or None if they share too few.

        Seeds are common random numbers, so two programs scored on different draws are not
        comparable however many seeds each of them holds.
        """
        common = sorted(set(t) & set(c))
        if len(common) < min_seeds:
            return None
        a = float(iqm(np.asarray([t[s] for s in common], dtype=float)))
        b = float(iqm(np.asarray([c[s] for s in common], dtype=float)))
        return a - b, len(common)

    pairs: list[dict[str, Any]] = []
    per_line_programs: dict[str, dict[str, list[dict[int, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for (line, chain), by_arm in sorted(by.items()):
        for arm in (treatment, control):
            per_line_programs[line][arm].extend(by_arm.get(arm, {}).values())
        if treatment not in by_arm or control not in by_arm:
            continue
        for tc, tps in sorted(by_arm[treatment].items()):
            for cc, cps in sorted(by_arm[control].items()):
                got = pair_delta(tps, cps)
                if got is None:
                    continue
                pairs.append({"line": line, "chain": chain, "treatment_program": tc,
                              "control_program": cc, "delta": got[0], "n_seeds": got[1]})

    if not pairs:
        # SAME SHAPE as the populated return, zeroed, so a caller never has to ask which keys exist.
        return {"n_pairs": 0, "n_lines": 0, "pairs_treatment_ahead": 0,
                "p_treatment_ahead": float("nan"), "p_treatment_ahead_ci95": [float("nan")] * 2,
                "p_treatment_ahead_across_runs": float("nan"), "n_pairs_across_runs": 0,
                "delta_sd": float("nan"), "delta_p10": float("nan"), "delta_p50": float("nan"),
                "delta_p90": float("nan"), "delta_min": float("nan"), "delta_max": float("nan"),
                "n_seeds_min": 0, "n_seeds_max": 0, "ci_n_boot": 0,
                "ci_resampling_unit": "line (model), clustered", "per_line": {},
                "note": f"no authoring run held both {treatment} and {control} programs"}

    d = np.asarray([p["delta"] for p in pairs], dtype=float)
    by_line: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        by_line[p["line"]].append(p["delta"])
    lo, hi = _cluster_bootstrap(by_line, lambda a: float(np.mean(a > 0.0)))

    # The same count with the run NOT held fixed, i.e. every treatment program of a model against
    # every control program of that model whichever run each came from. Reported, never substituted.
    crossed = [pd[0] for line, arms in sorted(per_line_programs.items())
               for t in arms.get(treatment, []) for c in arms.get(control, [])
               if (pd := pair_delta(t, c)) is not None]

    seeds = [p["n_seeds"] for p in pairs]
    out: dict[str, Any] = {
        "n_pairs": int(d.size),
        "n_lines": len(by_line),
        "pairs_treatment_ahead": int(np.sum(d > 0.0)),
        "p_treatment_ahead": float(np.mean(d > 0.0)),
        "p_treatment_ahead_ci95": [lo, hi],
        "p_treatment_ahead_across_runs": float(np.mean(np.asarray(crossed) > 0.0)),
        "n_pairs_across_runs": len(crossed),
        "delta_sd": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "delta_p10": float(np.percentile(d, 10)),
        "delta_p50": float(np.percentile(d, 50)),
        "delta_p90": float(np.percentile(d, 90)),
        "delta_min": float(np.min(d)),
        "delta_max": float(np.max(d)),
        "n_seeds_min": min(seeds), "n_seeds_max": max(seeds),
        "ci_n_boot": 2000,
        "ci_resampling_unit": "line (model), clustered — a model's programs move together",
    }
    # Per model, because the study's largest measured component is the model-by-arm interaction and
    # a pooled probability would hide it. These are DESCRIPTIVE: one model's ~44 pairs come from 10
    # programs, so they carry nothing like 44 degrees of freedom and get no interval.
    #
    # The individual differences travel with them, exactly as ``arm_contrast_over_authoring`` keeps
    # its 20 contrasts. An exhibit drawing this distribution must be able to read it from the
    # artefact rather than re-walking 14,000 record files, or the figure and the number it claims to
    # draw can drift apart without anything noticing.
    out["per_line"] = {
        line: {"n_pairs": len(v),
               "p_treatment_ahead": float(np.mean(np.asarray(v) > 0.0)),
               "iqm_delta": float(iqm(np.asarray(v, dtype=float))),
               "deltas": [float(x) for x in v]}
        for line, v in sorted(by_line.items())}
    return out


def analyse(rows: list[Record], contrast_rows: list[Record] | None = None,
            lines_loaded: set[str] | None = None) -> dict[str, Any]:
    """Variance components per (line, arm), pooled, plus the authoring-resampled contrast.

    ``contrast_rows`` defaults to ``rows`` but should be the UNTRUNCATED set. The decomposition
    needs a nest balanced per ``(line, arm)``; the contrast crosses two arms inside one chain and
    does its own seed matching. Feeding it the per-unit cut hands it two disjoint seed sets —
    measured 2026-08-24, that produced zero contrasts and printed nothing at all.
    """
    nested: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        nested[f"{r.line}|{r.arm}"][r.chain][r.candidate].append(r.score)

    cells: dict[str, dict[str, float]] = {}
    refused: list[dict[str, Any]] = []
    for key, by_chain in sorted(nested.items()):
        comp = nested_variance_components({c: dict(v) for c, v in by_chain.items()})
        if comp is not None:
            cells[key] = comp
            continue
        # A refusal used to vanish silently, so a run that decomposed 3 units of 55 read exactly
        # like one that decomposed all 55. Record the SHAPE rather than re-deriving the estimator's
        # rule: the estimator stays the authority on what is balanced, and this only says what it
        # saw. ``chains: 1`` or ``seed_counts: [3, 5]`` is enough to tell a reader which it was.
        refused.append({
            "unit": key,
            "chains": len(by_chain),
            "candidates_per_chain": sorted({len(v) for v in by_chain.values()}),
            "seed_counts": sorted({len(s) for v in by_chain.values() for s in v.values()}),
        })

    pooled: dict[str, float] = {}
    spread: dict[str, list[float]] = {}
    if cells:
        for level in ("sigma2_seed", "sigma2_candidate", "sigma2_chain"):
            vals = [c[level] for c in cells.values()]
            # An UNWEIGHTED mean across cells, so every (line, arm) counts once regardless of how
            # many candidates or seeds it happens to hold. That is the right weighting for the
            # question ("how big is authoring noise for a typical model x arm") but it means a
            # thin cell carries as much weight as a full one, so the SPREAD is reported beside it.
            pooled[level] = float(np.mean(vals))
            spread[level] = [float(np.min(vals)), float(np.max(vals))]
        for level in ("seed", "candidate", "chain"):
            pooled[f"sd_{level}"] = math.sqrt(pooled[f"sigma2_{level}"])
    present = {r.line for r in rows}
    # NOTE: A LINE WE CUT OURSELVES IS NOT A LINE THE RUN NEVER REACHED, and reporting the two as one
    # thing is precisely the "missing data read as a null" error this file exists to prevent. Until
    # ``lines_loaded`` was threaded through, the candidate-balancing step could remove a line and the
    # report would then announce it had NO RECORDS AT ALL.
    loaded = set(ALL_LINES) if lines_loaded is None else set(lines_loaded)
    return {
        "n_records": len(rows),
        "n_lines": len(present),
        "lines_absent": sorted(set(ALL_LINES) - loaded),
        "lines_cut_by_our_own_balancing": sorted(loaded - present),
        "n_chains": len({(r.line, r.chain) for r in rows}),
        "cells": cells,
        "spread": spread,
        # A component estimated from few degrees of freedom can come out negative and is clamped to
        # zero. Clamping is one-sided, so a run where most cells clamp has a pooled figure biased
        # UPWARD, and the reader has to be told how often it happened.
        "cells_clamped": int(sum(1 for c in cells.values() if c.get("clamped", 0) > 0)),
        "units_refused_as_unbalanced": refused,
        "n_units_seen": len(nested),
        "pooled": pooled,
        "contrast": arm_contrast_over_authoring(
            rows if contrast_rows is None else contrast_rows),
        "program_contrast": program_level_contrast(
            rows if contrast_rows is None else contrast_rows),
        # THE SAME 2 CONTRASTS AGAINST EVERY REGISTERED COMPARATOR, NOT ONLY AGAINST scalar.
        # Decision of 2026-08-27: "for 30 seeds for these 10 candidates we have had other arms as well
        # no? Should it be somehow integrated as well?" The sub-experiment ran all 5 arms and the
        # decomposition above already uses every decomposable cell, yet both contrasts reported
        # only distributional against scalar. NOTE: IT IS NOT A NEW AXIS. These 4 ARE the comparator
        # family the main campaign is registered against, so the resample now mirrors the campaign
        # rather than widening it, and reporting ALL of them is what keeps it from being a choice.
        "comparator_contrasts": {
            control: {
                "run": arm_contrast_over_authoring(
                    rows if contrast_rows is None else contrast_rows, control=control),
                "reward": program_level_contrast(
                    rows if contrast_rows is None else contrast_rows, control=control),
            }
            for control in COMPARATORS
        },
    }


def render(result: dict[str, Any], metric: str) -> str:
    """A human-readable report. The numbers carry their conditions, per the register."""
    L: list[str] = []
    L.append(f"AUTHORING VARIANCE — sealed-window {metric}")
    L.append("=" * 78)
    L.append(f"records {result['n_records']}   lines {result['n_lines']}   "
             f"(line, chain) pairs {result['n_chains']}   "
             f"decomposable (line, arm) cells {len(result['cells'])}")
    absent = result.get("lines_absent") or []
    if absent:
        L.append("")
        L.append(f"NOTE: NO RECORDS AT ALL for {len(absent)} of {len(ALL_LINES)} lines: "
                 f"{', '.join(absent)}")
        L.append("  These are NOT nulls and must not be reported as one. They are lines the run "
                 "has not reached.")
    cut = result.get("lines_cut_by_our_own_balancing") or []
    if cut:
        L.append("")
        L.append(f"NOTE: {len(cut)} further line(s) DO hold records but were removed by OUR OWN cuts, "
                 f"not by the run: {', '.join(cut)}")
        L.append("  A line we cut is not a line that produced nothing. It is a line whose chains do "
                 "not yet")
        L.append("  hold a balanced, seed-matched nest.")
    skipped = result.get("records_skipped_at_load") or {}
    if skipped:
        L.append("")
        L.append(f"NOTE: {sum(skipped.values())} record(s) on disk were SKIPPED at load and are in "
                 f"none of the numbers below:")
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            L.append(f"    {n:>6}  {reason}")
    refused = result.get("units_refused_as_unbalanced") or []
    if refused:
        L.append("")
        L.append(f"NOTE: {len(refused)} of {result.get('n_units_seen', '?')} (line, arm) units were "
                 f"REFUSED as unbalanced and contribute NOTHING to the pooled numbers below.")
        L.append("  A refusal is missing data, never a null. The shape the estimator saw:")
        for u in refused[:12]:
            L.append(f"    {u['unit']:<34} chains={u['chains']}  "
                     f"candidates/chain={u['candidates_per_chain']}  seeds={u['seed_counts']}")
        if len(refused) > 12:
            L.append(f"    ... and {len(refused) - 12} more")
    p = result.get("pooled") or {}
    if p:
        L.append("")
        L.append("Variance of one program's sealed-window score, by what is resampled:")
        L.append("")
        n_cells = len(result.get("cells") or {})
        L.append(f"  pooled over {n_cells} decomposable (line, arm) cell(s), each counted once")
        L.append("")
        L.append(f"  {'level':<38}{'sigma':>10}{'share':>9}"
                 f"{'sigma across cells (min-max)':>34}")
        spread = result.get("spread") or {}
        total = sum(p[f"sigma2_{k}"] for k in ("seed", "candidate", "chain"))
        rows = [("RL seed (what the campaign varied)", "seed"),
                ("which program of the generation", "candidate"),
                ("which authoring run (Okhrati's point)", "chain")]
        if total <= 0.0:
            L.append("  every component estimated at zero — the shares below are undefined, so "
                     "they are not printed")
            for label, k in rows:
                L.append(f"  {label:<38}{p[f'sd_{k}']:>10.4f}{'n/a':>9}")
        else:
            for label, k in rows:
                lo, hi = spread.get(f"sigma2_{k}", [float("nan"), float("nan")])
                rng = f"{math.sqrt(max(0.0, lo)):.4f} - {math.sqrt(max(0.0, hi)):.4f}"
                L.append(f"  {label:<38}{p[f'sd_{k}']:>10.4f}"
                         f"{100 * p[f'sigma2_{k}'] / total:>8.1f}%{rng:>34}")
        clamped = int(result.get("cells_clamped") or 0)
        if clamped:
            L.append("")
            L.append(f"  NOTE: {clamped} of {n_cells} cell(s) produced a NEGATIVE component estimate, "
                     f"clamped to zero.")
            L.append("    Clamping is one-sided, so the pooled figures above are biased UPWARD by "
                     "it. With two")
            L.append("    chains the chain component carries one degree of freedom per cell, so "
                     "read the pooled")
            L.append("    number and never a single cell.")
    c = result.get("contrast") or {}
    L.append("")
    L.append(f"{TREATMENT} minus {CONTROL}, one contrast per (line, chain), each the IQM over the "
             f"seeds BOTH arms hold,")
    L.append("with each arm's per-seed score the IQM over its whole final generation:")
    L.append("")
    if c.get("contrasts"):
        lo, hi = c.get("iqm_delta_ci95") or [float("nan"), float("nan")]
        n_seeds = [x["n_seeds"] for x in c["contrasts"]]
        L.append(f"  contrasts                     {c['n_line_chains']}")
        L.append(f"  matched seeds per contrast    {min(n_seeds)} to {max(n_seeds)}")
        L.append(f"  IQM of the contrast           {c['iqm_delta']:+.4f}"
                 f"   95% CI [{lo:+.4f}, {hi:+.4f}]  ({c.get('ci_n_boot', 0)} draws)")
        L.append(f"  mean of the contrast          {c['mean_delta']:+.4f}")
        L.append(f"  SD across (line, chain)       {c['sd_delta']:.4f}")
        L.append(f"  treatment ahead in            {c['treatment_wins']} of {c['n_line_chains']}")
    else:
        L.append(f"  NO CONTRAST COULD BE FORMED. {c.get('note', 'reason not recorded')}")
    # A dropped (line, chain) is missing data and must be visible. It printed nothing at all until
    # 2026-08-24, so a run that formed ZERO contrasts looked identical to one that had no arms.
    drop = c.get("dropped") or []
    if drop:
        why = defaultdict(int)
        for d in drop:
            why[d["why"]] += 1
        L.append("")
        L.append(f"  NOTE: {len(drop)} (line, chain) pair(s) formed NO contrast:")
        for reason, n in sorted(why.items(), key=lambda kv: -kv[1]):
            L.append(f"      {n:>4}  {reason}")
        if c.get("lines_with_both_chains"):
            L.append(f"  lines whose TWO CHAINS DISAGREE IN SIGN   "
                     f"{c['lines_where_chains_disagree_in_sign']} of {c['lines_with_both_chains']}")
            if c.get("sign_flip_lines"):
                L.append(f"    {', '.join(c['sign_flip_lines'])}")
            L.append("    (each of these is a model where a single authoring run would have")
            L.append("     reported the opposite direction — the point Dr Okhrati raised)")

    g = result.get("program_contrast") or {}
    L.append("")
    L.append(f"One {TREATMENT} program against one {CONTROL} program of the SAME authoring run, "
             f"seed-matched:")
    L.append("")
    if g.get("n_pairs"):
        lo, hi = g["p_treatment_ahead_ci95"]
        L.append(f"  program pairs                 {g['n_pairs']} over {g['n_lines']} model(s), "
                 f"{g['n_seeds_min']} to {g['n_seeds_max']} matched seeds each")
        L.append(f"  P({TREATMENT} ahead)     {g['p_treatment_ahead']:.4f}"
                 f"   95% CI [{lo:.4f}, {hi:.4f}]  ({g.get('ci_n_boot', 0)} draws, "
                 f"models resampled)")
        L.append(f"    {g['pairs_treatment_ahead']} of {g['n_pairs']} pairs, against 0.5000 if the "
                 f"arm decided nothing")
        L.append(f"  the same count across runs    {g['p_treatment_ahead_across_runs']:.4f}"
                 f"   ({g['n_pairs_across_runs']} pairs, the run no longer held fixed)")
        L.append(f"  spread of one pair            10th to 90th percentile "
                 f"{g['delta_p10']:+.4f} to {g['delta_p90']:+.4f}, "
                 f"SD {g['delta_sd']:.4f}")
        L.append(f"  widest pair either way        {g['delta_min']:+.4f} to {g['delta_max']:+.4f}")
        L.append("")
        L.append(f"  {'model':<20}{'pairs':>7}{'P(ahead)':>10}{'IQM of the pair':>18}")
        for line, v in g["per_line"].items():
            L.append(f"  {line:<20}{v['n_pairs']:>7}{v['p_treatment_ahead']:>10.3f}"
                     f"{v['iqm_delta']:>+18.4f}")
        L.append("")
        L.append("  (a per-model row comes from ~10 programs, not from its pair count, so it "
                 "carries no interval)")
    else:
        L.append(f"  NO PROGRAM PAIR COULD BE FORMED. {g.get('note', 'reason not recorded')}")

    cc = result.get("comparator_contrasts") or {}
    if cc:
        L.append("")
        L.append("THE SAME 2 CONTRASTS AGAINST EVERY REGISTERED COMPARATOR. All 4 are printed "
                 "together")
        L.append("so that quoting one of them cannot be a choice among four.")
        L.append("")
        L.append(f"  {'against':<18}{'runs ahead':>11}{'IQM':>9}{'95% CI':>22}"
                 f"{'pairs ahead':>13}{'P(ahead)':>10}{'95% CI':>20}")
        for control in COMPARATORS:
            r, w = cc[control]["run"], cc[control]["reward"]
            rlo, rhi = r.get("iqm_delta_ci95") or [float("nan"), float("nan")]
            wlo, whi = w.get("p_treatment_ahead_ci95") or [float("nan"), float("nan")]
            L.append(f"  {control:<18}{r['treatment_wins']:>5}/{r['n_line_chains']:<5}"
                     f"{r['iqm_delta']:>+9.4f}  [{rlo:+8.4f},{rhi:+8.4f}]"
                     f"{w['pairs_treatment_ahead']:>6}/{w['n_pairs']:<6}"
                     f"{w['p_treatment_ahead']:>10.4f}  [{wlo:.4f},{whi:.4f}]")
        excl = [c for c in COMPARATORS
                if (cc[c]["run"].get("iqm_delta_ci95") or [0.0, 0.0])[0] > 0.0
                or (cc[c]["run"].get("iqm_delta_ci95") or [0.0, 0.0])[1] < 0.0]
        L.append("")
        L.append(f"  run-level intervals that exclude zero: {len(excl)} of {len(COMPARATORS)}"
                 + (f" ({', '.join(excl)})" if excl else ""))
        cover = [c for c in COMPARATORS
                 if (cc[c]["reward"].get("p_treatment_ahead_ci95") or [1.0, 0.0])[0] <= 0.5
                 <= (cc[c]["reward"].get("p_treatment_ahead_ci95") or [1.0, 0.0])[1]]
        L.append(f"  reward-level intervals that cover a coin toss: {len(cover)} of "
                 f"{len(COMPARATORS)}")
    return "\n".join(L)


def selftest() -> int:
    """Recover known variance components from synthetic data, with falsifiers."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    check("candidate_id parses", parse_candidate_id("distributional-g1-c3") == ("distributional", 1, 3))
    check("a TEST LABEL is rejected (falsifier)", parse_candidate_id("distributional__c3") is None,
          "the record's `arm` field is a test label and must never be read as an arm")

    check("the line list comes from the driver and is archive-spelled",
          len(ALL_LINES) == 11 and "opus_5" in ALL_LINES and "qwen3_6_27b" in ALL_LINES,
          f"{len(ALL_LINES)} lines: {', '.join(ALL_LINES)}")

    trunc_rows = ([Record("L", 1, "scalar", 0, sd, 1.0) for sd in range(7)]
                  + [Record("L", 1, "scalar", 1, sd, 2.0) for sd in range(3)])
    kept, info = truncate_to_seed_floor(trunc_rows, 5)
    check("a unit whose candidates share fewer than n seeds is dropped WHOLE",
          kept == [] and info["units_dropped_for_no_common_floor"] == 1
          and info["candidates_dropped_as_short"] == 2,
          f"kept {len(kept)}, units dropped {info['units_dropped_for_no_common_floor']}")

    # The 2026-08-24 defect, as a test that FAILS against the old per-candidate rule: both
    # candidates hold five seeds, so the old code kept 0-4 from one and 0,1,7,8,9 from the other and
    # called the unit balanced. The seeds must MATCH, not merely be equal in number.
    holed = ([Record("L", 1, "scalar", 0, sd, 1.0) for sd in (0, 1, 2, 3, 4, 7, 8, 9)]
             + [Record("L", 1, "scalar", 1, sd, 2.0) for sd in (0, 1, 7, 8, 9)])
    kept_h, info_h = truncate_to_seed_floor(holed, 5)
    check("candidates with holed seeds are cut to their COMMON seeds, not their own lowest",
          {r.seed for r in kept_h} == {0, 1, 7, 8, 9} and len(kept_h) == 10
          and info_h["units_dropped_for_no_common_floor"] == 0,
          f"kept seeds {sorted({r.seed for r in kept_h})}, n={len(kept_h)}")

    balanced = [Record("L", 1, "scalar", c, sd, 1.0) for c in (0, 1) for sd in range(5)]
    check("truncation is a no-op when nothing exceeds the floor (falsifier)",
          sorted(truncate_to_seed_floor(balanced, 5)[0], key=lambda r: (r.candidate, r.seed))
          == balanced)

    rng = np.random.default_rng(0)
    sd_seed, sd_cand, sd_chain = 0.10, 0.30, 0.50
    by_chain: dict[int, dict[int, list[float]]] = {}
    for c in range(6):                       # more chains than the run has, so the check is sharp
        chain_effect = rng.normal(0, sd_chain)
        by_chain[c] = {}
        for k in range(5):
            cand_effect = chain_effect + rng.normal(0, sd_cand)
            by_chain[c][k] = [cand_effect + rng.normal(0, sd_seed) for _ in range(30)]
    comp = nested_variance_components(by_chain)
    assert comp is not None
    got = {k: math.sqrt(comp[f"sigma2_{k}"]) for k in ("seed", "candidate", "chain")}
    check("recovers sigma_seed", abs(got["seed"] - sd_seed) < 0.02, f"{got['seed']:.3f} vs {sd_seed}")
    check("recovers sigma_candidate", abs(got["candidate"] - sd_cand) < 0.10,
          f"{got['candidate']:.3f} vs {sd_cand}")
    check("recovers sigma_chain", abs(got["chain"] - sd_chain) < 0.30,
          f"{got['chain']:.3f} vs {sd_chain}")

    # The PERMANENT imbalance: a rejected g1 candidate leaves one chain with 4 against the other's
    # 5, and no quantity of seeds fixes it. Cut to the common count by LOWEST index (outcome-blind).
    bal_rows = ([Record("L", 1, "scalar", c, s, 1.0) for c in range(5) for s in range(3)]
                + [Record("L", 1, "scalar", c, s, 2.0) for c in (0, 1, 2, 4) for s in range(3)])
    # chain 1 holds candidates 0-4, chain 2 holds 0,1,2,4 -> both must end at {0,1,2}
    bal_rows = [r if r.score == 1.0 else Record(r.line, 2, r.arm, r.candidate, r.seed, r.score)
                for r in bal_rows]
    kept_b, info_b = balance_candidates_across_chains(bal_rows)
    per_chain = {ch: sorted({r.candidate for r in kept_b if r.chain == ch}) for ch in (1, 2)}
    # NOTE: The contract is equal COUNTS, not a common SET, and that is the opposite of the seed rule.
    # Seeds are CROSSED (common random numbers: seed 3 is the same draw in both chains, so the
    # values must match). Candidates are NESTED inside a chain: chain 1's c0 and chain 2's c0 are
    # two unrelated authored programs, so requiring the same indices would discard data for nothing.
    check("chains are cut to a COMMON candidate COUNT, keeping each chain's lowest indices",
          [len(v) for v in per_chain.values()] == [4, 4]
          and per_chain == {1: [0, 1, 2, 3], 2: [0, 1, 2, 4]}
          and info_b["units_kept"] == 1,
          f"{per_chain}")
    thin = ([Record("T", 1, "scalar", 0, s, 1.0) for s in range(3)]
            + [Record("T", 2, "scalar", c, s, 2.0) for c in range(5) for s in range(3)])
    check("a unit whose thinner chain holds ONE candidate is dropped (falsifier)",
          balance_candidates_across_chains(thin)[0] == []
          and balance_candidates_across_chains(thin)[1]["units_dropped_for_too_few_candidates"] == 1)

    # Falsifier: an UNBALANCED nest must refuse, not silently use the wrong estimator.
    unbalanced = {0: {0: [1.0, 2.0], 1: [1.0, 2.0]}, 1: {0: [1.0, 2.0]}}
    check("unbalanced nest returns None (falsifier)",
          nested_variance_components(unbalanced) is None)
    check("one chain returns None (falsifier)",
          nested_variance_components({0: {0: [1.0, 2.0], 1: [1.0, 2.0]}}) is None)

    # SEED MATCHING. The two arms hold {0,1,2,3} and {2,3,4,5}; on the two seeds they SHARE they are
    # identical, so the honest contrast is exactly zero. The old pooled-IQM version answered with the
    # control's private seeds 4 and 5 folded in, which is a different and much larger number — so
    # this check FAILS against it rather than merely passing with it.
    from src.inference.reporting import iqm as _iqm
    m_rows = ([Record("M", 1, TREATMENT, 0, s, 1.0) for s in (0, 1, 2, 3)]
              + [Record("M", 1, CONTROL, 0, s, 1.0) for s in (2, 3)]
              + [Record("M", 1, CONTROL, 0, s, -10.0) for s in (4, 5)])
    m_out = arm_contrast_over_authoring(m_rows)
    pooled_delta = float(_iqm(np.array([1.0] * 4)) - _iqm(np.array([1.0, 1.0, -10.0, -10.0])))
    check("the contrast uses only seeds BOTH arms hold",
          len(m_out["contrasts"]) == 1
          and abs(m_out["contrasts"][0]["delta"]) < 1e-12
          and m_out["contrasts"][0]["n_seeds"] == 2
          and abs(pooled_delta) > 1e-6,
          f"seed-matched {m_out['contrasts'][0]['delta']:+.4f} vs old pooled {pooled_delta:+.4f}")

    # A seed where only SOME of an arm's candidates have a record is not a generation score.
    p_rows = ([Record("P", 1, TREATMENT, c, s, 1.0) for c in (0, 1) for s in (0, 1)]
              + [Record("P", 1, TREATMENT, 0, 2, 9.0)]          # candidate 1 missing at seed 2
              + [Record("P", 1, CONTROL, 0, s, 0.0) for s in (0, 1, 2)])
    p_out = arm_contrast_over_authoring(p_rows)
    check("a seed missing one candidate is excluded from that arm (falsifier)",
          p_out["contrasts"][0]["n_seeds"] == 2
          and p_out["contrasts"][0]["seeds_dropped_treatment"] == 1,
          f"n_seeds={p_out['contrasts'][0]['n_seeds']}, "
          f"dropped={p_out['contrasts'][0]['seeds_dropped_treatment']}")

    # CLUSTERING. Each model here contributes two chains with the SAME delta, so a model is one
    # observation, not two. An interval that resampled contrasts would be too tight. This check
    # FAILS if the cluster structure is ever dropped.
    from src.inference.reporting import stratified_bootstrap_ci as _sbci
    cl_rows: list[Record] = []
    for li, d in enumerate([-0.6, -0.2, 0.0, 0.3, 0.9]):
        for ch in (1, 2):
            for sd in range(3):
                cl_rows.append(Record(f"L{li}", ch, TREATMENT, 0, sd, d))
                cl_rows.append(Record(f"L{li}", ch, CONTROL, 0, sd, 0.0))
    cl = arm_contrast_over_authoring(cl_rows)
    cl_lo, cl_hi = cl["iqm_delta_ci95"]
    naive_deltas = np.asarray([x["delta"] for x in cl["contrasts"]], dtype=float)
    _, n_lo, n_hi = _sbci(naive_deltas, n_boot=2000, ci=0.95, rng=np.random.default_rng(0))
    check("the interval resamples MODELS, so it is WIDER than resampling contrasts",
          cl["n_lines_resampled"] == 5 and len(cl["contrasts"]) == 10
          and (cl_hi - cl_lo) > (n_hi - n_lo) * 1.05,
          f"clustered width {cl_hi - cl_lo:.4f} vs naive {n_hi - n_lo:.4f}")

    check("a contrast with too few matched seeds is dropped, not silently weighted equally",
          len(arm_contrast_over_authoring(
              [Record("Z", 1, TREATMENT, 0, 0, 1.0), Record("Z", 1, CONTROL, 0, 0, 0.0)],
              min_seeds=2)["contrasts"]) == 0)

    # A sign flip between two chains of one line must be COUNTED, because it is the headline.
    rows = [r for sd in range(3) for r in (
        Record("m1", 1, TREATMENT, 0, sd, 1.0), Record("m1", 1, CONTROL, 0, sd, 0.0),
        Record("m1", 2, TREATMENT, 0, sd, 0.0), Record("m1", 2, CONTROL, 0, sd, 1.0))]
    out = arm_contrast_over_authoring(rows)
    check("a two-chain sign disagreement is counted",
          out["lines_where_chains_disagree_in_sign"] == 1, f"got {out}")

    # ---- the PROGRAM-level contrast: one authored program against one authored program ----------
    # Run 1 writes 1 treatment program that wins and 2 controls; run 2 writes 2 treatment programs
    # that lose and 1 control. Held within a run that is 2 wins and 2 losses, so 0.5. Crossed it is
    # 3 wins of 9, so 0.3333. The two numbers DIFFER by construction, which is what makes this a
    # falsifier: a version that quietly crossed runs would report 0.3333 as the headline.
    g_rows = ([Record("A", 1, TREATMENT, 0, s, 1.0) for s in range(3)]
              + [Record("A", 1, CONTROL, c, s, 0.0) for c in (0, 1) for s in range(3)]
              + [Record("A", 2, TREATMENT, c, s, -1.0) for c in (0, 1) for s in range(3)]
              + [Record("A", 2, CONTROL, 0, s, 0.0) for s in range(3)])
    g = program_level_contrast(g_rows)
    check("program pairs are counted within an authoring run",
          g["n_pairs"] == 4 and g["pairs_treatment_ahead"] == 2
          and abs(g["p_treatment_ahead"] - 0.5) < 1e-12,
          f"{g['pairs_treatment_ahead']} of {g['n_pairs']}")
    check("crossing runs is reported SEPARATELY and is a different number (falsifier)",
          g["n_pairs_across_runs"] == 9
          and abs(g["p_treatment_ahead_across_runs"] - 3 / 9) < 1e-12
          and abs(g["p_treatment_ahead"] - g["p_treatment_ahead_across_runs"]) > 0.1,
          f"within {g['p_treatment_ahead']:.4f} vs crossed "
          f"{g['p_treatment_ahead_across_runs']:.4f}")

    # SEED MATCHING again, one level down. The two programs share only seeds 2 and 3, where they are
    # identical, so the honest difference is exactly zero. Pooling each program's own seeds would
    # answer with the control's private 4 and 5 and return a large positive number instead.
    gm_rows = ([Record("M", 1, TREATMENT, 0, s, 1.0) for s in (0, 1, 2, 3)]
               + [Record("M", 1, CONTROL, 0, s, 1.0) for s in (2, 3)]
               + [Record("M", 1, CONTROL, 0, s, -10.0) for s in (4, 5)])
    gm = program_level_contrast(gm_rows)
    check("a program pair uses only the seeds BOTH programs hold (falsifier)",
          gm["n_pairs"] == 1 and abs(gm["delta_p50"]) < 1e-12
          and gm["n_seeds_min"] == gm["n_seeds_max"] == 2,
          f"delta {gm['delta_p50']:+.4f} on {gm['n_seeds_min']} matched seed(s)")

    check("a program pair sharing too few seeds forms nothing, in the same output shape",
          program_level_contrast(
              [Record("Z", 1, TREATMENT, 0, 0, 1.0), Record("Z", 1, CONTROL, 0, 0, 0.0)],
              min_seeds=2)["n_pairs"] == 0)

    # CLUSTERING, at the program level. Three models win every pair and two lose every pair, so the
    # answer depends entirely on WHICH MODELS are drawn. Resampling the 20 pairs instead would report
    # an interval several times too tight, and this check fails if the cluster unit is ever dropped.
    gc_rows: list[Record] = []
    for li in range(5):
        val = 1.0 if li < 3 else -1.0
        for c in (0, 1):
            for s in range(3):
                gc_rows.append(Record(f"L{li}", 1, TREATMENT, c, s, val))
                gc_rows.append(Record(f"L{li}", 1, CONTROL, c, s, 0.0))
    gc = program_level_contrast(gc_rows)
    gc_lo, gc_hi = gc["p_treatment_ahead_ci95"]
    _rng = np.random.default_rng(0)
    _flat = np.asarray([1.0] * 12 + [0.0] * 8)
    _naive = np.asarray([float(np.mean(_rng.choice(_flat, size=_flat.size, replace=True)))
                         for _ in range(2000)])
    _n_lo, _n_hi = float(np.quantile(_naive, 0.025)), float(np.quantile(_naive, 0.975))
    check("the program interval resamples MODELS, so it is WIDER than resampling pairs",
          gc["n_pairs"] == 20 and gc["n_lines"] == 5
          and abs(gc["p_treatment_ahead"] - 0.6) < 1e-12
          and (gc_hi - gc_lo) > (_n_hi - _n_lo) * 1.5,
          f"clustered width {gc_hi - gc_lo:.4f} vs pair-level {_n_hi - _n_lo:.4f}")

    # ---- the SAME 2 contrasts against every registered comparator -------------------------------
    # One line, 1 run. The treatment beats scalar, ties scalar_cvar5 and loses to placebo, so a
    # block that quietly reused ONE control for all of them would return 3 identical rows. That is
    # the failure this falsifier is built to catch, and it is the one a reader could never see.
    cmp_rows = ([Record("A", 1, TREATMENT, 0, s, 1.0) for s in range(3)]
                + [Record("A", 1, "scalar", 0, s, 0.0) for s in range(3)]
                + [Record("A", 1, "scalar_cvar5", 0, s, 1.0) for s in range(3)]
                + [Record("A", 1, "placebo", 0, s, 2.0) for s in range(3)]
                + [Record("A", 1, "placebo_shuffled", 0, s, -1.0) for s in range(3)])
    cc = analyse(cmp_rows)["comparator_contrasts"]
    check("every registered comparator gets its own contrast",
          tuple(cc) == COMPARATORS and len(COMPARATORS) == 4,
          f"got {tuple(cc)}")
    check("the 4 comparators return 4 DIFFERENT answers (falsifier)",
          cc["scalar"]["reward"]["p_treatment_ahead"] == 1.0
          and cc["scalar_cvar5"]["reward"]["p_treatment_ahead"] == 0.0
          and cc["placebo"]["reward"]["p_treatment_ahead"] == 0.0
          and cc["placebo_shuffled"]["reward"]["p_treatment_ahead"] == 1.0
          and abs(cc["placebo"]["run"]["iqm_delta"] + 1.0) < 1e-12
          and abs(cc["placebo_shuffled"]["run"]["iqm_delta"] - 2.0) < 1e-12,
          f"scalar {cc['scalar']['reward']['p_treatment_ahead']}, "
          f"placebo run {cc['placebo']['run']['iqm_delta']}")
    check("the registered contrast is the same object under both names",
          cc[CONTROL]["run"]["iqm_delta"] == analyse(cmp_rows)["contrast"]["iqm_delta"]
          and cc[CONTROL]["reward"]["n_pairs"] == analyse(cmp_rows)["program_contrast"]["n_pairs"],
          "the scalar row and the headline contrast disagree")

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


def main() -> int:
    # The report carries a warning sign, and a Windows console defaults to a codepage that cannot
    # encode it -- `print` then raises UnicodeEncodeError and the whole analysis is lost AFTER the
    # computation has finished. Caught 2026-08-25 by running this on partial data before the real
    # read. Setting PYTHONIOENCODING=utf-8 also works, but relying on the operator remembering it
    # puts a crash between a finished run and its result.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs/authoring_variance")
    ap.add_argument("--metric", default=DEFAULT_METRIC)
    ap.add_argument("--json-out", default=None, help="Also write the result as JSON.")
    ap.add_argument("--max-seeds", type=int, default=None,
                    help="Cut every (line, arm) unit to the N lowest seeds COMMON to all of its "
                         "candidates, and drop the unit when they share fewer, so a PRELIMINARY "
                         "read is balanced AND seed-matched. Without it a part-finished run "
                         "decomposes nothing, because the estimator refuses an unbalanced nest.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    rows, skipped = load_records(args.output_dir, metric=args.metric)
    raw_rows = list(rows)          # the contrast needs these, NOT the per-unit cut ones
    if not rows:
        print(f"no sealed-window records under {args.output_dir} yet")
        return 0
    # ALWAYS, not only under --max-seeds: the candidate imbalance is permanent (a rejected g1
    # candidate never returns), so the FINAL analysis needs this step exactly as much as a
    # preliminary one does. Measured 2026-08-24: 19 of the 55 pairs were unbalanced this way.
    rows, bal = balance_candidates_across_chains(rows)
    print(f"balanced the candidate count across chains: kept {bal['units_kept']} (line, arm) "
          f"unit(s), dropped {bal['units_dropped_for_too_few_candidates']} whose thinner chain "
          f"holds under 2 candidates, trimmed {bal['candidates_trimmed_for_balance']} candidate(s) "
          f"and {bal['records_discarded']} record(s) for balance")
    if not rows:
        print("no (line, arm) unit has two chains with two candidates each yet")
        return 0

    if args.max_seeds:
        rows, trunc = truncate_to_seed_floor(rows, args.max_seeds)
        print(f"balanced to {args.max_seeds} COMMON seed(s) per (line, arm) unit: kept "
              f"{trunc['candidates_kept']} candidate(s), dropped "
              f"{trunc['candidates_dropped_as_short']} in "
              f"{trunc['units_dropped_for_no_common_floor']} unit(s) whose candidates share fewer "
              f"than {args.max_seeds} seeds, discarded {trunc['records_discarded']} record(s) "
              f"outside the cut" + "\n")
        if not rows:
            print(f"nothing reaches {args.max_seeds} seeds yet")
            return 0
    result = analyse(rows, contrast_rows=raw_rows,
                     lines_loaded={r.line for r in raw_rows})
    result["records_skipped_at_load"] = skipped
    print(render(result, args.metric))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
