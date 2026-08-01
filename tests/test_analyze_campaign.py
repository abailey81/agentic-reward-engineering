"""Fast tests for the campaign PBO/CSCV analysis (scripts/analyze_campaign.py).

No torch, no real training: synthetic per-candidate run records with KNOWN per-period
validation vectors drive ``build_perf_matrix`` and ``campaign_pbo``. Covers:

  - ``build_perf_matrix`` returns a ``(T_val, N_candidates)`` matrix of the stacked vectors;
  - PBO ∈ [0, 1];
  - a CLEAN fixture (in-sample candidate rank == out-of-sample rank) -> PBO near 0;
  - a pure-NOISE fixture (IS-best is random OOS) -> PBO near/above 0.5;
  - an arm with too few candidates is handled gracefully (reported as skipped, no raise);
  - candidates lacking a validation vector are skipped (build_perf_matrix);
  - the markdown emitter renders a row per arm.

These mirror the proven ``tests/test_inference.py`` PBO fixtures, lifted to the campaign
record layer (the columns are an ARM's candidates, per the spec).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402

N_BLOCKS = 10


def _record(arm: str, cid: str, val_vec: np.ndarray | None) -> dict:
    """A minimal per-candidate record carrying ``metrics['val_returns']`` (or omitting it)."""
    metrics: dict = {"val_fitness": float(np.mean(val_vec)) if val_vec is not None else 0.0}
    if val_vec is not None:
        metrics["val_returns"] = [float(x) for x in val_vec]
    return {
        "run_id": f"{arm}-{cid}",
        "arm": arm,
        "seed": 0,
        "fold": 0,
        "candidate_id": cid,
        "generation": 0,
        "reward_source_hash": "deadbeef",
        "feedback_block": "",
        "metrics": metrics,
        "wall_clock": 0.0,
        "env_fingerprint": "test",
    }


def _records_from_matrix(arm: str, matrix: np.ndarray) -> list[dict]:
    """Wrap each COLUMN of ``matrix`` (a candidate's val vector) in a record."""
    return [
        _record(arm, f"{arm}-c{j:02d}", matrix[:, j])
        for j in range(matrix.shape[1])
    ]


# --------------------------------------------------------------------------- #
# winner_dsr — canonical cross-trial variance (final-acceptance-audit P1)      #
# --------------------------------------------------------------------------- #
def test_winner_dsr_var_sr_is_per_period_not_annualized() -> None:
    """winner_dsr deflates by the PER-PERIOD cross-trial Sharpe variance, not a 252x one.

    Regression for the final-acceptance-audit P1: analyze_campaign computed var_sr from
    ANNUALIZED per-candidate Sharpes (sharpe_ratio default periods_per_year=252) while
    deflated_sharpe_ratio compares a PER-PERIOD winner Sharpe to sr_star = sqrt(var_sr)*term,
    so var_sr was ~252x too large -> sr_star ~15.87x -> dsr_canonical collapsed spuriously to ~0.
    This calls winner_dsr DIRECTLY (the function was never invoked under test, so the units bug
    slipped through test_inference.py which hand-computes var_sr and bypasses the internal call).
    """
    rng = np.random.default_rng(7)
    t = 400
    # A heterogeneous (dispersed-skill) population with DISTINCT per-period Sharpes.
    vecs: list[np.ndarray] = []
    for mu, sd in [(0.012, 0.05), (0.004, 0.05), (0.008, 0.06), (0.001, 0.05), (0.015, 0.05)]:
        v = rng.standard_normal(t)
        v = (v - v.mean()) / v.std(ddof=1) * sd + mu  # force exact per-period (mu, sd)
        vecs.append(v)
    records = [_record("distributional", f"c{j}", v) for j, v in enumerate(vecs)]

    entry = AC.winner_dsr(records)["distributional"]
    assert entry["status"] == "ok"

    # The canonical input is the variance of the PER-PERIOD Sharpes, built with the SAME ddof=1 sample-std
    # convention winner_dsr uses (``_sample_moments`` — so var_sr's dispersion and the winner Sharpe the DSR
    # compares it against share one std convention), NOT bootstrap.sharpe_ratio's ddof=0 population std and
    # NOT the annualized (default 252) value the original P1 bug used.
    from src.inference.bootstrap import sharpe_ratio
    from src.inference.deflated_sharpe import _sample_moments
    pp = np.array([_sample_moments(v)[0] for v in vecs])
    ann = np.array([sharpe_ratio(v) for v in vecs])  # default periods_per_year=252 = the bug
    assert entry["var_sr"] == pytest.approx(float(np.var(pp, ddof=1)), rel=1e-9)
    assert entry["var_sr"] != pytest.approx(float(np.var(ann, ddof=1)), rel=1e-3)  # NOT the ~252x bug value
    # The winner (max val_fitness = max mean, per-period Sharpe ~0.30) clearly beats the N=5
    # expected-max benchmark -> a real, non-collapsed DSR (the bug pinned this near 0).
    assert 0.0 < entry["dsr_canonical"] <= 1.0
    assert entry["dsr_canonical"] > 0.5


# --------------------------------------------------------------------------- #
# build_perf_matrix                                                            #
# --------------------------------------------------------------------------- #
def test_build_perf_matrix_shape_and_values() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((120, 7))
    records = _records_from_matrix("scalar", matrix)
    # Add records from ANOTHER arm to confirm filtering.
    records += _records_from_matrix("placebo", rng.standard_normal((120, 3)))

    out = AC.build_perf_matrix(records, "scalar")
    assert out.shape == (120, 7)
    # Columns are ordered by candidate_id (c00..c06), matching the source columns.
    np.testing.assert_allclose(out, matrix)


def test_build_perf_matrix_skips_candidates_without_vector(caplog) -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((80, 4))
    records = _records_from_matrix("scalar", matrix)
    # One candidate has NO val_returns at all -> must be skipped.
    records.append(_record("scalar", "scalar-cZZ", None))

    out = AC.build_perf_matrix(records, "scalar")
    assert out.shape == (80, 4)  # the 4 with vectors; the bad one dropped


def test_build_perf_matrix_aligns_ragged_lengths() -> None:
    records = [
        _record("scalar", "scalar-c00", np.arange(50, dtype=float)),
        _record("scalar", "scalar-c01", np.arange(40, dtype=float)),
    ]
    out = AC.build_perf_matrix(records, "scalar")
    assert out.shape == (40, 2)  # aligned to the common leading length


def test_build_perf_matrix_empty_when_no_vectors() -> None:
    records = [_record("scalar", "scalar-c00", None)]
    out = AC.build_perf_matrix(records, "scalar")
    assert out.shape == (0, 0)


# --------------------------------------------------------------------------- #
# campaign_pbo — unit interval + the two regimes                              #
# --------------------------------------------------------------------------- #
def test_campaign_pbo_in_unit_interval() -> None:
    rng = np.random.default_rng(3)
    records = _records_from_matrix("scalar", rng.standard_normal((300, 10)))
    out = AC.campaign_pbo(records, n_blocks=N_BLOCKS, arms=("scalar",))
    e = out["scalar"]
    assert e["status"] == "ok"
    assert 0.0 <= e["pbo"] <= 1.0
    assert e["n_candidates"] == 10
    assert e["t_val"] == 300


def test_campaign_pbo_low_on_clean_monotone_ladder() -> None:
    """CLEAN fixture: a fixed quality ladder -> IS rank == OOS rank -> PBO near 0.

    Each candidate j has a distinct constant mean (a ladder) plus tiny noise, so the IS-best
    candidate is the same as the OOS-best on every split: in-sample selection carries full
    out-of-sample information and PBO collapses to ~0.
    """
    rng = np.random.default_rng(4)
    t, n = 400, 8
    ladder = np.linspace(0.0, 1.0, n)  # strictly increasing per-candidate mean
    matrix = ladder[None, :] + 0.01 * rng.standard_normal((t, n))
    records = _records_from_matrix("distributional", matrix)

    out = AC.campaign_pbo(records, n_blocks=N_BLOCKS, arms=("distributional",))
    e = out["distributional"]
    assert e["status"] == "ok"
    assert e["pbo"] == pytest.approx(0.0, abs=0.05)


def test_campaign_pbo_near_half_on_pure_noise() -> None:
    """NOISE fixture: i.i.d. equal-mean candidates -> IS-best random OOS -> PBO ~ 0.5."""
    vals = []
    for seed in range(5):
        rng = np.random.default_rng(200 + seed)
        records = _records_from_matrix("random_search", rng.standard_normal((400, 12)))
        out = AC.campaign_pbo(records, n_blocks=12, arms=("random_search",))
        e = out["random_search"]
        assert e["status"] == "ok"
        vals.append(e["pbo"])
    assert float(np.mean(vals)) == pytest.approx(0.5, abs=0.15)


def test_campaign_pbo_fully_enumerates_at_frozen_s16() -> None:
    """At the FROZEN S=16, C(16,8)=12,870 exceeds the module's old 4,000-split cap, so the PRIMARY
    overfitting guard must FULLY ENUMERATE the CSCV splits (deterministic), not draw a random
    ``max_combinations`` subsample. Behavioural proof: two DIFFERENT rng seeds yield the IDENTICAL
    PBO, because full enumeration never touches the rng (regression for the seed-dependent-headline
    defect — both technical audits flagged it)."""
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((480, 12))  # T_val=480 >> 16 blocks; non-degenerate PBO
    records = _records_from_matrix("distributional", matrix)
    a = AC.campaign_pbo(
        records, n_blocks=16, arms=("distributional",), rng=np.random.default_rng(1)
    )
    b = AC.campaign_pbo(
        records, n_blocks=16, arms=("distributional",), rng=np.random.default_rng(987)
    )
    assert a["distributional"]["status"] == "ok"
    # identical across two unrelated rng seeds <=> the splits were enumerated, not sampled
    assert a["distributional"]["pbo"] == b["distributional"]["pbo"]


# --------------------------------------------------------------------------- #
# graceful degradation                                                         #
# --------------------------------------------------------------------------- #
def test_campaign_pbo_skips_too_few_candidates() -> None:
    """An arm with < 2 candidates is reported as skipped (no raise)."""
    rng = np.random.default_rng(5)
    records = _records_from_matrix("bayes_opt", rng.standard_normal((300, 1)))
    out = AC.campaign_pbo(records, n_blocks=N_BLOCKS, arms=("bayes_opt",))
    e = out["bayes_opt"]
    assert e["status"] == "skipped"
    assert e["pbo"] is None
    assert e["n_candidates"] == 1
    assert "candidates" in e["reason"]


def test_campaign_pbo_skips_short_validation_window() -> None:
    """An arm whose validation window is shorter than n_blocks is skipped, not raised."""
    rng = np.random.default_rng(6)
    # T_val = 6 rows but n_blocks = 10 -> CSCV partition impossible.
    records = _records_from_matrix("scalar", rng.standard_normal((6, 5)))
    out = AC.campaign_pbo(records, n_blocks=N_BLOCKS, arms=("scalar",))
    e = out["scalar"]
    assert e["status"] == "skipped"
    assert e["pbo"] is None
    assert "shorter than" in e["reason"]


def test_campaign_pbo_absent_arm_is_skipped() -> None:
    """An arm with no records at all is reported as skipped with zero candidates."""
    out = AC.campaign_pbo([], n_blocks=N_BLOCKS, arms=("distributional",))
    e = out["distributional"]
    assert e["status"] == "skipped"
    assert e["n_candidates"] == 0


def test_campaign_pbo_all_six_arms_keyed() -> None:
    """The default report keys every pre-registered arm even when most are empty."""
    rng = np.random.default_rng(7)
    records = _records_from_matrix("scalar", rng.standard_normal((200, 10)))
    out = AC.campaign_pbo(records, n_blocks=N_BLOCKS)
    assert set(out) == set(AC.ARMS)
    assert out["scalar"]["status"] == "ok"
    # The five arms with no records are all skipped.
    for arm in AC.ARMS:
        if arm != "scalar":
            assert out[arm]["status"] == "skipped"


# --------------------------------------------------------------------------- #
# H1 — beat_human_baseline (Eureka §18-19; Ma et al. 2024)                      #
# --------------------------------------------------------------------------- #
def _test_record(arm: str, seed: int, returns: np.ndarray, *, val_returns: np.ndarray | None = None) -> dict:
    """A frozen-winner-style TEST record carrying metrics['test_returns'] (the H1 metric reads it).

    ``val_fitness`` defaults to NaN, matching the REAL campaign baseline TEST record
    (``run_campaign._baseline_winner_record`` / ``test_leg.build_test_record`` carry a NaN ``val_fitness``
    and no ``val_returns`` for the un-validation-selected baselines — DEEP_H1 §3.1), so the H1 best-baseline
    identity FALLS BACK to test-selection (flagged). Pass ``val_returns`` to exercise the validation-
    selection path (the data-snoop-free comparator, DEEP_H1 T-REF).
    """
    tr = [float(x) for x in returns]
    metrics: dict = {"val_fitness": float("nan"), "test_returns": tr}
    if val_returns is not None:
        metrics["val_returns"] = [float(x) for x in val_returns]
    return {
        "run_id": f"{arm}-s{seed}",
        "arm": arm,
        "seed": int(seed),
        "fold": 0,
        "candidate_id": f"{arm}-winner",
        "generation": 0,
        "reward_source_hash": "deadbeef",
        "feedback_block": "",
        "metrics": metrics,
        "wall_clock": 0.0,
        "env_fingerprint": "test",
        "frozen": True,
        "test_returns": tr,
    }


def _seeded_test_records(
    arm: str, *, mu: float, n_seeds: int = 8, t: int = 250, seed0: int = 0, val_mu: float | None = None
) -> list[dict]:
    """``n_seeds`` per-seed test records for ``arm``, each a return series with per-period mean ~``mu``.

    A higher ``mu`` (at fixed vol) => a higher per-seed test Sharpe, so the arms' Sharpe ORDER is
    controlled by ``mu`` — letting the test assert the LLM>best / LLM<all regimes deterministically. When
    ``val_mu`` is given each record ALSO carries a ``val_returns`` vector with that per-period mean (so the
    validation-selection path can be exercised independently of the test ordering).
    """
    out: list[dict] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + seed0 + s)
        v = rng.standard_normal(t)
        v = (v - v.mean()) / v.std(ddof=1) * 0.02 + mu  # force per-period (mu, sd=0.02)
        vv = None
        if val_mu is not None:
            vr = np.random.default_rng(5000 + seed0 + s).standard_normal(t)
            vv = (vr - vr.mean()) / vr.std(ddof=1) * 0.02 + val_mu
        out.append(_test_record(arm, s, v, val_returns=vv))
    return out


_H1_BASELINES = ["raw_return", "return_minus_variance", "return_minus_cvar", "differential_sharpe"]


def test_beat_human_baseline_llm_beats_best() -> None:
    """LLM winner Sharpe > every baseline => beat_fraction high, positive normalised improvement."""
    records = _seeded_test_records("distributional", mu=0.010)  # highest mean -> highest Sharpe
    for i, name in enumerate(_H1_BASELINES):
        records += _seeded_test_records(f"baseline_{name}", mu=0.002 + 0.001 * i, seed0=100 * (i + 1))

    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional",
                                winner_n_trials=30)
    assert h1["status"] == "ok"
    assert h1["winner_arm"] == "distributional"
    # The best baseline is the highest-mu one (differential_sharpe, i=3).
    assert h1["best_baseline"] == "return_minus_cvar" or h1["best_baseline"] == "differential_sharpe"
    assert h1["beats_best_baseline_median"] is True
    assert h1["beat_fraction"] > 0.5            # LLM clears the human bar on most seeds
    assert h1["norm_improvement"] is not None and h1["norm_improvement"] > 0.0
    assert h1["winner_n_trials"] == 30          # LLM keeps its searched multiplicity
    # DSR side present; LLM (searched N=30) vs best baseline (N=1).
    assert h1["winner_dsr"] is not None and h1["best_baseline_dsr"] is not None


def test_beat_human_baseline_llm_loses_to_all() -> None:
    """LLM winner Sharpe < every baseline => does NOT beat the best; low beat fraction."""
    records = _seeded_test_records("distributional", mu=0.001)  # lowest mean -> lowest Sharpe
    for i, name in enumerate(_H1_BASELINES):
        records += _seeded_test_records(f"baseline_{name}", mu=0.008 + 0.001 * i, seed0=100 * (i + 1))

    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional",
                                winner_n_trials=30)
    assert h1["status"] == "ok"
    assert h1["beats_best_baseline_median"] is False
    assert h1["beat_fraction"] < 0.5
    assert h1["norm_improvement"] is not None and h1["norm_improvement"] < 0.0


# --------------------------------------------------------------------------- #
# N6 (validity_tier) — the snoop-free IUT: does the LLM reward DOMINATE the      #
# hand-reward canon (== beat the BEST human, made precise)? Berger 1982.         #
# --------------------------------------------------------------------------- #
def _iut_seeded(arm: str, base_mu: float, *, n_seeds: int = 10, jitter: float = 0.0015, seed0: int = 0) -> list[dict]:
    """Per-seed records whose per-seed Sharpe VARIES across seeds (real across-seed variance for the paired
    bootstrap), with the arm's mean Sharpe set by ``base_mu``. Shared seed indices 0..n_seeds-1 across arms ->
    CRN-paired (unlike the deterministic ``_seeded_test_records`` which forces zero across-seed variance)."""
    out: list[dict] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(7000 + seed0 + s)
        mu_s = base_mu + float(rng.standard_normal()) * jitter   # per-seed mean jitter -> across-seed Sharpe spread
        v = rng.standard_normal(250)
        v = (v - v.mean()) / v.std(ddof=1) * 0.02 + mu_s         # force sd=0.02, mean=mu_s
        out.append(_test_record(arm, s, v))
    return out


def test_iut_dominates_canon_when_llm_beats_every_member() -> None:
    """N6: the LLM significantly beats EVERY hand reward => dominates_canon True; IUT p (max leg p) <= alpha."""
    records = _iut_seeded("distributional", 0.012)                       # clearly the highest Sharpe
    for i, name in enumerate(_H1_BASELINES):
        records += _iut_seeded(f"baseline_{name}", 0.002 + 0.0008 * i, seed0=100 * (i + 1))
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional",
                                winner_n_trials=30, n_boot=2000)
    iut = h1["iut"]
    assert iut["test"] == "llm_dominates_hand_reward_canon"
    assert iut["all_baselines_present"] is True
    assert iut["n_significantly_beaten"] == len(_H1_BASELINES)
    assert iut["dominates_canon"] is True
    assert iut["iut_pvalue"] is not None and iut["iut_pvalue"] <= 0.05
    assert len(iut["dominance_profile"]) == len(_H1_BASELINES)
    assert all(lg["verdict"] == "dominates" for lg in iut["dominance_profile"])


def test_iut_does_not_dominate_when_one_member_beats_llm() -> None:
    """N6: one hand reward beats the LLM => dominates_canon False even though the LLM beats the rest;
    IUT p is dominated by the losing leg (max over legs) => > alpha. The honest profile flags the loss."""
    records = _iut_seeded("distributional", 0.007)
    for i, name in enumerate(_H1_BASELINES[:-1]):                        # 3 weak baselines the LLM beats
        records += _iut_seeded(f"baseline_{name}", 0.002 + 0.0008 * i, seed0=100 * (i + 1))
    records += _iut_seeded(f"baseline_{_H1_BASELINES[-1]}", 0.013, seed0=900)   # 1 STRONG -> beats the LLM
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional",
                                winner_n_trials=30, n_boot=2000)
    iut = h1["iut"]
    assert iut["all_baselines_present"] is True
    assert iut["dominates_canon"] is False
    assert iut["n_behind_on_point_estimate"] >= 1
    strong = next(lg for lg in iut["dominance_profile"] if lg["baseline"] == _H1_BASELINES[-1])
    assert strong["beaten"] is False and strong["verdict"] == "behind"
    assert iut["iut_pvalue"] is not None and iut["iut_pvalue"] > 0.05


def test_iut_incomplete_when_a_member_missing() -> None:
    """N6: a canon member with no test records => dominance NOT certifiable (all_baselines_present False)."""
    records = _iut_seeded("distributional", 0.012)
    for i, name in enumerate(_H1_BASELINES[:-1]):                        # only 3 of 4 baselines present
        records += _iut_seeded(f"baseline_{name}", 0.003, seed0=100 * (i + 1))
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional",
                                winner_n_trials=30, n_boot=2000)
    iut = h1["iut"]
    assert iut["all_baselines_present"] is False
    assert iut["dominates_canon"] is False                              # cannot certify with a missing member
    missing = next(lg for lg in iut["dominance_profile"] if lg["baseline"] == _H1_BASELINES[-1])
    assert missing["present"] is False and missing["verdict"] == "not_testable"


def test_beat_human_baseline_skips_when_baselines_absent() -> None:
    """No baseline_<name> records => graceful skip (the baseline stage was not run)."""
    records = _seeded_test_records("distributional", mu=0.01)  # winner present, no baselines
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional")
    assert h1["status"] == "skipped"
    assert "baseline" in h1["reason"].lower()


def test_beat_human_baseline_skips_when_winner_absent() -> None:
    """No LLM winner records => graceful skip (records-only / winner arm missing)."""
    records: list[dict] = []
    for i, name in enumerate(_H1_BASELINES):
        records += _seeded_test_records(f"baseline_{name}", mu=0.005, seed0=100 * (i + 1))
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional")
    assert h1["status"] == "skipped"


def test_beat_human_baseline_handles_partial_baselines() -> None:
    """Only SOME configured baselines present => the best is chosen among the present ones (no crash)."""
    records = _seeded_test_records("distributional", mu=0.01)
    # Only 2 of the 4 baselines have records.
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)
    records += _seeded_test_records("baseline_return_minus_cvar", mu=0.006, seed0=300)
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional")
    assert h1["status"] == "ok"
    assert h1["best_baseline"] == "return_minus_cvar"  # the higher-mu of the two present
    assert h1["baselines"]["return_minus_variance"]["present"] is False
    assert h1["baselines"]["differential_sharpe"]["present"] is False


def test_h1_beat_human_disjoint_key_does_not_trip_frozen_family_assert() -> None:
    """The H1 key carries NO arm_a/arm_b/metric/level => assert_realized_family_matches_frozen is untouched.

    Builds the realized H2 family from the FROZEN pre-registration mirror (so the assert PASSES), then
    confirms the h1_beat_human dict is structurally disjoint: it never enters the `tests` list the assert
    inspects, and none of its keys collide with the family-tuple keys. This is the load-bearing guarantee
    that the report-only H1 panel cannot grow the frozen m=6 testing family.
    """
    from src.utils.config import load_config

    fam = load_config("preregistration").get("inference", {}).get("testing_family")
    if not fam or not fam.get("members"):
        pytest.skip("no frozen testing_family mirror in config/preregistration.yaml")

    # The realized family == the frozen members (so the assert is satisfied) -> no AssertionError.
    realized_tests = [
        {"arm_a": m["arm_a"], "arm_b": m["arm_b"], "metric": m["metric"], "level": m.get("level")}
        for m in fam["members"]
    ]
    cvar_levels = tuple(float(x) for x in fam.get("cvar_levels", [0.05]))
    AC.assert_realized_family_matches_frozen(realized_tests, cvar_levels=cvar_levels)  # must NOT raise

    # The H1 output dict is DISJOINT: it has none of the family-tuple keys, so it can never be a member.
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)
    h1 = AC.beat_human_baseline(records, baseline_names=_H1_BASELINES, winner_arm="distributional")
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(h1.keys()))


def test_h1_beat_human_markdown_renders_table_and_skip() -> None:
    """The H1 markdown renders the verdict + a per-baseline row, and an n/a block when skipped."""
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)
    h1 = AC.beat_human_baseline(records, baseline_names=["raw_return"], winner_arm="distributional",
                                winner_n_trials=30)
    md = AC.h1_beat_human_markdown(h1)
    assert "hand reward" in md.lower()        # the H1 verdict header (beat the BEST hand reward?)
    assert "Eureka" in md                     # cited as Eureka-STYLE (context, not the identical HNS)
    assert "raw_return" in md
    assert "Fraction beaten" in md            # the headline metric line (relabelled from "Beat fraction")
    # The skipped case renders an n/a block, not a table.
    md_skip = AC.h1_beat_human_markdown({"status": "skipped", "reason": "no baselines"})
    assert "n/a" in md_skip
    assert "beat-the-human" in md_skip.lower()


# --------------------------------------------------------------------------- #
# markdown emitter                                                             #
# --------------------------------------------------------------------------- #
def test_pbo_markdown_has_a_row_per_arm() -> None:
    rng = np.random.default_rng(8)
    records = _records_from_matrix("scalar", rng.standard_normal((200, 10)))
    results = AC.campaign_pbo(records, n_blocks=N_BLOCKS)
    md = AC.pbo_markdown(results, n_blocks=N_BLOCKS)
    assert "PBO / CSCV" in md
    for arm in AC.ARMS:
        assert arm in md
    # The 'ok' scalar arm shows a numeric PBO, not n/a.
    scalar_line = next(line for line in md.splitlines() if line.startswith("| scalar "))
    assert "n/a" not in scalar_line


# --------------------------------------------------------------------------- #
# H1 — best-baseline IDENTITY selected on VALIDATION (DEEP_H1 T-REF data-snoop) #
# --------------------------------------------------------------------------- #
def test_beat_human_selects_best_baseline_on_validation_when_archived() -> None:
    """When baselines carry val_returns, the best baseline is picked by VAL Sharpe, not test (T-REF fix).

    The baseline that is BEST on test (highest test mu) is deliberately made WORST on validation, and vice
    versa, so a test-snooping selection would pick a different baseline than the val-selecting one. The fix
    must select on validation and flag no data-snoop.
    """
    records = _seeded_test_records("distributional", mu=0.01)
    # baseline A: best on TEST (mu 0.009) but worst on VAL (val_mu 0.001).
    records += _seeded_test_records("baseline_raw_return", mu=0.009, seed0=200, val_mu=0.001)
    # baseline B: worse on TEST (mu 0.004) but BEST on VAL (val_mu 0.012).
    records += _seeded_test_records("baseline_return_minus_cvar", mu=0.004, seed0=300, val_mu=0.012)

    h1 = AC.beat_human_baseline(
        records, baseline_names=["raw_return", "return_minus_cvar"], winner_arm="distributional",
        winner_n_trials=30,
    )
    assert h1["status"] == "ok"
    assert h1["best_selected_on"] == "validation"
    assert h1["val_snoop_caveat"] is False
    # Validation selection picks the val-best baseline (return_minus_cvar), NOT the test-best (raw_return).
    assert h1["best_baseline"] == "return_minus_cvar"
    assert h1["best_baseline_val_sharpe"] is not None


def test_beat_human_falls_back_to_test_selection_and_flags_when_no_val() -> None:
    """No baseline val signal (val_fitness=NaN, no val_returns) => test-median selection, FLAGGED (T-REF)."""
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)        # no val_mu
    records += _seeded_test_records("baseline_return_minus_cvar", mu=0.006, seed0=300)  # no val_mu
    h1 = AC.beat_human_baseline(
        records, baseline_names=["raw_return", "return_minus_cvar"], winner_arm="distributional",
    )
    assert h1["status"] == "ok"
    assert h1["val_snoop_caveat"] is True
    assert "test" in h1["best_selected_on"]
    assert h1["best_baseline"] == "return_minus_cvar"  # the test-best of the two


# --------------------------------------------------------------------------- #
# T2.2 — the data-snoop disclosure is UNMISSABLE (structured status + loud md)  #
# --------------------------------------------------------------------------- #
def test_h1_inference_status_is_structured_and_descriptive_only_on_snoop() -> None:
    """The test-snoop fallback carries a STRUCTURED status + caveat (descriptive-only), not just a bool."""
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)        # no val signal
    records += _seeded_test_records("baseline_return_minus_cvar", mu=0.006, seed0=300)  # no val signal
    h1 = AC.beat_human_baseline(
        records, baseline_names=["raw_return", "return_minus_cvar"], winner_arm="distributional",
    )
    assert h1["inference_status"] == "test_snooped_descriptive_only"
    assert h1["val_snoop_caveat"] is True
    assert "DESCRIPTIVE-ONLY" in h1["caveat"]
    assert "PREREGISTRATION" in h1["caveat"]  # the proposed-amendment pointer is in the caveat text


def test_h1_inference_status_val_selected_when_validation_archived() -> None:
    """When a baseline val signal IS archived, the status is the defensible 'val_selected' (no caveat)."""
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.009, seed0=200, val_mu=0.001)
    records += _seeded_test_records("baseline_return_minus_cvar", mu=0.004, seed0=300, val_mu=0.012)
    h1 = AC.beat_human_baseline(
        records, baseline_names=["raw_return", "return_minus_cvar"], winner_arm="distributional",
        winner_n_trials=30,
    )
    assert h1["inference_status"] == "val_selected"
    assert h1["val_snoop_caveat"] is False
    assert h1["caveat"] == ""


def test_h1_markdown_surfaces_loud_warning_block_at_top_on_snoop() -> None:
    """The markdown puts a prominent NOTE: DATA-SNOOP blockquote ABOVE the report body (not a buried bullet)."""
    records = _seeded_test_records("distributional", mu=0.01)
    records += _seeded_test_records("baseline_raw_return", mu=0.003, seed0=200)
    h1 = AC.beat_human_baseline(records, baseline_names=["raw_return"], winner_arm="distributional")
    md = AC.h1_beat_human_markdown(h1)
    assert "DATA-SNOOP" in md and "DESCRIPTIVE-ONLY" in md
    # The warning is a blockquote that PRECEDES the 'POST-FREEZE' report body line.
    i_warn = md.index("DATA-SNOOP")
    i_body = md.index("POST-FREEZE")
    assert i_warn < i_body
    # The warning line is a markdown blockquote (starts with '>') and the section header is line 1.
    warn_line = next(ln for ln in md.splitlines() if "DATA-SNOOP" in ln)
    assert warn_line.lstrip().startswith(">")
    assert md.splitlines()[0].startswith("## H1")
    # The val-selected case shows NO warning block.
    records2 = _seeded_test_records("distributional", mu=0.01)
    records2 += _seeded_test_records("baseline_raw_return", mu=0.009, seed0=200, val_mu=0.012)
    h1_ok = AC.beat_human_baseline(records2, baseline_names=["raw_return"], winner_arm="distributional")
    assert "DATA-SNOOP" not in AC.h1_beat_human_markdown(h1_ok)


# --------------------------------------------------------------------------- #
# H4 — LLM winner vs random_search / bayes_opt (DEEP_H4)                        #
# --------------------------------------------------------------------------- #
def test_h4_llm_beats_all_search_controls() -> None:
    """LLM (high test Sharpe) beats all four search controls => every H4 leg rejects one-sided (N4 beat-max)."""
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=12)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=12, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.001, n_seeds=12, seed0=800)
    records += _seeded_test_records("cma_es", mu=0.001, n_seeds=12, seed0=1200)
    records += _seeded_test_records("tpe", mu=0.001, n_seeds=12, seed0=1600)
    h4 = AC.h4_search_controls(records, winner_arm="distributional", rng=np.random.default_rng(0))
    assert h4["status"] == "ok"
    assert h4["n_tests"] == 4
    assert {t["test"] for t in h4["tests"]} == {"h4a", "h4b", "h4c", "h4d"}
    assert all(t["direction_ok"] for t in h4["tests"])      # LLM is better in both
    assert all(t["reject_one_sided"] for t in h4["tests"])
    assert h4["all_supported"] is True
    # The 4-test multiplicity is reported (Bonferroni over 4 at alpha/4); all-supported = the N4 beat-max IUT.
    assert h4["bonferroni_alpha"] == pytest.approx(0.0125)
    assert "reject_one_sided_bonferroni" in h4["tests"][0]


def test_h4_llm_loses_does_not_support() -> None:
    """LLM worse than the controls => direction wrong, no rejection, all_supported False."""
    records = _seeded_test_records("distributional", mu=0.001, n_seeds=12)
    records += _seeded_test_records("random_search", mu=0.010, n_seeds=12, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.010, n_seeds=12, seed0=800)
    h4 = AC.h4_search_controls(records, winner_arm="distributional", rng=np.random.default_rng(0))
    assert h4["status"] == "ok"
    assert not any(t["reject_one_sided"] for t in h4["tests"])
    assert h4["all_supported"] is False


def test_h4_skips_when_a_control_absent() -> None:
    """A missing control arm => that leg is reported in `skipped` and all_supported is False (not all ran)."""
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=8)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=8, seed0=400)
    # bayes_opt absent.
    h4 = AC.h4_search_controls(records, winner_arm="distributional", rng=np.random.default_rng(0))
    assert h4["status"] == "ok"
    assert [t["test"] for t in h4["tests"]] == ["h4a"]
    assert any(s["test"] == "h4b" for s in h4["skipped"])
    assert h4["all_supported"] is False  # h4b did not run


def test_h4_skips_when_winner_absent() -> None:
    """No LLM winner test records => graceful skip."""
    records = _seeded_test_records("random_search", mu=0.001, n_seeds=8, seed0=400)
    h4 = AC.h4_search_controls(records, winner_arm="distributional")
    assert h4["status"] == "skipped"


def test_h4_disjoint_keys_no_family_tuple() -> None:
    """The H4 output carries NO arm_a/arm_b/metric/level family-tuple keys (frozen m=6 untouched)."""
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=8)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=8, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.001, n_seeds=8, seed0=800)
    h4 = AC.h4_search_controls(records)
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(h4.keys()))


def test_h4_markdown_renders_and_skips() -> None:
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=8)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=8, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.001, n_seeds=8, seed0=800)
    md = AC.h4_markdown(AC.h4_search_controls(records))
    assert "H4" in md and "H4A" in md and "H4B" in md
    assert "random_search" in md and "bayes_opt" in md
    assert "n/a" in AC.h4_markdown({"status": "skipped", "reason": "no winner"})


# --------------------------------------------------------------------------- #
# H3 — iterative reflection vs single-shot (DEEP_H3)                            #
# --------------------------------------------------------------------------- #
def test_h3_skips_when_single_shot_absent() -> None:
    """The headline graceful skip: no single-shot archive => H3 is not fabricated."""
    iter_recs = _seeded_test_records("distributional", mu=0.01, n_seeds=10)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, None, arm="distributional")
    assert h3["status"] == "skipped"
    assert "single-shot" in h3["reason"].lower()


def test_h3_difference_when_iterative_beats_single_shot() -> None:
    """Iterative clearly > single-shot => one-sided difference rejects."""
    iter_recs = _seeded_test_records("distributional", mu=0.012, n_seeds=12, seed0=0)
    ss_recs = _seeded_test_records("distributional", mu=0.001, n_seeds=12, seed0=600)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional", rng=np.random.default_rng(0))
    assert h3["status"] == "ok"
    assert h3["difference"]["direction_ok"] is True
    assert h3["difference"]["reject_one_sided"] is True
    assert "iterative > single-shot" in h3["verdict"]
    # Equivalence is reported alongside (a structured TOST dict).
    assert set(h3["equivalence"]) >= {"equivalent", "estimate", "ci_low", "ci_high", "margin"}


def test_h3_equivalence_when_iterative_matches_single_shot() -> None:
    """Iterative ~= single-shot (same mu) => no difference, TOST equivalence holds (bankable null)."""
    iter_recs = _seeded_test_records("distributional", mu=0.005, n_seeds=16, seed0=0)
    ss_recs = _seeded_test_records("distributional", mu=0.005, n_seeds=16, seed0=600)
    h3 = AC.h3_iterative_vs_singleshot(
        iter_recs, ss_recs, arm="distributional", equiv_margin=0.5, rng=np.random.default_rng(1)
    )
    assert h3["status"] == "ok"
    assert h3["difference"]["reject_one_sided"] is False
    assert h3["equivalence"]["equivalent"] is True
    assert "equivalent" in h3["verdict"]


def test_h3_skips_when_too_few_shared_seeds() -> None:
    """< 2 shared seeds across the two conditions => graceful skip."""
    iter_recs = _seeded_test_records("distributional", mu=0.01, n_seeds=1, seed0=0)
    ss_recs = _seeded_test_records("distributional", mu=0.01, n_seeds=1, seed0=600)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional")
    assert h3["status"] == "skipped"


def test_h3_disjoint_keys_and_markdown() -> None:
    iter_recs = _seeded_test_records("distributional", mu=0.012, n_seeds=10)
    ss_recs = _seeded_test_records("distributional", mu=0.001, n_seeds=10, seed0=600)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional", rng=np.random.default_rng(0))
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(h3.keys()))
    md = AC.h3_markdown(h3)
    assert "iterative" in md.lower() and "single-shot" in md.lower()
    assert "TOST" in md
    assert "n/a" in AC.h3_markdown({"status": "skipped", "reason": "absent"})


# --------------------------------------------------------------------------- #
# T3.4 (a) — H4 TOST equivalence bound (closing the H4-vs-H3 asymmetry)         #
# --------------------------------------------------------------------------- #
def test_h4_carries_tost_equivalence_per_leg() -> None:
    """Each H4 leg now carries a TOST equivalence dict + a verdict (mirrors H3)."""
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=12)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=12, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.001, n_seeds=12, seed0=800)
    records += _seeded_test_records("cma_es", mu=0.001, n_seeds=12, seed0=1200)
    records += _seeded_test_records("tpe", mu=0.001, n_seeds=12, seed0=1600)
    h4 = AC.h4_search_controls(records, winner_arm="distributional", rng=np.random.default_rng(0))
    assert "equiv_margin" in h4 and "all_equivalent" in h4
    for t in h4["tests"]:
        assert set(t["equivalence"]) >= {"equivalent", "estimate", "ci_low", "ci_high", "margin"}
        assert "verdict" in t


def test_h4_equivalence_bounded_null_when_llm_matches_controls() -> None:
    """LLM ~= both controls (same mu) + a wide margin => no rejection but the legs are equivalence-bounded."""
    records = _seeded_test_records("distributional", mu=0.005, n_seeds=16)
    records += _seeded_test_records("random_search", mu=0.005, n_seeds=16, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.005, n_seeds=16, seed0=800)
    records += _seeded_test_records("cma_es", mu=0.005, n_seeds=16, seed0=1200)
    records += _seeded_test_records("tpe", mu=0.005, n_seeds=16, seed0=1600)
    h4 = AC.h4_search_controls(
        records, winner_arm="distributional", equiv_margin=0.5, rng=np.random.default_rng(1)
    )
    assert h4["all_supported"] is False           # no difference
    assert h4["all_equivalent"] is True           # but the bankable equivalence holds
    assert all("equivalent" in t["verdict"] for t in h4["tests"])


# --------------------------------------------------------------------------- #
# T3.4 (b) — H4 procedure-vs-richness reference framing                         #
# --------------------------------------------------------------------------- #
def test_h4_reports_procedure_vs_richness_reference_framing() -> None:
    """H4a is labelled the IN-FAMILY reference; H4b the fixed-template reference (the promoted baseline)."""
    records = _seeded_test_records("distributional", mu=0.012, n_seeds=8)
    records += _seeded_test_records("random_search", mu=0.001, n_seeds=8, seed0=400)
    records += _seeded_test_records("bayes_opt", mu=0.001, n_seeds=8, seed0=800)
    h4 = AC.h4_search_controls(records, winner_arm="distributional", rng=np.random.default_rng(0))
    assert "reference_framing" in h4
    by_id = {t["test"]: t for t in h4["tests"]}
    assert "in-family" in by_id["h4a"]["reference"]
    assert "fixed-parametric-template" in by_id["h4b"]["reference"]
    md = AC.h4_markdown(h4)
    assert "procedure-vs-richness" in md or "procedure" in md
    # #100 (2026-07-27): these were ``"in-family ref"`` / ``"fixed-template ref"`` — the ABBREVIATIONS
    # the emitter hardcoded via ``"in-family ref" if test == "h4a" else "fixed-template ref"``. That
    # two-way branch mislabelled BOTH h4c (CMA-ES) and h4d (TPE) with h4b's Bayes-opt framing, and this
    # test could never see it: it supplies only random_search and bayes_opt records, so h4c/h4d are
    # always SKIPPED here and its coverage stayed frozen at the 2-leg era. The emitter now reads the
    # authoritative per-test ``reference`` field, so assert on THAT — matching this test's own
    # docstring and its ``by_id[...]["reference"]`` checks above. Full 4-leg coverage lives in
    # ``test_h4_markdown_derives_its_counts_and_uses_the_authoritative_reference_labels``.
    assert "in-family random-search reference" in md
    assert "fixed-parametric-template reference" in md
    assert "equiv" in md.lower()  # the TOST column header from (a)


# --------------------------------------------------------------------------- #
# T3.4 (c) — H3 paired placebo-relative uplift difference                       #
# --------------------------------------------------------------------------- #
def test_h3_placebo_relative_uplift_skips_without_placebo_singleshot() -> None:
    """The default headline single-shot stage has only distributional => the placebo-relative probe skips."""
    iter_recs = _seeded_test_records("distributional", mu=0.012, n_seeds=10)
    ss_recs = _seeded_test_records("distributional", mu=0.001, n_seeds=10, seed0=600)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional", rng=np.random.default_rng(0))
    pru = h3["placebo_relative_uplift"]
    assert pru["status"] == "skipped"
    assert "placebo" in pru["reason"].lower()
    # The markdown reports the n/a line (never fabricated).
    assert "Placebo-relative uplift" in AC.h3_markdown(h3)


def test_h3_placebo_relative_uplift_detects_information_signature() -> None:
    """When BOTH arms have iterative + single-shot conditions and the distributional uplift is LARGER, the
    placebo-relative difference rejects (an information-tracking signature)."""
    # distributional: a BIG iterative-over-single-shot uplift (0.012 vs 0.001).
    iter_recs = _seeded_test_records("distributional", mu=0.012, n_seeds=16, seed0=0)
    ss_recs = _seeded_test_records("distributional", mu=0.001, n_seeds=16, seed0=600)
    # placebo: NO uplift (iterative ~= single-shot, both ~0.001) -> content-free reflection adds nothing.
    iter_recs += _seeded_test_records("placebo", mu=0.001, n_seeds=16, seed0=1200)
    ss_recs += _seeded_test_records("placebo", mu=0.001, n_seeds=16, seed0=1800)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional", rng=np.random.default_rng(3))
    pru = h3["placebo_relative_uplift"]
    assert pru["status"] == "ok"
    assert pru["n_seeds"] == 16
    assert pru["mean_uplift_distributional"] > pru["mean_uplift_placebo"]
    assert pru["effect"] > 0.0
    assert pru["reject_two_sided"] is True
    assert "information-tracking signature" in pru["interpretation"]
    assert "information-tracking signature" in AC.h3_markdown(h3)


def test_h3_placebo_relative_uplift_null_when_uplifts_match() -> None:
    """Equal distributional + placebo uplifts => no information-tracking signature (the qualitative null)."""
    # BOTH arms get the SAME iterative-over-single-shot uplift => the difference-of-uplifts is ~0.
    iter_recs = _seeded_test_records("distributional", mu=0.010, n_seeds=16, seed0=0)
    ss_recs = _seeded_test_records("distributional", mu=0.004, n_seeds=16, seed0=600)
    iter_recs += _seeded_test_records("placebo", mu=0.010, n_seeds=16, seed0=1200)
    ss_recs += _seeded_test_records("placebo", mu=0.004, n_seeds=16, seed0=1800)
    h3 = AC.h3_iterative_vs_singleshot(iter_recs, ss_recs, arm="distributional", rng=np.random.default_rng(5))
    pru = h3["placebo_relative_uplift"]
    assert pru["status"] == "ok"
    assert pru["reject_two_sided"] is False
    assert "NO information-tracking signature" in pru["interpretation"]


# --------------------------------------------------------------------------- #
# H2 TOST equivalence (DEEP_H2 §5.3)                                            #
# --------------------------------------------------------------------------- #
def _h2_test_records(*, dist_mu: float, other_mu: float, n_seeds: int = 12) -> list[dict]:
    """Test-leg records for the four H2 arms: distributional vs {scalar, placebo, scalar_cvar5}."""
    recs = _seeded_test_records("distributional", mu=dist_mu, n_seeds=n_seeds, seed0=0)
    for i, arm in enumerate(("scalar", "placebo", "scalar_cvar5")):
        recs += _seeded_test_records(arm, mu=other_mu, n_seeds=n_seeds, seed0=1000 * (i + 1))
    return recs


def test_h2_tost_equivalent_legs_when_arms_match() -> None:
    """All H2 arms ~equal => every RA + Tail leg is equivalent within a wide margin."""
    records = _h2_test_records(dist_mu=0.005, other_mu=0.005, n_seeds=16)
    tost = AC.h2_tost(records, equiv_margin=1.0, rng=np.random.default_rng(0))
    assert tost["status"] == "ok"
    assert len(tost["ra"]) == 3 and len(tost["tail"]["legs"]) == 3
    assert all(leg["equivalent"] for leg in tost["ra"])
    assert all(leg["equivalent"] for leg in tost["tail"]["legs"])
    # In the test-statistic's own units, with structured CI fields.
    assert "test-statistic units" in tost["units"]
    assert set(tost["ra"][0]) >= {"contrast", "estimate", "ci_low", "ci_high", "equivalent", "n_seeds"}


def test_h2_tost_not_equivalent_when_far_apart_tight_margin() -> None:
    """A large distributional edge with a TIGHT margin => legs are NOT equivalent."""
    records = _h2_test_records(dist_mu=0.02, other_mu=-0.01, n_seeds=16)
    tost = AC.h2_tost(records, equiv_margin=0.001, rng=np.random.default_rng(0))
    assert tost["status"] == "ok"
    assert not all(leg["equivalent"] for leg in tost["ra"])


def test_h2_tost_skips_records_only() -> None:
    """No test-leg records => graceful skip (no crash)."""
    tost = AC.h2_tost([], rng=np.random.default_rng(0))
    assert tost["status"] == "skipped"


def test_h2_tost_disjoint_keys_and_markdown() -> None:
    records = _h2_test_records(dist_mu=0.005, other_mu=0.005, n_seeds=12)
    tost = AC.h2_tost(records, equiv_margin=1.0, rng=np.random.default_rng(0))
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(tost.keys()))
    md = AC.h2_tost_markdown(tost)
    assert "TOST" in md and "H2-RA" in md and "H2-Tail" in md
    assert "n/a" in AC.h2_tost_markdown({"status": "skipped"})


# --------------------------------------------------------------------------- #
# V9 (2026-06-26): DSR-units TOST companion                                  #
# --------------------------------------------------------------------------- #
def test_h2_tost_dsr_runs_in_validation_dsr_units() -> None:
    """The DSR-units leg computes the equivalence the power doc requires: per-seed Sharpe mapped to a
    CONSERVATIVE validation-DSR shift, TOST at the FROZEN ±0.05 SESOI. RA-only, structured, DISJOINT."""
    records = _h2_test_records(dist_mu=0.005, other_mu=0.005, n_seeds=16)
    t = AC.h2_tost_dsr(records, rng=np.random.default_rng(0))
    assert t["status"] == "ok"
    assert "validation-DSR" in t["units"]
    assert t["margin"] == pytest.approx(0.05)  # the FROZEN SESOI (inference.equivalence_margin)
    assert t["sharpe_to_dsr_factor"] > 0.0     # the φ(0)·√(T−1)/√252 ceiling factor
    assert len(t["ra"]) == 3                    # the three H2 RA contrasts
    leg = t["ra"][0]
    assert set(leg) >= {"contrast", "estimate", "ci_low", "ci_high", "equivalent", "inconclusive",
                        "n_seeds", "estimate_sharpe"}
    # The DSR-unit estimate is the Sharpe-unit estimate RESCALED by the linear ceiling factor (within MC).
    assert leg["estimate"] == pytest.approx(leg["estimate_sharpe"] * t["sharpe_to_dsr_factor"], abs=1e-9)
    # DISJOINT: no family-tuple keys -> assert_realized_family_matches_frozen untouched.
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(t.keys()))


def test_h2_tost_dsr_equivalent_when_arms_match_and_inconclusive_when_far() -> None:
    """Equal arms => the DSR CI sits inside ±0.05 (equivalent). A large edge => NOT equivalent =>
    INCONCLUSIVE (the conservative ceiling cannot certify a bound that small), never 'a difference'."""
    eq = AC.h2_tost_dsr(_h2_test_records(dist_mu=0.005, other_mu=0.005, n_seeds=16),
                        rng=np.random.default_rng(1))
    assert all(leg["equivalent"] and not leg["inconclusive"] for leg in eq["ra"])
    # A big distributional edge (Sharpe gap maps to a DSR shift > 0.05) => not equivalent => inconclusive.
    far = AC.h2_tost_dsr(_h2_test_records(dist_mu=0.05, other_mu=-0.02, n_seeds=16),
                         rng=np.random.default_rng(1))
    assert any(leg["inconclusive"] and not leg["equivalent"] for leg in far["ra"])


def test_h2_tost_dsr_skips_and_renders_markdown() -> None:
    """No shared-seed records => graceful skip; the markdown renders the DSR-units section + the n/a path."""
    assert AC.h2_tost_dsr([], rng=np.random.default_rng(0))["status"] == "skipped"
    t = AC.h2_tost_dsr(_h2_test_records(dist_mu=0.005, other_mu=0.005, n_seeds=12),
                       rng=np.random.default_rng(0))
    md = AC.h2_tost_dsr_markdown(t)
    assert "validation-DSR units" in md and "T2.5" in md and "ΔDSR" in md
    assert "n/a" in AC.h2_tost_dsr_markdown({"status": "skipped"})


# --------------------------------------------------------------------------- #
# H2 structure-vs-content control — distributional vs placebo_shuffled (R32)    #
# --------------------------------------------------------------------------- #
def _structure_records(*, dist_mu: float, shuffled_mu: float, n_seeds: int = 12) -> list[dict]:
    """Test-leg records for the structure control's two arms: distributional + placebo_shuffled.

    ``mu`` controls each arm's per-seed mean (hence its Sharpe AND its CVaR-5% ordering at fixed vol), so
    a higher ``dist_mu`` makes distributional beat placebo_shuffled on BOTH co-primary metrics — letting
    the test drive ``content_over_format`` deterministically (mirrors ``_h2_test_records``)."""
    return _seeded_test_records("distributional", mu=dist_mu, n_seeds=n_seeds, seed0=0) + _seeded_test_records(
        "placebo_shuffled", mu=shuffled_mu, n_seeds=n_seeds, seed0=500
    )


def test_h2_structure_control_content_over_format_when_distributional_beats_both() -> None:
    """distributional > placebo_shuffled on BOTH Sharpe AND CVaR-5% => content_over_format is True."""
    records = _structure_records(dist_mu=0.012, shuffled_mu=0.001, n_seeds=12)
    sc = AC.h2_structure_control(records, rng=np.random.default_rng(0))
    assert sc["status"] == "ok"
    # Both co-primary legs reject one-sided in the predicted direction (distributional strictly better).
    assert sc["sharpe"]["reject_one_sided"] is True and sc["sharpe"]["direction_ok"] is True
    assert sc["cvar"]["reject_one_sided"] is True and sc["cvar"]["direction_ok"] is True
    assert sc["content_over_format"] is True
    assert sc["contrast"] == "distributional>placebo_shuffled"
    assert sc["cvar_level"] == 0.05


def test_h2_structure_control_not_content_when_distributional_worse() -> None:
    """distributional WORSE than placebo_shuffled => neither leg rejects => content_over_format is False."""
    records = _structure_records(dist_mu=0.001, shuffled_mu=0.012, n_seeds=12)
    sc = AC.h2_structure_control(records, rng=np.random.default_rng(0))
    assert sc["status"] == "ok"
    assert sc["sharpe"]["reject_one_sided"] is False
    assert sc["cvar"]["reject_one_sided"] is False
    assert sc["content_over_format"] is False


def test_h2_structure_control_disjoint_keys_no_family_tuple() -> None:
    """The structure-control output carries NO arm_a/arm_b/metric/level keys, so it can NEVER be a member
    of the realized family the frozen-m=6 assert inspects (it cannot trip
    assert_realized_family_matches_frozen). This is the load-bearing 'reported, never a gate' guarantee."""
    from src.utils.config import load_config

    # The realized H2 family == the frozen members, so the assert is satisfied (does not raise) ...
    fam = load_config("preregistration").get("inference", {}).get("testing_family")
    if fam and fam.get("members"):
        realized_tests = [
            {"arm_a": m["arm_a"], "arm_b": m["arm_b"], "metric": m["metric"], "level": m.get("level")}
            for m in fam["members"]
        ]
        cvar_levels = tuple(float(x) for x in fam.get("cvar_levels", [0.05]))
        AC.assert_realized_family_matches_frozen(realized_tests, cvar_levels=cvar_levels)  # must NOT raise

    # ... and the structure-control dict is structurally disjoint from the family-tuple key set.
    records = _structure_records(dist_mu=0.012, shuffled_mu=0.001, n_seeds=8)
    sc = AC.h2_structure_control(records, rng=np.random.default_rng(0))
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(sc.keys()))


def test_h2_structure_control_skips_when_placebo_shuffled_absent() -> None:
    """No placebo_shuffled records (the 5th arm was not run) => graceful skip, NOT fabricated, no family keys."""
    records = _seeded_test_records("distributional", mu=0.01, n_seeds=12)
    sc = AC.h2_structure_control(records, rng=np.random.default_rng(0))
    assert sc["status"] == "skipped"
    assert "placebo_shuffled" in sc["reason"]
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(sc.keys()))


def test_h2_structure_markdown_renders_and_skips() -> None:
    records = _structure_records(dist_mu=0.012, shuffled_mu=0.001, n_seeds=8)
    md = AC.h2_structure_markdown(AC.h2_structure_control(records, rng=np.random.default_rng(0)))
    assert "Structure-vs-content control (R32)" in md
    assert "placebo_shuffled" in md
    assert "CONTENT over format" in md
    assert "n/a" in AC.h2_structure_markdown({"status": "skipped", "reason": "absent"})


# --------------------------------------------------------------------------- #
# Delisting-return sensitivity band (R33; PREREGISTRATION §7) — DATA-level      #
# --------------------------------------------------------------------------- #
import pandas as pd  # noqa: E402


def _synthetic_delisting_panel() -> "tuple[pd.DataFrame, pd.DataFrame]":
    """A tiny test-window panel + audit log: two names 'delist' (last valid session, NaN after).

    'A' dies at row 150 (audit delisting_return −0.30), 'B' at row 120 (−0.55). The other two names are
    continuously alive. The audit `date` is intentionally NOT used to locate the cell (the real audit
    date is the booking date, ~13 sessions late); the band finds each name's LAST VALID session in the
    panel, mirroring the verified univ4 mapping."""
    dates = pd.bdate_range("2018-01-02", periods=200)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((200, 4)) * 0.01, index=dates, columns=["A", "B", "C", "D"])
    df.iloc[151:, df.columns.get_loc("A")] = np.nan  # A's last valid session = row 150
    df.iloc[121:, df.columns.get_loc("B")] = np.nan  # B's last valid session = row 120
    audit = pd.DataFrame(
        {"ric": ["A", "B"], "delisting_return": [-0.30, -0.55], "value": [-0.30, -0.55]}
    )
    return df, audit


_TW = ("2018-01-02", "2026-01-01")  # a window covering the whole synthetic panel


def test_delisting_band_locates_cells_and_band_is_monotone() -> None:
    """The band finds the 2 delisting cells and the pooled CVaR gets MORE negative as d falls (0→−1)."""
    df, audit = _synthetic_delisting_panel()
    b = AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=_TW)
    assert b["status"] == "ok"
    assert b["n_delisting_cells_in_test"] == 2
    assert b["grid"] == [0.0, -0.30, -0.55, -1.00]
    # Pooled CVaR-5% is non-increasing across the (decreasing) delisting-return grid (more loss => worse tail).
    c5 = [b["cvar"]["0.05"][f"{d:g}"] for d in (0.0, -0.30, -0.55, -1.00)]
    assert all(c5[i] >= c5[i + 1] - 1e-12 for i in range(len(c5) - 1))
    # CVaR-1% (the EVT level) is also reported and strictly more extreme than CVaR-5% at the same d.
    c1_worst = b["cvar"]["0.01"]["-1"]
    c5_worst = b["cvar"]["0.05"]["-1"]
    assert c1_worst <= c5_worst + 1e-12


def test_delisting_band_flags_univ4_as_ma_contaminated_upper_bracket() -> None:
    """The −0.30 / −0.55 Shumway rows are flagged (``is_headline_extreme``) as the band's HEAVY end,
    and the note/keys frame it as an M&A-contaminated upper bracket — NOT the tail, NEITHER pole the truth
    (data-integrity audit 2026-06-25)."""
    df, audit = _synthetic_delisting_panel()
    b = AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=_TW)
    by_d = {row["d"]: row for row in b["rows"]}
    assert by_d[-0.30]["is_headline_extreme"] is True
    assert by_d[-0.55]["is_headline_extreme"] is True
    assert by_d[0.0]["is_headline_extreme"] is False
    assert by_d[-1.00]["is_headline_extreme"] is False
    # The band BRACKETS the tail: the note must say so and disclose the M&A contamination of the heavy end.
    note = b["note"].lower()
    assert "bracket" in note and "m&a" in note.lower()
    assert "ma_contamination" in b and "m&a" in b["ma_contamination"].lower()


def test_delisting_band_disjoint_keys_no_family_tuple() -> None:
    """The band output carries NO arm_a/arm_b/metric/level keys => assert_realized_family_matches_frozen
    is untouched (it is a DATA-level report-only secondary, never folded into the frozen m=6 union)."""
    df, audit = _synthetic_delisting_panel()
    b = AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=_TW)
    assert not ({"arm_a", "arm_b", "metric", "level"} & set(b.keys()))


def test_delisting_band_skips_when_no_cell_in_window() -> None:
    """When the delisting cells fall OUTSIDE the test window, the band skips (not fabricated)."""
    df, audit = _synthetic_delisting_panel()
    # A window AFTER both names' last valid sessions (rows 150/120 are ~2018), so no cell lands inside.
    b = AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=("2024-01-01", "2026-01-01"))
    assert b["status"] == "skipped"
    assert "test window" in b["reason"]


def test_delisting_band_skips_when_audit_ric_absent_from_panel() -> None:
    """An audit RIC not present in the panel contributes no cell => with none locatable, the band skips."""
    df, _ = _synthetic_delisting_panel()
    audit = pd.DataFrame({"ric": ["ZZZ"], "delisting_return": [-0.30], "value": [-0.30]})
    b = AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=_TW)
    assert b["status"] == "skipped"


def test_delisting_band_default_is_pinned_to_univ4_not_gold_suffix(monkeypatch) -> None:
    """V3 (2026-06-26): the band must LOCATE its cells from the suffix that carries them (univ4),
    INDEPENDENT of the headline ``gold_suffix()``. The headline default (post-R44 univ3; univ5
    post-Split-C) carries NO audit log — ``shumway_audit_log_univ5.parquet`` does NOT exist — so
    deferring to gold_suffix() silently skipped the LOAD-BEARING band. Here gold_suffix() is forced to
    the ACTIVE univ5 yet the arg-less band still reads univ4 and produces numbers (when the real univ4
    parquets are on disk)."""
    import src.data.loaders as _loaders

    monkeypatch.setattr(_loaders, "gold_suffix", lambda: "univ5")  # force the ACTIVE headline default
    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "data" / "gold" / "returns_panel_univ4.parquet").exists():
        pytest.skip("real univ4 panel/audit not on disk (records-only / synthetic install)")
    b = AC.delisting_band()  # ARG-LESS — exactly the call analyze() makes
    assert b["status"] == "ok", b.get("reason")
    assert b["headline_panel"] == "univ4" and b["cells_source"] == "univ4"  # pinned, not gold_suffix()
    assert b["n_delisting_cells_in_test"] > 0  # the 333 univ4 cells -> the test-window subset is located
    # The band PRODUCES the pooled tail at every grid point (no skip), and CVaR-5% is monotone non-increasing
    # as the delisting return falls (more loss => heavier tail) — d=0 is the univ3 zero-fill BRACKET end.
    c5 = [b["cvar"]["0.05"][f"{d:g}"] for d in (0.0, -0.30, -0.55, -1.00)]
    assert all(v == v for v in c5)  # all finite (not NaN)
    assert all(c5[i] >= c5[i + 1] - 1e-12 for i in range(len(c5) - 1))


def test_delisting_band_audit_suffix_override_is_honoured(monkeypatch) -> None:
    """The ``audit_suffix`` kwarg overrides the pinned default (future re-pull hook); a non-existent
    suffix skips cleanly WITHOUT touching gold_suffix() — proving the load path is fully decoupled."""
    import src.data.loaders as _loaders

    monkeypatch.setattr(_loaders, "gold_suffix", lambda: "univ4")  # even if gold_suffix() WERE univ4...
    b = AC.delisting_band(audit_suffix="univ_does_not_exist")
    assert b["status"] == "skipped"
    assert "univ_does_not_exist" in b["reason"]  # the override suffix drove the (failed) load, not univ4


def test_delisting_band_markdown_renders_and_skips() -> None:
    df, audit = _synthetic_delisting_panel()
    md = AC.delisting_band_markdown(AC.delisting_band(panel_df=df, audit_df=audit, test_window_dates=_TW))
    assert "Delisting-return sensitivity band (R33" in md
    assert "pooled CVaR-5%" in md and "pooled CVaR-1%" in md
    # The markdown must FRAME the band as a bracket and disclose the M&A contamination of the heavy end.
    assert "bracket" in md.lower() and "m&a" in md.lower()
    assert "n/a" in AC.delisting_band_markdown({"status": "skipped", "reason": "no panel"})


def test_delisting_band_runs_disk_free_via_injection(monkeypatch) -> None:
    """V3-regression guard that needs NO licensed data (records-only box).

    The on-disk default test (``test_delisting_band_default_is_pinned_to_univ4_not_gold_suffix``)
    pytest.skips when ``returns_panel_univ4.parquet`` is absent — a SILENT skip is exactly the
    V3 failure mode it is supposed to catch. This twin feeds a SYNTHETIC panel + audit through the
    injectable params so the band runs to ``status="ok"`` regardless of disk, and asserts the load
    is pinned to univ4 (``cells_source``). gold_suffix() is forced to univ5 (the ACTIVE headline
    default, which carries no audit log) to prove the band does NOT defer to it."""
    import src.data.loaders as _loaders

    monkeypatch.setattr(_loaders, "gold_suffix", lambda: "univ5")  # the ACTIVE headline default
    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)

    # A synthetic panel inside the frozen 2020-2026 evaluation window (SPLIT C); two names die WITH a
    # last-valid session that lands in the test window (rows 150 / 120), mirroring the verified univ4
    # last-valid mapping.
    dates = pd.bdate_range("2020-01-02", periods=200)
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(
        rng.standard_normal((200, 4)) * 0.01, index=dates, columns=["A", "B", "C", "D"]
    )
    panel.iloc[151:, panel.columns.get_loc("A")] = np.nan  # A's last valid session = row 150
    panel.iloc[121:, panel.columns.get_loc("B")] = np.nan  # B's last valid session = row 120
    # Audit columns mirror the real Shumway schema the band reads ('ric' + the 'value'/'delisting_return').
    audit = pd.DataFrame(
        {"ric": ["A", "B"], "delisting_return": [-0.30, -0.55], "value": [-0.30, -0.55]}
    )
    tw = ("2020-01-02", "2026-06-30")  # inside the frozen evaluation span (SPLIT C)

    b = AC.delisting_band(panel_df=panel, audit_df=audit, test_window_dates=tw)

    assert b["status"] == "ok"  # NOT "skipped" — the V3 silent-skip regression
    assert b["n_delisting_cells_in_test"] >= 1
    assert b["cells_source"] == "univ4"  # pinned to DELISTING_BAND_AUDIT_SUFFIX, not gold_suffix()=univ5
    # The band is monotone in d: a more-negative delisting return gives a CVaR no LESS negative.
    c5 = [b["cvar"]["0.05"][f"{d:g}"] for d in (0.0, -0.30, -0.55, -1.00)]
    assert all(v == v for v in c5)  # all finite
    assert all(c5[i] >= c5[i + 1] - 1e-12 for i in range(len(c5) - 1))


# --------------------------------------------------------------------------- #
# DSR effective-N sensitivity (DEEP_STATS A1)                                   #
# --------------------------------------------------------------------------- #
def test_dsr_effective_n_correlated_candidates_lowers_n_raises_dsr() -> None:
    """Correlated reflective candidates => rho_bar>0 => N_eff<N => DSR(N_eff) >= DSR(N) (benign direction)."""
    rng = np.random.default_rng(3)
    t, n = 400, 8
    common = rng.standard_normal(t)  # a shared component => positive cross-candidate correlation
    vecs = []
    for j in range(n):
        idio = rng.standard_normal(t)
        v = 0.85 * common + 0.15 * idio
        v = (v - v.mean()) / v.std(ddof=1) * 0.05 + (0.002 + 0.001 * j)  # a quality ladder for a winner
        vecs.append(v)
    records = [_record("distributional", f"c{j}", v) for j, v in enumerate(vecs)]
    d = AC.dsr_effective_n(records, winner_arm="distributional")
    assert d["status"] == "ok"
    assert d["rho_bar"] > 0.3           # strongly correlated by construction
    assert d["n_eff"] <= d["n_trials"]
    # Smaller N_eff => smaller expected-max deflation => DSR cannot fall (benign-direction, DEEP_STATS A8).
    assert d["dsr_eff_n"] >= d["dsr_raw_n"] - 1e-9


def test_dsr_effective_n_skips_with_one_candidate() -> None:
    """< 2 candidates with validation vectors => skipped (no cross-trial correlation)."""
    records = [_record("bayes_opt", "c0", np.arange(50, dtype=float))]
    d = AC.dsr_effective_n(records, winner_arm="bayes_opt")
    assert d["status"] == "skipped"


def test_dsr_effective_n_markdown_renders_and_skips() -> None:
    rng = np.random.default_rng(4)
    vecs = [rng.standard_normal(300) * 0.05 + 0.001 * j for j in range(6)]
    records = [_record("distributional", f"c{j}", v) for j, v in enumerate(vecs)]
    md = AC.dsr_effective_n_markdown(AC.dsr_effective_n(records, winner_arm="distributional"))
    assert "effective N" in md and "ρ̄" in md
    assert "n/a" in AC.dsr_effective_n_markdown({"status": "skipped", "reason": "too few"})


# --------------------------------------------------------------------------- #
# Cross-hypothesis multiplicity — Bonferroni-across-4 (DEEP_STATS A4)           #
# --------------------------------------------------------------------------- #
def test_cross_hypothesis_multiplicity_bonferroni_rows() -> None:
    """The sensitivity reports one row per hypothesis with the Bonferroni-across-4 hurdle applied."""
    h2 = {
        "verdict": "H2-RA + H2-Tail supported",
        "legs": [{"contrast": "distributional>scalar", "pvalue_one_sided": 0.004}],
        "tail_legs": [{"contrast": "distributional>scalar", "pvalue_one_sided": 0.20}],
    }
    h3 = {"status": "ok", "verdict": "equivalent", "difference": {"pvalue_one_sided": 0.30}}
    h4 = {"status": "ok", "all_supported": True,
          "tests": [{"pvalue_one_sided": 0.01}, {"pvalue_one_sided": 0.04}]}
    h1 = {"status": "ok", "beats_best_baseline_dsr": True}
    m = AC.cross_hypothesis_multiplicity(h1=h1, h2=h2, h3=h3, h4=h4, alpha=0.05, n_hypotheses=4)
    assert m["bonferroni_alpha"] == pytest.approx(0.0125)
    by = {r["hypothesis"]: r for r in m["rows"]}
    assert set(by) == {"H1", "H2", "H3", "H4"}
    # H1 is descriptive (no p) => survives_bonferroni None.
    assert by["H1"]["headline_p"] is None and by["H1"]["survives_bonferroni"] is None
    # H2's binding leg is the WORST one-sided p over both IUTs (0.20 from the tail leg) > 0.0125 => fails.
    assert by["H2"]["headline_p"] == pytest.approx(0.20)
    assert by["H2"]["survives_bonferroni"] is False
    # H4's binding p is the MAX over {0.01, 0.04} = 0.04 > 0.0125 => fails the stricter hurdle.
    assert by["H4"]["headline_p"] == pytest.approx(0.04)
    assert by["H4"]["survives_bonferroni"] is False
    # H3's 0.30 > 0.0125 => fails.
    assert by["H3"]["survives_bonferroni"] is False


def test_cross_hypothesis_multiplicity_handles_skipped_hypotheses() -> None:
    """Skipped/absent hypotheses contribute a row with None p / None survival (no crash)."""
    m = AC.cross_hypothesis_multiplicity(
        h1={"status": "skipped"}, h2={"verdict": "neither (null)", "legs": [], "tail_legs": []},
        h3={"status": "skipped"}, h4={"status": "skipped"}, alpha=0.05,
    )
    assert len(m["rows"]) == 4
    md = AC.cross_hypothesis_multiplicity_markdown(m)
    assert "Bonferroni-across-4" in md


# --------------------------------------------------------------------------- #
# EVT-consistency guard (DEEP_H2 §6.3)                                          #
# --------------------------------------------------------------------------- #
def _fed_search_record(arm: str, cid: str, vec: np.ndarray, *, fed_tail: bool) -> dict:
    """A search candidate in the CORRECTED fed shape (2026-07-06: the guard now shares the M13-fixed
    ``_was_fed_tail`` — prompt-first, generation-gated — so a FED candidate carries the tail labels
    in its archived PROMPT at generation >= 1; its own feedback_block is what it feeds FORWARD)."""
    rec = _record(arm, cid, vec)
    rec["generation"] = 1
    rec["prompt"] = (
        "Improve the reward. Feedback from the previous best:\n"
        "CVaR 5%: -0.03\nCVaR 1%: -0.06\nleft-tail mass: 0.02\n"
        if fed_tail else "Improve the reward. Feedback: Deflated Sharpe: 0.41"
    )
    rec["feedback_block"] = "CVaR 5%: -0.03\nCVaR 1%: -0.06\nleft-tail mass: 0.02\n" if fed_tail else ""
    return rec


def test_evt_consistency_guard_reports_per_arm_paths() -> None:
    """The guard re-derives each tail-FED arm's CVaR estimator path and reports cross-arm consistency."""
    rng = np.random.default_rng(9)
    recs: list[dict] = []
    for arm in ("distributional", "scalar_cvar5"):
        for j in range(4):
            v = rng.standard_normal(800) * 0.02 + 0.0005
            recs.append(_fed_search_record(arm, f"{arm}-c{j}", v, fed_tail=True))
    # A non-fed arm must NOT enter the consistency check (scalar sees only a scalar).
    for j in range(4):
        recs.append(_fed_search_record("scalar", f"scalar-c{j}", rng.standard_normal(800) * 0.02, fed_tail=False))
    g = AC.evt_consistency_guard(recs, levels=(0.05, 0.01))
    assert g["status"] == "ok"
    assert set(g["fed_arms"]) == {"distributional", "scalar_cvar5"}
    assert "scalar" not in g["per_arm"]                  # not tail-fed => excluded
    assert set(g["consistent"]) == {"0.05", "0.01"}
    assert "all_consistent" in g


def test_evt_consistency_guard_skips_with_one_fed_arm() -> None:
    """Fewer than 2 tail-fed arms => no cross-arm consistency to check => skipped."""
    rng = np.random.default_rng(10)
    recs = [
        _fed_search_record("distributional", f"d-c{j}", rng.standard_normal(800) * 0.02, fed_tail=True)
        for j in range(4)
    ]
    g = AC.evt_consistency_guard(recs)
    assert g["status"] == "skipped"


def test_evt_consistency_markdown_renders_and_skips() -> None:
    rng = np.random.default_rng(11)
    recs: list[dict] = []
    for arm in ("distributional", "scalar_cvar5"):
        for j in range(4):
            recs.append(_fed_search_record(arm, f"{arm}-c{j}", rng.standard_normal(800) * 0.02, fed_tail=True))
    md = AC.evt_consistency_markdown(AC.evt_consistency_guard(recs))
    assert "EVT-consistency" in md
    assert "n/a" in AC.evt_consistency_markdown({"status": "skipped", "reason": "one arm"})


# --------------------------------------------------------------------------- #
# divergence_report (R34) — cluster anomaly LINES into diverged RUNS           #
# --------------------------------------------------------------------------- #
def _write_anomalies(root: Path, runs: list[list[int]], kind: str = "critic_explosion") -> None:
    """Write a synthetic anomalies.jsonl: each inner list is one RUN's diverged STEPS (monotone up)."""
    lines: list[str] = []
    for steps in runs:
        for s in steps:
            lines.append(
                json.dumps({"ts": "2026-06-20T13:59:13", "kind": kind, "detail": "critic_loss spike", "step": s})
            )
    (root / "anomalies.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_divergence_report_clusters_lines_into_runs(tmp_path: Path) -> None:
    """64 critic_explosion LINES across 6 step-reset blocks => 6 diverged RUNS (the verified prototype fact)."""
    # Reproduce the prototype's 6 runs: long run (steps 1000..21000), 2 single-step (9000),
    # two long runs (1000..25000), one single-step (9000) — 64 lines total.
    runs = [
        list(range(1000, 21001, 1000))[:11],   # 11 lines (drops 6000/11000 like the real log; length 11)
        [9000],                                  # transient
        list(range(1000, 25001, 1000)),          # 25 lines
        [9000],                                  # transient
        list(range(1000, 25001, 1000)),          # 25 lines
        [9000],                                  # transient
    ]
    _write_anomalies(tmp_path, runs)
    n_lines = sum(len(r) for r in runs)
    out = AC.divergence_report(tmp_path)
    assert out["status"] == "ok"
    assert out["n_anomaly_lines"] == n_lines
    assert out["n_diverged_runs"] == 6           # the TRUE diverged-run count, not the line count
    assert out["n_diverged_runs"] < out["n_anomaly_lines"]
    assert out["transient_runs"] == 3            # the three single-step runs
    # Rate is runs / (candidates_per_arm * n_arms) from config (a small fraction).
    if out["divergence_rate"] is not None:
        assert 0.0 < out["divergence_rate"] < 0.2


def test_divergence_report_single_sustained_run(tmp_path: Path) -> None:
    """A single training that logs many monotone-increasing steps is ONE diverged run, not many."""
    _write_anomalies(tmp_path, [list(range(1000, 26000, 1000))])  # 25 lines, no reset
    out = AC.divergence_report(tmp_path)
    assert out["n_anomaly_lines"] == 25
    assert out["n_diverged_runs"] == 1
    assert out["transient_runs"] == 0


def test_divergence_report_winner_attribution_unavailable_without_ids(tmp_path: Path) -> None:
    """The prototype anomaly schema carries no candidate_id => winner attribution is 'unavailable', not faked."""
    _write_anomalies(tmp_path, [[1000, 2000], [500]])
    out = AC.divergence_report(tmp_path, winner_ids=("distributional-g6-c3",))
    assert out["n_diverged_runs"] == 2
    assert out["winner_diverged"] == []
    assert "unavailable" in out["winner_attribution"]


def test_divergence_report_winner_flagged_when_id_present(tmp_path: Path) -> None:
    """When events DO carry a candidate id, a winner's diverged run is flagged."""
    rows = [
        {"kind": "critic_explosion", "step": 1000, "candidate_id": "distributional-g6-c3"},
        {"kind": "critic_explosion", "step": 2000, "candidate_id": "distributional-g6-c3"},
        {"kind": "critic_explosion", "step": 500, "candidate_id": "scalar-g0-c1"},  # reset => run 2
    ]
    (tmp_path / "anomalies.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = AC.divergence_report(tmp_path, winner_ids=("distributional-g6-c3",))
    assert out["n_diverged_runs"] == 2
    assert out["winner_diverged"] == ["distributional-g6-c3"]
    assert out["winner_attribution"] == "ok"


def test_divergence_report_skips_when_no_log(tmp_path: Path) -> None:
    """No anomalies.jsonl (a clean run) => skipped, never raised."""
    out = AC.divergence_report(tmp_path)
    assert out["status"] == "skipped"


def test_divergence_report_reads_real_prototype_log() -> None:
    """The shipped outputs/prototype/anomalies.jsonl: 64 LINES => exactly 6 diverged RUNS (≈2.5%)."""
    log = Path(AC.__file__).resolve().parents[1] / "outputs" / "prototype" / "anomalies.jsonl"
    if not log.is_file():
        pytest.skip("prototype anomalies.jsonl not present")
    out = AC.divergence_report(log.parent)
    assert out["status"] == "ok"
    assert out["n_anomaly_lines"] == 64
    assert out["n_diverged_runs"] == 6


def test_divergence_markdown_renders_and_skips() -> None:
    md_skip = AC.divergence_markdown({"status": "skipped", "reason": "no log"})
    assert "n/a" in md_skip
    body = AC.divergence_markdown({
        "status": "ok", "n_anomaly_lines": 64, "n_diverged_runs": 6, "divergence_rate": 0.025,
        "n_candidates_budget": 240, "winner_diverged": [], "winner_attribution": "unavailable",
        "transient_runs": 3, "files_read": [], "note": "x",
    })
    assert "Training-divergence" in body
    assert "norm_reward=False" in body
    assert "NO winner" in body


# --------------------------------------------------------------------------- #
# compute_accounting (R35) — candidates attempted/accepted/failed + tokens     #
# --------------------------------------------------------------------------- #
def _write_llm_calls(arm_dir: Path, n_calls: int, in_tok: int = 1000, out_tok: int = 800) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "llm_calls.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n_calls):
            fh.write(json.dumps({
                "model": "m", "system": "s", "user": "u", "response": "r",
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }) + "\n")


def _write_failures(arm_dir: Path, n_failed: int) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "failures.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n_failed):
            fh.write(json.dumps({
                "candidate_id": f"c{i}", "prompt": "p", "reward_source": "bad", "error": "SandboxError",
            }) + "\n")


def test_compute_accounting_counts_and_tokens(tmp_path: Path) -> None:
    """LLM arm: accepted from records, failed from failures.jsonl, prompt-tokens summed from llm_calls.jsonl."""
    rng = np.random.default_rng(0)
    # 3 accepted distributional search candidates + a failures.jsonl with 1 gate failure + 4 llm calls.
    records = _records_from_matrix("distributional", rng.standard_normal((50, 3)))
    arm_dir = tmp_path / "distributional"
    _write_llm_calls(arm_dir, n_calls=4, in_tok=1000, out_tok=800)
    _write_failures(arm_dir, n_failed=1)

    out = AC.compute_accounting(records, tmp_path)
    assert out["status"] == "ok"
    row = next(r for r in out["rows"] if r["arm"] == "distributional")
    assert row["kind"] == "llm"
    assert row["n_accepted"] == 3
    assert row["n_failed"] == 1
    assert row["n_attempted"] == 4               # accepted + failed = the slots consumed
    assert row["n_llm_calls"] == 4
    assert row["prompt_tokens"] == 4000          # 4 calls * 1000
    assert row["completion_tokens"] == 3200      # 4 calls * 800
    assert row["resamples_to_full_slate"] is False
    assert row["tail_fed"] is True


def test_compute_accounting_search_arm_resamples_flag(tmp_path: Path) -> None:
    """Search arms carry the resamples-to-full-slate flag (the conservative-for-H2 asymmetry)."""
    rng = np.random.default_rng(1)
    records = _records_from_matrix("random_search", rng.standard_normal((40, 5)))
    out = AC.compute_accounting(records, tmp_path)
    row = next(r for r in out["rows"] if r["arm"] == "random_search")
    assert row["kind"] == "search"
    assert row["resamples_to_full_slate"] is True
    assert row["n_accepted"] == 5
    assert row["n_failed"] == 0                   # no failures.jsonl => 0, never raises
    assert row["prompt_tokens"] == 0             # search arms make no LLM calls
    assert row["tail_fed"] is False


def test_compute_accounting_campaign_search_leg_layout(tmp_path: Path) -> None:
    """The campaign writes provenance under search/<arm>/ — the finder probes there too."""
    _write_llm_calls(tmp_path / "search" / "scalar", n_calls=2, in_tok=500)
    _write_failures(tmp_path / "search" / "scalar", n_failed=3)
    out = AC.compute_accounting([], tmp_path, arms=("scalar",))
    row = out["rows"][0]
    assert row["n_failed"] == 3
    assert row["n_llm_calls"] == 2
    assert row["prompt_tokens"] == 1000


def test_compute_accounting_totals_and_markdown(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    records = _records_from_matrix("distributional", rng.standard_normal((30, 2)))
    _write_llm_calls(tmp_path / "distributional", n_calls=2, in_tok=100, out_tok=50)
    out = AC.compute_accounting(records, tmp_path)
    assert out["totals"]["prompt_tokens"] == 200
    assert out["totals"]["completion_tokens"] == 100
    md = AC.compute_accounting_markdown(out)
    assert "Compute-accounting" in md
    assert "burn" in md.lower() or "BURN" in md
    assert "n/a" in AC.compute_accounting_markdown({"status": "skipped", "reason": "x"})


def test_compute_accounting_reads_real_prototype(tmp_path: Path) -> None:
    """The shipped prototype archive: distributional has 40 LLM calls + 1 gate failure."""
    proto = Path(AC.__file__).resolve().parents[1] / "outputs" / "prototype"
    if not (proto / "distributional" / "llm_calls.jsonl").is_file():
        pytest.skip("prototype provenance not present")
    records = AC.load_campaign_records(proto)
    out = AC.compute_accounting(records, proto)
    row = next(r for r in out["rows"] if r["arm"] == "distributional")
    assert row["n_llm_calls"] == 40              # full slate of 40 LLM calls
    assert row["n_failed"] == 1                  # one gate failure burned a slot
    assert row["prompt_tokens"] > 0


# --------------------------------------------------------------------------- #
# campaign_pbo_dsr (R36, M3) — second PBO ranked on per-block Sharpe (DSR-proxy) #
# --------------------------------------------------------------------------- #
def test_pbo_dsr_in_unit_interval() -> None:
    rng = np.random.default_rng(20)
    records = _records_from_matrix("scalar", rng.standard_normal((300, 10)))
    out = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["scalar"]
    assert out["status"] == "ok"
    assert 0.0 <= out["pbo"] <= 1.0


def test_pbo_dsr_clean_fixture_low_pbo() -> None:
    """A fixture whose candidates have a stable Sharpe ORDER across blocks => low PBO on the Sharpe rule."""
    # Distinct, well-separated per-period means with the SAME low noise => the Sharpe order is stable
    # across any IS/OOS split, so the IS-best stays OOS-best => PBO near 0.
    t = 320
    rng = np.random.default_rng(21)
    cols = []
    for mu in np.linspace(0.001, 0.02, 8):
        v = rng.standard_normal(t) * 0.01 + mu
        cols.append(v)
    matrix = np.column_stack(cols)
    records = _records_from_matrix("distributional", matrix)
    out = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["distributional"]
    assert out["status"] == "ok"
    assert out["pbo"] < 0.3


def test_pbo_dsr_agrees_with_mean_return_pbo_when_vols_equal() -> None:
    """With λ=0, DSR is monotone in Sharpe; when all candidates share a vol, the per-block Sharpe rank ==
    the mean-return rank, so the two PBO columns should agree closely (the empirical A3 closure check)."""
    t = 320
    rng = np.random.default_rng(22)
    cols = [rng.standard_normal(t) * 0.01 + mu for mu in np.linspace(-0.005, 0.02, 9)]
    matrix = np.column_stack(cols)
    records = _records_from_matrix("scalar", matrix)
    primary = AC.campaign_pbo(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["scalar"]
    dsrcol = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["scalar"]
    assert primary["status"] == "ok" and dsrcol["status"] == "ok"
    # Equal-vol columns: the per-block Sharpe order tracks the per-block mean order closely.
    assert abs(primary["pbo"] - dsrcol["pbo"]) < 0.15


def test_pbo_dsr_does_not_mutate_frozen_pbo() -> None:
    """campaign_pbo_dsr must NOT alter overfitting.pbo: the primary PBO is identical before/after."""
    rng = np.random.default_rng(23)
    records = _records_from_matrix("scalar", rng.standard_normal((200, 6)))
    p1 = AC.campaign_pbo(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["scalar"]["pbo"]
    _ = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))
    p2 = AC.campaign_pbo(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))["scalar"]["pbo"]
    assert p1 == p2


def test_pbo_dsr_skips_too_few_candidates() -> None:
    records = _records_from_matrix("scalar", np.random.default_rng(24).standard_normal((200, 1)))
    out = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS)["scalar"]
    assert out["status"] == "skipped"


def test_pbo_dsr_markdown_two_columns_and_gap() -> None:
    rng = np.random.default_rng(25)
    records = _records_from_matrix("scalar", rng.standard_normal((300, 8)))
    primary = AC.campaign_pbo(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))
    dsrcol = AC.campaign_pbo_dsr(records, n_blocks=N_BLOCKS, rng=np.random.default_rng(0))
    md = AC.pbo_dsr_markdown(primary, dsrcol, n_blocks=N_BLOCKS)
    assert "DSR-proxy" in md
    assert "PRIMARY" in md
    assert "n/a" in AC.pbo_dsr_markdown(primary, {"status": "error", "reason": "x"}, n_blocks=N_BLOCKS)


def test_load_campaign_records_skips_h3_singleshot_subtrees(tmp_path: Path) -> None:
    """M15 (2026-07-05): the H3 single-shot control's ``*_h3_singleshot/`` subtrees live under the
    SAME campaign output_dir with arm='distributional' and colliding run_id patterns — the default
    walker must NEVER pool them into the headline records (the H3 analysis loads them explicitly
    via its own ``single_shot_root``)."""
    from src.io.results import write_run

    base = dict(
        arm="distributional", seed=0, fold=0, generation=0, reward_source_hash="h",
        feedback_block="", wall_clock=0.0, env_fingerprint="x",
    )
    write_run({**base, "run_id": "distributional-s0-distributional-g0-c0",
               "candidate_id": "c0", "metrics": {"val_fitness": 0.4}},
              tmp_path / "search" / "distributional")
    # The single-shot twin — a DIFFERENT candidate that must stay invisible to the default walk.
    write_run({**base, "run_id": "distributional-s0-distributional-g0-c99",
               "candidate_id": "c99", "metrics": {"val_fitness": 0.9}},
              tmp_path / "search_h3_singleshot" / "distributional")

    records = AC.load_campaign_records(tmp_path)
    ids = {r["run_id"] for r in records}
    assert "distributional-s0-distributional-g0-c0" in ids
    assert "distributional-s0-distributional-g0-c99" not in ids
    assert len(records) == 1


# --------------------------------------------------------------------------- #
# Duplicate (arm, seed) handling in _seed_scores (deep review 2026-07-26)      #
# --------------------------------------------------------------------------- #
def _dup_rec(arm: str, seed: int, vals: list[float]) -> dict:
    """Minimal frozen-winner TEST record: _test_returns reads metrics['test_returns']."""
    return {"arm": arm, "seed": seed, "metrics": {"test_returns": list(vals)}}


def test_seed_scores_allows_agreeing_duplicate_records() -> None:
    """An IDENTICAL duplicate (arm, seed) must NOT fail the analysis.

    `src/io/results.py::load_all` de-duplicates nothing, and `--resume` / winner re-runs can legitimately
    write a SECOND run directory for a seed that already has a record. Loop 6 of the deep review made ANY
    duplicate raise; loop 13 refined that, because hard-failing a valid campaign analysis at the last step
    on a benign re-write is a worse trade than the bug it prevents.
    """
    recs = [_dup_rec("winner", 1, [0.01, -0.02, 0.03]), _dup_rec("winner", 1, [0.01, -0.02, 0.03])]
    out = AC._seed_scores(recs, "winner", lambda v: float(np.mean(v)))
    assert set(out) == {1}


def test_seed_scores_raises_on_conflicting_duplicate_records() -> None:
    """A DISAGREEING duplicate must fail loud — this is the case last-wins silently got wrong.

    Every headline estimator is PAIRED on the seed, so arbitrarily keeping whichever record was written
    last would shift a paired difference with nothing in the output to say it happened.
    """
    recs = [_dup_rec("winner", 1, [0.01, 0.01, 0.01]), _dup_rec("winner", 1, [0.99, 0.99, 0.99])]
    with pytest.raises(ValueError, match="CONFLICTING duplicate test records"):
        AC._seed_scores(recs, "winner", lambda v: float(np.mean(v)))


# --- #100: the H4 report must track the REALISED contrast set, not a hardcoded 2 --------------

def test_h4_markdown_derives_its_counts_and_uses_the_authoritative_reference_labels():
    """The H4 family grew 2 -> 4 on 2026-07-26 (+cma_es/tpe) and the emitter did not follow.

    Two independent defects, both in a CONFIRMATORY node's reported table:
      * every count was hardcoded to 2 — "Two pre-registered difference tests", "Own 2-test family",
        "Bonferroni-over-2", a "Bonf-2" column header, and a fallback ``alpha`` default of 0.025
        (= alpha/2) — while the CODE had been correcting over 4 all along. The arithmetic was right;
        the reported description of the multiplicity was wrong by a factor of two.
      * the reference column was ``"in-family ref" if test == "h4a" else "fixed-template ref"``, a
        two-way branch that mislabelled BOTH h4c (CMA-ES) and h4d (TPE) with h4b's Bayes-opt framing,
        even though ``_H4_REFERENCE_FRAMING`` carries a distinct label for each and it is already
        attached to every test row.

    Pins the PROPERTIES (counts derived from the realised set; labels taken from the authoritative
    field) so a future expansion cannot silently re-stale the report.
    """
    from scripts.analyze_campaign import H4_CONTRASTS, _H4_REFERENCE_FRAMING, h4_markdown

    n = len(H4_CONTRASTS)
    tests = [
        {"test": tid, "a": a, "b": b, "reference": _H4_REFERENCE_FRAMING.get(tid, ""),
         "effect": 0.01, "pvalue_one_sided": 0.02, "reject_one_sided": True,
         "reject_one_sided_bonferroni": False, "equivalence": {"equivalent": True},
         "verdict": "beats", "n_seeds": 30}
        for tid, a, b in H4_CONTRASTS
    ]
    md = h4_markdown({
        "status": "ok", "winner_arm": "distributional", "alpha": 0.05, "n_tests": n,
        "bonferroni_alpha": 0.05 / n, "equiv_margin": 0.05, "tests": tests, "skipped": [],
        "all_supported": True, "all_supported_bonferroni": False, "all_equivalent": True,
    })

    assert f"Bonferroni-over-{n}" in md, "the reported Bonferroni count does not match the family size"
    assert f"Bonf-{n}" in md, "the table header's Bonferroni count is stale"
    assert f"Own {n}-test family" in md, "the family size in the prose is stale"
    assert "Bonferroni-over-2" not in md and "Bonf-2)" not in md

    # Each leg must carry ITS OWN framing, not a neighbour's.
    for tid, _a, _b in H4_CONTRASTS:
        lead = _H4_REFERENCE_FRAMING[tid].split(" \u2014 ")[0].split(" -- ")[0]
        assert lead in md, f"{tid} is missing its authoritative reference label ({lead!r})"


# --- #101: the N6 IUT must be exercised at the REGISTERED canon size, not a frozen subset -------

def test_n6_iut_is_exercised_at_the_FULL_registered_canon() -> None:
    """The confirmatory N6 node must be guarded at the size it actually runs at.

    The hand-reward canon was expanded **4 -> 11** on 2026-07-26 (R105/R108), but every N6 guard here
    uses ``_H1_BASELINES`` — the four pre-expansion names — so the node's behaviour at its REGISTERED
    comparator size was never exercised. That is the same shape as #100, where a guard whose coverage
    stayed frozen at 2 legs let a real mislabelling of h4c/h4d survive: a test that never runs the new
    members cannot fail on them.

    No production defect was found (the code is roster-agnostic and was verified correct at 11), so
    this closes the GUARD, not a bug. The canon is read from the frozen config rather than hardcoded,
    so a future expansion is picked up automatically instead of silently narrowing coverage again.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    canon = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))["h1_baselines"]
    assert len(canon) >= len(_H1_BASELINES), "the registered canon shrank below the legacy test set"
    assert set(_H1_BASELINES) <= set(canon), "the fast-path test names are no longer registered members"

    records = _seeded_test_records("distributional", mu=0.010)
    for i, name in enumerate(canon):
        records += _seeded_test_records(f"baseline_{name}", mu=0.001, seed0=100 * (i + 1))

    h1 = AC.beat_human_baseline(records, baseline_names=canon, winner_arm="distributional",
                                rng=np.random.default_rng(0))
    iut = h1["iut"]
    assert iut["n_baselines"] == len(canon), "the IUT did not span the whole registered canon"
    assert iut["n_testable"] == len(canon)
    assert iut["all_baselines_present"] is True
    assert len(iut["dominance_profile"]) == len(canon), "a canon member is missing from the profile"
    # Every member appears exactly once, so none is silently dropped from the conjunction.
    assert sorted(lg["baseline"] for lg in iut["dominance_profile"]) == sorted(canon)
    # The LLM beats all of them here, so the IUT p is the MAX leg p and dominance holds.
    assert iut["dominates_canon"] is True
    assert iut["iut_pvalue"] == max(lg["pvalue_one_sided"] for lg in iut["dominance_profile"])


def test_n6_refuses_dominance_when_a_registered_canon_member_is_MISSING() -> None:
    """Drop ONE member: the conjunction must go un-certifiable, not quietly succeed on the rest.

    This is the anti-conservative failure the ``all_baselines_present`` guard exists to prevent, and
    at the 4-name subset it was only ever exercised at a quarter of the registered breadth.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    canon = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))["h1_baselines"]

    records = _seeded_test_records("distributional", mu=0.010)
    for i, name in enumerate(canon[:-1]):          # the LAST registered member never ran
        records += _seeded_test_records(f"baseline_{name}", mu=0.001, seed0=100 * (i + 1))

    iut = AC.beat_human_baseline(records, baseline_names=canon, winner_arm="distributional",
                                 rng=np.random.default_rng(0))["iut"]
    assert iut["n_baselines"] == len(canon)
    assert iut["n_testable"] == len(canon) - 1
    assert iut["all_baselines_present"] is False
    assert iut["dominates_canon"] is False, (
        "dominance was certified while a registered canon member was never compared — the LLM would "
        "be claimed to beat 'the best human' without having faced one of them"
    )


def test_ARMS_matches_the_frozen_arms_yaml_roster() -> None:
    """`ARMS` is the DEFAULT arm set for five analyses — it must not drift from the frozen roster.

    #102 (2026-07-27): it listed SEVEN. `cma_es` and `tpe` were never added when the roster grew
    7 -> 9 on 2026-07-26, while the constant's own comment cited "PREREGISTRATION §3" as authority —
    and §3's heading reads "The nine arms". Because an arm absent from this tuple produces no row AT
    ALL (not even a "skipped" row), two registered arms were silently invisible to `campaign_pbo` (the
    PRIMARY overfitting diagnostic), `campaign_pbo_dsr`, `winner_dsr`, `compute_accounting` and
    `model_confidence_set_report` — whose MCS composition depends on which arms are entered.

    The pre-existing guard could not catch it: `assert set(out) == set(AC.ARMS)` derives the
    expectation from the SAME constant, so it is satisfied by any value. This asserts against the
    frozen config instead, mirroring the config-vs-code check that already keeps `_BENCHMARK_NAMES`
    honest.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "arms.yaml").read_text(encoding="utf-8"))
    entries = cfg.get("arms", cfg) if isinstance(cfg, dict) else cfg
    registered = [e["name"] if isinstance(e, dict) else e for e in entries]

    assert set(AC.ARMS) == set(registered), (
        f"analyze_campaign.ARMS drifted from config/arms.yaml — missing "
        f"{sorted(set(registered) - set(AC.ARMS))}, extra {sorted(set(AC.ARMS) - set(registered))}. "
        "An arm absent from ARMS is invisible to PBO/DSR/MCS, with no skipped row to reveal it."
    )
    assert list(AC.ARMS) == list(registered), "ARMS order drifted from the frozen roster"


def test_the_MCS_bootstrap_seed_is_a_REAL_knob_and_its_default_IS_the_registered_pin() -> None:
    """#108 (2026-07-27): `model_confidence_set_report` took an `rng` it silently ignored.

    The block ran `arch`'s 1,000-rep MCS bootstrap at a HARDCODED `seed=0` whatever the caller
    passed, so a bootstrap-sensitivity re-run — `rng=default_rng(999_999)` — returned BIT-IDENTICAL
    p-values (measured) and would have invited the false conclusion that the MCS carries no
    resampling noise. It does: at 1,000 reps the elimination p-values move by ~0.005-0.016 between
    seeds. The knob is now a real `seed`, and `0` (the R69-registered pin) is its default, so the
    campaign number is unchanged.

    Two assertions, because either alone is weak: the SPY pins the contract structurally (the seed
    must reach the estimator) even if a future `arch` made the noise vanish, and the BEHAVIOURAL half
    proves the threading actually bites rather than merely being passed somewhere harmless.
    """
    pytest.importorskip("arch")

    gen = np.random.default_rng(7)
    records = [
        {"run_id": f"{arm}-{s}", "arm": arm, "seed": s, "fold": 0,
         "metrics": {"test_returns": [float(x) for x in gen.normal(4e-4 + 4e-5 * i, 0.011, 260)]}}
        for i, arm in enumerate(AC.ARMS)
        for s in range(12)
    ]

    seen: list[int] = []
    real = AC.model_confidence_set_report

    import src.inference.model_confidence_set as MCS_MOD

    original = MCS_MOD.model_confidence_set

    def _spy(arm_scores, **kw):  # type: ignore[no-untyped-def]
        seen.append(kw.get("seed"))
        return original(arm_scores, **kw)

    MCS_MOD.model_confidence_set = _spy  # type: ignore[assignment]
    try:
        default = real(records)
        pinned = real(records, seed=0)
        moved = real(records, seed=12345)
    finally:
        MCS_MOD.model_confidence_set = original  # type: ignore[assignment]

    assert seen, "the report never reached src.inference.model_confidence_set"
    assert seen[:2] == [0, 0], f"the DEFAULT seed is not the registered R69 pin of 0; saw {seen[:2]}"
    assert 12345 in seen, f"the caller's seed never reached the estimator; seeds seen: {seen}"

    assert default == pinned, "the default no longer reproduces the registered seed=0 campaign number"
    assert moved != default, (
        "changing the MCS bootstrap seed left the report BIT-IDENTICAL — the seed is being ignored "
        "again, so a resampling-sensitivity check would silently report zero bootstrap noise"
    )


def test_H4_CONTRASTS_matches_the_registered_N4_comparator_portfolio() -> None:
    """H4 is DELIBERATELY outside the frozen-family guard, so its roster needs its own.

    `assert_realized_family_matches_frozen` runs on the real path and transitively protects
    `H2_CONTRASTS`: if it drifted, the realized tests would stop matching the frozen family and it
    would raise. H4 is deliberately excluded from that mechanism — `h4_search_controls` writes a
    DISJOINT key precisely "so ``assert_realized_family_matches_frozen`` never sees it" and the m=6
    family stays clean — but nothing was put in its place, leaving the CONFIRMATORY N4 node's
    comparator portfolio unguarded.

    That is the #102 gap (ARMS drifted 7 vs 9 with only a tautological test) on the node whose
    constant has ALREADY grown once: #100 caught H4 expanding 2 -> 4 with the report left behind.
    The registered source is ``inference.validity_tier.nodes.N4_h4.comparators`` — "beat the MAX of
    the {random, GP-EI, CMA-ES, TPE} portfolio". If a fifth optimiser arm were registered and this
    tuple not updated, N4 would quietly claim to beat "the best optimiser" while never facing it.
    """
    from pathlib import Path

    import yaml

    from scripts.analyze_campaign import H4_CONTRASTS, _H4_REFERENCE_FRAMING

    root = Path(__file__).resolve().parents[1]
    pre = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    registered = list(pre["inference"]["validity_tier"]["nodes"]["N4_h4"]["comparators"])
    coded = [b for _tid, _a, b in H4_CONTRASTS]

    assert coded == registered, (
        f"H4_CONTRASTS drifted from the registered N4 portfolio: code={coded}, "
        f"registered={registered}. N4 is the confirmatory beat-the-MAX IUT, so a comparator missing "
        "here is one the LLM is never tested against while the claim still says 'the best'."
    )
    # Every leg must have a reference-framing label, or the H4 table falls back to a bare test id
    # for that row — the #100 failure mode, one step upstream.
    missing = [tid for tid, _a, _b in H4_CONTRASTS if not _H4_REFERENCE_FRAMING.get(tid)]
    assert not missing, f"H4 legs with no _H4_REFERENCE_FRAMING label: {missing}"
