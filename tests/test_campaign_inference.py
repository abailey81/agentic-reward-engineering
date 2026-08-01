"""Fast tests for the Rank-8 campaign inference (scripts/analyze_campaign.py).

No torch, no real training: synthetic per-(arm, seed) TEST-leg records (carrying
``metrics['test_returns']``) drive the multiple-testing family, the H2 conjunction, and
the DeMiguel 1/N benchmark floor. Covers (per the punch-list):

  - the H2 conjunction is supported ONLY when ALL THREE legs reject in the predicted
    direction post-correction (one tied/worse leg -> not supported);
  - the Benjamini-Hochberg family rejection set matches ``benjamini_hochberg(pvals)``;
  - ``romano_wolf_joint`` draws ONE shared bootstrap path per replication (joint stepdown)
    and rejects a strong leg while sparing a null leg;
  - the benchmark floor rolls a known WEIGHT policy through the IDENTICAL costed env and
    the 1/N per-step gross return matches the hand-computed equal-weight value;
  - the floor gate flags pass/fail (winner DSR vs best-benchmark DSR), never re-selecting.

These mirror the proven ``tests/test_inference.py`` / ``tests/test_analyze_campaign.py``
fixtures, lifted to the TEST-leg record layer the campaign writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402

from src.baselines.strategies import equal_weight  # noqa: E402
from src.data.synthetic import make_synthetic_panel  # noqa: E402
from src.env.portfolio_env import PortfolioEnv, project_simplex  # noqa: E402
from src.inference.multiple_testing import benjamini_hochberg  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _test_record(arm: str, seed: int, vec: np.ndarray) -> dict:
    """A minimal per-(arm, seed) frozen-winner TEST record carrying metrics['test_returns']."""
    tr = [float(x) for x in vec]
    return {
        "run_id": f"{arm}-s{seed}",
        "arm": arm,
        "seed": seed,
        "fold": 0,
        "candidate_id": f"{arm}-winner",
        "generation": 0,
        "reward_source_hash": "deadbeef",
        "feedback_block": "",
        "metrics": {"val_fitness": 0.0, "test_returns": tr, "per_period_pnl": tr},
        "wall_clock": 0.0,
        "env_fingerprint": "test",
        "frozen": True,
        "test_returns": tr,
    }


_N_SEEDS = 12  # multi-seed: the per-seed rliable inference needs >= 2 (the campaign uses 30)


def _arm_records(means: dict[str, float], *, t: int = 400, n_seeds: int = _N_SEEDS, seed: int = 0) -> list[dict]:
    """One TEST record per (arm, SEED): each seed's i.i.d. normal series at the arm's mean + a small
    per-seed shift — the across-seed (training-RNG) variance the per-seed inference is built to carry.

    The headline inference (#9/#14 fix) is rliable per-seed (IQM + paired across-seed bootstrap), so a
    realistic test needs MULTIPLE seeds per arm; a single record per arm could not exercise it.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for arm, mu in means.items():
        for s in range(n_seeds):
            vec = rng.standard_normal(t) * 0.01 + mu + rng.normal(0.0, 0.0004)
            recs.append(_test_record(arm, s, vec))
    return recs


# --------------------------------------------------------------------------- #
# collect_family_pvalues — the arm x metric family + BH                         #
# --------------------------------------------------------------------------- #
def test_collect_family_enumerates_leg_by_metric_family() -> None:
    recs = _arm_records(
        {"distributional": 0.004, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=1,
    )
    fam = AC.collect_family_pvalues(recs, n_boot=400, rng=np.random.default_rng(1))
    # 3 contrasts x (1 Sharpe + 1 CVaR@0.05) = 6 hypotheses.
    assert fam["n_family"] == 6
    assert len(fam["labels"]) == 6
    assert {"sharpe"} <= {t["metric"] for t in fam["tests"]}
    assert {"cvar"} <= {t["metric"] for t in fam["tests"]}
    assert fam["pvals"].shape == (6,)


def test_collect_family_bh_matches_pvals() -> None:
    """The returned FDR set IS benjamini_hochberg(pvals, q) — no bespoke correction."""
    recs = _arm_records(
        {"distributional": 0.004, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=2,
    )
    fam = AC.collect_family_pvalues(recs, n_boot=400, q=0.05, rng=np.random.default_rng(2))
    expected = benjamini_hochberg(fam["pvals"], q=0.05)
    np.testing.assert_array_equal(fam["reject_bh"], expected)


def test_collect_family_extra_cvar_level_grows_family() -> None:
    """Passing the frozen cvar_01 level adds one CVaR hypothesis per contrast."""
    recs = _arm_records(
        {"distributional": 0.004, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=3,
    )
    fam = AC.collect_family_pvalues(
        recs, cvar_levels=(0.05, 0.01), n_boot=300, rng=np.random.default_rng(3)
    )
    # 3 contrasts x (1 Sharpe + 2 CVaR) = 9.
    assert fam["n_family"] == 9


def test_collect_family_skips_missing_arm() -> None:
    """A contrast whose arm has no test record is reported in 'skipped', never fabricated."""
    recs = _arm_records({"distributional": 0.004, "scalar": 0.0}, seed=4)
    fam = AC.collect_family_pvalues(recs, n_boot=300, rng=np.random.default_rng(4))
    # Only the distributional>scalar contrast survives; the other two are skipped.
    assert fam["n_family"] == 2
    skipped_pairs = {(s["arm_a"], s["arm_b"]) for s in fam["skipped"]}
    assert ("distributional", "placebo") in skipped_pairs
    assert ("distributional", "scalar_cvar5") in skipped_pairs


# --------------------------------------------------------------------------- #
# h2_conjunction — the pre-registered headline (ALL THREE legs)                 #
# --------------------------------------------------------------------------- #
def test_h2_supported_only_when_all_three_legs_reject() -> None:
    """distributional clearly beats scalar, placebo AND scalar_cvar5 -> H2 supported."""
    recs = _arm_records(
        {"distributional": 0.005, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=5,
    )
    res = AC.h2_conjunction(recs, q=0.05, n_boot=600, rng=np.random.default_rng(5))
    assert res["H2_supported"] is True
    assert len(res["legs"]) == 3
    assert all(leg["leg_supported"] for leg in res["legs"])
    assert all(leg["sharpe_direction_ok"] for leg in res["legs"])


def test_h2_not_supported_under_true_null() -> None:
    """When ALL arms share the SAME distribution (true null), the per-seed conjunction must NOT
    support H2. This is the #9/#14 fix: the prior seed-AVERAGED inference collapsed the across-seed
    variance and over-rejected a true null (≈21% vs the correct ≈5% in a representative 30-seed
    calibration — the inflation scales with the across-seed variance); the per-seed rliable test
    here is correctly sized, so a true null is (overwhelmingly) not declared a distributional win."""
    recs = _arm_records(
        {"distributional": 0.0, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=42,
    )
    res = AC.h2_conjunction(recs, q=0.05, n_boot=600, rng=np.random.default_rng(42))
    assert res["H2_supported"] is False


def test_h2_not_supported_when_one_leg_ties() -> None:
    """scalar_cvar5 as good as distributional -> that leg fails -> conjunction fails."""
    recs = _arm_records(
        {
            "distributional": 0.005,
            "scalar": 0.0,
            "placebo": 0.0,
            "scalar_cvar5": 0.005,  # ties distributional -> no rejection in direction
        },
        seed=6,
    )
    res = AC.h2_conjunction(recs, q=0.05, n_boot=600, rng=np.random.default_rng(6))
    assert res["H2_supported"] is False
    legs = {leg["contrast"]: leg for leg in res["legs"]}
    assert legs["distributional>scalar"]["leg_supported"] is True
    assert legs["distributional>placebo"]["leg_supported"] is True
    assert legs["distributional>scalar_cvar5"]["leg_supported"] is False


def test_h2_not_supported_when_wrong_direction() -> None:
    """If distributional is WORSE than a comparator, that leg cannot support H2."""
    recs = _arm_records(
        {
            "distributional": 0.0,
            "scalar": 0.0,
            "placebo": 0.0,
            "scalar_cvar5": 0.006,  # better than distributional -> wrong direction
        },
        seed=7,
    )
    res = AC.h2_conjunction(recs, q=0.05, n_boot=400, rng=np.random.default_rng(7))
    assert res["H2_supported"] is False
    legs = {leg["contrast"]: leg for leg in res["legs"]}
    assert legs["distributional>scalar_cvar5"]["sharpe_direction_ok"] is False


def test_h2_missing_arm_is_unsupported_not_error() -> None:
    """A null is credible: a missing comparator arm -> H2 unsupported, no exception."""
    recs = _arm_records({"distributional": 0.005, "scalar": 0.0}, seed=8)
    res = AC.h2_conjunction(recs, n_boot=300, rng=np.random.default_rng(8))
    assert res["H2_supported"] is False
    assert "distributional>placebo" in res["missing"]
    assert "distributional>scalar_cvar5" in res["missing"]


def test_analyze_entrypoint_wires_h2_into_report(tmp_path) -> None:
    """#18 regression: analyze() runs the H2 conjunction and write_report/h2_markdown emit it. The
    headline test was implemented + unit-tested but previously had NO caller in the analysis entry
    point, so the documented headline result never actually ran."""
    from src.io.results import write_run

    recs = _arm_records(
        {"distributional": 0.0016, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0}, seed=11
    )
    for r in recs:  # persist per-(arm,seed) records into a campaign archive
        write_run(r, tmp_path)
    result = AC.analyze(tmp_path, n_blocks=4)
    assert "h2" in result
    assert isinstance(result["h2"]["H2_supported"], bool)
    assert len(result["h2"]["legs"]) == 3
    report = AC.write_report(result, tmp_path)  # emits the H2 section
    md = report.read_text(encoding="utf-8")
    assert "H2 (distributional feedback)" in md
    assert "distributional>scalar" in AC.h2_markdown(result["h2"])


def test_h2_conjunction_romano_wolf_method() -> None:
    """The JOINT Romano-Wolf family also supports H2 only when all three legs reject."""
    recs = _arm_records(
        {"distributional": 0.005, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=9,
    )
    res = AC.h2_conjunction(recs, method="rw", alpha=0.05, n_boot=600, rng=np.random.default_rng(9))
    assert res["method"] == "rw"
    assert res["H2_supported"] is True
    assert all(leg["leg_supported"] for leg in res["legs"])


# --------------------------------------------------------------------------- #
# R25 — the TWO co-primary intersection-union tests (H2-RA Sharpe + H2-Tail CVaR-5%)  #
# --------------------------------------------------------------------------- #
def _tail_record(arm: str, seed: int, *, mu: float, sd: float, heavy: bool, rng_off: int, t: int = 600) -> dict:
    """A per-(arm, seed) TEST record with a controllable LEFT-TAIL severity at a matched (mu, sd).

    ``heavy=True`` draws a Student-t(df=3) body (fat left tail -> a MORE-negative CVaR-5%) rescaled to
    the SAME (mu, sd) as the Gaussian ``heavy=False`` body, so the per-seed Sharpe (~mu/sd) is matched
    across arms while the CVaR-5% diverges. This DECOUPLES the risk-adjusted (Sharpe) and tail (CVaR)
    dimensions, letting a test exercise "only H2-Tail rejects" deterministically. ``seed`` is the SHARED
    training-seed label (paired across arms by the inference); ``rng_off`` varies only the RNG stream so
    each arm draws an independent realization at the same seed index.
    """
    rng = np.random.default_rng(7000 + rng_off + seed)
    if heavy:
        x = rng.standard_t(3, size=t)
    else:
        x = rng.standard_normal(t)
    x = (x - x.mean()) / x.std(ddof=0)
    v = x * sd + mu
    return _test_record(arm, seed, v)


def _tail_records(arm: str, *, mu: float, sd: float, heavy: bool, n_seeds: int = _N_SEEDS, rng_off: int = 0) -> list[dict]:
    # The record `seed` is `s` (SHARED across arms so the paired-by-seed inference can align them); the
    # per-arm `rng_off` only changes the random STREAM, not the seed label.
    return [_tail_record(arm, s, mu=mu, sd=sd, heavy=heavy, rng_off=rng_off) for s in range(n_seeds)]


def test_h2_ra_supported_when_all_three_sharpe_reject() -> None:
    """All three Sharpe legs reject one-sided in direction -> H2-RA supported; two-tier verdict exposed."""
    recs = _arm_records(
        {"distributional": 0.005, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=51,
    )
    res = AC.h2_conjunction(recs, q=0.05, alpha=0.05, n_boot=800, rng=np.random.default_rng(51))
    # The two-tier structure is present and H2-RA (the Sharpe IUT) is supported.
    assert res["H2_RA"]["supported"] is True
    assert res["H2_supported"] is res["H2_RA"]["supported"]  # back-compat alias mirrors the Sharpe IUT
    assert len(res["H2_RA"]["legs"]) == 3
    assert all(leg["leg_supported"] for leg in res["H2_RA"]["legs"])
    # Each RA leg decision is the genuinely ONE-SIDED reject (not the BH-over-6 set).
    assert all("pvalue_one_sided" in leg for leg in res["H2_RA"]["legs"])
    assert res["verdict"].startswith("H2-RA")


def test_h2_tail_supported_ra_not_when_only_cvar_rejects() -> None:
    """Distributional has a thinner LEFT tail (better CVaR-5%) at a TIED Sharpe -> only H2-Tail rejects.

    Comparators draw a fat-left-tail Student-t body at the SAME (mu, sd) as distributional's Gaussian
    body (plus a hair more mean), so the per-seed Sharpe IQM is ~tied (H2-RA must NOT reject) while the
    CVaR-5% IQM is clearly less-negative for distributional (H2-Tail rejects in direction). This is the
    prototype's 'CVaR p<0.05 at parity of risk-adjusted mean' pattern that R25 makes bankable.
    """
    recs = (
        _tail_records("distributional", mu=0.0009, sd=0.012, heavy=False, rng_off=0)
        + _tail_records("scalar", mu=0.0010, sd=0.012, heavy=True, rng_off=1000)
        + _tail_records("placebo", mu=0.0010, sd=0.012, heavy=True, rng_off=2000)
        + _tail_records("scalar_cvar5", mu=0.0010, sd=0.012, heavy=True, rng_off=3000)
    )
    res = AC.h2_conjunction(recs, q=0.05, alpha=0.05, n_boot=1500, rng=np.random.default_rng(5))
    assert res["H2_Tail"]["supported"] is True
    assert res["H2_RA"]["supported"] is False
    assert res["verdict"] == "H2-Tail supported (risk-adjusted not)"
    # All three TAIL legs reject one-sided in direction; the Sharpe legs do not.
    assert all(leg["leg_supported"] for leg in res["tail_legs"])
    assert not any(leg["leg_supported"] for leg in res["legs"])
    # The tail IUT is gated at the headline CVaR level (frozen 0.05) and flags FZ0/ES corroboration.
    assert res["H2_Tail"]["level"] == 0.05
    assert res["H2_Tail"]["corroborated_by"] == "fz0_var_es_comparative_backtest"


def test_h2_iut_needs_no_bh_single_non_rejecting_leg_fails_family() -> None:
    """The IUT is the correction: ONE non-rejecting leg fails the family with NO Benjamini-Hochberg step.

    Two legs are strong (distributional >> placebo, scalar_cvar5) but the distributional>scalar Sharpe
    leg is a true null. Under the R25 one-sided IUT (no leg correction) the family is NOT supported, and
    the gate is decided on ``reject_one_sided`` — never on the BH-over-6 set (``reject_bh``). We also
    confirm the BH set would NOT rescue the family (the IUT, not BH, is what gates).
    """
    recs = _arm_records(
        {"distributional": 0.0, "scalar": 0.0, "placebo": -0.006, "scalar_cvar5": -0.006},
        seed=61,
    )
    res = AC.h2_conjunction(recs, q=0.05, alpha=0.05, n_boot=800, rng=np.random.default_rng(61))
    assert res["H2_RA"]["supported"] is False  # the scalar leg is null -> the conjunction fails
    legs = {leg["contrast"]: leg for leg in res["legs"]}
    assert legs["distributional>scalar"]["leg_supported"] is False
    assert legs["distributional>placebo"]["leg_supported"] is True
    assert legs["distributional>scalar_cvar5"]["leg_supported"] is True
    # The leg decision is the one-sided reject, and a single non-rejecting leg is decisive: the family
    # verdict is the AND of the per-leg one-sided rejects, with NO BH involved in the gate.
    sharpe_one_sided = {
        (t["arm_a"], t["arm_b"]): t["reject_one_sided"]
        for t in res["family"]["tests"] if t["metric"] == "sharpe"
    }
    assert sharpe_one_sided[("distributional", "scalar")] is False
    assert res["H2_RA"]["supported"] == all(
        sharpe_one_sided[(a, b)] for (a, b) in AC.H2_CONTRASTS
    )


def test_h2_tail_gates_at_headline_level_not_optin_cvar01() -> None:
    """The tail IUT gates at the HEADLINE CVaR-5% even when the opt-in cvar_01 is added (it must NOT gate).

    cvar_01 is more extreme (high-variance; Bauer 2025) and is reporting-only; passing it must leave the
    H2-Tail gating level at 0.05 and the verdict driven by the 0.05 legs, with the 0.01 legs only swelling
    the reported family.
    """
    recs = _arm_records(
        {"distributional": 0.005, "scalar": 0.0, "placebo": 0.0, "scalar_cvar5": 0.0},
        seed=71,
    )
    res = AC.h2_conjunction(recs, q=0.05, alpha=0.05, cvar_levels=(0.05, 0.01), n_boot=600,
                            rng=np.random.default_rng(71))
    assert res["H2_Tail"]["level"] == 0.05            # gated at the headline level, NOT 0.01
    assert len(res["tail_legs"]) == 3                  # exactly the 3 CVaR-5% legs, not 6
    assert res["family"]["n_family"] == 9              # the 0.01 legs still swell the reported family


def test_h2_two_family_frozen_assert_partitions_union() -> None:
    """The frozen two co-primary IUT sub-families partition the m=6 union (R25); the realized assert holds.

    Builds the realized family from the FROZEN mirror's m=6 ``members`` and confirms
    ``assert_realized_family_matches_frozen`` passes (it now also checks the ``families.{h2_ra, h2_tail}``
    sub-families are disjoint, sized to their own m, and union to the 6). A doctored realized family that
    drops one member must still fail-loud.
    """
    fam = load_config("preregistration").get("inference", {}).get("testing_family")
    if not fam or not fam.get("members"):
        pytest.skip("no frozen testing_family mirror in config/preregistration.yaml")
    if not fam.get("families"):
        pytest.skip("frozen testing_family has no two-family (R25) structure")

    cvar_levels = tuple(float(x) for x in fam.get("cvar_levels", [0.05]))
    realized = [
        {"arm_a": m["arm_a"], "arm_b": m["arm_b"], "metric": m["metric"], "level": m.get("level")}
        for m in fam["members"]
    ]
    # The honest realized family validates (union match + sub-family partition).
    AC.assert_realized_family_matches_frozen(realized, cvar_levels=cvar_levels)

    # The two sub-families are disjoint and sum to the union.
    ra = fam["families"]["h2_ra"]
    tail = fam["families"]["h2_tail"]
    assert ra["m"] == len(ra["members"]) == 3
    assert tail["m"] == len(tail["members"]) == 3
    assert ra["m"] + tail["m"] == fam["m"] == 6
    assert {m["metric"] for m in ra["members"]} == {"sharpe"}
    assert {m["metric"] for m in tail["members"]} == {"cvar"}

    # A drifted realized family (one member dropped) must fail-loud, exactly as before R25.
    with pytest.raises(AssertionError):
        AC.assert_realized_family_matches_frozen(realized[:-1], cvar_levels=cvar_levels)


# --------------------------------------------------------------------------- #
# romano_wolf_joint — one shared bootstrap path per replication                 #
# --------------------------------------------------------------------------- #
def test_romano_wolf_joint_rejects_strong_spares_null() -> None:
    rng = np.random.default_rng(10)
    # Per-seed Sharpe SCORES per arm (the rliable unit, aligned by seed): distributional strong, rest null.
    scores = {
        "distributional": rng.normal(8.0, 0.6, _N_SEEDS),  # high per-seed Sharpe
        "scalar": rng.normal(0.0, 0.6, _N_SEEDS),          # null vs distributional
        "placebo": rng.normal(0.0, 0.6, _N_SEEDS),         # null
        "scalar_cvar5": rng.normal(0.0, 0.6, _N_SEEDS),    # null
    }
    out = AC.romano_wolf_joint(scores, n_boot=600, alpha=0.05, rng=np.random.default_rng(11))
    rej = {lbl: bool(r) for lbl, r in zip(out["labels"], out["reject_rw"])}
    assert rej["distributional>scalar"] is True
    assert out["stats"].shape == (3,)
    # All three legs share the SAME comparator family direction (distributional better).
    assert all(out["direction_ok"])


def test_romano_wolf_joint_missing_arm_raises() -> None:
    scores = {"distributional": np.zeros(_N_SEEDS), "scalar": np.zeros(_N_SEEDS)}  # placebo/scalar_cvar5 absent
    with pytest.raises(KeyError):
        AC.romano_wolf_joint(scores, n_boot=50)


# --------------------------------------------------------------------------- #
# benchmark_floor — roll 1/N through the IDENTICAL costed env                   #
# --------------------------------------------------------------------------- #
def test_weight_policy_one_over_n_gross_matches_hand_value() -> None:
    """The 1/N WeightPolicy's per-step GROSS return == cross-sectional mean of returns.

    The env reports info['gross'] = w[:N] @ r_t with equal weights w_i = 1/N, so the
    realized gross is exactly the cross-sectional mean of the realized return vector — a
    hand-computable invariant. (port_ret = gross - cost, with cost from turnover.)
    """
    cfg = load_config("environment")
    panel = make_synthetic_panel(n_assets=4, n_days=90, seed=21)
    lookback = int(cfg["state"]["lookback_days"])
    projection = str(cfg["action"]["projection"])
    start = lookback

    policy = AC.WeightPolicy(
        equal_weight, lookback=lookback, n_assets=panel.N, projection=projection, cfg=cfg
    )

    # The action round-trips to uniform risky weights under the FROZEN projection.
    obs = np.concatenate([np.zeros(lookback * panel.N), np.ones(64)])
    action, state = policy.predict(obs)
    assert state is None
    w = project_simplex(action, projection)
    np.testing.assert_allclose(w[: panel.N], 1.0 / panel.N, atol=1e-9)
    assert abs(w[panel.N]) < 1e-9  # benchmark holds no cash

    # Roll one step through the real env and check the realized gross == equal-weight mean.
    env = PortfolioEnv(panel, cfg, lambda *a: (0.0, {}, None), start=start, end=panel.T)
    obs0, _ = env.reset()
    a0, _ = policy.predict(obs0)
    _obs, _r, _term, _trunc, info = env.step(np.asarray(a0).ravel())
    hand_gross = float(panel.returns[start, : panel.N].mean())
    assert info["gross"] == pytest.approx(hand_gross, abs=1e-12)
    assert info["port_ret"] == pytest.approx(info["gross"] - info["cost"], abs=1e-12)


def test_benchmark_floor_reports_all_benchmarks_and_costed_rollout() -> None:
    cfg = load_config("environment")
    panel = make_synthetic_panel(n_assets=6, n_days=160, seed=22)
    lookback = int(cfg["state"]["lookback_days"])
    test_window = (lookback, panel.T)

    out = AC.benchmark_floor(panel, cfg, test_window)
    bench = out["benchmarks"]
    # All eight frozen benchmarks (R19) present, each with the reported metrics.
    assert set(bench) == set(AC._BENCHMARK_NAMES)
    n_steps = test_window[1] - test_window[0]
    for name, m in bench.items():
        assert m["n_steps"] == n_steps
        for key in ("sharpe", "cvar", "max_drawdown", "dsr"):
            assert np.isfinite(m[key]), (name, key)
        assert 0.0 <= m["dsr"] <= 1.0
        assert m["max_drawdown"] >= 0.0
    assert out["gate"] is None  # no winner passed -> no gate


def test_benchmark_floor_gate_pass_and_fail() -> None:
    """The floor gate flags pass/fail: winner DSR vs the BEST benchmark DSR (no re-select)."""
    cfg = load_config("environment")
    panel = make_synthetic_panel(n_assets=6, n_days=160, seed=23)
    lookback = int(cfg["state"]["lookback_days"])
    test_window = (lookback, panel.T)
    n_steps = test_window[1] - test_window[0]
    rng = np.random.default_rng(24)

    strong = rng.standard_normal(n_steps) * 0.004 + 0.004   # high, smooth -> DSR ~ 1
    weak = rng.standard_normal(n_steps) * 0.02 - 0.004       # poor -> low DSR

    res_pass = AC.benchmark_floor(panel, cfg, test_window, winner_test_returns=strong)
    gate_pass = res_pass["gate"]
    assert gate_pass is not None
    assert gate_pass["best_benchmark"] in AC._BENCHMARK_NAMES
    assert gate_pass["passed"] is True
    assert gate_pass["winner_dsr"] > gate_pass["best_benchmark_dsr"]

    res_fail = AC.benchmark_floor(panel, cfg, test_window, winner_test_returns=weak)
    gate_fail = res_fail["gate"]
    assert gate_fail["passed"] is False
    assert gate_fail["winner_dsr"] <= gate_fail["best_benchmark_dsr"]


def test_benchmark_floor_l1_projection_round_trip() -> None:
    """Under the l1_normalize_of_clipped projection the 1/N action also yields 1/N weights."""
    cfg = load_config("environment")
    panel = make_synthetic_panel(n_assets=5, n_days=90, seed=25)
    lookback = int(cfg["state"]["lookback_days"])

    policy = AC.WeightPolicy(
        equal_weight,
        lookback=lookback,
        n_assets=panel.N,
        projection="l1_normalize_of_clipped",
        cfg=cfg,
    )
    obs = np.concatenate([np.zeros(lookback * panel.N), np.ones(40)])
    action, _ = policy.predict(obs)
    w = project_simplex(action, "l1_normalize_of_clipped")
    np.testing.assert_allclose(w[: panel.N], 1.0 / panel.N, atol=1e-9)
    assert abs(w[panel.N]) < 1e-9


# --------------------------------------------------------------------------- #
# R20 — additive risk-free robustness of the H2 Sharpe conjunction            #
# --------------------------------------------------------------------------- #
def test_h2_rf_robustness_quantifies_low_vol_shrinkage() -> None:
    """The rf penalty mean(rf)*sqrt(252)/sigma is larger for LOWER-vol arms, so threading the rf must
    SHRINK a lower-vol distributional arm's Sharpe edge (effect_shrinkage > 0) — and the frozen rf=0
    headline must be left untouched (rf=0 robustness call reproduces the rf=0 effect with no shrinkage)."""
    rng = np.random.default_rng(11)
    T = 400
    recs: list[dict] = []
    for s in range(8):
        recs.append(_test_record("distributional", s, rng.standard_normal(T) * 0.008 + 0.0006))  # low vol
        recs.append(_test_record("scalar", s, rng.standard_normal(T) * 0.011 + 0.0006))
        recs.append(_test_record("placebo", s, rng.standard_normal(T) * 0.011 + 0.0005))
        recs.append(_test_record("scalar_cvar5", s, rng.standard_normal(T) * 0.010 + 0.00055))

    rf = np.full(T, 2.0 / 100.0 / 252.0)  # ~2%/yr daily T-bill
    rob = AC.h2_sharpe_rf_robustness(recs, rf, rng=np.random.default_rng(1))
    assert set(rob) == {"survives", "contrasts", "rf_annualised_pct", "note"}
    assert rob["rf_annualised_pct"] == pytest.approx(2.0, abs=0.05)
    assert len(rob["contrasts"]) == 3
    ds = next(r for r in rob["contrasts"] if r["contrast"] == "distributional>scalar")
    assert ds["effect_shrinkage"] > 0.0  # rf reduces the LOWER-vol arm's edge (the R20 insight)

    # rf=0 must reproduce the frozen effect exactly (no shrinkage) — the headline convention is untouched.
    rob0 = AC.h2_sharpe_rf_robustness(recs, np.zeros(T), rng=np.random.default_rng(1))
    for r in rob0["contrasts"]:
        assert r["effect_shrinkage"] == pytest.approx(0.0, abs=1e-12)
        assert r["effect_rf0"] == pytest.approx(r["effect_excess"], abs=1e-12)


def test_analyze_produces_benchmark_floor_when_panel_supplied(tmp_path) -> None:
    """analyze() with a panel runs + renders the DeMiguel floor (previously invoked by NO entry point);
    records-only (no panel) preserves the prior behaviour. (critical-review #2/#6/#12)."""
    from src.io.results import write_run

    rng = np.random.default_rng(5)
    cfg = load_config("environment")
    panel = make_synthetic_panel(n_assets=6, n_days=160, seed=7)
    lookback = int(cfg["state"]["lookback_days"])
    test_window = (lookback, panel.T)
    t_len = test_window[1] - test_window[0]
    for arm in ("distributional", "scalar", "placebo", "scalar_cvar5"):
        for seed in range(3):
            write_run(_test_record(arm, seed, rng.standard_normal(t_len) * 0.01 + 0.0004), tmp_path)

    result = AC.analyze(tmp_path, n_blocks=4, panel=panel, cfg=cfg, test_window=test_window, winner_n_trials=40)
    floor = result.get("benchmark_floor")
    assert floor and "benchmarks" in floor
    assert set(floor["benchmarks"]) == set(AC._BENCHMARK_NAMES)        # every registered allocator rolled (the ladder grew 8 -> 9 when min_cvar was added;
        # the assertion is derived from _BENCHMARK_NAMES, so only this count was stale)
    assert floor.get("gate") and floor["gate"]["winner_n_trials"] == 40  # winner deflated by the budget (#17)
    assert floor["gate"]["winner_dsr_method"] == "median_per_seed"  # robust per-seed gate, NOT seed-mean (#2)
    md = AC.benchmark_floor_markdown(floor)
    assert "Benchmark floor" in md and "Winner-vs-floor gate" in md

    # R20 risk-free robustness is ALSO produced + rendered (additive; frozen rf=0 headline unchanged).
    # It is GATED on the public FRED risk-free series being on disk: `analyze` degrades silently when
    # it is absent (`load_risk_free_daily(...).available is False`), and a bare clone legitimately has
    # no `data/raw/`. Asserting unconditionally made this a data-gated test that could not skip, so
    # the precondition is now checked rather than assumed.
    rob = result.get("h2_rf_robustness")
    if rob is None:
        pytest.skip("risk-free reference series not on disk; the R20 robustness panel is data-gated")
    assert "contrasts" in rob and len(rob["contrasts"]) == 3
    for r in rob["contrasts"]:
        assert "effect_rf0" in r and "effect_excess" in r and "effect_shrinkage" in r
    assert "R20" in AC.h2_rf_robustness_markdown(rob)

    # records-only path (no panel) must NOT produce a floor or robustness (default behaviour preserved)
    records_only = AC.analyze(tmp_path, n_blocks=4)
    assert "benchmark_floor" not in records_only and "h2_rf_robustness" not in records_only


def test_every_registered_benchmark_has_a_callable_implementation() -> None:
    """A registered benchmark NAME with no implementation crashes the whole ladder.

    `benchmark_floor` builds its strategy map with
    ``{name: getattr(_strat, name) for name in _BENCHMARK_NAMES}``. A registered name with no
    attribute raises `AttributeError` and takes the ENTIRE ladder down — including the T0 floor the
    replication-leg filter is pinned to (R84).

    Failing loudly there is the RIGHT behaviour, not a bug: it is the "registered NAME with no
    registered VALUE" class that `pretrain_validate.check_registered_nodes_computable` exists to catch
    for validity-tier nodes. Nothing did it for benchmarks, so the discovery point was analysis time
    rather than pre-launch. A later review verified `_BENCHMARK_NAMES` matches `config: benchmarks`, so the
    remaining unguarded hop is NAME -> IMPLEMENTATION: adding a benchmark to both the config and the
    tuple (satisfying that guard) while forgetting the function would pass every existing check.

    Currently correct — all nine resolve — so this closes the guard, not a live defect (#105).
    """
    from src.baselines import strategies as _strat

    missing = [n for n in AC._BENCHMARK_NAMES if not hasattr(_strat, n)]
    assert not missing, (
        f"registered benchmark(s) with no implementation in src/baselines/strategies.py: {missing} — "
        "benchmark_floor's getattr would raise and take the whole ladder, and the T0 floor, with it"
    )
    not_callable = [n for n in AC._BENCHMARK_NAMES if not callable(getattr(_strat, n))]
    assert not not_callable, f"registered benchmark(s) present but not callable: {not_callable}"
