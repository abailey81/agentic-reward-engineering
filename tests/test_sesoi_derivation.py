"""R104: bind the DERIVED SESOI economic band into the test suite so the linchpin is VERIFIED, not asserted.

The SESOI VALUE (0.05 validation-DSR) is unchanged; these tests prove its *registered justification*
(`inference.sesoi_derivation`) is internally self-consistent, cross-consistent with the environment's
transaction cost, and that 0.05 DSR (= 0.0756 ann-Sharpe) sits inside the defensible band
[cost-breakeven, practitioner-material].
"""
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _prereg():
    return yaml.safe_load((ROOT / "config" / "preregistration.yaml").read_text(encoding="utf-8"))


def _env():
    return yaml.safe_load((ROOT / "config" / "environment.yaml").read_text(encoding="utf-8"))


def test_sesoi_value_unchanged():
    inf = _prereg()["inference"]
    assert inf["sesoi"] == 0.05
    assert inf["equivalence_margin"] == 0.05


def test_sesoi_derivation_block_present_and_shaped():
    d = _prereg()["inference"]["sesoi_derivation"]
    for k in (
        "dsr_per_ann_sharpe", "sesoi_ann_sharpe_equiv", "book_sigma_ann",
        "book_turnover_ann_oneway", "cost_bps_oneway", "lower_bound_cost_breakeven_sharpe",
        "upper_bound_practitioner_sharpe", "verdict", "source",
    ):
        assert k in d, f"sesoi_derivation missing {k!r}"
    assert (ROOT / d["source"]).exists(), f"derivation source doc missing: {d['source']}"


def test_dsr_to_sharpe_map_consistent():
    # sesoi_ann_sharpe_equiv == sesoi_DSR / k   (k = DSR per ann-Sharpe, the conservative delta-method ceiling)
    inf = _prereg()["inference"]
    d = inf["sesoi_derivation"]
    implied = inf["sesoi"] / d["dsr_per_ann_sharpe"]
    assert math.isclose(implied, d["sesoi_ann_sharpe_equiv"], rel_tol=0.02), (implied, d["sesoi_ann_sharpe_equiv"])


def test_cost_breakeven_floor_recomputed_from_inputs():
    # floor = cost(one-way, fractional) * annual turnover / annual sigma  (Sharpe-edge equal to the cost drag)
    d = _prereg()["inference"]["sesoi_derivation"]
    implied = d["cost_bps_oneway"] * 1e-4 * d["book_turnover_ann_oneway"] / d["book_sigma_ann"]
    assert math.isclose(implied, d["lower_bound_cost_breakeven_sharpe"], rel_tol=0.05), (
        implied, d["lower_bound_cost_breakeven_sharpe"])


def test_sesoi_inside_defensible_band():
    d = _prereg()["inference"]["sesoi_derivation"]
    lo = d["lower_bound_cost_breakeven_sharpe"]
    hi = d["upper_bound_practitioner_sharpe"]
    s = d["sesoi_ann_sharpe_equiv"]
    assert lo < s < hi, f"SESOI {s} not inside the derived band [{lo}, {hi}]"
    assert d["verdict"] == "sesoi_inside_band"
    assert s / lo > 5.0, "SESOI should be comfortably (>5x) above the cost-breakeven floor (exploitable)"
    assert s < hi, "SESOI must be stricter than practitioner-material (conservative equivalence)"


def test_cost_bps_matches_environment_headline():
    # cross-file non-fragility: the SESOI floor's cost input must equal the env's headline transaction cost
    d = _prereg()["inference"]["sesoi_derivation"]
    env = _env()
    assert d["cost_bps_oneway"] == env["costs"]["headline_bps"], (
        d["cost_bps_oneway"], env["costs"]["headline_bps"])
