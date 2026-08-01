"""Deterministic coverage tests for the platform layer.

REAL unit tests targeting previously-uncovered guard/branch/edge paths in
``src/data``, ``src/utils``, ``src/env``, ``src/feedback`` and ``src/reward``.
Every assertion is exact (closed-form or byte-identical); synthetic + pure-logic
only — any path that would read the licensed gold panel is guarded or avoided.

Local fixtures only (nothing added to conftest). Run order-independent:
    .venv/Scripts/python.exe -m pytest tests/test_platform_coverage.py -q -p no:randomly
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.data.panel import Panel
from src.data.synthetic import make_synthetic_panel


# --------------------------------------------------------------------------- #
# Local fixtures (NOT conftest)
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_panel() -> Panel:
    """A tiny deterministic synthetic panel (contemporaneous-VIX convention)."""
    return make_synthetic_panel(n_assets=4, n_days=200, seed=7)


@pytest.fixture
def env_cfg() -> dict:
    """A minimal, fully-literal environment config (no YAML / no gold)."""
    return {
        "state": {
            "lookback_days": 10,
            "realized_vol_windows": [5, 10],
            "include_vix": True,
            "include_prev_weights": True,
            "cash_daily_rate": 0.0,
        },
        "action": {"projection": "softmax", "bound": 10.0},
        "costs": {"headline_bps": 10.0},
    }


def _const_reward(weights, returns, prev_weights, port_ret, info):
    """Contract-conforming reward returning port_ret with a step counter state."""
    state = info.get("reward_state")
    count = 0 if state is None else int(state) + 1
    return float(port_ret), {"port_ret": float(port_ret)}, count


# =========================================================================== #
# src/utils/logging.py
# =========================================================================== #
def test_jsonable_coerces_numpy_scalar_array_and_other():
    from src.utils.logging import _jsonable

    assert _jsonable(np.int64(5)) == 5
    assert _jsonable(np.float64(1.5)) == 1.5
    assert _jsonable(np.array([1, 2, 3])) == [1, 2, 3]
    # Non-numpy, non-jsonable object falls through to str().
    obj = object()
    assert _jsonable(obj) == str(obj)


def test_jsonl_formatter_emits_one_json_object_with_fields_and_exc():
    from src.utils.logging import _JsonlFormatter

    fmt = _JsonlFormatter()
    rec = logging.LogRecord("L", logging.INFO, __file__, 1, "hello", None, None)
    rec.event = "unit_event"
    rec.fields = {"a": 1, "b": np.float64(2.0)}
    line = fmt.format(rec)
    obj = json.loads(line)
    assert obj["event"] == "unit_event"
    assert obj["level"] == "INFO"
    assert obj["msg"] == "hello"
    assert obj["a"] == 1 and obj["b"] == 2.0
    assert "exc" not in obj

    # With exc_info the formatter adds an "exc" key.
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec2 = logging.LogRecord(
            "L", logging.ERROR, __file__, 2, "fail", None, sys.exc_info()
        )
        obj2 = json.loads(fmt.format(rec2))
    assert "boom" in obj2["exc"]
    # No `fields` attr → no spurious keys beyond the base record.
    assert set(obj2) >= {"ts", "level", "logger", "event", "msg", "exc"}


def test_configure_logging_is_idempotent_and_resets_level():
    import src.utils.logging as L

    L.configure_logging(logging.WARNING)
    assert L._configured is True
    root = logging.getLogger()
    n_handlers = len(root.handlers)
    # Second call: idempotent (no new handler) but level is updated.
    L.configure_logging(logging.DEBUG)
    assert len(root.handlers) == n_handlers
    assert root.level == logging.DEBUG


def test_attach_run_logging_writes_files_and_is_idempotent(tmp_path):
    import src.utils.logging as L

    run_dir = tmp_path / "run01"
    paths = L.attach_run_logging(run_dir)
    assert paths["log"].name == "run.log"
    assert paths["events"].name == "events.jsonl"
    assert run_dir.is_dir()
    root = logging.getLogger()
    after_first = len(root.handlers)
    # Idempotent for the same dir: no extra file handlers.
    paths2 = L.attach_run_logging(run_dir)
    assert paths2 == paths
    assert len(root.handlers) == after_first

    # log_event writes a parseable JSONL line carrying the structured fields.
    log = L.get_logger("cov.test")
    L.log_event(log, "candidate_done", level=logging.INFO, arm="scalar", fitness=0.0125)
    for h in root.handlers:
        h.flush()
    events = paths["events"].read_text(encoding="utf-8").strip().splitlines()
    matching = [json.loads(x) for x in events if json.loads(x).get("event") == "candidate_done"]
    assert matching, "the structured event was not written to events.jsonl"
    rec = matching[-1]
    assert rec["arm"] == "scalar"
    assert rec["fitness"] == pytest.approx(0.0125)

    # Detach the per-run handlers so other tests' root logger is unaffected.
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()
    L._run_attached.discard(str(run_dir.resolve()))


def test_short_formats_floats_and_truncates_long_strings():
    from src.utils.logging import _short

    assert _short(0.123456789) == "0.123457"
    assert _short(42) == "42"
    long = "x" * 200
    out = _short(long)
    assert out.endswith("...") and len(out) == 80


# =========================================================================== #
# src/utils/config.py
# =========================================================================== #
def test_dotdict_attribute_access_nested_and_missing():
    from src.utils.config import DotDict

    d = DotDict({"splits": {"train": {"start": "2005"}}})
    assert d.splits.train.start == "2005"
    with pytest.raises(AttributeError):
        _ = d.nope


def test_dotdict_require_missing_and_null_and_present():
    from src.utils.config import DotDict

    d = DotDict({"present": 1, "nested": {"k": 2}, "nullable": None})
    assert d.require("present") == 1
    assert d.require("nested").k == 2  # nested dict wrapped
    with pytest.raises(KeyError, match="required config key 'absent'"):
        d.require("absent")
    with pytest.raises(ValueError, match="is null"):
        d.require("nullable")


def test_cfg_get_dict_object_and_none():
    from src.utils.config import DotDict, cfg_get

    assert cfg_get(None, "k", "default") == "default"
    assert cfg_get({"k": 5}, "k") == 5
    assert cfg_get({"k": 5}, "missing", -1) == -1
    assert cfg_get(DotDict({"k": 9}), "k") == 9

    class _Obj:
        attr = 3

    assert cfg_get(_Obj(), "attr") == 3
    assert cfg_get(_Obj(), "nope", "fallback") == "fallback"


def test_load_config_unknown_name_raises_filenotfound():
    from src.utils.config import load_config

    with pytest.raises(FileNotFoundError, match="no config"):
        load_config("definitely_not_a_config_file_xyz")


# =========================================================================== #
# src/utils/seeding.py
# =========================================================================== #
def test_set_global_seed_seeds_numpy_reproducibly_and_returns_seed():
    from src.utils.seeding import set_global_seed

    assert set_global_seed(123) == 123
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_rejects_non_int():
    from src.utils.seeding import set_global_seed

    with pytest.raises(TypeError, match="seed must be int"):
        set_global_seed(1.5)


def test_set_global_seed_deterministic_torch_flag_is_safe():
    # deterministic_torch=True must not raise even when torch is absent.
    from src.utils.seeding import set_global_seed

    assert set_global_seed(7, deterministic_torch=True) == 7


def test_rng_is_isolated_and_reproducible():
    from src.utils.seeding import rng

    g1 = rng(99)
    g2 = rng(99)
    np.testing.assert_array_equal(g1.standard_normal(10), g2.standard_normal(10))


# =========================================================================== #
# src/utils/provenance.py
# =========================================================================== #
def test_provenance_hashes_are_stable_and_distinct():
    from src.utils import provenance as P

    assert P.sha256_text("abc") == P.sha256_bytes(b"abc")
    # canonical-JSON object hash is key-order-independent
    assert P.sha256_obj({"a": 1, "b": 2}) == P.sha256_obj({"b": 2, "a": 1})
    assert P.sha256_obj({"a": 1}) != P.sha256_obj({"a": 2})


def test_sha256_file_matches_bytes(tmp_path):
    from src.utils import provenance as P

    f = tmp_path / "blob.bin"
    data = b"deterministic-content-1234"
    f.write_bytes(data)
    assert P.sha256_file(f) == P.sha256_bytes(data)


def test_env_fingerprint_has_expected_keys():
    from src.utils import provenance as P

    fp = P.env_fingerprint()
    assert {"python", "platform", "git_commit", "git_dirty", "packages"} <= set(fp)
    assert "numpy" in fp["packages"]


# =========================================================================== #
# src/reward/contract.py
# =========================================================================== #
def test_validate_signature_accepts_conforming_and_rejects_variants():
    from src.reward.contract import validate_signature

    def good(weights, returns, prev_weights, port_ret, info):
        return 0.0, {}, None

    assert validate_signature(good) is True
    # Not callable.
    assert validate_signature(42) is False
    # Wrong names.
    assert validate_signature(lambda a, b, c, d, e: None) is False

    # *args / **kwargs forms rejected.
    def varargs(*args):
        return 0.0, {}, None

    assert validate_signature(varargs) is False

    # keyword-only param rejected.
    def kwonly(weights, returns, prev_weights, port_ret, *, info):
        return 0.0, {}, None

    assert validate_signature(kwonly) is False


def test_validate_signature_handles_uninspectable_callable():
    from src.reward.contract import validate_signature

    # A builtin with no Python signature → inspect.signature raises → False.
    assert validate_signature(len) is False


# =========================================================================== #
# src/feedback/schema.py
# =========================================================================== #
_TAIL = {
    "cvar_01": -0.067,
    "cvar_05": -0.041,
    "cvar_10": -0.029,
    "cvar_25": -0.016,
    "left_tail_mass": 0.061,
    "robust_skew": -0.38,
}


def test_the_fed_scalar_resolves_differences_well_below_the_SESOI():
    """#87: the scalar arm's ENTIRE signal is this one number, so its RESOLUTION is the treatment.

    At the previous ``.2f`` the median archived fitness (0.000914) rendered as "0.00"; 328 of 591
    real rendered headers were that identical string, and only 52.8 % of genuinely-different
    candidate pairs were distinguishable. That makes the primary H2 comparator degenerate by
    construction and SQ1 responsiveness unmeasurable for it. Candidates differing by a SESOI — or a
    tenth of one — must render DIFFERENTLY.
    """
    from src.feedback.schema import build_block

    sesoi = 0.05
    base = 0.000914  # the measured median archived fitness
    assert build_block("scalar", base, None) != build_block("scalar", base + sesoi, None)
    assert build_block("scalar", base, None) != build_block("scalar", base + sesoi / 10, None)
    # and it must resolve the MODE of the distribution, where .2f collapsed everything to "0.00"
    assert build_block("scalar", 0.0009, None) != build_block("scalar", 0.0011, None)


def test_the_fed_scalar_never_renders_in_scientific_notation():
    """#87: fixed-point is load-bearing, because two parsers read this number back out.

    ``scripts/analyze_campaign.py::_FED_SCALAR_RE`` and ``src/inference/information_gap.py::
    _SCALAR_RE`` both match ``\\d+(?:\\.\\d+)?``, which does not accept an exponent. A ``.3g``-style
    format scores similarly on discrimination but emits ``1.11e-05`` for small values, and both
    regexes would silently capture the MANTISSA ALONE — corrupting the responsiveness analysis
    instead of failing loudly.
    """
    import re

    from src.feedback.schema import build_block

    parser = re.compile(r"Your previous reward scored:\s*([+-]?\d+(?:\.\d+)?)\s")
    for v in (0.0, 1e-7, 1.11e-05, 0.000914, 0.4988, 1.0):
        block = build_block("scalar", v, None)
        m = parser.search(block)
        assert m is not None, f"{v!r} rendered unparseably: {block!r}"
        assert float(m.group(1)) == pytest.approx(v, abs=5e-7), (
            f"{v!r} did not round-trip through the parser: {block!r}")


def test_build_block_scalar_and_distributional():
    from src.feedback.schema import build_block

    s = build_block("scalar", 0.83, None)
    assert s == "Your previous reward scored: 0.830000 (validation Deflated Sharpe)."

    d = build_block("distributional", 0.83, _TAIL)
    lines = d.splitlines()
    assert len(lines) == 8  # header + intro + 6 fields
    assert "CVaR 5%: -0.041" in d
    assert "(high-variance estimate)" in d  # appended to the CVaR-1% line


def test_build_block_scalar_cvar5_and_placebo():
    from src.feedback.schema import build_block

    c5 = build_block("scalar_cvar5", 0.1, _TAIL).splitlines()
    assert len(c5) == 2
    assert "CVaR 5%: -0.041" in c5[1]

    p = build_block("placebo", 0.1, None).splitlines()
    assert len(p) == 8  # matched line count to distributional
    assert "reference value 1: +0.000" in p[2]


def test_build_block_placebo_shuffled_is_derangement_and_replayable():
    from src.feedback.schema import build_block, shuffle_seed_from_id

    seed = shuffle_seed_from_id("cand-007")
    a = build_block("placebo_shuffled", 0.1, _TAIL, shuffle_seed=seed)
    b = build_block("placebo_shuffled", 0.1, _TAIL, shuffle_seed=seed)
    assert a == b  # replayable for the same seed
    # Same MARGINAL set of values as distributional, structure identical.
    real_vals = sorted(round(v, 3) for v in _TAIL.values())

    def _vals(block: str) -> list[float]:
        out = []
        for ln in block.splitlines()[2:]:
            tok = ln.split(":")[1].split("(")[0].strip()
            out.append(round(float(tok), 3))
        return sorted(out)

    assert _vals(a) == real_vals
    # Derangement: at least one label's value differs from the canonical-order value.
    dist = build_block("distributional", 0.1, _TAIL)
    assert a.splitlines()[2:] != dist.splitlines()[2:]


def test_build_block_error_paths():
    from src.feedback.schema import build_block

    with pytest.raises(ValueError, match="unknown arm"):
        build_block("nonsense", 0.1, _TAIL)
    with pytest.raises(ValueError, match="requires tail_stats"):
        build_block("distributional", 0.1, None)
    with pytest.raises(ValueError, match="requires tail_stats"):
        build_block("scalar_cvar5", 0.1, None)
    with pytest.raises(ValueError, match="requires tail_stats"):
        build_block("placebo_shuffled", 0.1, None)
    with pytest.raises(ValueError, match="shuffle_seed"):
        build_block("placebo_shuffled", 0.1, _TAIL)


def test_block_fields_per_arm_and_unknown():
    from src.feedback.schema import block_fields

    assert block_fields("scalar") == ["scalar_metric"]
    assert block_fields("scalar_cvar5") == ["scalar_metric", "cvar_05"]
    assert len(block_fields("distributional")) == 7
    assert len(block_fields("placebo")) == 7
    assert block_fields("placebo_shuffled") == block_fields("distributional")
    with pytest.raises(ValueError, match="unknown arm"):
        block_fields("???")


# =========================================================================== #
# src/feedback/measurement.py
# =========================================================================== #
def test_return_distribution_empty_input_raises():
    from src.feedback.measurement import ReturnDistribution

    with pytest.raises(ValueError, match="non-empty and finite"):
        ReturnDistribution().fit(np.array([np.nan, np.inf]))


def test_return_distribution_query_before_fit_raises():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution()
    with pytest.raises(RuntimeError, match="fit must be called"):
        rd.quantiles([0.5])
    with pytest.raises(RuntimeError, match="fit must be called"):
        rd._bootstrap_cvars(0.05, 10, None, 0)


def test_cvar_bad_alpha_and_unknown_method():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution().fit(np.linspace(-0.05, 0.05, 500))
    with pytest.raises(ValueError, match="alpha must be in"):
        rd.cvar(0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        rd.cvar(1.0)
    with pytest.raises(ValueError, match="unknown method"):
        rd.cvar(0.05, method="bogus")


def test_cvar_methods_agree_in_sign_and_monotone():
    from src.feedback.measurement import ReturnDistribution

    rng = np.random.default_rng(0)
    rd = ReturnDistribution().fit(rng.standard_normal(5000) * 0.01)
    emp = rd.cvar(0.05, method="empirical")
    evt = rd.cvar(0.05, method="evt")
    assert emp < 0 and evt < 0
    # Deeper tail is at least as severe.
    assert rd.cvar(0.01) <= rd.cvar(0.05) + 1e-9
    ts = rd.tail_stats()
    assert set(ts) == {"cvar_01", "cvar_05", "cvar_10", "cvar_25",
                       "left_tail_mass", "robust_skew"}


def test_evt_falls_back_degenerate_and_shallow_alpha():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution().fit(np.full(50, 0.001))  # all-equal → degenerate tail fit
    # Degenerate or shallow-alpha → empirical fallback (no crash, finite value).
    assert np.isfinite(rd.cvar(0.05, method="evt"))
    # A reason string is returned (not None) for a clearly shallow alpha vs exceed_frac.
    rd2 = ReturnDistribution(threshold_q=0.10).fit(np.random.default_rng(1).standard_normal(800) * 0.01)
    assert rd2._evt_falls_back(0.5) == "alpha_gt_exceed_frac"


def test_reliability_tiers_and_exceedance_count():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution().fit(np.random.default_rng(2).standard_normal(800) * 0.01)
    assert rd.exceedance_count(0.05) == int(np.ceil(0.05 * 800))
    assert rd.reliability(0.05) == "high"        # ceil(0.05*800)=40 > 30
    assert rd.reliability(0.01) == "medium"      # ceil(0.01*800)=8 in [7,30]
    small = ReturnDistribution().fit(np.random.default_rng(3).standard_normal(100) * 0.01)
    assert small.reliability(0.01) == "low"      # ceil(0.01*100)=1 < 7


def test_fed_estimator_log_records_and_resets():
    from src.feedback import measurement as M

    M.reset_fed_estimator_log()
    rd = M.ReturnDistribution().fit(np.random.default_rng(4).standard_normal(800) * 0.01)
    rd.cvar(0.05)  # FED headline level → records a path
    log = M.fed_estimator_log()
    assert 0.05 in log and len(log[0.05]) == 1
    M.reset_fed_estimator_log()
    assert M.fed_estimator_log() == {}


def test_cvar_ci_and_bias_are_deterministic():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution().fit(np.random.default_rng(5).standard_normal(600) * 0.01)
    lo1, hi1 = rd.cvar_ci(0.05, n_boot=50, seed=11)
    lo2, hi2 = rd.cvar_ci(0.05, n_boot=50, seed=11)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1
    b1 = rd.cvar_bias(0.05, n_boot=50, seed=11)
    b2 = rd.cvar_bias(0.05, n_boot=50, seed=11)
    assert b1 == b2 and np.isfinite(b1)


def test_cvar_uncertainty_report_and_threshold_sensitivity():
    from src.feedback.measurement import ReturnDistribution

    rd = ReturnDistribution().fit(np.random.default_rng(6).standard_normal(600) * 0.01)
    rep = rd.cvar_uncertainty_report(n_boot=40, seed=1)
    assert "cvar_05" in rep
    entry = rep["cvar_05"]
    assert {"point", "ci_lo", "ci_hi", "bias", "n_exceedances", "reliability"} <= set(entry)
    assert entry["ci_lo"] <= entry["ci_hi"]

    ts = rd.threshold_sensitivity(alpha=0.05, threshold_qs=(0.05, 0.10, 0.20))
    assert "spread" in ts and "cv" in ts
    assert ts["spread"] >= 0.0


# =========================================================================== #
# src/env/portfolio_env.py
# =========================================================================== #
def test_project_simplex_softmax_l1_and_unknown():
    from src.env.portfolio_env import project_simplex

    w = project_simplex(np.array([1.0, 2.0, 3.0]), "softmax")
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0)

    w2 = project_simplex(np.array([-1.0, 0.0, 4.0]), "l1_normalize_of_clipped")
    assert w2[0] == 0.0 and w2.sum() == pytest.approx(1.0)

    # All-zero clipped → uniform fallback.
    w3 = project_simplex(np.array([-1.0, -2.0, -3.0]), "l1_normalize_of_clipped")
    np.testing.assert_allclose(w3, np.full(3, 1.0 / 3.0))

    with pytest.raises(ValueError, match="unknown simplex projection"):
        project_simplex(np.zeros(3), "bogus")


def test_env_construction_guard_raises(small_panel, env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    # start < lookback
    with pytest.raises(ValueError, match="must be >= lookback"):
        PortfolioEnv(small_panel, env_cfg, _const_reward, start=2)
    # end > T
    with pytest.raises(ValueError, match="must be <= panel.T"):
        PortfolioEnv(small_panel, env_cfg, _const_reward, end=small_panel.T + 5)
    # start >= end
    with pytest.raises(ValueError, match="must be <"):
        PortfolioEnv(small_panel, env_cfg, _const_reward, start=50, end=50)
    # vol window > lookback
    bad = {**env_cfg, "state": {**env_cfg["state"], "realized_vol_windows": [50]}}
    with pytest.raises(ValueError, match="must be <= lookback"):
        PortfolioEnv(small_panel, bad, _const_reward)


def test_env_cost_bps_override(small_panel, env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    base = PortfolioEnv(small_panel, env_cfg, _const_reward)
    assert base.cost == pytest.approx(10.0 * 1e-4)
    assert base.cost_bps is None
    swept = PortfolioEnv(small_panel, env_cfg, _const_reward, cost_bps=25.0)
    assert swept.cost == pytest.approx(25.0 * 1e-4)
    assert swept.cost_bps == 25.0


def test_env_cash_daily_rate_nonzero_changes_gross(small_panel, env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    cfg_cash = {**env_cfg, "state": {**env_cfg["state"], "cash_daily_rate": 0.001}}
    env = PortfolioEnv(small_panel, cfg_cash, _const_reward)
    assert env.cash_daily_rate == 0.001
    env.reset(seed=0)
    # An all-into-cash action: gross should equal the cash sleeve's money-market return.
    action = np.array([-50.0, -50.0, -50.0, -50.0, 50.0], dtype=float)  # N=4 + cash
    _obs, _rew, _term, _trunc, info = env.step(action)
    # w_cash ~ 1, so gross ~ cash_daily_rate.
    assert info["gross"] == pytest.approx(0.001, abs=2e-4)


def test_env_no_vix_no_prev_weights_obs_dim(small_panel, env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    cfg = {
        **env_cfg,
        "state": {**env_cfg["state"], "include_vix": False, "include_prev_weights": False},
    }
    env = PortfolioEnv(small_panel, cfg, _const_reward)
    n = small_panel.N
    lb = 10
    expected = lb * n + 2 * n + 1  # lookback + 2 vol windows + cash marker, no vix/no prev_w
    assert env._obs_dim() == expected
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == expected


def test_env_port_growth_wipeout_raises(env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    # The `port_growth <= 0` guard fires when the book that actually HOLDS through the step is wiped by
    # a combined -100% move. Since #92 it keys off the POST-trade book (`w @ growth`, identically
    # `1 + gross`) rather than the prior one — the prior book is not the one earning r_t.
    #
    # That makes the reachability condition precise: the guard can only fire when the projection admits
    # a ZERO-CASH allocation, because the cash sleeve grows at 1.0 and contributes `w_cash > 0` to the
    # sum. Under `softmax` a wipeout is therefore mathematically IMPOSSIBLE (every weight is strictly
    # positive), so this uses the other frozen projection, `l1_normalize_of_clipped` (audit C-8), which
    # clips at zero and can put the whole book in risky assets.
    T, N = 40, 2
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((T, N)) * 0.005
    vix = np.full(T, 0.2)
    dates = np.arange("2005-01-03", T, dtype="datetime64[D]")
    panel = Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(N))
    cfg = {**env_cfg,
           "state": {**env_cfg["state"], "realized_vol_windows": [5]},
           "action": {**env_cfg.get("action", {}), "projection": "l1_normalize_of_clipped"}}
    env = PortfolioEnv(panel, cfg, _const_reward)
    env.reset(seed=0)
    t0 = env.t
    panel.returns[t0, :] = -1.0  # both risky assets -100% on this step
    # Clipped+L1-normalised -> [0.5, 0.5, 0.0]: an all-risky, zero-cash book on a -100% day.
    with pytest.raises(FloatingPointError, match="non-positive portfolio growth"):
        env.step(np.array([1.0, 1.0, -1.0], dtype=float))


def test_env_T_property(small_panel, env_cfg):
    from src.env.portfolio_env import PortfolioEnv

    env = PortfolioEnv(small_panel, env_cfg, _const_reward)
    assert env.T == small_panel.T


# =========================================================================== #
# src/data/loaders.py  (pure-logic paths; no gold reads)
# =========================================================================== #
def test_seed_leading_vix_interior_ffill_and_leading_seed():
    from src.data.loaders import _seed_leading_vix

    # Full cash frame with a known pre-`start` value.
    idx = pd.date_range("2005-01-01", periods=6, freq="D")
    cash = pd.DataFrame({"vix": [0.20, 0.21, np.nan, np.nan, np.nan, np.nan]}, index=idx)
    start = idx[2]
    window = cash.loc[start:, "vix"].copy()  # leading NaN at `start`
    out = _seed_leading_vix(window, cash, start)
    # Leading gap seeded from the last PAST close (0.21 at idx[1]), not bfilled from future.
    assert out.iloc[0] == pytest.approx(0.21)
    assert out.notna().all()


def test_seed_leading_vix_global_first_falls_back_to_bfill():
    from src.data.loaders import _seed_leading_vix

    idx = pd.date_range("2005-01-01", periods=4, freq="D")
    cash = pd.DataFrame({"vix": [np.nan, np.nan, 0.30, 0.31]}, index=idx)
    start = idx[0]  # global-first: no prior observation anywhere
    window = cash.loc[start:, "vix"].copy()
    out = _seed_leading_vix(window, cash, start)
    # No past data → last-resort bfill from the first later valid (0.30).
    assert out.iloc[0] == pytest.approx(0.30)


def test_expected_sha256_returns_none_without_manifest(tmp_path):
    from src.data.loaders import _expected_sha256

    missing_manifest = tmp_path / "no_manifest.jsonl"
    assert _expected_sha256(tmp_path / "x.parquet", missing_manifest) is None


def test_expected_sha256_basename_and_relpath_match(tmp_path):
    from src.data.loaders import _expected_sha256

    target = tmp_path / "returns_panel_test.parquet"
    target.write_bytes(b"x")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join([
            "   ",  # blank line tolerated
            json.dumps({"name": "other.parquet", "sha256": "aa"}),
            json.dumps({"relpath": "x/returns_panel_test.parquet", "sha256": "bb"}),
        ]),
        encoding="utf-8",
    )
    # basename match (relpath doesn't resolve to this tmp tree) → "bb".
    assert _expected_sha256(target, manifest) == "bb"


def test_verify_checksum_raises_without_manifest_entry(tmp_path):
    """C2: verification requested but no manifest entry -> FAIL LOUD (was a silent skip)."""
    import pytest

    from src.data import loaders as Ld

    f = tmp_path / "returns_panel_xyz.parquet"
    f.write_bytes(b"data")
    # Point the module manifest at a nonexistent file → the requested verification cannot be proven,
    # so it now RAISES rather than silently skipping (silent-skip on the headline panel is the bug C2 fixes).
    empty_manifest = tmp_path / "none.jsonl"
    monkey = Ld._MANIFEST
    try:
        Ld._MANIFEST = empty_manifest
        with pytest.raises(ValueError, match="no manifest entry"):
            Ld._verify_checksum(f)
    finally:
        Ld._MANIFEST = monkey


def test_gold_suffix_respects_env(monkeypatch):
    from src.data import loaders as Ld

    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert Ld.gold_suffix() == "univ5"  # ACTIVE Split-C panel (ADR-044/051; config gold.suffix governs)
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "_univ4")
    assert Ld.gold_suffix() == "univ4"  # leading underscore stripped


def test_embargoed_val_start_synthetic_fallback():
    from src.data.loaders import embargoed_val_start

    # No split table at this gold_dir → fallback path (purge = max(embargo, lookback)).
    dates = np.arange("2005-01-03", 400, dtype="datetime64[D]").astype("datetime64[ns]")
    train_end = dates[100]
    # embargo-only (lookback=0): val starts strictly after the abutting train index.
    idx0 = embargoed_val_start(dates, train_end, embargo_days=21, lookback=0, gold_dir="/nonexistent")
    train_idx = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_end))))
    assert idx0 > train_idx
    # lookback dominates embargo: a larger lookback pushes val later.
    idx_lb = embargoed_val_start(dates, train_end, embargo_days=21, lookback=60, gold_dir="/nonexistent")
    assert idx_lb > idx0
    # Clamped to len(dates).
    assert idx_lb <= len(dates)


def test_materialized_val_post_embargo_missing_table():
    from src.data.loaders import _materialized_val_post_embargo
    from pathlib import Path

    boundary, embargo = _materialized_val_post_embargo("development", Path("/nonexistent_dir"))
    assert boundary is None and embargo is None


# =========================================================================== #
# src/data/pipeline.py  (synthetic; deterministic)
# =========================================================================== #
def test_gold_pipeline_runs_and_is_deterministic():
    from src.data.pipeline import GoldPipeline, DataPipeline

    cfg = {
        "splits": {
            "train": {"start": "2005-01-03", "end": "2005-06-01"},
            "val": {"start": "2005-06-02", "end": "2005-09-01"},
            "test": {"start": "2005-09-02", "end": "2006-01-01"},
        },
        "embargo_days": 5,
    }
    p1 = GoldPipeline(cfg, n_assets=12, top_n=4, n_days=400, seed=42, lookback_days=10)
    panel, manifest = p1.run()
    assert panel.N == 4
    assert "splits" in manifest and {"train", "val", "test"} <= set(manifest["splits"])
    assert manifest["embargo_days"] == max(5, 10)  # purge = max(embargo, lookback)
    # 13 stage checksums present.
    assert sum(1 for k in manifest if k[:2].isdigit()) == 13

    p2 = GoldPipeline(cfg, n_assets=12, top_n=4, n_days=400, seed=42, lookback_days=10)
    _panel2, manifest2 = p2.run()
    assert manifest["13_freeze_panel"] == manifest2["13_freeze_panel"]
    assert DataPipeline is GoldPipeline  # back-compat alias


def test_gold_pipeline_proportional_fallback_when_dates_out_of_range():
    from src.data.pipeline import GoldPipeline

    # Config split dates far outside the synthetic 2005-start calendar → proportional fallback.
    cfg = {
        "splits": {
            "train": {"start": "1990-01-01", "end": "1990-06-01"},
            "val": {"start": "1990-06-02", "end": "1990-09-01"},
            "test": {"start": "1990-09-02", "end": "1991-01-01"},
        },
        "embargo_days": 3,
    }
    p = GoldPipeline(cfg, n_assets=12, top_n=4, n_days=300, seed=1, lookback_days=0)
    panel, manifest = p.run()
    splits = manifest["splits"]
    t = panel.T
    # Proportional 60/80 split applied (then purge=max(3,0)=3 trims later starts).
    assert splits["train"]["start"] == 0
    assert splits["train"]["end"] == int(t * 0.6)
    assert splits["test"]["end"] == t


def test_default_lookback_resolves_from_config():
    from src.data.pipeline import GoldPipeline

    lb = GoldPipeline._default_lookback()
    assert isinstance(lb, int) and lb >= 0


# =========================================================================== #
# src/data/market_reference.py  (file-absent fallbacks; no gold)
# =========================================================================== #
def test_market_reference_loaders_fallback_when_files_absent(tmp_path):
    from src.data.market_reference import (
        load_risk_free_daily,
        load_market_proxy_returns,
        load_ff_factors,
    )

    dates = np.arange("2005-01-03", 50, dtype="datetime64[D]")
    rf = load_risk_free_daily(dates, raw_dir=tmp_path)
    assert rf.available is False
    assert rf.daily.shape == (50,) and np.all(rf.daily == 0.0)

    mp = load_market_proxy_returns(dates, suffix="testonly", gold_dir=tmp_path)
    assert mp.available is False
    assert mp.returns.shape == (50,)

    ff = load_ff_factors(dates, raw_dir=tmp_path)
    assert ff.available is False and ff.factors == {}


def test_risk_free_missing_source_column_falls_back(tmp_path):
    from src.data.market_reference import load_risk_free_daily

    # File exists but lacks the requested source column → available=False.
    csv = tmp_path / "fred_macro.csv"
    pd.DataFrame({"observation_date": ["2005-01-03"], "OTHER": [1.0]}).to_csv(csv, index=False)
    dates = np.array(["2005-01-03"], dtype="datetime64[D]")
    rf = load_risk_free_daily(dates, source="DGS3MO", raw_dir=tmp_path)
    assert rf.available is False
