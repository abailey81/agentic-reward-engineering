"""Tests for the live H4 reward family (P5; src/baselines/reward_family.py).

The family must obey the LIVE 5-arg/3-tuple contract (the archived version used the old
RewardContext contract), expose a (6,2) search box, and accumulate rolling stats via reward_state.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.reward_family import WEIGHT_KEYS, family_bounds, params_to_reward
from src.reward.contract import validate_signature


def test_family_bounds_shape_and_order() -> None:
    b = family_bounds()
    assert b.shape == (len(WEIGHT_KEYS), 2)
    assert (b[:, 0] <= b[:, 1]).all()
    # Frozen ranges (mirror eureka_loop.yaml).
    assert tuple(b[0]) == (0.0, 2.0)  # w_return
    assert tuple(b[2]) == (0.0, 0.02)  # w_turnover


def test_h4_search_box_is_identical_in_all_three_sources() -> None:
    """#77: the H4 search box is TRIPLICATED, every copy claims to mirror the others, and nothing
    enforced it — on a CONFIRMATORY node.

    The three copies are ``config/prototype.yaml: reward_family.weights``,
    ``config/eureka_loop.yaml: reward_family.weights``, and
    ``src/baselines/reward_family._DEFAULT_BOUNDS``. Each carries a comment asserting the mirror
    ("mirrors eureka_loop.yaml", "Frozen ranges (mirror config/eureka_loop.yaml…)"), yet the only
    enforced eureka mirror in the repo is R97's ``baseline_rewards``
    (``tests/test_baselines.py::test_secondary_panel_config_matches_reward_canon``).

    Why it matters: H4 is the beat-the-max IUT — free-form LLM code versus the BEST member of this
    fixed parametric family. The BOX defines what "best member" means, so drift silently redefines the
    comparator and hence the H4 verdict. And the LIVE source is NOT the documented one: every campaign
    consumer reads **prototype.yaml** (``run_campaign.py:674``, ``run_campaign_cluster.py:160``,
    ``run_prototype.py:529/613``), while ``reward_family.py``'s own docstring points at
    ``eureka_loop.yaml`` — so editing the authoritative-LOOKING file would change nothing that runs.
    Neither YAML is in freeze's ``_BOUND_CONFIGS``, so drift also leaves ``--check`` green.

    NOTE: NOT closed by this test, and NOT mine to decide: the numeric ranges appear NOWHERE in the frozen
    pre-registration (``config/preregistration.yaml`` has no ``reward_family``; PREREGISTRATION.md R28
    registers the six primitives and the fixed ``cvar_alpha=0.05``/``window=20`` in prose, but not the
    weight box). Registering them is a design decision for supervision; this test only locks the three
    live copies to each other.
    """
    from pathlib import Path

    import yaml

    from src.baselines.reward_family import _DEFAULT_BOUNDS

    root = Path(__file__).resolve().parents[1]

    def _box(rel: str) -> dict[str, tuple[float, float]]:
        cfg = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        weights = cfg["reward_family"]["weights"]
        return {k: (float(v["low"]), float(v["high"])) for k, v in weights.items()}

    proto = _box("config/prototype.yaml")
    eureka = _box("config/eureka_loop.yaml")
    code = {k: (float(lo), float(hi)) for k, (lo, hi) in _DEFAULT_BOUNDS.items()}

    assert set(proto) == set(WEIGHT_KEYS), "prototype.yaml box does not cover exactly the six weights"
    assert proto == eureka, (
        "config/prototype.yaml (the LIVE H4 box every campaign consumer reads) has drifted from "
        f"config/eureka_loop.yaml (the documented one): {proto} vs {eureka}"
    )
    assert proto == code, (
        "the code fallback src/baselines/reward_family._DEFAULT_BOUNDS has drifted from the YAML box "
        f"the campaign actually searches: {code} vs {proto}"
    )


def test_family_bounds_reads_cfg() -> None:
    cfg = {"reward_family": {"weights": {"w_return": {"low": 0.5, "high": 1.5}}}}
    b = family_bounds(cfg)
    assert tuple(b[0]) == (0.5, 1.5)


def test_params_to_reward_obeys_live_contract() -> None:
    reward = params_to_reward(np.array([1.0, 0.5, 0.01, 0.05, 1.0, 0.5]))
    assert validate_signature(reward)  # exactly (weights, returns, prev_weights, port_ret, info)
    w = np.full(4, 0.25)
    r = np.array([0.01, -0.02, 0.0])
    total, components, state = reward(w, r, w, 0.005, {})
    assert isinstance(total, float) and np.isfinite(total)
    assert isinstance(components, dict) and "cvar" in components
    assert state is not None  # stateful (history, peak, cum)


def test_params_to_reward_threads_state_and_window() -> None:
    reward = params_to_reward([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], window=5)
    w = np.full(4, 0.25)
    r = np.array([0.01, -0.02, 0.0])
    _t1, _c1, s1 = reward(w, r, w, -0.03, {})
    t2, _c2, s2 = reward(w, r, w, -0.05, {"reward_state": s1})
    assert len(s2[0]) == 2  # rolling history accumulated
    assert np.isfinite(t2)


def test_params_to_reward_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        params_to_reward([1.0, 2.0])


def test_default_bounds_MIRROR_the_frozen_config_exactly() -> None:
    """``_DEFAULT_BOUNDS`` claims to mirror ``config/eureka_loop.yaml: reward_family.weights``.

    A silently-drifted mirror would make the H4 search box differ from the registered one whenever a
    caller omits ``cfg`` — and ``family_bounds`` falls back to the defaults without complaint. This is
    the stale-mirror class this review keeps finding, so it is pinned mechanically (loop 79)."""
    import itertools
    from pathlib import Path

    import yaml

    from src.baselines.reward_family import _DEFAULT_BOUNDS

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "eureka_loop.yaml").read_text(encoding="utf-8")
    )["reward_family"]["weights"]
    assert set(cfg) == set(WEIGHT_KEYS), "the config weight table and WEIGHT_KEYS have diverged"
    for key in WEIGHT_KEYS:
        assert _DEFAULT_BOUNDS[key] == (float(cfg[key]["low"]), float(cfg[key]["high"])), (
            f"{key}: code default {_DEFAULT_BOUNDS[key]} != frozen config "
            f"({cfg[key]['low']}, {cfg[key]['high']})"
        )
    box = family_bounds()
    assert np.isfinite(box).all() and (box[:, 0] < box[:, 1]).all()
    # the itertools import is used by the box-corner sweep below; keep the box small enough to enumerate
    assert len(list(itertools.product(*[(lo, hi) for lo, hi in box]))) == 2 ** len(WEIGHT_KEYS)


def test_turnover_is_TWO_WAY_and_differs_from_the_H1_canon_member() -> None:
    """PIN the turnover convention: the family uses TWO-WAY ``sum|w - w_prev|``, range [0, 2].

    Finding #52 (loop 79): ``config/eureka_loop.yaml`` documented this term as "one-way, in [0, 1]",
    which is what the H1 canon member ``rewards.py::return_minus_turnover`` actually uses
    (Garleanu-Pedersen ``0.5*sum|.|``). The family computes the two-way sum, so at the maximum
    rebalance the term reaches ``0.02 * 2.0 = 0.04`` per step — twice the config's own stated
    "~+/-0.02 per step" sizing parity. The code is deliberately unchanged (altering it would change
    the REGISTERED H4 search space); the docs were corrected to match. This test pins the convention
    so a silent flip in either direction is caught."""
    from src.baselines.rewards import return_minus_turnover

    w = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    wp = np.array([0.0, 0.0, 0.0, 0.0, 1.0])   # fully disjoint = the maximum possible rebalance

    fam = params_to_reward([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # isolate w_turnover
    _, comp, _ = fam(w, None, wp, 0.0, {"reward_state": None})
    assert comp["turnover"] == 2.0, "the H4 family must use the TWO-WAY sum|w - w_prev|"

    _, comp_h1, _ = return_minus_turnover(w, None, wp, 0.0, {})
    assert comp_h1["turnover"] == 1.0, "the H1 canon member must stay ONE-WAY (0.5*sum|.|)"

    # and the consequence the config now documents
    assert 0.02 * comp["turnover"] == pytest.approx(0.04)


def test_materialised_source_is_GATE_CLEAN_and_bit_exact_across_the_box() -> None:
    """Every point in the box must materialise to sandbox-admissible source that matches the closure.

    Extends the existing ``test_source_and_closure_parity`` (one coefficient vector, fixed weights,
    benign returns, bare ``exec``) along the three axes it does not cover (loop 79):
      * the WHOLE box — all 64 corners plus interior points, not a single vector;
      * the real admissibility path — ``ast_gate`` AND ``defines_reward``, which is precisely what
        stops a winner being archived as ``winner_not_testable``; a bare ``exec`` cannot catch that;
      * PATHOLOGICAL returns — a <= -100% step exercises the ``log1p`` clip branch that permanently
        poisons the stateful cum/peak carry if it regresses, and which benign draws never reach.
    A drift here means the sealed TEST leg runs a different reward than the search scored, and the
    archive would not notice: it hash-verifies the SOURCE TEXT, not the behaviour."""
    import itertools

    from src.baselines.reward_family import params_to_source
    from src.sandbox.executor import ast_gate, defines_reward

    box = family_bounds()
    rng = np.random.default_rng(0)
    corners = [np.array(p, dtype=float) for p in itertools.product(*[(lo, hi) for lo, hi in box])]
    points = corners + [rng.uniform(box[:, 0], box[:, 1]) for _ in range(12)] + [np.zeros(6)]

    for p in points:
        src = params_to_source(p)
        assert ast_gate(src), f"generated source failed ast_gate at coeffs={list(p)}"
        assert defines_reward(src), f"generated source binds no reward at coeffs={list(p)}"

    def _materialise(src: str):
        ns: dict = {"np": np}
        exec(compile(src, "<family>", "exec"), ns)  # noqa: S102 - source we generated ourselves
        return ns["reward"]

    # bit-exact parity on a stressed call sequence, over a sample of the box
    pathological = {5: -1.5, 9: -0.999999, 14: 3.0}   # wipeout / at-the-clip / huge gain
    for p in corners[::7] + points[-3:]:
        closure = params_to_reward(p)
        materialised = _materialise(params_to_source(p))
        st_c = st_m = None
        w_prev = np.full(5, 0.2)
        for t in range(25):
            w = rng.dirichlet(np.ones(5))          # turnover VARIES, unlike the existing test
            r = pathological.get(t, float(rng.normal(0.0, 0.02)))
            tc, cc, st_c = closure(w, None, w_prev, r, {"reward_state": st_c})
            tm, cm, st_m = materialised(w, None, w_prev, r, {"reward_state": st_m})
            assert tc == tm, f"total drift at t={t}, coeffs={list(p)}: {tc} != {tm}"
            assert cc == cm, f"component drift at t={t}, coeffs={list(p)}"
            assert np.isfinite(tc), f"non-finite reward at t={t} (r={r})"
            w_prev = w
