"""Tests for the report-only mechanism sub-experiment driver + its three default-off seams (ADR-039).

Stub-designer path only — NO real LLM / SAC / torch. The driver's collaborators (env_builder, trainer,
measurement) and reward author (transport) are INJECTED via fakes, exactly as tests/test_loop.py fakes the
trainer + transport. Covered:

  1. schema.build_block(legible=True) preserves every _DIST_FIELDS label substring AND renders bps + decile;
     legible=False is BYTE-IDENTICAL to the current raw output (the frozen-path byte-identity guard);
  2. _was_fed_tail returns True on a legible-rendered block;
  3. run_subexperiment in each mode writes records carrying the right label/condition discriminator and
     (named) equal-length pairable lists;
  4. END-TO-END ACCEPTANCE: feed the produced named_blinded_root / legible_root into analyze() and assert
     both legs flip from no_data to status == "ok".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_campaign as AC  # noqa: E402
import run_subexperiment as RS  # noqa: E402

from src.feedback import schema  # noqa: E402
from src.feedback.measurement import ReturnDistribution  # noqa: E402

# Six well-defined tail stats (the frozen set) for the schema byte-identity / label-preservation tests.
_TAIL = {
    "cvar_05": -0.0410,
    "cvar_10": -0.0290,
    "cvar_25": -0.0160,
    "cvar_01": -0.0670,
    "left_tail_mass": 0.0610,
    "robust_skew": -0.38,
}


# --------------------------------------------------------------------------- #
# Seam 1 — schema.build_block(legible=...)
# --------------------------------------------------------------------------- #
def test_legible_preserves_labels_and_renders_bps() -> None:
    """legible=True keeps EVERY _DIST_FIELDS label substring (else _was_fed_tail filters every record) and
    renders CVaR/tail-mass as basis points with a decile-rank tag."""
    block = schema.build_block("distributional", 0.83, _TAIL, legible=True)
    for _fid, label in schema._DIST_FIELDS:
        assert f"{label}:" in block, f"legible block dropped the {label!r} label substring"
    assert "CVaR 5%:" in block
    assert "bps" in block
    assert "decile" in block
    # -0.0410 -> round(-0.0410 * 1e4) = -410 bps (the worked example).
    assert "-410 bps" in block
    # robust_skew renders dimensionless 2-dp, NOT bps.
    assert "left-tail skew: -0.38" in block
    # the CVaR-1% high-variance annotation survives the legible rendering (matched structure).
    assert "(high-variance estimate)" in block


def test_legible_decile_rises_with_tail_risk_on_every_field() -> None:
    """The decile tag must mean the SAME thing on all six lines: HIGHER decile = MORE tail risk.

    The tag exists to give an LLM a coarse ORDINAL framing it can compare without float arithmetic
    (the numeracy-bottleneck device, ADR-039). Until 2026-07-26 the direction FLIPPED between lines —
    each field was bucketed positionally over its own band, so a risky tail rendered as
    "CVaR 1%: decile 1/10" beside "left-tail mass: decile 9/10", both meaning "worst". A tag whose
    direction is inconsistent is worse than no tag, and it biases
    ``responsiveness.legible_format_responsiveness_differential`` toward a spurious null.
    """
    safe = {"cvar_05": -0.005, "cvar_10": -0.004, "cvar_25": -0.003, "cvar_01": -0.008,
            "left_tail_mass": 0.010, "robust_skew": +0.20}
    risky = {"cvar_05": -0.090, "cvar_10": -0.070, "cvar_25": -0.050, "cvar_01": -0.110,
             "left_tail_mass": 0.085, "robust_skew": -0.80}

    def _deciles(stats: dict) -> dict[str, int]:
        out: dict[str, int] = {}
        for line in schema.build_block("distributional", 0.5, stats, legible=True).splitlines():
            if "(decile" in line:
                out[line.split(":")[0].strip()] = int(line.split("(decile ")[1].split("/")[0])
        return out

    safe_d, risky_d = _deciles(safe), _deciles(risky)
    assert len(risky_d) == len(schema._DIST_FIELDS) == 6
    for label, risky_val in risky_d.items():
        assert risky_val > safe_d[label], (
            f"{label}: decile must RISE with tail risk (safe={safe_d[label]}, risky={risky_val})"
        )
    assert all(1 <= d <= 10 for d in {**safe_d, **risky_d}.values())


def test_legible_false_is_byte_identical_to_raw() -> None:
    """The frozen-path guard: legible=False reproduces the current raw bytes EXACTLY (default-off)."""
    raw_default = schema.build_block("distributional", 0.83, _TAIL)
    raw_explicit = schema.build_block("distributional", 0.83, _TAIL, legible=False)
    assert raw_default == raw_explicit
    # And it still matches the documented raw worked-example renderings.
    assert "CVaR 5%: -0.041" in raw_default
    assert "bps" not in raw_default


def test_other_arms_unaffected_by_legible_flag() -> None:
    """legible only touches the distributional six-line block; every other arm is byte-unchanged."""
    assert schema.build_block("scalar", 0.5, None, legible=True) == schema.build_block("scalar", 0.5, None)
    assert schema.build_block("placebo", 0.5, None, legible=True) == schema.build_block("placebo", 0.5, None)
    assert (
        schema.build_block("scalar_cvar5", 0.5, _TAIL, legible=True)
        == schema.build_block("scalar_cvar5", 0.5, _TAIL)
    )


# --------------------------------------------------------------------------- #
# Seam 1/analysis bridge — _was_fed_tail fires on a legible block
# --------------------------------------------------------------------------- #
def test_was_fed_tail_true_on_legible_block() -> None:
    """The legible rendering preserves the exact label substrings, so the fed-tail gate still fires —
    checked on the CORRECTED archive semantics (2026-07-05 M13): the fed text lives in the archived
    PROMPT; a bare own-``feedback_block`` counts only for legacy prompt-less records at generation>=1
    (a generation-0 own block must NOT pass — the designer was fed nothing)."""
    ir = AC._inspect_rewards()
    legible_block = schema.build_block("distributional", 0.83, _TAIL, legible=True)
    raw_block = schema.build_block("distributional", 0.83, _TAIL, legible=False)
    # the fed block in the PROMPT (the corrected read) — legible and raw both match
    assert ir._was_fed_tail({"prompt": f"Improve the reward.\n{legible_block}"}) is True
    assert ir._was_fed_tail({"prompt": f"Improve the reward.\n{raw_block}"}) is True
    # legacy prompt-less records: own block gates only at generation >= 1
    assert ir._was_fed_tail({"feedback_block": legible_block, "generation": 1}) is True
    assert ir._was_fed_tail({"feedback_block": raw_block, "generation": 1}) is True
    assert ir._was_fed_tail({"feedback_block": raw_block, "generation": 0}) is False


# --------------------------------------------------------------------------- #
# Injected fakes (no torch / no real LLM) — mirror tests/test_loop.py
# --------------------------------------------------------------------------- #
# Reward sources carrying a VARYING number of tail-shaped constructs (cvar / quantile / drawdown / sortino /
# left_tail), so _construct_prevalence recovers a per-candidate tail-construct count M that is NON-constant.
# Each line introduces ONE more tail-shaped construct keyword (cvar / quantile / drawdown / sortino /
# left_tail) into the REWARD CODE (not a comment — _construct_prevalence strips comments first), so the
# per-candidate tail-construct count M = _construct_prevalence(...) is NON-constant across candidates.
_TAIL_CODE_LINES = (
    "    cvar = float(np.mean(returns))\n",
    "    q = float(np.percentile(returns, 5))\n",
    "    drawdown = float(np.min(returns))\n",
    "    sortino = float(np.std(returns))  # downside\n",
    "    left_tail = float(np.sum(returns))\n",
)


def _reward_src(n_tail: int) -> str:
    n = max(1, min(n_tail, len(_TAIL_CODE_LINES)))
    body = "".join(_TAIL_CODE_LINES[:n])
    return (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        f"{body}"
        "    total = float(port_ret)\n"
        "    return total, {'pnl': total}, None\n"
    )


class _CyclingTransport:
    """A keyless transport cycling reward sources with ascending tail-construct counts (so M varies).

    The cycle starts at the SEED offset (2026-07-06): it previously ignored its seed, so every seed
    authored the identical archetype window and the per-seed MEAN construct count was constant —
    degenerate for the S8 seed-cell analysis in a way the real (Opus) leg is not."""

    def __init__(self, seed: int = 0) -> None:
        self._n = int(seed)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        n_tail = (self._n % len(_TAIL_CODE_LINES)) + 1
        self._n += 1
        return _reward_src(n_tail)


class _FakePolicy:
    def __init__(self, level: float) -> None:
        self.level = level


class _FakeEnv:
    """Per-reward fake env yielding NON-constant train returns per candidate (so cvar_05 = X varies)."""

    _counter = 0

    def __init__(self, reward_fn) -> None:
        self.reward_fn = reward_fn
        _FakeEnv._counter += 1
        self.level = float(_FakeEnv._counter)

    def train_env(self):
        return self

    def train_returns(self, policy) -> np.ndarray:
        rng = np.random.default_rng(int(self.level) * 7 + 1)
        # Negative-mean, heavy-tailed: a well-defined, candidate-varying left tail (cvar_05).
        return -0.001 * self.level + 0.01 * rng.standard_t(5, size=512)

    def val_returns(self, policy) -> np.ndarray:
        rng = np.random.default_rng(int(self.level) * 13 + 3)
        return policy.level * 0.001 + 0.01 * rng.standard_normal(512)


def _fake_trainer():
    state = {"n": 0}

    def _train(train_env) -> _FakePolicy:
        state["n"] += 1
        return _FakePolicy(level=float(state["n"]))

    return _train


def _collaborators_factory(seed: int) -> dict:
    """Inject the fakes + the SAME minimal prompts the loop would render; no torch/panel/SAC."""
    _FakeEnv._counter = 0  # reset per seed so candidate levels are deterministic

    class _Panel:
        T = 600
        N = 30
        dates = np.arange(600).astype("datetime64[D]")

    return {
        "panel": _Panel(),
        "env_builder": _FakeEnv,
        "agent_trainer": _fake_trainer(),
        "measurement": ReturnDistribution,
        "prompts": {"system": "SYS", "initial": "INIT"},
        "ric_by_id": {0: "AAA.OQ", 1: "BBB.N"},
        "regime_names": ["calm", "normal", "stress"],
    }


def _transport_factory(seed: int):
    return _CyclingTransport(seed=seed), f"stub/seed{seed}"


def _run(mode: str, output_dir: Path, *, seeds=(0, 1), candidates: int = 4) -> dict:
    return RS.run_subexperiment(
        mode,
        seeds=list(seeds),
        candidates=candidates,
        output_dir=output_dir,
        sub_cfg={},
        collaborators_factory=_collaborators_factory,
        transport_factory=_transport_factory,
        # mocked transport (no paid Opus), so the inert-legible money guard is not relevant here.
        allow_inert_legible=True,
    )


# --------------------------------------------------------------------------- #
# Seam 3 — the driver stamps the right discriminator + (named) equal-length pairable lists
# --------------------------------------------------------------------------- #
def test_named_mode_writes_label_and_equal_length_pairs(tmp_path: Path) -> None:
    summary = _run("named", tmp_path, seeds=(0, 1), candidates=4)
    assert summary["mode"] == "named"
    assert summary["conditions"] == ["named", "blinded"]
    assert summary["n_paired"] > 0

    ir = AC._inspect_rewards()
    recs = AC.load_campaign_records(tmp_path)
    named = [r for r in ir._by_generation(recs) if r.get("label") == "named"]
    blinded = [r for r in ir._by_generation(recs) if r.get("label") == "blinded"]
    assert named and blinded
    # FLAG B: after _prune_to_intersection the two legs are equal-length and (gen, candidate_id)-aligned.
    assert len(named) == len(blinded)
    for n, b in zip(named, blinded):
        assert n["candidate_id"] == b["candidate_id"]
        assert n["generation"] == b["generation"]
    # The NAMED leg's initial prompt carried the data-provenance paragraph; blinded did not.
    assert any("DATA PROVENANCE" in str(r.get("prompt", "")) for r in named)
    assert all("DATA PROVENANCE" not in str(r.get("prompt", "")) for r in blinded)


def test_legible_mode_writes_condition_and_legible_rendering(tmp_path: Path) -> None:
    summary = _run("legible", tmp_path, seeds=(0, 1), candidates=4)
    assert summary["mode"] == "legible"
    assert summary["conditions"] == ["legible", "raw"]

    recs = AC.load_campaign_records(tmp_path)
    legible = [r for r in recs if r.get("condition") == "legible"]
    raw = [r for r in recs if r.get("condition") == "raw"]
    assert legible and raw
    # The legible leg rendered the tail block in basis points; the raw leg did not.
    assert all("bps" in str(r.get("feedback_block", "")) for r in legible)
    assert all("bps" not in str(r.get("feedback_block", "")) for r in raw)
    # Both legs keep the CVaR 5%: label so _was_fed_tail (the mechanism gate) still fires.
    ir = AC._inspect_rewards()
    assert all(ir._was_fed_tail(r) for r in legible + raw)


# --------------------------------------------------------------------------- #
# END-TO-END ACCEPTANCE — analyze() flips both legs from no_data to ok
# --------------------------------------------------------------------------- #
def test_named_leg_analyze_flips_to_ok(tmp_path: Path) -> None:
    named_root = tmp_path / "named"
    _run("named", named_root, seeds=(0, 1, 2), candidates=4)
    out = AC.analyze(tmp_path / "empty_main", named_blinded_root=named_root)
    nvb = out["named_vs_blinded_structural"]
    assert nvb["status"] == "ok", nvb
    assert nvb["n_seeds"] >= 2


def test_legible_leg_analyze_flips_to_ok(tmp_path: Path) -> None:
    # S8 (2026-07-06): the differential's bootstrap unit is the SEED CELL (x is cell-constant by
    # design), so >= 3 seeds per condition are required for a coefficient at all — 4 here, mirroring
    # the real leg's >= 5. (The old 2-seed fixture passed only because candidate-level rows
    # fabricated n=8 from 2 independent units — the anti-conservative clustering S8 closed.)
    legible_root = tmp_path / "legible"
    # candidates=3 = a PARTIAL archetype cycle, so the per-seed mean m varies across the seed-offset
    # windows (a full cycle of 4 would make the mean offset-invariant — degenerate by arithmetic).
    _run("legible", legible_root, seeds=(0, 1, 2, 3), candidates=3)
    out = AC.analyze(tmp_path / "empty_main", legible_root=legible_root)
    leg = out["legible_format_responsiveness"]
    assert leg["status"] == "ok", leg
    # both conditions produced a non-degenerate responsiveness coefficient over the seed cells
    assert leg["legible"]["status"] == "ok" and leg["raw"]["status"] == "ok"
    assert leg["legible"]["n"] == 4 and leg["raw"]["n"] == 4  # one row per SEED (4 seeds), not per candidate
    assert "clustering" in leg  # the S8 disclosure rides in the output
