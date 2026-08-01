"""DEEP, advanced tests for the LLM reward-design loop layer (dissertation H2).

The feedback BLOCK is the H2 contribution, so the prompt + schema that carry it are
load-bearing — and they must NEVER leak tickers, dates, or raw ids. These tests go
beyond the happy-path coverage already in tests/test_loop.py, test_prompts.py,
test_schema.py, test_stub_designer.py, and test_llm_transport.py with:

  * property-based (Hypothesis, derandomize=True via conftest's "deterministic" profile),
  * metamorphic (information-monotonicity across arms, seed-invariance),
  * adversarial (info dicts laced with date/ticker strings must not reach any prompt),
  * boundary (extreme metrics, NaN/inf tail values, empty/whitespace LLM output).

EVERYTHING external is faked / monkeypatched — NO real network or LLM call is ever made.
Covered modules: src/llm/loop.py, src/llm/prompts.py, src/llm/client.py,
src/llm/stub_designer.py, src/feedback/schema.py.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from src.feedback import schema
from src.feedback.measurement import ReturnDistribution
from src.llm import loop
from src.llm.client import FakeTransport, LLMClient, ProvenanceRecord
from src.llm.prompts import build_prompt_set, render_env_interface
from src.llm.stub_designer import ARCHETYPES, StubDesignerTransport
from src.sandbox.executor import ast_gate, validate_once
from src.selection.fitness import held_out_fitness
from src.utils.config import load_config

try:
    from hypothesis import given
    from hypothesis import strategies as st

    _HAS_HYP = True
except ImportError:  # pragma: no cover - hypothesis is a declared dep
    _HAS_HYP = False


# The six FROZEN tail keys (schema._DIST_FIELDS) — the H2 information set. Pinned here as a
# literal so a silent rename/reorder of the frozen design fails this test loudly.
FROZEN_TAIL_KEYS = ("cvar_05", "cvar_10", "cvar_25", "cvar_01", "left_tail_mass", "robust_skew")

TAIL_STATS = {
    "cvar_01": -0.067,
    "cvar_05": -0.041,
    "cvar_10": -0.029,
    "cvar_25": -0.016,
    "left_tail_mass": 0.061,
    "robust_skew": -0.38,
}

# Shape-faithful to the production call (fixture-parity fix 2026-07-03): weights carries the cash
# slot (N+1) while returns is the risky leg (N) — run_loop's validation fixture now mirrors that, so
# the old `np.dot(weights, returns)` (equal-length assumption) would be REJECTED at validate_once.
_VALID_REWARD_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    total = float(np.dot(weights[:-1], returns))\n"
    "    return total, {'pnl': total}, None\n"
)


# --------------------------------------------------------------------------- #
# Shared fakes (independent of tests/test_loop.py's module-level fakes)         #
# --------------------------------------------------------------------------- #
class _CannedLLM:
    """Injected LLM: returns a fixed (already-extracted) source; records every prompt + cfg."""

    def __init__(self, source: str = _VALID_REWARD_SRC) -> None:
        self.source = source
        self.cfg = {"model": "fake-model-2026"}
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.source


class _Policy:
    def __init__(self, level: float) -> None:
        self.level = level


class _LeveledEnv:
    """Per-reward fake env; a distinct level per construction so candidates differ."""

    _counter = 0

    def __init__(self, reward_fn) -> None:
        self.reward_fn = reward_fn
        _LeveledEnv._counter += 1
        self.level = float(_LeveledEnv._counter)

    def train_env(self):
        return self

    def train_returns(self, _policy) -> np.ndarray:
        rng = np.random.default_rng(int(self.level) * 7 + 1)
        return -0.001 + 0.01 * rng.standard_t(5, size=512)

    def val_returns(self, policy) -> np.ndarray:
        rng = np.random.default_rng(int(self.level) * 13 + 3)
        return policy.level * 0.001 + 0.01 * rng.standard_normal(512)


def _ascending_trainer():
    state = {"n": 0}

    def _train(_train_env) -> _Policy:
        state["n"] += 1
        return _Policy(level=float(state["n"]))

    return _train


def _run(arm: str, cfg: dict, archive_root, *, source: str = _VALID_REWARD_SRC):
    _LeveledEnv._counter = 0
    llm = _CannedLLM(source)
    archive = loop.run_loop(
        arm=arm,
        env_builder=_LeveledEnv,
        llm=llm,
        agent_trainer=_ascending_trainer(),
        measurement=ReturnDistribution,
        fitness_fn=held_out_fitness,
        cfg=cfg,
        archive_root=archive_root,
    )
    return archive, llm


# A date-like / ticker-like adversarial probe set. If any of these substrings ever appear in a
# rendered prompt, the anonymisation contract (N3) is broken.
_LEAK_PROBES = (
    "2021-03-15",        # ISO date
    "03/15/2021",        # US date
    "15 Jan 2020",       # prose date
    "AAPL",              # ticker
    "RIC:VOD.L",         # Refinitiv RIC
    "GME US Equity",     # Bloomberg-style id
    "Q1-2022",
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


# =========================================================================== #
# prompts.py — anonymisation is load-bearing (N3)                              #
# =========================================================================== #
def test_env_interface_leaks_no_date_or_ticker_even_with_laced_config() -> None:
    """ADVERSARIAL: lace the env config with date/ticker strings; the rendered interface (built
    from SHAPES only) must not echo any of them — render_env_interface reads numbers, never ids."""
    base = load_config("environment")
    # Build a dict overlay that injects poison into every field render_env_interface reads.
    poisoned = dict(base) if isinstance(base, dict) else {}
    poisoned["state"] = {
        "lookback_days": 60,
        "realized_vol_windows": [20, 60],
        "include_vix": True,
        "include_prev_weights": True,
        "cash_daily_rate": 0.0,
        # poison fields the renderer should IGNORE
        "first_date": "2021-03-15",
        "tickers": ["AAPL", "VOD.L"],
        "ric": "RIC:VOD.L",
    }
    s = render_env_interface(poisoned, n_assets=30)
    for probe in _LEAK_PROBES:
        assert probe not in s, f"interface leaked {probe!r}"
    assert not _DATE_RE.search(s), "interface contains a date-like token"
    # And it still rendered the real (anonymised) contract.
    assert "def reward(weights, returns, prev_weights, port_ret, info)" in s


def test_env_interface_render_is_deterministic_in_inputs() -> None:
    """Same (env_cfg, n_assets) -> byte-identical interface (replayable prompts, C-2)."""
    cfg = load_config("environment")
    assert render_env_interface(cfg, 30) == render_env_interface(cfg, 30)
    # Distinct asset counts give distinct renders (the shape is actually threaded through).
    assert render_env_interface(cfg, 30) != render_env_interface(cfg, 8)


def test_build_prompt_set_initial_carries_no_date_or_ticker() -> None:
    """The assembled gen-0 prompt (system + filled initial) contains no date/ticker token."""
    cfg = load_config("environment")
    ps = build_prompt_set(cfg, n_assets=30)
    for text in (ps.system, ps.initial):
        for probe in _LEAK_PROBES:
            assert probe not in text
        assert not _DATE_RE.search(text), "assembled prompt has a date-like token"
    assert "{ENV_INTERFACE}" not in ps.initial  # placeholder filled


def test_cash_term_renders_only_when_rate_nonzero() -> None:
    """BOUNDARY: cash_daily_rate=0 omits the cash term; a nonzero rate surfaces it (mirrors env)."""
    base = load_config("environment")
    common = {
        "lookback_days": 60, "realized_vol_windows": [20, 60],
        "include_vix": True, "include_prev_weights": True,
    }
    z = dict(base) if isinstance(base, dict) else {}
    z["state"] = {**common, "cash_daily_rate": 0.0}
    nz = dict(base) if isinstance(base, dict) else {}
    nz["state"] = {**common, "cash_daily_rate": 0.0001}
    assert "cash_daily_rate" not in render_env_interface(z, 10)
    assert "cash_daily_rate" in render_env_interface(nz, 10)


# =========================================================================== #
# loop.py prompt assembly — the DISTRIBUTIONAL arm carries the six tail        #
# scalars; SCALAR does not; PLACEBO matches length but is inert; no leakage    #
# =========================================================================== #
def test_distributional_reflection_prompt_contains_all_six_tail_scalars(tmp_path) -> None:
    """The gen-1 reflection prompt for the distributional arm carries ALL SIX frozen tail labels —
    the full H2 information set, not just one CVaR line. (Existing tests check only a subset.)"""
    cfg = {"generations": 2, "candidates_per_gen": 1, "budget": 100, "n_trials": 5}
    _, llm = _run("distributional", cfg, tmp_path)
    gen1_user = llm.prompts[1][1]
    for label in ("CVaR 5%", "CVaR 10%", "CVaR 25%", "CVaR 1%", "left-tail mass", "left-tail skew"):
        assert label in gen1_user, f"distributional prompt missing {label!r}"
    assert "high-variance" in gen1_user.lower()  # CVaR-1% B-7 annotation carried into the prompt


def test_scalar_arm_prompt_carries_only_the_scalar(tmp_path) -> None:
    """METAMORPHIC: the scalar arm's reflection prompt has the scalar header and NO tail content."""
    cfg = {"generations": 2, "candidates_per_gen": 1, "budget": 100, "n_trials": 5}
    _, llm = _run("scalar", cfg, tmp_path)
    gen1_user = llm.prompts[1][1]
    assert "Deflated Sharpe" in gen1_user
    assert "CVaR" not in gen1_user
    assert "tail" not in gen1_user.lower()


def test_placebo_reflection_matches_distributional_linecount_but_is_inert(tmp_path) -> None:
    """The placebo reflection prompt matches the distributional one in LINE COUNT (token-count
    control) yet carries inert reference values and no tail diagnostics (isolates information)."""
    cfg = {"generations": 2, "candidates_per_gen": 1, "budget": 100, "n_trials": 5}
    _, dist_llm = _run("distributional", cfg, tmp_path / "d")
    _, plac_llm = _run("placebo", cfg, tmp_path / "p")
    dist_body = dist_llm.prompts[1][1]
    plac_body = plac_llm.prompts[1][1]
    assert len(dist_body.splitlines()) == len(plac_body.splitlines())
    assert "CVaR" not in plac_body
    assert "reference value" in plac_body
    assert "inert" in plac_body.lower()


def test_no_reflection_prompt_leaks_dates_or_tickers_across_all_arms(tmp_path) -> None:
    """ADVERSARIAL/METAMORPHIC: the feedback block is rendered from FLOATS only; no arm's prompt
    can carry a date/ticker even though the loop's measurement feeds it real return statistics."""
    cfg = {"generations": 2, "candidates_per_gen": 1, "budget": 100, "n_trials": 5}
    for i, arm in enumerate(("distributional", "scalar", "placebo", "scalar_cvar5", "placebo_shuffled")):
        _, llm = _run(arm, cfg, tmp_path / f"arm{i}")
        for _system, user in llm.prompts:
            assert not _DATE_RE.search(user), f"{arm} prompt has a date-like token"
            for probe in _LEAK_PROBES:
                assert probe not in user


# =========================================================================== #
# schema.py — frozen keys + validate-correct / reject-malformed                #
# =========================================================================== #
def test_frozen_tail_key_set_is_exactly_the_six() -> None:
    """The frozen tail key SET (schema._DIST_FIELDS) is EXACTLY the six H2 scalars — no more,
    no fewer, no rename. Cross-checks the live measurement.tail_stats() keys too."""
    schema_keys = {fid for fid, _ in schema._DIST_FIELDS}
    assert schema_keys == set(FROZEN_TAIL_KEYS)
    # The estimator's emitted keys must match the schema's expected keys exactly.
    dist = ReturnDistribution()
    dist.fit(-0.001 + 0.01 * np.random.default_rng(0).standard_t(5, size=2048))
    assert set(dist.tail_stats().keys()) == set(FROZEN_TAIL_KEYS)


def test_build_block_accepts_a_correct_tail_stats_dict_and_round_trips() -> None:
    """A well-formed tail dict renders, and every value is recoverable from the text — a round-trip
    of the H2 payload through the prompt-facing string.

    The expected strings come from ``schema._fmt`` rather than a hardcoded precision. This test
    duplicated ``+.3f`` and so silently asserted a PRECISION it was not written to test: when #87b
    raised the renderer to ``.4f`` (because ``cvar_25`` at ``.3f`` left a quarter of genuinely
    different values indistinguishable) this failed on formatting alone, while the round-trip
    property it exists to check was never in doubt. Deriving from ``_fmt`` keeps it testing the
    round-trip and lets the treatment's resolution be set where it is reasoned about.
    """
    block = schema.build_block("distributional", 0.83, TAIL_STATS)
    for _fid, label in schema._DIST_FIELDS:
        assert label in block
    # Recover the rendered numbers and check they equal the renderer's output for the fed values.
    rendered = {
        ln.split(":", 1)[0].strip(): ln.split(":", 1)[1].strip().split()[0]
        for ln in block.splitlines() if ln.startswith("  ")
    }
    assert rendered["CVaR 5%"] == schema._fmt(TAIL_STATS["cvar_05"])
    assert rendered["left-tail skew"] == schema._fmt(TAIL_STATS["robust_skew"])
    # And the rendered text really does round-trip back to the fed value at the renderer's precision.
    assert float(rendered["CVaR 5%"]) == pytest.approx(TAIL_STATS["cvar_05"], abs=5e-5)


def test_build_block_rejects_missing_field_short_block() -> None:
    """A tail dict MISSING a frozen field is rejected (KeyError) — a short/malformed block cannot
    silently render with a dropped diagnostic."""
    short = {k: v for k, v in TAIL_STATS.items() if k != "robust_skew"}
    with pytest.raises(KeyError):
        schema.build_block("distributional", 0.83, short)


def test_build_block_ignores_extra_fields_but_renders_only_frozen_six() -> None:
    """An EXTRA (non-frozen) field is ignored: the rendered block carries exactly the six frozen
    labels and nothing derived from the smuggled extra key."""
    laced = {**TAIL_STATS, "secret_alpha": 9.99, "ticker": "AAPL"}
    block = schema.build_block("distributional", 0.83, laced)
    assert "9.99" not in block
    assert "AAPL" not in block
    indented = [ln for ln in block.splitlines() if ln.startswith("  ")]
    assert len(indented) == len(schema._DIST_FIELDS)  # exactly six value lines


def test_build_block_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        schema.build_block("not-an-arm", 0.5, TAIL_STATS)


@pytest.mark.parametrize("arm", ["distributional", "scalar_cvar5", "placebo_shuffled"])
def test_tail_carrying_arms_require_tail_stats(arm) -> None:
    """Every tail-carrying arm fails loud (ValueError) when given tail_stats=None."""
    kw = {"shuffle_seed": 1} if arm == "placebo_shuffled" else {}
    with pytest.raises(ValueError):
        schema.build_block(arm, 0.5, None, **kw)


if _HAS_HYP:
    @given(
        metric=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        vals=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=6, max_size=6,
        ),
    )
    def test_property_build_block_is_pure_and_matched_structure(metric, vals) -> None:
        """PROPERTY (derandomized): for arbitrary inputs the block is (a) pure/deterministic,
        (b) the distributional and placebo blocks ALWAYS share a line count (the matched-structure
        invariant the H2 claim depends on), and (c) the scalar block is a single line."""
        ts = dict(zip(FROZEN_TAIL_KEYS, vals))
        dist = schema.build_block("distributional", metric, ts)
        assert dist == schema.build_block("distributional", metric, ts)  # pure
        placebo = schema.build_block("placebo", metric, None)
        assert len(dist.splitlines()) == len(placebo.splitlines())  # matched structure
        scalar = schema.build_block("scalar", metric, None)
        assert len(scalar.splitlines()) == 1

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_property_placebo_shuffled_preserves_marginal_multiset(seed) -> None:
        """PROPERTY: for ANY seed, placebo_shuffled renders the SAME multiset of numbers as the
        distributional block (it permutes labels, never invents/drops a value)."""
        ts = {k: TAIL_STATS[k] for k in FROZEN_TAIL_KEYS}
        dist = schema.build_block("distributional", 0.5, ts)
        shuf = schema.build_block("placebo_shuffled", 0.5, ts, shuffle_seed=seed)

        def _nums(b: str) -> list[str]:
            return sorted(
                ln.split(":", 1)[1].strip().split()[0]
                for ln in b.splitlines() if ln.startswith("  ")
            )

        assert _nums(dist) == _nums(shuf)


def test_build_block_handles_nonfinite_tail_values() -> None:
    """BOUNDARY: NaN/inf tail values render (np float-format) without raising — the loop must not
    crash if a degenerate return series yields a non-finite CVaR; it surfaces it honestly."""
    bad = {**TAIL_STATS, "cvar_01": float("nan"), "cvar_05": float("inf"), "cvar_10": float("-inf")}
    block = schema.build_block("distributional", 0.5, bad)
    assert "nan" in block.lower()
    assert "inf" in block.lower()


# =========================================================================== #
# stub_designer.py — deterministic by seed, gate-valid, varies with seed       #
# =========================================================================== #
def test_stub_same_seed_same_first_source_gate_valid() -> None:
    """Same seed -> identical first reward source, and that source passes the AST gate + validates."""
    a = StubDesignerTransport(seed=123)("sys", "usr")
    b = StubDesignerTransport(seed=123)("sys", "usr")
    assert a == b
    assert ast_gate(a) is True
    # Production shape parity (2026-07-03): returns has ONE FEWER element than weights/prev_weights
    # (weights carries the cash slot) — the same contract shapes run_loop's validation fixture uses.
    fixture = (
        np.full(31, 1.0 / 31), np.full(30, 0.001), np.full(31, 1.0 / 31), 0.0, {},
    )
    fn = validate_once(a, fixture, timeout_s=4.0)  # raises SandboxError on failure
    total, components, _state = fn(*fixture)
    assert np.isfinite(float(total))
    assert isinstance(components, dict)


def test_stub_varies_with_seed() -> None:
    """METAMORPHIC: different seeds give different coefficient sequences (variety is seed-driven).
    Compared on the coefficient-bearing archetypes (the first, return-only, is coefficient-free)."""
    n = len(ARCHETYPES)
    # Draw a full cycle from ONE transport per seed (a fresh transport each call would always
    # return the coefficient-free first archetype). The coefficient grids are small, so compare
    # over a couple of cycles to be robust to an occasional collision on a single draw.
    stub0, stub1 = StubDesignerTransport(seed=0), StubDesignerTransport(seed=1)
    seq0 = [stub0("s", "u") for _ in range(2 * n)]
    seq1 = [stub1("s", "u") for _ in range(2 * n)]
    # At least one position differs (the RNG-filled coefficients differ across seeds).
    assert any(x != y for x, y in zip(seq0, seq1))


def test_stub_is_a_pure_function_of_seed_and_call_index() -> None:
    """2026-07-06 (crash-rehearsal catch): call ``n`` depends ONLY on ``(seed, n)`` — reset()
    therefore replays the cycle IDENTICALLY, and advance(k) skips positions so a resume that
    replays k archived candidates leaves the next fresh candidate byte-identical to the one an
    uninterrupted run would have authored at that index."""
    stub = StubDesignerTransport(seed=5)
    first = [stub("s", "u") for _ in range(len(ARCHETYPES))]
    stub.reset()
    second = [stub("s", "u") for _ in range(len(ARCHETYPES))]
    assert first == second  # pure f(seed, index): the replayed cycle is byte-identical
    # advance(k) == "the first k slots were replayed from the archive": the next call must equal
    # the uninterrupted run's candidate at index k.
    stub.reset()
    stub.advance(2)
    assert stub("s", "u") == first[2]


# =========================================================================== #
# loop.py — one gen step archives prompt+source+feedback; reflect-on-BEST;      #
# invalid candidate is skipped not fatal                                       #
# =========================================================================== #
def test_one_step_archives_prompt_source_and_feedback(tmp_path) -> None:
    """A single accepted candidate records the exact sent prompt, raw source, its hash, and the
    feedback block built from THIS candidate's results — the replay payload (C-2)."""
    cfg = {"generations": 1, "candidates_per_gen": 1, "budget": 1, "n_trials": 5}
    archive, llm = _run("distributional", cfg, tmp_path)
    assert len(archive.candidates) == 1
    rec = archive.candidates[0]
    sent_system, sent_user = llm.prompts[0]
    assert rec.prompt == sent_user                       # archived prompt == sent prompt
    assert rec.reward_source == _VALID_REWARD_SRC
    assert rec.reward_hash and len(rec.reward_hash) == 64  # sha256 hex
    assert "Deflated Sharpe" in rec.feedback_block        # block built from this candidate
    assert set(rec.tail_stats.keys()) == set(FROZEN_TAIL_KEYS)
    # F15: budget_spent records the ACTUAL draws (1 here), not the cfg 'budget' plan.
    assert archive.meta["budget_spent"] == 1


def test_reflect_on_best_not_last_seeds_next_generation(tmp_path) -> None:
    """M5: with candidates_per_gen>1 the NEXT generation's reflection prompt is seeded by the
    generation's BEST candidate's feedback block, NOT the last drawn one. The ascending trainer
    makes the LAST candidate the best, so we make the best NOT last by reversing fitness."""
    # Trainer giving DESCENDING levels: candidate 0 is the best within the generation, the last is worst.
    state = {"n": 5}

    def descending_trainer(_env) -> _Policy:
        state["n"] -= 1
        return _Policy(level=float(state["n"]))

    _LeveledEnv._counter = 0
    llm = _CannedLLM(_VALID_REWARD_SRC)
    cfg = {"generations": 2, "candidates_per_gen": 3, "budget": 60, "n_trials": 5}
    archive = loop.run_loop(
        arm="distributional", env_builder=_LeveledEnv, llm=llm,
        agent_trainer=descending_trainer, measurement=ReturnDistribution,
        fitness_fn=held_out_fitness, cfg=cfg, archive_root=tmp_path,
    )
    gen0 = [c for c in archive.candidates if c.generation == 0]
    best_gen0 = max(gen0, key=lambda c: c.val_fitness)
    # The gen-1 prompts (candidates 3,4,5) embed the BEST gen-0 block, not the last gen-0 one.
    gen1_user = llm.prompts[3][1]
    assert best_gen0.feedback_block in gen1_user
    worst_gen0 = min(gen0, key=lambda c: c.val_fitness)
    if worst_gen0.feedback_block != best_gen0.feedback_block:
        assert worst_gen0.feedback_block not in gen1_user


def test_invalid_candidate_skipped_then_valid_one_accepted(tmp_path) -> None:
    """A sandbox-rejected candidate is logged in failures and SKIPPED (not fatal); a later valid
    candidate in the same generation is still accepted. Uses a per-call alternating LLM."""
    bad = "import os\ndef reward(weights, returns, prev_weights, port_ret, info):\n    return 0.0, {}, None\n"

    class _Alternating:
        def __init__(self) -> None:
            self.cfg = {"model": "m"}
            self.prompts: list[tuple[str, str]] = []
            self._i = 0

        def complete(self, system: str, user: str) -> str:
            self.prompts.append((system, user))
            src = bad if self._i == 0 else _VALID_REWARD_SRC
            self._i += 1
            return src

    _LeveledEnv._counter = 0
    llm = _Alternating()
    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 40, "n_trials": 5}
    archive = loop.run_loop(
        arm="scalar", env_builder=_LeveledEnv, llm=llm,
        agent_trainer=_ascending_trainer(), measurement=ReturnDistribution,
        fitness_fn=held_out_fitness, cfg=cfg, archive_root=tmp_path,
    )
    assert len(archive.failures) == 1
    assert "error" in archive.failures[0] and archive.failures[0]["candidate_id"].endswith("-c0")
    assert len(archive.candidates) == 1            # the second (valid) candidate accepted
    assert archive.winner() is not None
    assert archive.meta["accepted"] == 1 and archive.meta["failed"] == 1


def test_loop_extracts_fenced_reward_source(tmp_path) -> None:
    """The loop runs the raw completion through extract_reward_source, so a markdown-FENCED reward
    (what a real Opus/Gemini reply looks like) is salvaged + accepted, not rejected for formatting."""
    fenced = (
        "Here is my reward function:\n```python\n"
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    return float(port_ret), {'r': float(port_ret)}, None\n"
        "```\nHope this helps!"
    )
    archive, _ = _run("scalar", {"generations": 1, "candidates_per_gen": 1, "budget": 10, "n_trials": 3},
                      tmp_path, source=fenced)
    assert len(archive.candidates) == 1
    src = archive.candidates[0].reward_source
    assert src.startswith("def reward(")        # fence + prose stripped
    assert "```" not in src and "Hope this helps" not in src


# =========================================================================== #
# client.py — provider-neutral wiring; clear error on missing key; never logs   #
# the key                                                                      #
# =========================================================================== #
def test_client_reads_model_provider_and_key_env_from_cfg() -> None:
    """LLMClient resolves model/provider/api_key_env from cfg (provider-neutral wiring)."""
    c = LLMClient({"model": "claude-opus-4-8", "provider": "anthropic"})
    assert c.model == "claude-opus-4-8"
    assert c.provider == "anthropic"
    assert c.api_key_env == "ANTHROPIC_API_KEY"  # default for anthropic
    c2 = LLMClient({"model": "gemini-3.5-flash", "provider": "gemini"})
    assert c2.api_key_env == "GEMINI_API_KEY"
    c3 = LLMClient({"model": "m", "provider": "openai", "api_key_env": "MY_CUSTOM_KEY"})
    assert c3.api_key_env == "MY_CUSTOM_KEY"  # explicit override honored


def test_client_with_injected_transport_archives_provenance_no_network() -> None:
    """With an injected FakeTransport the client returns the canned source and appends a
    ProvenanceRecord (model + prompts + response) — no network call."""
    archive: list[ProvenanceRecord] = []
    c = LLMClient(
        {"model": "claude-opus-4-8", "provider": "anthropic"},
        transport=FakeTransport(response=_VALID_REWARD_SRC),
        archive=archive,
    )
    out = c.complete("SYSTEM", "USER")
    assert out == _VALID_REWARD_SRC
    assert len(archive) == 1
    rec = archive[0]
    assert rec.model == "claude-opus-4-8"
    assert rec.system == "SYSTEM" and rec.user == "USER"
    assert rec.response == _VALID_REWARD_SRC


def test_client_missing_key_raises_clear_named_error(monkeypatch) -> None:
    """No injected transport + missing key -> a clear RuntimeError naming the env var, and NO
    network call (the real transport build raises before any SDK/network use)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = LLMClient({"model": "claude-opus-4-8", "provider": "anthropic"})
    with pytest.raises(RuntimeError) as excinfo:
        c.complete("s", "u")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_client_no_model_raises_before_building_real_transport(monkeypatch) -> None:
    """BOUNDARY: an empty model id raises a clear error (a real pinned snapshot is required)
    rather than silently floating to an alias."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-used")
    c = LLMClient({"model": "", "provider": "anthropic"})
    with pytest.raises(RuntimeError, match="model"):
        c.complete("s", "u")


def test_api_key_is_never_logged_or_archived(monkeypatch, caplog) -> None:
    """SECURITY: the API key never appears in the provenance archive nor in any emitted log
    record. Drive the real Anthropic transport with a faked SDK module so no network is used."""
    import logging
    import sys
    import types

    secret = "sk-ant-SUPER-SECRET-KEY-do-not-leak"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    # Fake the anthropic SDK so make_anthropic_transport builds without network.
    class _Msg:
        def __init__(self) -> None:
            self.content = [types.SimpleNamespace(text=_VALID_REWARD_SRC, type="text")]
            self.usage = types.SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            )

    class _Anthropic:
        def __init__(self, **kwargs) -> None:
            self._key = kwargs.get("api_key")
            self.messages = types.SimpleNamespace(create=lambda **kw: _Msg())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

    archive: list[ProvenanceRecord] = []
    c = LLMClient({"model": "claude-opus-4-8", "provider": "anthropic"}, archive=archive)
    with caplog.at_level(logging.DEBUG):
        out = c.complete("SYSTEM PROMPT", "USER PROMPT")
    assert out == _VALID_REWARD_SRC
    # The key is nowhere in the archived provenance.
    for rec in archive:
        assert secret not in rec.model
        assert secret not in rec.system
        assert secret not in rec.user
        assert secret not in rec.response
        assert secret not in str(rec.usage)
    # The key is nowhere in any captured log line.
    assert secret not in caplog.text


class _SwappingTransport:
    """A transport whose provider silently changes the SERVED snapshot mid-run (R71's failure mode)."""

    def __init__(self, served: list[str]) -> None:
        self._served = list(served)
        self._i = 0
        self.last_served_model: str | None = None

    def __call__(self, system: str, user: str) -> str:
        self.last_served_model = self._served[min(self._i, len(self._served) - 1)]
        self._i += 1
        return _VALID_REWARD_SRC


def test_served_model_registry_is_silent_when_stable_and_loud_on_a_mid_run_swap(caplog) -> None:
    """R71 ANCHOR INTEGRITY: the write-up quotes the SERVED snapshot as the reproducibility anchor, so a
    provider that swaps snapshots mid-run silently invalidates that anchor and mixes the candidate set.

    Stable serving must stay silent (one logged snapshot, no warning); a swap must warn LOUDLY exactly
    ONCE per requested id and expose both snapshots for audit."""
    from src.llm.client import reset_served_model_log, served_model_log

    reset_served_model_log()
    try:
        stable = LLMClient(
            {"model": "claude-opus-5", "provider": "anthropic"},
            transport=_SwappingTransport(["claude-opus-5-20260115"] * 3),
        )
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                stable.complete("SYS", "USER")
        assert served_model_log() == {"claude-opus-5": ["claude-opus-5-20260115"]}
        assert "llm_served_model_SWITCHED" not in caplog.text  # stable serving is SILENT

        caplog.clear()
        swapping = LLMClient(
            {"model": "qwen/qwen3-coder", "provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY"},
            transport=_SwappingTransport(["qwen-snapshot-A", "qwen-snapshot-B", "qwen-snapshot-B"]),
        )
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                swapping.complete("SYS", "USER")

        assert caplog.text.count("llm_served_model_SWITCHED") == 1  # loud, but ONCE — never a log flood
        assert "qwen-snapshot-B" in caplog.text and "qwen-snapshot-A" in caplog.text
        assert served_model_log()["qwen/qwen3-coder"] == ["qwen-snapshot-A", "qwen-snapshot-B"]
        # the stable client's entry is untouched: the registry is per-requested-id
        assert served_model_log()["claude-opus-5"] == ["claude-opus-5-20260115"]
    finally:
        reset_served_model_log()
