"""Fast tests for the reward-forensics tool (scripts/inspect_rewards.py).

FAST, no torch, no real training, no LLM: synthetic per-candidate run records across
generations/arms (mirroring the schema written by ``src/llm/loop.py``) drive the three
analyses. Covers the Phase-4.C / §6.1 acceptance criteria:

  - ``per_generation_summary`` keys by (arm, generation) and carries the fitness +
    reward-code complexity + tail-usage trend;
  - ``feedback_responsiveness`` returns a FINITE per-arm correlation and DISTINGUISHES a
    constructed responsive fixture (edits track the fed-back tail delta) from an
    unresponsive one (edits decoupled from the tail delta);
  - ``hacking_taxonomy`` FLAGS a constructed specification-gaming example (tail-referencing
    reward whose held-out fitness collapsed) and a tautology;
  - the markdown report is emitted into a tmp dir (and never touches a run archive).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inspect_rewards as IR  # noqa: E402

# A tail-aware reward source (trips the _TAIL_TERMS interpretability lens).
_TAIL_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    cvar = np.sort(returns)[:3].mean()\n"
    "    return port_ret + 0.5 * cvar, {}, None\n"
)
# A plain proxy with NO risk structure.
_PLAIN_SRC = (
    "def reward(weights, prev_weights, returns, port_ret, info):\n"
    "    return port_ret, {}, None\n"
)


def _tail_stats(scale: float) -> dict:
    """A frozen-shape tail-stat dict; ``scale`` shifts every level (drives the delta)."""
    return {
        "cvar_01": -0.080 * scale,
        "cvar_05": -0.040 * scale,
        "cvar_10": -0.028 * scale,
        "cvar_25": -0.015 * scale,
        "left_tail_mass": 0.05 * scale,
        "robust_skew": -0.30 * scale,
    }


def _record(
    arm: str,
    gen: int,
    cidx: int,
    *,
    source: str,
    fitness: float,
    tail_scale: float | None,
) -> dict:
    """A per-candidate record mirroring the loop's write_run payload (no torch)."""
    metrics: dict = {"val_fitness": float(fitness)}
    feedback_block = f"Your previous reward scored: {fitness:.2f} (validation Deflated Sharpe)."
    if tail_scale is not None:
        ts = _tail_stats(tail_scale)
        metrics["tail_stats"] = ts
        feedback_block += (
            "\nRealized-return tail diagnostics (training period):"
            f"\n  CVaR 5%: {ts['cvar_05']:+.3f}"
            f"\n  CVaR 10%: {ts['cvar_10']:+.3f}"
            f"\n  CVaR 25%: {ts['cvar_25']:+.3f}"
            f"\n  CVaR 1%: {ts['cvar_01']:+.3f}  (high-variance estimate)"
            f"\n  left-tail mass: {ts['left_tail_mass']:+.3f}"
            f"\n  left-tail skew: {ts['robust_skew']:+.3f}"
        )
    return {
        "run_id": f"{arm}-g{gen}-c{cidx}",
        "arm": arm,
        "seed": 0,
        "fold": 0,
        "candidate_id": f"{arm}-g{gen}-c{cidx}",
        "generation": gen,
        "reward_source": source,
        "reward_source_hash": f"h-{arm}-{gen}-{cidx}",
        "feedback_block": feedback_block,
        "metrics": metrics,
        "wall_clock": 0.0,
        "env_fingerprint": "test",
    }


def _morph_source(n_lines: int) -> str:
    """A reward source whose textual SIZE is set ONLY by ``n_lines`` (a controllable edit lever).

    Two sources with the SAME ``n_lines`` are byte-identical (edit distance 0); a larger gap in
    ``n_lines`` between successive candidates produces a proportionally larger gen N->N+1 source
    change. The body lines are fixed templates (no step-index in their names) so the edit
    magnitude is a clean function of the line-count delta — the lever the correlation reads.
    """
    body = [
        "def reward(weights, returns, prev_weights, port_ret, info):",
        "    cvar = np.sort(returns)[:5].mean()",
    ]
    for k in range(n_lines):
        body.append(f"    term_{k:03d} = np.quantile(returns, 0.05) * 1.0")
    body.append("    return port_ret + cvar, {}, None")
    return "\n".join(body) + "\n"


# --------------------------------------------------------------------------- #
# per_generation_summary                                                       #
# --------------------------------------------------------------------------- #
def test_per_generation_summary_keys_by_arm_and_generation() -> None:
    records = [
        _record("distributional", 0, 0, source=_PLAIN_SRC, fitness=0.20, tail_scale=1.0),
        _record("distributional", 1, 0, source=_TAIL_SRC, fitness=0.55, tail_scale=1.2),
        _record("scalar", 0, 0, source=_PLAIN_SRC, fitness=0.30, tail_scale=None),
    ]
    summary = IR.per_generation_summary(records)

    assert ("distributional", 0) in summary
    assert ("distributional", 1) in summary
    assert ("scalar", 0) in summary
    # gen-0 distributional used the plain proxy; gen-1 used a tail-aware reward.
    assert summary[("distributional", 0)]["frac_uses_tail"] == 0.0
    assert summary[("distributional", 1)]["frac_uses_tail"] == 1.0
    assert summary[("distributional", 1)]["best_fitness"] == np.float64(0.55)
    # complexity trend is populated and finite.
    assert summary[("distributional", 1)]["mean_loc"] > 0
    assert np.isfinite(summary[("distributional", 0)]["mean_ops"])


def test_per_generation_summary_accepts_arm_mapping() -> None:
    """It accepts the {arm: [records]} mapping from load_arms, not just a flat list."""
    mapping = {
        "distributional": [
            _record("distributional", 0, 0, source=_TAIL_SRC, fitness=0.4, tail_scale=1.0),
        ],
    }
    summary = IR.per_generation_summary(mapping)
    assert summary[("distributional", 0)]["n"] == 1


def test_per_generation_mean_fitness_is_nan_safe() -> None:
    """A record missing val_fitness must not poison the mean (nan-skipping)."""
    recs = [
        _record("scalar", 0, 0, source=_PLAIN_SRC, fitness=0.5, tail_scale=None),
        _record("scalar", 0, 1, source=_PLAIN_SRC, fitness=0.3, tail_scale=None),
    ]
    recs[1]["metrics"].pop("val_fitness")
    s = IR.per_generation_summary(recs)[("scalar", 0)]
    assert s["mean_fitness"] == np.float64(0.5)  # the finite one only


# --------------------------------------------------------------------------- #
# feedback_responsiveness — the H2 core (responsive vs unresponsive)           #
# --------------------------------------------------------------------------- #
# Each plan is (tail_scale, n_source_lines) per generation. The fed-back tail DELTA across a
# step is proportional to |Δ tail_scale|; the source-EDIT magnitude grows with |Δ n_source_lines|.
# RESPONSIVE: the two move together (big distribution shift -> big code rewrite).
_RESPONSIVE_PLAN = [(1.0, 2), (1.5, 8), (1.6, 10), (3.0, 40), (3.1, 42)]
#   step Δscale: 0.5, 0.1, 1.4, 0.1   |   step Δlines: 6, 2, 30, 2   -> strongly co-monotone.
# UNRESPONSIVE: the biggest distribution shifts get the SMALLEST edits (anti-correlated).
_UNRESPONSIVE_PLAN = [(1.0, 2), (3.0, 4), (3.1, 44), (1.1, 46), (1.0, 8)]
#   step Δscale: 2.0, 0.1, 2.0, 0.1   |   step Δlines: 2, 40, 2, 38   -> anti-correlated.


def _arm_from_plan(arm: str, plan: list[tuple[float, int]]) -> list[dict]:
    return [
        _record(arm, gen, 0, source=_morph_source(n), fitness=0.3 + 0.02 * gen, tail_scale=scale)
        for gen, (scale, n) in enumerate(plan)
    ]


def _responsive_arm(arm: str) -> list[dict]:
    """Edit magnitude TRACKS the fed-back tail delta -> high responsiveness correlation."""
    return _arm_from_plan(arm, _RESPONSIVE_PLAN)


def _unresponsive_arm(arm: str) -> list[dict]:
    """Edit magnitude is ANTI-correlated with the fed-back tail delta -> negative score."""
    return _arm_from_plan(arm, _UNRESPONSIVE_PLAN)


def test_feedback_responsiveness_returns_finite_score() -> None:
    out = IR.feedback_responsiveness(_responsive_arm("distributional"))
    e = out["distributional"]
    assert e["score"] is not None
    assert np.isfinite(e["score"])
    assert -1.0 <= e["score"] <= 1.0
    assert e["n_steps"] == 4  # 5 candidates -> 4 successive steps


def test_feedback_responsiveness_distinguishes_responsive_from_unresponsive() -> None:
    resp = IR.feedback_responsiveness(_responsive_arm("distributional"))["distributional"]
    unresp = IR.feedback_responsiveness(_unresponsive_arm("distributional"))["distributional"]
    # The constructed responsive arm tracks the distribution; the unresponsive one does not.
    assert resp["score"] > 0.5
    assert resp["score"] > unresp["score"]
    assert unresp["score"] < 0.0


def test_feedback_responsiveness_none_for_scalar_arm() -> None:
    """An arm with no tail feedback (scalar) reports score=None — nothing to track."""
    recs = [
        _record("scalar", g, 0, source=_morph_source(g + 1), fitness=0.3, tail_scale=None)
        for g in range(3)
    ]
    e = IR.feedback_responsiveness(recs)["scalar"]
    assert e["score"] is None
    assert "not fed a tail" in e["reason"]


def test_feedback_responsiveness_ignores_measured_tail_when_not_fed() -> None:
    """REGRESSION (real-data confound): a scalar arm whose candidates carry a MEASURED
    ``metrics['tail_stats']`` (the off-critic measurement is archived for EVERY arm) but whose
    feedback shows only a scalar — the designer was never shown the tail — must score ``None``.
    Pre-fix, the gate read the MEASURED vector and scored this arm a number instead of ``None`` (here
    0.0 — the fixture's constant scale gives a constant delta; on real varying-tail data a genuinely
    SPURIOUS correlation, as the prototype showed). The synthetic fixtures coupled ``tail_stats`` to the
    rendered tail text and so never exercised this case (it surfaced only on the real prototype archive).
    """
    recs = []
    for g in range(4):
        r = _record("scalar", g, 0, source=_morph_source(g + 1), fitness=0.3, tail_scale=1.0)
        assert "tail_stats" in r["metrics"]  # the confound precondition: a measured tail is present
        # ...but strip the rendered tail block: a real scalar arm sees ONLY the scalar line.
        r["feedback_block"] = "Your previous reward scored: 0.30 (validation Deflated Sharpe)."
        recs.append(r)
    e = IR.feedback_responsiveness(recs)["scalar"]
    assert e["score"] is None
    assert "not fed a tail" in e["reason"]


def test_feedback_responsiveness_parses_tail_from_feedback_block() -> None:
    """When metrics['tail_stats'] is absent, the tail vector is recovered from the block text."""
    recs = _responsive_arm("distributional")
    for r in recs:
        r["metrics"].pop("tail_stats")  # force the feedback_block text parse path
    e = IR.feedback_responsiveness(recs)["distributional"]
    assert e["score"] is not None and np.isfinite(e["score"])
    assert e["n_steps"] == 4


# --------------------------------------------------------------------------- #
# hacking_taxonomy                                                             #
# --------------------------------------------------------------------------- #
def test_hacking_taxonomy_flags_specification_gaming() -> None:
    """A tail-referencing reward whose held-out fitness COLLAPSED is specification gaming."""
    records = [
        # gaming: references CVaR (tail) yet fitness <= 0 -> proxy up, true objective down.
        _record("distributional", 0, 0, source=_TAIL_SRC, fitness=-0.12, tail_scale=1.0),
        # healthy: tail-aware AND positive fitness -> no flag.
        _record("distributional", 1, 0, source=_TAIL_SRC, fitness=0.40, tail_scale=1.0),
    ]
    tax = IR.hacking_taxonomy(records)
    by_id = {t["candidate_id"]: t for t in tax["candidates"]}
    assert "specification_gaming" in by_id["distributional-g0-c0"]["flags"]
    assert by_id["distributional-g1-c0"]["flags"] == []
    assert tax["counts"]["specification_gaming"] == 1
    assert tax["n_flagged"] == 1


def test_hacking_taxonomy_flags_tautology_and_proxy() -> None:
    """`return port_ret` is a tautology; a no-tail collapsed reward is proxy_no_tail."""
    records = [
        _record("scalar", 0, 0, source=_PLAIN_SRC, fitness=-0.05, tail_scale=None),
    ]
    tax = IR.hacking_taxonomy(records)
    flags = tax["candidates"][0]["flags"]
    assert "tautology" in flags
    assert "proxy_no_tail" in flags


def test_hacking_taxonomy_clean_archive_flags_nothing() -> None:
    records = [
        _record("distributional", 0, 0, source=_TAIL_SRC, fitness=0.5, tail_scale=1.0),
    ]
    tax = IR.hacking_taxonomy(records)
    assert tax["n_flagged"] == 0
    assert tax["counts"] == {
        "unbounded_magnitude": 0, "specification_gaming": 0, "proxy_no_tail": 0, "tautology": 0,
    }


# --------------------------------------------------------------------------- #
# reward_magnitude_audit (MEASURED — executes the reward; FIX #4)              #
# --------------------------------------------------------------------------- #
# The EXACT verified prototype pathology (scalar-g5-c3): a per-step Sharpe with the std floored at 1e-8,
# so a single early return becomes mu/1e-8 ~ 1e4 — the critic-explosion source.
_EXPLODING_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    st = info.get('reward_state') or {'hist': []}\n"
    "    h = st['hist']; h.append(float(port_ret))\n"
    "    arr = np.array(h, dtype=np.float64); n = len(arr)\n"
    "    if n >= 5:\n"
    "        mu = float(np.mean(arr)); sigma = float(np.std(arr, ddof=1)) + 1e-8\n"
    "        sharpe = mu / sigma\n"
    "    else:\n"
    "        sharpe = (float(np.mean(arr)) / 1e-8 * 0.1) if n > 0 else 0.0\n"
    "    return float(sharpe), {'sharpe': float(sharpe)}, {'hist': h}\n"
)


def test_reward_magnitude_audit_measures_and_flags_explosion() -> None:
    """The audit EXECUTES each reward and flags the variance-floor one whose realized |total| > bound."""
    records = [
        _record("scalar", 5, 3, source=_EXPLODING_SRC, fitness=0.0019, tail_scale=None),
        _record("scalar", 0, 0, source=_PLAIN_SRC, fitness=0.01, tail_scale=None),   # benign port_ret
    ]
    audit = IR.reward_magnitude_audit(records, bound=1.0e4)
    by_id = {d["candidate_id"]: d for d in audit["candidates"]}

    bad = by_id["scalar-g5-c3"]
    assert bad["over_bound"] is True
    assert bad["max_abs_total"] is not None and bad["max_abs_total"] > 1.0e4  # mu/1e-8 ~ 1e4+

    good = by_id["scalar-g0-c0"]
    assert good["over_bound"] is False
    assert good["max_abs_total"] is not None and good["max_abs_total"] < 1.0  # |port_ret| << 1

    assert audit["n_over_bound"] == 1
    # sorted worst-first: the exploding candidate leads the table.
    assert audit["candidates"][0]["candidate_id"] == "scalar-g5-c3"


def test_reward_magnitude_audit_report_only_does_not_mutate_records() -> None:
    """The audit is READ-ONLY — it never writes back to or alters the input records."""
    rec = _record("scalar", 5, 3, source=_EXPLODING_SRC, fitness=0.0019, tail_scale=None)
    snapshot = dict(rec)
    IR.reward_magnitude_audit([rec], bound=1.0e4)
    assert rec == snapshot  # untouched


def test_reward_magnitude_audit_handles_empty_and_unconstructable_source() -> None:
    """An empty / non-constructable reward is reported ok=False, never raised, and is not over-bound."""
    records = [
        _record("scalar", 0, 0, source="", fitness=0.0, tail_scale=None),
        _record("scalar", 0, 1, source="def reward(:\n  pass\n", fitness=0.0, tail_scale=None),  # syntax error
    ]
    audit = IR.reward_magnitude_audit(records)
    assert audit["n_over_bound"] == 0
    assert all(d["ok"] is False for d in audit["candidates"])


# --------------------------------------------------------------------------- #
# hacking_taxonomy — tautology FALSE-POSITIVE fix (verified on the real archive) #
# --------------------------------------------------------------------------- #
# An INTERMEDIATE assignment named ``*_reward = port_ret`` that the function later combines into a
# SHAPED total is NOT a tautology (the real prototype's distributional-g4-c0 / scalar_cvar5-g3-c2).
_INTERMEDIATE_VAR_SHAPED_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    core_reward = port_ret\n"                       # intermediate component, NOT the return
    "    cvar = np.sort(returns)[:3].mean()\n"
    "    total = core_reward + 0.3 * min(cvar, 0.0)\n"   # the ACTUAL reward is multi-term
    "    return float(total), {}, None\n"
)
# A REAL tautology: the function returns port_ret WHOLE (the degenerate proxy).
_REAL_TAUTOLOGY_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    return port_ret, {}, None\n"
)


def test_hacking_taxonomy_intermediate_reward_var_is_not_tautology() -> None:
    """REGRESSION: ``core_reward = port_ret`` (an intermediate term of a SHAPED reward) is NOT flagged.

    Pre-fix, the bare ``reward = port_ret`` arm matched any line ENDING in ``…reward = port_ret``, so
    the prototype's distributional-g4-c0 (``core_reward = port_ret`` then a 6-term ``total``) was
    mis-flagged a tautology. The statement-anchored pattern must no longer trip on the suffix.
    """
    tax = IR.hacking_taxonomy(
        [_record("distributional", 0, 0, source=_INTERMEDIATE_VAR_SHAPED_SRC, fitness=0.4, tail_scale=1.0)]
    )
    assert "tautology" not in tax["candidates"][0]["flags"]


def test_hacking_taxonomy_real_tautology_still_flagged() -> None:
    """The fix must NOT under-fire: a genuine ``return port_ret`` whole-proxy is still a tautology."""
    tax = IR.hacking_taxonomy(
        [_record("scalar", 0, 0, source=_REAL_TAUTOLOGY_SRC, fitness=0.2, tail_scale=None)]
    )
    assert "tautology" in tax["candidates"][0]["flags"]


# --------------------------------------------------------------------------- #
# hacking_taxonomy — variance-floor / UNBOUNDED-MAGNITUDE class (the real failure mode) #
# --------------------------------------------------------------------------- #
# The critic-explosion source: a return divided by a FLOORED volatility, unbounded as variance -> 0.
# CRUCIALLY this carries POSITIVE fitness on the prototype, so the collapse-gated specification_gaming
# check MISSED it — the new class flags it on the code SHAPE, independent of fitness.
_VARIANCE_FLOOR_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    std = float(np.std(returns)) \n"
    "    ret_component = port_ret / (std + 1e-8)\n"      # UNBOUNDED above as std -> 0
    "    return float(ret_component), {}, None\n"
)
_UNFLOORED_DIVIDE_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    ema_std = float(np.std(returns))\n"
    "    return float(port_ret / ema_std), {}, None\n"   # the un-floored divide (diverges hardest)
)
# A benign normalization that DIVIDES but by a non-volatility quantity -> must NOT trip the class.
_BENIGN_NORMALIZE_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    hhi = float(np.sum(weights ** 2))\n"
    "    n = len(weights)\n"
    "    hhi_norm = (hhi - 1.0 / n) / (1.0 - 1.0 / n + 1e-8)\n"   # concentration ratio, bounded
    "    return float(port_ret - 0.05 * hhi_norm), {}, None\n"
)


def test_hacking_taxonomy_flags_variance_floor_with_positive_fitness() -> None:
    """A ``port_ret / (std + eps)`` reward is flagged unbounded_magnitude EVEN with positive fitness.

    This is the class the old collapse-gated check missed: on the prototype the worst offenders carried
    POSITIVE val_fitness (e.g. distributional-g1-c2 fit=+0.0016) yet are the critic-explosion source.
    """
    rec = _record("distributional", 3, 0, source=_VARIANCE_FLOOR_SRC, fitness=0.30, tail_scale=1.0)
    tax = IR.hacking_taxonomy([rec])
    t = tax["candidates"][0]
    assert "unbounded_magnitude" in t["flags"]
    assert tax["counts"]["unbounded_magnitude"] == 1
    # The flag NAMES the offending divide so the report can quote it.
    assert any("port_ret / (std + 1e-8)" in term for term in t["variance_floor_terms"])
    # ...and it is NOT a fitness-collapse flag (fitness was positive).
    assert "specification_gaming" not in t["flags"] and "proxy_no_tail" not in t["flags"]


def test_hacking_taxonomy_flags_unfloored_return_over_vol_divide() -> None:
    """The un-floored ``port_ret / ema_std`` (no +eps) is also flagged (it diverges hardest)."""
    tax = IR.hacking_taxonomy(
        [_record("distributional", 7, 0, source=_UNFLOORED_DIVIDE_SRC, fitness=0.5, tail_scale=1.0)]
    )
    assert "unbounded_magnitude" in tax["candidates"][0]["flags"]


def test_hacking_taxonomy_benign_normalization_not_flagged_unbounded() -> None:
    """A divide by a NON-volatility quantity (HHI concentration ratio) must NOT trip the class."""
    tax = IR.hacking_taxonomy(
        [_record("distributional", 0, 0, source=_BENIGN_NORMALIZE_SRC, fitness=0.4, tail_scale=1.0)]
    )
    t = tax["candidates"][0]
    assert "unbounded_magnitude" not in t["flags"]
    assert t["variance_floor_terms"] == []


def test_variance_floor_terms_ignores_comment_lines() -> None:
    """A return/std divide inside a ``# comment`` must NOT be detected (only executable code counts)."""
    assert IR._variance_floor_terms("    # port_ret / std is the classic gaming shape") == []
    assert IR._variance_floor_terms("    x = port_ret / (std + 1e-8)  # trailing note") != []


# --------------------------------------------------------------------------- #
# report emission                                                              #
# --------------------------------------------------------------------------- #
def test_write_report_emits_markdown_to_out_dir(tmp_path) -> None:
    records = (
        _responsive_arm("distributional")
        + [_record("scalar", 0, 0, source=_PLAIN_SRC, fitness=-0.05, tail_scale=None)]
    )
    result = {
        "arms": ["distributional", "scalar"],
        "per_generation": IR.per_generation_summary(records),
        "responsiveness": IR.feedback_responsiveness(records),
        "hacking": IR.hacking_taxonomy(records),
    }
    out_dir = tmp_path / "tables"
    report = IR.write_report(result, out_dir)

    assert report.exists()
    assert report.name == "reward_forensics.md"
    text = report.read_text(encoding="utf-8")
    assert "Reward forensics" in text
    assert "responsiveness" in text.lower()
    assert "taxonomy" in text.lower()
    # The JSON sidecar is emitted with stringified per-generation keys.
    assert (out_dir / "reward_forensics.json").exists()


def test_report_headlines_responsiveness_first(tmp_path) -> None:
    """Responsiveness is the HEADLINE forensics signal: its section precedes the taxonomy + per-gen ones."""
    records = (
        _responsive_arm("distributional")
        + [_record("scalar", 0, 0, source=_PLAIN_SRC, fitness=-0.05, tail_scale=None)]
    )
    result = {
        "arms": ["distributional", "scalar"],
        "per_generation": IR.per_generation_summary(records),
        "responsiveness": IR.feedback_responsiveness(records),
        "hacking": IR.hacking_taxonomy(records),
    }
    text = IR.write_report(result, tmp_path / "tables").read_text(encoding="utf-8")
    i_resp = text.index("HEADLINE")
    i_tax = text.lower().index("reward-hacking taxonomy")
    i_pergen = text.index("Per-generation summary")
    assert i_resp < i_tax < i_pergen  # responsiveness headlines, taxonomy second, per-gen supporting
    # The taxonomy section advertises the new unbounded_magnitude class + count.
    assert "unbounded_magnitude" in text


def test_report_taxonomy_table_lists_only_flagged_with_unbounded_terms(tmp_path) -> None:
    """The taxonomy table lists flagged candidates and shows the offending unbounded divide term."""
    records = [
        _record("distributional", 0, 0, source=_VARIANCE_FLOOR_SRC, fitness=0.3, tail_scale=1.0),
        _record("distributional", 1, 0, source=_TAIL_SRC, fitness=0.5, tail_scale=1.0),  # clean, unlisted
    ]
    result = {
        "arms": ["distributional"],
        "per_generation": IR.per_generation_summary(records),
        "responsiveness": IR.feedback_responsiveness(records),
        "hacking": IR.hacking_taxonomy(records),
    }
    text = IR.write_report(result, tmp_path / "tables").read_text(encoding="utf-8")
    assert "unbounded_magnitude" in text
    assert "port_ret / (std + 1e-8)" in text  # the offending term is quoted in the table


def test_inspect_end_to_end_over_written_archive(tmp_path) -> None:
    """End-to-end: write real per-arm record.json archives, then inspect() reads them back.

    Exercises the load_arms archive walk (audit C-1) WITHOUT touching the live outputs/
    tree and confirms the archive is read-only (no files added under the run root).
    """
    from src.io.results import write_run

    root = tmp_path / "runs"
    for r in _responsive_arm("distributional"):
        write_run(r, root / "distributional")
    for g in range(2):
        write_run(
            _record("scalar", g, 0, source=_PLAIN_SRC, fitness=0.3, tail_scale=None),
            root / "scalar",
        )

    before = sorted(p.name for p in (root / "distributional").iterdir())
    result = IR.inspect(root)
    after = sorted(p.name for p in (root / "distributional").iterdir())

    assert result["arms"] == ["distributional", "scalar"]
    assert ("distributional", 0) in result["per_generation"]
    assert result["responsiveness"]["distributional"]["score"] is not None
    assert result["responsiveness"]["scalar"]["score"] is None
    assert "program_differential" in result  # T3.1 wired into inspect()
    assert before == after  # inspect() never wrote into the archive


# --------------------------------------------------------------------------- #
# reward_program_differential (T3.1 — theory->code mechanism loop)             #
# --------------------------------------------------------------------------- #
# A tail-RICH distributional-style program: CVaR + drawdown + turnover + Herfindahl + a declared
# components dict + a hard-coded 5% tail level. This is the shape of the real prototype winners.
_TAIL_RICH_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    st = info.get('reward_state') or {}\n"
    "    hist = list(st.get('hist', [])); hist.append(port_ret)\n"
    "    arr = np.array(hist)\n"
    "    thr = np.percentile(arr, 5)\n"               # quantile_tail + level 0.05
    "    cvar = float(np.mean(arr[arr <= thr])) if arr.size else 0.0\n"  # cvar construct
    "    std = max(float(np.std(arr)), 1e-8)\n"        # rolling_vol
    "    peak = max(st.get('peak', 1.0), 1.0)\n"
    "    drawdown = (peak - 1.0) / peak\n"             # drawdown
    "    turnover = float(np.sum(np.abs(weights - prev_weights)))\n"  # turnover
    "    hhi = float(np.sum(weights[:-1] ** 2))\n"     # herfindahl
    "    total = port_ret / std - 3.0 * abs(cvar) - 2.0 * drawdown - 0.3 * turnover - 0.1 * hhi\n"
    "    components = {'cvar_penalty': -3.0 * abs(cvar), 'turnover_penalty': -0.3 * turnover,\n"
    "                  'drawdown': drawdown}\n"
    "    return float(total), components, {'hist': hist, 'peak': peak}\n"
)
# A tail-POOR scalar-style program: a plain risk-adjusted return, NO CVaR / quantile / drawdown.
_TAIL_POOR_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    st = info.get('reward_state') or {}\n"
    "    n = st.get('n', 0) + 1; mean = st.get('mean', 0.0) + (port_ret - st.get('mean', 0.0)) / n\n"
    "    std = max(abs(mean), 1e-8)\n"                 # rolling_vol only
    "    total = mean / std\n"                          # online_sharpe, no tail
    "    components = {'sharpe': total}\n"
    "    return float(total), components, {'n': n, 'mean': mean}\n"
)


def test_construct_prevalence_detects_tail_and_generic_constructs() -> None:
    """The construct probe finds CVaR/quantile/drawdown/turnover/HHI in a tail-rich source."""
    p = IR._construct_prevalence(_TAIL_RICH_SRC)
    assert p["cvar"] and p["quantile_tail"] and p["drawdown"]
    assert p["turnover"] and p["herfindahl"] and p["rolling_vol"]
    # The tail-poor source has NO tail constructs.
    q = IR._construct_prevalence(_TAIL_POOR_SRC)
    assert not q["cvar"] and not q["quantile_tail"] and not q["drawdown"]
    assert q["rolling_vol"] and q["online_sharpe"]


def test_construct_prevalence_ignores_comment_lines() -> None:
    """A construct named ONLY in a comment does not count as shaped code."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    # this would be a cvar / drawdown penalty if implemented\n"
        "    return port_ret, {}, None\n"
    )
    p = IR._construct_prevalence(src)
    assert not p["cvar"] and not p["drawdown"]


def test_cvar_levels_referenced_recovers_percent_and_named_levels() -> None:
    """A percentile(arr, 5) and a named cvar_05 both map to the 0.05 tail probability; uppers excluded."""
    assert IR._cvar_levels_referenced("x = np.percentile(arr, 5)") == [0.05]
    assert IR._cvar_levels_referenced("y = np.quantile(arr, 0.01)") == [0.01]
    assert IR._cvar_levels_referenced("z = cvar_05 + cvar_01") == [0.01, 0.05]
    # An upper-tail percentile and the median are NOT tail levels.
    assert IR._cvar_levels_referenced("a = np.percentile(arr, 95); b = np.quantile(arr, 0.5)") == []


def test_declared_components_parses_keys_not_runtime_values() -> None:
    """The declared component KEYS are parsed from the source dict literal (no runtime values needed)."""
    keys = IR._declared_components(_TAIL_RICH_SRC)
    assert keys == ["cvar_penalty", "drawdown", "turnover_penalty"]
    # A reward with an empty components dict declares none.
    assert IR._declared_components(_PLAIN_SRC) == []


def test_coefficient_magnitudes_keeps_weights_drops_plumbing() -> None:
    """Shaping weights (3.0, 0.3, 0.1) are kept; eps floors / 252 / trivial 0/1/2 are dropped."""
    coefs = IR._coefficient_magnitudes(_TAIL_RICH_SRC)
    assert 3.0 in coefs and 0.3 in coefs and 0.1 in coefs
    assert 1e-8 not in coefs            # eps floor dropped (tiny)
    # 252 annualisation and trivial 0/1/2 are structural, not shaping coefficients.
    src = "def reward(w, r, p, pr, i):\n    x = pr * 252.0 + 1.0 - 2.0 + 0.0\n    return x, {}, None\n"
    assert IR._coefficient_magnitudes(src) == []


def test_reward_program_differential_quantifies_cross_arm_tail_gap() -> None:
    """Distributional (tail-rich) references tail constructs MORE than a tail-poor scalar arm."""
    records = (
        [_record("distributional", g, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0) for g in range(3)]
        + [_record("scalar", g, 0, source=_TAIL_POOR_SRC, fitness=0.3, tail_scale=None) for g in range(3)]
    )
    out = IR.reward_program_differential(records)
    assert out["n_total"] == 6
    da = out["per_arm"]["distributional"]
    sa = out["per_arm"]["scalar"]
    # Distributional shapes a tail level on every program; the scalar arm on none.
    assert da["n_programs_with_cvar_level"] == 3
    assert sa["n_programs_with_cvar_level"] == 0
    assert da["tail_construct_rate"] > sa["tail_construct_rate"]
    # The differential block reports the positive distributional-minus-scalar tail gap.
    diff = out["differential"]["scalar"]
    assert diff["tail_rate_delta"] > 0.0
    assert diff["distributional_references_tail_more"] is True
    # Declared components surfaced for the distributional arm.
    assert any(k == "cvar_penalty" for k, _v in da["top_declared_components"])


def test_reward_program_differential_excludes_search_arms_from_llm_differential() -> None:
    """V11 (2026-06-26): random_search / bayes_opt are fixed-skeleton human templates (constant tail count,
    only coefficients vary), so they are EXCLUDED from the LLM tail-construct differential and reported in
    ``search_templates`` (fixed_template=True). The LLM-comparator delta is unchanged by their presence."""
    # A fixed-skeleton search template: same tail constructs (cvar + quantile via percentile) every program,
    # only the shaping COEFFICIENT varies (so distinct source bytes, but a constant tail-construct count).
    def _search_src(coef: float) -> str:
        return (
            "def reward(weights, returns, prev_weights, port_ret, info):\n"
            "    st = info.get('reward_state') or {}\n"
            "    hist = list(st.get('hist', [])); hist.append(port_ret)\n"
            "    arr = np.array(hist)\n"
            "    thr = np.percentile(arr, 5)\n"
            "    cvar = float(np.mean(arr[arr <= thr])) if arr.size else 0.0\n"
            f"    total = port_ret - {coef:.4f} * abs(cvar)\n"
            "    return float(total), {}, {'hist': hist}\n"
        )
    records = (
        [_record("distributional", g, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0) for g in range(3)]
        + [_record("scalar", g, 0, source=_TAIL_POOR_SRC, fitness=0.3, tail_scale=None) for g in range(3)]
        # search arms: 4 DISTINCT sources each (coefficients differ) but a FIXED tail skeleton.
        + [_record("random_search", g, 0, source=_search_src(0.1 * (g + 1)), fitness=0.2, tail_scale=None) for g in range(4)]
        + [_record("bayes_opt", g, 0, source=_search_src(0.5 + g), fitness=0.2, tail_scale=None) for g in range(4)]
    )
    out = IR.reward_program_differential(records)
    # The search arms are NOT in the cross-arm LLM differential (the category-error fix)...
    assert "random_search" not in out["differential"]
    assert "bayes_opt" not in out["differential"]
    assert set(out["differential"]) <= {"scalar", "placebo", "scalar_cvar5"}
    assert out["llm_comparators"] == ["scalar", "placebo", "scalar_cvar5"]
    # ...but ARE reported in search_templates, flagged fixed-skeleton (constant tail count, distinct sources).
    st = out["search_templates"]
    assert set(st) == {"random_search", "bayes_opt"}
    for arm in ("random_search", "bayes_opt"):
        assert st[arm]["fixed_template"] is True          # zero tail-count variance
        assert st[arm]["tail_count_variance"] == 0.0
        assert st[arm]["n_distinct_sources"] == 4         # coefficients varied -> distinct sources
        assert st[arm]["tail_construct_rate"] > 0.0
    # The LLM-comparator delta is identical whether or not the search arms are present (no contamination).
    base = IR.reward_program_differential(
        [_record("distributional", g, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0) for g in range(3)]
        + [_record("scalar", g, 0, source=_TAIL_POOR_SRC, fitness=0.3, tail_scale=None) for g in range(3)]
    )
    assert out["differential"]["scalar"]["tail_rate_delta"] == base["differential"]["scalar"]["tail_rate_delta"]


def test_reward_program_differential_flags_components_not_archived() -> None:
    """When NO record carries a per-step components field, components_archived is False (declared-only)."""
    records = [_record("distributional", 0, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0)]
    assert IR.reward_program_differential(records)["components_archived"] is False
    # ...but if a record DID archive per-step components, the flag flips True.
    r = _record("distributional", 0, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0)
    r["metrics"]["components"] = {"cvar_penalty": -0.01}
    assert IR.reward_program_differential([r])["components_archived"] is True


def test_reward_program_differential_markdown_renders_section_and_scope(tmp_path) -> None:
    """The differential renders a section with the scope caveat when components are not archived."""
    records = (
        [_record("distributional", 0, 0, source=_TAIL_RICH_SRC, fitness=0.3, tail_scale=1.0)]
        + [_record("scalar", 0, 0, source=_TAIL_POOR_SRC, fitness=0.3, tail_scale=None)]
        + [_record("random_search", 0, 0, source=_TAIL_RICH_SRC, fitness=0.2, tail_scale=None)]
    )
    result = {
        "arms": ["distributional", "scalar", "random_search"],
        "per_generation": IR.per_generation_summary(records),
        "responsiveness": IR.feedback_responsiveness(records),
        "hacking": IR.hacking_taxonomy(records),
        "reward_magnitude": IR.reward_magnitude_audit(records),
        "program_differential": IR.reward_program_differential(records),
    }
    text = IR.write_report(result, tmp_path / "tables").read_text(encoding="utf-8")
    assert "Reward-program differential" in text
    assert "theory" in text.lower() and "mechanism loop" in text.lower()
    assert "components" in text and "NOT in the archive" in text  # the no-fabrication scope note
    assert "Cross-arm tail differential" in text
    # V11: the search baseline is rendered in its OWN fixed-skeleton section, flagged out of the differential.
    assert "Search baselines" in text and "fixed-skeleton" in text.lower()
    assert "NOT in the LLM tail differential" in text
