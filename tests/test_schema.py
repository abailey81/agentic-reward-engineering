"""Tests for src/feedback/schema.py — matched-structure feedback blocks (F.3, B.5)."""

from __future__ import annotations

import pytest

from src.feedback.schema import _DIST_FIELDS, _fmt, block_fields, build_block

ARMS = ["distributional", "scalar", "placebo", "scalar_cvar5"]

TAIL_STATS = {
    "cvar_01": -0.067,
    "cvar_05": -0.041,
    "cvar_10": -0.029,
    "cvar_25": -0.016,
    "left_tail_mass": 0.061,
    "robust_skew": -0.38,
}
METRIC = 0.83


def test_all_blocks_build_without_error() -> None:
    """All four feedback arms render blocks without error (B.5)."""
    for arm in ARMS:
        block = build_block(arm, METRIC, TAIL_STATS)
        assert isinstance(block, str) and len(block) > 0


def test_placebo_matches_distributional_field_count_and_length() -> None:
    """Placebo block matches distributional in line-count and length (inert content)."""
    dist = build_block("distributional", METRIC, TAIL_STATS)
    placebo = build_block("placebo", METRIC, None)

    assert len(dist.splitlines()) == len(placebo.splitlines())
    assert len(block_fields("placebo")) == len(block_fields("distributional"))

    # Char-length within +/-15%.
    ratio = len(placebo) / len(dist)
    assert 0.85 <= ratio <= 1.15


def test_scalar_cvar5_is_scalar_plus_exactly_one_field() -> None:
    """scalar_cvar5 block == scalar block + exactly the CVaR-5% line."""
    scalar = build_block("scalar", METRIC, None)
    cvar5 = build_block("scalar_cvar5", METRIC, TAIL_STATS)

    scalar_lines = scalar.splitlines()
    cvar5_lines = cvar5.splitlines()
    assert len(cvar5_lines) == len(scalar_lines) + 1
    assert cvar5_lines[:len(scalar_lines)] == scalar_lines
    assert "CVaR 5%" in cvar5_lines[-1]

    assert block_fields("scalar_cvar5") == block_fields("scalar") + ["cvar_05"]


def test_scalar_block_adds_nothing_beyond_header() -> None:
    """The scalar arm adds no tail content beyond the shared scalar header."""
    scalar = build_block("scalar", METRIC, None)
    assert len(scalar.splitlines()) == 1
    assert "CVaR" not in scalar
    assert "tail" not in scalar.lower()
    assert block_fields("scalar") == ["scalar_metric"]


def test_serialization_is_deterministic() -> None:
    """Same (arm, scalar_metric, tail_stats) yields byte-identical output."""
    for arm in ARMS:
        a = build_block(arm, METRIC, TAIL_STATS)
        b = build_block(arm, METRIC, TAIL_STATS)
        assert a == b


def test_distributional_block_flags_cvar1_high_variance() -> None:
    """The rendered distributional block flags CVaR-1% as high-variance (B-7)."""
    dist = build_block("distributional", METRIC, TAIL_STATS)
    assert "CVaR 1%" in dist
    assert "high-variance" in dist.lower()


def test_block_fields_consistent_with_rendered_block() -> None:
    """block_fields length matches the rendered field/line layout."""
    # distributional: header + intro + 6 field lines = fields + 1 intro line.
    dist = build_block("distributional", METRIC, TAIL_STATS)
    # number of content lines (header + field lines) excluding intro == len(fields).
    assert len(block_fields("distributional")) == len(_DIST_CONTENT_LINES(dist))


def _DIST_CONTENT_LINES(block: str) -> list[str]:
    """Lines that correspond to a field (header + per-field lines), excluding intros."""
    return [
        ln for ln in block.splitlines()
        if "diagnostics" not in ln  # drop the intro line
    ]


# --- placebo_shuffled: the structure-vs-content control (R32) -------------------------------
def _rendered_values(block: str) -> list[str]:
    """The rendered numeric values, in label order (the lines indented two spaces)."""
    return [
        ln.split(":", 1)[1].strip().split()[0]
        for ln in block.splitlines() if ln.startswith("  ")
    ]


def test_placebo_shuffled_matches_distributional_structure() -> None:
    """placebo_shuffled shares the distributional block's EXACT structure — same header, intro,
    labels, line count, and CVaR-1% annotation; only the numeric VALUES differ (R32)."""
    dist = build_block("distributional", METRIC, TAIL_STATS)
    shuf = build_block("placebo_shuffled", METRIC, TAIL_STATS, shuffle_seed=7)
    dl, sl = dist.splitlines(), shuf.splitlines()
    assert len(dl) == len(sl)
    assert dl[0] == sl[0] and dl[1] == sl[1]  # header + intro identical
    assert [ln.split(":")[0] for ln in dl] == [ln.split(":")[0] for ln in sl]  # labels identical
    assert "high-variance" in shuf.lower()  # the CVaR-1% annotation is preserved
    assert block_fields("placebo_shuffled") == block_fields("distributional")


def test_placebo_shuffled_is_a_derangement_of_the_real_values() -> None:
    """The six tail values are permuted with NO value left in its own label slot — the coherent
    label->value mapping (the tail SHAPE) is broken while the MARGINAL multiset is preserved."""
    shuf = build_block("placebo_shuffled", METRIC, TAIL_STATS, shuffle_seed=3)
    rendered = _rendered_values(shuf)
    real = [_fmt(float(TAIL_STATS[fid])) for fid, _ in _DIST_FIELDS]
    assert sorted(rendered) == sorted(real)  # same marginal multiset of numbers
    assert all(r != v for r, v in zip(rendered, real))  # derangement: nothing in its own slot


def test_placebo_shuffled_is_seeded_and_replayable() -> None:
    """Same seed -> byte-identical (replayable from the archive); seeds actually vary the block."""
    a = build_block("placebo_shuffled", METRIC, TAIL_STATS, shuffle_seed=42)
    b = build_block("placebo_shuffled", METRIC, TAIL_STATS, shuffle_seed=42)
    assert a == b
    variants = {
        build_block("placebo_shuffled", METRIC, TAIL_STATS, shuffle_seed=s) for s in range(8)
    }
    assert len(variants) > 1


def test_placebo_shuffled_requires_tail_stats_and_seed() -> None:
    """placebo_shuffled fails loud without tail_stats or without a shuffle_seed."""
    with pytest.raises(ValueError, match="tail_stats"):
        build_block("placebo_shuffled", METRIC, None, shuffle_seed=1)
    with pytest.raises(ValueError, match="shuffle_seed"):
        build_block("placebo_shuffled", METRIC, TAIL_STATS)


# --- #98: the LEGIBLE rendering must not carry less information than the RAW one ----------------

def test_legible_rendering_has_RESOLUTION_PARITY_with_the_raw_rendering() -> None:
    """The legibility leg varies FRAMING; it must not also vary how much information is carried.

    ``responsiveness.legible_format_responsiveness_differential`` asks whether re-rendering the SAME
    numbers more legibly changes the designer's responsiveness. That identification only holds if the
    legible arm resolves the same distinctions the raw arm does — otherwise a measured "legibility
    effect" is partly information loss, biasing the numeracy-bottleneck leg toward a spurious null
    (the same direction this module already guards for the decile tag).

    Measured on the archive before the fix: ``robust_skew`` at ``+.2f`` collapsed 277 distinct values
    to 14 (99.8 % -> 88.7 % of pairs distinguishable) and ``left_tail_mass`` at ``.1f%`` lost 7.8
    points. The four CVaR fields were already at parity because integer bps IS a 1e-4 step — but only
    since #87b raised the raw renderer to ``.4f``; before that the legible arm was 10x FINER.

    Pins the PROPERTY, not the format strings, so either renderer may be retuned as long as they stay
    matched.
    """
    import numpy as np

    from src.feedback.schema import _DIST_FIELDS, _fmt, _legible_value

    rng = np.random.default_rng(0)
    # Values in the real observed bands (CVaR ~ -0.06..-0.01, mass ~ 0.01..0.05, skew ~ -0.15..0.05).
    bands = {
        "cvar_01": (-0.09, -0.03), "cvar_05": (-0.06, -0.015),
        "cvar_10": (-0.045, -0.01), "cvar_25": (-0.03, -0.005),
        "left_tail_mass": (0.005, 0.06), "robust_skew": (-0.15, 0.05),
    }
    for field_id, _label in _DIST_FIELDS:
        lo, hi = bands[field_id]
        xs = rng.uniform(lo, hi, size=300)
        raw = [_fmt(x) for x in xs]
        leg = [_legible_value(field_id, x) for x in xs]
        raw_pairs = sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if raw[i] != raw[j])
        leg_pairs = sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if leg[i] != leg[j])
        assert leg_pairs >= raw_pairs, (
            f"{field_id}: the LEGIBLE rendering distinguishes {leg_pairs} value pairs but the RAW one "
            f"distinguishes {raw_pairs} — the legible arm is carrying LESS information, so the "
            f"legibility contrast is confounded by resolution"
        )


def test_every_tail_field_has_a_DELIBERATE_decile_direction() -> None:
    """A tail field with an unclassified decile direction silently reverses the legibility device.

    `_DECILE_INVERTED_FIELDS` partitions the six fed fields into "higher value = MORE tail risk"
    (kept positional) and "higher value = LESS tail risk" (inverted so a higher decile always means
    worse). The module already records what goes wrong when that partition is incomplete: a direction
    that flips between lines is "worse than no tag", and it "biases the numeracy-bottleneck leg
    (``responsiveness.legible_format_responsiveness_differential``) toward a spurious null".

    Nothing guarded it. The tail set itself is frozen and triple-guarded (see
    `test_deep_p23_cvar_levels_reconcile` and `test_llm_deep::test_frozen_tail_key_set_is_exactly_the_six`),
    but those catch an ADDED field — not a field added *without* classifying its direction. This
    asserts the partition is TOTAL, so any change to the fed vector fails here until the direction is
    decided on purpose (#104, 2026-07-27; same unguarded-invariant class as #103).
    """
    from src.feedback.schema import _DECILE_INVERTED_FIELDS, _DIST_FIELDS

    fields = {fid for fid, _label in _DIST_FIELDS}
    assert _DECILE_INVERTED_FIELDS <= fields, (
        f"_DECILE_INVERTED_FIELDS names fields that are not fed: "
        f"{sorted(_DECILE_INVERTED_FIELDS - fields)}"
    )
    # The partition must be TOTAL: every fed field is classified one way or the other, and the only
    # NON-inverted field is the one whose value rises with risk (a left-tail probability).
    non_inverted = fields - _DECILE_INVERTED_FIELDS
    assert non_inverted == {"left_tail_mass"}, (
        f"unclassified decile direction for {sorted(non_inverted - {'left_tail_mass'})} — a fed field "
        "was added or renamed without deciding whether a HIGHER value means more or less tail risk. "
        "Inverting is the default (CVaR/skew fall as risk rises); only a probability that RISES with "
        "risk keeps its positional decile."
    )
