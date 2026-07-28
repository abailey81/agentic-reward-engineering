"""R114 drift guard: the REGISTERED fed-block precision must equal what the renderer emits.

R114 registered the rendered precision of the fed block as a design parameter — because the reward
designer never sees a float, it sees a STRING, so quantization is part of the stimulus. The values
live in ``config/preregistration.yaml: fed_rendering`` and are implemented in
``src/feedback/schema.py``.

NOTE: WHY THIS TEST EXISTS. Nothing compared the two. The freeze binds ``schema.py`` into the canonical
hash (#97), which makes a POST-freeze edit detectable — but PRE-freeze the registered number and the
emitted number could differ silently, and the freeze would then have sealed a registration that
misdescribes the instrument. Every comparably load-bearing value (SESOI, the equivalence margin, the
m=6 family, lambda_cvar, tf32) has a prose<->yaml assertion in ``freeze.py``; ``fed_rendering`` had
none, and neither did the suite. This is the same shape as the R110 JZS-prior gap that was repaired
the day before, and as R84's "a registered NAME requires a registered VALUE".

Deliberately a TEST rather than a freeze check: importing ``schema.py`` into the gate would drag the
feedback stack into a step that must stay import-light — the same reasoning R110 recorded for the
JZS pin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))  # freeze.py lives in scripts/, not on the package path


@pytest.fixture(scope="module")
def registered() -> dict:
    cfg = yaml.safe_load((_ROOT / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    block = cfg.get("fed_rendering")
    assert block, "config/preregistration.yaml has no `fed_rendering` block (R114)"
    return block


def test_scalar_header_renders_at_the_registered_precision(registered: dict) -> None:
    """The header is the scalar arm's ENTIRE feedback content, and the scalar arm is the PRIMARY H2
    comparator. At the pre-R114 ``.2f``, 229/240 archived val_fitness values rendered literally
    "0.00" — the tail vector was being contrasted against a near-CONSTANT."""
    from src.feedback.schema import _HEADER

    want = int(registered["scalar_header_decimals"])
    rendered = _HEADER.format(metric=0.000243)
    decimals = len(rendered.split("scored: ")[1].split(" ")[0].split(".")[1])
    assert decimals == want, f"header renders {decimals} decimals, register says {want}"
    assert "0.000243" in rendered, "the measured median fitness must survive the rendering"


def test_tail_lines_render_at_the_registered_precision(registered: dict) -> None:
    """R76 measured the paired candidate-to-candidate diff-SE of a fed CVaR-5% at 1e-4 to 8e-4. The
    pre-R114 ``.3f`` step of 1e-3 was LARGER than that entire range, so 90.1% of sibling-close pairs
    rendered as the SAME STRING."""
    from src.feedback.schema import _fmt

    want = int(registered["tail_line_decimals"])
    assert len(_fmt(-0.012345).split(".")[1]) == want
    # the R76 boundary case: a 1e-4 separation must survive as two DIFFERENT strings
    assert _fmt(-0.0123) != _fmt(-0.0124)


def test_fixed_point_never_scientific(registered: dict) -> None:
    """A fed number that renders as ``1e-04`` is a different stimulus from ``0.0001``."""
    from src.feedback.schema import _HEADER, _fmt

    assert registered["notation"] == "fixed_point_never_scientific"
    for text in (_fmt(-1e-4), _fmt(1e-6), _HEADER.format(metric=1e-6)):
        assert "e-" not in text.lower(), f"scientific notation leaked into the stimulus: {text}"


def test_legible_rendering_has_resolution_parity_with_raw(registered: dict) -> None:
    """Before R114 the legible re-rendering resolved COARSER than raw on two of six fields
    (``robust_skew`` collapsed 277 distinct values to 14; ``left_tail_mass`` 54 to 18), so any
    measured 'legibility effect' was partly INFORMATION LOSS — biased toward a spurious null, which
    is the exact direction that module already guards against."""
    from src.feedback.schema import _fmt, _legible_value

    assert registered["legible_resolution_parity"] == "required"
    fields = ("cvar_05", "cvar_10", "cvar_25", "cvar_01", "left_tail_mass", "robust_skew")
    step = 10.0 ** -int(registered["tail_line_decimals"])
    for field in fields:
        base = 0.0123
        raw_distinct = {_fmt(base), _fmt(base + step)}
        leg_distinct = {_legible_value(field, base), _legible_value(field, base + step)}
        assert len(leg_distinct) >= len(raw_distinct), (
            f"{field}: the legible rendering collapses a separation the raw one preserves "
            f"({sorted(leg_distinct)} vs {sorted(raw_distinct)})")


def test_the_renderer_is_hash_bound(registered: dict) -> None:
    """The registered values are only as strong as the file that implements them being sealed."""
    from freeze import _BOUND_TREATMENT

    bound = {str(p).replace("\\", "/") for p in _BOUND_TREATMENT}
    assert registered["bound_by"] in bound, (
        f"{registered['bound_by']} is named as the binder but is not in _BOUND_TREATMENT: {bound}")
