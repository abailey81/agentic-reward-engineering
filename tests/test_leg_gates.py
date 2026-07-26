"""Offline tests for the per-leg gates (R80/R82): the transport seam takes a fake — no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from leg_gates import _flagged, _screen_probes, run_leg_gates  # noqa: E402

# A PRICED model id (audit 2026-07-24: the old unpriced "v/m" fixture only passed because the
# price_key "review" verdict was being clobbered by a clean screen — the exact MAJOR bug).
_LEG = {"label": "test-leg", "provider": "openrouter", "model": "qwen/qwen3.5-9b",
        "api_key_env": "K", "max_tokens": 100}


class _FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.last_cost_usd = 0.001
        self.last_served_model = "fake-served"
        self.last_stop_reason = "stop"

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0) if self._responses else "I cannot determine that."


def test_screen_probes_are_deterministic():
    a, b = _screen_probes(), _screen_probes()
    assert a == b and [p["name"] for p in a] == ["continuation", "identification", "canary"]


def test_flag_rules():
    assert _flagged("This is clearly the Lehman collapse of 2008.")
    assert _flagged("These are the returns of Apple during covid.")
    assert not _flagged("The data are anonymized; I cannot determine the source.")


def test_clean_model_passes_screen(tmp_path: Path):
    fake = _FakeTransport(["SMOKE-OK"] + ["I cannot determine that; the series is anonymized."] * 3)
    s = run_leg_gates(_LEG, tmp_path, which=("smoke", "screen"),
                      transport_factory=lambda leg: fake)
    assert s["smoke_ok"] is True
    assert s["screen_verdict"] == "pass" and s["screen_flags"] == []
    rows = [json.loads(x) for x in (tmp_path / "test-leg.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == 4 and all(r["cost_usd"] == 0.001 for r in rows)  # all archived
    assert (tmp_path / "test-leg.summary.json").exists()


def test_contaminated_model_is_flagged_not_dropped(tmp_path: Path):
    fake = _FakeTransport(["I recognize this — clearly the 2008 Lehman window."] * 3)
    s = run_leg_gates(_LEG, tmp_path, which=("screen",), transport_factory=lambda leg: fake)
    assert s["screen_verdict"] == "FLAG->review"          # routed for review, never silently dropped


class _StopReasonTransport(_FakeTransport):
    """A transport whose completions carry a chosen ``last_stop_reason`` (truncated/refused)."""

    def __init__(self, responses, stop_reason: str):
        super().__init__(responses)
        self.last_stop_reason = stop_reason


def test_empty_screen_answers_are_unverified_not_clean(tmp_path: Path):
    """An UNANSWERED screen must not certify a clean screen (deep review 2026-07-26).

    A refusal / content_filter / max_tokens completion is EMPTY without raising, and ``_flagged("")``
    is False -- so an all-empty screen used to report ``pass``, certifying "no contamination markers"
    having asked and learnt NOTHING. Under ``--only screen`` no compliance gate runs to catch it.
    """
    fake = _FakeTransport(["", "", ""])
    s = run_leg_gates(_LEG, tmp_path, which=("screen",), transport_factory=lambda leg: fake)

    assert s["screen_flags"] == []                        # nothing was FLAGGED...
    assert len(s["screen_unusable"]) == 3                 # ...because nothing was ANSWERED
    assert s["screen_verdict"].startswith("UNVERIFIED->review"), s["screen_verdict"]


def test_truncated_screen_answers_are_unverified_not_clean(tmp_path: Path):
    """A PARTIAL answer cannot clear the screen either -- the markers may sit past the cut."""
    fake = _StopReasonTransport(["The series appears to be"] * 3, "max_tokens")
    s = run_leg_gates(_LEG, tmp_path, which=("screen",), transport_factory=lambda leg: fake)

    assert s["screen_verdict"].startswith("UNVERIFIED->review"), s["screen_verdict"]


def test_a_real_flag_outranks_unusable_probes(tmp_path: Path):
    """Precedence: a genuine contamination marker still wins over unanswered probes."""
    fake = _FakeTransport(["I recognize this — clearly the 2008 Lehman window.", "", ""])
    s = run_leg_gates(_LEG, tmp_path, which=("screen",), transport_factory=lambda leg: fake)

    assert s["screen_verdict"] == "FLAG->review"
    assert set(s["screen_flags"]) <= {"continuation", "identification", "canary"}
    assert len(s["screen_flags"]) >= 1


def test_compliance_rate_scored_on_real_prompts(tmp_path: Path):
    good = "```python\ndef reward(weights, returns, prev_weights, port_ret, info):\n" \
           "    return float(port_ret), {}, None\n```"
    fake = _FakeTransport([good, good, "sorry, no code here", good])
    s = run_leg_gates(_LEG, tmp_path, which=("compliance",), n_compliance=4,
                      transport_factory=lambda leg: fake)
    assert s["compliance_rate"] == pytest.approx(0.75)
    assert s["compliance_verdict"].startswith("LOW")   # 0.75 < 1.0 strict floor (2026-07-25)
    # the REAL frozen prompts were used (same-exam principle)
    system_sent = fake.calls[0][0]
    assert "ANONYMIZED numeric arrays" in system_sent


def test_compliance_floor_is_extremely_strict():
    from scripts.leg_gates import _COMPLIANCE_FLOOR
    assert _COMPLIANCE_FLOOR == 1.0   # 2026-07-25: any sub-1.0 leg flags for review


def test_spend_rides_the_advisory_ledger(tmp_path: Path):
    fake = _FakeTransport(["SMOKE-OK"])
    led = tmp_path / "ledger.jsonl"
    run_leg_gates(_LEG, tmp_path, which=("smoke",), transport_factory=lambda leg: fake,
                  ledger_path=led)
    from src.llm.spend_ledger import spend_summary
    assert spend_summary(led)["n_calls"] == 1


def test_r85_usage_roundtrip_evidence_archived(tmp_path: Path):
    """R85: the smoke archives full usage + surfaces a pin-roundtrip verdict for reasoning legs."""
    fake = _FakeTransport(["SMOKE-OK"])
    fake.last_usage = {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 42}
    leg = dict(_LEG, reasoning={"effort": "low"})
    s = run_leg_gates(leg, tmp_path, which=("smoke",), transport_factory=lambda lg: fake)
    # 2026-07-25 (repro-audit HOLE 2): the verdict now VERIFIES from the reasoning-token count, not
    # merely bool(usage) — 42 > 0 tokens on an ENABLE pin => it functioned.
    assert s["pin_roundtrip"].startswith("verified")
    assert s["reasoning_tokens_observed"] == 42
    assert s["usage_observed"]["reasoning_tokens"] == 42
    row = json.loads((tmp_path / "test-leg.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["usage"]["reasoning_tokens"] == 42        # the per-call evidence on disk


def test_r85_pinned_reasoning_with_no_usage_flags_review(tmp_path: Path):
    fake = _FakeTransport(["SMOKE-OK"])           # _FakeTransport has no last_usage attribute
    leg = dict(_LEG, reasoning={"mode": "think-high"})
    s = run_leg_gates(leg, tmp_path, which=("smoke",), transport_factory=lambda lg: fake)
    assert s["pin_roundtrip"].startswith("UNVERIFIED")   # never silently trusted


def test_r85_disable_pin_verified_when_no_reasoning(tmp_path: Path):
    """A DISABLING reasoning pin ({"enabled": false}) is VERIFIED iff ~0 reasoning tokens are served
    (the qwen thinking-off fix: the pin's whole point is to STOP the budget-burning reasoning)."""
    good = "```python\ndef reward(w, r, p, pr, i):\n    return float(pr), {}, None\n```"
    fake = _FakeTransport([good] * 3)
    fake.last_usage = {"input_tokens": 8, "output_tokens": 20, "reasoning_tokens": 0}
    leg = dict(_LEG, reasoning={"enabled": False})
    s = run_leg_gates(leg, tmp_path, which=("compliance",), n_compliance=3,
                      transport_factory=lambda lg: fake)
    assert s["pin_roundtrip"].startswith("verified")
    assert s["reasoning_tokens_observed"] == 0


def test_r85_enable_pin_fictional_when_zero_reasoning_on_compliance(tmp_path: Path):
    """An ENABLE reasoning pin that yields a MEASURED 0 reasoning tokens on real authoring calls is
    FICTIONAL (the pin was silently ignored — the R85 defect the archive could not previously detect)."""
    good = "```python\ndef reward(w, r, p, pr, i):\n    return float(pr), {}, None\n```"
    fake = _FakeTransport([good] * 3)
    fake.last_usage = {"input_tokens": 8, "output_tokens": 20, "reasoning_tokens": 0}  # MEASURED 0
    leg = dict(_LEG, reasoning={"mode": "pro"})
    s = run_leg_gates(leg, tmp_path, which=("compliance",), n_compliance=3,
                      transport_factory=lambda lg: fake)
    assert s["pin_roundtrip"].startswith("FICTIONAL")
    assert s["screen_verdict"] == "review"               # routed for review, never silently trusted


def test_r85_pin_unverified_when_reasoning_field_absent(tmp_path: Path):
    """CRITICAL fix M1/m1: when the provider returns NO reasoning_tokens field, the pin's effect is
    UNMEASURABLE -> UNVERIFIED for BOTH directions. Previously a DISABLE pin failed OPEN ('verified'
    on absent evidence) and an ENABLE pin mislabelled absence as 'FICTIONAL'."""
    good = "```python\ndef reward(w, r, p, pr, i):\n    return float(pr), {}, None\n```"
    for i, pin in enumerate(({"enabled": False}, {"mode": "pro"})):   # disable AND enable both fail-safe
        fake = _FakeTransport([good] * 3)
        fake.last_usage = {"input_tokens": 8, "output_tokens": 20}    # NO reasoning_tokens field
        leg = dict(_LEG, reasoning=pin)
        s = run_leg_gates(leg, tmp_path / f"u{i}", which=("compliance",), n_compliance=3,
                          transport_factory=lambda lg: fake)
        assert s["pin_roundtrip"].startswith("UNVERIFIED"), (pin, s["pin_roundtrip"])
        assert s["reasoning_tokens_observed"] is None


def test_m3_served_provider_adjudicated_against_pin(tmp_path: Path):
    """m3: a served provider OUTSIDE provider_pin.only flags MISROUTE->review; an in-pin match is
    'verified'. Previously served_provider was archived but never adjudicated."""
    good = "```python\ndef reward(w, r, p, pr, i):\n    return float(pr), {}, None\n```"
    leg = dict(_LEG, provider_pin={"only": ["siliconflow"], "allow_fallbacks": False})
    fake_ok = _FakeTransport([good] * 3)
    fake_ok.last_served_provider = "siliconflow"
    s_ok = run_leg_gates(leg, tmp_path / "ok", which=("compliance",), n_compliance=3,
                         transport_factory=lambda lg: fake_ok)
    assert s_ok["provider_roundtrip"].startswith("verified")
    fake_bad = _FakeTransport([good] * 3)
    fake_bad.last_served_provider = "someone-else"
    s_bad = run_leg_gates(leg, tmp_path / "bad", which=("compliance",), n_compliance=3,
                          transport_factory=lambda lg: fake_bad)
    assert s_bad["provider_roundtrip"].startswith("MISROUTE")
    assert s_bad["screen_verdict"] == "review"


def test_compliance_floor_flags_zero_compliance(tmp_path: Path):
    """A leg that authors 0 compliant rewards must be flagged for review, not stamped 'pass'
    (2026-07-25: closes the qwen3.5-9b 0.0->'pass' verdict-logic gap)."""
    fake = _FakeTransport(["no code here"] * 4)
    s = run_leg_gates(_LEG, tmp_path, which=("compliance",), n_compliance=4,
                      transport_factory=lambda lg: fake)
    assert s["compliance_rate"] == 0.0
    assert s["screen_verdict"] == "review"
    assert s["compliance_verdict"].startswith("LOW")


def test_price_key_review_survives_a_clean_screen(tmp_path, monkeypatch):
    """Audit 2026-07-24 MAJOR regression: the unpriced-model 'review' verdict must NOT be
    clobbered back to 'pass' by a clean contamination screen."""
    import scripts.leg_gates as lg

    leg = {"label": "x", "model": "unpriced/model", "provider": "openrouter"}

    class _T:
        def __call__(self, system, user):
            return "I cannot determine that."  # hedged -> screen passes
    s = lg.run_leg_gates(leg, tmp_path, which=("screen",), transport_factory=lambda _lg: _T())
    assert s["price_key"] == "MISSING->review"
    assert s["screen_verdict"] != "pass"


# --- provider round-trip normalisation (2026-07-26, measured false alarm) ----------------------

def test_provider_pin_matches_the_display_name_the_API_actually_returns():
    """MEASURED: the pin is OpenRouter's slug `siliconflow`, the response says `SiliconFlow`, and a
    literal comparison declared MISROUTE on BOTH qwen legs while routing was exactly as pinned."""
    from scripts.leg_gates import _norm_provider

    assert _norm_provider("SiliconFlow") == _norm_provider("siliconflow")
    assert _norm_provider("Silicon Flow") == _norm_provider("siliconflow")
    assert _norm_provider("SILICONFLOW") == _norm_provider("siliconflow")


def test_a_GENUINELY_different_provider_still_fails_the_pin():
    """The fold must not become a rubber stamp: a real silent re-route (different provider, and so
    potentially a different quantization) is exactly what this check exists to catch."""
    from scripts.leg_gates import _norm_provider

    for other in ("Together", "DeepInfra", "Fireworks", "Novita"):
        assert _norm_provider(other) != _norm_provider("siliconflow")
