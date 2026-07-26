"""v2 leg-transport tests (R80/R82): legs.yaml integrity, the loader's translation contract,
the rolling-alias ban, extra_body passthrough, and per-call cost capture."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import client as C  # noqa: E402
from src.llm.legs import leg_by_label, load_legs, transport_kwargs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
QUEUE = ["deepseek-v4-pro", "glm-5.2", "qwen3.6-27b", "qwen3.5-9b", "haiku-4.5",
         "gpt-5.6-luna", "nemotron-3-super", "sonnet-5",  # R90/R92: sonnet-4.6 removed, sonnet-5 stays
         "gemini-2.5-flash", "kimi-k3"]  # R95; R106 (2026-07-27): 3.5-flash -> 2.5-flash (3.5's
                                         # reasoning is MANDATORY, so it could not join uniform-off)


# --------------------------------------------------------------------------- #
# legs.yaml integrity vs the registered model_suite                            #
# --------------------------------------------------------------------------- #
def test_legs_match_registered_queue_order():
    legs = load_legs()["legs"]
    ms = yaml.safe_load((REPO / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    assert [leg["label"] for leg in legs] == ms["model_suite"]["queue_order"] == QUEUE


def test_qwen_pair_invariant_same_provider_and_quant():
    """The confound-free family pair: EVERY serving knob identical (R80/R103).

    Extended 2026-07-26 (deep review #64). This asserted only ``provider_pin`` and
    ``quantizations``, while ``config/legs.yaml`` declares the invariant more broadly —
    "PAIR INVARIANT: identical reasoning config across the qwen pair". The unguarded half was the
    one that actually bit: R103 records siliconflow serving Qwen3 in THINKING mode by default,
    which burned the whole output budget on hidden reasoning and produced EMPTY authored code
    (compliance 0.4/10 at the frozen config, 0.4 -> 1.0 once disabled).

    The pair IS the capability gradient (9b is the bottom anchor, ~17% gate-pass; 27b ~83%), so a
    divergence in ANY serving knob — reasoning, decoding temperature, output budget, or the key/route
    — would confound exactly the comparison the pair exists to make. Assert the whole contract, not
    the two fields someone happened to write down first."""
    q27, q9 = leg_by_label("qwen3.6-27b"), leg_by_label("qwen3.5-9b")
    assert q27["provider_pin"] == q9["provider_pin"]
    assert q27["quantizations"] == q9["quantizations"] == ["fp8"]
    assert q27["provider_pin"]["allow_fallbacks"] is False
    # the knobs that would confound the gradient if they ever drifted apart
    for field in ("reasoning", "temperature", "max_tokens", "api_key_env"):
        assert q27.get(field) == q9.get(field), (
            f"qwen pair invariant broken on {field!r}: 27b={q27.get(field)!r} vs 9b={q9.get(field)!r}"
        )
    # and the reasoning pin is present and DISABLING on both — the R103 fix, not merely equal
    assert q27["reasoning"] == {"enabled": False}, q27["reasoning"]

    # the invariant must survive the TRANSLATION too, not just the yaml: identical extra_body
    # serving config is what the provider actually receives
    eb27, eb9 = transport_kwargs(q27)["extra_body"], transport_kwargs(q9)["extra_body"]
    assert eb27 == eb9, f"translated serving config diverges: {eb27} vs {eb9}"


def test_every_leg_has_max_tokens_and_no_alias():
    for leg in load_legs()["legs"]:
        assert isinstance(leg["max_tokens"], int)
        assert "~" not in leg["model"] and not leg["model"].endswith("-latest")


def test_planning_prices_cover_all_legs():
    cfg = load_legs()
    priced = set(cfg["planning_prices"])
    assert {leg["model"] for leg in cfg["legs"]} <= priced


# --------------------------------------------------------------------------- #
# The loader's translation contract                                            #
# --------------------------------------------------------------------------- #
def test_transport_kwargs_openrouter_assembles_extra_body():
    kw = transport_kwargs(leg_by_label("qwen3.6-27b"))
    assert kw["provider"] == "openrouter" and kw["max_tokens"] == 16384   # R106 uniform cap
    eb = kw["extra_body"]
    assert eb["provider"] == {"only": ["siliconflow"], "allow_fallbacks": False,
                              "quantizations": ["fp8"]}
    assert eb["usage"] == {"include": True}  # per-call cost for the advisory ledger


def test_every_leg_is_reasoning_off_and_caps_are_matched():
    """R106 (ratified with the supervisor on 2026-07-26, implemented 2026-07-27): UNIFORM conditions.

    Two invariants, and the second matters as much as the first. **Reasoning off everywhere** removes
    the masking confound (a reasoning scratchpad can silently reformat the fed floats, which would
    contaminate the numeracy headline) — and every leg that ever ran with reasoning ON has been
    measured truncating: qwen 0.0/10, glm 0.6, kimi 0.8, gemini 0.1, deepseek 0.9. **Matched caps**
    close the ledger's HIGH fragility "Haiku (4096, no reasoning) vs Opus-5 (8192) — the DiD
    conflates capability with token-budget/thinking": with unequal caps a capability contrast is not
    a capability contrast.
    """
    legs = load_legs()["legs"]
    caps, not_off = set(), []
    for leg in legs:
        caps.add(leg["max_tokens"])
        kw = transport_kwargs(leg)
        if leg["provider"] == "anthropic":
            off = kw.get("thinking") == {"type": "disabled"}
        else:
            off = (kw.get("extra_body") or {}).get("reasoning") == {"enabled": False}
        if not off:
            not_off.append(leg["label"])
    assert not not_off, f"R106 violated — leg(s) not pinned reasoning-off: {not_off}"
    assert caps == {16384}, f"R106 violated — caps are not matched: {sorted(caps)}"


def test_anthropic_legs_carry_the_thinking_pin_not_extra_body():
    """The Anthropic transport REJECTS extra_body, so `thinking` is the only pin channel it has.

    Before R106 these legs carried no reasoning key at all — reasoning-off by VENDOR DEFAULT, not by
    pin. That is exactly how R102's "Opus 5 runs adaptive thinking by default" came to be asserted in
    the registration and proved empirically false.
    """
    for label in ("sonnet-5", "haiku-4.5"):
        kw = transport_kwargs(leg_by_label(label))
        assert kw["provider"] == "anthropic"
        assert "extra_body" not in kw, f"{label}: extra_body is rejected by that transport"
        assert kw["thinking"] == {"type": "disabled"}, f"{label}: reasoning pin missing"


def test_loader_rejects_alias_and_missing_fields(tmp_path: Path):
    bad = tmp_path / "legs.yaml"
    bad.write_text("legs:\n  - {label: x, provider: openrouter, model: openai/gpt-latest, "
                   "api_key_env: K, max_tokens: 100}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rolling alias"):
        load_legs(bad)
    bad.write_text("legs:\n  - {label: x, provider: openrouter, model: a/b, api_key_env: K}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="max_tokens"):
        load_legs(bad)


# --------------------------------------------------------------------------- #
# The transport factory: alias ban + extra_body plumbing + cost capture        #
# --------------------------------------------------------------------------- #
def test_factory_bans_rolling_aliases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="rolling 'latest' alias"):
        C.build_transport("openrouter", "~openai/gpt-latest")
    with pytest.raises(ValueError, match="rolling 'latest' alias"):
        C.build_transport("openrouter", "moonshotai/kimi-latest")


def test_factory_rejects_extra_body_on_anthropic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(ValueError, match="anthropic"):
        C.build_transport("anthropic", "claude-sonnet-4-6",
                          extra_body={"provider": {"only": ["x"]}})


class _FakeCompletions:
    def __init__(self, response):  # captures kwargs; returns the canned response
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def _fake_openai_client(response):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(response)))


def _fake_response(cost=None):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=cost)
    choice = SimpleNamespace(finish_reason="stop",
                             message=SimpleNamespace(content="def reward(): ..."))
    return SimpleNamespace(usage=usage, choices=[choice], id="req-1", model="served/slug")


def test_extra_body_reaches_the_wire_and_cost_is_captured():
    resp = _fake_response(cost=0.00123)
    fake = _fake_openai_client(resp)
    t = C._OpenAITransport(fake, "z-ai/glm-5.2", temperature=None, retrying=None,
                           max_tokens=4096,
                           extra_body={"provider": {"only": ["p"]}, "usage": {"include": True}})
    out = t("sys", "user")
    sent = fake.chat.completions.last_kwargs
    assert sent["extra_body"]["provider"] == {"only": ["p"]}
    assert sent["max_tokens"] == 4096
    assert out == "def reward(): ..."
    assert t.last_cost_usd == pytest.approx(0.00123)
    assert t.last_served_model == "served/slug"


def test_cost_none_when_provider_returns_no_cost():
    t = C._OpenAITransport(_fake_openai_client(_fake_response(cost=None)), "m",
                           temperature=None, retrying=None, max_tokens=100)
    t("s", "u")
    assert t.last_cost_usd is None
