"""Tests for the LLM completion-integrity + cost upgrades (2026-06-29).

Covers: (1) ``LLMClient.complete`` archives ``stop_reason`` + ``request_id`` when the transport exposes
them; (2) the real ``_AnthropicTransport`` / ``_OpenAITransport`` capture stop_reason + request_id from a
faked provider message and WARN-log a truncation/refusal; (3) ``summarize_llm_cost`` computes cache-aware
USD + a completion-integrity tally from archived records. All offline — provider clients are faked, no
network, no torch.
"""

from __future__ import annotations

import logging

from src.llm.client import (
    LLMClient,
    ProvenanceRecord,
    _AnthropicTransport,
    _OpenAITransport,
)
from src.llm.cost import summarize_llm_cost


# ---- (1) archival of stop_reason + request_id via LLMClient.complete -------------------------------- #
def test_complete_archives_stop_reason_and_request_id() -> None:
    class _MetaTransport:
        last_usage = {"input_tokens": 10, "output_tokens": 4}
        last_stop_reason = "max_tokens"
        last_request_id = "req_abc123"
        last_served_model = "qwen/qwen3-coder-served-snapshot"  # R71 reproducibility anchor

        def __call__(self, system: str, user: str) -> str:
            return "def reward(...): ...  # truncated"

    archive: list[ProvenanceRecord] = []
    LLMClient({"model": "claude-opus-4-8"}, transport=_MetaTransport(), archive=archive).complete("S", "U")
    assert archive[0].stop_reason == "max_tokens"
    assert archive[0].request_id == "req_abc123"
    assert archive[0].served_model == "qwen/qwen3-coder-served-snapshot"


def test_complete_meta_defaults_none_for_plain_transport() -> None:
    from src.llm.client import FakeTransport

    archive: list[ProvenanceRecord] = []
    LLMClient({"model": "m"}, transport=FakeTransport(response="x"), archive=archive).complete("S", "U")
    assert archive[0].stop_reason is None and archive[0].request_id is None
    assert archive[0].served_model is None


# ---- (2) real transports capture provider metadata + warn on incomplete ----------------------------- #
class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicMessage:
    def __init__(self, text: str, stop_reason: str) -> None:
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason
        self._request_id = "req_anthropic_1"
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20,
                                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})()


class _FakeAnthropicClient:
    def __init__(self, msg: _FakeAnthropicMessage) -> None:
        self.messages = type("M", (), {"create": lambda _self, **kw: msg})()


def test_anthropic_transport_captures_stop_reason_and_warns(caplog) -> None:
    msg = _FakeAnthropicMessage("def reward(w, r, p, pr, i):\n    return float(pr), {}, None", "max_tokens")
    tr = _AnthropicTransport(
        _FakeAnthropicClient(msg), "claude-opus-4-8",
        temperature=None, max_tokens=4096, cache_system=True, retrying=None,
    )
    with caplog.at_level(logging.WARNING, logger="llm.client"):
        out = tr("SYS", "USER")
    assert "def reward" in out
    assert tr.last_stop_reason == "max_tokens" and tr.last_request_id == "req_anthropic_1"
    assert any("llm_incomplete_completion" in r.message for r in caplog.records)


def test_anthropic_transport_no_warn_on_end_turn(caplog) -> None:
    msg = _FakeAnthropicMessage("ok", "end_turn")
    tr = _AnthropicTransport(
        _FakeAnthropicClient(msg), "claude-opus-4-8",
        temperature=None, max_tokens=4096, cache_system=True, retrying=None,
    )
    with caplog.at_level(logging.WARNING, logger="llm.client"):
        tr("S", "U")
    assert tr.last_stop_reason == "end_turn"
    assert not any("llm_incomplete_completion" in r.message for r in caplog.records)


class _FakeOpenAIChoice:
    def __init__(self, text: str, finish_reason: str) -> None:
        self.message = type("Msg", (), {"content": text})()
        self.finish_reason = finish_reason


class _FakeOpenAIResponse:
    def __init__(self, text: str, finish_reason: str) -> None:
        self.choices = [_FakeOpenAIChoice(text, finish_reason)]
        self.id = "chatcmpl_xyz"
        self.usage = type("U", (), {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59})()


class _FakeOpenAIClient:
    def __init__(self, resp: _FakeOpenAIResponse) -> None:
        self.chat = type("C", (), {"completions": type("CC", (), {"create": lambda _s, **kw: resp})()})()


def test_openai_transport_captures_finish_reason_and_id() -> None:
    tr = _OpenAITransport(
        _FakeOpenAIClient(_FakeOpenAIResponse("code", "length")),
        "gemini-2.5", temperature=1.0, retrying=None, max_tokens=4096,
    )
    out = tr("S", "U")
    assert out == "code"
    assert tr.last_stop_reason == "length" and tr.last_request_id == "chatcmpl_xyz"


# ---- (3) cost reducer ------------------------------------------------------------------------------- #
def _rec(model, usage, stop="end_turn"):
    return ProvenanceRecord(model=model, system="", user="", response="", usage=usage, stop_reason=stop)


def test_summarize_llm_cost_cache_aware_and_integrity() -> None:
    records = [
        _rec("claude-opus-4-8", {"input_tokens": 1_000_000, "output_tokens": 200_000,
                                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
        _rec("claude-opus-4-8", {"input_tokens": 0, "output_tokens": 0,
                                 "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}),
        _rec("claude-opus-4-8", None, stop="max_tokens"),   # truncated, no usage
    ]
    s = summarize_llm_cost(records)
    assert s["calls"] == 3 and s["costed_calls"] == 2
    assert s["input_tokens"] == 1_000_000 and s["output_tokens"] == 200_000
    assert s["cache_write_tokens"] == 1_000_000 and s["cache_read_tokens"] == 1_000_000
    # rec1: 1M*5 + 0.2M*25 = 5 + 5 = 10 ; rec2: cache write 1M*5*1.25=6.25 + read 1M*5*0.1=0.5 = 6.75
    assert abs(s["usd"] - 16.75) < 1e-6
    assert s["incomplete"] == 1 and s["incomplete_by_reason"] == {"max_tokens": 1}
    assert s["by_model"]["claude-opus-4-8"]["usd"] == s["usd"]


def test_summarize_llm_cost_unknown_model_not_costed() -> None:
    s = summarize_llm_cost([_rec("mystery", {"input_tokens": 100, "output_tokens": 5})])
    assert s["calls"] == 1 and s["costed_calls"] == 0 and s["usd"] == 0.0
    assert s["input_tokens"] == 100  # tokens still counted, just not priced
