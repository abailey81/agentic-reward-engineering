"""Tests for the provider transport internals (src/llm/client.py — ADR-033/034 wiring).

These exercise the Anthropic transport WITHOUT installing ``anthropic`` or ``tenacity``: the
callable ``_AnthropicTransport`` takes its SDK client as a constructor argument, so a fake client
drives the real request-shaping logic. Covered:

  - prompt-caching: the static system block is sent as a content block with ``cache_control``
    (the ADR-016 shared-context cache lever), and ``cache_system=False`` sends a plain string;
  - temperature is passed through only when set (the Eureka loop needs temperature > 0 — ADR-016 —
    so ``None`` must leave the provider default rather than forcing 0);
  - per-call token ``usage`` is captured on ``last_usage`` (archived for cost accounting, C-2);
  - the retry predicate ``_is_transient_api_error`` classifies transient vs terminal failures;
  - an injected ``retrying`` wrapper is actually used (transparent to the SDK, ``max_retries=0``);
  - the no-key path raises clearly BEFORE importing the SDK (no network, no dependency needed).
"""
from __future__ import annotations

import pytest

from src.llm.client import (
    _AnthropicTransport,
    _is_transient_api_error,
    _make_retrying,
    _usage_dict,
    make_anthropic_transport,
)


# --------------------------------------------------------------------------- #
# Fakes standing in for the anthropic SDK client / message objects            #
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, text: str, type_: str = "text") -> None:
        self.text = text
        self.type = type_


class _Usage:
    input_tokens = 10
    output_tokens = 4
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 8


class _Message:
    def __init__(self, blocks: list[_Block], usage: object | None) -> None:
        self.content = blocks
        self.usage = usage


class _Messages:
    def __init__(self, recorder: list[dict], blocks: list[_Block], usage: object | None) -> None:
        self._recorder = recorder
        self._blocks = blocks
        self._usage = usage

    def create(self, **kwargs: object) -> _Message:
        self._recorder.append(kwargs)
        return _Message(self._blocks, self._usage)


class _FakeClient:
    def __init__(self, blocks: list[_Block] | None = None, usage: object | None = _Usage()) -> None:
        self.calls: list[dict] = []
        self.messages = _Messages(
            self.calls, blocks or [_Block("def reward(): return 0.0")], usage
        )


def _transport(client: _FakeClient, **kw: object) -> _AnthropicTransport:
    defaults: dict = dict(temperature=1.0, max_tokens=4096, cache_system=True, retrying=None)
    defaults.update(kw)
    return _AnthropicTransport(client, "claude-sonnet-4-6", **defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# prompt-cache + request shape                                                #
# --------------------------------------------------------------------------- #
def test_caches_system_block_and_threads_model_and_messages() -> None:
    client = _FakeClient()
    out = _transport(client)("SYS", "USER")
    assert out == "def reward(): return 0.0"
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["messages"] == [{"role": "user", "content": "USER"}]
    # System sent as a cache_control content block (the shared-context cache lever).
    assert call["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]


def test_cache_disabled_sends_plain_system_string() -> None:
    client = _FakeClient()
    _transport(client, cache_system=False)("SYS", "USER")
    assert client.calls[0]["system"] == "SYS"


def test_only_text_blocks_are_concatenated() -> None:
    client = _FakeClient(blocks=[_Block("a"), _Block("IGNORED", type_="thinking"), _Block("b")])
    assert _transport(client)("S", "U") == "ab"


# --------------------------------------------------------------------------- #
# temperature threading — None leaves the provider default (Eureka needs > 0)  #
# --------------------------------------------------------------------------- #
def test_temperature_passed_when_set() -> None:
    client = _FakeClient()
    _transport(client, temperature=1.0)("S", "U")
    assert client.calls[0]["temperature"] == 1.0


def test_temperature_omitted_when_none() -> None:
    client = _FakeClient()
    _transport(client, temperature=None)("S", "U")
    assert "temperature" not in client.calls[0]


# --------------------------------------------------------------------------- #
# usage capture                                                               #
# --------------------------------------------------------------------------- #
def test_last_usage_captured_after_call() -> None:
    client = _FakeClient()
    t = _transport(client)
    assert t.last_usage is None
    t("S", "U")
    assert t.last_usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 8,
        # R106: Anthropic reports NO reasoning_tokens, so the count of `thinking` CONTENT BLOCKS is
        # the only direct evidence a disable pin took effect. Recorded per call; 0 here because the
        # fake response carries a text block only.
        "thinking_blocks": 0,
    }


def test_thinking_blocks_are_counted_so_a_disable_pin_can_be_adjudicated() -> None:
    """R106: the evidence that REPLACES reasoning_tokens on Anthropic, which reports none.

    It has to track reality in BOTH directions — a counter stuck at 0 would rubber-stamp every
    disable pin as "verified", which is precisely the fictional-pin failure the three-state verdict
    exists to prevent.
    """
    t = _transport(_FakeClient())
    t("S", "U")
    assert t.last_usage["thinking_blocks"] == 0        # text-only response -> thinking did NOT run

    thinking = _FakeClient(blocks=[_Block("scratchpad...", "thinking"), _Block("answer")])
    t2 = _transport(thinking)
    out = t2("S", "U")
    assert t2.last_usage["thinking_blocks"] == 1       # a thinking block IS seen when present
    assert out == "answer"                             # and thinking is never mistaken for the answer


def test_thinking_pin_is_sent_only_when_requested() -> None:
    """R106 must be inert unless asked for: omitting the pin has to leave the request byte-identical
    to the pre-R106 behaviour, or every existing archived call would become non-comparable."""
    client = _FakeClient()
    _transport(client)("S", "U")
    assert "thinking" not in client.calls[0]

    pinned = _FakeClient()
    _transport(pinned, thinking={"type": "disabled"})("S", "U")
    assert pinned.calls[0]["thinking"] == {"type": "disabled"}


def test_usage_dict_none_when_message_has_no_usage() -> None:
    assert _usage_dict(_Message([_Block("x")], None)) is None
    client = _FakeClient(usage=None)
    t = _transport(client)
    t("S", "U")
    assert t.last_usage is None


# --------------------------------------------------------------------------- #
# retry policy                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["APIConnectionError", "APITimeoutError", "RateLimitError",
                                  "InternalServerError", "APIError"])
def test_transient_errors_classified_retryable(name: str) -> None:
    exc = type(name, (Exception,), {})()
    assert _is_transient_api_error(exc) is True


def test_5xx_status_is_retryable_4xx_is_not() -> None:
    server = type("BadGateway", (Exception,), {})()
    server.status_code = 502  # type: ignore[attr-defined]
    client_err = type("BadRequest", (Exception,), {})()
    client_err.status_code = 400  # type: ignore[attr-defined]
    assert _is_transient_api_error(server) is True
    assert _is_transient_api_error(client_err) is False
    assert _is_transient_api_error(ValueError("bad source")) is False


def test_injected_retrying_is_used() -> None:
    """The transport defers to the injected ``retrying`` wrapper (tenacity in production)."""
    attempts = {"n": 0}

    class _Flaky:
        def create(self, **kwargs: object) -> _Message:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient boom")
            return _Message([_Block("ok")], None)

    client = _FakeClient()
    client.messages = _Flaky()  # type: ignore[assignment]

    def retry_thrice(fn):  # type: ignore[no-untyped-def]
        last: Exception | None = None
        for _ in range(3):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - test double mirrors tenacity reraise
                last = exc
        raise last  # pragma: no cover - not reached in this test

    out = _AnthropicTransport(
        client, "m", temperature=None, max_tokens=16, cache_system=True, retrying=retry_thrice
    )("S", "U")
    assert out == "ok"
    assert attempts["n"] == 2  # failed once, retried, succeeded


def test_make_retrying_disabled_returns_none() -> None:
    assert _make_retrying(0) is None
    assert _make_retrying(-1) is None


# --------------------------------------------------------------------------- #
# no-key path raises before importing the SDK (no network, no dependency)      #
# --------------------------------------------------------------------------- #
def test_make_anthropic_transport_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        make_anthropic_transport("claude-sonnet-4-6", "ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# OpenAI-compatible transport (OpenAI / Gemini / DeepSeek) — fake SDK client    #
# --------------------------------------------------------------------------- #
class _OAIChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _OAIUsage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _OAIResponse:
    #: The model id the provider REPORTS it served (``response.model``) — for OpenRouter this is the
    #: exact snapshot behind the requested slug (the R71 reproducibility anchor).
    model = "qwen/qwen3-coder-served-snapshot"

    def __init__(self, content: str, usage: object | None) -> None:
        self.choices = [_OAIChoice(content)]
        self.usage = usage


class _FakeOAIClient:
    """Stands in for an ``openai.OpenAI`` client: records ``chat.completions.create`` kwargs."""

    def __init__(self, content: str = "def reward(): return 0.0", usage: object | None = None) -> None:
        self.calls: list[dict] = []
        usage = _OAIUsage() if usage is None else usage
        outer = self

        class _Completions:
            def create(self, **kwargs: object) -> _OAIResponse:
                outer.calls.append(kwargs)
                return _OAIResponse(content, None if usage == "NONE" else usage)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _oai(client: object, **kw: object):  # type: ignore[no-untyped-def]
    from src.llm.client import _OpenAITransport
    defaults: dict = dict(temperature=1.0, retrying=None)
    defaults.update(kw)
    return _OpenAITransport(client, "gemini-3.5-flash", **defaults)  # type: ignore[arg-type]


def test_openai_transport_threads_temperature_and_message_shape() -> None:
    client = _FakeOAIClient()
    out = _oai(client)("SYS", "USER")
    assert out == "def reward(): return 0.0"
    call = client.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert call["temperature"] == 1.0


def test_openai_transport_omits_temperature_when_none() -> None:
    client = _FakeOAIClient()
    _oai(client, temperature=None)("S", "U")
    assert "temperature" not in client.calls[0]


def test_openai_transport_captures_usage() -> None:
    client = _FakeOAIClient()
    t = _oai(client)
    assert t.last_usage is None
    t("S", "U")
    assert t.last_usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


def test_openai_transport_usage_none_when_response_has_none() -> None:
    client = _FakeOAIClient(usage="NONE")
    t = _oai(client)
    t("S", "U")
    assert t.last_usage is None


def test_openai_transport_uses_injected_retrying() -> None:
    from src.llm.client import _OpenAITransport
    attempts = {"n": 0}

    class _Flaky:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: object) -> _OAIResponse:
                    attempts["n"] += 1
                    if attempts["n"] == 1:
                        raise RuntimeError("transient boom")
                    return _OAIResponse("ok", None)

    def retry_thrice(fn):  # type: ignore[no-untyped-def]
        last: Exception | None = None
        for _ in range(3):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise last  # pragma: no cover

    out = _OpenAITransport(_Flaky(), "m", temperature=None, retrying=retry_thrice)("S", "U")
    assert out == "ok"
    assert attempts["n"] == 2


# --------------------------------------------------------------------------- #
# Provider registry + build_transport (single dispatch point)                  #
# --------------------------------------------------------------------------- #
def test_default_key_env_per_provider() -> None:
    from src.llm.client import default_key_env
    assert default_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert default_key_env("gemini") == "GEMINI_API_KEY"
    assert default_key_env("openai") == "OPENAI_API_KEY"
    assert default_key_env("deepseek") == "DEEPSEEK_API_KEY"
    assert default_key_env("openrouter") == "OPENROUTER_API_KEY"
    assert default_key_env("NoNsEnSe") == "OPENAI_API_KEY"  # case-folded fallback


def test_build_transport_unknown_provider_raises() -> None:
    from src.llm.client import build_transport
    with pytest.raises(RuntimeError) as excinfo:
        build_transport("bogus", "m")
    assert "bogus" in str(excinfo.value)


def test_build_transport_gemini_routes_to_openai_compat(monkeypatch) -> None:
    """gemini -> the OpenAI SDK pointed at Gemini's OpenAI-compatible base URL + GEMINI_API_KEY."""
    import sys
    import types

    import src.llm.client as client_mod

    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    captured: dict = {}
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kw: captured.update(kw))
    )
    client_mod.build_transport("gemini", "gemini-3.5-flash")
    assert captured["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert captured["api_key"] == "k-test"
    assert captured["max_retries"] == 0


def test_build_transport_deepseek_routes_to_deepseek_base(monkeypatch) -> None:
    import sys
    import types

    import src.llm.client as client_mod

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k-ds")
    captured: dict = {}
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kw: captured.update(kw))
    )
    client_mod.build_transport("deepseek", "deepseek-chat")
    assert captured["base_url"] == "https://api.deepseek.com"


def test_build_transport_anthropic_uses_native_no_key(monkeypatch) -> None:
    """anthropic -> the native Anthropic transport (no-key raises naming ANTHROPIC_API_KEY)."""
    from src.llm.client import build_transport

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        build_transport("anthropic", "claude-opus-4-8")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# OpenRouter — the R71 secondary open-weights designer (Qwen3-Coder) wiring     #
# --------------------------------------------------------------------------- #
def test_build_transport_openrouter_routes_to_openrouter_base(monkeypatch) -> None:
    """openrouter -> the OpenAI SDK pointed at OpenRouter's base URL + OPENROUTER_API_KEY (R71)."""
    import sys
    import types

    import src.llm.client as client_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "k-or")
    captured: dict = {}
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kw: captured.update(kw))
    )
    client_mod.build_transport("openrouter", "qwen/qwen3-coder")
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "k-or"
    assert captured["max_retries"] == 0  # ADR-034: tenacity is the single backoff policy


def test_build_transport_openrouter_without_key_raises(monkeypatch) -> None:
    """Key-gated (R71): no OPENROUTER_API_KEY -> a clear RuntimeError naming the env var, no network."""
    from src.llm.client import build_transport

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        build_transport("openrouter", "qwen/qwen3-coder")
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_openrouter_request_body_and_served_model_anchor() -> None:
    """The request carries the pinned slug/messages/max_tokens; the response's ``model`` (the exact
    snapshot OpenRouter served) is captured on ``last_served_model`` — the R71 reproducibility anchor."""
    from src.llm.client import _OpenAITransport

    client = _FakeOAIClient(content="OK")
    t = _OpenAITransport(client, "qwen/qwen3-coder", temperature=None, retrying=None, max_tokens=64)  # type: ignore[arg-type]
    assert t.last_served_model is None
    out = t("SYS", "USER")
    assert out == "OK"
    call = client.calls[0]
    assert call["model"] == "qwen/qwen3-coder"
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert call["max_tokens"] == 64
    assert t.last_served_model == "qwen/qwen3-coder-served-snapshot"


def test_llmclient_pinned_openrouter_cfg_routes_to_registry(monkeypatch) -> None:
    """The R71 pinned config (provider=openrouter) defaults the key env and lazily routes through
    ``build_transport`` — the single dispatch point the campaign path uses."""
    import src.llm.client as client_mod
    from src.llm.client import FakeTransport, LLMClient

    cfg = {"provider": "openrouter", "model": "qwen/qwen3-coder"}
    client = LLMClient(cfg)
    assert client.api_key_env == "OPENROUTER_API_KEY"

    routed: dict = {}

    def _fake_build(provider, model, key_env, **kw):  # type: ignore[no-untyped-def]
        routed.update(provider=provider, model=model, key_env=key_env)
        return FakeTransport(response="OK")

    monkeypatch.setattr(client_mod, "build_transport", _fake_build)
    assert client.complete("S", "U") == "OK"
    assert routed == {
        "provider": "openrouter", "model": "qwen/qwen3-coder", "key_env": "OPENROUTER_API_KEY",
    }


def test_openrouter_429_is_retried_by_tenacity() -> None:
    """A mocked 429 (RateLimitError) is retried by the REAL tenacity policy from ``_make_retrying``
    and the call succeeds on the second attempt (ADR-034 backoff wiring)."""
    from src.llm.client import _OpenAITransport

    rate_limit_cls = type("RateLimitError", (Exception,), {"status_code": 429})
    attempts = {"n": 0}

    class _RateLimited:
        class chat:  # noqa: N801 - mirrors the SDK surface
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: object) -> _OAIResponse:
                    attempts["n"] += 1
                    if attempts["n"] == 1:
                        raise rate_limit_cls("429 too many requests")
                    return _OAIResponse("OK", None)

    t = _OpenAITransport(
        _RateLimited(), "qwen/qwen3-coder",  # type: ignore[arg-type]
        temperature=None, retrying=_make_retrying(3), max_tokens=64,
    )
    assert t("S", "U") == "OK"
    assert attempts["n"] == 2  # 429 once, retried, succeeded


# --------------------------------------------------------------------------------------------- #
# 2026-07-19 AUTOMATIC KEY FAILOVER: a key-dead error (credit exhausted / 401)          #
# switches PERMANENTLY to <primary>_FALLBACK and retries once; everything else is untouched.     #
# --------------------------------------------------------------------------------------------- #
class _FakeApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _FoFakeMessage:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.model = "claude-opus-4-8"
        self.usage = None


class _FoFakeClient:
    """messages.create raises each exception in `errors` first, then returns `text`."""

    def __init__(self, text, errors=()):
        self.calls = 0
        self._errors = list(errors)
        self._text = text
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls += 1
                if outer._errors:
                    raise outer._errors.pop(0)
                return _FoFakeMessage(outer._text)

        self.messages = _Messages()


def _mk_transport(primary, factory, fallback_env="TEST_FALLBACK_KEY"):
    from src.llm.client import _AnthropicTransport

    return _AnthropicTransport(
        primary, "claude-opus-4-8", temperature=None, max_tokens=64,
        cache_system=False, retrying=None, fallback_key_env=fallback_env,
        client_factory=factory,
    )


def test_failover_on_credit_exhaustion_is_sticky_and_returns_the_completion(monkeypatch):
    monkeypatch.setenv("TEST_FALLBACK_KEY", "sk-fallback")
    primary = _FoFakeClient("never", errors=[_FakeApiError(400, "Your credit balance is too low")])
    fallback = _FoFakeClient("from-fallback")
    built = []
    t = _mk_transport(primary, lambda key: built.append(key) or fallback)
    assert t("sys", "user") == "from-fallback"
    assert built == ["sk-fallback"]          # the factory got the FALLBACK key
    assert t("sys", "again") == "from-fallback"   # sticky: later calls stay on the fallback
    assert primary.calls == 1 and fallback.calls == 2


def test_failover_on_401_auth_error(monkeypatch):
    monkeypatch.setenv("TEST_FALLBACK_KEY", "sk-fallback")
    primary = _FoFakeClient("never", errors=[_FakeApiError(401, "invalid x-api-key")])
    fallback = _FoFakeClient("ok")
    t = _mk_transport(primary, lambda key: fallback)
    assert t("s", "u") == "ok"


def test_no_fallback_env_means_unchanged_raise(monkeypatch):
    monkeypatch.delenv("TEST_FALLBACK_KEY", raising=False)
    primary = _FoFakeClient("never", errors=[_FakeApiError(400, "credit balance is too low")])
    t = _mk_transport(primary, lambda key: (_ for _ in ()).throw(AssertionError("must not build")))
    with pytest.raises(_FakeApiError):
        t("s", "u")


def test_non_billing_errors_never_rotate(monkeypatch):
    monkeypatch.setenv("TEST_FALLBACK_KEY", "sk-fallback")
    for err in (_FakeApiError(400, "max_tokens exceeds model limit"),   # other 400: a real bug
                _FakeApiError(429, "rate limited"),                      # transient: tenacity's job
                _FakeApiError(529, "overloaded")):
        primary = _FoFakeClient("never", errors=[err])
        t = _mk_transport(primary, lambda key: (_ for _ in ()).throw(AssertionError("must not rotate")))
        with pytest.raises(_FakeApiError):
            t("s", "u")


# --------------------------------------------------------------------------- #
# R106: the archive must record the REQUEST, not only what was served          #
# --------------------------------------------------------------------------- #
def test_archive_records_the_request_pins_so_a_replayer_needs_no_second_artifact() -> None:
    """Strict reproducibility (the project's first priority; Stefan's criterion 3): the replay
    archive must stand ALONE.

    Every archived field described what the provider SERVED (served_model, served_provider, usage,
    stop_reason). Nothing recorded what was ASKED FOR, so an independent replayer had to open
    `config/legs.yaml` at the frozen commit to learn the decoding configuration — a second artifact,
    and a chain that only holds while the freeze cross-check does.

    It also collapsed a distinction that matters: under a disable pin, `reasoning_tokens > 0` was
    indistinguishable from "the pin was never sent". With the request recorded, those are now
    different observations. This is the R85/R103 lesson (a pin nobody can verify is fictional)
    applied to the request side.
    """
    from dataclasses import asdict

    from src.llm.client import LLMClient

    client = _FakeClient()
    transport = _transport(client, thinking={"type": "disabled"})
    archive: list = []
    LLMClient({"model": "claude-sonnet-4-6"}, transport=transport, archive=archive).complete("S", "U")

    rec = asdict(archive[0])
    pins = rec["request_pins"]
    assert pins is not None, "the archive does not record what was requested"
    assert pins["thinking"] == {"type": "disabled"}      # the R106 pin, per call
    assert pins["max_tokens"] == 4096
    assert "temperature" in pins


def test_request_pins_are_absent_not_fabricated_when_a_transport_declares_none() -> None:
    """A stub/fake with no pins must archive None — never an invented default.

    Fabricating a plausible-looking config would be worse than recording nothing: it would make the
    archive assert a decoding configuration that was never sent.
    """
    from dataclasses import asdict

    from src.llm.client import FakeTransport, LLMClient

    archive: list = []
    LLMClient({"model": "m"}, transport=FakeTransport("x"), archive=archive).complete("S", "U")
    assert asdict(archive[0])["request_pins"] is None
