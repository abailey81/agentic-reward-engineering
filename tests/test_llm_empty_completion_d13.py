"""D13 — a provider reply with no completion CONTAINER must be a retryable transport fault.

Regression for the campaign execution record §25 and the deferred-fix register item 1.

**What happened.** OpenRouter answered HTTP 200 twice on `nemotron-3-super` with `choices = None`.
`response.choices[0]` raised `TypeError: 'NoneType' object is not subscriptable`. The retry
predicate is duck-typed on API-error CLASS NAMES, so a `TypeError` is terminal to it: the exception
escaped the transport, propagated through `_complete_with_outage_tolerance`, and killed FIVE whole
arm pipelines — one of which (`leg7`) then ran 8 h 29 m with 3 of its 5 arms.

**The extension found on 2026-08-01 (RUN 10).** The deferred spec covered only the OpenAI path.
Grepping every response-extraction site found the identical defect on `_AnthropicTransport`
(`blocks = list(message.content)` -> `TypeError: 'NoneType' object is not iterable`) — which is the
transport the CONFIRMATORY core line runs on. Both are covered here.

**The line this must NOT cross, and it is the reason for `test_*_legitimately_empty_*` below.**
A response that is well-formed but says nothing — a truncation, a refusal, a completion carrying
only `thinking` blocks — is a LEGITIMATE authoring outcome. It must reach the archive as an
authoring failure, because that is the capability signal the per-model reliability result is
measured from. Retrying it would silently change which candidates exist. The predicate is
*the container is missing*, never *the text is empty*.
"""
from __future__ import annotations

import pytest

from src.llm.client import (
    EmptyCompletionError,
    _AnthropicTransport,
    _is_transient_api_error,
    _OpenAITransport,
)


# --------------------------------------------------------------------------- #
# Fakes — OpenAI-compatible                                                    #
# --------------------------------------------------------------------------- #
class _OAMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _OAChoice:
    def __init__(self, message: object | None, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _OAResponse:
    def __init__(self, choices: object) -> None:
        self.choices = choices
        self.id = "resp_test"
        self.model = "served/model"
        self.usage = None


class _OACompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.n_calls = 0

    def create(self, **_kw: object) -> object:
        self.n_calls += 1
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class _OAClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = type("_Chat", (), {})()
        self.completions = _OACompletions(responses)
        self.chat.completions = self.completions


def _oa(responses: list[object], retrying: object = None) -> _OpenAITransport:
    return _OpenAITransport(  # type: ignore[arg-type]
        _OAClient(responses), "test/model", temperature=None, retrying=retrying)


# --------------------------------------------------------------------------- #
# Fakes — Anthropic                                                            #
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, text: str, type_: str = "text") -> None:
        self.text = text
        self.type = type_


class _AntMessage:
    def __init__(self, content: object) -> None:
        self.content = content
        self.usage = None
        self.stop_reason = "end_turn"
        self.model = "claude-test"


class _AntMessages:
    def __init__(self, message: object) -> None:
        self._message = message

    def create(self, **_kw: object) -> object:
        return self._message


class _AntClient:
    def __init__(self, message: object) -> None:
        self.messages = _AntMessages(message)


def _ant(message: object) -> _AnthropicTransport:
    return _AnthropicTransport(  # type: ignore[arg-type]
        _AntClient(message), "claude-test",
        temperature=None, max_tokens=64, cache_system=False, retrying=None)


# --------------------------------------------------------------------------- #
# The malformed-body cases — each must be a NAMED, RETRYABLE fault             #
# --------------------------------------------------------------------------- #
def test_openai_choices_none_raises_named_transport_fault() -> None:
    """The exact live failure: HTTP 200 with `choices = None`.

    Pre-fix this raised `TypeError: 'NoneType' object is not subscriptable`, which
    `_is_transient_api_error` classifies as terminal.
    """
    with pytest.raises(EmptyCompletionError):
        _oa([_OAResponse(None)])("SYS", "USER")


def test_openai_choices_empty_list_raises_named_transport_fault() -> None:
    """`choices = []` is equally malformed — the API guarantees at least one choice.

    Pre-fix this raised `IndexError`, also terminal to the retry predicate.
    """
    with pytest.raises(EmptyCompletionError):
        _oa([_OAResponse([])])("SYS", "USER")


def test_openai_choice_without_message_raises_named_transport_fault() -> None:
    """A choice carrying no `message` object. Pre-fix: `AttributeError` on `.content`."""
    with pytest.raises(EmptyCompletionError):
        _oa([_OAResponse([_OAChoice(None)])])("SYS", "USER")


def test_anthropic_content_none_raises_named_transport_fault() -> None:
    """The CONFIRMATORY transport, same defect class.

    Pre-fix: `list(None)` -> `TypeError: 'NoneType' object is not iterable`.
    """
    with pytest.raises(EmptyCompletionError):
        _ant(_AntMessage(None))("SYS", "USER")


def test_anthropic_content_not_iterable_raises_named_transport_fault() -> None:
    """A scalar where a block list belongs. Pre-fix: a bare `TypeError`."""
    with pytest.raises(EmptyCompletionError):
        _ant(_AntMessage(42))("SYS", "USER")


def test_the_named_fault_is_classified_transient() -> None:
    """Without this the rename buys nothing — the whole point is that it RETRIES."""
    assert _is_transient_api_error(EmptyCompletionError("no choices")) is True
    # and the guard against over-broadening: an ordinary programming error stays terminal
    assert _is_transient_api_error(TypeError("not subscriptable")) is False


def test_a_retrying_wrapper_actually_recovers_from_the_fault() -> None:
    """End-to-end: malformed body on attempt 1, well-formed on attempt 2 -> the call succeeds.

    This is the property the campaign needed and did not have: five arm pipelines died because
    a recoverable transport fault was terminal.
    """
    from src.llm.client import _make_retrying

    good = _OAResponse([_OAChoice(_OAMessage("def reward(): return 0.0"))])
    t = _oa([_OAResponse(None), good], retrying=_make_retrying(3))
    assert t("SYS", "USER") == "def reward(): return 0.0"


# --------------------------------------------------------------------------- #
# THE SCIENCE GUARD — a legitimately empty completion must NOT be retried    #
# --------------------------------------------------------------------------- #
def test_openai_legitimately_empty_completion_is_returned_not_retried() -> None:
    """A well-formed reply whose text is empty is an AUTHORING failure, not a transport one.

    It must reach the archive as `""` so the per-model authoring-reliability measurement sees it.
    Retrying it would silently change which candidates exist — a forking path, not a fix.
    """
    t = _oa([_OAResponse([_OAChoice(_OAMessage(None), finish_reason="length")])])
    assert t("SYS", "USER") == ""
    assert t.last_stop_reason == "length"


def test_anthropic_thinking_only_completion_is_returned_not_retried() -> None:
    """Container present, zero TEXT blocks -> `""`, no exception. Same reasoning as above."""
    t = _ant(_AntMessage([_Block("reasoning...", type_="thinking")]))
    assert t("SYS", "USER") == ""


def test_anthropic_empty_block_list_is_returned_not_retried() -> None:
    """An empty block LIST is still a present container -> `""`, not a retry."""
    assert _ant(_AntMessage([]))("SYS", "USER") == ""


# --------------------------------------------------------------------------- #
# The same recovery property on the CONFIRMATORY transport                     #
# --------------------------------------------------------------------------- #
class _AntSequenceMessages:
    def __init__(self, messages: list[object]) -> None:
        self._messages = list(messages)

    def create(self, **_kw: object) -> object:
        return self._messages.pop(0) if len(self._messages) > 1 else self._messages[0]


class _AntSequenceClient:
    def __init__(self, messages: list[object]) -> None:
        self.messages = _AntSequenceMessages(messages)


def test_anthropic_retrying_wrapper_actually_recovers_from_the_fault() -> None:
    """Malformed body on attempt 1, well-formed on attempt 2 -> the core line's call succeeds.

    This is the test that caught the real defect: the deferred spec placed the check AFTER
    `self._retrying(_call)` returned, i.e. outside tenacity's scope, so the named error would
    have retried exactly zero times. Classification alone proves nothing -- recovery does.
    """
    from src.llm.client import _make_retrying

    good = _AntMessage([_Block("def reward(): return 0.0")])
    t = _AnthropicTransport(  # type: ignore[arg-type]
        _AntSequenceClient([_AntMessage(None), good]), "claude-test",
        temperature=None, max_tokens=64, cache_system=False, retrying=_make_retrying(3))
    assert t("SYS", "USER") == "def reward(): return 0.0"
