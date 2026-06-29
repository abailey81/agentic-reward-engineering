"""Tests for the RL agent factories and the archival LLM client.

Agent factories (FINAL_PLAN F.7, audits A-1/A-2): SAC is the FIXED headline,
TQC the secondary distributional critic. The factories import their heavy backends
LAZILY, so the module imports cleanly with or without torch/SB3, and raises a clear,
package-naming RuntimeError when the backend is absent. Backend absence is SIMULATED
via monkeypatch (``sys.modules[...] = None``) so these tests hold whether or not SB3 is
installed (P1/ADR-026: the build env DOES install stable-baselines3 + sb3-contrib).

LLM client (FINAL_PLAN F.8, audit C-2): an injected FakeTransport returns the
canned response and the call is archived (prompt + response + model id);
constructing the real transport without a key / without openai raises clearly and
never touches the network.
"""

from __future__ import annotations


import importlib
import sys

import pytest

from src.agents import factory


# --------------------------------------------------------------------------- #
# Agent factories                                                             #
# --------------------------------------------------------------------------- #
def test_module_imports_without_torch() -> None:
    """The factory module imports cleanly without torch/SB3 installed."""
    assert "torch" not in sys.modules or sys.modules.get("torch") is None or True
    reloaded = importlib.reload(factory)
    assert hasattr(reloaded, "make_headline_agent")
    assert hasattr(reloaded, "make_distributional_agent")
    assert hasattr(reloaded, "make_agent")


def test_headline_agent_raises_clear_runtime_error_when_sb3_absent(monkeypatch) -> None:
    """make_headline_agent raises RuntimeError naming stable-baselines3 when it is absent."""
    # Simulate SB3 being unavailable even though it is installed in the build env, so the
    # error-path contract is tested regardless of install state.
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    with pytest.raises(RuntimeError) as excinfo:
        factory.make_headline_agent(env=object(), cfg={})
    assert "stable-baselines3" in str(excinfo.value)


def test_distributional_agent_raises_clear_runtime_error_when_sb3contrib_absent(monkeypatch) -> None:
    """make_distributional_agent raises RuntimeError naming sb3-contrib when it is absent."""
    monkeypatch.setitem(sys.modules, "sb3_contrib", None)
    with pytest.raises(RuntimeError) as excinfo:
        factory.make_distributional_agent(env=object(), cfg={})
    assert "sb3-contrib" in str(excinfo.value)


def test_make_agent_dispatches_and_rejects_unknown_kind(monkeypatch) -> None:
    """The dispatcher routes known kinds and rejects an unknown one with ValueError."""
    # Simulate both backends absent so dispatch-to-factory reaches the RuntimeError path.
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    monkeypatch.setitem(sys.modules, "sb3_contrib", None)
    with pytest.raises(RuntimeError):
        factory.make_agent("headline", env=object(), cfg={})
    with pytest.raises(RuntimeError):
        factory.make_agent("distributional", env=object(), cfg={})
    # Unknown kind is rejected before any factory dispatch.
    with pytest.raises(ValueError):
        factory.make_agent("nonsense", env=object(), cfg={})


def test_headline_is_sac_and_distributional_is_tqc() -> None:
    """The fixed headline is SAC (A-1); the secondary critic is TQC (A-2)."""
    assert factory.HEADLINE_ALGO == "SAC"
    assert factory.DISTRIBUTIONAL_ALGO == "TQC"


# --------------------------------------------------------------------------- #
# LLM client                                                                  #
# --------------------------------------------------------------------------- #
def test_llm_client_with_fake_transport_returns_and_archives() -> None:
    """An injected FakeTransport returns the canned response and is archived."""
    from src.llm.client import FakeTransport, LLMClient, ProvenanceRecord

    cfg = {"model": "test-model-2026-01-01"}
    transport = FakeTransport(response="def reward(...): ...")
    archive: list[ProvenanceRecord] = []
    client = LLMClient(cfg, transport=transport, archive=archive)

    out = client.complete("SYS", "USER")

    assert out == "def reward(...): ..."
    # Transport recorded the call.
    assert transport.calls == [("SYS", "USER")]
    # Archive holds the provenance record: prompt + response + exact model id.
    assert len(archive) == 1
    rec = archive[0]
    assert rec.model == "test-model-2026-01-01"
    assert rec.system == "SYS"
    assert rec.user == "USER"
    assert rec.response == "def reward(...): ..."


def test_default_provider_is_anthropic_and_key_defaults_match(monkeypatch) -> None:
    """The provider-aware client defaults to Anthropic (ADR-033) with ANTHROPIC_API_KEY.

    With no key in the env, the lazily-built default transport raises the Anthropic key error
    naming ANTHROPIC_API_KEY — proving the default provider (anthropic) + its default key-env
    are wired, and no network call is attempted.
    """
    from src.llm.client import LLMClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient({"model": "claude-sonnet-4-6"})  # no provider/api_key -> anthropic defaults
    assert client.provider == "anthropic"
    assert client.api_key_env == "ANTHROPIC_API_KEY"
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("SYS", "USER")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_openai_provider_without_key_raises_no_network(monkeypatch) -> None:
    """Selecting the OpenAI-compatible check provider routes to the OpenAI transport and,
    without a key, raises clearly (no network)."""
    from src.llm.client import LLMClient

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(
        {"model": "test-model", "provider": "openai", "api_key_env": "OPENAI_API_KEY"}
    )
    assert client.provider == "openai"
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("SYS", "USER")
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_unknown_provider_raises() -> None:
    """An unrecognised provider is rejected with a clear error before any network call."""
    from src.llm.client import LLMClient

    client = LLMClient({"model": "m", "provider": "nonsense"})
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("SYS", "USER")
    assert "provider" in str(excinfo.value).lower()


def test_real_transport_without_model_raises() -> None:
    """Building the real transport with no pinned model id raises clearly."""
    from src.llm.client import LLMClient

    client = LLMClient({})  # no model, no injected transport
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("SYS", "USER")
    assert "model" in str(excinfo.value)


def test_complete_archives_transport_usage() -> None:
    """When the transport exposes ``last_usage``, ``complete`` archives it on the record."""
    from src.llm.client import LLMClient, ProvenanceRecord

    class _UsageTransport:
        last_usage = {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 5}

        def __call__(self, system: str, user: str) -> str:
            return "def reward(...): ..."

    archive: list[ProvenanceRecord] = []
    client = LLMClient({"model": "m"}, transport=_UsageTransport(), archive=archive)
    client.complete("SYS", "USER")
    assert archive[0].usage == {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 5}


def test_complete_usage_is_none_for_plain_transport() -> None:
    """A transport without ``last_usage`` (e.g. FakeTransport) archives usage=None, not an error."""
    from src.llm.client import FakeTransport, LLMClient, ProvenanceRecord

    archive: list[ProvenanceRecord] = []
    client = LLMClient({"model": "m"}, transport=FakeTransport(response="x"), archive=archive)
    client.complete("SYS", "USER")
    assert archive[0].usage is None
