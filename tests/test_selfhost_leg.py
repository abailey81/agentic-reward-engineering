"""The A5 self-hosted vLLM Qwen leg (the lineage's first fully-pinned open-weight author): provider
dispatch (base_url REQUIRED from the env + fail-loud) and the transport-kwargs translation (thinking-off
pinned at the request level via Qwen3's chat template; none of the OpenRouter aggregator extras leak in).
No network: build_transport constructs the OpenAI client but makes no call."""
import pytest

from src.llm.client import PROVIDERS, build_transport, default_key_env
from src.llm.legs import transport_kwargs


def test_vllm_selfhost_is_a_registered_openai_compatible_provider() -> None:
    assert "vllm_selfhost" in PROVIDERS
    assert default_key_env("vllm_selfhost") == "VLLM_API_KEY"


def test_vllm_selfhost_requires_base_url_and_fails_loud(monkeypatch) -> None:
    # An unset VLLM_BASE_URL must NOT silently fall back to the real OpenAI endpoint.
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
        build_transport("vllm_selfhost", "qwen3.5-9b", max_tokens=4096)


def test_vllm_selfhost_routes_to_the_served_node(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("VLLM_BASE_URL", "http://node42:8000/v1")
    t = build_transport("vllm_selfhost", "qwen3.5-9b", max_tokens=4096)
    assert "node42:8000" in str(t._client.base_url)  # the served node, not api.openai.com


def test_transport_kwargs_pins_thinking_off_with_no_openrouter_extras() -> None:
    leg = {
        "label": "qwen3.5-9b-selfhost", "provider": "vllm_selfhost", "model": "qwen3.5-9b",
        "api_key_env": "VLLM_API_KEY", "max_tokens": 4096, "temperature": 1.0,
        "reasoning": {"enabled": False},
        # these OpenRouter-only fields must be IGNORED for a self-hosted leg:
        "provider_pin": {"only": ["siliconflow"]}, "quantizations": ["bf16"],
    }
    kw = transport_kwargs(leg)
    assert kw["provider"] == "vllm_selfhost" and kw["model"] == "qwen3.5-9b"
    assert kw["max_tokens"] == 4096 and kw["temperature"] == 1.0
    # thinking-off enforced explicitly at the request level (the R103 pin, verifiable via the archive):
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # NONE of the OpenRouter aggregator extras leak in (vLLM would reject provider-routing/usage):
    assert "provider" not in kw["extra_body"] and "usage" not in kw["extra_body"]
    assert "quantizations" not in str(kw["extra_body"])


def test_transport_kwargs_pins_thinking_on_when_enabled() -> None:
    leg = {"label": "x", "provider": "vllm_selfhost", "model": "m", "api_key_env": "VLLM_API_KEY",
           "max_tokens": 100, "reasoning": {"enabled": True}}
    assert transport_kwargs(leg)["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
