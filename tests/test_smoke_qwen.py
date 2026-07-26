"""Tests for scripts/smoke_qwen.py — the R71 secondary open-weights designer smoke.

All offline: ``load_env`` is stubbed so a real gitignored ``.env`` cannot leak into the key-missing
test, and the transport factory is faked so the key-present path exercises the script's real flow
(config pin -> build_transport -> one call -> report) with no network. The PINNED provider/model/key
are read from ``config/llm.yaml`` AT TEST TIME (not hardcoded), so the assertions track the R71
provider decision — DashScope today (2026-07-06), OpenRouter the documented fallback — and never
drift again on a provider migration. Covered:

  - key MISSING -> exit code 2 with an actionable "add <KEY_ENV>=... to .env" message (NOT an
    exception), naming the pinned model;
  - key PRESENT -> exit code 0, routes the PINNED config through ``build_transport``, prints the
    served-model reproducibility anchor + response text + usage, and NEVER prints the key.
"""
from __future__ import annotations

from typing import Any

from scripts import smoke_qwen


def _stub_load_env(monkeypatch) -> None:
    """Keep the real gitignored .env out of the test (the script calls src.utils.env.load_env)."""
    import src.utils.env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)


def _pinned() -> tuple[str, str, str]:
    """The (provider, model, key_env) the script resolves from config/llm.yaml — resolved the SAME
    way main() does so the test asserts against the real pin, not a hardcoded provider."""
    from src.llm.client import default_key_env
    from src.utils.config import cfg_get, load_config

    cfg = load_config("llm")
    model = str(cfg.require("open_weights_check_model"))
    provider = str(cfg_get(cfg, "open_weights_provider", "openrouter"))
    key_env = str(cfg_get(cfg, "open_weights_api_key_env", default_key_env(provider)))
    return provider, model, key_env


def test_missing_key_exits_2_with_actionable_message(monkeypatch, capsys) -> None:
    _stub_load_env(monkeypatch)
    _provider, model, key_env = _pinned()
    monkeypatch.delenv(key_env, raising=False)  # force the missing-key path for the ACTUAL key_env
    rc = smoke_qwen.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert key_env in out and ".env" in out  # actionable: what to set, where
    assert model in out  # names the pinned model so the operator knows what will run


def test_key_present_makes_one_call_and_reports_anchor(monkeypatch, capsys) -> None:
    _stub_load_env(monkeypatch)
    provider, model, key_env = _pinned()
    monkeypatch.setenv(key_env, "sk-SECRET-not-to-print")

    calls: list[tuple[str, str]] = []
    built: dict[str, Any] = {}

    class _FakeTransport:
        last_served_model = f"{model}-served-snapshot"
        last_usage = {"input_tokens": 21, "output_tokens": 1, "total_tokens": 22}
        last_request_id = "gen-123"

        def __call__(self, system: str, user: str) -> str:
            calls.append((system, user))
            return "OK"

    def _fake_build(provider, model, key_env, **kw):  # type: ignore[no-untyped-def]
        built.update(provider=provider, model=model, key_env=key_env)
        return _FakeTransport()

    import src.llm.client as client_mod

    monkeypatch.setattr(client_mod, "build_transport", _fake_build)
    rc = smoke_qwen.main()
    out = capsys.readouterr().out

    assert rc == 0
    # Routed the PINNED R71 config (config/llm.yaml) through the single dispatch point.
    assert built == {"provider": provider, "model": model, "key_env": key_env}
    # Exactly ONE tiny call carrying the OK instruction.
    assert len(calls) == 1
    assert "Reply with the single word OK" in calls[0][1]
    # Reports the reproducibility anchor + text + usage; the key is NEVER printed.
    assert f"{model}-served-snapshot" in out
    assert "'OK'" in out
    assert "output_tokens" in out
    assert "sk-SECRET-not-to-print" not in out
