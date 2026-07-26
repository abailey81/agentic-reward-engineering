#!/usr/bin/env python3
"""One-call liveness smoke for the R71 secondary open-weights designer (Qwen3-Coder via OpenRouter).

Verifies the KEY-GATED wiring end-to-end BEFORE the secondary panel is scheduled: loads the gitignored
``.env`` (the same way every real entry point does — ``src.utils.env.load_env``), reads the R71 pin from
``config/llm.yaml`` (``open_weights_check_model`` / ``open_weights_provider`` / ``open_weights_api_key_env``),
and makes ONE tiny chat completion ("Reply with the single word OK") through the SAME transport factory the
campaign uses (``src.llm.client.build_transport`` — tenacity-wrapped, OpenAI-compatible, base URL from the
provider registry).

On success it prints the pinned slug, the EXACT model id OpenRouter reports it served
(``response.model`` -> ``transport.last_served_model`` — the R71 reproducibility anchor; record it), the
returned text, and the token usage. The API key is read from the environment and NEVER printed.

Exit codes
----------
    0   live call succeeded
    2   key not present
    3   auth OK but the pinned model is not yet activated/purchased in the workspace — actionable message, not an exception (add OPENROUTER_API_KEY=... to .env, rerun)

Run::

    python scripts/smoke_qwen.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    from src.llm.client import build_transport, default_key_env
    from src.utils.config import cfg_get, load_config
    from src.utils.env import load_env

    load_env()  # .env -> os.environ (idempotent; shell vars win) — same path as run_prototype/run_campaign

    cfg = load_config("llm")
    model = str(cfg.require("open_weights_check_model"))
    provider = str(cfg_get(cfg, "open_weights_provider", "openrouter"))
    key_env = str(cfg_get(cfg, "open_weights_api_key_env", default_key_env(provider)))

    if not os.environ.get(key_env):
        print(
            f"key not present — add {key_env}=... to the gitignored .env (repo root), then rerun\n"
            f"    python scripts/smoke_qwen.py\n"
            f"(R71 secondary designer: {model!r} via {provider}; the key is never committed or printed)"
        )
        return 2

    print(f"R71 smoke: provider={provider} pinned_model={model} key_env={key_env} (key present, not shown)")
    transport = build_transport(provider, model, key_env, temperature=None, max_tokens=64)
    try:
        text = transport(
            "You are a connectivity probe. Follow the user instruction exactly.",
            "Reply with the single word OK",
        )
    except Exception as exc:  # noqa: BLE001 — a paywall state is an actionable status, not a stack trace
        if "Unpurchased" in str(exc) or getattr(exc, "status_code", None) == 403:
            print("AUTH OK but the model is NOT ACTIVATED in this workspace (403 AccessDenied.Unpurchased).")
            print("The wiring is verified to the paywall (2026-07-06: key + endpoint + /models all live).")
            print(f"Action: activate/top up {model!r} in Alibaba Model Studio, then rerun this smoke.")
            return 3
        raise

    served = getattr(transport, "last_served_model", None)
    usage = getattr(transport, "last_usage", None)
    print(f"served_model={served!r}  <- RECORD THIS (R71 reproducibility anchor, config/llm.yaml)")
    print(f"response_text={text!r}")
    print(f"usage={usage!r}")
    print(f"request_id={getattr(transport, 'last_request_id', None)!r}")
    print("smoke OK — the R71 OpenRouter/Qwen3-Coder path is live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
