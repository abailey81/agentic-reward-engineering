"""One-call Opus authoring smoke — the pre-launch key/model/balance live check (~$0.05).

Campaign-day pre-flight (CAMPAIGN_DAY_RUNBOOK step 4): the campaign's FIRST real Opus call
otherwise happens mid-C1, after the canary — a dead key, an unfunded account, or a renamed
model snapshot would then burn queue time and stall the reflection chain behind the 2 h
authoring-outage budget. This makes exactly ONE tiny call through the SAME provider/model/
key-env plumbing the campaign authoring path uses (``build_transport`` from the campaign
llm block — nothing re-implemented), prints the round-trip + token usage, and exits 0/1.

Usage::

    python scripts/author_smoke.py                 # campaign.yaml llm block (the real check)
    python scripts/author_smoke.py --from prototype  # prototype.yaml block (rehearsals)
    python scripts/author_smoke.py --fallback      # smoke the <key_env>_FALLBACK key instead

``--fallback`` (2026-07-20, the key-failover follow-through): validates the SECOND key end-to-end
through the identical plumbing. With the primary near-exhausted, the fallback IS the plan — a
dead/unfunded fallback must be discovered HERE, not mid-campaign at the exact moment the primary
dies. Run BOTH smokes at pre-flight step 4.
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="One-call authoring smoke (key/model/balance).")
    p.add_argument("--from", dest="source", choices=["campaign", "prototype"], default="campaign",
                   help="Which config's llm block to smoke (default: campaign — the real check).")
    p.add_argument("--fallback", action="store_true",
                   help="Smoke the <key_env>_FALLBACK key instead of the primary (validates the "
                        "failover key end-to-end BEFORE launch).")
    args = p.parse_args(argv)

    import os

    from src.llm.client import build_transport, default_key_env
    from src.utils.config import cfg_get, load_config
    from src.utils.env import load_env

    load_env()
    blk = dict(cfg_get(load_config(args.source), "llm", {}) or {})
    if not blk:
        print(f"[smoke] FAIL: config/{args.source}.yaml has no llm block", flush=True)
        return 1
    provider = str(blk.get("provider", ""))
    model = str(blk.get("model_snapshot", ""))
    key_env = blk.get("api_key_env")
    if args.fallback:
        resolved_env = str(key_env or default_key_env(provider))
        fb_env = f"{resolved_env}_FALLBACK"
        fb_key = os.environ.get(fb_env, "").strip()
        if not fb_key:
            print(f"[smoke] FAIL: --fallback but {fb_env} is unset/empty in the environment/.env "
                  f"— the failover key does not exist; set it before relying on failover",
                  flush=True)
            return 1
        # Process-local swap: the transport reads the primary env var, so pointing it at the
        # fallback value smokes the SECOND key through the byte-identical plumbing.
        os.environ[resolved_env] = fb_key
        key_env = resolved_env
        print(f"[smoke] FAILOVER-KEY MODE: smoking the key from {fb_env}")
    print(f"[smoke] provider={provider} model={model} key_env={key_env or '<default>'}")
    try:
        transport = build_transport(provider, model, key_env, max_tokens=32)
        t0 = time.perf_counter()
        out = transport("Reply with exactly: OK", "Say OK.")
        secs = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 — the whole point is to surface this BEFORE launch
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(f"[smoke] OK in {secs:.1f}s — response: {str(out)[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
