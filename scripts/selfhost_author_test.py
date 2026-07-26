#!/usr/bin/env python
"""A5 EXHAUSTIVE authoring-reliability test for the self-hosted, HF-commit-pinned, bf16 Qwen leg.

It answers the two questions the "lineage's first fully-pinned author" claim rests on:
  1. Does the fully-pinned bf16 endpoint AUTHOR executable reward code, and at what rate? (the
     reliability figure -- the API-served fp8 variant measured ~17% for qwen3.5-9b; a faithful
     self-host should reproduce it, which is itself the serving-variant reproducibility evidence.)
  2. Is thinking-off actually SERVED? (the R103/R85 round-trip: no ``<think>`` block in the response
     AND reasoning_tokens==0 in the archived usage -- a pin nobody can verify is fictional.)

The scorer reuses ``src.sandbox.executor.extract_reward_source`` (no reimplementation); the certified
executable-yield (AST gate + sandbox exec) remains ``scripts/leg_gates.py``. This harness drives a
directly-constructed ``vllm_selfhost`` transport (VLLM_BASE_URL -> the served node), so it needs no
model_suite roster entry -- keeping the demonstration off the R101 lockstep seed ladder (no power cost).

Real run (on/after serving): python -m scripts.selfhost_author_test --served-model-name qwen3.5-9b --n 20
(requires VLLM_BASE_URL + VLLM_API_KEY in the env). The scorer + aggregator are unit-tested off-GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_DEF_REWARD = re.compile(r"def\s+reward\s*\(")


def score_response(text: str) -> dict[str, bool]:
    """Score ONE authored response: does it carry a ``reward`` function, and was thinking off?

    Reuses the executor's extractor so 'has code' means exactly what the campaign's sandbox accepts.
    ``no_thinking`` is the request-level round-trip check (Qwen wraps reasoning in ``<think>...</think>``);
    combined with reasoning_tokens==0 from the archived usage it proves the R103 pin functioned.
    """
    text = text or ""
    no_thinking = "<think>" not in text
    try:
        from src.sandbox.executor import extract_reward_source
        src = extract_reward_source(text)
    except Exception:  # noqa: BLE001 -- no extractable code (or extractor rejects) == not compliant
        src = None
    has_code = bool(src)
    defines_reward = bool(src and _DEF_REWARD.search(src))
    return {
        "has_code": has_code,
        "defines_reward": defines_reward,
        "no_thinking": no_thinking,
        "compliant": has_code and defines_reward,
    }


def run_selfhost_author_test(
    transport: Callable[[str, str], str], system: str, user: str, n: int,
    *, reasoning_tokens_of: Optional[Callable[[], Optional[int]]] = None,
) -> dict[str, Any]:
    """Make ``n`` authoring calls through ``transport`` and aggregate the reliability + thinking-off rates.

    ``reasoning_tokens_of`` (optional) reads the transport's last archived reasoning-token count after
    each call (the strongest thinking-off evidence); absent, ``no_thinking`` (no ``<think>``) is used.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        text = transport(system, user)
        row = score_response(text)
        if reasoning_tokens_of is not None:
            rt = reasoning_tokens_of()
            row["reasoning_tokens"] = rt
            row["reasoning_off"] = (rt == 0) if rt is not None else row["no_thinking"]
        rows.append(row)
    m = len(rows)
    return {
        "n": m,
        "compliance_rate": sum(r["compliant"] for r in rows) / m,
        "thinking_off_rate": sum(r.get("reasoning_off", r["no_thinking"]) for r in rows) / m,
        "rows": rows,
    }


def _load_prompts() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    system = (root / "prompts" / "system.txt").read_text(encoding="utf-8")
    user = (root / "prompts" / "initial_generation.txt").read_text(encoding="utf-8")
    return system, user


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover (needs the served endpoint)
    p = argparse.ArgumentParser(description="A5 authoring-reliability test vs the pinned bf16 Qwen endpoint.")
    p.add_argument("--served-model-name", dest="served", default="qwen3.5-9b",
                   help="the vLLM --served-model-name the endpoint exposes")
    p.add_argument("--n", type=int, default=10, help="authoring calls (leg_gates default is 10)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--baseline", type=float, default=0.17, help="the fp8-served reference rate to reproduce")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from src.llm.client import build_transport
    transport = build_transport(  # base_url REQUIRED from VLLM_BASE_URL (fails loud if unset)
        "vllm_selfhost", args.served, max_tokens=args.max_tokens, temperature=1.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    system, user = _load_prompts()
    summary = run_selfhost_author_test(
        transport, system, user, args.n,
        reasoning_tokens_of=lambda: (transport.last_usage or {}).get("reasoning_tokens"),
    )
    summary["served_model_name"] = args.served
    summary["fp8_baseline"] = args.baseline
    out = Path(args.out) if args.out else Path(f"selfhost_author_test-{args.served}.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
