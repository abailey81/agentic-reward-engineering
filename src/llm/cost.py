"""Report-only LLM cost + completion-integrity accounting from the replay archive.

Every reward-authoring call archives a :class:`src.llm.client.ProvenanceRecord` carrying the provider
``usage`` (input/output + prompt-cache write/read token counts), the ``stop_reason``, and the model id
(audit C-2). This module reduces a list of those records into an honest, cache-aware **USD cost** and a
**completion-integrity tally** (how many calls were truncated/refused) — both computed entirely from the
archive, so the campaign's compute-accounting is reproducible from the frozen run rather than estimated.

Cache-aware pricing follows Anthropic's published model: cached-write input tokens bill at 1.25x the base
input rate and cached-read input tokens at 0.1x (see ``shared/prompt-caching.md``); for this study the
shared prefix is sub-floor so caches never engage (cache token counts are 0), but the accounting is
correct either way. Report-only: DISJOINT from the frozen m=6 family; never gates a hypothesis.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["PRICES_PER_MTOK", "CACHE_WRITE_MULT", "CACHE_READ_MULT", "summarize_llm_cost"]

#: Anthropic list price per 1M tokens (input, output), Jan 2026 — keyed by substring of the model id.
#: Mirrors ``scripts/monitor.py``'s table; this one additionally prices cache write/read tokens.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "fable-5": (10.0, 50.0),
    "opus-5": (5.0, 25.0),    # confirmatory author (R102, 2026-07-25); same $5/$25 as 4.8
    "opus-4-8": (5.0, 25.0),  # retained: M2 generation-pair partner + legacy
    "opus-4-7": (5.0, 25.0),
    "opus-4-6": (5.0, 25.0),
    "sonnet-5": (2.0, 10.0),  # seated leg (R90/R92), ANTHROPIC-billed. Introductory pricing through
                              # 2026-08-31 (covers the campaign window), standard $3/$15 after — mirrors
                              # config/legs.yaml. ADDED 2026-07-26 (deep review): it was ABSENT
                              # from both Anthropic price tables, so every sonnet-5 call booked $0.00.
    "sonnet-4-6": (3.0, 15.0),  # legacy: the sonnet-4.6 LEG was removed by R92; kept for archived calls
    "haiku-4-5": (1.0, 5.0),
}
CACHE_WRITE_MULT = 1.25  # cached-write input tokens bill at 1.25x base input (5-min ephemeral)
CACHE_READ_MULT = 0.10   # cached-read input tokens bill at 0.10x base input


def _price(model: str | None) -> tuple[float, float] | None:
    if not model:
        return None
    m = model.lower()
    for key, price in PRICES_PER_MTOK.items():
        if key in m:
            return price
    return None


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) else 0


def summarize_llm_cost(
    records: Iterable[Any],
    *,
    incomplete_reasons: Iterable[str] = ("max_tokens", "refusal", "length", "content_filter"),
) -> dict[str, Any]:
    """Reduce archived LLM calls into cache-aware USD cost + a completion-integrity tally.

    Parameters
    ----------
    records :
        Iterable of :class:`~src.llm.client.ProvenanceRecord` (or any objects/dicts exposing ``model``,
        ``usage`` and ``stop_reason``). ``usage`` is the archived provider dict with keys
        ``input_tokens``/``output_tokens``/``cache_creation_input_tokens``/``cache_read_input_tokens``.
    incomplete_reasons :
        Stop/finish reasons that mean the call did NOT return a complete candidate (truncation/refusal);
        counted separately so a capped/refused call is not conflated with a genuine bad candidate.

    Returns
    -------
    dict
        ``{calls, costed_calls, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens,
        usd, by_model, incomplete, incomplete_by_reason}``. ``usd`` is ``None``-safe (0.0 when no
        record's model matches the price table). ``by_model`` maps each model id to its own sub-summary.
    """
    incomplete_set = set(incomplete_reasons)
    tot: dict[str, Any] = {
        "calls": 0, "costed_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_write_tokens": 0, "cache_read_tokens": 0, "usd": 0.0, "incomplete": 0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    incomplete_by_reason: dict[str, int] = {}

    for rec in records:
        model = getattr(rec, "model", None) if not isinstance(rec, dict) else rec.get("model")
        usage = getattr(rec, "usage", None) if not isinstance(rec, dict) else rec.get("usage")
        stop = getattr(rec, "stop_reason", None) if not isinstance(rec, dict) else rec.get("stop_reason")
        tot["calls"] += 1
        m = str(model or "?")
        bm = by_model.setdefault(m, {
            "calls": 0, "costed_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_write_tokens": 0, "cache_read_tokens": 0, "usd": 0.0, "incomplete": 0,
        })
        bm["calls"] += 1
        if stop in incomplete_set:
            tot["incomplete"] += 1
            bm["incomplete"] += 1
            incomplete_by_reason[str(stop)] = incomplete_by_reason.get(str(stop), 0) + 1
        if isinstance(usage, dict):
            it = _int(usage.get("input_tokens"))
            ot = _int(usage.get("output_tokens"))
            cw = _int(usage.get("cache_creation_input_tokens"))
            cr = _int(usage.get("cache_read_input_tokens"))
            for k, v in (("input_tokens", it), ("output_tokens", ot),
                         ("cache_write_tokens", cw), ("cache_read_tokens", cr)):
                tot[k] += v
                bm[k] += v
            price = _price(model)
            if price is not None:
                in_p, out_p = price
                usd = (it * in_p + cw * in_p * CACHE_WRITE_MULT + cr * in_p * CACHE_READ_MULT
                       + ot * out_p) / 1e6
                tot["usd"] += usd
                bm["usd"] += usd
                tot["costed_calls"] += 1
                bm["costed_calls"] += 1

    tot["usd"] = round(tot["usd"], 4)
    for bm in by_model.values():
        bm["usd"] = round(bm["usd"], 4)
    tot["by_model"] = by_model
    tot["incomplete_by_reason"] = incomplete_by_reason
    return tot
