"""Cross-provider LLM spend LEDGER — advisory tracking, never a gate (R81 softened by R83).

Records every LLM call's realized cost (captured from the provider response where available,
else computed from token usage x the planning prices) to a crash-safe JSONL ledger, and reports
totals against the ADVISORY planning ceiling. Per the standing instruction (2026-07-21) this module
NEVER refuses or blocks a call: it warns loudly at thresholds and the spending decision rests
with the researcher. The design's integrity-critical exogenous stops are the seed-rung rule and
the leg calendar gate — not the dollar figure; the ledger's scientific role is the transparency
metric ("the whole study's realized LLM spend was $X", reported in CH4/CH6 and the NatWest
brief).

Crash-safety: one JSON object per line, appended atomically per call (open-append-close); a torn
final line is skipped on read with a logged warning (never sinks an analysis).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

#: Advisory warning thresholds as fractions of the planning ceiling (0.8 -> warn at 80%).
WARN_FRACTIONS: tuple[float, ...] = (0.8, 1.0)

#: Fallback planning ceiling (USD) when config/preregistration.yaml is unreadable.
DEFAULT_CEILING_USD: float = 30.0


def planning_ceiling(repo_root: Path | None = None) -> float:
    """The ADVISORY ceiling from ``config/preregistration.yaml: model_suite.spend_ceiling_usd``.

    Falls back to :data:`DEFAULT_CEILING_USD` with a logged warning if unreadable — the ledger
    must never crash a run over a config read (advisory means advisory).
    """
    try:
        import yaml

        root = repo_root or Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
        return float(cfg["model_suite"]["spend_ceiling_usd"])
    except Exception as exc:  # noqa: BLE001 — advisory path: warn, never crash
        _LOG.warning("spend_ledger: could not read the planning ceiling (%r); using $%.2f",
                     exc, DEFAULT_CEILING_USD)
        return DEFAULT_CEILING_USD


def estimate_cost_usd(
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
    prices: dict[str, Any] | None = None,
) -> float | None:
    """Estimate one call's cost from tokens × the legs.yaml ``planning_prices`` ($/MTok in, out).

    The REALIZED per-call cost from the provider response is always the authority when present
    (OpenRouter ``usage.cost``); this estimator covers providers that do not return cost (the
    native Anthropic transport). Returns None when the model is unpriced or tokens are absent —
    the caller records the call with an explicit note rather than a fabricated number.
    """
    if tokens_in is None and tokens_out is None:
        return None
    if prices is None:
        try:
            import yaml

            root = Path(__file__).resolve().parents[2]
            cfg = yaml.safe_load((root / "config" / "legs.yaml").read_text(encoding="utf-8")) or {}
            prices = dict(cfg.get("planning_prices") or {})
        except Exception as exc:  # noqa: BLE001 — advisory path: warn, never crash
            _LOG.warning("spend_ledger: could not read planning_prices (%r)", exc)
            return None
    pair = prices.get(model)
    if not pair:
        return None
    p_in, p_out = float(pair[0]), float(pair[1])
    return (int(tokens_in or 0) / 1e6) * p_in + (int(tokens_out or 0) / 1e6) * p_out


def record_spend(
    ledger_path: str | Path,
    *,
    provider: str,
    model: str,
    cost_usd: float,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    note: str = "",
    stop_reason: str | None = None,
) -> float:
    """Append one call's realized cost; return the new cumulative total (USD).

    Emits a WARNING log line when the cumulative total first crosses a
    :data:`WARN_FRACTIONS` threshold of the advisory ceiling. NEVER raises on
    threshold contact (R83): the caller keeps full control.
    """
    if cost_usd < 0:
        raise ValueError(f"cost_usd must be >= 0, got {cost_usd!r}")
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = spend_summary(path)["total_usd"]
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": str(provider),
        "model": str(model),
        "cost_usd": float(cost_usd),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "note": note,
        # 2026-07-28: the provider's own completion verdict, persisted STRUCTURALLY. A value in
        # client._INCOMPLETE_STOP_REASONS ({max_tokens, refusal, length, content_filter}) means the
        # completion was CUT OFF — which arrives at the sandbox as "source defines no callable named
        # 'reward'", i.e. indistinguishable from a model that could not write the code. Without this
        # field the per-model AUTHORING-RELIABILITY table (a registered deliverable) cannot separate
        # "the model failed" from "our cap truncated it", and attribution degrades to grepping
        # driver logs. It was previously captured by the client and only WARN-logged.
        "stop_reason": stop_reason,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    total = before + float(cost_usd)
    ceiling = planning_ceiling()
    for frac in WARN_FRACTIONS:
        threshold = frac * ceiling
        if before < threshold <= total:
            _LOG.warning(
                "spend_ledger: cumulative LLM spend $%.2f has crossed %.0f%% of the ADVISORY "
                "$%.2f planning ceiling (advisory only — no action taken; R83).",
                total, frac * 100, ceiling,
            )
    return total


def spend_summary(ledger_path: str | Path) -> dict[str, Any]:
    """Totals from the ledger: overall, by provider, by model, and the call count.

    A torn final line (crash mid-append) is skipped with a warning — reporting must not sink.
    """
    path = Path(ledger_path)
    total = 0.0
    by_provider: dict[str, float] = {}
    by_model: dict[str, float] = {}
    n_calls = 0
    if path.is_file():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                _LOG.warning("spend_ledger: skipping unparseable ledger line %d (torn append?)", i + 1)
                continue
            c = float(row.get("cost_usd", 0.0))
            total += c
            by_provider[row.get("provider", "?")] = by_provider.get(row.get("provider", "?"), 0.0) + c
            by_model[row.get("model", "?")] = by_model.get(row.get("model", "?"), 0.0) + c
            n_calls += 1
    return {
        "total_usd": round(total, 6),
        "by_provider": {k: round(v, 6) for k, v in sorted(by_provider.items())},
        "by_model": {k: round(v, 6) for k, v in sorted(by_model.items())},
        "n_calls": n_calls,
        "advisory_ceiling_usd": planning_ceiling(),
    }
