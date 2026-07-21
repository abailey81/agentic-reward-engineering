"""Pre-launch per-leg gates (R80/R82): author smoke + format-compliance baseline + the
behavioural contamination screen. One command per leg (or --all), verdicts archived as JSONL.

Usage
-----
    # everything, all legs (needs OpenRouter/Anthropic credit; ~\\$1-2 total):
    python scripts/leg_gates.py --all --out outputs/leg_gates

    # one leg, screen only:
    python scripts/leg_gates.py --leg deepseek-v4-pro --only screen --out outputs/leg_gates

Gates
-----
1. **smoke** — one real call through the leg's PINNED transport (provider/quant/reasoning/max
   tokens): proves the pin route is live and the key valid. Archives served model + cost.
2. **compliance** — N (default 10) authoring calls with the REAL frozen prompts on a stub
   interface block; scores the parse rate (``extract_reward_source`` succeeds AND the source
   defines ``def reward(``). This is the per-model FORMAT-COMPLIANCE BASELINE the
   authoring-reliability table is interpreted against (R82): identical prompts for every model —
   the same-exam principle — so failures here are the model's, not the prompt's.
3. **screen** — the behavioural CONTAMINATION screen (registered; a screen, not a proof): three
   probe families over anonymised numeric arrays — (a) SERIES CONTINUATION (memorization probe: a
   real-looking anonymised return window; any confident claim to recognise/continue the actual
   series is a flag), (b) IDENTIFICATION (must NOT name a specific asset/date/event for
   anonymised numbers; hedged refusal = pass), (c) SYNTHETIC CANARY (a fabricated distinctive
   sequence; claiming to recognise it = confabulation flag). Scored by conservative keyword
   rules; every response archived verbatim for human review. VERDICT: pass / flag (a flag routes
   for review — the gate never silently drops a leg; the pre-declared fallback chain decides).

All calls ride the R83 advisory spend ledger. Offline testing: ``--transport-factory`` is a seam
(tests inject a fake); production builds from ``src.llm.legs`` + ``src.llm.client``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.legs import load_legs, transport_kwargs  # noqa: E402

#: Phrases that mark a CONFIDENT identification/recognition claim (case-insensitive). Hedged
#: language ("cannot determine", "anonymized", "impossible to say") passes; naming passes only
#: through the named-entity check below.
_CONFIDENT_MARKERS = (
    "this is ", "these are the returns of", "i recognize", "i recognise",
    "clearly the", "definitely", "this appears to be the stock",
)
#: Entities whose CONFIDENT appearance in an identification answer flags the screen. Deliberately
#: famous names — an anonymised integer-indexed array gives no legitimate basis to name any.
_ENTITY_MARKERS = (
    "lehman", "enron", "apple", "aapl", "tesla", "tsla", "nvidia", "nvda", "gamestop", "gme",
    "s&p", "sp500", "2008", "2020", "covid", "dot-com", "black monday",
)


def _flagged(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _CONFIDENT_MARKERS) or any(m in low for m in _ENTITY_MARKERS)


def _screen_probes(seed: int = 20260721) -> list[dict[str, str]]:
    """The three registered probe families (deterministic content; archived with the verdicts)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    window = ", ".join(f"{x:+.4f}" for x in rng.standard_normal(30) * 0.012)
    canary = ", ".join(f"{x:+.4f}" for x in np.linspace(-0.0917, 0.0917, 20))
    return [
        {"name": "continuation",
         "user": f"Here is a series of anonymized daily returns: [{window}]. "
                 "Do you recognize this specific series? If so, continue it for 5 more values "
                 "and name its source. If not, say so plainly."},
        {"name": "identification",
         "user": f"An anonymized asset produced these daily returns: [{window}]. "
                 "Which company is this, and over which dates? Answer directly."},
        {"name": "canary",
         "user": f"These daily returns are from a well-known market episode: [{canary}]. "
                 "Name the episode and the asset."},
    ]


#: Below this compliance rate a leg is flagged for review: a leg that authors little/no
#: contract-compliant code is unusable regardless of the contamination screen — this closes the
#: qwen3.5-9b gap (0.0 compliance was stamped "pass" because the verdict keyed off screen_flags only).
#: NOT an auto-fail (the pre-declared fallback chain decides), but never a silent "pass".
#: 2026-07-25: RAISED 0.5 → 1.0 = EXTREMELY STRICT. ANY authoring call that fails to emit a
#: contract-compliant reward now flags the leg for review — a leg must be PERFECTLY compliant to pass
#: silently. (Review, not block: the pre-declared fallback chain still decides the final roster.)
_COMPLIANCE_FLOOR = 1.0


def _norm_provider(name: str) -> str:
    """Fold a provider identifier for comparison: ``SiliconFlow`` == ``siliconflow`` == ``Silicon Flow``.

    The pin is OpenRouter's routing SLUG; the response reports a human DISPLAY NAME. Comparing them
    literally produced a permanent false MISROUTE on both qwen legs (measured 2026-07-26). Only case
    and separators are folded — a different provider (``Together``, ``DeepInfra``) still fails, which
    is the condition the check exists to catch.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def run_leg_gates(
    leg: dict[str, Any],
    out_dir: Path,
    *,
    which: tuple[str, ...] = ("smoke", "compliance", "screen"),
    n_compliance: int = 10,
    transport_factory: Callable[[dict[str, Any]], Any] | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Run the selected gates for one leg; archive per-call rows + return the verdict summary."""
    # ``_INCOMPLETE_STOP_REASONS`` is module-private but is the SINGLE SOURCE OF TRUTH for "this
    # completion is truncated/refused"; re-declaring the set here would be exactly the config<->code
    # drift this repo guards against elsewhere. Imported, not copied.
    from src.llm.client import _INCOMPLETE_STOP_REASONS, build_transport
    from src.llm.spend_ledger import record_spend
    from src.sandbox.executor import extract_reward_source

    label = leg["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{label}.jsonl"
    if transport_factory is None:
        transport_factory = lambda lg: build_transport(**transport_kwargs(lg))  # noqa: E731
    transport = transport_factory(leg)

    reasoning_seen: list[int] = []   # reasoning-token counts across calls -> pin round-trip verdict
    providers_seen: list[str] = []   # served providers across calls -> provider round-trip verdict (m3)
    #: R106: Anthropic reports NO reasoning_tokens, so the token round-trip is unmeasurable there.
    #: Extended thinking arrives as ``thinking`` CONTENT BLOCKS, and their count IS observable — the
    #: only direct evidence available for those legs that a disable pin actually took effect.
    thinking_blocks_seen: list[int] = []

    def _call(system: str, user: str, gate: str) -> str:
        text = transport(system, user)
        cost = getattr(transport, "last_cost_usd", None)
        usage = getattr(transport, "last_usage", None)
        served_provider = getattr(transport, "last_served_provider", None)
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "leg": label, "gate": gate, "user": user, "response": text,
            "served_model": getattr(transport, "last_served_model", None),
            "served_provider": served_provider,
            "stop_reason": getattr(transport, "last_stop_reason", None),
            # R85 round-trip evidence: full usage (incl. reasoning-token counts, now captured by
            # client.py) archived per call — the record of whether a reasoning/temperature pin
            # actually FUNCTIONED (an API silently ignoring a pass-through key leaves the pin fictional).
            "usage": usage,
            "cost_usd": cost,
        }
        _rt = (usage or {}).get("reasoning_tokens")
        if _rt is not None:
            reasoning_seen.append(int(_rt))
        _tb = (usage or {}).get("thinking_blocks")
        if _tb is not None:
            thinking_blocks_seen.append(int(_tb))
        if served_provider:
            providers_seen.append(str(served_provider))
        with rows_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if ledger_path is not None and cost is not None:
            record_spend(ledger_path, provider=str(leg["provider"]), model=str(leg["model"]),
                         cost_usd=float(cost), note=f"leg_gates:{gate}")
        return text

    summary: dict[str, Any] = {"leg": label, "model": leg["model"]}

    # row 30f (audit 2026-07-22, the $0-booking drift guard): every leg model id MUST resolve in
    # config/legs.yaml planning_prices — an unpriced id would silently book cost_usd=0.0 for the
    # whole leg ("cost-unknown(model-unpriced)"), corrupting the reported spend transparency.
    from src.llm.spend_ledger import estimate_cost_usd
    if estimate_cost_usd(str(leg["model"]), 1000, 1000) is None:
        summary["price_key"] = "MISSING->review"
        summary["screen_verdict"] = "review"
    else:
        summary["price_key"] = "ok"

    if "smoke" in which:
        text = _call("You are a terse assistant.", "Reply with exactly: SMOKE-OK", "smoke")
        summary["smoke_ok"] = "SMOKE-OK" in text
        # R85: surface the pin round-trip observables in the verdict — a leg registering a
        # reasoning pin whose smoke shows NO usage metadata gets an explicit review flag
        # (never silently trusted).
        usage = getattr(transport, "last_usage", None)
        summary["usage_observed"] = usage
        # The reasoning-pin round-trip VERDICT is computed after the gates from the reasoning-token
        # counts (a trivial smoke prompt is an unreliable reasoning trigger; the compliance calls fire it).

    if "compliance" in which:
        system = (Path("prompts/system.txt").read_text(encoding="utf-8"))
        user = Path("prompts/initial_generation.txt").read_text(encoding="utf-8").replace(
            "{ENV_INTERFACE}", "# 31 weights (incl cash), 30 returns; contract as stated above.")
        ok = 0
        for _ in range(int(n_compliance)):
            text = _call(system, user, "compliance")
            try:
                src = extract_reward_source(text)
                ok += bool(re.search(r"def\s+reward\s*\(", src))
            except Exception:  # noqa: BLE001 — a non-extractable response is simply a miss
                pass
        summary["compliance_rate"] = ok / max(1, int(n_compliance))
        summary["compliance_n"] = int(n_compliance)
        # A leg that authors little/no compliant code is UNUSABLE regardless of the contamination
        # screen (2026-07-25: qwen3.5-9b scored 0.0 yet passed because the verdict keyed off
        # screen_flags only). Gate the verdict on compliance too: below the floor -> review.
        if summary["compliance_rate"] < _COMPLIANCE_FLOOR:
            summary["compliance_verdict"] = (
                f"LOW->review ({summary['compliance_rate']:.0%} < {_COMPLIANCE_FLOOR:.0%} floor)")
            if summary.get("screen_verdict") in (None, "pass"):
                summary["screen_verdict"] = "review"

    if "screen" in which:
        flags: list[str] = []
        unusable: list[str] = []
        for probe in _screen_probes():
            text = _call(
                "You answer questions about financial data honestly and concisely.",
                probe["user"], f"screen:{probe['name']}")
            # Deep review 2026-07-26: the screen was FAIL-OPEN on absent evidence. A refusal /
            # content_filter / max_tokens completion is EMPTY or partial WITHOUT raising
            # (client.py `_INCOMPLETE_STOP_REASONS` + `_warn_if_incomplete`, which only WARN-logs
            # because it assumes the AST/compliance gate downstream catches it) — and `_flagged("")`
            # is False, so an all-empty screen certified "no contamination markers" having asked and
            # learnt NOTHING. Under `--only screen` no compliance gate runs to catch it. Same
            # discipline as the R85 pin block below: NEVER read absent evidence as a clean result.
            stop_reason = getattr(transport, "last_stop_reason", None)
            if not text.strip() or stop_reason in _INCOMPLETE_STOP_REASONS:
                unusable.append(f"{probe['name']}({stop_reason or 'empty'})")
            elif _flagged(text):
                flags.append(probe["name"])
        summary["screen_flags"] = flags
        summary["screen_unusable"] = unusable
        # Audit 2026-07-24 (MAJOR): never DOWNGRADE an existing review verdict — the price_key
        # guard sets "review" up front, and this line used to clobber it back to "pass" on a
        # clean screen, making the $0-booking guard inert in the documented --all invocation.
        if flags:
            screen_result = "FLAG->review"
        elif unusable:
            screen_result = (f"UNVERIFIED->review ({len(unusable)} probe(s) returned no usable "
                             f"answer: {unusable} — an unanswered screen is not a clean screen)")
        else:
            screen_result = "pass"
        if summary.get("screen_verdict") in (None, "pass"):
            summary["screen_verdict"] = screen_result
        elif flags:
            summary["screen_verdict"] = f"{summary['screen_verdict']}+FLAG"
        elif unusable:
            summary["screen_verdict"] = f"{summary['screen_verdict']}+UNVERIFIED"

    # R85 pin round-trip VERIFICATION (repro-audit 2026-07-25, HOLE 2): reasoning-token counts are
    # now archived (client.py), so a reasoning pin's EFFECT is measurable instead of assumed.
    # THREE-STATE, direction-aware (repro-audit fix M1/m1 — never claim verified OR fictional on
    # ABSENT evidence): (a) NO reasoning_tokens field on any call -> UNMEASURABLE -> UNVERIFIED for
    # BOTH directions; (b) a MEASURED count -> an ENABLE pin (mode/effort/budget) must be >0 (0 on
    # real authoring calls = silently ignored = FICTIONAL), a DISABLE pin ({"enabled": false}) must
    # be ~0 (>0 = IGNORED); (c) a trivial smoke prompt may not trigger reasoning, so a measured 0 on
    # an ENABLE pin is only conclusive when a compliance (authoring) call ran.
    reasoning_pin = leg.get("reasoning")
    if reasoning_pin is not None:
        max_rt = max(reasoning_seen) if reasoning_seen else None
        summary["reasoning_tokens_observed"] = max_rt
        disabling = isinstance(reasoning_pin, dict) and reasoning_pin.get("enabled") is False
        measured = max_rt is not None            # did ANY call return a reasoning_tokens field?
        reasoned = bool(max_rt and max_rt > 0)   # a POSITIVE reasoning-token count was measured
        # R106: a provider that reports no reasoning_tokens (Anthropic) can still be adjudicated when
        # it exposes thinking CONTENT BLOCKS — their absence across every call is direct evidence the
        # disable took effect. Strictly weaker than a token count, so it says so in the verdict, and
        # it is only ever accepted for a DISABLE pin: absence of thinking blocks cannot prove an
        # ENABLE pin functioned.
        blocks_measured = bool(thinking_blocks_seen)
        max_tb = max(thinking_blocks_seen) if blocks_measured else None
        summary["thinking_blocks_observed"] = max_tb
        if not measured and disabling and blocks_measured and max_tb == 0:
            rt_verdict = (f"verified via content blocks (reasoning disabled; 0 thinking blocks in "
                          f"{len(thinking_blocks_seen)} response(s); this provider reports no "
                          "reasoning_tokens, so block absence is the available evidence)")
        elif not measured and disabling and blocks_measured:
            rt_verdict = (f"IGNORED->review (reasoning-DISABLE pinned but {max_tb} thinking block(s) "
                          "returned)")
        elif not measured:
            rt_verdict = ("UNVERIFIED->review (no reasoning_tokens field returned by the provider — "
                          "the pin's effect cannot be confirmed from the archive)")
        elif disabling:
            rt_verdict = (
                f"IGNORED->review (reasoning-DISABLE pinned but {max_rt} reasoning tokens served)"
                if reasoned else "verified (reasoning disabled; 0 reasoning tokens measured)")
        elif reasoned:
            rt_verdict = f"verified ({max_rt} reasoning tokens served; enable pin functioned)"
        elif "compliance" in which:
            rt_verdict = ("FICTIONAL->review (reasoning-ENABLE pinned but 0 reasoning tokens MEASURED "
                          "on authoring calls — pin silently ignored)")
        else:
            rt_verdict = ("UNVERIFIED->review (reasoning-ENABLE pinned; 0 reasoning tokens on the smoke "
                          "call — run --compliance to verify)")
        summary["pin_roundtrip"] = rt_verdict
        if "->review" in rt_verdict and summary.get("screen_verdict") in (None, "pass"):
            summary["screen_verdict"] = "review"

    # m3 (repro-audit 2026-07-25): adjudicate the SERVED PROVIDER against the provider_pin, symmetric
    # with the reasoning verdict — a silent mis-route (served provider not in the pinned only-list) now
    # flags review instead of merely being archived for a human to notice.
    pin_only = [str(x) for x in ((leg.get("provider_pin") or {}).get("only") or [])]
    if pin_only:
        served = {p for p in providers_seen if p}
        # NORMALISE before comparing (2026-07-26). The pin carries OpenRouter's routing SLUG
        # (`siliconflow`) but the response reports the DISPLAY NAME (`SiliconFlow`), so a literal
        # set difference declared MISROUTE on both qwen legs while the routing was in fact exactly
        # as pinned. That is worse than cosmetic: a check that cries wolf on every run trains the
        # operator to ignore it, and this is the check that would catch a REAL silent re-route to a
        # provider serving a different quantization — the R85/Stefan-#3 reproducibility risk it
        # exists for. Case and separators are folded; a genuinely different provider still fails.
        unmatched = sorted(p for p in served if _norm_provider(p) not in
                           {_norm_provider(q) for q in pin_only})
        if not served:
            summary["provider_roundtrip"] = "unrecorded (no served_provider field returned)"
        elif unmatched:
            summary["provider_roundtrip"] = (
                f"MISROUTE->review (served {unmatched} not in pin {pin_only})")
            if summary.get("screen_verdict") in (None, "pass"):
                summary["screen_verdict"] = "review"
        else:
            summary["provider_roundtrip"] = f"verified (served {sorted(served)} within pin)"

    (out_dir / f"{label}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — these gates cost real money; never lose one to a codepage
    p = argparse.ArgumentParser(description="Pre-launch per-leg gates (smoke/compliance/screen).")
    p.add_argument("--leg", help="One leg label from config/legs.yaml.")
    p.add_argument("--all", action="store_true", help="Run every leg in queue order.")
    p.add_argument("--only", choices=["smoke", "compliance", "screen"],
                   help="Run a single gate instead of all three.")
    p.add_argument("--n-compliance", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", default="outputs/spend_ledger.jsonl")
    args = p.parse_args(argv)

    # Check-day catch (2026-07-23): the campaign drivers load .env themselves but this script
    # did NOT — a fresh shell crashed on the unset OPENROUTER_API_KEY. Load it here, same as
    # the drivers (keys stay in the gitignored .env, never in code or logs).
    from src.utils.env import load_env
    load_env()
    legs = load_legs()["legs"]
    if args.all:
        chosen = legs
    else:
        # Audit m4: an unknown --leg used to die with a raw StopIteration.
        chosen = [lg for lg in legs if lg["label"] == args.leg]
        if not chosen:
            raise SystemExit(f"--leg {args.leg!r}: no such leg; valid: "
                             f"{[lg['label'] for lg in legs]}")
    which = (args.only,) if args.only else ("smoke", "compliance", "screen")
    out = Path(args.out)
    verdicts = []
    for leg in chosen:
        try:
            s = run_leg_gates(leg, out, which=which, n_compliance=args.n_compliance,
                              ledger_path=Path(args.ledger))
        except Exception as exc:  # noqa: BLE001 - one leg's API error must not kill the other nine (check-day catch)
            s = {"leg": leg.get("label", "?"), "error": f"{type(exc).__name__}: {exc}"[:300],
                 "screen_verdict": "ERROR->review"}
            # Audit m4: persist the ERROR verdict so a stale previous summary can never be
            # read as current evidence (T7/F14 consume these files).
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{s['leg']}.summary.json").write_text(
                json.dumps(s, indent=2), encoding="utf-8")
            print(f"[gate] {s['leg']}: ERROR - {s['error']}")
        verdicts.append(s)
        print(json.dumps(s))
    flagged = [v["leg"] for v in verdicts if v.get("screen_verdict", "pass") != "pass"]
    if flagged:
        print(f"SCREEN FLAGS -> review (fallback chain decides): {flagged}")
    # Audit m4: hard errors are an exit-code signal (flags stay rc 0 — they are review-routed,
    # not failures); an all-errored run must not read as success to a calling script.
    return 1 if any(v.get("error") for v in verdicts) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
