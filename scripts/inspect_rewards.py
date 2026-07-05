"""Qualitative reward inspection — the §6.1 GREEN-gate "open the black box" (FINAL_PLAN Phase 4.C).

Purpose
-------
Open the black box of LLM-authored reward functions. The headline metric gap (H2)
is necessary but not sufficient: §6.1 demands *qualitative* evidence that the
distributional feedback was actually USED. This tool produces DIRECTIONAL such evidence —
it COMPLEMENTS (does not replace) the CAUSAL test, the matched-budget ablation lattice
(distributional vs scalar / placebo / scalar_cvar5; cf. Eureka 2024 §4.3 reward-reflection ablation):

  1. ``per_generation_summary`` — per-arm-per-generation best/mean fitness plus a
     reward-source size/complexity trend and a tail-term-usage trend across
     generations (does the authored code grow more risk-aware as the loop runs?).
  2. ``feedback_responsiveness`` — within an arm, correlate successive reward-source
     EDITS (gen N → N+1) with the tail-statistic deltas the LLM was FED back in the
     gen-N ``feedback_block`` (a DIRECTIONAL "did it use the information" probe — the causal
     H2 test is the ablation lattice above; no number from it enters the dissertation). A
     responsive designer changes its code more when the distribution shifts more.
  3. ``hacking_taxonomy`` — tag candidates by specification-gaming / proxy / tautology
     signals through the reward-hacking lens (Skalse et al. 2022; Hadfield-Menell et
     al. 2017): e.g. a reward that maximizes a logged component while OOS fitness
     collapses, or a tautological "reward = port_ret" proxy.

Reads results ONLY through ``src.io.results.load_all`` (audit C-1) and treats the
archive as READ-ONLY (it never writes a run record). It REUSES the interpretability
lens (``_TAIL_TERMS`` / ``interpretability``) and the ``load_arms`` archive-walk from
``scripts/analyze_results.py`` rather than duplicating them.

DIRECTIONAL: like ``analyze_results`` this is qualitative evidence on the mechanism,
not a result — on a 1-seed development archive it is a go/no-go narrative, not a number
that enters the dissertation.

Flags
-----
  --results-dir   Archived runs directory; per-arm subdirs (default outputs/runs).
  --out-dir       Where to write the qualitative report (default outputs/tables).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# REUSE the analyze_results archive-walk + interpretability lens (audit C-1; no
# duplication). scripts/ is added to sys.path by the campaign entry points and the
# tests; fall back to a path insert so the module is importable standalone.
try:  # pragma: no cover - import plumbing
    from analyze_results import _TAIL_TERMS, interpretability, load_arms
except ImportError:  # pragma: no cover - standalone invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_results import _TAIL_TERMS, interpretability, load_arms

# The theory->code construct vocabulary MOVED to src/inference/reward_taxonomy.py (the taxonomy labels
# need it and a src module must not import from scripts/); re-imported here under the original private
# names so every existing call site (this module, ``ir._construct_prevalence`` in analyze_campaign.py,
# the tests) keeps reading the ONE shared table. ``analyze_results`` already imports ``src.*``, so this
# adds no new import constraint.
from src.inference.reward_taxonomy import (  # noqa: E402  (after the sys.path plumbing above)
    CONSTRUCTS as _CONSTRUCTS,
    TAIL_CONSTRUCTS as _TAIL_CONSTRUCTS,
    construct_prevalence as _construct_prevalence,
)

__all__ = [
    "per_generation_summary",
    "feedback_responsiveness",
    "hacking_taxonomy",
    "reward_magnitude_audit",
    "inspect",
    "write_report",
]

#: A reward maximizing a single logged proxy while held-out fitness is non-positive is the
#: canonical specification-gaming signature (Skalse 2022): the proxy and the true objective
#: have decoupled. This threshold flags a *collapsed* out-of-sample fitness.
_FITNESS_COLLAPSE = 0.0

#: VARIANCE-FLOOR / UNBOUNDED-MAGNITUDE specification gaming — the REAL prototype failure mode (the
#: critic-explosion source; outputs/prototype/anomalies.jsonl logged 64 critic_loss spikes up to
#: 1.1e+07). A reward whose dominant term divides a return by a *floored* volatility/variance —
#: ``port_ret / (std + 1e-8)``, ``ema_ret / ema_std``, ``r / (vol + eps)`` — is UNBOUNDED ABOVE: as the
#: agent drives realized variance toward zero the per-step reward diverges (Sharpe ~1e4), the SAC critic
#: target blows up, and training destabilizes — a Goodhart proxy (Skalse 2022; Pan et al. 2022, "The
#: Effects of Reward Misspecification"). This is ORTHOGONAL to fitness-collapse: on the prototype the
#: worst offenders (distributional-g1-c2 ``port_ret/(std+1e-6)``, g7-c3/c4 ``port_ret/std``) carried
#: POSITIVE val_fitness, so the existing collapse-gated specification_gaming check MISSED all of them.
#: ``_VARIANCE_FLOOR_PATTERNS`` matches ``<numerator> / (<vol-ish> + <eps>)`` and the un-floored
#: ``<ret-ish> / <std-ish>`` divide; the numerator/denominator token sets are deliberately
#: return/volatility-named so a benign normalization (e.g. ``hhi / (max_hhi + eps)``) does NOT trip it.
_VOL_DENOM = r"(?:std|stdev|sigma|vol|volatility|var|variance|ema_std|ema_var|daily_vol|std_ret|downside)\w*"
_RET_NUMER = r"(?:port_ret|ema_ret|mean_?ret\w*|roll_mean|mean|mu|excess\w*|ret|reward)"
_VARIANCE_FLOOR_PATTERNS = (
    # ret-ish / (vol-ish + tiny eps)  — the FLOORED divide (the canonical 1/(var+eps) gaming shape).
    rf"{_RET_NUMER}\s*/\s*\(\s*{_VOL_DENOM}\s*\+\s*\d*\.?\d*e?-?\d+\s*\)",
    # ret-ish / vol-ish            — the UN-floored divide (diverges hardest; e.g. ``port_ret / std``).
    rf"\bport_ret\s*/\s*{_VOL_DENOM}\b",
    rf"\bema_ret\s*/\s*{_VOL_DENOM}\b",
)

#: A finite, generous sanity bound on a per-step reward MAGNITUDE. The contract emits per-step rewards
#: on the scale of a daily portfolio return (|r| well under 1); a constant or literal far above this in
#: the reward body (a hand-coded huge bonus/penalty) is an unbounded-magnitude smell. Used only as a
#: SOURCE-level heuristic companion to the divide patterns (we cannot re-execute the archived reward
#: here), kept loose so ordinary coefficients (0.1, 0.5, 252 annualization) never trip it.
_SANE_MAGNITUDE_BOUND = 1.0e4

#: Tautology proxies: a reward that just re-emits the environment's own objective term
#: (no risk shaping) is a degenerate proxy — it cannot encode the tail the arm was fed. Each
#: pattern requires the proxy term to be the WHOLE returned reward (followed by a comma, the
#: reward-contract tuple separator, or end-of-line) — NOT part of a larger expression like
#: ``return port_ret + cvar`` (which is a shaped reward, not a tautology). ``_DELIM`` is that
#: terminator. ``[^\S\n]`` is "horizontal whitespace" (space/tab) so a trailing newline counts
#: as a delimiter but does not let the match run across lines.
#:
#: FALSE-POSITIVE FIX (verified on the real prototype archive 2026-06-25): the bare ``reward =
#: port_ret`` arm matched any line ENDING in ``…reward = port_ret``, so an INTERMEDIATE assignment
#: like ``core_reward = port_ret`` / ``sharpe_reward = port_ret`` (a named component the code later
#: combines into a SHAPED total — e.g. distributional-g4-c0, scalar_cvar5-g3-c2) was mis-flagged a
#: tautology even though the function ``return``s a multi-term reward. ``_LHS_START`` anchors the
#: assignment to a STATEMENT boundary (start-of-line indentation, then the LHS name with no
#: identifier char before it), so only ``reward``/``total``/``r`` assigned port_ret WHOLE — not a
#: ``*_reward`` suffix — trips it. The ``return`` arms were already statement-anchored.
_DELIM = r"[^\S\n]*(?:,|$)"
#: Start-of-statement anchor for an assignment LHS: line start + indentation, then a word boundary so
#: the captured name is the WHOLE identifier (``reward``), never the tail of ``core_reward``.
_LHS_START = r"(?m)^[^\S\n]*\b"
_TAUTOLOGY_PATTERNS = (
    rf"(?m)^[^\S\n]*return[^\S\n]+port_ret{_DELIM}",
    rf"(?m)^[^\S\n]*return[^\S\n]+float\([^\S\n]*port_ret[^\S\n]*\){_DELIM}",
    rf"{_LHS_START}(?:reward|total)[^\S\n]*=[^\S\n]*port_ret{_DELIM}",
    rf"(?m)^[^\S\n]*return[^\S\n]+np\.sum\([^\S\n]*weights[^\S\n]*\*[^\S\n]*returns[^\S\n]*\){_DELIM}",
    rf"(?m)^[^\S\n]*return[^\S\n]+\(?[^\S\n]*weights[^\S\n]*\*[^\S\n]*returns[^\S\n]*\)?\.sum\(\){_DELIM}",
)


# --------------------------------------------------------------------------- #
# record-field accessors (READ-ONLY; tolerant of partial records)             #
# --------------------------------------------------------------------------- #
def _generation(record: dict[str, Any]) -> int:
    """The 0-based generation that produced a candidate (records always carry it)."""
    return int(record.get("generation", 0))


def _fitness(record: dict[str, Any]) -> float:
    """Held-out (validation) fitness; ``nan`` when absent so means stay finite-skipping."""
    metrics = record.get("metrics") or {}
    val = metrics.get("val_fitness")
    return float(val) if val is not None else float("nan")


def _reward_source(record: dict[str, Any]) -> str:
    """The archived reward source string (``''`` when a record carries none)."""
    return str(record.get("reward_source") or "")


def _variance_floor_terms(source: str) -> list[str]:
    """Return the matched ``return/var``-floor divide expressions in a reward source (``[]`` if none).

    Detects the variance-floor / unbounded-magnitude gaming shape (``port_ret / (std + 1e-8)`` and the
    un-floored ``ret / std``) via :data:`_VARIANCE_FLOOR_PATTERNS`. The matched substrings name WHICH
    expression is unbounded, so the report can quote the exact offending term (e.g. the prototype's
    ``port_ret / (std + 1e-6)``). Comment lines are stripped first so a divide inside a ``# …`` note
    never trips the detector.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
    hits: list[str] = []
    for pat in _VARIANCE_FLOOR_PATTERNS:
        for m in re.finditer(pat, code):
            frag = m.group(0).strip()
            if frag not in hits:
                hits.append(frag)
    return hits


def _source_complexity(source: str) -> dict[str, int]:
    """Cheap size/complexity proxies for the authored reward code.

    ``chars`` and ``loc`` (non-blank lines) size the function; ``ops`` counts numpy
    calls + arithmetic/comparison operators as a structural-complexity proxy. None of
    these need an AST parse (the source may be a raw LLM string that never validated).
    """
    lines = [ln for ln in source.splitlines() if ln.strip()]
    ops = len(re.findall(r"np\.[a-zA-Z_]+|[+\-*/<>]", source))
    return {"chars": len(source), "loc": len(lines), "ops": ops}


#: Tail-statistic field ids the distributional / scalar_cvar5 feedback blocks carry, mapped
#: to the human label rendered in the block text (src/feedback/schema.py). Used to recover
#: the numeric tail vector the LLM was actually shown, from either the structured
#: ``metrics['tail_stats']`` dict OR the rendered ``feedback_block`` text.
_TAIL_FIELDS: tuple[tuple[str, str], ...] = (
    ("cvar_05", "CVaR 5%"),
    ("cvar_10", "CVaR 10%"),
    ("cvar_25", "CVaR 25%"),
    ("cvar_01", "CVaR 1%"),
    ("left_tail_mass", "left-tail mass"),
    ("robust_skew", "left-tail skew"),
)


# --------------------------------------------------------------------------- #
# reward-program differential: theory -> code construct vocabulary (T3.1)      #
# --------------------------------------------------------------------------- #
# ``_CONSTRUCTS`` / ``_TAIL_CONSTRUCTS`` / ``_construct_prevalence`` now live in
# src/inference/reward_taxonomy.py (imported at the top of this module) — the taxonomy's kind labels and
# this forensics report must count constructs against the SAME vocabulary, and only one copy may exist.

#: A CVaR/quantile LEVEL the code references (the LLM hard-codes the tail probability it shapes): matches a
#: percentile like ``percentile(arr, 5)`` / ``quantile(arr, 0.05)`` and a named ``cvar_05`` / ``cvar5`` /
#: ``cvar_01``. The captured number is normalised to a (0, 1] tail probability so a per-arm distribution of
#: shaped tail levels can be reported (e.g. "distributional codes shape CVaR at {0.05, 0.10} more than
#: scalar"). This is the count + the level-set, NOT a re-derivation of the fed level (audit-safe).
_PCT_CALL = re.compile(r"\b(?:np\.)?(?:percentile|quantile)\s*\([^,]+,\s*([0-9]*\.?[0-9]+)\s*[,)]")
_CVAR_NAMED = re.compile(r"\bcvar[_]?([0-9]{1,3})\b")

#: A numeric COEFFICIENT in the reward body: a float/int literal that is NOT part of an identifier, an
#: ``eps`` floor (``1e-8``), an array index, or an annualisation constant (252). Captures the magnitudes the
#: LLM chose for its penalty/bonus weights so the per-arm coefficient-magnitude distribution can be summarised
#: (median / max |coef|) — a coarse "how hard does this arm push its shaping terms" proxy. We strip comments
#: first and skip the small set of structural constants below so ordinary weights (0.1, 0.3, 3.0, 8.0)
#: dominate the distribution rather than plumbing literals.
_NUM_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d+|\d+e[+-]?\d+|\d+)(?![\w.])", re.IGNORECASE)
#: Structural literals that are NOT shaping coefficients (eps floors, annualisation, common window sizes,
#: trivial 0/1, and the index constants that appear in slicing). Excluded from the coefficient distribution.
_NON_COEF_LITERALS: frozenset[float] = frozenset(
    {0.0, 1.0, 2.0, 252.0, 100.0, 12.0, 5.0, 10.0, 20.0, 50.0, 1000.0}
)


def _declared_components(source: str) -> list[str]:
    """The KEYS of the ``components = {...}`` dict the reward DECLARES (the shaped terms it logs), or ``[]``.

    The reward contract returns ``(total, components, state)`` where ``components`` is a ``{str: float}`` dict
    LOGGED every step (``src/reward/contract.py``). Those per-step VALUES are NOT archived in the run records
    (verified: 0/239 prototype records carry a ``components`` field — they hold only ``val_fitness`` /
    ``val_returns`` / ``tail_stats``), so "which shaped term actually MOVES at run time" is UNAVAILABLE from
    the archive. What IS recoverable is the set of component NAMES the program declares in its source — the
    shaped terms the designer chose to expose — parsed here from the first ``components = { "k": …, … }``
    literal. This is a STRUCTURAL (declared-term) signal, explicitly NOT a runtime-activity one; the
    distinction is stated in the report so no runtime claim is fabricated.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
    m = re.search(r"components\s*=\s*\{(.*?)\}", code, flags=re.DOTALL)
    if m is None:
        return []
    # Keys are the string literals on the LHS of each ``"key": value`` pair inside the brace block.
    return sorted({k for k in re.findall(r"""['"]([A-Za-z_][\w]*)['"]\s*:""", m.group(1))})


def _cvar_levels_referenced(source: str) -> list[float]:
    """Distinct CVaR/quantile tail LEVELS (as probabilities in ``(0, 1]``) the source hard-codes, sorted.

    Recovers the tail probability the program shapes from two shapes: a ``percentile/quantile(arr, p)`` call
    (``p`` read as a percent if ``> 1`` else a fraction) and a named ``cvar_05`` / ``cvar5`` / ``cvar_01``
    (the trailing digits read as a percent). Only plausible LEFT-tail levels (``0 < level < 0.5``, strictly
    below the median) are kept, so a ``percentile(arr, 95)`` upper-tail probe or a ``quantile(.., 0.5)``
    median does not masquerade as a tail level. Report-only: this COUNTS the shaped levels, it does not
    re-derive the fed signal.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
    levels: set[float] = set()
    for raw in _PCT_CALL.findall(code):
        try:
            v = float(raw)
        except ValueError:
            continue
        prob = v / 100.0 if v > 1.0 else v
        if 0.0 < prob < 0.5:
            levels.add(round(prob, 4))
    for digits in _CVAR_NAMED.findall(code):
        prob = int(digits) / 100.0
        if 0.0 < prob < 0.5:
            levels.add(round(prob, 4))
    return sorted(levels)


def _coefficient_magnitudes(source: str) -> list[float]:
    """The shaping-coefficient magnitudes ``|literal|`` in the reward body (plumbing constants removed).

    Strips comments, pulls every numeric literal (:data:`_NUM_LITERAL`), drops the structural constants
    (:data:`_NON_COEF_LITERALS` — eps floors are < the smallest kept value and excluded by being tiny; 252
    annualisation; trivial 0/1/2; common window sizes), and returns the remaining magnitudes. These are the
    weights the LLM chose for its penalty/bonus terms; their per-arm distribution (median / max) is a coarse
    "how hard does this arm push its shaping" proxy, reported descriptively.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
    out: list[float] = []
    for raw in _NUM_LITERAL.findall(code):
        try:
            v = float(raw)
        except ValueError:
            continue
        a = abs(v)
        # Drop eps-floors / tiny plumbing constants (< 1e-3) and the structural literal set.
        if a < 1e-3 or v in _NON_COEF_LITERALS:
            continue
        out.append(a)
    return out


def _tail_vector(record: dict[str, Any]) -> np.ndarray | None:
    """The measured tail-stat vector for this candidate over :data:`_TAIL_FIELDS`, or ``None``.

    Returns the structured ``metrics['tail_stats']`` (what the loop measured off-critic for
    EVERY arm), falling back to parsing a rendered ``feedback_block``. NOTE: because the
    measurement is archived for *every* arm, this is NOT a "was the designer fed the tail"
    signal — that gate is :func:`_was_fed_tail`, applied per-arm in
    :func:`feedback_responsiveness`, which must judge responsiveness from what the LLM SAW
    (the rendered prompt), never from what was measured (else scalar/placebo get a spurious
    correlation against a distribution they were never shown).
    """
    metrics = record.get("metrics") or {}
    stats = metrics.get("tail_stats")
    if isinstance(stats, dict) and any(fid in stats for fid, _ in _TAIL_FIELDS):
        return np.asarray([float(stats.get(fid, np.nan)) for fid, _ in _TAIL_FIELDS], dtype=float)

    block = str(record.get("feedback_block") or "")
    if not block:
        return None
    vec: list[float] = []
    found = False
    for _fid, label in _TAIL_FIELDS:
        m = re.search(
            rf"{re.escape(label)}\s*:\s*([+-]?\d*\.?\d+)", block
        )
        if m is not None:
            vec.append(float(m.group(1)))
            found = True
        else:
            vec.append(float("nan"))
    return np.asarray(vec, dtype=float) if found else None


def _fed_text(record: dict[str, Any]) -> str:
    """The feedback text this candidate's designer actually SAW — the archived PROMPT.

    2026-07-05 (M13/M14 construct fix): the record's own ``feedback_block`` is the block built
    FROM this candidate's results — on the serial path it is fed to the NEXT generation, never to
    this candidate's own designer — so reading it as "what was fed" reversed the estimand and let
    generation-0 candidates (fed nothing) pass the fed-tail gate via their own block. The rendered
    prompt is archived for every candidate (loop.py "Rank 14") and carries the fed block verbatim
    for every generation >= 1; the generation-0 prompt is the tail-neutral base prompt. Records
    from a pre-Rank-14 archive (no ``prompt`` key) return ``""`` — callers must then treat fed
    VALUES as unrecoverable rather than substitute the candidate's own measured tail.
    """
    return str(record.get("prompt") or "")


def _fed_tail_vector(record: dict[str, Any]) -> np.ndarray | None:
    """The tail vector the DESIGNER WAS FED (parsed from the archived prompt), or ``None``.

    This — not :func:`_tail_vector` (the candidate's OWN post-training measurement) — is the
    ``X`` of the pre-registered SQ1/SQ2 mechanism estimands (PREREGISTRATION §2a: "does the fed
    tail signal change the authored reward code?"). Returns ``None`` for generation-0 candidates
    (fed nothing), non-tail arms, and legacy records without an archived prompt.
    """
    fed = _fed_text(record)
    if not fed:
        return None
    vec: list[float] = []
    found = False
    for _fid, label in _TAIL_FIELDS:
        m = re.search(rf"{re.escape(label)}\s*:\s*([+-]?\d*\.?\d+)", fed)
        if m is not None:
            vec.append(float(m.group(1)))
            found = True
        else:
            vec.append(float("nan"))
    return np.asarray(vec, dtype=float) if found else None


def _was_fed_tail(record: dict[str, Any]) -> bool:
    """Did the rendered feedback this candidate's designer SAW carry tail diagnostics?

    Decided from the archived PROMPT (see :func:`_fed_text`) — what the designer actually saw —
    so a generation-0 candidate (base prompt, no feedback) is correctly NOT fed, and the serial
    archive's own-``feedback_block`` (fed to the NEXT generation) can no longer leak a candidate
    into the fed set (2026-07-05 M13 fix; the old ``feedback_block``-first read did both). A
    ``scalar`` arm sees only a Deflated-Sharpe scalar and ``placebo`` sees inert reference
    constants, so neither matches a :data:`_TAIL_FIELDS` label; the search arms carry no LLM
    prompt at all. Legacy records without an archived prompt fall back to the own-block label
    check GATED on generation >= 1 (the block schema is fixed per arm, so at gen >= 1 it is a
    faithful was-fed indicator even though its VALUES are the candidate's own).
    """
    fed_text = _fed_text(record)
    if fed_text:
        return any(re.search(rf"{re.escape(label)}\s*:", fed_text) for _fid, label in _TAIL_FIELDS)
    if _generation(record) < 1:
        return False
    own_block = str(record.get("feedback_block") or "")
    return any(re.search(rf"{re.escape(label)}\s*:", own_block) for _fid, label in _TAIL_FIELDS)


def _by_generation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records for one arm sorted by (generation, candidate_id) — deterministic order."""
    return sorted(
        records,
        key=lambda r: (_generation(r), str(r.get("candidate_id", r.get("run_id", "")))),
    )


# --------------------------------------------------------------------------- #
# 1. per-generation summary                                                   #
# --------------------------------------------------------------------------- #
def per_generation_summary(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Per-arm-per-generation fitness + reward-code complexity / tail-usage trend.

    Parameters
    ----------
    records : list[dict] or dict[str, list[dict]]
        Either a flat list of per-candidate records (each carrying ``arm`` and
        ``generation``) or the ``{arm: [records]}`` mapping from :func:`load_arms`.

    Returns
    -------
    dict
        Keyed by ``(arm, generation)``. Each value carries ``n``, ``best_fitness``,
        ``mean_fitness`` (nan-skipping), ``mean_loc`` / ``mean_ops`` /
        ``mean_chars`` (the reward-source size/complexity trend) and
        ``frac_uses_tail`` + ``mean_tail_terms`` (the §6.1 interpretability lens applied
        across generations — does the authored code reference more tail structure over
        time?).
    """
    flat = _flatten(records)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in flat:
        groups.setdefault((str(r.get("arm", "?")), _generation(r)), []).append(r)

    summary: dict[tuple[str, int], dict[str, Any]] = {}
    for key in sorted(groups):
        recs = groups[key]
        fits = np.asarray([_fitness(r) for r in recs], dtype=float)
        finite = fits[np.isfinite(fits)]
        complexities = [_source_complexity(_reward_source(r)) for r in recs]
        interps = [interpretability(_reward_source(r)) for r in recs]
        n_terms = [len(i["terms"]) for i in interps]
        summary[key] = {
            "arm": key[0],
            "generation": key[1],
            "n": len(recs),
            "best_fitness": float(np.max(finite)) if finite.size else float("nan"),
            "mean_fitness": float(np.mean(finite)) if finite.size else float("nan"),
            "mean_chars": float(np.mean([c["chars"] for c in complexities])),
            "mean_loc": float(np.mean([c["loc"] for c in complexities])),
            "mean_ops": float(np.mean([c["ops"] for c in complexities])),
            "frac_uses_tail": float(np.mean([1.0 if i["uses_tail"] else 0.0 for i in interps])),
            "mean_tail_terms": float(np.mean(n_terms)) if n_terms else 0.0,
        }
    return summary


# --------------------------------------------------------------------------- #
# 2. feedback responsiveness (DIRECTIONAL probe; ablation lattice is the causal H2 test)      #
# --------------------------------------------------------------------------- #
def feedback_responsiveness(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Does the LLM's revision track the distribution it was fed back? (DIRECTIONAL probe.)

    For one arm we walk its candidates in generation order. The gen-``N``
    ``feedback_block`` carried the tail statistics the LLM saw BEFORE writing the
    gen-``N+1`` reward. For each successive pair we compute:

      - ``edit``     : the magnitude of the reward-source edit gen ``N`` → ``N+1``
        (a normalized character-level Levenshtein-style distance ratio);
      - ``feedback`` : the L1 norm of the tail-stat DELTA the LLM was shown across that
        same step (how much the fed-back distribution moved).

    A *responsive* designer edits MORE when the distribution moved MORE, so the per-arm score
    is the SPEARMAN (rank) correlation between ``edit`` and ``feedback`` across steps (rank-based
    — robust to the heterogeneous tail-stat scales + non-linearity; ``None`` when undefined: <2
    usable steps or a constant series). DIRECTIONAL only — the CAUSAL "did it use the information"
    test is the matched-budget ablation across arms (cf. Eureka 2024 §4.3), NOT this correlation;
    an arm fed only a scalar (``scalar`` / ``placebo``) has no tail delta to track → ``None``.

    Parameters
    ----------
    records : list[dict] or dict[str, list[dict]]
        Flat per-candidate records or the ``{arm: [records]}`` mapping.

    Returns
    -------
    dict
        ``{arm: {"score": float | None, "n_steps": int, "edits": [...],
        "feedback_deltas": [...], "reason": str}}``. ``score`` is a SPEARMAN (rank)
        correlation in ``[-1, 1]``, or ``None`` (undefined, or an arm fed no tail).
    """
    arms = _group_by_arm(records)
    out: dict[str, dict[str, Any]] = {}
    for arm, recs in arms.items():
        ordered = _by_generation(recs)
        # H2 gate: only an arm actually FED a tail distribution can be responsive to it. The
        # off-critic measurement archives metrics['tail_stats'] for EVERY arm, so we decide from
        # what the designer SAW (the rendered prompt / feedback_block), NOT what was measured —
        # else scalar (a Sharpe scalar) / placebo (inert constants) / the search arms get a
        # spurious correlation against a tail they were never shown (see _was_fed_tail).
        if not any(_was_fed_tail(r) for r in ordered):
            out[arm] = {
                "score": None,
                "n_steps": 0,
                "edits": [],
                "feedback_deltas": [],
                "reason": "arm not fed a tail distribution (scalar/placebo/search) — nothing to track",
            }
            continue

        edits: list[float] = []
        feedback_deltas: list[float] = []
        prev_src: str | None = None
        prev_tail: np.ndarray | None = None
        for r in ordered:
            src = _reward_source(r)
            tail = _tail_vector(r)
            if prev_src is not None and prev_tail is not None and tail is not None:
                edits.append(_edit_distance_ratio(prev_src, src))
                delta = np.abs(np.asarray(tail) - np.asarray(prev_tail))
                feedback_deltas.append(float(np.nansum(delta)))
            prev_src = src
            prev_tail = tail if tail is not None else prev_tail

        score, reason = _finite_correlation(edits, feedback_deltas)
        out[arm] = {
            "score": score,
            "n_steps": len(edits),
            "edits": [float(e) for e in edits],
            "feedback_deltas": [float(d) for d in feedback_deltas],
            "reason": reason,
        }
    return out


# --------------------------------------------------------------------------- #
# 3. reward-hacking taxonomy                                                   #
# --------------------------------------------------------------------------- #
def hacking_taxonomy(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Tag candidates against a reward-hacking taxonomy (Skalse 2022; Hadfield-Menell 2017).

    Each candidate is screened through the :data:`_TAIL_TERMS` interpretability lens and
    its held-out fitness for FOUR failure modes:

      - ``unbounded_magnitude`` — the reward's dominant term divides a return by a FLOORED
        volatility (``port_ret / (std + 1e-8)``, ``ema_ret / ema_std``): UNBOUNDED ABOVE as
        realized variance -> 0, so the agent games it by collapsing variance, the SAC critic
        target diverges, and training destabilizes. This is the REAL prototype failure mode
        (the 64 logged critic explosions, critic_loss up to 1.1e+07) and is ORTHOGONAL to
        fitness-collapse — on the prototype the worst offenders carried POSITIVE val_fitness,
        so it is flagged on the SOURCE shape regardless of fitness (matched by
        :data:`_VARIANCE_FLOOR_PATTERNS`; Skalse 2022; Pan et al. 2022).
      - ``specification_gaming`` — the reward references a logged component (tail term)
        yet held-out fitness COLLAPSED (``val_fitness <= 0``): the proxy was driven up
        while the true objective fell, the canonical reward-gaming signature.
      - ``proxy_no_tail`` — fitness is non-positive AND the code references NO tail/risk
        structure: it optimizes a non-risk proxy that ignored the fed-back distribution.
      - ``tautology`` — the reward just re-emits the environment objective
        (``return port_ret`` / ``sum(weights*returns)``): a degenerate proxy that cannot
        encode the tail at all (matched by :data:`_TAUTOLOGY_PATTERNS`).

    Parameters
    ----------
    records : list[dict] or dict[str, list[dict]]
        Flat per-candidate records or the ``{arm: [records]}`` mapping.

    Returns
    -------
    dict
        ``{"candidates": [per-candidate tag dicts], "counts": {tag: int},
        "n_flagged": int, "n_total": int}``. A candidate's ``flags`` list is empty when it
        trips no signal; ``counts`` tallies each tag across the archive.
    """
    flat = _flatten(records)
    tagged: list[dict[str, Any]] = []
    counts = {"unbounded_magnitude": 0, "specification_gaming": 0, "proxy_no_tail": 0, "tautology": 0}
    for r in flat:
        src = _reward_source(r)
        fit = _fitness(r)
        interp = interpretability(src)
        uses_tail = bool(interp["uses_tail"])
        # The specific logged tail/risk components this reward references (the proxy it could
        # be over-optimizing) — drawn straight from the shared _TAIL_TERMS lens so the signal
        # names WHICH logged component decoupled from out-of-sample fitness.
        logged_components = sorted({t for t in _TAIL_TERMS if t in src.lower()})
        collapsed = np.isfinite(fit) and fit <= _FITNESS_COLLAPSE
        is_tautology = any(re.search(p, src, flags=re.MULTILINE) for p in _TAUTOLOGY_PATTERNS)
        # The variance-floor / unbounded-magnitude divide(s) this reward contains — the critic-explosion
        # source. Flagged on the SOURCE shape (NOT gated on fitness): it is unbounded above as variance
        # -> 0 regardless of the validation number, the very confound that hid it from the collapse gate.
        variance_floor_terms = _variance_floor_terms(src)

        flags: list[str] = []
        # unbounded_magnitude: a floored/un-floored return-over-volatility divide -> Sharpe diverges as
        # variance collapses (the real prototype failure mode; orthogonal to the collapse gate below).
        if variance_floor_terms:
            flags.append("unbounded_magnitude")
        # specification_gaming: maximizes a logged tail component yet OOS fitness collapsed.
        if collapsed and logged_components:
            flags.append("specification_gaming")
        if collapsed and not uses_tail:
            flags.append("proxy_no_tail")
        if is_tautology:
            flags.append("tautology")
        for f in flags:
            counts[f] += 1

        tagged.append(
            {
                "arm": str(r.get("arm", "?")),
                "candidate_id": str(r.get("candidate_id", r.get("run_id", "?"))),
                "generation": _generation(r),
                "val_fitness": fit,
                "uses_tail": uses_tail,
                "tail_terms": interp["terms"],
                "variance_floor_terms": variance_floor_terms,
                "flags": flags,
            }
        )
    return {
        "candidates": tagged,
        "counts": counts,
        "n_flagged": sum(1 for t in tagged if t["flags"]),
        "n_total": len(tagged),
    }


# --------------------------------------------------------------------------- #
# 4. reward-program differential (T3.1 — theory->code->outcome mechanism loop)  #
# --------------------------------------------------------------------------- #
def reward_program_differential(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """PER-ARM reward-program characterization + the cross-arm tail-construct DIFFERENTIAL (T3.1).

    Opens the 239 archived reward PROGRAMS (their ``reward_source`` + declared ``components`` + the recorded
    tail signal) and builds, per arm, the theory->code half of the mechanism loop:

      * **construct prevalence** — the fraction of the arm's programs that reference each risk-shaping
        construct (:data:`_CONSTRUCTS`: CVaR / quantile-tail / drawdown / Sortino-downside / left-tail-mass /
        rolling-vol / turnover / online-Sharpe / Herfindahl), split into a TAIL set (what H2 predicts the
        distributional feedback should grow) and the generic risk controls;
      * **CVaR-level count** — how many programs hard-code a CVaR/quantile tail LEVEL, and the union of the
        tail levels referenced (:func:`_cvar_levels_referenced`);
      * **coefficient-magnitude distribution** — the median / max |shaping coefficient| the arm chose
        (:func:`_coefficient_magnitudes`), a coarse "how hard does it push its shaping" proxy;
      * **declared-component activity** — the most-frequent ``components`` dict KEYS the arm's programs declare
        (:func:`_declared_components`). NB this is the DECLARED (logged-term) set, NOT runtime activity: the
        per-step component VALUES are not archived (0/239 records carry a ``components`` field), so "which
        shaped term actually MOVES" is unavailable from the archive and is reported as such, never fabricated.

    The premise it TESTS (the directional H2 mechanism question): did the distributional arm's CODE reference
    tail statistics MORE than the scalar / placebo / search arms? It reports, per arm, ``tail_construct_rate``
    = the mean number of TAIL constructs per program, and a ``differential`` block = ``distributional`` minus
    each comparator on that rate (+ a per-construct prevalence delta). DIRECTIONAL only (1-seed development
    archive; no number enters the dissertation). On the prototype the base prompt pre-seeded the tail
    vocabulary (PREREGISTRATION R38), so EVERY arm wrote real CVaR/drawdown code and the differential is
    expected to be SMALL — which is exactly the confound R38 removes for the campaign; the report states this
    so a near-zero differential reads as the known prompt-leak signature, not a null mechanism.

    Parameters
    ----------
    records : list[dict] or dict[str, list[dict]]
        Flat per-candidate records or the ``{arm: [records]}`` mapping.

    Returns
    -------
    dict
        ``{"per_arm": {arm: {...}}, "differential": {comparator: {...}}, "tail_constructs": [...],
        "components_archived": bool, "n_total": int}``. ``components_archived`` is ``False`` whenever NO
        record carries a per-step ``components`` field (the archive case), flagging that the component signal
        is the DECLARED set only.
    """
    flat = _flatten(records)
    by_arm = _group_by_arm(flat)
    # Detect whether ANY record actually persisted per-step components (vs only the declared source dict).
    components_archived = any(
        ("components" in r) or ("components" in (r.get("metrics") or {})) for r in flat
    )

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(by_arm):
        recs = by_arm[arm]
        sources = [_reward_source(r) for r in recs]
        n = len(recs)
        prevalences = [_construct_prevalence(s) for s in sources]
        # Per-construct prevalence = fraction of the arm's programs that reference it.
        prevalence = {
            name: (float(np.mean([1.0 if p[name] else 0.0 for p in prevalences])) if n else 0.0)
            for name, _pat, _is_tail in _CONSTRUCTS
        }
        # Mean number of TAIL constructs per program (the headline cross-arm differential statistic).
        tail_counts = [sum(1 for name in _TAIL_CONSTRUCTS if p[name]) for p in prevalences]
        all_counts = [sum(1 for name in p if p[name]) for p in prevalences]
        # CVaR-level usage across the arm's programs.
        level_lists = [_cvar_levels_referenced(s) for s in sources]
        n_with_level = sum(1 for lv in level_lists if lv)
        level_union = sorted({x for lv in level_lists for x in lv})
        # Coefficient magnitudes pooled over the arm's programs.
        coefs = [c for s in sources for c in _coefficient_magnitudes(s)]
        # Declared-component key frequency (the logged shaped terms — DECLARED, not runtime).
        comp_counter: dict[str, int] = {}
        for s in sources:
            for k in _declared_components(s):
                comp_counter[k] = comp_counter.get(k, 0) + 1
        top_components = sorted(comp_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        per_arm[arm] = {
            "n_programs": n,
            "construct_prevalence": prevalence,
            "tail_construct_rate": float(np.mean(tail_counts)) if tail_counts else 0.0,
            "any_construct_rate": float(np.mean(all_counts)) if all_counts else 0.0,
            "n_programs_with_cvar_level": int(n_with_level),
            "cvar_levels_referenced": [float(x) for x in level_union],
            "coef_median": float(np.median(coefs)) if coefs else None,
            "coef_max": float(np.max(coefs)) if coefs else None,
            "coef_n": len(coefs),
            "top_declared_components": [[k, int(v)] for k, v in top_components],
        }

    # Cross-arm DIFFERENTIAL: distributional minus each LLM comparator on the tail-construct rate + per-
    # construct prevalence. The H2 mechanism prediction is that these deltas are POSITIVE (distributional
    # codes more tail structure); on the prompt-leaked prototype they are expected to be small (R38).
    #
    # V11 (2026-06-26): the differential pools ONLY the LLM arms ({scalar, placebo, scalar_cvar5}). The two
    # SEARCH baselines (random_search, bayes_opt) are EXCLUDED from this cross-arm "LLM tail-construct
    # prevalence" differential. Each is a HUMAN-AUTHORED search template whose tail-construct SKELETON is
    # FIXED: every candidate references the SAME tail constructs (verified on the prototype — random_search =
    # cvar+quantile_tail on 100% of programs, bayes_opt = cvar+drawdown on 100%; tail_rate ≡ 2.000 with ZERO
    # program-to-program variance), only the sampled COEFFICIENTS differ (so the source bytes vary but the
    # construct set does not). The tail-construct rate is therefore a property of the template DESIGN, not an
    # LLM authoring CHOICE — differencing it against the LLM-authored distribution is a category error in the
    # one mechanism exhibit (it would read a hand-fixed tail_rate as if the search "decided" to code the
    # tail). They are reported SEPARATELY in ``search_templates`` with that flag, never inside ``differential``.
    _LLM_COMPARATORS = ("scalar", "placebo", "scalar_cvar5")
    _SEARCH_ARMS = ("random_search", "bayes_opt")
    differential: dict[str, Any] = {}
    if "distributional" in per_arm:
        d = per_arm["distributional"]
        for comp in _LLM_COMPARATORS:
            if comp not in per_arm:
                continue
            c = per_arm[comp]
            differential[comp] = {
                "tail_rate_delta": float(d["tail_construct_rate"] - c["tail_construct_rate"]),
                "per_construct_delta": {
                    name: float(d["construct_prevalence"][name] - c["construct_prevalence"][name])
                    for name in _TAIL_CONSTRUCTS
                },
                "distributional_references_tail_more": bool(
                    d["tail_construct_rate"] > c["tail_construct_rate"]
                ),
            }

    # The search baselines' tail-construct rate, reported as a fixed-template DESCRIPTOR (NOT a differential):
    # the tail-construct SKELETON is hand-fixed (constant tail count across programs), so the rate is not
    # interpretable as an LLM tail choice. ``fixed_template`` keys off ZERO variance in the per-program tail
    # count (the construct set is constant) — NOT source-identity, since the sampled coefficients vary.
    search_templates: dict[str, Any] = {}
    for comp in _SEARCH_ARMS:
        if comp not in per_arm:
            continue
        c = per_arm[comp]
        n = int(c.get("n_programs", 0))
        recs_for_arm = by_arm.get(comp, [])
        sources_for_arm = {_reward_source(r) for r in recs_for_arm}
        tail_counts_arm = [
            sum(1 for name in _TAIL_CONSTRUCTS if _construct_prevalence(_reward_source(r))[name])
            for r in recs_for_arm
        ]
        tail_count_var = float(np.var(tail_counts_arm)) if tail_counts_arm else 0.0
        search_templates[comp] = {
            "tail_construct_rate": float(c["tail_construct_rate"]),
            "n_programs": n,
            "n_distinct_sources": len(sources_for_arm),
            "tail_count_variance": tail_count_var,
            # FIXED skeleton iff the per-program tail-construct COUNT never varies (rate is hand-determined).
            "fixed_template": bool(tail_count_var <= 1e-12),
            "note": (
                "human-authored search template with a FIXED tail-construct skeleton (every program "
                "references the same tail constructs — zero variance in the tail count; only sampled "
                "coefficients differ). The tail-construct rate is a property of the template DESIGN, NOT an "
                "LLM authoring choice; EXCLUDED from the cross-arm LLM tail-construct differential (V11)"
            ),
        }

    return {
        "per_arm": per_arm,
        "differential": differential,
        "search_templates": search_templates,
        "llm_comparators": list(_LLM_COMPARATORS),
        "tail_constructs": list(_TAIL_CONSTRUCTS),
        "components_archived": bool(components_archived),
        "n_total": len(flat),
    }


#: A short representative probe for the MEASURED reward-magnitude audit. Daily-return-scaled inputs over
#: 30 risky assets + cash (the dev universe), driven for a few dozen steps so a STATEFUL reward whose scale
#: blows up only after history accumulates (the ``mu / (std + eps)`` cold-start that hits ``mu / 1e-8`` on
#: the FIRST sample — the verified prototype shape) is actually exercised. Deterministic (fixed seed).
_PROBE_STEPS = 48
_PROBE_ASSETS = 30


def _measure_reward_magnitude(source: str, *, steps: int = _PROBE_STEPS, n_assets: int = _PROBE_ASSETS) -> dict[str, Any]:
    """Execute an archived reward on a representative probe; return its empirical max ``|total|``.

    This turns the SOURCE-shape ``unbounded_magnitude`` flag into a MEASURED number: it reproduces the
    verification that prototype ``scalar-g5-c3`` emits ``|total| ≈ 1.15e4`` on an ordinary step (a per-step
    Sharpe with ``sigma`` floored at ``1e-8`` → ``mu/1e-8`` on the first sample). Read-only: the reward is
    run in the same in-process sandbox the env uses (``safe_call`` substitutes ``SAFE_DEFAULT`` and flags on
    raise / non-finite), the archive is never written, and the reward itself is NOT altered — a candidate
    whose reward explodes already trains a diverged critic and loses selection; this only SURFACES it.

    Returns ``{"ok", "max_abs_total", "finite", "error"}``. ``ok=False`` (with ``error``) when the source
    cannot even be constructed; a reward that raises mid-probe is reported with whatever magnitude it reached.
    """
    out: dict[str, Any] = {"ok": False, "max_abs_total": None, "finite": True, "error": None}
    if not source.strip():
        out["error"] = "empty source"
        return out
    try:
        from src.sandbox.executor import safe_call, validate_once

        # The anonymised validate-once fixture (no tickers/dates), sized to the probe universe.
        fixture = (
            np.full(n_assets + 1, 1.0 / (n_assets + 1)),
            np.full(n_assets, 0.001),
            np.full(n_assets + 1, 1.0 / (n_assets + 1)),
            0.0,
            {},
        )
        reward_fn = validate_once(source, fixture)
    except Exception as exc:  # noqa: BLE001 - a non-constructable reward is reported, not raised
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    rng = np.random.default_rng(0)
    state: Any = None
    max_abs = 0.0
    finite = True
    for _ in range(steps):
        w = rng.random(n_assets + 1)
        w = w / w.sum()
        wp = rng.random(n_assets + 1)
        wp = wp / wp.sum()
        r = rng.normal(0.0, 0.012, n_assets)  # ~typical daily equity vol
        port_ret = float(w[:n_assets] @ r)
        info = {"reward_state": state}
        total, _components, state = safe_call(reward_fn, w, r, wp, port_ret, info)
        if not np.isfinite(total):
            finite = False
            continue
        max_abs = max(max_abs, abs(float(total)))
    out.update(ok=True, max_abs_total=float(max_abs), finite=bool(finite))
    return out


def reward_magnitude_audit(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    *,
    bound: float = _SANE_MAGNITUDE_BOUND,
) -> dict[str, Any]:
    """REPORT-ONLY measured-magnitude diagnostic: empirical max ``|reward total|`` per candidate.

    Companion to :func:`hacking_taxonomy`'s source-shape ``unbounded_magnitude`` flag — it EXECUTES each
    archived reward on a representative probe and records the realized peak magnitude, flagging any whose
    ``max_abs_total`` exceeds ``bound`` (default ``1e4``, the sane per-step-reward ceiling). Such a reward
    drives the SAC Bellman target toward ``max|r|/(1-gamma)`` and is the verified critic-explosion source
    (``outputs/prototype/anomalies.jsonl``). It does NOT alter the reward (a diverged candidate already
    loses selection); it makes the pathology AUDITABLE with a number, and corroborates that the PopArt
    scale-normalization in the trainer is what tames it.
    """
    flat = _flatten(records)
    rows: list[dict[str, Any]] = []
    n_over = 0
    worst = 0.0
    for r in flat:
        src = _reward_source(r)
        meas = _measure_reward_magnitude(src)
        mx = meas.get("max_abs_total")
        over = bool(mx is not None and mx > bound)
        n_over += int(over)
        if mx is not None and np.isfinite(mx):
            worst = max(worst, mx)
        rows.append(
            {
                "arm": str(r.get("arm", "?")),
                "candidate_id": str(r.get("candidate_id", r.get("run_id", "?"))),
                "generation": _generation(r),
                "val_fitness": _fitness(r),
                "max_abs_total": mx,
                "finite": meas.get("finite", True),
                "over_bound": over,
                "ok": meas.get("ok", False),
                "error": meas.get("error"),
            }
        )
    rows.sort(key=lambda d: (-(d["max_abs_total"] or -1.0)))
    return {
        "bound": float(bound),
        "n_total": len(rows),
        "n_over_bound": n_over,
        "n_nonfinite": sum(1 for d in rows if not d["finite"]),
        "worst_max_abs_total": float(worst),
        "candidates": rows,
    }


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _flatten(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Accept either a flat record list or the ``{arm: [records]}`` mapping.

    When given the mapping, the ``arm`` key is stamped onto each record (defensively) so
    downstream grouping by ``record['arm']`` is consistent with the flat-list path.
    """
    if isinstance(records, dict):
        flat: list[dict[str, Any]] = []
        for arm, recs in records.items():
            for r in recs:
                r = {**r, "arm": r.get("arm", arm)}
                flat.append(r)
        return flat
    return list(records)


def _group_by_arm(
    records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records by arm (passthrough when already the ``{arm: [records]}`` mapping)."""
    if isinstance(records, dict):
        return {k: list(v) for k, v in records.items()}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get("arm", "?")), []).append(r)
    return groups


def _edit_distance_ratio(a: str, b: str) -> float:
    """Normalized edit distance in ``[0, 1]`` between two reward sources.

    Uses :func:`difflib.SequenceMatcher` (Python stdlib, no torch): ``1 - ratio`` is a
    cheap, deterministic character-level edit magnitude — ``0`` for identical source,
    approaching ``1`` for a full rewrite. The magnitude (not the diff) is what correlates
    against the fed-back tail delta.
    """
    from difflib import SequenceMatcher

    if not a and not b:
        return 0.0
    return float(1.0 - SequenceMatcher(None, a, b).ratio())


def _finite_correlation(xs: list[float], ys: list[float]) -> tuple[float | None, str]:
    """SPEARMAN (rank) correlation, or ``None`` when undefined, plus a reason.

    Rank-based (not Pearson): the fed tail-stat L1-delta sums six heterogeneous-scale statistics
    and the edit/feedback relation need not be linear, so a monotone rank measure is the defensible
    choice. Returns ``None`` (NOT 0.0 — which would masquerade as "no responsiveness") when there
    are <2 paired points or either series is constant (the correlation is undefined there); the
    string reason records WHY for the report.
    """
    from scipy.stats import spearmanr

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return None, f"too few usable steps ({x.size}) to correlate"
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None, "a series is constant — correlation undefined"
    r = float(spearmanr(x, y)[0])
    if not np.isfinite(r):
        return None, "correlation evaluated non-finite"
    return r, "ok"


# --------------------------------------------------------------------------- #
# orchestration + markdown                                                     #
# --------------------------------------------------------------------------- #
def inspect(root: str | Path) -> dict[str, Any]:
    """Run the full reward forensics over an archive root (per-arm subdirs).

    Reads ``{arm: [records]}`` via :func:`load_arms` (audit C-1; READ-ONLY) and returns
    the three analyses plus the arm list, ready for :func:`write_report`.
    """
    arms = load_arms(root)
    return {
        "arms": sorted(arms),
        "per_generation": per_generation_summary(arms),
        "responsiveness": feedback_responsiveness(arms),
        "hacking": hacking_taxonomy(arms),
        "reward_magnitude": reward_magnitude_audit(arms),
        "program_differential": reward_program_differential(arms),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:+.4f}"
    return str(v)


def _program_differential_markdown(diff: dict[str, Any]) -> list[str]:
    """Render the T3.1 reward-program differential section (the theory->code mechanism loop)."""
    per_arm = diff.get("per_arm", {})
    tail_constructs = diff.get("tail_constructs", [])
    lines: list[str] = [
        "",
        "## 4. Reward-program differential (theory→code mechanism loop; T3.1)",
        "Per-arm characterization of the archived reward PROGRAMS: which risk-shaping constructs the code "
        "references, how many programs hard-code a CVaR/quantile tail level, the shaping-coefficient "
        "magnitudes, and the DECLARED `components` keys. The headline cross-arm question (H2 mechanism): does "
        "the **distributional** arm's CODE reference tail statistics MORE than scalar / placebo / search? "
        f"TAIL constructs = {', '.join(tail_constructs)}.",
        "",
    ]
    if not diff.get("components_archived", False):
        lines += [
            "> **Scope (no fabrication):** per-step `components` VALUES are NOT in the archive (0 records carry "
            "a `components` field), so *which shaped term actually moves at run time* is unavailable. The "
            "`components` column below is the DECLARED (logged-term) set parsed from each program's source — a "
            "structural signal, not a runtime-activity one.",
            "",
        ]
    lines += [
        "| arm | n | tail-construct rate | n w/ CVaR level | CVaR levels | coef median | coef max | top declared components |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, a in per_arm.items():
        levels = ", ".join(f"{x:g}" for x in a["cvar_levels_referenced"]) or "—"
        comps = ", ".join(k for k, _v in a["top_declared_components"][:5]) or "—"
        lines.append(
            f"| {arm} | {a['n_programs']} | {a['tail_construct_rate']:.2f} | "
            f"{a['n_programs_with_cvar_level']} | {levels} | {_fmt(a['coef_median'])} | "
            f"{_fmt(a['coef_max'])} | {comps} |"
        )

    differential = diff.get("differential", {})
    if differential:
        lines += [
            "",
            "### Cross-arm tail differential (distributional − LLM comparator)",
            "Positive ⇒ the distributional code references MORE tail structure (the H2 prediction). Pooled over "
            "the **LLM arms only** ({scalar, placebo, scalar_cvar5}); the two search baselines are EXCLUDED "
            "(V11 — each is one fixed human template, not an LLM authoring choice; see below). On the prototype "
            "the base prompt pre-seeded the tail vocabulary (PREREGISTRATION R38), so every LLM arm wrote real "
            "CVaR/drawdown code and these deltas are EXPECTED to be small — a near-zero differential reads as the "
            "known prompt-leak signature (the confound R38 removes for the campaign), NOT a null mechanism. "
            "DIRECTIONAL only; no number enters the dissertation.",
            "",
            "| LLM comparator | tail-rate Δ | distributional references tail more? | per-construct Δ (tail) |",
            "|---|---|---|---|",
        ]
        for comp, d in differential.items():
            pcd = ", ".join(
                f"{name} {val:+.2f}" for name, val in d["per_construct_delta"].items() if abs(val) > 1e-9
            ) or "all ~0"
            lines.append(
                f"| {comp} | {d['tail_rate_delta']:+.3f} | {d['distributional_references_tail_more']} | {pcd} |"
            )

    search_templates = diff.get("search_templates", {})
    if search_templates:
        lines += [
            "",
            "### Search baselines — fixed-skeleton templates (NOT in the LLM tail differential; V11)",
            "`random_search` / `bayes_opt` are HUMAN-AUTHORED search templates with a FIXED tail-construct "
            "skeleton: every candidate references the same tail constructs (zero variance in the tail count — "
            "the rate is a constant property of the template DESIGN), and only the sampled coefficients differ. "
            "So the tail-construct rate is not interpretable as an LLM tail-construct CHOICE. Listed for "
            "completeness; deliberately kept OUT of the cross-arm differential above.",
            "",
            "| search arm | n programs | distinct sources | tail-count variance | fixed skeleton? | tail-construct rate |",
            "|---|---|---|---|---|---|",
        ]
        for comp, s in search_templates.items():
            lines.append(
                f"| {comp} | {s['n_programs']} | {s['n_distinct_sources']} | "
                f"{s.get('tail_count_variance', 0.0):.3f} | {s['fixed_template']} | "
                f"{s['tail_construct_rate']:.2f} |"
            )
    lines.append("")
    return lines


def _markdown(result: dict[str, Any]) -> str:
    """Render the qualitative reward-forensics report as markdown.

    Ordering puts RESPONSIVENESS first (the HEADLINE forensics signal — does reflection steer the
    generated code toward the fed tail distribution?), then the reward-hacking taxonomy (incl. the
    variance-floor / unbounded-magnitude class — the real critic-explosion failure mode), then the
    per-generation complexity/tail-usage trend as supporting context.
    """
    resp = result["responsiveness"]
    fed_arms = {a: e for a, e in resp.items() if e["score"] is not None}
    lines: list[str] = [
        "# Reward forensics — opening the black box (FINAL_PLAN Phase 4.C; §6.1 GREEN gate)",
        "",
        "Qualitative evidence that the LLM reward-designer USED the distributional feedback "
        "(H2), not just that a metric gap exists. DIRECTIONAL on a 1-seed development archive "
        "— a go/no-go narrative, not a number for the dissertation.",
        "",
        f"Arms inspected: {', '.join(result['arms']) or '(none)'}.",
        "",
        "## 1. HEADLINE — feedback responsiveness (does reflection steer the code toward the fed tail?)",
        "The headline forensics question: when the LLM is FED the realized-return tail distribution, does "
        "its next reward-code revision move MORE when the distribution moves MORE? Per-arm SPEARMAN (rank) "
        "correlation between the reward-source EDIT magnitude (gen N→N+1) and the L1 tail-stat DELTA the LLM "
        "was shown that step. Higher ⇒ more responsive. `n/a` = arm carries NO tail feedback "
        "(scalar/placebo/search) — there is nothing to track, by construction. DIRECTIONAL only: the CAUSAL "
        "'did it use the distribution' test is the matched-budget ablation contrast across arms (cf. Eureka "
        "§4.3); no number in this table enters the dissertation.",
        "",
        "| arm | responsiveness (Spearman) | n steps | note |",
        "|---|---|---|---|",
    ]
    for arm, e in resp.items():
        lines.append(
            f"| {arm} | {_fmt(e['score'])} | {e['n_steps']} | {e['reason']} |"
        )
    if fed_arms:
        summary_bits = ", ".join(f"{a}={_fmt(e['score'])} (n={e['n_steps']})" for a, e in fed_arms.items())
        lines += [
            "",
            f"**Tail-fed arms (the only ones this signal is defined for): {summary_bits}.** A responsiveness "
            "near zero / negative on the development archive says the reflection edits did NOT track the fed "
            "tail-stat deltas — the qualitative counterpart to a null headline (it tells the forensics story "
            "of WHAT the LLM invents about tail risk, not that distributional feedback was causally used).",
        ]

    h = result["hacking"]
    lines += [
        "",
        "## 2. Reward-hacking taxonomy (Skalse 2022; Pan 2022; Hadfield-Menell 2017)",
        f"Flagged {h['n_flagged']} / {h['n_total']} candidates. "
        f"unbounded_magnitude={h['counts']['unbounded_magnitude']}, "
        f"specification_gaming={h['counts']['specification_gaming']}, "
        f"proxy_no_tail={h['counts']['proxy_no_tail']}, "
        f"tautology={h['counts']['tautology']}.",
        "",
        "`unbounded_magnitude` = a return-over-(floored)-volatility divide (`port_ret / (std + 1e-8)`, "
        "`ema_ret / ema_std`) that is UNBOUNDED ABOVE as realized variance → 0 — the agent games it by "
        "collapsing variance, the SAC critic target diverges, and training destabilizes. This is the REAL "
        "prototype failure mode (the critic explosions in `outputs/prototype/anomalies.jsonl`) and is "
        "ORTHOGONAL to fitness-collapse: it is flagged on the code SHAPE regardless of the validation number.",
        "",
        "| arm | candidate | gen | val fitness | uses tail | flags | unbounded term(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in h["candidates"]:
        if not t["flags"]:
            continue  # the taxonomy table lists only FLAGGED candidates (the archive can hold hundreds)
        flags = ", ".join(t["flags"])
        vterms = "; ".join(t.get("variance_floor_terms", [])) or "—"
        lines.append(
            f"| {t['arm']} | {t['candidate_id']} | {t['generation']} | "
            f"{_fmt(t['val_fitness'])} | {t['uses_tail']} | {flags} | `{vterms}` |"
        )
    if h["n_flagged"] == 0:
        lines.append("| _(none flagged)_ | | | | | | |")

    mag = result.get("reward_magnitude")
    if mag is not None:
        lines += [
            "",
            "## 3. MEASURED reward-magnitude audit (the critic-explosion source, quantified)",
            f"Each archived reward is EXECUTED on a representative probe; the table reports its empirical "
            f"peak `|reward total|`. {mag['n_over_bound']} / {mag['n_total']} candidates exceed the sane "
            f"per-step bound {mag['bound']:.0f} (worst = {mag['worst_max_abs_total']:.4g}); "
            f"{mag['n_nonfinite']} emitted a non-finite reward. A reward of magnitude `R` drives the SAC "
            "Bellman target toward `R/(1-gamma)` (`~1e6` at `gamma=0.99` for `R~1e4`), which is the verified "
            "source of the `~5e6` critic-loss spikes in `outputs/prototype/anomalies.jsonl`. REPORT-ONLY: a "
            "diverged candidate already loses selection, and the PopArt scale-normalization in the trainer "
            "(`src/agents/popart.py`) makes the critic invariant to this scale — this table makes the "
            "pathology auditable, it does NOT change any reward.",
            "",
            "| arm | candidate | gen | val fitness | max \\|reward\\| | finite | over bound |",
            "|---|---|---|---|---|---|---|",
        ]
        shown = [d for d in mag["candidates"] if d["over_bound"] or not d["finite"]]
        for d in shown[:25]:  # the worst offenders (sorted by magnitude); the archive can hold hundreds
            mx = f"{d['max_abs_total']:.4g}" if d["max_abs_total"] is not None else "n/a"
            lines.append(
                f"| {d['arm']} | {d['candidate_id']} | {d['generation']} | "
                f"{_fmt(d['val_fitness'])} | {mx} | {d['finite']} | {d['over_bound']} |"
            )
        if not shown:
            lines.append("| _(none over bound)_ | | | | | | |")

    diff = result.get("program_differential")
    if diff is not None:
        lines += _program_differential_markdown(diff)

    lines += [
        "",
        "## 5. Per-generation summary (fitness + reward-code complexity + tail-usage trend)",
        "Supporting context: does the authored code grow more complex / more tail-aware as the loop runs?",
        "",
        "| arm | gen | n | best fit | mean fit | mean LOC | mean ops | frac uses-tail | mean tail-terms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (arm, gen), s in result["per_generation"].items():
        lines.append(
            f"| {arm} | {gen} | {s['n']} | {_fmt(s['best_fitness'])} | {_fmt(s['mean_fitness'])} | "
            f"{s['mean_loc']:.1f} | {s['mean_ops']:.1f} | {s['frac_uses_tail']:.2f} | "
            f"{s['mean_tail_terms']:.2f} |"
        )

    lines += [
        "",
        "## Notes",
        "- Reads results ONLY through `src.io.results.load_all` (audit C-1) and never writes a run "
        "record (the archive is read-only).",
        "- The interpretability lens (`_TAIL_TERMS`) and the archive walk (`load_arms`) are REUSED "
        "from `scripts/analyze_results.py`.",
        "- unbounded_magnitude = a return / (floored volatility) divide that diverges as variance → 0 "
        "(the critic-explosion source); flagged on the code shape, independent of fitness.",
        "- specification_gaming = a reward that references a logged tail component while held-out "
        "fitness collapsed (≤ 0): the proxy was driven up while the true objective fell.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``reward_forensics.md`` + ``reward_forensics.json`` into ``out_dir``.

    Writes ONLY into ``out_dir`` (never the run archive). Returns the markdown path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "reward_forensics.md"
    report.write_text(_markdown(result), encoding="utf-8")
    # Tuple keys in per_generation are not JSON-serializable -> stringify for the JSON dump.
    serializable = dict(result)
    serializable["per_generation"] = {
        f"{arm}::g{gen}": v for (arm, gen), v in result["per_generation"].items()
    }
    (out / "reward_forensics.json").write_text(
        json.dumps(serializable, indent=2, default=str), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect LLM-authored rewards — reward forensics (FINAL_PLAN Phase 4.C).",
    )
    parser.add_argument("--results-dir", default="outputs/runs")
    parser.add_argument("--out-dir", default="outputs/tables")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = inspect(args.results_dir)
    report = write_report(result, args.out_dir)
    h = result["hacking"]
    print(
        f"[inspect_rewards] {len(result['arms'])} arm(s) from {args.results_dir} -> {report}"
    )
    print("  HEADLINE responsiveness (does reflection track the fed tail? — tail-fed arms only):")
    for arm, e in result["responsiveness"].items():
        print(f"    {arm:>14}: responsiveness={_fmt(e['score'])} (steps={e['n_steps']})")
    print(
        f"  reward-hacking flags: {h['n_flagged']}/{h['n_total']} "
        f"(unbounded_magnitude={h['counts']['unbounded_magnitude']}, "
        f"specification_gaming={h['counts']['specification_gaming']}, "
        f"proxy_no_tail={h['counts']['proxy_no_tail']}, tautology={h['counts']['tautology']})"
    )
    pd = result.get("program_differential") or {}
    if pd.get("differential"):
        print("  reward-program tail-construct differential (distributional − comparator; DIRECTIONAL):")
        for comp, d in pd["differential"].items():
            print(
                f"    vs {comp:>14}: tail-rate Δ={d['tail_rate_delta']:+.3f} "
                f"(more={d['distributional_references_tail_more']})"
            )
        if not pd.get("components_archived", False):
            print("    NB: per-step `components` not archived — declared-term set only (no runtime activity).")


if __name__ == "__main__":
    main()
