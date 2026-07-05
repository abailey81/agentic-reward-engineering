"""Information-utilization gap — report-only mechanism instrument (PREREGISTRATION §2a micro-anchor (d)).

A predicted null invites the objection *"the fed tail vector carried no information beyond the scalar
anyway."* This module measures that directly on the **ACTUAL fed feedback sequences archived per
generation**: the redundancy of the six-component tail vector given the concurrently fed scalar summary
(the validation-DSR header line), pooled over generations. The complement ``1 − redundancy`` is the
**fed-but-unusable-by-a-scalar information**; contrasted with the SQ1 responsiveness estimate it yields
the **information-utilization gap** — how much extra signal the designer was GIVEN vs how much its code
USED. Registered direction under the numeracy hypothesis: substantial non-redundant information,
near-zero utilization.

Data + units. Every reflective candidate's archived record embeds the feedback it was fed (rendered into
``feedback_block`` or the full ``prompt`` by :mod:`src.feedback.schema`). All candidates of one generation
see the SAME block (best-of-previous feedback), so parsed blocks are DEDUPLICATED per (arm, generation):
the observation unit is one distinct fed block ≈ one generation. Two channels are reported:

* ``fed_rendered`` — the values exactly as the LLM saw them, parsed back out of the rendered text (the
  header scalar at 6 dp, the tail lines at 4 dp — NOTE: these were 2 dp / 3 dp until 2026-07-27; the
  precisions are NOT cosmetic here because "render-precision quantization is part of the estimand"
  below, so raising them MOVED this channel's quantization floor. #87 measured the old header at
  `.2f` rendering 328 of 591 archived headers as literally "0.00" — i.e. the `scalar_degenerate`
  branch below was the LIVE case, not a defensive one — and #98 raised `_fmt` to `.4f` so the legible
  and raw arms carry equal resolution. Parsing is precision-AGNOSTIC by construction (both regexes
  accept any decimal count, and the ``fed_underlying`` match re-renders through the live
  ``schema._fmt``), so only this description needed correcting; no behaviour changed). This is the
  REGISTERED channel and the only one in
  which the ``placebo_shuffled`` calibration floor is meaningful (its rendered vector's label↔value
  linkage is destroyed by the candidate-seeded derangement; its ARCHIVED metrics keep the true linkage).
  Render-precision quantization is part of the estimand here (the same legibility phenomenon SQ3 probes):
  a fed scalar that quantizes to a constant is flagged ``scalar_degenerate`` and the vector is then fully
  non-redundant BY CONSTRUCTION (redundancy 0; a constant explains nothing).
* ``fed_underlying`` — the same fed observations at FULL archived precision, recovered by matching each
  fed block back to the parent record whose measured ``tail_stats`` render to the fed strings (the
  derangement preserves the value multiset, so matching is multiset-based). This channel removes the
  quantization floor and is the primary magnitude readout; it is NOT reported for the floor arm (whose
  underlying linkage is intact by construction).

Redundancy per component ``j``: the R² of the simple OLS regression of the fed component on the fed
scalar (= squared Pearson r for one regressor) AND a Spearman-based rank redundancy ``ρ_s²`` — both
reported, pooled as the across-component mean, with percentile bootstrap CIs over the generation-level
fed observations (Efron & Tibshirani 1993). Deterministic (seeded bootstrap; numpy + scipy only).

Report-only, DISJOINT from the frozen ``m = 6`` testing family; never gates a hypothesis. Honest
degradation: too few fed observations, an absent floor arm, or a missing responsiveness estimate degrade
to ``{"executed": False, "reason": ...}`` sub-blocks — never fabricated numbers.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import stats

# Single source of truth for the fed-block labels + the raw renderer (src/feedback/schema.py). Importing
# the private names keeps the parser in exact lockstep with the renderer (a renamed label breaks tests,
# not silently the parser) — the same private-reuse precedent as analyze_campaign -> inspect_rewards.
from src.feedback.schema import _DIST_FIELDS, _fmt
from src.inference.responsiveness import MIN_BOOT_VALID_FRACTION

__all__ = ["extract_fed_observations", "information_gap"]

#: The scalar-summary header line every arm's block carries (src/feedback/schema.py _HEADER, 2 dp).
_SCALAR_RE = re.compile(r"Your previous reward scored:\s*([+-]?\d+(?:\.\d+)?)")

#: Per-component value patterns, in _DIST_FIELDS order. "CVaR 1%" cannot false-match inside "CVaR 10%"
#: because the literal "%" must follow the digits.
_COMPONENT_RES: list[tuple[str, re.Pattern[str]]] = [
    (fid, re.compile(re.escape(label) + r"\s*:\s*([+-]?\d+(?:\.\d+)?)"))
    for fid, label in _DIST_FIELDS
]

#: Field ids in canonical order (the vector layout used throughout this module).
_FIELD_IDS: tuple[str, ...] = tuple(fid for fid, _ in _DIST_FIELDS)


def _parse_fed_block(text: str) -> dict[str, Any] | None:
    """Parse one rendered feedback text into ``{"scalar", "scalar_str", "vector", "vector_strs"}``.

    Returns ``None`` unless the header scalar AND all six tail components are present (so scalar /
    scalar_cvar5 / placebo blocks — which carry at most one tail line — never produce an observation).
    The captured strings are kept verbatim for the multiset parent-matching in the underlying channel.
    """
    if not text:
        return None
    s = _SCALAR_RE.search(text)
    if s is None:
        return None
    values: list[float] = []
    strs: list[str] = []
    for _fid, pat in _COMPONENT_RES:
        m = pat.search(text)
        if m is None:
            return None
        strs.append(m.group(1))
        values.append(float(m.group(1)))
    return {
        "scalar": float(s.group(1)),
        "scalar_str": s.group(1),
        "vector": np.asarray(values, dtype=float),
        "vector_strs": tuple(strs),
    }


def _generation(record: dict[str, Any]) -> int:
    try:
        return int(record.get("generation", 0))
    except (TypeError, ValueError):
        return 0


def _tail_stats_vector(record: dict[str, Any]) -> np.ndarray | None:
    """The archived measured tail vector in ``_FIELD_IDS`` order, or ``None`` when absent/incomplete."""
    stats_d = (record.get("metrics") or {}).get("tail_stats")
    if not isinstance(stats_d, dict) or not all(fid in stats_d for fid in _FIELD_IDS):
        return None
    vec = np.asarray([float(stats_d[fid]) for fid in _FIELD_IDS], dtype=float)
    return vec if np.all(np.isfinite(vec)) else None


def extract_fed_observations(
    records: list[dict[str, Any]], arm: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The DEDUPLICATED fed six-vector observations for one arm, with parent-matched underlying values.

    Walks the arm's records in (generation, candidate_id) order, parses each record's fed text
    (``feedback_block`` falling back to ``prompt``), and deduplicates identical parsed blocks within a
    generation (the loop feeds every candidate of a generation the same best-of-previous block). Each
    observation then gets an UNDERLYING full-precision pair when exactly one previous-generation record's
    measured ``tail_stats`` renders (via the schema's raw ``_fmt``) to the fed value MULTISET (multiset,
    not ordered, so the placebo_shuffled derangement still matches its parent) AND that parent carries a
    finite ``val_fitness``; ambiguous or missing parents are counted, never guessed.

    Returns ``(observations, counters)`` where each observation is ``{"generation", "scalar", "vector",
    "underlying_scalar", "underlying_vector"}`` (underlying entries ``None`` when unmatched) and counters
    report ``{"n_records_parsed", "n_parent_unmatched", "n_parent_ambiguous"}``.
    """
    arm_records = sorted(
        [r for r in records if r.get("arm") == arm],
        key=lambda r: (_generation(r), str(r.get("candidate_id", r.get("run_id", "")))),
    )
    by_generation: dict[int, list[dict[str, Any]]] = {}
    for r in arm_records:
        by_generation.setdefault(_generation(r), []).append(r)

    observations: list[dict[str, Any]] = []
    counters = {"n_records_parsed": 0, "n_parent_unmatched": 0, "n_parent_ambiguous": 0}
    for gen in sorted(by_generation):
        seen: set[tuple[str, ...]] = set()
        for r in by_generation[gen]:
            # 2026-07-05 (M14 construct fix): the fed text is the archived PROMPT — what the designer
            # actually SAW (the previous generation's best block rides in it verbatim). The record's
            # own ``feedback_block`` is the block built FROM this candidate (fed to the NEXT
            # generation): the old block-first read (a) computed the redundancy on the wrong,
            # off-by-one-generation sequence and (b) defeated the sibling de-duplication (every
            # candidate's own block is distinct, inflating one true fed observation per generation
            # into K pseudo-observations). Own-block survives only as the legacy fallback for
            # pre-Rank-14 archives without a prompt, gated to generation >= 1 (gen-0 was fed nothing).
            fed_text = str(r.get("prompt") or "")
            if not fed_text and _generation(r) >= 1:
                fed_text = str(r.get("feedback_block") or "")
            parsed = _parse_fed_block(fed_text)
            if parsed is None:
                continue
            counters["n_records_parsed"] += 1
            key = (parsed["scalar_str"], *parsed["vector_strs"])
            if key in seen:  # same block fed to a sibling candidate — one observation per distinct block
                continue
            seen.add(key)

            # NOTE: RAW-RENDERING ONLY (verified 2026-07-26). The match below compares against the schema's
            # RAW ``_fmt`` (``-0.090``). A LEGIBLE-rendered block (``legible_render: true`` — the
            # numeracy sub-experiment, ADR-039) renders basis points / percent (``-900 bps``, ``8.5%``),
            # which this parser HAPPILY PARSES (the regex stops before the unit) but which can never
            # equal the raw multiset — so every such record lands in ``n_parent_unmatched``. That
            # degradation is honest (counted, never guessed), but the SYMPTOM is misleading: a 100%
            # unmatched rate on a legible leg is a FORMAT mismatch, NOT missing parents. Read the
            # counters with the leg's ``legible_render`` setting in hand; extend ``_COMPONENT_RES`` +
            # this comparison to the bps/percent units before running this instrument on a legible leg.
            #
            # Underlying full-precision parent: the g-1 record whose measured tail_stats render (via the
            # schema's raw _fmt) to the fed value MULTISET (multiset => derangement-proof) and which
            # carries a FINITE val_fitness (only so underlying_scalar exists). The fed SCALAR is
            # deliberately NOT re-matched against the parent's val_fitness render: the 6-value tail
            # multiset already pins the parent, and uniqueness is enforced below — 0 or 2+ matches are
            # counted (unmatched/ambiguous), never guessed.
            fed_multiset = tuple(sorted(parsed["vector_strs"]))
            matches: list[dict[str, Any]] = []
            for p in by_generation.get(gen - 1, []):
                vec = _tail_stats_vector(p)
                fitness = (p.get("metrics") or {}).get("val_fitness")
                if vec is None or fitness is None or not np.isfinite(float(fitness)):
                    continue
                if tuple(sorted(_fmt(v) for v in vec)) == fed_multiset:
                    matches.append(p)
            underlying_scalar: float | None = None
            underlying_vector: np.ndarray | None = None
            if len(matches) == 1:
                underlying_scalar = float(matches[0]["metrics"]["val_fitness"])
                underlying_vector = _tail_stats_vector(matches[0])
            elif len(matches) == 0:
                counters["n_parent_unmatched"] += 1
            else:
                counters["n_parent_ambiguous"] += 1

            observations.append(
                {
                    "generation": gen,
                    "scalar": parsed["scalar"],
                    "vector": parsed["vector"],
                    "underlying_scalar": underlying_scalar,
                    "underlying_vector": underlying_vector,
                }
            )
    return observations, counters


def _r2(s: np.ndarray, v: np.ndarray) -> float:
    """R² of the simple OLS regression ``v ~ s`` (= squared Pearson r); NaN when either side is constant."""
    if s.size < 2 or np.ptp(s) == 0 or np.ptp(v) == 0:
        return float("nan")
    r = float(np.corrcoef(s, v)[0, 1])
    return r * r


def _rank_rho2(s: np.ndarray, v: np.ndarray) -> float:
    """Spearman rank redundancy ``ρ_s²``; NaN when either side is rank-degenerate."""
    if s.size < 2 or np.ptp(s) == 0 or np.ptp(v) == 0:
        return float("nan")
    rho = float(stats.spearmanr(s, v).statistic)
    return rho * rho


def _row_nanmean(mat: np.ndarray) -> np.ndarray:
    """Per-row nanmean that leaves all-NaN rows (degenerate resamples) NaN without a RuntimeWarning."""
    out = np.full(mat.shape[0], np.nan)
    has = np.isfinite(mat).any(axis=1)
    if has.any():
        out[has] = np.nanmean(mat[has], axis=1)
    return out


def _ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _channel_redundancy(
    s: np.ndarray,
    v: np.ndarray,
    *,
    min_obs: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Per-component + pooled redundancy of the fed vector ``v`` (n × 6) given the fed scalar ``s`` (n).

    Percentile bootstrap over the n generation-level fed observations (rows); degenerate resamples yield
    NaN and are dropped per statistic, with ``ci_reliable`` gated on the valid-boot fraction exactly as in
    :mod:`src.inference.responsiveness`. A zero-variance scalar (e.g. every fed header quantized to the
    same 2-dp value) short-circuits: redundancy is 0 BY CONSTRUCTION (a constant explains nothing), rank
    redundancy is undefined, and ``scalar_degenerate`` flags the finding.
    """
    n = int(s.size)
    if n < min_obs:
        return {"status": "no_data", "reason": f"need >= {min_obs} fed observations; got {n}", "n": n}

    if np.ptp(s) == 0:
        components = {
            fid: {
                "r2": 0.0,
                "r2_ci_low": float("nan"),
                "r2_ci_high": float("nan"),
                "rank_rho2": float("nan"),
                "rank_ci_low": float("nan"),
                "rank_ci_high": float("nan"),
                "degenerate": bool(np.ptp(v[:, j]) == 0),
            }
            for j, fid in enumerate(_FIELD_IDS)
        }
        return {
            "status": "ok",
            "n": n,
            "scalar_degenerate": True,
            "components": components,
            "pooled": {
                "mean_r2": 0.0,
                "mean_r2_ci_low": float("nan"),
                "mean_r2_ci_high": float("nan"),
                "mean_rank_rho2": float("nan"),
                "mean_rank_ci_low": float("nan"),
                "mean_rank_ci_high": float("nan"),
                "non_redundant_r2": 1.0,
                "non_redundant_rank": float("nan"),
            },
            "n_boot_valid": 0,
            "ci_reliable": False,
        }

    k = len(_FIELD_IDS)
    point_r2 = np.asarray([_r2(s, v[:, j]) for j in range(k)])
    point_rho2 = np.asarray([_rank_rho2(s, v[:, j]) for j in range(k)])

    boot_r2 = np.full((n_boot, k), np.nan)
    boot_rho2 = np.full((n_boot, k), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sb, vb = s[idx], v[idx]
        if np.ptp(sb) == 0:
            continue  # degenerate resample -> NaN row, dropped per statistic
        for j in range(k):
            boot_r2[b, j] = _r2(sb, vb[:, j])
            boot_rho2[b, j] = _rank_rho2(sb, vb[:, j])
    pooled_r2_boots = _row_nanmean(boot_r2)
    pooled_rho2_boots = _row_nanmean(boot_rho2)

    components: dict[str, dict[str, Any]] = {}
    for j, fid in enumerate(_FIELD_IDS):
        r2_lo, r2_hi = _ci(boot_r2[:, j])
        rk_lo, rk_hi = _ci(boot_rho2[:, j])
        components[fid] = {
            "r2": float(point_r2[j]),
            "r2_ci_low": r2_lo,
            "r2_ci_high": r2_hi,
            "rank_rho2": float(point_rho2[j]),
            "rank_ci_low": rk_lo,
            "rank_ci_high": rk_hi,
            "degenerate": bool(not np.isfinite(point_r2[j])),
        }

    mean_r2 = float(np.nanmean(point_r2)) if np.isfinite(point_r2).any() else float("nan")
    mean_rho2 = float(np.nanmean(point_rho2)) if np.isfinite(point_rho2).any() else float("nan")
    p_r2_lo, p_r2_hi = _ci(pooled_r2_boots)
    p_rk_lo, p_rk_hi = _ci(pooled_rho2_boots)
    n_boot_valid = int(np.isfinite(pooled_r2_boots).sum())
    return {
        "status": "ok",
        "n": n,
        "scalar_degenerate": False,
        "components": components,
        "pooled": {
            "mean_r2": mean_r2,
            "mean_r2_ci_low": p_r2_lo,
            "mean_r2_ci_high": p_r2_hi,
            "mean_rank_rho2": mean_rho2,
            "mean_rank_ci_low": p_rk_lo,
            "mean_rank_ci_high": p_rk_hi,
            "non_redundant_r2": float(1.0 - mean_r2) if np.isfinite(mean_r2) else float("nan"),
            "non_redundant_rank": float(1.0 - mean_rho2) if np.isfinite(mean_rho2) else float("nan"),
        },
        "n_boot_valid": n_boot_valid,
        "ci_reliable": bool(n_boot_valid >= MIN_BOOT_VALID_FRACTION * int(n_boot)),
    }


def _arm_channels(
    observations: list[dict[str, Any]],
    *,
    min_obs: int,
    n_boot: int,
    rng: np.random.Generator,
    include_underlying: bool,
) -> dict[str, Any]:
    """Both redundancy channels for one arm's fed observations (underlying optional — floor arm skips it)."""
    s = np.asarray([o["scalar"] for o in observations], dtype=float)
    v = (
        np.stack([o["vector"] for o in observations])
        if observations
        else np.empty((0, len(_FIELD_IDS)))
    )
    channels: dict[str, Any] = {
        "fed_rendered": _channel_redundancy(s, v, min_obs=min_obs, n_boot=n_boot, rng=rng)
    }
    if include_underlying:
        matched = [o for o in observations if o["underlying_vector"] is not None]
        su = np.asarray([o["underlying_scalar"] for o in matched], dtype=float)
        vu = (
            np.stack([o["underlying_vector"] for o in matched])
            if matched
            else np.empty((0, len(_FIELD_IDS)))
        )
        channels["fed_underlying"] = _channel_redundancy(
            su, vu, min_obs=min_obs, n_boot=n_boot, rng=rng
        )
        channels["fed_underlying"]["n_matched_parents"] = len(matched)
    return channels


def _pooled_non_redundant(channel: dict[str, Any]) -> float:
    """The channel's pooled non-redundant fraction — rank-based when finite, else the R² complement."""
    pooled = channel.get("pooled", {})
    nr = pooled.get("non_redundant_rank", float("nan"))
    if not np.isfinite(nr):
        nr = pooled.get("non_redundant_r2", float("nan"))
    return float(nr)


def information_gap(
    records: list[dict[str, Any]],
    *,
    responsiveness: dict[str, Any] | None = None,
    arms: tuple[str, ...] = ("distributional",),
    floor_arm: str = "placebo_shuffled",
    n_boot: int = 2000,
    min_obs: int = 3,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The information-utilization gap over an archive's fed feedback sequences (micro-anchor (d)).

    Parameters
    ----------
    records : list of dict
        Per-candidate run records (``load_campaign_records`` / ``src.io.results.load_all``). Only the
        rendered fed text (``feedback_block``/``prompt``), ``metrics.tail_stats``, ``metrics.val_fitness``,
        ``arm`` and ``generation`` are read — VALIDATION-side quantities only.
    responsiveness : dict or None
        The SQ1 responsiveness output (``src.inference.responsiveness.responsiveness``), supplied by the
        caller — NEVER recomputed here. ``None`` degrades the utilization-gap sub-block honestly.
    arms : tuple of str
        The tail-VECTOR-fed arms to analyse (default: the ``distributional`` arm — the only confirmatory
        arm fed the six-component vector with intact linkage).
    floor_arm : str
        The calibration-floor arm whose FED vector's label↔value linkage is destroyed by construction
        (default ``placebo_shuffled``); analysed on the rendered channel only.
    n_boot, min_obs, rng
        Percentile-bootstrap replicates over fed observations; the minimum observations per channel; the
        seeded generator (default ``default_rng(0)`` — byte-deterministic).

    Returns
    -------
    dict
        ``{"status", "executed", "arms": {arm: {"n_obs", "n_generations", counters, "channels"}},
        "floor", "floor_comparison", "utilization_gap", "n_boot"}``. Every sub-block degrades to an
        ``executed: False`` / ``no_data`` form rather than fabricating numbers.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    out: dict[str, Any] = {"status": "ok", "executed": True, "n_boot": int(n_boot), "arms": {}}
    any_ok = False
    for arm in arms:
        observations, counters = extract_fed_observations(records, arm)
        channels = _arm_channels(
            observations, min_obs=min_obs, n_boot=n_boot, rng=rng, include_underlying=True
        )
        entry = {
            "n_obs": len(observations),
            "n_generations": len({o["generation"] for o in observations}),
            **counters,
            "channels": channels,
        }
        out["arms"][arm] = entry
        if channels["fed_rendered"].get("status") == "ok":
            any_ok = True

    # Calibration floor — rendered channel ONLY (the archived metrics of the floor arm keep the TRUE
    # vector<->scalar linkage; only the RENDERED, deranged values realise the destroyed-linkage floor).
    floor_obs, floor_counters = extract_fed_observations(records, floor_arm)
    floor_channels = _arm_channels(
        floor_obs, min_obs=min_obs, n_boot=n_boot, rng=rng, include_underlying=False
    )
    if floor_channels["fed_rendered"].get("status") == "ok":
        out["floor"] = {
            "arm": floor_arm,
            "executed": True,
            "n_obs": len(floor_obs),
            **floor_counters,
            "fed_rendered": floor_channels["fed_rendered"],
        }
    else:
        out["floor"] = {
            "arm": floor_arm,
            "executed": False,
            "reason": floor_channels["fed_rendered"].get(
                "reason", f"no parseable fed blocks for floor arm {floor_arm!r}"
            ),
        }

    # Floor comparison (rendered channel): linkage-attributable redundancy = arm minus floor.
    primary = arms[0] if arms else None
    prim_rendered = (
        out["arms"].get(primary, {}).get("channels", {}).get("fed_rendered", {})
        if primary
        else {}
    )
    if (
        out["floor"].get("executed")
        and prim_rendered.get("status") == "ok"
        and not prim_rendered.get("scalar_degenerate")
        and not out["floor"]["fed_rendered"].get("scalar_degenerate")
    ):
        fl = out["floor"]["fed_rendered"]["pooled"]
        pr = prim_rendered["pooled"]
        out["floor_comparison"] = {
            "executed": True,
            "arm": primary,
            "floor_arm": floor_arm,
            "arm_mean_r2": pr["mean_r2"],
            "floor_mean_r2": fl["mean_r2"],
            "linkage_attributable_r2": float(pr["mean_r2"] - fl["mean_r2"]),
            "arm_mean_rank_rho2": pr["mean_rank_rho2"],
            "floor_mean_rank_rho2": fl["mean_rank_rho2"],
        }
    else:
        reasons = []
        if not out["floor"].get("executed"):
            reasons.append(f"floor arm {floor_arm!r} unavailable")
        if prim_rendered.get("status") != "ok":
            reasons.append(f"primary arm {primary!r} rendered channel unavailable")
        if prim_rendered.get("scalar_degenerate") or (
            out["floor"].get("executed")
            and out["floor"]["fed_rendered"].get("scalar_degenerate")
        ):
            reasons.append("fed scalar degenerate at render precision (comparison undefined)")
        out["floor_comparison"] = {"executed": False, "reason": "; ".join(reasons) or "unavailable"}

    # Utilization gap — GIVEN (non-redundant fed fraction) vs USED (|SQ1 responsiveness coefficient|).
    # A descriptive index (both terms bounded [0, 1]; different estimands), never a parameter estimate.
    if responsiveness is None or responsiveness.get("status") != "ok":
        out["utilization_gap"] = {
            "executed": False,
            "reason": (
                "no SQ1 responsiveness estimate supplied"
                if responsiveness is None
                else f"responsiveness status={responsiveness.get('status')!r}"
            ),
        }
    else:
        channel_name, channel = None, None
        arm_channels = out["arms"].get(primary, {}).get("channels", {}) if primary else {}
        for name in ("fed_underlying", "fed_rendered"):  # full precision preferred
            cand = arm_channels.get(name, {})
            if cand.get("status") == "ok" and np.isfinite(_pooled_non_redundant(cand)):
                channel_name, channel = name, cand
                break
        if channel is None:
            out["utilization_gap"] = {
                "executed": False,
                "reason": f"no usable redundancy channel for arm {primary!r}",
            }
        else:
            given = _pooled_non_redundant(channel)
            used = abs(float(responsiveness.get("coef", float("nan"))))
            out["utilization_gap"] = {
                "executed": True,
                "arm": primary,
                "channel": channel_name,
                "non_redundant_fed": given,
                "responsiveness_abs_coef": used,
                "responsiveness_ci": [
                    responsiveness.get("ci_low"),
                    responsiveness.get("ci_high"),
                ],
                "gap": float(given - used),
                "note": (
                    "descriptive index: GIVEN = 1 - pooled redundancy (rank-based when finite), "
                    "USED = |SQ1 Spearman coefficient|; both bounded [0, 1] but different estimands"
                ),
            }

    if not any_ok:
        out["status"] = "no_data"
        out["executed"] = False
        out["reason"] = (
            f"no analysed arm produced >= {min_obs} parseable fed six-vector observations "
            f"(arms={list(arms)})"
        )
    return out
