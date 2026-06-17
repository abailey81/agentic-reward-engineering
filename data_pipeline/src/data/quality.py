"""Quality scoreboard, lineage map, and datasheet generation (stage 12).

Per-series quality score = weighted {coverage, reconciliation agreement, integrity
flag rate, freshness} (weights: config data.platform.quality_weights). The datasheet
follows Gebru et al., "Datasheets for Datasets" (CACM 2021), auto-filled from
manifests; anything not yet measurable stays ⟨TBD⟩ — never fabricated. The
stage-seed file drafts ONE dissertation data-chapter paragraph per pipeline stage,
each slot pulling a real number when it exists.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import get
from .vault import lineage_chain, manifest_entries

ROOT = Path(__file__).resolve().parent.parent.parent


def series_quality(coverage: float, recon_agreement: float, integrity_flag_rate: float,
                   freshness: float) -> float:
    """Weighted score in [0,1]: coverage & reconciliation & freshness reward presence;
    the integrity term rewards a LOW flag rate (1 - rate)."""
    w = get("data.platform.quality_weights")
    for v in (coverage, recon_agreement, integrity_flag_rate, freshness):
        if not 0.0 <= v <= 1.0:
            raise ValueError("quality inputs must be fractions in [0, 1]")
    return float(w["coverage"] * coverage + w["reconciliation"] * recon_agreement
                 + w["integrity"] * (1.0 - integrity_flag_rate) + w["freshness"] * freshness)


def coverage_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-name coverage stats over the panel's session index."""
    rows = []
    for col in wide.columns:
        s = wide[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        live = s.loc[first:last] if first is not None else s.iloc[0:0]
        rows.append({"name": col,
                     "first_obs": first, "last_obs": last,
                     "n_obs": int(s.notna().sum()),
                     "coverage_live_span": round(float(live.notna().mean()), 6) if len(live) else 0.0,
                     "coverage_full_panel": round(float(s.notna().mean()), 6)})
    return pd.DataFrame(rows)


def scoreboard(coverage_df: pd.DataFrame, recon_rows: list | None = None,
               integrity_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """Assemble the per-series quality scoreboard from stage outputs."""
    recon_by_name = {}
    for r in recon_rows or []:
        agree = 1.0 if r.n_overlap == 0 else max(0.0, 1.0 - r.days_over_tol / r.n_overlap)
        recon_by_name[r.name] = agree
    integrity_counts = integrity_counts or {}
    rows = []
    for _, c in coverage_df.iterrows():
        name = c["name"]
        cov = float(c["coverage_live_span"])
        rec = float(recon_by_name.get(name, 0.0))
        n_obs = max(1, int(c["n_obs"]))
        flag_rate = min(1.0, integrity_counts.get(name, 0) / n_obs)
        fresh = 1.0 if pd.notna(c["last_obs"]) else 0.0
        rows.append({"name": name, "coverage": round(cov, 4),
                     "recon_agreement": round(rec, 4),
                     "integrity_flag_rate": round(flag_rate, 6),
                     "freshness": fresh,
                     "quality_score": round(series_quality(cov, rec, flag_rate, fresh), 4)})
    return pd.DataFrame(rows).sort_values("quality_score", ascending=False).reset_index(drop=True)


def lineage_map_markdown(gold_relpaths: list[str]) -> str:
    """Raw->gold hash chains for the named gold artifacts (audit view)."""
    lines = ["# Lineage map (raw -> gold hash chain)", ""]
    for rel in gold_relpaths:
        lines.append(f"## {rel}")
        chain = lineage_chain(rel)
        if not chain:
            lines.append("*(no recorded lineage — artifact predates the lineage log "
                         "or is a raw pull)*")
            continue
        for edge in chain:
            lines.append(f"- `{edge['output']['relpath']}` `{edge['output']['sha256'][:12]}…` "
                         f"← stage **{edge['stage']}** ←")
            for i in edge["inputs"]:
                lines.append(f"    - `{i['relpath']}` `{i['sha256'][:12]}…`")
        lines.append("")
    return "\n".join(lines)


_GEBRU_SECTIONS = (
    ("Motivation", "Created to answer the pre-registered research question (PREREGISTRATION §1): "
     "LLM-evolved rewards under distributional feedback vs hand-designed rewards on a "
     "survivorship-bias-free 30-stock US large-cap universe, 2005–2025. Funded/required by the "
     "UCL MSc IFTE0008 dissertation; no commercial purpose."),
    ("Composition", None),
    ("Collection process", None),
    ("Preprocessing / cleaning / labeling", "Medallion pipeline (src/data): structural validation, "
     "corporate-action integrity, Ince–Porter screens (flag-only), Shumway delisting corrections "
     "(logged), missing-data taxonomy with zero interpolation (R4). Raw layer is immutable; every "
     "transform is lineage-recorded."),
    ("Uses", "Training/evaluating deep-RL portfolio policies inside this dissertation only; vendor "
     "licence terms prohibit redistribution of raw data — the repo ships manifests, not payloads."),
    ("Distribution", "NOT distributed. SHA-256 manifests + pull provenance allow an entitled party "
     "to reproduce byte-exact artifacts."),
    ("Maintenance", "Maintained until dissertation submission (2026-09-01); "
     "manifest + ADR log document every change."),
)


def generate_datasheet(out_path: Path | None = None) -> Path:
    """docs/DATASHEET_v1.md from live manifests (Gebru et al. structure)."""
    entries = manifest_entries()
    by_layer: dict[str, list[dict]] = {}
    for e in entries:
        by_layer.setdefault(e["layer"], []).append(e)
    comp_lines = []
    for layer in ("raw", "staged", "clean", "gold"):
        arts = by_layer.get(layer, [])
        rows = sum(a["rows"] for a in arts)
        comp_lines.append(f"- **{layer}**: {len(arts)} artifacts, {rows:,} rows"
                          + (f" (e.g. {arts[0]['name']})" if arts else " ⟨TBD — no pulls frozen yet⟩"))
    vendors = sorted({e.get("vendor") for e in entries if e.get("vendor")})
    collection = (f"Pulled via src/data/acquire.py (journaled, rate-governed, provenance-sidecar'd) "
                  f"from vendors: {', '.join(map(str, vendors)) if vendors else '⟨TBD⟩'}. "
                  "Entitlement evidence: docs/evidence/entitlement_report.md.")

    lines = [f"# Datasheet v1 — generated {datetime.now(timezone.utc).date().isoformat()} "
             "(Gebru et al., CACM 2021; auto-filled from data/manifest)", ""]
    for title, body in _GEBRU_SECTIONS:
        lines.append(f"## {title}")
        if title == "Composition":
            lines += comp_lines
        elif title == "Collection process":
            lines.append(collection)
        else:
            lines.append(body)
        lines.append("")
    out_path = out_path or ROOT / "docs" / "DATASHEET_v1.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def stage_seed_paragraphs(numbers: dict[str, str] | None = None) -> str:
    """One drafted dissertation paragraph per pipeline stage; ⟨TBD⟩ slots fill from
    `numbers` (stage -> formatted number/claim) as runs produce them."""
    n = numbers or {}
    seeds = [
        ("Acquisition", f"Data were acquired through a journaled, rate-governed pipeline with "
         f"request-level provenance ({n.get('acquisition', '⟨TBD: chunk/row counts⟩')}); every "
         "artifact is SHA-256-manifested at the moment of receipt."),
        ("Raw vault", f"The raw layer is write-once and checksum-verified on every read "
         f"({n.get('raw', '⟨TBD: artifact count⟩')}), making fabrication or silent revision "
         "detectable by construction (R4)."),
        ("Security master", f"All joins key on a security master that resolves renames, share-class "
         f"splits and dead-RIC suffixes ({n.get('master', '⟨TBD: n securities⟩')}), eliminating "
         "ticker-reuse contamination."),
        ("Validation", f"Structural validation enforced schema, XNYS calendar alignment and "
         f"duplicate detection ({n.get('validation', '⟨TBD: violation counts⟩')})."),
        ("Corporate actions", f"Total-return internal consistency and unadjusted-split signatures "
         f"were screened ({n.get('corp_actions', '⟨TBD: flag counts⟩')}); flags never mutate values."),
        ("Outliers", f"Ince–Porter screens plus cross-sectional context classification preserved "
         f"genuine crisis tails while quarantining suspect prints ({n.get('outliers', '⟨TBD⟩')})."),
        ("Survivorship", f"Point-in-time membership with a spliced 2016 overlap validation "
         f"({n.get('membership', '⟨TBD: jaccard⟩')}) and logged Shumway corrections removes the "
         "0.9–1.4%/yr survivorship inflation documented by Elton–Gruber–Blake."),
        ("Missing data", f"Every missing cell is classified and counted "
         f"({n.get('missing', '⟨TBD: taxonomy counts⟩')}); returns are never interpolated."),
        ("Reconciliation", f"Two-vendor reconciliation across the universe "
         f"({n.get('reconciliation', '⟨TBD: corr/discrepancy stats⟩')}) clusters discrepancies on "
         "ex-dividend and split days, deciding per-field vendor authority."),
        ("Gold construction", f"Model-ready panels are built exclusively through availability-"
         f"lagged as-of joins with embargoed splits materialized to explicit session lists "
         f"({n.get('gold', '⟨TBD: split session counts⟩')}), and leakage is asserted by tests."),
        ("EDA", f"Profiling confirmed fat tails, volatility clustering and correlation-regime "
         f"dynamics ({n.get('eda', '⟨TBD: kurtosis/Hill numbers⟩')}), motivating the CVaR-penalised "
         "fitness, the 60-day lookback and the 3-state filtered HMM."),
        ("Quality", f"A per-series scoreboard ({n.get('quality', '⟨TBD: median score⟩')}) and a "
         "raw→gold lineage map close the loop from API call to model input."),
    ]
    out = ["# Data-chapter seed paragraphs (one per pipeline stage)", ""]
    for title, para in seeds:
        out += [f"**{title}.** {para}", ""]
    return "\n".join(out)


def write_quality_reports(coverage_df: pd.DataFrame, score_df: pd.DataFrame,
                          gold_relpaths: list[str], numbers: dict | None = None) -> dict[str, Path]:
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    out: dict[str, Path] = {}
    out["scoreboard"] = rep / "data_quality_scoreboard.md"
    out["scoreboard"].write_text(
        "# Data quality scoreboard\n\n" + score_df.to_markdown(index=False)
        + "\n\n## Coverage matrix\n\n" + coverage_df.to_markdown(index=False) + "\n")
    ev = ROOT / get("data.platform.evidence_dir")
    ev.mkdir(parents=True, exist_ok=True)
    out["lineage"] = ev / "lineage_map.md"
    out["lineage"].write_text(lineage_map_markdown(gold_relpaths))
    out["datasheet"] = generate_datasheet()
    out["seeds"] = rep / "data_chapter_seeds.md"
    out["seeds"].write_text(stage_seed_paragraphs(numbers))
    out["scoreboard_json"] = ev / "quality_scoreboard.json"
    out["scoreboard_json"].write_text(json.dumps(score_df.to_dict("records"), indent=2, default=str))
    return out
