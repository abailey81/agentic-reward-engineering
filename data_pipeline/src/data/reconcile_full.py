"""Full-universe two-vendor reconciliation (stage 9) — extends the pilot
(src/reconcile.py) to the membership union, with discrepancy CLUSTERING, per-field
vendor-authority merging, and a reason-coded quarantine queue.

Two vendors disagreeing is INFORMATION (Ince & Porter 2006: naive single-vendor use
inflated equal-weighted returns ~72% vs CRSP) — the clustering step mines WHERE the
disagreement lives (ex-dividend days? split days? index-exit windows?) because that
geography decides which vendor anchors which field (config
data.platform.vendor_authority) and what tolerance the clean layer enforces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get


@dataclass
class ReconRow:
    name: str
    n_overlap: int
    corr: float
    max_abs_gap: float
    days_over_tol: int
    cluster_ex_div: int = 0
    cluster_split: int = 0
    cluster_exit_window: int = 0
    cluster_unexplained: int = 0
    notes: str = ""


@dataclass
class ReconReport:
    tolerance: float
    rows: list[ReconRow] = field(default_factory=list)
    quarantine: pd.DataFrame | None = None

    def to_markdown(self) -> str:
        lines = [
            "| Name | overlap | corr(daily ret) | max \\|Δ\\| | days>tol | ex-div | split | exit-win | unexplained | notes |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.name} | {r.n_overlap} | {r.corr:.6f} | {r.max_abs_gap:.2e} "
                f"| {r.days_over_tol} | {r.cluster_ex_div} | {r.cluster_split} "
                f"| {r.cluster_exit_window} | {r.cluster_unexplained} | {r.notes} |"
            )
        return "\n".join(lines)


def reconcile_returns(
    primary: pd.DataFrame,
    shadow: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    exit_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    tol: float | None = None,
) -> ReconReport:
    """Per-name reconciliation of two daily-return panels + discrepancy clustering.

    Clusters (priority order, one cluster per discrepancy day): ex-dividend day
    (vendor adjustment mismatches concentrate there), split day, index-exit window,
    else unexplained -> quarantine queue with reason codes.
    """
    tol = float(tol if tol is not None else get("data.vendors.reconciliation_tolerance.daily_return_abs"))
    report = ReconReport(tolerance=tol)
    qrows = []
    for name in sorted(set(primary.columns) & set(shadow.columns)):
        pair = pd.concat([primary[name], shadow[name]], axis=1, keys=["a", "b"]).dropna()
        if pair.empty:
            report.rows.append(ReconRow(name, 0, np.nan, np.nan, 0, notes="no overlap"))
            continue
        gap = (pair["a"] - pair["b"]).abs()
        over = gap[gap > tol]
        row = ReconRow(
            name=name, n_overlap=int(len(pair)),
            corr=float(pair["a"].corr(pair["b"])) if len(pair) > 2 else np.nan,
            max_abs_gap=float(gap.max()), days_over_tol=int(len(over)),
        )
        div_dates = set()
        if dividends is not None and name in dividends.columns:
            d = dividends[name]
            div_dates = set(d[d.fillna(0) != 0].index)
        split_dates = set()
        if splits is not None and name in splits.columns:
            s = splits[name]
            split_dates = set(s[(s.fillna(0) != 0) & (s.fillna(1) != 1)].index)
        exit_win = exit_windows.get(name) if exit_windows else None
        for date, g in over.items():
            if date in div_dates:
                row.cluster_ex_div += 1
            elif date in split_dates:
                row.cluster_split += 1
            elif exit_win is not None and exit_win[0] <= date <= exit_win[1]:
                row.cluster_exit_window += 1
            else:
                row.cluster_unexplained += 1
                qrows.append({"name": name, "date": date, "abs_gap": float(g),
                              "primary": float(pair["a"].loc[date]),
                              "shadow": float(pair["b"].loc[date]),
                              "reason": "unexplained_vendor_gap"})
        report.rows.append(row)
    report.quarantine = pd.DataFrame(qrows)
    return report


def authority_merge(frames_by_vendor: dict[str, pd.DataFrame], field_name: str) -> tuple[pd.DataFrame, dict]:
    """Per-field clean-layer merge: take the configured authority vendor's series;
    fall back (column-wise) to the fallback vendor ONLY where the authority has no
    column at all — never cell-wise blending, which would manufacture a synthetic
    series no vendor ever published (R4). Returns (merged, decision_record)."""
    auth_cfg = get("data.platform.vendor_authority")
    authority = auth_cfg.get(field_name, auth_cfg["fallback"])
    fallback = auth_cfg["fallback"]
    decision = {"field": field_name, "authority": authority, "fallback": fallback,
                "authority_columns": [], "fallback_columns": []}
    primary = frames_by_vendor.get(authority)
    backup = frames_by_vendor.get(fallback)
    if primary is None and backup is None:
        raise ValueError(f"no vendor frame supplied for field '{field_name}'")
    if primary is None:
        decision["fallback_columns"] = list(backup.columns)
        return backup.copy(), decision
    merged = primary.copy()
    decision["authority_columns"] = list(primary.columns)
    if backup is not None:
        extra = [c for c in backup.columns if c not in merged.columns]
        if extra:
            merged = pd.concat([merged, backup[extra]], axis=1)
            decision["fallback_columns"] = extra
    return merged.sort_index(), decision
