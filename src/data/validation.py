"""Data-contract validation for a frozen :class:`~src.data.panel.Panel`.

WHY THIS EXISTS. ``Panel.__post_init__`` already enforces the *structural* contract (2-D shape,
matching ``(T, N)`` dimensions, finite returns). This module adds the *semantic / research*
invariants the constructor cannot express — the ones whose violation would be a silent data error
or, worse, a **look-ahead leak** (the cardinal sin in a finance backtest). It consolidates the checks
that were previously implicit/scattered (checksum, NaN policy, the leakage-free VIX fill) into one
auditable contract that can be asserted on load and unit-tested directly.

It is a *report-only* validator: it reads an already-built panel and never mutates it, so it is
deterministic and adds no dependency. Scope discipline: this is NOT supervised-learning data
validation (there are no labels/classes here) — it is the leakage + integrity contract for an RL
returns panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .panel import Panel

if TYPE_CHECKING:  # pragma: no cover - typing only (pandas is a heavy import kept lazy)
    import pandas as pd

__all__ = ["PanelContractError", "validate_panel", "PanelOverlapDiff", "panel_overlap_diff"]

#: A long simple return can lose AT MOST 100%, i.e. ``r >= -1.0``; exactly ``-1.0`` is a legitimate total
#: loss (delisting / bankruptcy — and this project's delisting band includes -100%). Only ``r < -1.0``
#: (losing MORE than 100%) is impossible and signals a mis-scaled action/units error. Returns whose
#: magnitude exceeds ``_MAX_ABS_RETURN`` are implausible at daily frequency (split/units artifact).
_IMPOSSIBLE_RETURN = -1.0  # checked with strict "<" so -1.0 (total loss) is allowed
_MAX_ABS_RETURN = 10.0  # +1000% in one day — a screen flag, not a hard cap


class PanelContractError(ValueError):
    """Raised when a :class:`Panel` violates the data contract (``validate_panel(strict=True)``)."""


def validate_panel(
    panel: Panel,
    *,
    strict: bool = True,
    max_abs_return: float = _MAX_ABS_RETURN,
) -> list[str]:
    """Assert the semantic + leakage invariants of a built ``Panel``.

    Checks (each a distinct, named violation):

    - **Dates strictly increasing** — no duplicate, unsorted, or non-monotone sessions. This is the
      load-bearing **leakage** invariant: an out-of-order or duplicated date silently admits future
      information into a window.
    - **Returns sane** — every simple return is ``> -1.0`` (a long position cannot lose more than
      100%) and ``|r| <= max_abs_return`` (implausible daily moves flag a split/units artifact).
    - **VIX non-negative + finite** — a volatility index cannot be negative.
    - **asset_ids unique + integer** — the anonymisation contract (ids carry no semantic meaning and
      must be distinct; never tickers/strings).
    - **market_caps non-negative** (when present).
    - **No all-zero return column** — a dead / gap-filled asset surfaced as a constant-0 series.

    Structural invariants (shape, finiteness) are already enforced by ``Panel.__post_init__`` and are
    re-checked here defensively.

    Parameters
    ----------
    panel:
        The built panel to validate (never mutated).
    strict:
        If ``True`` (default) raise :class:`PanelContractError` on the first non-empty violation list;
        if ``False`` return the list of violation strings (empty == clean) for the caller to handle.
    max_abs_return:
        The implausible-daily-move screen threshold (default ``10.0`` == +1000%).

    Returns
    -------
    list[str]
        Violation messages (empty if the panel satisfies the contract). Only returned when
        ``strict=False``; otherwise a non-empty list raises.
    """
    issues: list[str] = []

    # --- structural (defensive; Panel.__post_init__ also enforces these) ---
    if panel.returns.ndim != 2:
        issues.append(f"returns must be 2-D (T, N); got {panel.returns.shape}")
        # cannot continue meaningfully without a 2-D matrix
        return _finish(issues, strict)
    if not np.isfinite(panel.returns).all():
        issues.append("returns contains non-finite values (post-NaN-policy finiteness invariant)")

    # --- leakage: dates strictly increasing (no duplicate / unsorted / future) ---
    dates = np.asarray(panel.dates)
    if dates.size >= 2:
        # int64 view works for datetime64[*] (epoch units) and integer date encodings alike.
        order = dates.astype("int64")
        d = np.diff(order)
        if not (d > 0).all():
            n_bad = int((d <= 0).sum())
            issues.append(
                f"dates not strictly increasing ({n_bad} non-positive gap(s)) -- duplicate/unsorted "
                "sessions admit look-ahead leakage"
            )

    # --- returns sanity ---
    r = panel.returns
    if (r < _IMPOSSIBLE_RETURN).any():
        n_bad = int((r < _IMPOSSIBLE_RETURN).sum())
        issues.append(
            f"{n_bad} return(s) < {_IMPOSSIBLE_RETURN} -- impossible for a long simple return "
            "(can lose at most 100%; exactly -1.0 total loss IS allowed for delisting)"
        )
    if (np.abs(r) > max_abs_return).any():
        n_bad = int((np.abs(r) > max_abs_return).sum())
        issues.append(
            f"{n_bad} return(s) with |r| > {max_abs_return} -- implausible daily move (split/units artifact?)"
        )
    # all-zero column = dead / gap-filled asset
    n_zero_cols = int((np.abs(r).sum(axis=0) == 0.0).sum())
    if n_zero_cols:
        issues.append(f"{n_zero_cols} all-zero return column(s) -- dead or gap-filled asset")

    # --- vix ---
    vix = np.asarray(panel.vix, dtype=float)
    if not np.isfinite(vix).all():
        issues.append("vix contains non-finite values")
    if (vix < 0.0).any():
        issues.append("vix < 0 -- a volatility index cannot be negative")

    # --- asset_ids: anonymisation contract ---
    aid = np.asarray(panel.asset_ids)
    if not np.issubdtype(aid.dtype, np.integer):
        issues.append(f"asset_ids must be integer (anonymisation contract); got dtype {aid.dtype}")
    if np.unique(aid).size != aid.size:
        issues.append("asset_ids not unique")

    # --- market caps ---
    if panel.market_caps is not None:
        mc = np.asarray(panel.market_caps, dtype=float)
        # Batch-4 audit #8 (2026-07-03): the two conditions are independent — the old combined
        # `isfinite(mc).all() and (mc < 0).any()` silently SKIPPED the negativity check whenever any
        # cap was NaN/inf (and never flagged the non-finite caps themselves).
        if not np.isfinite(mc).all():
            issues.append("market_caps contain non-finite values")
        finite = mc[np.isfinite(mc)]
        if (finite < 0.0).any():
            issues.append("market_caps < 0 -- a market capitalisation cannot be negative")

    return _finish(issues, strict)


def _finish(issues: list[str], strict: bool) -> list[str]:
    if strict and issues:
        raise PanelContractError(
            "Panel data-contract violated:\n  - " + "\n  - ".join(issues)
        )
    return issues


# --------------------------------------------------------------------------- #
# Rebuild validator: byte-diff a candidate returns panel vs the frozen one     #
# --------------------------------------------------------------------------- #
@dataclass
class PanelOverlapDiff:
    """Cell-level diff of a CANDIDATE returns panel against a FROZEN reference over their overlap.

    The Phase-1 data rebuild produces a new-suffix panel; before it can replace the frozen ``univ3``
    headline panel we must know EXACTLY which cells it changes over the shared 2005-2025 (date × RIC)
    overlap — a rebuild that silently perturbs historical returns would move the headline tail. This
    is a *report-only* comparison; it never mutates either panel.

    Attributes
    ----------
    n_overlap_rows, n_overlap_cols:
        Size of the shared (date × RIC) overlap actually compared.
    n_changed_cells:
        Count of overlap cells whose values differ (NaN-aware: NaN==NaN is treated as EQUAL, so
        aligned missing data does not register as a change; NaN-vs-value DOES).
    max_abs_delta:
        Largest absolute value change over the overlap (``0.0`` if identical / no numeric overlap).
    changed_examples:
        Up to ``max_examples`` ``(date, ric, ref_value, cand_value)`` tuples for the first changed
        cells (for a human-readable report / test assertion).
    ref_only_cols, cand_only_cols:
        RICs present in only one panel (schema drift — reported, not counted as changed cells).
    ref_only_rows, cand_only_rows:
        Session counts present in only one panel's date index (calendar drift).
    """

    n_overlap_rows: int
    n_overlap_cols: int
    n_changed_cells: int
    max_abs_delta: float
    changed_examples: list[tuple[str, str, float, float]] = field(default_factory=list)
    ref_only_cols: list[str] = field(default_factory=list)
    cand_only_cols: list[str] = field(default_factory=list)
    ref_only_rows: int = 0
    cand_only_rows: int = 0

    @property
    def identical_over_overlap(self) -> bool:
        """True iff no overlap cell changed (schema/calendar drift is reported separately)."""
        return self.n_changed_cells == 0

    def summary(self) -> str:
        """One-block human-readable summary (for the ``verify_gold`` CLI + logs)."""
        head = (
            f"overlap {self.n_overlap_rows} rows x {self.n_overlap_cols} cols; "
            f"changed cells: {self.n_changed_cells}; max |delta|: {self.max_abs_delta:.3e}"
        )
        schema = (
            f"ref-only cols: {len(self.ref_only_cols)} {self.ref_only_cols[:5]}; "
            f"cand-only cols: {len(self.cand_only_cols)} {self.cand_only_cols[:5]}; "
            f"ref-only rows: {self.ref_only_rows}; cand-only rows: {self.cand_only_rows}"
        )
        lines = [head, schema]
        if self.changed_examples:
            lines.append("first changed cells (date, ric, ref -> cand):")
            for dt, ric, a, b in self.changed_examples:
                lines.append(f"    {dt}  {ric}  {a!r} -> {b!r}")
        return "\n  ".join(lines)


def _read_returns_frame(source: "str | Path | pd.DataFrame") -> "pd.DataFrame":
    """Load a returns panel (date-indexed, RIC-columned) from a parquet path or accept a DataFrame."""
    import pandas as pd

    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_parquet(Path(source))


def panel_overlap_diff(
    candidate: "str | Path | pd.DataFrame",
    reference: "str | Path | pd.DataFrame",
    *,
    max_examples: int = 20,
) -> PanelOverlapDiff:
    """Byte-diff a CANDIDATE returns panel against a FROZEN ``reference`` over their (date × RIC) overlap.

    Both inputs are date-indexed, RIC-columned returns frames (the ``returns_panel_<suffix>.parquet``
    shape) OR a parquet path to one. The comparison aligns on the INTERSECTION of dates and columns and
    reports every differing cell (NaN-aware: two aligned NaNs are EQUAL; NaN-vs-number is a change).
    Schema drift (columns/rows present in only one panel) is reported separately, not counted as a
    changed cell — a rebuild legitimately adds/retires names, but must NOT alter shared historical cells.

    This is the lean, working validator the Phase-1 rebuild runs to prove a new-suffix panel does not
    perturb the frozen 2005-2025 history (``scripts/verify_gold.py`` wires it against frozen ``univ3``).
    """
    ref = _read_returns_frame(reference)
    cand = _read_returns_frame(candidate)

    ref_cols = [str(c) for c in ref.columns]
    cand_cols = [str(c) for c in cand.columns]
    ref_col_set, cand_col_set = set(ref_cols), set(cand_cols)
    shared_cols = [c for c in ref_cols if c in cand_col_set]  # ref order, stable
    ref_only_cols = [c for c in ref_cols if c not in cand_col_set]
    cand_only_cols = [c for c in cand_cols if c not in ref_col_set]

    ref_idx, cand_idx = ref.index, cand.index
    shared_idx = ref_idx.intersection(cand_idx)
    ref_only_rows = int(len(ref_idx) - len(shared_idx))
    cand_only_rows = int(len(cand_idx) - len(shared_idx))

    if len(shared_idx) == 0 or not shared_cols:
        return PanelOverlapDiff(
            n_overlap_rows=int(len(shared_idx)), n_overlap_cols=len(shared_cols),
            n_changed_cells=0, max_abs_delta=0.0,
            ref_only_cols=ref_only_cols, cand_only_cols=cand_only_cols,
            ref_only_rows=ref_only_rows, cand_only_rows=cand_only_rows,
        )

    a = ref.loc[shared_idx, shared_cols]
    b = cand.loc[shared_idx, shared_cols]
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)

    # NaN-aware inequality: differ where values are unequal AND not both-NaN.
    both_nan = np.isnan(av) & np.isnan(bv)
    differ = (av != bv) & ~both_nan
    n_changed = int(differ.sum())

    delta = np.abs(av - bv)
    delta[~np.isfinite(delta)] = 0.0  # NaN-vs-number deltas are non-finite; counted via `differ`, not size
    max_abs_delta = float(delta[differ].max()) if n_changed and np.isfinite(delta[differ]).any() else 0.0

    examples: list[tuple[str, str, float, float]] = []
    if n_changed:
        rows, cols = np.nonzero(differ)
        for r_i, c_i in zip(rows[:max_examples], cols[:max_examples]):
            examples.append(
                (str(shared_idx[r_i]), shared_cols[c_i], float(av[r_i, c_i]), float(bv[r_i, c_i]))
            )

    return PanelOverlapDiff(
        n_overlap_rows=int(len(shared_idx)), n_overlap_cols=len(shared_cols),
        n_changed_cells=n_changed, max_abs_delta=max_abs_delta,
        changed_examples=examples,
        ref_only_cols=ref_only_cols, cand_only_cols=cand_only_cols,
        ref_only_rows=ref_only_rows, cand_only_rows=cand_only_rows,
    )
