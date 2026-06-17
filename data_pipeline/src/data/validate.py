"""Structural validation (stage 4) + missing-data engine (stage 8).

Schema enforcement is a hand-rolled minimal core ("pandera or equivalent" — zero
extra dependency, identical guarantees for our table shapes): declared dtypes,
nullability, bounds, monotone-unique date index.

Calendar: XNYS sessions via `exchange_calendars`. Gap CLASSIFICATION taxonomy:
    holiday          non-session date         -> calendar-drop (counted)
    pre_ipo          before first valid obs   -> masked (counted)
    post_delisting   after last valid obs     -> terminal (counted; the panel stage
                                                liquidates leavers, never drops them)
    interior         in-session, mid-history  -> halt_or_vendor_missing: FLAGGED,
                                                never interpolated (R4 — zero
                                                fabrication of market data)
Every decision is COUNTED in MissingReport and reported; nothing is silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import get


# ---------------------------------------------------------------------- schema core
@dataclass(frozen=True)
class ColumnSpec:
    dtype: str                      # 'float' | 'int' | 'str' | 'date'
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: dict[str, ColumnSpec]
    datetime_index: bool = True
    monotone_unique_index: bool = True


@dataclass
class ValidationReport:
    table: str
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def validate_frame(df: pd.DataFrame, schema: TableSchema) -> ValidationReport:
    rep = ValidationReport(schema.name)
    if schema.datetime_index:
        if not isinstance(df.index, pd.DatetimeIndex):
            rep.violations.append("index is not DatetimeIndex")
        else:
            if df.index.tz is not None:
                rep.violations.append("index must be tz-naive (normalize_timezone first)")
            if schema.monotone_unique_index:
                if not df.index.is_monotonic_increasing:
                    rep.violations.append("index not monotone increasing")
                if not df.index.is_unique:
                    rep.violations.append("index not unique")
    for col, spec in schema.columns.items():
        if col not in df.columns:
            rep.violations.append(f"missing column: {col}")
            continue
        s = df[col]
        if spec.dtype == "float" and not pd.api.types.is_float_dtype(s):
            rep.violations.append(f"{col}: expected float dtype, got {s.dtype}")
        if spec.dtype == "int" and not pd.api.types.is_integer_dtype(s):
            rep.violations.append(f"{col}: expected int dtype, got {s.dtype}")
        if not spec.nullable and s.isna().any():
            rep.violations.append(f"{col}: {int(s.isna().sum())} nulls in non-nullable column")
        if pd.api.types.is_numeric_dtype(s):           # bounds only meaningful on numerics
            if spec.minimum is not None and (s.dropna() < spec.minimum).any():
                rep.violations.append(f"{col}: values below {spec.minimum}")
            if spec.maximum is not None and (s.dropna() > spec.maximum).any():
                rep.violations.append(f"{col}: values above {spec.maximum}")
    return rep


def coerce_frame(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Explicit dtype coercion (errors surface as NaN + a validation violation —
    coercion never invents values)."""
    out = df.copy()
    if schema.datetime_index and not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    for col, spec in schema.columns.items():
        if col not in out.columns:
            continue
        if spec.dtype == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        elif spec.dtype == "int":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif spec.dtype == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Vendor timestamps -> tz-naive dates (daily bars; one calendar, one convention)."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


# ------------------------------------------------------------------------- calendar
def trading_sessions(start: str, end: str, calendar: str | None = None) -> pd.DatetimeIndex:
    """XNYS sessions in [start, end], tz-naive. The calendar is constructed from the
    configured `calendar_start` bound because exchange_calendars defaults to a ~20y
    lookback that would clip the 2005 window start."""
    import exchange_calendars as xcals
    cal = xcals.get_calendar(calendar or get("data.platform.calendar"),
                             start=get("data.platform.calendar_start"))
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return sessions.tz_localize(None) if sessions.tz is not None else sessions


def align_to_calendar(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Reindex to exchange sessions. Off-calendar rows are REPORTED and set aside
    (almost always vendor artefacts on holidays), in-calendar rows pass through."""
    df = normalize_timezone(df)
    off = df.index.difference(sessions)
    aligned = df.reindex(sessions)
    report = {"sessions": int(len(sessions)),
              "off_calendar_rows_dropped": int(len(off)),
              "off_calendar_dates": [d.date().isoformat() for d in off[:20]]}
    return aligned, report


# ----------------------------------------------------------------------- duplicates
def find_duplicates(long_df: pd.DataFrame, keys: list[str]) -> dict:
    """Exact duplicates (identical rows) vs CONFLICTS (same key, different values —
    the dangerous kind; routed to the quarantine queue by the caller)."""
    exact = long_df[long_df.duplicated(keep=False)]
    by_key = long_df.drop_duplicates().groupby(keys, dropna=False).size()
    conflict_keys = by_key[by_key > 1].index.tolist()
    return {"exact_duplicate_rows": int(exact.shape[0]) // 2 if len(exact) else 0,
            "conflicting_keys": conflict_keys,
            "n_conflicts": len(conflict_keys)}


# ------------------------------------------------------------ gaps & missing engine
@dataclass
class MissingReport:
    table: str
    total_cells: int = 0
    observed: int = 0
    holiday_dropped: int = 0
    pre_ipo_masked: int = 0
    post_delisting_terminal: int = 0
    interior_flagged: int = 0           # halt / vendor-missing — NEVER interpolated

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def classify_missing(wide: pd.DataFrame, sessions: pd.DatetimeIndex,
                     raw_index: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, MissingReport]:
    """Classify every missing cell of a session-aligned wide panel.

    Returns (flag_frame, report): flag_frame holds {'', 'pre_ipo', 'post_delisting',
    'interior'} per cell; holiday count comes from the calendar alignment (dates the
    vendor supplied off-session, plus sessions the vendor never covers are NOT
    holidays — holidays are non-session dates in the RAW index)."""
    rep = MissingReport(table="panel")
    if raw_index is not None:
        raw_norm = pd.DatetimeIndex(raw_index).normalize()
        rep.holiday_dropped = int(len(raw_norm.difference(sessions)))
    flags = pd.DataFrame("", index=wide.index, columns=wide.columns, dtype=object)
    rep.total_cells = int(wide.size)
    rep.observed = int(wide.notna().sum().sum())
    for col in wide.columns:
        s = wide[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            flags[col] = "pre_ipo"
            rep.pre_ipo_masked += len(s)
            continue
        pre = s.index < first
        post = s.index > last
        interior = s.isna() & ~pre & ~post
        flags.loc[pre, col] = "pre_ipo"
        flags.loc[post, col] = "post_delisting"
        flags.loc[interior, col] = "interior"
        rep.pre_ipo_masked += int(pre.sum())
        rep.post_delisting_terminal += int(post.sum())
        rep.interior_flagged += int(interior.sum())
    return flags, rep


def missingness_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Boolean missing-cell matrix (heatmap input for the EDA stage)."""
    return wide.isna()
