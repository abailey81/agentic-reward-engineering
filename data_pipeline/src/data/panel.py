"""Model-ready (gold) construction — stage 10. Leakage-proof BY CONSTRUCTION, then
leakage-ASSERTED by tests (tests/test_data_panel.py): perturbing inputs at t >= m
must leave every gold row < m unchanged.

Framework rule: every derived feature declares its information availability lag, and
`as_of_join` (a shift-by-lag join on the session index) is the ONLY join allowed
into model-ready tables — plain index-joins of derived features are forbidden by
convention and absent from this module. The cash-row features themselves come from
`src.features.build_cash_features`, which is shift(1)-lagged and leakage-tested.

Splits are materialized to EXPLICIT session-date lists matching PREREGISTRATION §6
byte-for-byte intent: development train/validation, walk-forward 2018->2025
(train 5y / test 1y / step 1y), a 21-TRADING-DAY embargo at every boundary (López de
Prado 2018), and CPCV S=16 contiguous purged blocks over the evaluation span.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get
from ..features import build_cash_features
from .membership import top30_at
from .vault import FrozenArtifact, freeze, record_lineage


# ----------------------------------------------------------------- as-of framework
@dataclass(frozen=True)
class AsOfFeature:
    """A feature plus its declared information timestamp: values dated t become
    AVAILABLE at t + available_lag_sessions (>=1 for anything computed from day-t
    data — the R3 'strictly before the decision' law)."""

    name: str
    frame: pd.DataFrame | pd.Series
    available_lag_sessions: int = 1

    def __post_init__(self) -> None:
        if self.available_lag_sessions < 0:
            raise ValueError("availability lag cannot be negative")


def as_of_join(base_index: pd.DatetimeIndex, feature: AsOfFeature) -> pd.DataFrame:
    """Join `feature` onto `base_index` so the row at session t carries the LATEST
    feature value whose availability date (value date + lag sessions) is <= t."""
    f = feature.frame if isinstance(feature.frame, pd.DataFrame) else feature.frame.to_frame(feature.name)
    f = f.sort_index()
    shifted = f.copy()
    shifted.index = pd.DatetimeIndex(shifted.index)
    # availability calendar = base_index sessions: value dated session i becomes
    # available at session i + lag -> implemented as a positional shift after
    # aligning value dates onto the session grid (asof for off-grid dates).
    aligned = shifted.reindex(base_index, method="ffill")
    return aligned.shift(feature.available_lag_sessions)


# ------------------------------------------------------------------------- splits
def _embargo_trim(sessions: pd.DatetimeIndex, embargo: int) -> pd.DatetimeIndex:
    return sessions[embargo:] if embargo > 0 else sessions


def materialize_splits(sessions: pd.DatetimeIndex) -> dict:
    """PREREG §6 splits as explicit ISO session lists (config inference.splits).

    Embargo convention: the first `embargo_trading_days` sessions of every
    out-of-sample segment (dev validation; each walk-forward test year) are dropped,
    creating a >=21-trading-day information gap after the adjoining train window.
    CPCV: S contiguous equal blocks over the evaluation span; purge width recorded
    for use-time application (purging is relative to each IS/OOS assignment)."""
    cfg = get("inference.splits")
    embargo = int(cfg["embargo_trading_days"])
    sessions = pd.DatetimeIndex(sessions)

    def _between(a: str, b: str) -> pd.DatetimeIndex:
        return sessions[(sessions >= pd.Timestamp(a)) & (sessions <= pd.Timestamp(b))]

    def _span(ix: pd.DatetimeIndex) -> list[str] | None:
        """[first, last] as ISO dates; None when the supplied sessions don't cover
        the segment (sub-span builds — the full-window build populates everything)."""
        return [d.date().isoformat() for d in ix[[0, -1]]] if len(ix) else None

    dev = cfg["development"]
    dev_train = _between(*dev["train"])
    dev_val = _embargo_trim(_between(*dev["validation"]), embargo)

    ev = cfg["evaluation"]
    span_start, span_end = pd.Timestamp(ev["span"][0]), pd.Timestamp(ev["span"][1])
    folds = []
    year = span_start.year
    while year <= span_end.year:
        train = _between(f"{year - ev['train_years']}-01-01", f"{year - 1}-12-31")
        test = _embargo_trim(_between(f"{year}-01-01", f"{year}-12-31"), embargo)
        if len(test) > 0 and len(train) > 0:
            folds.append({"test_year": year,
                          "train": _span(train),
                          "test": _span(test),
                          "n_train_sessions": int(len(train)),
                          "n_test_sessions": int(len(test))})
        year += int(ev["step_years"])

    eval_sessions = _between(ev["span"][0], ev["span"][1])
    n_blocks = int(cfg["cpcv"]["n_blocks"])
    blocks = np.array_split(np.arange(len(eval_sessions)), n_blocks)
    cpcv_blocks = [{"block": i,
                    "start": eval_sessions[b[0]].date().isoformat(),
                    "end": eval_sessions[b[-1]].date().isoformat(),
                    "n_sessions": int(len(b))} for i, b in enumerate(blocks) if len(b)]

    return {
        "embargo_trading_days": embargo,
        "development": {
            "train": _span(dev_train),
            "validation_post_embargo": _span(dev_val),
            "n_train_sessions": int(len(dev_train)),
            "n_validation_sessions": int(len(dev_val)),
        },
        "walk_forward": folds,
        "cpcv": {"n_blocks": n_blocks, "purge": bool(cfg["cpcv"]["purge"]),
                 "purge_trading_days": embargo, "blocks": cpcv_blocks},
    }


# --------------------------------------------------------------------- gold build
def build_gold(
    clean_returns: pd.DataFrame,
    vix: pd.Series,
    sessions: pd.DatetimeIndex,
    clean_mcap: pd.DataFrame | None = None,
    pit_membership: pd.DataFrame | None = None,
    run_id: str = "gold",
    input_artifacts: list[FrozenArtifact | dict] | None = None,
    suffix: str = "",
) -> dict:
    """Construct and freeze the gold layer from clean inputs.

    Artifacts (parquet, manifested, lineage-recorded):
        returns_panel       (T x N) daily simple total returns, session-aligned
        cash_features       [vol20, vol20/vol60, vix] — shift(1)-lagged (ADR-007)
        market_proxy        equal-weight universe return (HMM input; fit scope is
                            enforced downstream by regimes.py fit-on-train)
        splits              PREREG §6 materialization (json inside a 1-col frame)
        top30_selection     per walk-forward window start (when mcap+membership exist)
    """
    aligned = clean_returns.reindex(sessions)
    vix_aligned = vix.reindex(sessions)
    if vix_aligned.isna().all():
        raise ValueError("VIX series empty after session alignment — gold needs the macro pull")
    # VIX gaps on sessions (vendor missing): carry last published level for the
    # FEATURE only (a published index level remains the latest information until
    # replaced — this is information carry, not return fabrication; returns are
    # never filled, R4).
    vix_ff = vix_aligned.ffill()

    feats = build_cash_features(aligned.to_numpy(dtype=float), vix_ff.to_numpy(dtype=float))
    cash = pd.DataFrame(feats, index=sessions, columns=["vol20", "vol20_over_vol60", "vix"])
    market = aligned.mean(axis=1).rename("market_ew")

    arts: dict[str, FrozenArtifact] = {}
    inputs = input_artifacts or []
    arts["returns_panel"] = freeze(aligned, "gold", f"returns_panel{suffix}.parquet",
                                   {"vendor": "derived", "stage": "panel.build_gold"},
                                   fmt="parquet", run_id=run_id)
    arts["cash_features"] = freeze(cash, "gold", f"cash_features{suffix}.parquet",
                                   {"vendor": "derived", "stage": "panel.build_gold",
                                    "lag_convention": "shift(1) — ADR-007"},
                                   fmt="parquet", run_id=run_id)
    arts["market_proxy"] = freeze(market.to_frame(), "gold", f"market_proxy{suffix}.parquet",
                                  {"vendor": "derived", "stage": "panel.build_gold"},
                                  fmt="parquet", run_id=run_id)
    splits = materialize_splits(sessions)
    splits_frame = pd.DataFrame({"splits_json": [json.dumps(splits, sort_keys=True)]})
    arts["splits"] = freeze(splits_frame, "gold", f"splits{suffix}.parquet",
                            {"vendor": "derived", "stage": "panel.materialize_splits",
                             "prereg": "§6"}, fmt="parquet", run_id=run_id)

    if clean_mcap is not None and pit_membership is not None:
        rows = []
        window_starts = []
        if splits["development"]["train"] is not None:      # the DEV window — where the
            window_starts.append(("development", splits["development"]["train"][0]))
        window_starts += [("walk_forward", f["train"][0]) for f in splits["walk_forward"]]
        for phase, start_iso in window_starts:
            start = pd.Timestamp(start_iso)
            try:
                rows.append({"phase": phase, "window_start": start.date().isoformat(),
                             "selection": top30_at(start, clean_mcap, pit_membership)})
            except ValueError as e:
                rows.append({"phase": phase, "window_start": start.date().isoformat(),
                             "selection": None, "blocked": str(e)})
        arts["top30_selection"] = freeze(pd.DataFrame(rows), "gold",
                                         f"top30_selection{suffix}.parquet",
                                         {"vendor": "derived", "stage": "panel.top30",
                                          "rule": "PREREG §5 top30_marketcap_at_window_start"},
                                         fmt="parquet", run_id=run_id)
    for art in arts.values():
        record_lineage(art, inputs, stage="panel.build_gold",
                       params={"suffix": suffix, "n_sessions": int(len(sessions))})
    return {"artifacts": arts, "splits": splits}
