"""Backtesting & performance analytics (block B11).

A comprehensive, citation-anchored performance/risk metric suite for the sealed-test evaluation and the
benchmark comparison, with regime-conditional breakdowns and a markdown tearsheet. Reuses the audited
``src.inference`` primitives (Sharpe, CVaR, deflated Sharpe, IQM) — never re-derives them.
"""
from __future__ import annotations

from src.backtest.metrics import (
    PERIODS_PER_YEAR,
    compute_metrics,
    drawdown_series,
    regime_conditional_metrics,
    tearsheet_markdown,
)

__all__ = [
    "PERIODS_PER_YEAR",
    "compute_metrics",
    "drawdown_series",
    "regime_conditional_metrics",
    "tearsheet_markdown",
]
