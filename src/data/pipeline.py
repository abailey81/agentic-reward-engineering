"""Point-in-time, survivorship-free data pipeline (FINAL_PLAN F.15).

Purpose
-------
Build the frozen return :class:`~src.data.panel.Panel` that every experiment trains
and evaluates on, in a way that is point-in-time (PIT) and survivorship-free
(audit B-2, C-3). The pipeline is expressed as a documented 13-stage sequence, one
method per stage, run end-to-end by :meth:`GoldPipeline.run`.

Because the licensed gold data is not redistributable (audit C-3), this class operates on a
**synthetic source** (:func:`src.data.synthetic.make_synthetic_panel`) of identical shape, so the entire
pipeline runs end-to-end and is testable without the gold data. Each stage is deterministic given the
config and seed, and its output is hashed with :func:`src.utils.provenance.sha256_obj`; the manifest of
per-stage SHA-256 checksums is returned alongside the frozen panel.

.. note::
   The **real** survivorship-free gold panel (Refinitiv/LSEG) already exists in ``data/gold/`` — built by
   the separate, self-contained acquisition stack in ``data_pipeline/`` (ADR-019…022), not by this class.
   To train on the real data, load it with :func:`src.data.loaders.load_gold_panel`. This ``GoldPipeline``
   is the SYNTHETIC, shape-identical pipeline used for tests/shipping. Do not confuse the two.

The 13 stages
-------------
1.  ``ingest_membership`` — index-membership history per date.
2.  ``ingest_prices_returns`` — daily total-return series (primary vendor).
3.  ``ingest_market_cap`` — market-capitalisation series for top-N selection.
4.  ``ingest_corporate_actions`` — corporate actions / delisting events.
5.  ``map_identifiers`` — reconcile vendor identifiers onto a stable internal id.
6.  ``reconstruct_membership_pit`` — rebuild PIT membership (no survivorship).
7.  ``correct_delisting_returns`` — apply delisting-return corrections.
8.  ``select_top_n_pit`` — select PIT top-N names by market cap.
9.  ``build_return_panel`` — assemble the aligned return panel.
10. ``engineer_features`` — feature notes (lookback, vol, LAGGED VIX, cash row).
11. ``reconcile_vendors`` — two-vendor reconciliation (synthetic: two seeds).
12. ``split_and_embargo`` — map config split dates to indices, purged by
    ``max(embargo_days, lookback_days)`` sessions (R18).
13. ``freeze_and_checksum`` — freeze the panel and emit the checksum manifest.

FINAL_PLAN refs
---------------
- F.15 (data pipeline).
- audit B-2 (feedback measured on train only), C-3 (ship checksums + pipeline +
  synthetic panel, not the data).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.panel import Panel
from src.data.synthetic import make_synthetic_panel
from src.utils.provenance import sha256_obj

__all__ = ["GoldPipeline", "DataPipeline"]


def _checksum(obj: Any) -> str:
    """SHA-256 of a stage output, normalising numpy arrays to nested lists."""

    def _norm(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.datetime64):
                return x.astype("datetime64[D]").astype(str).tolist()
            return np.round(x, 10).tolist() if np.issubdtype(x.dtype, np.floating) else x.tolist()
        if isinstance(x, dict):
            return {k: _norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_norm(v) for v in x]
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x

    return sha256_obj(_norm(obj))


class GoldPipeline:
    """The 13-stage point-in-time, survivorship-free panel builder.

    Parameters
    ----------
    cfg : Any
        Parsed ``config/data.yaml`` (period, splits, embargo, vendors, sources).
    n_assets : int
        Raw universe size *before* PIT top-N selection. Defaults to a small
        multiple of ``top_n`` so selection actually discriminates.
    top_n : int
        Number of assets kept by PIT top-N market-cap selection.
    n_days : int
        Number of trading days to synthesise.
    seed : int
        Master RNG seed (determinism / provenance).
    lookback_days : int | None
        Feature observation window (``config/environment.yaml::state.lookback_days``).
        Stage 12 purges adjacent splits by ``max(embargo_days, lookback_days)`` so no
        observation window ``returns[t-lookback:t]`` straddles a split boundary (R18);
        the bare ``embargo_days`` (21) < ``lookback`` (60) would under-purge. ``None``
        (the default) resolves it from ``environment.yaml`` so the pre-registered purge
        holds without an explicit caller; an unreadable config falls back to ``0``
        (legacy embargo-only).

    Notes
    -----
    Stages are pure functions of the config, seed and prior-stage outputs so the
    final artifact is reproducible and its per-stage checksums are deterministic.
    """

    def __init__(
        self,
        cfg: Any,
        n_assets: int = 30,
        top_n: int = 8,
        n_days: int = 600,
        seed: int = 12345,
        lookback_days: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.n_assets = int(n_assets)
        self.top_n = int(top_n)
        self.n_days = int(n_days)
        self.seed = int(seed)
        # R18: the stage-12 inter-split purge must cover the FEATURE LOOKBACK, not just the
        # embargo (the data cfg holds only ``embargo_days``=21; the lookback=60 lives in
        # environment.yaml). Resolve it from environment.yaml when unset — mirroring
        # orchestration/parallel.py — so the pre-registered purge holds by default. An
        # unreadable config degrades to 0 (legacy embargo-only) rather than hard-failing the
        # synthetic pipeline.
        self.lookback_days = int(lookback_days) if lookback_days is not None else self._default_lookback()
        self.manifest: dict[str, str] = {}
        # populated during run()
        self._raw: Panel | None = None
        self._secondary: Panel | None = None
        self._membership: np.ndarray | None = None
        self._caps: np.ndarray | None = None
        self._selected: np.ndarray | None = None
        self._delisting: dict[str, Any] | None = None
        self._panel: Panel | None = None

    @staticmethod
    def _default_lookback() -> int:
        """Resolve the feature lookback from ``environment.yaml`` (R18 purge input).

        Mirrors ``orchestration/parallel.py`` (``load_config('environment')['state']
        ['lookback_days']``). Returns ``0`` if the config cannot be read, so the synthetic
        pipeline still runs (degrading stage 12 to the legacy embargo-only gap) rather than
        raising on a config-less install.
        """
        try:
            from src.utils.config import load_config

            return int(load_config("environment")["state"]["lookback_days"])
        except Exception:  # noqa: BLE001 — config absent/malformed: degrade to embargo-only.
            return 0

    # -- Stage 1 --------------------------------------------------------------
    def ingest_membership(self) -> np.ndarray:
        """Stage 1: index-membership history (synthetic: all names always members)."""
        membership = np.ones((self.n_days, self.n_assets), dtype=bool)
        self._membership = membership
        self.manifest["01_ingest_membership"] = _checksum(membership)
        return membership

    # -- Stage 2 --------------------------------------------------------------
    def ingest_prices_returns(self) -> Panel:
        """Stage 2: daily total-return series from the primary (synthetic) vendor."""
        panel = make_synthetic_panel(
            n_assets=self.n_assets, n_days=self.n_days, seed=self.seed
        )
        self._raw = panel
        self.manifest["02_ingest_prices_returns"] = _checksum(panel.returns)
        return panel

    # -- Stage 3 --------------------------------------------------------------
    def ingest_market_cap(self) -> np.ndarray:
        """Stage 3: market-capitalisation series for PIT top-N selection."""
        assert self._raw is not None and self._raw.market_caps is not None
        caps = self._raw.market_caps
        self._caps = caps
        self.manifest["03_ingest_market_cap"] = _checksum(caps)
        return caps

    # -- Stage 4 --------------------------------------------------------------
    def ingest_corporate_actions(self) -> dict[str, Any]:
        """Stage 4: corporate actions / delisting events.

        The synthetic source has no real delistings, so the event table is empty.
        The structure is retained so the correction path (stage 7) is exercised.
        """
        delisting: dict[str, Any] = {"delisted_assets": [], "delisting_returns": {}}
        self._delisting = delisting
        self.manifest["04_ingest_corporate_actions"] = _checksum(delisting)
        return delisting

    # -- Stage 5 --------------------------------------------------------------
    def map_identifiers(self) -> np.ndarray:
        """Stage 5: map vendor identifiers onto a stable internal id."""
        assert self._raw is not None
        ids = self._raw.asset_ids
        self.manifest["05_map_identifiers"] = _checksum(ids)
        return ids

    # -- Stage 6 --------------------------------------------------------------
    def reconstruct_membership_pit(self) -> np.ndarray:
        """Stage 6: reconstruct PIT membership (synthetic: identity, no survivorship)."""
        assert self._membership is not None
        pit = self._membership.copy()
        self.manifest["06_reconstruct_membership_pit"] = _checksum(pit)
        return pit

    # -- Stage 7 --------------------------------------------------------------
    def correct_delisting_returns(self) -> Panel:
        """Stage 7: apply delisting-return corrections for dead names.

        Synthetic data has no delistings, so this is a no-op pass-through; the path
        is still exercised so a gold run would behave identically in structure.
        """
        assert self._raw is not None and self._delisting is not None
        returns = self._raw.returns.copy()
        for asset, (day, ret) in self._delisting["delisting_returns"].items():
            returns[day, int(asset)] = ret
        corrected = Panel(
            returns=returns,
            vix=self._raw.vix,
            dates=self._raw.dates,
            asset_ids=self._raw.asset_ids,
            market_caps=self._raw.market_caps,
        )
        self._raw = corrected
        self.manifest["07_correct_delisting_returns"] = _checksum(corrected.returns)
        return corrected

    # -- Stage 8 --------------------------------------------------------------
    def select_top_n_pit(self) -> np.ndarray:
        """Stage 8: select PIT top-N names by market cap.

        Selection uses the market cap on the *first* trading day (point-in-time at
        universe construction). Returns the sorted asset-column indices kept.
        """
        assert self._caps is not None
        first_day_caps = self._caps[0]
        order = np.argsort(first_day_caps)[::-1]  # descending
        selected = np.sort(order[: self.top_n])
        self._selected = selected
        self.manifest["08_select_top_n_pit"] = _checksum(selected)
        return selected

    # -- Stage 9 --------------------------------------------------------------
    def build_return_panel(self) -> Panel:
        """Stage 9: assemble the aligned return panel over the PIT universe."""
        assert self._raw is not None and self._selected is not None
        sel = self._selected
        panel = Panel(
            returns=self._raw.returns[:, sel],
            vix=self._raw.vix,
            dates=self._raw.dates,
            asset_ids=np.arange(sel.size, dtype=np.int64),
            market_caps=None if self._raw.market_caps is None else self._raw.market_caps[:, sel],
        )
        self._panel = panel
        self.manifest["09_build_return_panel"] = _checksum(panel.returns)
        return panel

    # -- Stage 10 -------------------------------------------------------------
    def engineer_features(self) -> dict[str, Any]:
        """Stage 10: feature notes (lookback, realized vol, LAGGED VIX, cash row).

        The environment (:mod:`src.env.portfolio_env`) computes these features
        causally at decision time; the pipeline records the feature contract so the
        manifest documents what is (and is not) available without look-ahead.
        """
        notes = {
            "lookback_returns": "per-asset window strictly before t",
            "realized_vol": "rolling std over configured windows, strictly before t",
            "vix": "LAGGED (vix[t-1]) — knowable at decision time t",
            "cash_row": "always-available cash asset (constant marker)",
        }
        self.manifest["10_engineer_features"] = _checksum(notes)
        return notes

    # -- Stage 11 -------------------------------------------------------------
    def reconcile_vendors(self) -> dict[str, Any]:
        """Stage 11: two-vendor reconciliation (synthetic: two seeds reconciled).

        A secondary synthetic vendor (different seed) is generated; the per-asset
        mean absolute return discrepancy is computed and recorded. The primary
        vendor is authoritative; large discrepancies would be flagged for review.
        """
        assert self._panel is not None
        secondary = make_synthetic_panel(
            n_assets=self.n_assets, n_days=self.n_days, seed=self.seed + 1
        )
        self._secondary = secondary
        assert self._selected is not None
        sec_ret = secondary.returns[:, self._selected]
        discrepancy = float(np.abs(self._panel.returns - sec_ret).mean())
        report = {
            "primary_authoritative": True,
            "mean_abs_return_discrepancy": round(discrepancy, 10),
        }
        self.manifest["11_reconcile_vendors"] = _checksum(report)
        return report

    # -- Stage 12 -------------------------------------------------------------
    def split_and_embargo(self) -> tuple[dict[str, dict[str, int]], int]:
        """Stage 12: map config split dates to indices with a purge gap.

        Maps the ``train`` / ``val`` / ``test`` calendar ranges in
        ``config/data.yaml`` onto contiguous, non-overlapping index ranges, then
        removes ``max(embargo_days, lookback_days)`` sessions from the *start* of each
        later split so adjacent splits are separated by that gap. The purge must cover
        the feature lookback, not merely the embargo: every observation reads
        ``returns[t-lookback:t]``, so a bare ``embargo``(21) < ``lookback``(60) would
        leave a later split's first ``lookback - embargo`` windows reading the prior
        split's returns (R18; mirrors :func:`src.data.loaders.embargoed_val_start` and
        :func:`src.env.runner.make_env_builder`).

        Returns
        -------
        tuple
            ``(splits, purge)`` where ``splits`` maps each of ``train`` / ``val`` /
            ``test`` to ``{'start': int, 'end': int}`` (homogeneous, subscript-safe for
            every entry) and ``purge`` is the applied ``max(embargo, lookback)`` gap.
        """
        assert self._panel is not None
        dates = self._panel.dates.astype("datetime64[D]")
        splits_cfg = self.cfg["splits"] if "splits" in self.cfg else self.cfg.splits
        embargo = int(self.cfg["embargo_days"] if "embargo_days" in self.cfg else self.cfg.embargo_days)
        # R18: the purge that separates splits is max(embargo, lookback), NOT bare embargo,
        # so no observation window returns[t-lookback:t] straddles a split boundary. This
        # mirrors loaders.embargoed_val_start (purge = max(emb, lookback)) and runner.
        purge = max(embargo, int(self.lookback_days))

        def _range(name: str) -> tuple[int, int]:
            block = splits_cfg[name]
            start = np.datetime64(str(block["start"]), "D")
            end = np.datetime64(str(block["end"]), "D")
            idx = np.flatnonzero((dates >= start) & (dates <= end))
            if idx.size == 0:
                # Date range falls outside the synthetic calendar: fall back to a
                # proportional split so the pipeline still runs end-to-end.
                return (-1, -1)
            return int(idx[0]), int(idx[-1] + 1)

        raw = {name: _range(name) for name in ("train", "val", "test")}

        if any(lo < 0 for lo, _ in raw.values()):
            # Proportional fallback (synthetic calendar shorter than config period).
            t = self._panel.T
            a, b = int(t * 0.6), int(t * 0.8)
            raw = {"train": (0, a), "val": (a, b), "test": (b, t)}

        # Apply the purge gap: shrink each later split's start by `purge` (R18).
        ordered = ["train", "val", "test"]
        splits: dict[str, dict[str, int]] = {}
        prev_end = 0
        for i, name in enumerate(ordered):
            lo, hi = raw[name]
            if i > 0:
                lo = max(lo, prev_end + purge)
            lo = min(lo, hi)
            splits[name] = {"start": int(lo), "end": int(hi)}
            prev_end = hi
        # Keep `splits` homogeneous (every value is {'start','end'}); the purge is returned
        # separately and provenance-tracked via the stage checksum, rather than smuggled into
        # the splits dict under an 'embargo_days' int (which broke the dict's element type and
        # would make a generic `block['start']` consumer raise TypeError).
        self.manifest["12_split_and_embargo"] = _checksum({"splits": splits, "purge_days": purge})
        return splits, purge

    # -- Stage 13 -------------------------------------------------------------
    def freeze_and_checksum(self) -> dict[str, str]:
        """Stage 13: freeze the panel and emit the full SHA-256 checksum manifest."""
        assert self._panel is not None
        self.manifest["13_freeze_panel"] = _checksum(
            {
                "returns": self._panel.returns,
                "vix": self._panel.vix,
                "dates": self._panel.dates,
                "asset_ids": self._panel.asset_ids,
            }
        )
        return dict(self.manifest)

    # -- Orchestration --------------------------------------------------------
    def run(self) -> tuple[Panel, dict[str, Any]]:
        """Execute stages 1-13 in order.

        Returns
        -------
        tuple
            ``(panel, manifest)`` — the frozen :class:`Panel` and a manifest dict
            of per-stage SHA-256 checksums (one entry per stage, deterministic for a
            fixed seed). The split indices are stored under ``manifest['splits']`` (a
            homogeneous ``{name: {'start','end'}}`` map) and the applied purge gap
            ``max(embargo, lookback)`` under ``manifest['embargo_days']``.
        """
        self.ingest_membership()
        self.ingest_prices_returns()
        self.ingest_market_cap()
        self.ingest_corporate_actions()
        self.map_identifiers()
        self.reconstruct_membership_pit()
        self.correct_delisting_returns()
        self.select_top_n_pit()
        panel = self.build_return_panel()
        self.engineer_features()
        self.reconcile_vendors()
        # Stage 12 now returns the homogeneous splits dict and the purge separately
        # (the purge is no longer smuggled into `splits` under an int 'embargo_days' key).
        splits, purge = self.split_and_embargo()
        # The RETURNED manifest is genuinely heterogeneous (str stage-checksums + the splits dict + the int
        # purge), so it is typed dict[str, Any]; self.manifest itself stays str-only (checksums). This is the
        # honest type for the mixed payload — no per-line ``type: ignore`` smuggle (audit 2026-06-20).
        out_manifest: dict[str, Any] = dict(self.freeze_and_checksum())
        out_manifest["splits"] = splits
        out_manifest["embargo_days"] = purge  # applied gap = max(embargo, lookback); kept discoverable
        return panel, out_manifest


#: Backwards-compatible alias for the scaffold name.
DataPipeline = GoldPipeline
