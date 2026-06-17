"""Market-regime labelling (FINAL_PLAN F.12, audit B-6).

A regime label is assigned per date so out-of-sample performance can be reported
stratified by market regime. The default definition is a VIX-threshold rule
(``config/regimes.yaml``): calm (VIX < calm), normal (calm <= VIX <= stress), and
stress (VIX > stress), mapped to integer labels ``{0, 1, 2}``.

The *independent-regime count* — the number of contiguous regime episodes
(run-length blocks) — is what feeds the H2 power analysis: a 20-year sample still
contains only single-digit independent stress regimes, which bounds the achievable
power. It is therefore the episode count, not the per-date count, that matters.

Audit refs: B-6 (regime-stratified evaluation; the independent-regime count bounds
H2 power).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.panel import Panel

__all__ = ["label_regimes", "independent_regime_count"]

#: Integer labels for the VIX-threshold definition.
CALM, NORMAL, STRESS = 0, 1, 2


def label_regimes(panel: Panel, cfg: Any) -> np.ndarray:
    """Assign an integer regime label to every date of ``panel``.

    Parameters
    ----------
    panel : Panel
        The market panel; its ``vix`` series drives the VIX-threshold definition.
    cfg : Any
        Parsed ``config/regimes.yaml``. ``cfg["definition"]`` selects the variant;
        ``cfg["vix_thresholds"]`` gives ``{"calm": <float>, "stress": <float>}``.

    Returns
    -------
    np.ndarray, shape (T,)
        Per-date integer labels. For the VIX-threshold definition the labels are
        in ``{0, 1, 2}`` for calm / normal / stress.

    Raises
    ------
    NotImplementedError
        For the unimplemented ``nber_dates`` / ``hmm_2state`` / ``hmm_3state``
        variants. The ``vix_threshold`` variant is fully implemented.
    """
    definition = str(cfg["definition"]) if "definition" in cfg else str(cfg.definition)

    if definition == "vix_threshold":
        thresholds = cfg["vix_thresholds"] if "vix_thresholds" in cfg else cfg.vix_thresholds
        calm = float(thresholds["calm"])
        stress = float(thresholds["stress"])
        if calm > stress:
            raise ValueError(f"calm threshold ({calm}) must be <= stress threshold ({stress})")
        vix = np.asarray(panel.vix, dtype=np.float64)
        labels = np.full(vix.shape, NORMAL, dtype=np.int64)
        labels[vix < calm] = CALM
        labels[vix > stress] = STRESS
        return labels

    if definition in ("nber_dates", "hmm_2state", "hmm_3state"):
        raise NotImplementedError(
            f"regime definition {definition!r} is not implemented "
            "(only 'vix_threshold' is available)"
        )

    raise ValueError(f"unknown regime definition {definition!r}")


def independent_regime_count(labels: np.ndarray) -> int:
    """Count contiguous regime episodes (run-length blocks).

    Adjacent dates sharing a label form one episode; a new episode begins each time
    the label changes. This episode count — not the per-date count — is the number
    of *independent* regimes fed to the power analysis.

    Parameters
    ----------
    labels : np.ndarray, shape (T,)
        Per-date integer labels from :func:`label_regimes`.

    Returns
    -------
    int
        The number of contiguous run-length blocks (0 for an empty array).
    """
    arr = np.asarray(labels).ravel()
    if arr.size == 0:
        return 0
    changes = int(np.count_nonzero(arr[1:] != arr[:-1]))
    return changes + 1
