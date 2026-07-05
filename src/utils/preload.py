"""Native-library preload that prevents a torch/pyarrow ABI segfault (verified 2026-06-20).

On this stack (torch 2.6 + pyarrow, Windows), importing ``pyarrow(.parquet)`` — which
``src.data.loaders.load_gold_panel`` uses to read the REAL gold parquet — AFTER torch has been imported
SEGFAULTS (SIGSEGV/exit 139): a first-loader-wins native-runtime conflict. Verified: torch-then-gold => crash,
gold-then-torch => OK, and pyarrow-then-torch-then-gold => OK. Every process that uses BOTH the real gold data
AND torch must therefore import pyarrow FIRST.

This matters precisely because the project trains ONLY on real survivorship-free gold data — there is no
synthetic fallback in a real run — so the gold load must be rock-solid. Call :func:`preload` at the very top
of each entry point (and the pool worker init) BEFORE any torch import.
"""

from __future__ import annotations


def preload(*, strict: bool = False) -> None:
    """Import pyarrow before torch is ever imported, avoiding the ABI segfault on real-gold parquet loads.

    Parameters
    ----------
    strict:
        The PRODUCTION contract (H2). When ``True``, a failure to import pyarrow RAISES
        :class:`ImportError` — the real campaign trains ONLY on real gold parquet, so a missing/broken
        pyarrow is a hard, fail-loud precondition, not something to swallow (a swallowed failure would
        let torch import first and then SIGSEGV at the gold load). Every real entry point (run_campaign
        / run_prototype / run_sigma_pilot_train / learning_curve / smoke_test / run_subexperiment)
        passes ``strict=True``. The default ``False`` keeps the TEST/conftest path tolerant so the suite
        still collects in a torch-less / pyarrow-optional minimal env.
    """
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:  # pyarrow optional only in a torch-less minimal env
        if strict:
            raise ImportError(
                "preload(strict=True): pyarrow could not be imported before torch, but the production "
                "run reads the real gold parquet — importing torch first would SIGSEGV at load time "
                "(src/utils/preload). Install pyarrow into the run environment before launching."
            ) from exc
