"""Deterministic Jupyter-notebook writer shared by the notebook builders (report-only tooling).

The dissertation's notebooks are GENERATED artifacts: ``scripts/build_notebook_results.py`` and
``scripts/build_notebook_provenance.py`` author their cells as plain Python strings and emit the
``.ipynb`` through this one writer, so both notebooks share one byte-stability contract:

* **Fixed cell ids** — ``<prefix>-NNN`` by position (nbformat 4.5 requires ids; random ids would
  make every regeneration a spurious diff).
* **No timestamps, no randomness** — the notebook JSON is a pure function of the builder source.
* **Canonical JSON** — ``sort_keys`` + ``indent=1`` + ``ensure_ascii`` + trailing newline, matching
  ``nbformat.write`` conventions while pinning the byte layout, so regeneration is byte-identical
  (asserted by each builder via a double-build hash check).
* **Clean by construction** — code cells carry ``execution_count: null`` and empty ``outputs``; the
  shipped notebooks store no results (execution happens in CI/nbconvert, never in the repo copy).

Validated with :func:`nbformat.validate` before writing — a malformed cell fails the build, never
lands on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["code", "md", "build_notebook", "write_notebook", "sha256_bytes"]

#: Static notebook metadata (mirrors the repo's existing results_walkthrough.ipynb — no timestamps).
_NB_METADATA: dict[str, Any] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "authorship": "Tamer Atesyakar — UCL MSc dissertation",
}


def md(source: str) -> dict[str, Any]:
    """A markdown cell from a source string (id assigned positionally in :func:`build_notebook`)."""
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict[str, Any]:
    """A code cell (never pre-executed: ``execution_count`` null, ``outputs`` empty)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _source_lines(source: str) -> list[str]:
    """nbformat's canonical source encoding: a list of lines, each keeping its ``\\n``."""
    return source.splitlines(keepends=True)


def build_notebook(cells: list[dict[str, Any]], *, id_prefix: str) -> dict[str, Any]:
    """Assemble the notebook dict with positional ``<id_prefix>-NNN`` cell ids (deterministic)."""
    out_cells: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        c = dict(cell)
        c["id"] = f"{id_prefix}-{i:03d}"
        c["source"] = _source_lines(str(c["source"]))
        out_cells.append(c)
    return {
        "cells": out_cells,
        "metadata": dict(_NB_METADATA),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(nb: dict[str, Any], path: Path | str) -> Path:
    """Validate with nbformat, then write canonical JSON (sorted keys, ASCII, LF, one trailing NL)."""
    import nbformat

    nbformat.validate(nbformat.from_dict(nb))  # fail loud BEFORE touching disk
    payload = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("ascii"))  # LF endings on every platform (no text-mode CRLF)
    return path


def sha256_bytes(path: Path | str) -> str:
    """Hex SHA-256 of a file's bytes — the builders' double-build byte-stability check."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
