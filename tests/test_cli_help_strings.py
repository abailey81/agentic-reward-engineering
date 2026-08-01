"""Every ``scripts/*.py`` CLI must be able to PRINT its own ``--help``.

argparse runs printf-style ``%``-substitution on ``help=`` strings (``argparse._expand_help``:
``self._get_help_string(action) % params``). A bare ``%`` followed by a conversion character is
therefore a live format directive, and it raises only at ``format_help()`` time -- so it is
INVISIBLE to a test suite that never types ``--help``, and surfaces the first time a human does.

Found live on 2026-07-27, on the two CLIs that matter most:

* ``run_campaign_cluster.py`` -- ``"93-97% of all trainings"`` -> ``"% o"`` = space-flagged octal
  -> ``TypeError: %o format: an integer is required, not dict``. The CAMPAIGN LAUNCHER could not
  print its own help, hours before launch.
* ``p6_authored_ladder.py`` -- ``"a genuine 95% " "t-interval"`` -> ``"% t"`` ->
  ``ValueError: unsupported format character 't'``.

Both were prose percentages in help text. The fix is ``%%``; this test is the guard.

Static (AST) rather than ``subprocess ... --help`` on purpose: 57 subprocesses that each import
torch would cost minutes, and the defect is fully determined by the source text.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# Every argparse CLI in the repo, not just scripts/: src/cluster/run_one.py is the entry point that
# runs on EVERY cluster node, and src/cluster/bayes_chain.py runs the whole GP chain in one job.
_CLI_PATHS = sorted((_ROOT / "scripts").glob("*.py")) + [
    _ROOT / "src" / "cluster" / "run_one.py",
    _ROOT / "src" / "cluster" / "bayes_chain.py",
]

# Scanned LEFT TO RIGHT, consuming '%%' as one atom -- a lookahead alone is wrong, because in
# '20%% of' the SECOND '%' of the escaped pair is followed by a space and would look live.
# '%(' is the named substitution argparse itself supplies (e.g. '%(default)s') and is legitimate.
_PERCENT_ATOM = re.compile(r"%%|%(.)", re.S)


def _first_live_directive(text: str) -> int | None:
    """Index of the first '%' argparse would try to interpret, or ``None`` if the text is safe."""
    for m in _PERCENT_ATOM.finditer(text):
        if m.group(0) == "%%" or m.group(1) == "(":
            continue
        return m.start()
    # A trailing bare '%' with nothing after it is also a format error ("incomplete format").
    if text.endswith("%") and not text.endswith("%%"):
        return len(text) - 1
    return None


def _help_strings(path: Path) -> list[tuple[str, str]]:
    """Return ``(option, help_text)`` for every literal ``help=`` in ``add_argument`` calls.

    Implicit string concatenation ("a" "b") is already folded into one ``ast.Constant`` by the
    parser, which is exactly how the two real defects were formed -- the '%' ended one chunk and
    the conversion character began the next.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        opt = next(
            (a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)),
            "<positional>",
        )
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    found.append((opt, kw.value.value))
    return found


def test_no_live_percent_directive_in_any_cli_help_string() -> None:
    offenders: list[str] = []
    scanned = 0
    for path in _CLI_PATHS:
        assert path.is_file(), f"{path} is gone -- fix this list, do not let coverage silently shrink"
        for opt, text in _help_strings(path):
            scanned += 1
            i = _first_live_directive(text)
            if i is None:
                continue
            ctx = text[max(0, i - 40) : i + 20]
            offenders.append(f"{path.name} {opt}: ...{ctx}... (escape it as '%%')")
    assert scanned > 100, f"the AST walk found only {scanned} help strings -- it stopped working"
    assert not offenders, (
        "argparse %-formats help strings, so these would raise at --help time:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_catches_the_two_real_2026_07_27_defects(tmp_path: Path) -> None:
    """Falsifiability: reconstruct both real defects and prove the regex flags them."""
    assert _first_live_directive("(93-97% of all trainings)") is not None  # -> '%o'
    assert _first_live_directive("a genuine 95% " "t-interval spans zero") is not None  # -> '%t'
    # ...and does NOT flag the forms that are legitimate.
    assert _first_live_directive("~half throughput on ~20%% of the work") is None
    assert _first_live_directive("the tc throttle (default: %(default)s)") is None
    assert _first_live_directive("no percent here at all") is None
    # A trailing bare '%' is an incomplete format, not a safe literal.
    assert _first_live_directive("utilisation reached 97%") is not None
    assert _first_live_directive("utilisation reached 97%%") is None
