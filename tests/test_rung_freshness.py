"""Behaviour tests for the rung-freshness compile gate (scripts/check_rung_freshness.py).

Tag semantics (CH6 reporting rule 5): RUNG:n must equal the achieved rung; LEG-TIER must be the
frozen 30; --final additionally fails unfilled [FROM CAMPAIGN...] slots. The achieved rung must be
stated (flag or json) and lie on the frozen E1 ladder — never assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_rung_freshness as G  # noqa: E402


def _paper(tmp_path: Path, text: str, name: str = "CH6_results.md") -> Path:
    d = tmp_path / "paper"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
    return d


def _run(paper_dir: Path, *extra: str) -> int:
    return G.main(["--paper-dir", str(paper_dir), *extra])


def test_fresh_tags_pass_and_stale_rung_fails(tmp_path: Path, capsys) -> None:
    d = _paper(tmp_path, "IQM 0.41 <!--RUNG:30--> and CVaR -0.052 <!--LEG-TIER:30-->\n")
    assert _run(d, "--achieved", "30") == 0
    assert _run(d, "--achieved", "100") == 1  # the same tag is stale once rung-100 lands
    out = capsys.readouterr().out
    assert "stale RUNG:30" in out and "CH6_results.md:1" in out  # file:line named


def test_wrong_leg_tier_fails_even_when_rung_fresh(tmp_path: Path) -> None:
    d = _paper(tmp_path, "leg diff +0.003 <!--LEG-TIER:100-->\n")
    assert _run(d, "--achieved", "30") == 1


def test_final_mode_fails_unfilled_slots_default_mode_ignores_them(tmp_path: Path) -> None:
    d = _paper(tmp_path, "TOST bound `[FROM CAMPAIGN: CI]` pending.\n")
    assert _run(d, "--achieved", "403") == 0          # per-rung mode: slots are legitimate
    assert _run(d, "--achieved", "403", "--final") == 1  # pre-submission: they are not


def test_achieved_resolves_from_json_and_missing_source_fails_loud(tmp_path: Path) -> None:
    d = _paper(tmp_path, "x <!--RUNG:189-->\n")
    j = tmp_path / "achieved_rung.json"
    j.write_text(json.dumps({"achieved_rung": 189}), encoding="utf-8")
    assert G.main(["--paper-dir", str(d), "--achieved-json", str(j)]) == 0
    with pytest.raises(SystemExit):  # no flag, no json — never assumed
        G.main(["--paper-dir", str(d), "--achieved-json", str(tmp_path / "absent.json")])


def test_achieved_off_the_frozen_ladder_is_a_usage_error(tmp_path: Path) -> None:
    d = _paper(tmp_path, "x\n")
    with pytest.raises(SystemExit):
        _run(d, "--achieved", "200")


def test_drafts_files_are_excluded_from_the_gate(tmp_path: Path) -> None:
    d = _paper(tmp_path, "clean\n")
    (d / "DRAFTS_scratch.md").write_text("stale <!--RUNG:30--> [FROM CAMPAIGN: x]", encoding="utf-8")
    assert _run(d, "--achieved", "403", "--final") == 0


def test_whitespace_tolerant_tag_parse(tmp_path: Path) -> None:
    d = _paper(tmp_path, "y <!-- RUNG:30 --> z <!-- LEG-TIER:30 -->\n")
    assert _run(d, "--achieved", "30") == 0
    assert _run(d, "--achieved", "279") == 1
