"""Commit certification — the launch gate's unit of truth.

A suite result certifies a TREE STATE, not a folder. With four sessions sharing this working
directory, the only previous remedy was social ("everyone stop"), which is fragile and must be
repeated. Git already has immutable tree states, and the launch deploys a COMMIT, so the commit is
the correct unit. These tests pin the property that makes the certificate trustworthy: it must go
INVALID the moment it no longer describes what you are about to deploy.
"""
from __future__ import annotations

import json

from scripts import certify_commit as C


def _write(tmp_path, monkeypatch, payload):
    cert = tmp_path / "commit_certificate.json"
    cert.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(C, "CERT_PATH", cert)
    return cert


def test_a_certificate_for_a_DIFFERENT_commit_is_INVALID(tmp_path, monkeypatch):
    """The whole point. Green at some earlier commit says nothing about what you are launching."""
    _write(tmp_path, monkeypatch, {"commit": "0" * 40, "certified": True})
    v = C.check("HEAD")
    assert v["valid"] is False
    assert "HEAD is" in v["reason"] and "Re-certify" in v["reason"]


def test_a_FAILED_certificate_is_never_treated_as_valid(tmp_path, monkeypatch):
    head = C.git("rev-parse", "HEAD")
    _write(tmp_path, monkeypatch, {"commit": head, "certified": False,
                                   "failed_steps": ["pytest_full"]})
    v = C.check("HEAD")
    assert v["valid"] is False and "FAILED" in v["reason"]


def test_a_matching_PASSED_certificate_is_valid(tmp_path, monkeypatch):
    head = C.git("rev-parse", "HEAD")
    _write(tmp_path, monkeypatch, {"commit": head, "certified": True, "utc": "now"})
    assert C.check("HEAD")["valid"] is True


def test_NO_certificate_is_invalid_rather_than_assumed_fine(tmp_path, monkeypatch):
    """Absence of evidence must never read as evidence of readiness."""
    monkeypatch.setattr(C, "CERT_PATH", tmp_path / "does_not_exist.json")
    v = C.check("HEAD")
    assert v["valid"] is False and "no certificate has ever been produced" in v["reason"]


def test_the_certificate_states_its_own_scope():
    """A certificate that does not say what it covers invites being over-read."""
    src = (C.REPO / "scripts" / "certify_commit.py").read_text(encoding="utf-8")
    assert "THIS COMMIT ONLY" in src
    assert "nothing about the working" in src and "Re-certify" in src
