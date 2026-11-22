"""Tests for the Guarded mode privacy system (guarded.py).

Covers:
- cloud_access_error(): the central policy gate
- consent_matches(): consent field validation
- PrivacyMode enum parsing
- AuditLog: append-only logging
- GuardedModeController: verification tracking and full access check
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "common")]

from jarvis_backend.guarded import (  # noqa: E402
    AuditEventType,
    AuditLog,
    GuardedModeController,
    PrivacyMode,
    cloud_access_error,
    consent_matches,
)


class TestPrivacyMode(unittest.TestCase):
    def test_valid_strict(self):
        self.assertEqual(PrivacyMode.from_str("strict"), PrivacyMode.STRICT)

    def test_valid_guarded(self):
        self.assertEqual(PrivacyMode.from_str("guarded"), PrivacyMode.GUARDED)

    def test_invalid_defaults_to_strict(self):
        self.assertEqual(PrivacyMode.from_str("invalid"), PrivacyMode.STRICT)
        self.assertEqual(PrivacyMode.from_str(""), PrivacyMode.STRICT)


class TestCloudAccessError(unittest.TestCase):
    """Test the core policy gate function."""

    def test_local_always_allowed_strict(self):
        self.assertIsNone(cloud_access_error("strict", provider_is_local=True, consent_given=False))

    def test_local_always_allowed_guarded(self):
        self.assertIsNone(cloud_access_error("guarded", provider_is_local=True, consent_given=False))

    def test_strict_blocks_cloud(self):
        err = cloud_access_error("strict", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        self.assertIn("Strict", err)

    def test_guarded_allows_cloud_with_consent_and_verification(self):
        self.assertIsNone(cloud_access_error("guarded", provider_is_local=False,
                                             consent_given=True, provider_verified=True))

    def test_guarded_blocks_without_consent(self):
        err = cloud_access_error("guarded", provider_is_local=False,
                                 consent_given=False, provider_verified=True)
        self.assertIsNotNone(err)
        self.assertIn("confirmation", err)

    def test_guarded_blocks_without_verification(self):
        err = cloud_access_error("guarded", provider_is_local=False,
                                 consent_given=True, provider_verified=False)
        self.assertIsNotNone(err)
        self.assertIn("verified", err)

    def test_invalid_mode_treated_as_strict(self):
        err = cloud_access_error("invalid_mode", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        self.assertIn("Strict", err)


class TestConsentMatches(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(consent_matches("gemini", "gemini-2.0-flash", "gemini", "gemini-2.0-flash"))

    def test_provider_mismatch(self):
        self.assertFalse(consent_matches("openai", "gemini-2.0-flash", "gemini", "gemini-2.0-flash"))

    def test_model_mismatch(self):
        self.assertFalse(consent_matches("gemini", "gemini-1.5-pro", "gemini", "gemini-2.0-flash"))

    def test_none_consent_fields(self):
        self.assertFalse(consent_matches(None, None, "gemini", "gemini-2.0-flash"))


class TestAuditLog(unittest.TestCase):
    def test_records_in_memory(self):
        log = AuditLog()
        log.record(AuditEventType.CLOUD_REQUEST_ALLOWED, provider_id="gemini", detail="test")
        entries = log.recent()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event"], "cloud_request_allowed")
        self.assertEqual(entries[0]["provider_id"], "gemini")

    def test_writes_to_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        try:
            log = AuditLog(path=path)
            log.record(AuditEventType.PROVIDER_VERIFY_OK, provider_id="openai", model="gpt-4o", detail="ok")
            log.record(AuditEventType.CLOUD_REQUEST_BLOCKED, provider_id="openai", detail="strict mode")
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["event"], "provider_verify_ok")
            second = json.loads(lines[1])
            self.assertEqual(second["event"], "cloud_request_blocked")
        finally:
            path.unlink(missing_ok=True)

    def test_max_memory_cap(self):
        log = AuditLog(max_memory=5)
        for i in range(10):
            log.record(AuditEventType.CLOUD_REQUEST_ALLOWED, detail=f"req-{i}")
        entries = log.recent(100)
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[0]["detail"], "req-5")  # oldest retained


class TestGuardedModeController(unittest.TestCase):
    def setUp(self):
        self.ctrl = GuardedModeController()

    def test_unverified_by_default(self):
        self.assertFalse(self.ctrl.is_provider_verified("gemini"))

    def test_verification_success(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=True, latency_ms=120)
        self.assertTrue(self.ctrl.is_provider_verified("gemini"))
        status = self.ctrl.verification_status("gemini")
        self.assertTrue(status["verified"])
        self.assertEqual(status["latency_ms"], 120)

    def test_verification_failure(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=False, error="timeout")
        self.assertFalse(self.ctrl.is_provider_verified("gemini"))
        status = self.ctrl.verification_status("gemini")
        self.assertFalse(status["verified"])
        self.assertEqual(status["error"], "timeout")

    def test_check_access_local_always_ok(self):
        err = self.ctrl.check_access("strict", "ollama", provider_is_local=True, consent_given=False)
        self.assertIsNone(err)

    def test_check_access_strict_blocks_cloud(self):
        err = self.ctrl.check_access("strict", "gemini", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        # Verify audit recorded it
        entries = self.ctrl.audit.recent()
        self.assertTrue(any(e["event"] == "cloud_request_blocked" for e in entries))

    def test_check_access_guarded_requires_verification(self):
        err = self.ctrl.check_access("guarded", "gemini", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        self.assertIn("verified", err)

    def test_check_access_guarded_requires_consent(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=True, latency_ms=100)
        err = self.ctrl.check_access("guarded", "gemini", provider_is_local=False, consent_given=False)
        self.assertIsNotNone(err)
        self.assertIn("confirmation", err)

    def test_check_access_guarded_full_success(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=True, latency_ms=100)
        err = self.ctrl.check_access("guarded", "gemini", provider_is_local=False, consent_given=True)
        self.assertIsNone(err)
        # Verify audit recorded the allowed event
        entries = self.ctrl.audit.recent()
        self.assertTrue(any(e["event"] == "cloud_request_allowed" for e in entries))

    def test_clear_session_resets_verifications(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=True, latency_ms=100)
        self.assertTrue(self.ctrl.is_provider_verified("gemini"))
        self.ctrl.clear_session()
        self.assertFalse(self.ctrl.is_provider_verified("gemini"))

    def test_audit_entries_for_provider(self):
        self.ctrl.record_verification("gemini", "gemini-2.0-flash", ok=True, latency_ms=100)
        self.ctrl.record_verification("openai", "gpt-4o", ok=True, latency_ms=200)
        gemini_entries = self.ctrl.audit.entries_for_provider("gemini")
        self.assertTrue(all(e["provider_id"] == "gemini" for e in gemini_entries))


if __name__ == "__main__":
    unittest.main()
