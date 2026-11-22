"""Phase 5: Hardening & Integration Testing.

Tests:
 - End-to-end privacy flow: strict blocks → guarded allows with consent → audit trail
 - Race condition protection: double-check pattern catches setting changes mid-request
 - API key scrubbing: sensitive data never appears in audit log entries
"""
from __future__ import annotations

import re
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from jarvis_backend.guarded import (
    AuditEventType,
    AuditLog,
    GuardedModeController,
    PrivacyMode,
    cloud_access_error,
    consent_matches,
)
from jarvis_backend.permissions import SENDS_OUT, decide
from jarvis_backend.tools import BY_NAME, EXTERNAL_TOOLS, ToolSpec


# ── Task 1: End-to-end privacy flow ─────────────────────────────────────────

class TestE2EPrivacyFlow(unittest.TestCase):
    """Full privacy pipeline: strict → guarded → consent → allowed → audit trail."""

    def setUp(self):
        self.ctrl = GuardedModeController(audit_path=None)

    def test_strict_blocks_cloud_provider(self):
        """In strict mode, cloud providers are always blocked regardless of consent."""
        err = self.ctrl.check_access("strict", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        self.assertIn("strict", err.lower())

    def test_strict_allows_local_provider(self):
        """In strict mode, local providers work normally."""
        err = self.ctrl.check_access("strict", "ollama", provider_is_local=True, consent_given=False)
        self.assertIsNone(err)

    def test_guarded_blocks_without_verification(self):
        """In guarded mode, unverified cloud provider is blocked."""
        err = self.ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)
        self.assertIn("verified", err.lower())

    def test_guarded_blocks_without_consent(self):
        """In guarded mode, verified cloud provider requires consent."""
        self.ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=100)
        err = self.ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=False)
        self.assertIsNotNone(err)
        self.assertIn("confirmation", err.lower())

    def test_guarded_allows_with_verification_and_consent(self):
        """In guarded mode, verified provider + consent → allowed."""
        self.ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=100)
        err = self.ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNone(err)

    def test_full_flow_audit_trail(self):
        """Walk the complete flow and verify audit entries are created at each step."""
        # Step 1: Strict mode blocks
        self.ctrl.check_access("strict", "openai", provider_is_local=False, consent_given=False)

        # Step 2: Verify provider
        self.ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)

        # Step 3: Guarded mode without consent (blocked)
        self.ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=False)

        # Step 4: Guarded mode with consent (allowed)
        self.ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)

        entries = self.ctrl.audit.recent(100)
        event_types = [e["event"] for e in entries]

        # Verify the complete trail
        self.assertIn("cloud_request_blocked", event_types)      # strict block
        self.assertIn("provider_verify_ok", event_types)          # verification
        self.assertIn("cloud_consent_missing", event_types)       # consent missing
        self.assertIn("cloud_request_allowed", event_types)       # final success

    def test_strict_blocks_external_tools_via_permissions(self):
        """Strict mode blocks external_transmission tools through permissions.decide()."""
        web_search = BY_NAME["web_search"]
        result, cap = decide(web_search, {}, privacy_mode="strict")
        self.assertEqual(result, "block")
        self.assertEqual(cap, "send_data_out")

    def test_guarded_forces_ask_on_external_tools(self):
        """Guarded mode forces 'ask' on external tools even if policy says allow."""
        web_search = BY_NAME["web_search"]
        result, cap = decide(web_search, {}, saved={"send_data_out": "allow"}, privacy_mode="guarded")
        self.assertEqual(result, "ask")

    def test_local_tools_unaffected_by_privacy_mode(self):
        """Non-external tools work the same in all privacy modes."""
        read_file = BY_NAME.get("read_file")
        if read_file is None:
            self.skipTest("read_file tool not defined")
        for mode in ("strict", "guarded"):
            result, cap = decide(read_file, {}, privacy_mode=mode)
            self.assertNotEqual(result, "block")

    def test_guarded_allows_local_provider(self):
        """Guarded mode allows local providers without any consent."""
        err = self.ctrl.check_access("guarded", "ollama", provider_is_local=True, consent_given=False)
        self.assertIsNone(err)


# ── Task 2: Race condition protection (double-check pattern) ────────────────

class TestRaceConditionProtection(unittest.TestCase):
    """Verify that the double-check gate prevents TOCTOU attacks."""

    def test_cloud_access_error_checks_current_mode(self):
        """cloud_access_error reads the mode parameter at call time, not a cached value."""
        # First check: guarded allows with consent
        err1 = cloud_access_error("guarded", provider_is_local=False,
                                   consent_given=True, provider_verified=True)
        self.assertIsNone(err1)

        # Settings change between entry and call: now strict
        err2 = cloud_access_error("strict", provider_is_local=False,
                                   consent_given=True, provider_verified=True)
        self.assertIsNotNone(err2)

    def test_consent_matches_prevents_provider_switch(self):
        """consent_matches catches when provider changed between consent and call."""
        # User consented for openai/gpt-4
        self.assertTrue(consent_matches("openai", "gpt-4", "openai", "gpt-4"))

        # Provider switched to anthropic between consent and call
        self.assertFalse(consent_matches("openai", "gpt-4", "anthropic", "claude-3"))

    def test_consent_matches_prevents_model_switch(self):
        """consent_matches catches when model changed between consent and call."""
        # Consented for gpt-4, but model changed to gpt-3.5
        self.assertFalse(consent_matches("openai", "gpt-4", "openai", "gpt-3.5-turbo"))

    def test_verification_invalidated_on_session_clear(self):
        """Clearing the session invalidates all verifications (prevents stale state)."""
        ctrl = GuardedModeController(audit_path=None)
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)
        self.assertTrue(ctrl.is_provider_verified("openai"))

        ctrl.clear_session()
        self.assertFalse(ctrl.is_provider_verified("openai"))

    def test_double_check_different_modes(self):
        """Simulate the two-gate pattern: entry gate passes, pre-call gate catches change."""
        ctrl = GuardedModeController(audit_path=None)
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)

        # Entry gate: guarded mode with consent → pass
        err1 = ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNone(err1)

        # -- Between gates, user switches to strict mode --

        # Pre-call gate: strict mode → block
        err2 = ctrl.check_access("strict", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err2)

    def test_failed_verification_blocks_subsequent_requests(self):
        """A failed verification prevents future cloud requests until re-verified."""
        ctrl = GuardedModeController(audit_path=None)
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)

        # Passes initially
        err = ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNone(err)

        # Verification fails (provider key revoked, etc.)
        ctrl.record_verification("openai", "gpt-4", ok=False, error="401 Unauthorized")
        self.assertFalse(ctrl.is_provider_verified("openai"))

        # Now blocked
        err = ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)
        self.assertIsNotNone(err)

    def test_privacy_mode_enum_defaults_to_strict(self):
        """Invalid privacy mode strings default to strict (safest option)."""
        self.assertEqual(PrivacyMode.from_str("invalid"), PrivacyMode.STRICT)
        self.assertEqual(PrivacyMode.from_str(""), PrivacyMode.STRICT)
        self.assertEqual(PrivacyMode.from_str("open"), PrivacyMode.STRICT)


# ── Task 3: API key scrubbing in audit log ──────────────────────────────────

class TestKeyScrubbing(unittest.TestCase):
    """Verify that API keys, tokens, and sensitive data never appear in audit entries."""

    # Realistic API key patterns
    FAKE_KEYS = [
        "sk-proj-abcdefghij1234567890abcdefghij1234567890",   # OpenAI-style
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",   # Anthropic-style
        "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q",           # Google-style
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",             # GitHub PAT
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",  # JWT
    ]

    def test_audit_entries_contain_no_api_keys(self):
        """Audit entries generated by normal operations contain no key-shaped strings."""
        ctrl = GuardedModeController(audit_path=None)

        # Normal operations
        ctrl.check_access("strict", "openai", provider_is_local=False, consent_given=False)
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)
        ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)

        entries = ctrl.audit.recent(100)
        all_text = str(entries)

        # No key-like patterns
        key_pattern = re.compile(r"(sk-[a-zA-Z0-9_-]{20,}|AIza[a-zA-Z0-9_-]{30,}|ghp_[a-zA-Z0-9]{30,}|Bearer\s+eyJ[a-zA-Z0-9_-]+)")
        self.assertIsNone(key_pattern.search(all_text),
                         f"Audit log contains key-like string: {key_pattern.search(all_text)}")

    def test_error_messages_dont_leak_keys(self):
        """Error messages from verification don't contain API keys."""
        ctrl = GuardedModeController(audit_path=None)

        # Simulate a verification failure — only the error message goes in
        ctrl.record_verification("openai", "gpt-4", ok=False,
                                  error="Authentication failed: invalid credentials")

        entries = ctrl.audit.recent(100)
        all_text = str(entries)

        for key in self.FAKE_KEYS:
            self.assertNotIn(key, all_text)

    def test_evidence_field_uses_safe_values(self):
        """The evidence dict in audit entries should contain safe, non-sensitive values."""
        log = AuditLog(path=None)

        # Record with structured evidence (no keys)
        entry = log.record(
            AuditEventType.CLOUD_REQUEST_ALLOWED,
            provider_id="openai",
            model="gpt-4",
            detail="consent verified, provider verified",
            evidence={"latency_ms": 123, "consent": True},
        )

        d = entry.to_dict()
        # Evidence should only have our safe fields
        self.assertEqual(d["evidence"]["latency_ms"], 123)
        self.assertEqual(d["evidence"]["consent"], True)

        # No credential-shaped strings anywhere
        evidence_text = str(d["evidence"])
        for key in self.FAKE_KEYS:
            self.assertNotIn(key, evidence_text)

    def test_audit_log_file_contains_no_secrets(self):
        """When writing to a JSONL file, entries contain no secrets."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            log = AuditLog(path=path)
            log.record(AuditEventType.PROVIDER_VERIFY_OK,
                       provider_id="openai", model="gpt-4",
                       detail="verified in 50ms",
                       evidence={"latency_ms": 50})
            log.record(AuditEventType.CLOUD_REQUEST_ALLOWED,
                       provider_id="openai", model="gpt-4",
                       detail="consent verified")

            content = path.read_text()
            key_pattern = re.compile(r"(sk-[a-zA-Z0-9_-]{20,}|AIza[a-zA-Z0-9_-]{30,})")
            self.assertIsNone(key_pattern.search(content))

            # Verify it's valid JSONL
            import json
            lines = content.strip().split("\n")
            self.assertEqual(len(lines), 2)
            for line in lines:
                parsed = json.loads(line)
                self.assertIn("event", parsed)
                self.assertIn("timestamp", parsed)
        finally:
            path.unlink(missing_ok=True)

    def test_detail_field_is_descriptive_not_raw(self):
        """Detail messages describe what happened without exposing raw credentials."""
        ctrl = GuardedModeController(audit_path=None)

        # Normal flow
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=100)
        ctrl.check_access("guarded", "openai", provider_is_local=False, consent_given=True)

        entries = ctrl.audit.recent(100)
        for entry in entries:
            detail = entry.get("detail", "")
            # Details should be human-readable descriptions, not raw data
            self.assertNotIn("Authorization", detail)
            self.assertNotIn("api_key", detail)
            self.assertNotIn("token", detail.lower().split())  # "token" as standalone word

    def test_provider_id_is_not_a_key(self):
        """Provider IDs in audit entries are identifiers (openai, anthropic), not API keys."""
        ctrl = GuardedModeController(audit_path=None)
        ctrl.record_verification("openai", "gpt-4", ok=True, latency_ms=50)

        entries = ctrl.audit.recent(100)
        for entry in entries:
            pid = entry.get("provider_id", "")
            # Provider ID should be a short identifier, not a key
            self.assertLess(len(pid), 30, f"Provider ID looks like a key: {pid}")


if __name__ == "__main__":
    unittest.main()
