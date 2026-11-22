"""Guarded mode: privacy-first cloud provider access control.

Cloud providers (anything that sends prompts over the internet) require explicit
per-session verification and per-message consent when privacy_mode is "guarded".
In "strict" mode all cloud access is blocked; only local providers (Ollama,
llama.cpp, the local gateway) are allowed.

The flow:
1. User selects a cloud provider and enters an API key.
2. Before any real conversation is sent, a synthetic (non-sensitive) probe is
   sent to verify connectivity and response quality.
3. The verification result is logged in the audit trail.
4. Each subsequent cloud request requires explicit per-message consent from the
   frontend, which must identify the provider and model being consented to.
5. All cloud interactions (verification, chat, errors) are audit-logged with
   timestamps, provider, model, and redacted evidence.

Ported from ChatGPT's Phase 0 working copy (core/providers.py cloud_chat_policy_error)
and extended with audit logging and synthetic verification per the Phase 2 handoff.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger("jarvis.guarded")


# ── Privacy modes ───────────────────────────────────────────────────────────

class PrivacyMode(str, Enum):
    STRICT = "strict"      # local providers only; cloud is fully blocked
    GUARDED = "guarded"    # cloud allowed with per-message consent + verification

    @classmethod
    def from_str(cls, value: str) -> "PrivacyMode":
        try:
            return cls(value)
        except ValueError:
            log.warning("invalid privacy_mode %r, defaulting to strict", value)
            return cls.STRICT


# ── Verification state ──────────────────────────────────────────────────────

@dataclass
class ProviderVerification:
    """Tracks whether a cloud provider has been verified this session."""
    provider_id: str
    verified: bool = False
    verified_at: float = 0.0
    model_used: str = ""
    latency_ms: int = 0
    error: str = ""


# ── Audit log ───────────────────────────────────────────────────────────────

class AuditEventType(str, Enum):
    PRIVACY_MODE_SET = "privacy_mode_set"
    PROVIDER_VERIFY_START = "provider_verify_start"
    PROVIDER_VERIFY_OK = "provider_verify_ok"
    PROVIDER_VERIFY_FAIL = "provider_verify_fail"
    CLOUD_REQUEST_ALLOWED = "cloud_request_allowed"
    CLOUD_REQUEST_BLOCKED = "cloud_request_blocked"
    CLOUD_CONSENT_MISSING = "cloud_consent_missing"
    CLOUD_CONSENT_MISMATCH = "cloud_consent_mismatch"
    PROVIDER_SWITCHED = "provider_switched"


@dataclass
class AuditEntry:
    timestamp: float
    event: AuditEventType
    provider_id: str = ""
    model: str = ""
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "event": self.event.value,
            "provider_id": self.provider_id,
            "model": self.model,
            "detail": self.detail,
            "evidence": self.evidence,
        }


class AuditLog:
    """Append-only audit log for cloud provider interactions.

    Writes to a JSONL file so entries survive restarts. The log is never
    truncated by JARVIS; the user can clear it manually.
    """

    def __init__(self, path: Path | None = None, max_memory: int = 1000) -> None:
        self._path = path
        self._entries: list[AuditEntry] = []
        self._lock = Lock()
        self._max_memory = max_memory

    def record(self, event: AuditEventType, *, provider_id: str = "", model: str = "",
               detail: str = "", evidence: dict[str, Any] | None = None) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            event=event,
            provider_id=provider_id,
            model=model,
            detail=detail,
            evidence=evidence or {},
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_memory:
                self._entries = self._entries[-self._max_memory:]
        if self._path:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                log.exception("failed to write audit log entry")
        log.info("audit: %s provider=%s model=%s %s", event.value, provider_id, model, detail)
        return entry

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._entries[-n:]]

    def entries_for_provider(self, provider_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._entries if e.provider_id == provider_id]


# ── Policy check ────────────────────────────────────────────────────────────

SYNTHETIC_PROMPT = 'Reply with exactly: {"status":"ok","echo":"jarvis-verify"}'
SYNTHETIC_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}, "echo": {"type": "string"}},
    "required": ["status", "echo"],
}


def cloud_access_error(
    privacy_mode: str,
    provider_is_local: bool,
    consent_given: bool,
    provider_verified: bool = True,
) -> str | None:
    """Return an error message if a cloud request should be blocked, or None if allowed.

    This is the central policy gate. It is checked:
    - Before every chat request that would go to a cloud provider.
    - Both at request entry (in the API handler) and again immediately before
      the actual provider call (in case settings changed during preparation).

    Mirrors ChatGPT's cloud_chat_policy_error() with added verification check.
    """
    mode = PrivacyMode.from_str(privacy_mode)

    if provider_is_local:
        return None  # local providers are always allowed in any mode

    if mode == PrivacyMode.STRICT:
        return ("Strict privacy mode allows local providers only. "
                "Switch to a local provider or change to Guarded mode in Settings.")

    # mode == GUARDED
    if not provider_verified:
        return ("This cloud provider has not been verified yet. "
                "Run provider verification in Settings before sending requests.")

    if not consent_given:
        return ("This cloud request needs your confirmation before anything is sent. "
                "Nothing was transmitted to the provider.")

    return None


def consent_matches(consent_provider: str | None, consent_model: str | None,
                    active_provider: str, active_model: str) -> bool:
    """Check that the consent fields match the currently selected provider/model.

    Prevents a race where the user consents for provider A but the selection
    has changed to provider B by the time the message is processed.
    """
    return (consent_provider == active_provider and consent_model == active_model)


# ── Guarded mode controller ────────────────────────────────────────────────

class GuardedModeController:
    """Session-scoped controller for Guarded mode cloud access.

    Manages provider verification state and audit logging for one running
    instance of JARVIS. Instantiated once at startup alongside the LLM and
    SettingsStore.
    """

    def __init__(self, audit_path: Path | None = None) -> None:
        self.audit = AuditLog(path=audit_path)
        self._verifications: dict[str, ProviderVerification] = {}
        self._lock = Lock()

    def is_provider_verified(self, provider_id: str) -> bool:
        with self._lock:
            v = self._verifications.get(provider_id)
            return v is not None and v.verified

    def record_verification(self, provider_id: str, model: str, ok: bool,
                            latency_ms: int = 0, error: str = "") -> None:
        """Record the result of a synthetic verification probe."""
        with self._lock:
            self._verifications[provider_id] = ProviderVerification(
                provider_id=provider_id,
                verified=ok,
                verified_at=time.time() if ok else 0.0,
                model_used=model,
                latency_ms=latency_ms,
                error=error,
            )
        if ok:
            self.audit.record(
                AuditEventType.PROVIDER_VERIFY_OK,
                provider_id=provider_id, model=model,
                detail=f"verified in {latency_ms}ms",
                evidence={"latency_ms": latency_ms},
            )
        else:
            self.audit.record(
                AuditEventType.PROVIDER_VERIFY_FAIL,
                provider_id=provider_id, model=model,
                detail=error,
                evidence={"error": error},
            )

    def verification_status(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            v = self._verifications.get(provider_id)
            if v is None:
                return {"provider_id": provider_id, "verified": False}
            return {
                "provider_id": v.provider_id,
                "verified": v.verified,
                "verified_at": v.verified_at,
                "model_used": v.model_used,
                "latency_ms": v.latency_ms,
                "error": v.error,
            }

    def all_verification_statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.verification_status(pid) for pid in self._verifications]

    def check_access(self, privacy_mode: str, provider_id: str,
                     provider_is_local: bool, consent_given: bool) -> str | None:
        """Full access check: privacy mode + verification + consent.

        Returns an error message or None if access is allowed.
        Logs the decision to the audit trail.
        """
        verified = self.is_provider_verified(provider_id) if not provider_is_local else True

        error = cloud_access_error(
            privacy_mode=privacy_mode,
            provider_is_local=provider_is_local,
            consent_given=consent_given,
            provider_verified=verified,
        )

        if error:
            event = (AuditEventType.CLOUD_REQUEST_BLOCKED if "strict" in error.lower() or "verified" in error.lower()
                     else AuditEventType.CLOUD_CONSENT_MISSING)
            self.audit.record(event, provider_id=provider_id, detail=error)
        else:
            if not provider_is_local:
                self.audit.record(
                    AuditEventType.CLOUD_REQUEST_ALLOWED,
                    provider_id=provider_id,
                    detail="consent verified, provider verified",
                )

        return error

    def clear_session(self) -> None:
        """Reset verification state on session clear (new conversation)."""
        with self._lock:
            self._verifications.clear()
        self.audit.record(
            AuditEventType.PRIVACY_MODE_SET,
            detail="session cleared, verifications reset",
        )
