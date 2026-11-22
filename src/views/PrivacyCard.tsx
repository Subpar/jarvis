/**
 * PrivacyCard — Privacy mode toggle, provider verification, and audit log viewer.
 *
 * Shows in the Settings "PC control & safety" tab alongside PermissionsCard.
 * Implements Phase 4 tasks: privacy_settings_ui, verification_ui, audit_viewer.
 */
import { useCallback, useEffect, useState } from "react";
import type { Connection } from "../shell";
import type { AuditEntry, Settings, VerificationResult } from "../api";
import { getAuditLog, getVerificationStatus, putSettings, verifyProvider } from "../api";

// ── Privacy mode toggle ─────────────────────────────────────────────────────

function ModeToggle({ mode, onChange, busy }: { mode: "strict" | "guarded"; onChange: (m: "strict" | "guarded") => void; busy: boolean }) {
  return (
    <div className="privacy-mode-toggle">
      <h3 style={{ margin: "0 0 6px" }}>Privacy mode</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button
          className={`privacy-btn ${mode === "strict" ? "active" : ""}`}
          disabled={busy}
          onClick={() => onChange("strict")}
          title="Only local models — nothing leaves this machine"
        >
          🔒 Strict
        </button>
        <button
          className={`privacy-btn ${mode === "guarded" ? "active" : ""}`}
          disabled={busy}
          onClick={() => onChange("guarded")}
          title="Cloud allowed with per-message consent and verification"
        >
          🛡️ Guarded
        </button>
      </div>
      <p className="privacy-desc" style={{ margin: 0, fontSize: "0.85em", opacity: 0.8 }}>
        {mode === "strict"
          ? "Local only — all data stays on this machine. Cloud providers are blocked."
          : "Cloud providers allowed with per-message consent. Each request requires explicit approval and provider verification."}
      </p>
    </div>
  );
}

// ── Provider verification ───────────────────────────────────────────────────

interface VerifyStatus {
  provider_id: string;
  model: string;
  verified: boolean;
  last_check: number;
  latency_ms?: number;
}

function VerificationPanel({ conn }: { conn: Connection }) {
  const [statuses, setStatuses] = useState<VerifyStatus[]>([]);
  const [verifying, setVerifying] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await getVerificationStatus(conn);
      setStatuses(r.verifications);
    } catch { /* guarded controller may not be initialised */ }
  }, [conn]);

  useEffect(() => { load(); }, [load]);

  const runVerify = async () => {
    setVerifying(true);
    setResult(null);
    try {
      const r = await verifyProvider(conn);
      setResult(r);
      await load(); // refresh statuses
    } catch (e) {
      setResult({ provider_id: "?", verified: false, detail: (e as Error).message });
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="verification-panel" style={{ marginTop: 12 }}>
      <h3 style={{ margin: "0 0 6px" }}>Provider verification</h3>
      <button disabled={verifying} onClick={runVerify} className="privacy-btn">
        {verifying ? "Verifying…" : "▶ Verify current provider"}
      </button>
      {result && (
        <div className={`verify-result ${result.verified ? "ok" : "fail"}`} style={{ marginTop: 6, fontSize: "0.85em" }}>
          {result.verified ? "✓ Verified" : "✕ Failed"}{" "}
          {result.provider_id !== "?" && `(${result.provider_id})`}{" "}
          {result.latency_ms !== undefined && `— ${result.latency_ms}ms`}
          {result.detail && !result.verified && <div style={{ opacity: 0.7, marginTop: 2 }}>{result.detail}</div>}
        </div>
      )}
      {statuses.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: "0.8em", opacity: 0.6, marginBottom: 4 }}>Recent verifications</div>
          {statuses.map((s, i) => (
            <div key={i} style={{ fontSize: "0.82em", padding: "2px 0" }}>
              {s.verified ? "✓" : "✕"} {s.provider_id} / {s.model}
              {s.latency_ms !== undefined && ` (${s.latency_ms}ms)`}
              {" — "}
              {new Date(s.last_check * 1000).toLocaleTimeString()}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Audit log viewer ────────────────────────────────────────────────────────

function AuditLogViewer({ conn }: { conn: Connection }) {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await getAuditLog(conn, 100);
      setEntries(r.entries);
    } catch { /* not available */ }
    setLoading(false);
  }, [conn]);

  useEffect(() => { load(); }, [load]);

  const LABELS: Record<string, string> = {
    cloud_access_blocked: "🚫 Cloud blocked",
    cloud_consent_given: "✓ Consent given",
    cloud_request_sent: "→ Cloud request",
    provider_verify_start: "🔍 Verify start",
    provider_verify_pass: "✓ Verify pass",
    provider_verify_fail: "✕ Verify fail",
    tool_blocked_strict: "🔒 Tool blocked (strict)",
    tool_gated_guarded: "🛡️ Tool gated (guarded)",
  };

  if (entries.length === 0 && !loading) {
    return (
      <div style={{ marginTop: 12 }}>
        <h3 style={{ margin: "0 0 6px" }}>Audit log</h3>
        <p style={{ fontSize: "0.85em", opacity: 0.6 }}>No privacy-related events recorded yet.</p>
      </div>
    );
  }

  return (
    <div className="audit-log" style={{ marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>Audit log</h3>
        <button onClick={load} disabled={loading} className="privacy-btn" style={{ fontSize: "0.75em", padding: "2px 8px" }}>
          {loading ? "…" : "↻ Refresh"}
        </button>
      </div>
      <div style={{ maxHeight: 240, overflowY: "auto", marginTop: 6 }}>
        {entries.map((e, i) => (
          <div key={i} style={{ fontSize: "0.82em", padding: "3px 0", borderBottom: "1px solid var(--border, rgba(255,255,255,0.08))" }}>
            <span style={{ opacity: 0.5 }}>{new Date(e.timestamp * 1000).toLocaleString()}</span>{" "}
            <span>{LABELS[e.event_type] ?? e.event_type}</span>
            {e.provider_id && <span style={{ opacity: 0.6 }}> — {e.provider_id}</span>}
            {e.model && <span style={{ opacity: 0.5 }}> ({e.model})</span>}
            {e.detail && <div style={{ opacity: 0.6, fontSize: "0.9em", marginLeft: 8 }}>{e.detail}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main card ───────────────────────────────────────────────────────────────

export default function PrivacyCard({ conn, settings, onChanged }: {
  conn: Connection;
  settings: Settings;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);

  const setMode = async (mode: "strict" | "guarded") => {
    if (mode === settings.privacy_mode) return;
    setBusy(true);
    try {
      await putSettings(conn, { privacy_mode: mode });
      onChanged();
    } catch (e) {
      console.error("Failed to set privacy mode:", e);
    }
    setBusy(false);
  };

  return (
    <div className="card privacy-card">
      <ModeToggle mode={settings.privacy_mode} onChange={setMode} busy={busy} />
      {settings.privacy_mode === "guarded" && <VerificationPanel conn={conn} />}
      <AuditLogViewer conn={conn} />
    </div>
  );
}
