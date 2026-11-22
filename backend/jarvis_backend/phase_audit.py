"""Phase-gated development with LLM-powered audit at every checkpoint.

Each JARVIS development phase is a sequence of tasks. Before a task is marked
complete, the local LLM audits the implementation against acceptance criteria.
No phase advances until every task in it passes audit.

The audit runs against the local model (Ollama / gateway) so no code or
context ever leaves the machine. Each audit result is logged to the guarded
audit trail and persisted as a JSONL record.

Usage from Claude Code or any AI agent working on JARVIS:

    from jarvis_backend.phase_audit import PHASES, PhaseRunner

    runner = PhaseRunner(llm, audit_log, repo_root="/path/to/jarvis")
    # Before moving to next task:
    result = await runner.audit_task("phase2", "guarded_policy_gate")
    if not result.passed:
        print(result.findings)  # fix these before proceeding
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.phase_audit")


# ── Phase & task definitions ────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AUDITING = "auditing"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AuditCriterion:
    """One thing the auditor must verify."""
    description: str
    check_type: str = "code_review"  # code_review | test_run | file_exists | grep_check
    target_files: list[str] = field(default_factory=list)  # repo-relative paths to inspect
    grep_pattern: str = ""  # for grep_check: pattern that must (or must not) appear
    grep_must_exist: bool = True  # True = pattern must be found; False = must NOT be found


@dataclass
class PhaseTask:
    """A single implementation task within a phase."""
    id: str
    name: str
    description: str
    acceptance_criteria: list[AuditCriterion] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # task IDs that must pass first
    status: TaskStatus = TaskStatus.PENDING
    audit_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "criteria_count": len(self.acceptance_criteria),
            "audit_result": self.audit_result,
        }


@dataclass
class Phase:
    """A development phase: a named group of tasks with a gate."""
    id: str
    name: str
    description: str
    tasks: list[PhaseTask] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return all(t.status == TaskStatus.PASSED for t in self.tasks)

    @property
    def progress(self) -> dict[str, int]:
        counts = {}
        for t in self.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "complete": self.complete, "progress": self.progress,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ── The JARVIS phase plan ──────────────────────────────────────────────────
# Each phase builds on the previous. Phases 0-2 are complete (Phase 0-1 from
# ChatGPT's working copy, Phase 2 ported to main repo). Phases 3+ are next.

PHASES: dict[str, Phase] = {
    "phase0": Phase(
        id="phase0", name="Phase 0: Baseline Review",
        description="Initial codebase audit. Verify project structure, dependencies, and basic functionality.",
        tasks=[
            PhaseTask("baseline_structure", "Repository structure audit",
                      "Verify the Tauri + React + Python backend structure is intact and buildable.",
                      status=TaskStatus.PASSED),
            PhaseTask("baseline_deps", "Dependency audit",
                      "All imports resolve, no circular dependencies, common requirements available.",
                      status=TaskStatus.PASSED),
        ],
    ),

    "phase1": Phase(
        id="phase1", name="Phase 1: Local Action Controls",
        description="Permission system, tool approval, and local-only operation.",
        tasks=[
            PhaseTask("permissions_system", "Capability-based permissions",
                      "permissions.py with Allow/Ask/Block per capability, anti-prompt-injection tracking.",
                      status=TaskStatus.PASSED),
            PhaseTask("tool_risk_levels", "Tool risk classification",
                      "Every tool has a risk level (auto/confirm/danger) enforced before execution.",
                      status=TaskStatus.PASSED),
        ],
    ),

    "phase2": Phase(
        id="phase2", name="Phase 2: Guarded Mode",
        description="Privacy-first cloud provider access with per-message consent and audit logging.",
        tasks=[
            PhaseTask("privacy_mode_setting", "Privacy mode in settings",
                      "privacy_mode added to settings.py DEFAULTS, PROFILE_FIELDS, validation, and public().",
                      acceptance_criteria=[
                          AuditCriterion("privacy_mode exists in DEFAULTS with default 'strict'",
                                         target_files=["backend/jarvis_backend/settings.py"],
                                         check_type="grep_check", grep_pattern='privacy_mode.*strict'),
                          AuditCriterion("privacy_mode is in PROFILE_FIELDS",
                                         target_files=["backend/jarvis_backend/settings.py"],
                                         check_type="grep_check", grep_pattern='privacy_mode'),
                          AuditCriterion("Validation rejects invalid privacy_mode values",
                                         target_files=["backend/jarvis_backend/settings.py"],
                                         check_type="code_review"),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("provider_is_local", "Provider local/cloud classification",
                      "Every provider preset has an is_local flag. gateway and ollama are local; rest are cloud.",
                      acceptance_criteria=[
                          AuditCriterion("is_local flag on every provider in PRESETS",
                                         target_files=["backend/jarvis_backend/providers/__init__.py"],
                                         check_type="grep_check", grep_pattern='is_local'),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("guarded_policy_gate", "Cloud access policy gate",
                      "cloud_access_error() checks privacy mode, verification, and consent. "
                      "Double-check pattern in stream_chat() and run_agent().",
                      acceptance_criteria=[
                          AuditCriterion("cloud_access_error function exists in guarded.py",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="grep_check", grep_pattern='def cloud_access_error'),
                          AuditCriterion("Double-check pattern: gate at entry AND before provider call",
                                         target_files=["backend/jarvis_backend/llm.py"],
                                         check_type="code_review"),
                          AuditCriterion("Strict mode blocks all cloud providers",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="code_review"),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("consent_validation", "Per-message consent validation",
                      "ChatIn has cloud_consent/cloud_provider/cloud_model fields. "
                      "API validates consent matches active provider/model.",
                      acceptance_criteria=[
                          AuditCriterion("cloud_consent field on ChatIn model",
                                         target_files=["backend/jarvis_backend/api.py"],
                                         check_type="grep_check", grep_pattern='cloud_consent.*bool'),
                          AuditCriterion("consent_matches() validates provider/model agreement",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="grep_check", grep_pattern='def consent_matches'),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("audit_logging", "Audit trail for all cloud interactions",
                      "AuditLog writes JSONL. Every cloud allow/block/verify event is recorded.",
                      acceptance_criteria=[
                          AuditCriterion("AuditLog class with JSONL persistence",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="grep_check", grep_pattern='class AuditLog'),
                          AuditCriterion("GET /audit endpoint exposes recent entries",
                                         target_files=["backend/jarvis_backend/api.py"],
                                         check_type="grep_check", grep_pattern='/audit'),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("verification_endpoint", "Provider verification endpoint",
                      "POST /provider/verify runs a synthetic probe. GET returns status.",
                      acceptance_criteria=[
                          AuditCriterion("POST /provider/verify endpoint exists",
                                         target_files=["backend/jarvis_backend/api.py"],
                                         check_type="grep_check", grep_pattern='/provider/verify'),
                          AuditCriterion("Synthetic probe uses non-sensitive test data",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="code_review"),
                      ],
                      status=TaskStatus.PASSED),
            PhaseTask("guarded_tests", "Guarded mode test suite",
                      "27+ tests covering policy gate, consent, audit log, and controller.",
                      acceptance_criteria=[
                          AuditCriterion("test_guarded.py exists with comprehensive tests",
                                         target_files=["backend/tests/test_guarded.py"],
                                         check_type="file_exists"),
                          AuditCriterion("Tests cover: strict blocks cloud, guarded needs consent+verification, local always allowed",
                                         target_files=["backend/tests/test_guarded.py"],
                                         check_type="code_review"),
                      ],
                      status=TaskStatus.PASSED),
        ],
    ),

    "phase3": Phase(
        id="phase3", name="Phase 3: External Transmission Controls",
        description="Block tools marked external_transmission in strict mode. "
                    "Integrate with the existing permissions system for cloud-aware tool gating.",
        tasks=[
            PhaseTask("tool_transmission_flag", "Flag tools that send data externally",
                      "Add external_transmission flag to ToolSpec. Mark tools that send data "
                      "to external services (web_research, send_email, etc.).",
                      acceptance_criteria=[
                          AuditCriterion("external_transmission field on ToolSpec",
                                         target_files=["backend/jarvis_backend/tools.py"],
                                         check_type="grep_check", grep_pattern='external_transmission'),
                          AuditCriterion("Relevant tools are flagged (web_research, send_email, etc.)",
                                         target_files=["backend/jarvis_backend/tools.py"],
                                         check_type="code_review"),
                      ],
                      depends_on=["guarded_policy_gate"]),
            PhaseTask("tool_gate_integration", "Gate external tools in strict mode",
                      "Before executing a tool flagged external_transmission, check privacy_mode. "
                      "In strict mode, block it. In guarded mode, require approval.",
                      acceptance_criteria=[
                          AuditCriterion("Tool execution checks external_transmission against privacy_mode",
                                         target_files=["backend/jarvis_backend/api.py"],
                                         check_type="code_review"),
                          AuditCriterion("Blocked tool calls are audit-logged",
                                         target_files=["backend/jarvis_backend/guarded.py"],
                                         check_type="code_review"),
                      ],
                      depends_on=["tool_transmission_flag"]),
            PhaseTask("permissions_privacy_merge", "Merge privacy mode with permissions system",
                      "send_data_out capability in permissions.py interacts with privacy mode. "
                      "Strict mode forces send_data_out to Block.",
                      acceptance_criteria=[
                          AuditCriterion("Privacy mode overrides permissions for send_data_out when strict",
                                         target_files=["backend/jarvis_backend/permissions.py"],
                                         check_type="code_review"),
                      ],
                      depends_on=["tool_gate_integration"]),
            PhaseTask("phase3_tests", "Phase 3 test suite",
                      "Tests for tool gating, permissions interaction, and audit logging of blocked tools.",
                      acceptance_criteria=[
                          AuditCriterion("Tests cover: strict blocks external tools, guarded requires approval",
                                         check_type="code_review"),
                      ],
                      depends_on=["permissions_privacy_merge"]),
        ],
    ),

    "phase4": Phase(
        id="phase4", name="Phase 4: Frontend Privacy UI",
        description="React UI for privacy mode selection, consent dialogs, verification status, "
                    "and audit log viewer.",
        tasks=[
            PhaseTask("privacy_settings_ui", "Privacy mode toggle in Settings",
                      "Settings panel shows strict/guarded toggle with explanation of each mode.",
                      depends_on=["guarded_tests"]),
            PhaseTask("consent_dialog", "Per-message consent dialog",
                      "When guarded mode is active and a cloud provider is selected, "
                      "each message shows a consent checkbox with provider/model identification.",
                      depends_on=["privacy_settings_ui"]),
            PhaseTask("verification_ui", "Provider verification UI",
                      "Button to run provider verification. Status indicator showing verified/unverified.",
                      depends_on=["privacy_settings_ui"]),
            PhaseTask("audit_viewer", "Audit log viewer",
                      "Panel showing recent cloud interaction audit entries with timestamps and details.",
                      depends_on=["verification_ui"]),
        ],
    ),

    "phase5": Phase(
        id="phase5", name="Phase 5: Hardening & Integration Testing",
        description="End-to-end testing, edge cases, and security hardening.",
        tasks=[
            PhaseTask("e2e_privacy_flow", "End-to-end privacy flow test",
                      "Full flow: set strict → verify cloud blocked → switch to guarded → "
                      "verify → consent → cloud request succeeds → audit shows all events.",
                      depends_on=["consent_dialog", "phase3_tests"]),
            PhaseTask("race_condition_audit", "Race condition protection",
                      "Verify double-check pattern prevents TOCTOU: settings change between "
                      "entry gate and provider call is caught.",
                      depends_on=["e2e_privacy_flow"]),
            PhaseTask("key_scrubbing", "API key scrubbing in audit log",
                      "Verify that API keys, tokens, and sensitive data never appear in audit entries.",
                      depends_on=["e2e_privacy_flow"]),
        ],
    ),
}


# ── Audit result ────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    task_id: str
    phase_id: str
    passed: bool
    timestamp: float = 0.0
    findings: list[str] = field(default_factory=list)
    model_used: str = ""
    criteria_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "phase_id": self.phase_id,
            "passed": self.passed, "timestamp": self.timestamp,
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "findings": self.findings, "model_used": self.model_used,
            "criteria_results": self.criteria_results,
        }


# ── Audit prompt templates ──────────────────────────────────────────────────

AUDIT_SYSTEM_PROMPT = """You are a code auditor for JARVIS, a Tauri + React + TypeScript + Python AI assistant.
Your job is to verify that implementation tasks meet their acceptance criteria.

For each criterion, respond with a JSON object:
{
  "criterion": "<description>",
  "passed": true/false,
  "finding": "<what you found — quote relevant code if failing>"
}

After all criteria, give a final verdict:
{
  "verdict": "PASS" or "FAIL",
  "summary": "<one sentence summary>",
  "blocking_issues": ["<issue 1>", "<issue 2>"]
}

Be strict. A criterion passes only if you can see concrete evidence in the code.
If code is missing, incomplete, or has obvious bugs, the criterion FAILS.
Do not speculate about what might work — only what you can verify from the provided code."""

AUDIT_TASK_PROMPT = """## Audit Task: {task_name}

**Phase:** {phase_name}
**Description:** {task_description}

### Acceptance Criteria to Verify:
{criteria_list}

### Code to Review:
{code_sections}

### Instructions:
1. For each criterion, examine the provided code carefully.
2. Output one JSON object per criterion with passed/finding.
3. End with a final verdict JSON object.
4. Be thorough — check for edge cases, missing validation, and security issues."""


# ── Phase runner ────────────────────────────────────────────────────────────

class PhaseRunner:
    """Runs audit checks using the local LLM against implementation tasks.

    Designed to be used by Claude Code or any agent implementing JARVIS phases.
    The audit always runs on the local model so no source code leaves the machine.
    """

    def __init__(self, repo_root: str | Path, audit_path: Path | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.audit_path = audit_path
        self._results: list[AuditResult] = []

    def phase(self, phase_id: str) -> Phase | None:
        return PHASES.get(phase_id)

    def all_phases(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in PHASES.values()]

    def next_tasks(self, phase_id: str) -> list[PhaseTask]:
        """Return tasks in this phase that are ready to work on (dependencies met, not yet passed)."""
        phase = PHASES.get(phase_id)
        if not phase:
            return []
        # Collect all passed task IDs across ALL phases (deps can be cross-phase)
        all_passed: set[str] = set()
        for p in PHASES.values():
            for t in p.tasks:
                if t.status == TaskStatus.PASSED:
                    all_passed.add(t.id)
        return [t for t in phase.tasks
                if t.status not in (TaskStatus.PASSED, TaskStatus.SKIPPED)
                and all(d in all_passed for d in t.depends_on)]

    def _read_file(self, repo_relative: str) -> str:
        """Read a file from the repo, returning its content or an error message."""
        path = self.repo_root / repo_relative
        if not path.exists():
            return f"[FILE NOT FOUND: {repo_relative}]"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[ERROR READING {repo_relative}: {e}]"

    def build_audit_prompt(self, phase_id: str, task_id: str) -> tuple[str, str] | None:
        """Build the system and user prompts for auditing a task.

        Returns (system_prompt, user_prompt) or None if task not found.
        """
        phase = PHASES.get(phase_id)
        if not phase:
            return None
        task = next((t for t in phase.tasks if t.id == task_id), None)
        if not task:
            return None

        # Build criteria list
        criteria_lines = []
        for i, c in enumerate(task.acceptance_criteria, 1):
            criteria_lines.append(f"{i}. [{c.check_type}] {c.description}")
            if c.target_files:
                criteria_lines.append(f"   Files: {', '.join(c.target_files)}")
            if c.grep_pattern:
                criteria_lines.append(f"   Pattern: `{c.grep_pattern}` (must {'exist' if c.grep_must_exist else 'NOT exist'})")

        # Gather code sections
        all_files: set[str] = set()
        for c in task.acceptance_criteria:
            all_files.update(c.target_files)

        code_parts = []
        for f in sorted(all_files):
            content = self._read_file(f)
            code_parts.append(f"### {f}\n```python\n{content}\n```")

        user_prompt = AUDIT_TASK_PROMPT.format(
            task_name=task.name,
            phase_name=phase.name,
            task_description=task.description,
            criteria_list="\n".join(criteria_lines) or "(no specific criteria — do a general code review)",
            code_sections="\n\n".join(code_parts) or "(no specific files — audit based on description)",
        )
        return AUDIT_SYSTEM_PROMPT, user_prompt

    def record_result(self, result: AuditResult) -> None:
        """Record an audit result and update the task status."""
        self._results.append(result)
        phase = PHASES.get(result.phase_id)
        if phase:
            task = next((t for t in phase.tasks if t.id == result.task_id), None)
            if task:
                task.status = TaskStatus.PASSED if result.passed else TaskStatus.FAILED
                task.audit_result = result.to_dict()

        if self.audit_path:
            try:
                with open(self.audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                log.exception("failed to write phase audit result")

    def results(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._results]

    def status_report(self) -> dict[str, Any]:
        """Full status of all phases and their tasks."""
        return {
            "phases": self.all_phases(),
            "audit_results": len(self._results),
            "current_phase": next(
                (p.id for p in PHASES.values() if not p.complete), None
            ),
        }


# ── Claude Code integration instructions ────────────────────────────────────

CLAUDE_CODE_INSTRUCTIONS = """
# JARVIS Phase Development Protocol

## Before starting any task:
1. Run `PhaseRunner.next_tasks(phase_id)` to find tasks ready to implement
2. Check task.depends_on — blocked tasks cannot start until dependencies pass audit
3. Read the task description and acceptance criteria carefully

## While implementing:
- Write code that satisfies every acceptance criterion
- Add tests for each new behavior
- Keep changes focused on the current task — don't drift into future tasks

## Before marking a task complete:
1. Run all existing tests to verify no regressions
2. Build the audit prompt: `runner.build_audit_prompt(phase_id, task_id)`
3. Send the prompt to the LOCAL model (Ollama or gateway) — never cloud
4. Parse the model's response for pass/fail on each criterion
5. If ANY criterion fails:
   a. Fix the issue
   b. Re-run the audit
   c. Repeat until all criteria pass
6. Record the result: `runner.record_result(result)`
7. Only then move to the next task

## Phase gate:
- A phase is complete ONLY when all its tasks have status PASSED
- Do not start tasks from the next phase until the current phase is complete
- Record all audit results in the audit log

## Audit model selection:
- ALWAYS use the local model for audits (privacy: source code stays on-machine)
- If no local model is available, flag this and do NOT send code to a cloud provider
"""
