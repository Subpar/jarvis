"""Tests for the phase-gated audit system."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from jarvis_backend.phase_audit import (
    PHASES,
    AuditCriterion,
    AuditResult,
    Phase,
    PhaseRunner,
    PhaseTask,
    TaskStatus,
)


# ── TaskStatus ──────────────────────────────────────────────────────────────

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.AUDITING == "auditing"
        assert TaskStatus.PASSED == "passed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.SKIPPED == "skipped"


# ── PhaseTask ───────────────────────────────────────────────────────────────

class TestPhaseTask:
    def test_to_dict(self):
        task = PhaseTask("t1", "Test Task", "A test task",
                         acceptance_criteria=[AuditCriterion("check something")],
                         depends_on=["t0"])
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["name"] == "Test Task"
        assert d["status"] == "pending"
        assert d["depends_on"] == ["t0"]
        assert d["criteria_count"] == 1
        assert d["audit_result"] is None

    def test_default_status_is_pending(self):
        task = PhaseTask("t1", "Task", "Desc")
        assert task.status == TaskStatus.PENDING


# ── Phase ───────────────────────────────────────────────────────────────────

class TestPhase:
    def test_complete_all_passed(self):
        phase = Phase("p1", "Phase 1", "Desc", tasks=[
            PhaseTask("t1", "T1", "D1", status=TaskStatus.PASSED),
            PhaseTask("t2", "T2", "D2", status=TaskStatus.PASSED),
        ])
        assert phase.complete is True

    def test_not_complete_with_pending(self):
        phase = Phase("p1", "Phase 1", "Desc", tasks=[
            PhaseTask("t1", "T1", "D1", status=TaskStatus.PASSED),
            PhaseTask("t2", "T2", "D2", status=TaskStatus.PENDING),
        ])
        assert phase.complete is False

    def test_progress_counts(self):
        phase = Phase("p1", "Phase 1", "Desc", tasks=[
            PhaseTask("t1", "T1", "D1", status=TaskStatus.PASSED),
            PhaseTask("t2", "T2", "D2", status=TaskStatus.PENDING),
            PhaseTask("t3", "T3", "D3", status=TaskStatus.FAILED),
        ])
        prog = phase.progress
        assert prog == {"passed": 1, "pending": 1, "failed": 1}

    def test_to_dict(self):
        phase = Phase("p1", "Phase 1", "Desc", tasks=[
            PhaseTask("t1", "T1", "D1", status=TaskStatus.PASSED),
        ])
        d = phase.to_dict()
        assert d["id"] == "p1"
        assert d["complete"] is True
        assert len(d["tasks"]) == 1


# ── AuditCriterion ──────────────────────────────────────────────────────────

class TestAuditCriterion:
    def test_defaults(self):
        c = AuditCriterion("check something")
        assert c.check_type == "code_review"
        assert c.target_files == []
        assert c.grep_pattern == ""
        assert c.grep_must_exist is True

    def test_grep_check(self):
        c = AuditCriterion("no debug prints", check_type="grep_check",
                           target_files=["src/main.py"],
                           grep_pattern="print\\(", grep_must_exist=False)
        assert c.grep_must_exist is False


# ── AuditResult ─────────────────────────────────────────────────────────────

class TestAuditResult:
    def test_to_dict(self):
        r = AuditResult("t1", "p1", passed=True, timestamp=1700000000.0,
                         findings=["all good"], model_used="ollama/qwen2.5")
        d = r.to_dict()
        assert d["passed"] is True
        assert d["task_id"] == "t1"
        assert d["phase_id"] == "p1"
        assert d["model_used"] == "ollama/qwen2.5"
        assert "time_iso" in d

    def test_failed_result(self):
        r = AuditResult("t2", "p1", passed=False,
                         findings=["missing validation", "no tests"])
        d = r.to_dict()
        assert d["passed"] is False
        assert len(d["findings"]) == 2


# ── PHASES constant ─────────────────────────────────────────────────────────

class TestPhasePlan:
    def test_phase_ids_are_ordered(self):
        ids = list(PHASES.keys())
        assert ids == ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5"]

    def test_phase0_through_2_are_complete(self):
        for pid in ("phase0", "phase1", "phase2"):
            assert PHASES[pid].complete, f"{pid} should be marked complete"

    def test_phase3_is_not_complete(self):
        assert not PHASES["phase3"].complete

    def test_all_tasks_have_ids(self):
        for phase in PHASES.values():
            for task in phase.tasks:
                assert task.id, f"Task in {phase.id} missing id"
                assert task.name, f"Task {task.id} missing name"

    def test_dependencies_reference_valid_tasks(self):
        """Every depends_on value must refer to a task that exists somewhere."""
        all_task_ids = set()
        for phase in PHASES.values():
            for task in phase.tasks:
                all_task_ids.add(task.id)

        for phase in PHASES.values():
            for task in phase.tasks:
                for dep in task.depends_on:
                    assert dep in all_task_ids, (
                        f"Task {task.id} depends on {dep} which doesn't exist"
                    )

    def test_phase3_has_external_transmission_tasks(self):
        p3 = PHASES["phase3"]
        task_ids = [t.id for t in p3.tasks]
        assert "tool_transmission_flag" in task_ids
        assert "tool_gate_integration" in task_ids


# ── PhaseRunner ─────────────────────────────────────────────────────────────

class TestPhaseRunner:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.runner = PhaseRunner(repo_root=self.tmp)

    def test_all_phases(self):
        phases = self.runner.all_phases()
        assert len(phases) == 6
        assert phases[0]["id"] == "phase0"

    def test_next_tasks_phase3(self):
        """Phase 3 task 1 has no unmet deps (its dep is in phase2 which is passed)."""
        tasks = self.runner.next_tasks("phase3")
        assert len(tasks) >= 1
        assert tasks[0].id == "tool_transmission_flag"

    def test_next_tasks_respects_dependencies(self):
        """tool_gate_integration depends on tool_transmission_flag, so it shouldn't be in next_tasks."""
        tasks = self.runner.next_tasks("phase3")
        task_ids = [t.id for t in tasks]
        assert "tool_gate_integration" not in task_ids

    def test_next_tasks_invalid_phase(self):
        assert self.runner.next_tasks("nonexistent") == []

    def test_read_file_missing(self):
        content = self.runner._read_file("nonexistent.py")
        assert "FILE NOT FOUND" in content

    def test_read_file_exists(self):
        p = Path(self.tmp) / "test.py"
        p.write_text("print('hello')")
        content = self.runner._read_file("test.py")
        assert "print('hello')" in content

    def test_build_audit_prompt(self):
        result = self.runner.build_audit_prompt("phase3", "tool_transmission_flag")
        assert result is not None
        system, user = result
        assert "code auditor" in system.lower()
        assert "external_transmission" in user

    def test_build_audit_prompt_invalid(self):
        assert self.runner.build_audit_prompt("bad", "bad") is None

    def test_record_result_updates_task_status(self):
        result = AuditResult("tool_transmission_flag", "phase3", passed=True,
                              timestamp=1700000000.0)
        self.runner.record_result(result)
        task = next(t for t in PHASES["phase3"].tasks if t.id == "tool_transmission_flag")
        assert task.status == TaskStatus.PASSED
        # Reset for other tests
        task.status = TaskStatus.PENDING
        task.audit_result = None

    def test_record_result_persists_to_file(self):
        audit_path = Path(self.tmp) / "audit.jsonl"
        runner = PhaseRunner(repo_root=self.tmp, audit_path=audit_path)
        result = AuditResult("tool_transmission_flag", "phase3", passed=False,
                              timestamp=1700000000.0, findings=["missing flag"])
        runner.record_result(result)
        # Reset
        task = next(t for t in PHASES["phase3"].tasks if t.id == "tool_transmission_flag")
        task.status = TaskStatus.PENDING
        task.audit_result = None

        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["passed"] is False
        assert "missing flag" in entry["findings"]

    def test_status_report(self):
        report = self.runner.status_report()
        assert "phases" in report
        assert report["current_phase"] == "phase3"  # first incomplete phase

    def test_results_starts_empty(self):
        assert self.runner.results() == []
