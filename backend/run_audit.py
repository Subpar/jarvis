#!/usr/bin/env python3
"""Run the JARVIS phase audit against the local Ollama model.

Usage:
    python run_audit.py                       # Audit all phases
    python run_audit.py --phase phase2        # Audit one phase
    python run_audit.py --phase phase2 --task guarded_tests  # Audit one task
    python run_audit.py --model qwen3-8b-32k  # Use a different model
    python run_audit.py --list                # List all phases and task statuses
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_backend.phase_audit import PHASES, PhaseRunner


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = REPO_ROOT / "backend" / "audit_results.jsonl"
DEFAULT_MODEL = "qwen2.5-coder-32k"


def list_phases():
    """Print the status of all phases and tasks."""
    for pid, phase in PHASES.items():
        status = "✅ COMPLETE" if phase.complete else "⏳ IN PROGRESS"
        print(f"\n{'='*60}")
        print(f"  {phase.name}  [{status}]")
        print(f"  {phase.description[:80]}...")
        print(f"{'='*60}")
        for task in phase.tasks:
            icon = {"passed": "✅", "failed": "❌", "pending": "⏳",
                    "in_progress": "🔄", "auditing": "🔍", "skipped": "⏭️"
                    }.get(task.status, "?")
            deps = f" (needs: {', '.join(task.depends_on)})" if task.depends_on else ""
            print(f"  {icon} {task.id}: {task.name}{deps}")
            for c in task.acceptance_criteria:
                print(f"      └─ [{c.check_type}] {c.description}")


async def run_audit(phase_id: str | None, task_id: str | None, model: str):
    """Run the audit for specified phase/task or all phases."""
    runner = PhaseRunner(repo_root=REPO_ROOT, audit_path=AUDIT_LOG)

    # Verify Ollama is reachable
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"✅ Ollama is running. Available models: {', '.join(models)}")
            if model not in models and f"{model}:latest" not in models:
                # Try with :latest suffix
                model_check = model if ":" in model else f"{model}:latest"
                if model_check not in models:
                    print(f"⚠️  Model '{model}' not found. Available: {', '.join(models)}")
                    return
    except Exception as e:
        print(f"❌ Cannot reach Ollama at localhost:11434: {e}")
        print("   Make sure Ollama is running (check system tray).")
        return

    # Determine which tasks to audit
    tasks_to_audit: list[tuple[str, str]] = []

    if phase_id and task_id:
        tasks_to_audit.append((phase_id, task_id))
    elif phase_id:
        phase = PHASES.get(phase_id)
        if not phase:
            print(f"❌ Phase '{phase_id}' not found.")
            return
        for task in phase.tasks:
            tasks_to_audit.append((phase_id, task.id))
    else:
        for pid, phase in PHASES.items():
            for task in phase.tasks:
                tasks_to_audit.append((pid, task.id))

    print(f"\n🔍 Auditing {len(tasks_to_audit)} task(s) with model '{model}'...\n")

    passed = 0
    failed = 0

    for pid, tid in tasks_to_audit:
        phase = PHASES[pid]
        task = next(t for t in phase.tasks if t.id == tid)
        print(f"{'─'*50}")
        print(f"📋 {phase.name} → {task.name}")

        if not task.acceptance_criteria:
            print(f"   ⏭️  No acceptance criteria — skipping")
            continue

        result = await runner.audit_task(pid, tid, model=model)

        if result.passed:
            print(f"   ✅ PASSED")
            passed += 1
        else:
            print(f"   ❌ FAILED")
            failed += 1
            for finding in result.findings:
                print(f"      • {finding[:120]}")

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"  Audit log: {AUDIT_LOG}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="JARVIS Phase Audit Runner")
    parser.add_argument("--phase", help="Phase ID to audit (e.g. phase2)")
    parser.add_argument("--task", help="Task ID to audit (requires --phase)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--list", action="store_true", help="List phases and tasks")
    args = parser.parse_args()

    if args.list:
        list_phases()
        return

    if args.task and not args.phase:
        print("❌ --task requires --phase")
        sys.exit(1)

    asyncio.run(run_audit(args.phase, args.task, args.model))


if __name__ == "__main__":
    main()
