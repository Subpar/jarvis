"""Phase 3 tests: external transmission controls and privacy-mode tool gating."""
from __future__ import annotations

import pytest

from jarvis_backend.tools import BY_NAME, EXTERNAL_TOOLS, TOOLS, ToolSpec
from jarvis_backend.permissions import decide, SENDS_OUT


# ── Task 1: external_transmission flag ──────────────────────────────────────

class TestExternalTransmissionFlag:
    def test_flag_exists_on_toolspec(self):
        spec = ToolSpec("test", "test", {})
        assert hasattr(spec, "external_transmission")
        assert spec.external_transmission is False

    def test_sends_out_tools_are_flagged(self):
        """Every tool in the existing SENDS_OUT set should have external_transmission=True."""
        for name in SENDS_OUT:
            spec = BY_NAME.get(name)
            if spec:
                assert spec.external_transmission, f"{name} is in SENDS_OUT but not flagged external_transmission"

    def test_local_only_tools_not_flagged(self):
        """Read-only local tools should NOT have external_transmission."""
        local_tools = ["system_status", "read_file", "clipboard_get", "list_folder",
                        "find_files", "read_window", "list_windows"]
        for name in local_tools:
            spec = BY_NAME.get(name)
            if spec:
                assert not spec.external_transmission, f"{name} should not be flagged external_transmission"

    def test_external_tools_set_matches_flag(self):
        """EXTERNAL_TOOLS convenience set matches tools with external_transmission=True."""
        expected = {t.name for t in TOOLS if t.external_transmission}
        assert EXTERNAL_TOOLS == expected

    def test_web_research_flagged(self):
        assert BY_NAME["web_research"].external_transmission is True

    def test_maps_search_flagged(self):
        assert BY_NAME["maps_search"].external_transmission is True

    def test_deep_research_flagged(self):
        assert BY_NAME["deep_research"].external_transmission is True

    def test_at_least_8_tools_flagged(self):
        """We expect a meaningful number of tools to be flagged."""
        assert len(EXTERNAL_TOOLS) >= 8


# ── Task 2: tool gating in strict and guarded modes ────────────────────────

class TestToolGatingStrict:
    """In strict mode, all external_transmission tools must be blocked."""

    def test_strict_blocks_external_tool(self):
        spec = BY_NAME["web_research"]
        result, cap = decide(spec, {"query": "test"}, privacy_mode="strict")
        assert result == "block"
        assert cap == "send_data_out"

    def test_strict_blocks_call_api(self):
        spec = BY_NAME["call_api"]
        result, cap = decide(spec, {"url": "https://api.example.com"}, privacy_mode="strict")
        assert result == "block"

    def test_strict_blocks_ask_agent(self):
        spec = BY_NAME["ask_agent"]
        result, cap = decide(spec, {"url": "http://agent.local", "text": "hi"}, privacy_mode="strict")
        assert result == "block"

    def test_strict_allows_local_tool(self):
        """Local-only tools (like read_file) are unaffected by strict mode."""
        spec = BY_NAME["read_file"]
        result, cap = decide(spec, {"path": "/home/user/test.py"}, privacy_mode="strict")
        assert result == "allow"

    def test_strict_allows_system_status(self):
        spec = BY_NAME["system_status"]
        result, _ = decide(spec, {}, privacy_mode="strict")
        assert result == "allow"


class TestToolGatingGuarded:
    """In guarded mode, external_transmission tools are forced to at least Ask."""

    def test_guarded_forces_ask_on_external_tool(self):
        spec = BY_NAME["web_research"]
        result, cap = decide(spec, {"query": "test"}, privacy_mode="guarded")
        assert result == "ask"
        assert cap == "send_data_out"

    def test_guarded_forces_ask_even_with_full_control(self):
        """Full control doesn't bypass guarded mode for external tools."""
        spec = BY_NAME["call_api"]
        result, cap = decide(spec, {"url": "https://api.example.com"},
                              privacy_mode="guarded", full_control=True)
        assert result == "ask"

    def test_guarded_allows_local_tool(self):
        spec = BY_NAME["read_file"]
        result, _ = decide(spec, {"path": "/test.py"}, privacy_mode="guarded")
        assert result == "allow"

    def test_guarded_respects_explicit_block(self):
        """If the user explicitly blocked send_data_out, guarded doesn't downgrade to ask."""
        spec = BY_NAME["web_research"]
        result, cap = decide(spec, {"query": "test"},
                              saved={"send_data_out": "block"}, privacy_mode="guarded")
        assert result == "block"


# ── Task 3: privacy mode overrides permissions for send_data_out ────────────

class TestPrivacyModePermissionsInteraction:
    def test_strict_overrides_allow_policy(self):
        """Even with send_data_out set to allow, strict mode blocks."""
        spec = BY_NAME["open_url"]
        result, _ = decide(spec, {"url": "https://example.com"},
                            saved={"send_data_out": "allow"}, privacy_mode="strict")
        assert result == "block"

    def test_normal_non_external_tool_unaffected(self):
        """Non-external tools keep their normal permission logic."""
        spec = BY_NAME["write_file"]
        result, cap = decide(spec, {"path": "test.py", "content": "hi"},
                              privacy_mode="strict")
        # write_file → change_files capability, normally "ask"
        assert result == "ask"
        assert cap == "change_files"

    def test_backward_compatible_without_privacy_mode(self):
        """Default privacy_mode is 'strict' — same behavior as before for tools without
        external_transmission (they're not affected by the gate)."""
        spec = BY_NAME["system_status"]
        result, _ = decide(spec, {})
        assert result == "allow"
