"""Tests for system prompt service."""

from __future__ import annotations

from bot.services.system_prompt import DEFAULT_SYSTEM_PROMPT, get_system_prompt


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_default_system_prompt_not_empty(self):
        """Default system prompt must contain meaningful text."""
        assert DEFAULT_SYSTEM_PROMPT
        assert len(DEFAULT_SYSTEM_PROMPT) > 20

    def test_get_system_prompt_without_user_name(self):
        """get_system_prompt() without user_name returns base prompt only."""
        result = get_system_prompt()
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_get_system_prompt_with_user_name(self):
        """get_system_prompt(user_name=...) appends user name line."""
        result = get_system_prompt(user_name="Алиса")
        assert "Алиса" in result
        assert result.startswith(DEFAULT_SYSTEM_PROMPT)
        assert len(result) > len(DEFAULT_SYSTEM_PROMPT)
