"""Backward-compatible shim. Import from bot.conversation.episode_switcher instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.conversation.episode_switcher")
sys.modules[__name__] = _canonical
