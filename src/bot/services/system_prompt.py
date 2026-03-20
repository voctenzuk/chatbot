"""Backward-compatible shim. Import from bot.conversation.system_prompt instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.conversation.system_prompt")
sys.modules[__name__] = _canonical
