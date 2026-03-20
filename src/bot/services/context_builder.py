"""Backward-compatible shim. Import from bot.conversation.context_builder instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.conversation.context_builder")
sys.modules[__name__] = _canonical
