"""Backward-compatible shim. Import from bot.memory.cognee_service instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.memory.cognee_service")
sys.modules[__name__] = _canonical
