"""Backward-compatible shim. Import from bot.memory.models instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.memory.models")
sys.modules[__name__] = _canonical
