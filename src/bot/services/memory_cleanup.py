"""Backward-compatible shim. Import from bot.memory.cleanup instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.memory.cleanup")
sys.modules[__name__] = _canonical
