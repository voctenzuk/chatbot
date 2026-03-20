"""Backward-compatible shim. Import from bot.llm.service instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.llm.service")
sys.modules[__name__] = _canonical
