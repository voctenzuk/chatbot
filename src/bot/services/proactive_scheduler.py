"""Backward-compatible shim. Import from bot.adapters.proactive_scheduler instead."""

import importlib
import sys

_canonical = importlib.import_module("bot.adapters.proactive_scheduler")
sys.modules[__name__] = _canonical
