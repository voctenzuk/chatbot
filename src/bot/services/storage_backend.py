"""Backward-compatible shim. Import from bot.media.storage_backend instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.media.storage_backend")
sys.modules[__name__] = _canonical
