"""Backward-compatible shim. Import from bot.media.artifact_service instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.media.artifact_service")
sys.modules[__name__] = _canonical
