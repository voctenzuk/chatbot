"""Backward-compatible shim. Import from bot.infra.langfuse_service instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.infra.langfuse_service")
sys.modules[__name__] = _canonical
