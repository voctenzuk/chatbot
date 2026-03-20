"""Backward-compatible shim. Import from bot.infra.db_client instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.infra.db_client")
sys.modules[__name__] = _canonical
