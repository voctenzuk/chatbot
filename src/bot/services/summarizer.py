"""Backward-compatible shim. Import from bot.conversation.summarizer instead."""

import importlib
import sys

# Make this module an alias for the canonical module so patches work
_canonical = importlib.import_module("bot.conversation.summarizer")
sys.modules[__name__] = _canonical
