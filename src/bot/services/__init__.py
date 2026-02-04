"""Services module for bot functionality."""

try:
    from bot.services.mem0_memory_service import (
        Mem0MemoryService,
        get_memory_service as get_mem0_memory_service,
        set_memory_service as set_mem0_memory_service,
    )
except ImportError:
    Mem0MemoryService = None  # type: ignore
    get_mem0_memory_service = None  # type: ignore
    set_mem0_memory_service = None  # type: ignore

try:
    from bot.services.memory_models import (
        MemoryCategory,
        MemoryFact,
        MemoryType,
    )
except ImportError:
    MemoryCategory = None  # type: ignore
    MemoryFact = None  # type: ignore
    MemoryType = None  # type: ignore

__all__ = [
    # Mem0 Memory Service
    "Mem0MemoryService",
    "get_mem0_memory_service",
    "set_mem0_memory_service",
    # Memory Models
    "MemoryCategory",
    "MemoryFact",
    "MemoryType",
]
