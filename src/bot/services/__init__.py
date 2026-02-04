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

try:
    from bot.services.context_builder import (
        ContextBuilder,
        ContextAssemblyConfig,
        ContextPart,
        ConversationMessage,
        MessageRole,
        RunningSummary,
        get_context_builder,
        set_context_builder,
    )
except ImportError:
    ContextBuilder = None  # type: ignore
    ContextAssemblyConfig = None  # type: ignore
    ContextPart = None  # type: ignore
    ConversationMessage = None  # type: ignore
    MessageRole = None  # type: ignore
    RunningSummary = None  # type: ignore
    get_context_builder = None  # type: ignore
    set_context_builder = None  # type: ignore

try:
    from bot.services.summarizer import (
        Summarizer,
        SummarizerConfig,
        SummaryKind,
        SummaryResult,
        SummaryJSON,
        FactCandidate,
        ArtifactReference,
        get_summarizer,
        set_summarizer,
    )
except ImportError:
    Summarizer = None  # type: ignore
    SummarizerConfig = None  # type: ignore
    SummaryKind = None  # type: ignore
    SummaryResult = None  # type: ignore
    SummaryJSON = None  # type: ignore
    FactCandidate = None  # type: ignore
    ArtifactReference = None  # type: ignore
    get_summarizer = None  # type: ignore
    set_summarizer = None  # type: ignore

__all__ = [
    # Mem0 Memory Service
    "Mem0MemoryService",
    "get_mem0_memory_service",
    "set_mem0_memory_service",
    # Memory Models
    "MemoryCategory",
    "MemoryFact",
    "MemoryType",
    # Context Builder
    "ContextBuilder",
    "ContextAssemblyConfig",
    "ContextPart",
    "ConversationMessage",
    "MessageRole",
    "RunningSummary",
    "get_context_builder",
    "set_context_builder",
    # Summarizer
    "Summarizer",
    "SummarizerConfig",
    "SummaryKind",
    "SummaryResult",
    "SummaryJSON",
    "FactCandidate",
    "ArtifactReference",
    "get_summarizer",
    "set_summarizer",
]
