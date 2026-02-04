"""Services module for bot functionality."""

# Episode Manager Service
try:
    from bot.services.episode_manager import (
        EpisodeManager,
        EpisodeManagerConfig,
        MessageResult,
        get_episode_manager,
        set_episode_manager,
    )
except ImportError:
    EpisodeManager = None  # type: ignore
    EpisodeManagerConfig = None  # type: ignore
    MessageResult = None  # type: ignore
    get_episode_manager = None  # type: ignore
    set_episode_manager = None  # type: ignore

# Database Client
try:
    from bot.services.db_client import (
        DatabaseClient,
        Episode,
        EpisodeMessage,
        EpisodeSummary,
        Thread,
        get_db_client,
        set_db_client,
    )
except ImportError:
    DatabaseClient = None  # type: ignore
    Episode = None  # type: ignore
    EpisodeMessage = None  # type: ignore
    EpisodeSummary = None  # type: ignore
    Thread = None  # type: ignore
    get_db_client = None  # type: ignore
    set_db_client = None  # type: ignore

# Episode Switcher (base functionality)
try:
    from bot.services.episode_switcher import (
        Episode as EpisodeSwitcherEpisode,
        EpisodeConfig,
        Message as EpisodeSwitcherMessage,
        SimpleEmbeddingProvider,
        SwitchDecision,
        get_episode_manager as get_episode_switcher_manager,
        set_episode_manager as set_episode_switcher_manager,
    )
except ImportError:
    EpisodeSwitcherEpisode = None  # type: ignore
    EpisodeConfig = None  # type: ignore
    EpisodeSwitcherMessage = None  # type: ignore
    SimpleEmbeddingProvider = None  # type: ignore
    SwitchDecision = None  # type: ignore
    get_episode_switcher_manager = None  # type: ignore
    set_episode_switcher_manager = None  # type: ignore

# Mem0 Memory Service
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

# Memory Models
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

# Context Builder
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

__all__ = [
    # Episode Manager Service
    "EpisodeManager",
    "EpisodeManagerConfig",
    "MessageResult",
    "get_episode_manager",
    "set_episode_manager",
    # Database Client
    "DatabaseClient",
    "Episode",
    "EpisodeMessage",
    "EpisodeSummary",
    "Thread",
    "get_db_client",
    "set_db_client",
    # Episode Switcher
    "EpisodeSwitcherEpisode",
    "EpisodeConfig",
    "EpisodeSwitcherMessage",
    "SimpleEmbeddingProvider",
    "SwitchDecision",
    "get_episode_switcher_manager",
    "set_episode_switcher_manager",
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
]
