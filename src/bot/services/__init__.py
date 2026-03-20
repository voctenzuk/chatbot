"""Services module for bot functionality."""

from loguru import logger

# Episode Manager Service
try:
    from bot.services.episode_manager import (
        EpisodeManager,
        EpisodeManagerConfig,
        MessageResult,
        get_episode_manager,
        set_episode_manager,
    )
except ImportError as _e:
    logger.warning("Failed to import episode_manager: {}", _e)
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
except ImportError as _e:
    logger.warning("Failed to import db_client: {}", _e)
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
except ImportError as _e:
    logger.warning("Failed to import episode_switcher: {}", _e)
    EpisodeSwitcherEpisode = None  # type: ignore
    EpisodeConfig = None  # type: ignore
    EpisodeSwitcherMessage = None  # type: ignore
    SimpleEmbeddingProvider = None  # type: ignore
    SwitchDecision = None  # type: ignore
    get_episode_switcher_manager = None  # type: ignore
    set_episode_switcher_manager = None  # type: ignore

# Cognee Memory Service
try:
    from bot.services.cognee_memory_service import (
        CogneeMemoryService,
        get_memory_service as get_cognee_memory_service,
        set_memory_service as set_cognee_memory_service,
    )
except ImportError as _e:
    logger.warning("Failed to import cognee_memory_service: {}", _e)
    CogneeMemoryService = None  # type: ignore
    get_cognee_memory_service = None  # type: ignore
    set_cognee_memory_service = None  # type: ignore

# Memory Models
try:
    from bot.services.memory_models import (
        MemoryCategory,
        MemoryFact,
        MemoryType,
    )
except ImportError as _e:
    logger.warning("Failed to import memory_models: {}", _e)
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
except ImportError as _e:
    logger.warning("Failed to import context_builder: {}", _e)
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
except ImportError as _e:
    logger.warning("Failed to import summarizer: {}", _e)
    Summarizer = None  # type: ignore
    SummarizerConfig = None  # type: ignore
    SummaryKind = None  # type: ignore
    SummaryResult = None  # type: ignore
    SummaryJSON = None  # type: ignore
    FactCandidate = None  # type: ignore
    ArtifactReference = None  # type: ignore
    get_summarizer = None  # type: ignore
    set_summarizer = None  # type: ignore

# Artifact Service
try:
    from bot.services.artifact_service import (
        Artifact,
        ArtifactProcessingStatus,
        ArtifactService,
        ArtifactText,
        ArtifactType,
        CreateArtifactRequest,
        CreateArtifactResult,
        TextSurrogateForContext,
        TextSurrogateKind,
        get_artifact_service,
        set_artifact_service,
    )
except ImportError as _e:
    logger.warning("Failed to import artifact_service: {}", _e)
    Artifact = None  # type: ignore
    ArtifactProcessingStatus = None  # type: ignore
    ArtifactService = None  # type: ignore
    ArtifactText = None  # type: ignore
    ArtifactType = None  # type: ignore
    CreateArtifactRequest = None  # type: ignore
    CreateArtifactResult = None  # type: ignore
    TextSurrogateForContext = None  # type: ignore
    TextSurrogateKind = None  # type: ignore
    get_artifact_service = None  # type: ignore
    set_artifact_service = None  # type: ignore

# Storage Backend
try:
    from bot.services.storage_backend import (
        LocalStorageBackend,
        S3StorageBackend,
        StorageBackend,
        StorageReference,
        get_storage_backend,
        set_storage_backend,
    )
except ImportError as _e:
    logger.warning("Failed to import storage_backend: {}", _e)
    LocalStorageBackend = None  # type: ignore
    S3StorageBackend = None  # type: ignore
    StorageBackend = None  # type: ignore
    StorageReference = None  # type: ignore
    get_storage_backend = None  # type: ignore
    set_storage_backend = None  # type: ignore

# System Prompt
try:
    from bot.services.system_prompt import (
        DEFAULT_SYSTEM_PROMPT,
        get_system_prompt,
    )
except ImportError as _e:
    logger.warning("Failed to import system_prompt: {}", _e)
    DEFAULT_SYSTEM_PROMPT = None  # type: ignore
    get_system_prompt = None  # type: ignore

# Langfuse Observability Service
from bot.services.langfuse_service import (
    LangfuseService,
    get_langfuse_service,
    set_langfuse_service,
)

# LLM Service
try:
    from bot.services.llm_service import (
        LLMResponse,
        LLMService,
        get_llm_service,
        set_llm_service,
    )
except ImportError as _e:
    logger.warning("Failed to import llm_service: {}", _e)
    LLMResponse = None  # type: ignore
    LLMService = None  # type: ignore
    get_llm_service = None  # type: ignore
    set_llm_service = None  # type: ignore

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
    # Cognee Memory Service
    "CogneeMemoryService",
    "get_cognee_memory_service",
    "set_cognee_memory_service",
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
    # Artifact Service
    "Artifact",
    "ArtifactProcessingStatus",
    "ArtifactService",
    "ArtifactText",
    "ArtifactType",
    "CreateArtifactRequest",
    "CreateArtifactResult",
    "TextSurrogateForContext",
    "TextSurrogateKind",
    "get_artifact_service",
    "set_artifact_service",
    # Storage Backend
    "LocalStorageBackend",
    "S3StorageBackend",
    "StorageBackend",
    "StorageReference",
    "get_storage_backend",
    "set_storage_backend",
    # System Prompt
    "DEFAULT_SYSTEM_PROMPT",
    "get_system_prompt",
    # Langfuse Observability Service
    "LangfuseService",
    "get_langfuse_service",
    "set_langfuse_service",
    # LLM Service
    "LLMResponse",
    "LLMService",
    "get_llm_service",
    "set_llm_service",
]
