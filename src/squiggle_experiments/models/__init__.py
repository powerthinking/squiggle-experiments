"""Model architectures for squiggle experiments."""

from .tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from .research_transformer import (
    ResearchTransformerConfig,
    ResearchTransformerLM,
    get_research_config_350m,
    get_research_config_1b,
    get_research_config_debug,
)

__all__ = [
    # Scout/tiny models
    "TinyTransformerConfig",
    "TinyTransformerLM",
    # Research models
    "ResearchTransformerConfig",
    "ResearchTransformerLM",
    "get_research_config_350m",
    "get_research_config_1b",
    "get_research_config_debug",
]
