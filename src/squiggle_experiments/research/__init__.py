"""Research-grade training configuration and runners."""

from .config import (
    ResearchCfg,
    ResearchModelCfg,
    TrainingCfg,
    ScalarsCfg,
    CheckpointCfg,
    ProbesCfg,
    ActivationCheckpointCfg,
    load_research_config,
    get_research_default_config,
    get_research_debug_config,
)
from .trainer import run_research

__all__ = [
    # Config classes
    "ResearchCfg",
    "ResearchModelCfg",
    "TrainingCfg",
    "ScalarsCfg",
    "CheckpointCfg",
    "ProbesCfg",
    "ActivationCheckpointCfg",
    # Config loaders
    "load_research_config",
    "get_research_default_config",
    "get_research_debug_config",
    # Runner
    "run_research",
]
