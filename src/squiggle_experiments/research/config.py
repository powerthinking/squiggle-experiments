"""Configuration for research-grade training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


# -------------------------
# A. Scalars Config (every step)
# -------------------------

@dataclass(frozen=True)
class ScalarsCfg:
    """Configuration for scalar logging (every step)."""

    enabled: bool = True
    # What to log
    global_loss: bool = True
    per_task_loss: bool = True
    per_task_accuracy: bool = True  # Small online probe
    grad_norm_global: bool = True
    grad_norm_per_block: bool = False  # Optional, expensive
    lr: bool = True
    curriculum_mix: bool = True  # Task mix ratios


# -------------------------
# B. Checkpoints Config (weights)
# -------------------------

@dataclass(frozen=True)
class CheckpointCfg:
    """Configuration for model weight checkpoints."""

    enabled: bool = True
    # Early phase: more frequent
    early_until_step: int = 2000
    early_every_steps: int = 250  # 250-500 range
    # Mid/late phase: less frequent
    late_every_steps: int = 1000  # 1k-2k range
    # Retention
    keep_last_n: int = 5
    save_optimizer: bool = True


# -------------------------
# C. Probes Config (critical section)
# -------------------------

@dataclass(frozen=True)
class FixedProbeCfg:
    """Fixed probe configuration - always run."""

    enabled: bool = True
    # Frozen token situations
    n_examples: int = 500  # 250-1000 range
    seed: int = 123


@dataclass(frozen=True)
class ExtendedProbeCfg:
    """Extended probe configuration - run at milestones."""

    enabled: bool = True
    n_examples: int = 5000  # 5k-10k range
    seed: int = 456
    # Milestone percentages of total training (0.0 to 1.0)
    milestone_fractions: tuple = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ProbeMetricsCfg:
    """What metrics to compute on probes."""

    embeddings_output: bool = True
    residual_stream: bool = True  # At specified layers
    attention_stats: bool = True  # Attention pattern statistics
    entropy: bool = True  # Output distribution entropy
    top_k_mass: bool = True  # Mass in top-k predictions
    top_k_values: tuple = (1, 5, 10)  # k values for top-k mass


@dataclass(frozen=True)
class ProbesCfg:
    """Combined probes configuration."""

    fixed: FixedProbeCfg = field(default_factory=FixedProbeCfg)
    extended: ExtendedProbeCfg = field(default_factory=ExtendedProbeCfg)
    metrics: ProbeMetricsCfg = field(default_factory=ProbeMetricsCfg)
    # Layers to capture for residual stream
    layers: List[int] = field(default_factory=lambda: [0, 6, 12, 18, 23])
    # How often to run fixed probes
    fixed_every_steps: int = 100


# -------------------------
# D. Activation Checkpoints Config
# -------------------------

@dataclass(frozen=True)
class ActivationCaptureCfg:
    """What to capture at each activation checkpoint."""

    x_in: bool = True  # Input to block
    attn_out: bool = True  # Attention output (before residual add)
    x_mid: bool = True  # After attention + residual, before MLP
    mlp_out: bool = True  # MLP output (before residual add)
    x_out: bool = True  # Final block output


@dataclass(frozen=True)
class ActivationCheckpointCfg:
    """Configuration for activation checkpoints."""

    enabled: bool = True
    # All layers captured at these intervals:
    # Early phase (steps 0-2000): every 500-1000 steps
    early_until_step: int = 2000
    early_every_steps: int = 500
    # Mid phase (steps 2000-10000): every 2000-5000 steps
    mid_until_step: int = 10000
    mid_every_steps: int = 2000
    # Late phase (steps 10000+): every 5000-10000 steps
    late_every_steps: int = 5000
    # What to capture
    capture: ActivationCaptureCfg = field(default_factory=ActivationCaptureCfg)
    # Source data for captures
    source: Literal["train_batch", "probe_fixed", "mixed"] = "probe_fixed"
    probe_n_examples: int = 256
    probe_seed: int = 123


# -------------------------
# E. Full Attention Matrices Config (DEFERRED)
# -------------------------

@dataclass(frozen=True)
class AttentionMatrixCfg:
    """
    Configuration for full attention matrix capture.

    NOTE: DEFERRED - Do not implement until signatures exist.
    Only capture on:
    - Fixed probe runs
    - Extended probe milestones
    All layers when captured.
    """

    enabled: bool = False  # DEFERRED - set to False
    on_fixed_probes: bool = True
    on_extended_milestones: bool = True
    all_layers: bool = True


# -------------------------
# Model config
# -------------------------

@dataclass(frozen=True)
class ResearchModelCfg:
    """Configuration for research transformer model."""

    size: Literal["debug", "350m", "1b"] = "1b"
    vocab_size: int = 32000
    max_seq_len: int = 2048
    # Override specific params (None = use preset defaults)
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: float = 0.0
    tie_embeddings: bool = True


# -------------------------
# Data config
# -------------------------

@dataclass(frozen=True)
class DataCfg:
    """Configuration for training data."""

    # HuggingFace dataset or local path
    dataset: str = "nvidia/OpenMathReasoning"
    split: str = "cot"
    # Local family data directory (overrides dataset if set)
    family_data_dir: Optional[str] = None
    # Families to include (None = all)
    families: Optional[List[str]] = None
    # Data mixing
    max_samples: Optional[int] = None
    shuffle_seed: int = 42


# -------------------------
# Training config
# -------------------------

@dataclass(frozen=True)
class TrainingCfg:
    """Training hyperparameters."""

    # Basic
    steps: int = 50000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    # LR schedule
    warmup_steps: int = 500
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1
    # Precision
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    # Gradient clipping
    max_grad_norm: float = 1.0


# -------------------------
# Root config
# -------------------------

@dataclass(frozen=True)
class ResearchCfg:
    """Root configuration for research training runs."""

    # Run identification
    run_id: Optional[str] = None
    run_name: str = "research_1b"
    seed: int = 1337

    # Hardware
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0"

    # Sub-configs
    model: ResearchModelCfg = field(default_factory=ResearchModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)

    # Logging hierarchy (A-E)
    scalars: ScalarsCfg = field(default_factory=ScalarsCfg)  # A
    checkpoints: CheckpointCfg = field(default_factory=CheckpointCfg)  # B
    probes: ProbesCfg = field(default_factory=ProbesCfg)  # C
    activations: ActivationCheckpointCfg = field(default_factory=ActivationCheckpointCfg)  # D
    attention_matrices: AttentionMatrixCfg = field(default_factory=AttentionMatrixCfg)  # E (deferred)


# -------------------------
# Coercion helpers
# -------------------------

def _coerce_model(d: Dict[str, Any]) -> ResearchModelCfg:
    return ResearchModelCfg(**{k: v for k, v in (d or {}).items() if v is not None})


def _coerce_data(d: Dict[str, Any]) -> DataCfg:
    dd = dict(d or {})
    if "families" in dd and dd["families"] is not None:
        dd["families"] = list(dd["families"])
    return DataCfg(**dd)


def _coerce_training(d: Dict[str, Any]) -> TrainingCfg:
    dd = dict(d or {})
    if "betas" in dd:
        dd["betas"] = tuple(dd["betas"])
    # Ensure numeric types
    if "lr" in dd:
        dd["lr"] = float(dd["lr"])
    if "weight_decay" in dd:
        dd["weight_decay"] = float(dd["weight_decay"])
    if "min_lr_ratio" in dd:
        dd["min_lr_ratio"] = float(dd["min_lr_ratio"])
    if "max_grad_norm" in dd:
        dd["max_grad_norm"] = float(dd["max_grad_norm"])
    return TrainingCfg(**dd)


def _coerce_scalars(d: Dict[str, Any]) -> ScalarsCfg:
    return ScalarsCfg(**(d or {}))


def _coerce_checkpoint(d: Dict[str, Any]) -> CheckpointCfg:
    return CheckpointCfg(**(d or {}))


def _coerce_fixed_probe(d: Dict[str, Any]) -> FixedProbeCfg:
    return FixedProbeCfg(**(d or {}))


def _coerce_extended_probe(d: Dict[str, Any]) -> ExtendedProbeCfg:
    dd = dict(d or {})
    if "milestone_fractions" in dd:
        dd["milestone_fractions"] = tuple(dd["milestone_fractions"])
    return ExtendedProbeCfg(**dd)


def _coerce_probe_metrics(d: Dict[str, Any]) -> ProbeMetricsCfg:
    dd = dict(d or {})
    if "top_k_values" in dd:
        dd["top_k_values"] = tuple(dd["top_k_values"])
    return ProbeMetricsCfg(**dd)


def _coerce_probes(d: Dict[str, Any]) -> ProbesCfg:
    dd = dict(d or {})
    fixed = _coerce_fixed_probe(dd.get("fixed", {}))
    extended = _coerce_extended_probe(dd.get("extended", {}))
    metrics = _coerce_probe_metrics(dd.get("metrics", {}))
    layers = dd.get("layers", [0, 6, 12, 18, 23])
    fixed_every = dd.get("fixed_every_steps", 100)
    return ProbesCfg(
        fixed=fixed,
        extended=extended,
        metrics=metrics,
        layers=list(layers),
        fixed_every_steps=fixed_every,
    )


def _coerce_activation_capture(d: Dict[str, Any]) -> ActivationCaptureCfg:
    return ActivationCaptureCfg(**(d or {}))


def _coerce_activations(d: Dict[str, Any]) -> ActivationCheckpointCfg:
    dd = dict(d or {})
    capture = _coerce_activation_capture(dd.pop("capture", {}))
    return ActivationCheckpointCfg(capture=capture, **dd)


def _coerce_attention_matrices(d: Dict[str, Any]) -> AttentionMatrixCfg:
    return AttentionMatrixCfg(**(d or {}))


# -------------------------
# Loader
# -------------------------

def load_research_config(path: str | Path) -> ResearchCfg:
    """Load research config from YAML file."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}

    return ResearchCfg(
        run_id=(str(raw.get("run_id")) if raw.get("run_id") is not None else None),
        run_name=str(raw.get("run_name", "research_1b")),
        seed=int(raw.get("seed", 1337)),
        device=str(raw.get("device", "auto")),
        model=_coerce_model(raw.get("model", {})),
        data=_coerce_data(raw.get("data", {})),
        training=_coerce_training(raw.get("training", {})),
        scalars=_coerce_scalars(raw.get("scalars", {})),
        checkpoints=_coerce_checkpoint(raw.get("checkpoints", {})),
        probes=_coerce_probes(raw.get("probes", {})),
        activations=_coerce_activations(raw.get("activations", {})),
        attention_matrices=_coerce_attention_matrices(raw.get("attention_matrices", {})),
    )


# -------------------------
# Default configs
# -------------------------

def get_research_default_config() -> ResearchCfg:
    """Get default research config for 1.3B model with full logging."""
    return ResearchCfg(
        run_name="research_1b_default",
        model=ResearchModelCfg(size="1b"),
        training=TrainingCfg(
            steps=50000,
            batch_size=32,
            gradient_accumulation_steps=8,
            lr=3e-4,
            warmup_steps=1000,
        ),
        scalars=ScalarsCfg(
            grad_norm_per_block=False,  # Enable if needed
        ),
        checkpoints=CheckpointCfg(
            early_until_step=2000,
            early_every_steps=250,
            late_every_steps=1000,
        ),
        probes=ProbesCfg(
            fixed=FixedProbeCfg(n_examples=500),
            extended=ExtendedProbeCfg(
                n_examples=5000,
                milestone_fractions=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
            ),
            layers=[0, 4, 8, 12, 16, 20, 23],
            fixed_every_steps=100,
        ),
        activations=ActivationCheckpointCfg(
            early_until_step=2000,
            early_every_steps=500,
            mid_until_step=10000,
            mid_every_steps=2000,
            late_every_steps=5000,
        ),
        attention_matrices=AttentionMatrixCfg(enabled=False),  # DEFERRED
    )


def get_research_debug_config() -> ResearchCfg:
    """Get debug config for testing."""
    return ResearchCfg(
        run_name="research_debug",
        model=ResearchModelCfg(size="debug"),
        training=TrainingCfg(
            steps=100,
            batch_size=4,
            gradient_accumulation_steps=1,
            lr=1e-3,
            warmup_steps=10,
        ),
        scalars=ScalarsCfg(),
        checkpoints=CheckpointCfg(
            early_until_step=50,
            early_every_steps=10,
            late_every_steps=25,
        ),
        probes=ProbesCfg(
            fixed=FixedProbeCfg(n_examples=32),
            extended=ExtendedProbeCfg(n_examples=64),
            layers=[0, 2, 3],
            fixed_every_steps=10,
        ),
        activations=ActivationCheckpointCfg(
            early_until_step=50,
            early_every_steps=10,
            mid_until_step=75,
            mid_every_steps=15,
            late_every_steps=25,
        ),
    )
