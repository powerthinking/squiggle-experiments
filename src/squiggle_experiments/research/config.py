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
    """Configuration for activation checkpoints.

    Phase boundaries can be specified as either:
    - Absolute steps (early_until_step, mid_until_step)
    - Fractions of total_steps (early_until_fraction, mid_until_fraction)

    Fractions take precedence if set, and scale with step_multiplier automatically.
    Set early_until_fraction=1.0 for uniform capture across entire run.
    """

    enabled: bool = True
    # All layers captured at these intervals:
    # Early phase (steps 0-early_until): every early_every_steps
    early_until_step: int = 2000
    early_until_fraction: Optional[float] = None  # If set, overrides early_until_step
    early_every_steps: int = 500
    # Mid phase (early_until-mid_until): every mid_every_steps
    mid_until_step: int = 10000
    mid_until_fraction: Optional[float] = None  # If set, overrides mid_until_step
    mid_every_steps: int = 2000
    # Late phase (mid_until+): every late_every_steps
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
    # Training split directory (overrides family_data_dir if set)
    # Should contain train.jsonl, val_random.jsonl, val_family.jsonl
    split_dir: Optional[str] = None
    # Families to include (None = all)
    families: Optional[List[str]] = None
    # Data mixing
    max_samples: Optional[int] = None
    shuffle_seed: int = 42


# -------------------------
# Curriculum config
# -------------------------


@dataclass(frozen=True)
class CurriculumCfg:
    """Configuration for curriculum-driven training.

    When enabled, training samples are selected according to a time-varying
    policy over problem families defined in a curriculum YAML spec.
    """

    enabled: bool = False
    # Path to curriculum YAML spec (e.g., "curricula/family_ramp.yaml")
    spec_path: Optional[str] = None
    # Write sample trace for exact replay (can be large)
    trace_enabled: bool = False
    # Log family distribution every N steps (0 = disabled)
    log_distribution_every: int = 100


# -------------------------
# Loss Milestones Config
# -------------------------


@dataclass(frozen=True)
class LossMilestonesCfg:
    """Configuration for loss milestone tracking.

    Milestones are recorded for analysis but do NOT affect training termination.
    Use relative mode for cross-curriculum comparability.
    """

    enabled: bool = True
    # "relative" = % improvement from initial to best loss
    # "absolute" = fixed loss values
    mode: Literal["relative", "absolute"] = "relative"
    # For relative mode: fractions of improvement (0.2 = 20% of the way from initial to best)
    relative_fractions: tuple = (0.2, 0.4, 0.6, 0.8)
    # For absolute mode: specific loss values to track
    absolute_values: Optional[tuple] = None  # e.g., (4.0, 3.5, 3.0, 2.5)


# -------------------------
# Training config
# -------------------------


@dataclass(frozen=True)
class TrainingCfg:
    """Training hyperparameters.

    Step budget resolution (in order of precedence):
    1. total_steps (if set) - canonical, recommended
    2. epochs (if set) - converted to steps based on dataset size
    3. steps (fallback default)

    The step_multiplier scales the resolved budget (useful for extension experiments).
    """

    # Step budget - total_steps is canonical (epochs/steps are legacy/convenience)
    total_steps: Optional[int] = None  # PRIMARY: explicit step budget
    epochs: Optional[int] = None  # Convenience: converted to steps from dataset size
    steps: int = 50000  # Fallback default
    step_multiplier: float = 1.0  # Extension multiplier (1.0, 2.0, 3.0 for ladder)

    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    # LR schedule
    warmup_steps: int = 500  # Used if warmup_fraction is None
    warmup_fraction: Optional[float] = 0.05  # 5% of total steps (overrides warmup_steps)
    min_warmup_steps: int = 10  # Guardrail: minimum warmup regardless of fraction
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1  # Floor for cosine decay (0.1 = 10% of base LR)
    min_floor_steps: int = 50  # Guardrail: minimum steps at LR floor
    # Precision
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    # Gradient clipping
    max_grad_norm: float = 1.0
    # Validation
    val_every_epoch: bool = True  # Run validation every epoch (if epochs set)
    val_every_steps: Optional[int] = None  # Run validation every N steps (if step-based)

    # Loss milestone tracking (for analysis, not termination)
    loss_milestones: LossMilestonesCfg = field(default_factory=LossMilestonesCfg)


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

    # Curriculum-driven training
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)


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
    # Handle epochs (may be int or None)
    if "epochs" in dd and dd["epochs"] is not None:
        dd["epochs"] = int(dd["epochs"])
    # Handle total_steps (may be int or None)
    if "total_steps" in dd and dd["total_steps"] is not None:
        dd["total_steps"] = int(dd["total_steps"])
    # Handle val_every_steps (may be int or None)
    if "val_every_steps" in dd and dd["val_every_steps"] is not None:
        dd["val_every_steps"] = int(dd["val_every_steps"])
    # Handle step_multiplier
    if "step_multiplier" in dd:
        dd["step_multiplier"] = float(dd["step_multiplier"])
    # Handle guardrails
    if "min_warmup_steps" in dd:
        dd["min_warmup_steps"] = int(dd["min_warmup_steps"])
    if "min_floor_steps" in dd:
        dd["min_floor_steps"] = int(dd["min_floor_steps"])
    # Handle loss milestones sub-config
    if "loss_milestones" in dd:
        dd["loss_milestones"] = _coerce_loss_milestones(dd["loss_milestones"])
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


def _coerce_curriculum(d: Dict[str, Any]) -> CurriculumCfg:
    return CurriculumCfg(**(d or {}))


def _coerce_loss_milestones(d: Dict[str, Any]) -> LossMilestonesCfg:
    dd = dict(d or {})
    if "relative_fractions" in dd:
        dd["relative_fractions"] = tuple(dd["relative_fractions"])
    if "absolute_values" in dd and dd["absolute_values"] is not None:
        dd["absolute_values"] = tuple(dd["absolute_values"])
    return LossMilestonesCfg(**dd)


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
        curriculum=_coerce_curriculum(raw.get("curriculum", {})),
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
