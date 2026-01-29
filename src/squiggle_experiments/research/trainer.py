"""Research-grade training runner with comprehensive logging.

Implements the A-E logging hierarchy:
- A. Scalars: Global loss, per-task metrics, grad norms, LR (every step)
- B. Checkpoints: Model weights + optimizer state (configurable cadence)
- C. Probes: Fixed and extended probe evaluation with metric capture
- D. Activations: Layer-wise activation snapshots
- E. Attention matrices: Full attention patterns (DEFERRED)
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from squiggle_core import paths
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from squiggle_experiments.models.research_transformer import (
    ResearchTransformerConfig,
    ResearchTransformerLM,
    get_research_config_1b,
    get_research_config_350m,
    get_research_config_debug,
)
from squiggle_experiments.utils.logging import write_meta_json
from squiggle_experiments.utils.run_id import make_run_id
from squiggle_experiments.utils.seed import set_seed

from .config import (
    ResearchCfg,
    get_research_debug_config,
    get_research_default_config,
    load_research_config,
)
from .curriculum import (
    CurriculumSampler,
    CurriculumSpec,
    SampleTraceWriter,
    write_curriculum_manifest,
)
from .data import FamilyDataset, SplitDataset, get_tokenizer

# --- Checkpoint staging infrastructure ---
# Save checkpoints to fast local SSD first, then move to final destination in background.
# This avoids slow I/O blocking training when data dir is on a slow filesystem (e.g., WSL + Windows drive).

_CHECKPOINT_STAGE_DIR = Path(tempfile.gettempdir()) / "squiggle-checkpoint-stage"
_checkpoint_move_executor: Optional[ThreadPoolExecutor] = None


def _get_checkpoint_executor() -> ThreadPoolExecutor:
    """Get or create the background checkpoint move executor."""
    global _checkpoint_move_executor
    if _checkpoint_move_executor is None:
        _checkpoint_move_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ckpt-move"
        )
        atexit.register(_shutdown_checkpoint_executor)
    return _checkpoint_move_executor


def _shutdown_checkpoint_executor() -> None:
    """Shutdown the checkpoint move executor, waiting for pending moves."""
    global _checkpoint_move_executor
    if _checkpoint_move_executor is not None:
        _checkpoint_move_executor.shutdown(wait=True)
        _checkpoint_move_executor = None


def _move_checkpoint_background(src: Path, dst: Path) -> None:
    """Move checkpoint from staging to final location (runs in background thread)."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except Exception as e:
        # Log but don't crash - the staged file remains as backup
        print(f"[Checkpoint] Warning: background move failed {src} -> {dst}: {e}")


def _pick_device(device_setting: str) -> str:
    """Resolve device setting to actual device string."""
    if device_setting == "cpu":
        return "cpu"
    if device_setting.startswith("cuda"):
        return device_setting if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model_config(cfg: ResearchCfg) -> ResearchTransformerConfig:
    """Get model config based on size setting and overrides."""
    if cfg.model.size == "debug":
        base = get_research_config_debug()
    elif cfg.model.size == "350m":
        base = get_research_config_350m()
    else:  # 1b
        base = get_research_config_1b()

    # Apply overrides if specified
    overrides = {}
    if cfg.model.vocab_size != 32000:
        overrides["vocab_size"] = cfg.model.vocab_size
    if cfg.model.max_seq_len != 2048:
        overrides["max_seq_len"] = cfg.model.max_seq_len
    if cfg.model.d_model is not None:
        overrides["d_model"] = cfg.model.d_model
    if cfg.model.n_layers is not None:
        overrides["n_layers"] = cfg.model.n_layers
    if cfg.model.n_heads is not None:
        overrides["n_heads"] = cfg.model.n_heads
    if cfg.model.d_ff is not None:
        overrides["d_ff"] = cfg.model.d_ff
    if cfg.model.dropout != 0.0:
        overrides["dropout"] = cfg.model.dropout
    if not cfg.model.tie_embeddings:
        overrides["tie_embeddings"] = cfg.model.tie_embeddings

    if overrides:
        # Create new config with overrides
        return ResearchTransformerConfig(
            vocab_size=overrides.get("vocab_size", base.vocab_size),
            max_seq_len=overrides.get("max_seq_len", base.max_seq_len),
            d_model=overrides.get("d_model", base.d_model),
            n_layers=overrides.get("n_layers", base.n_layers),
            n_heads=overrides.get("n_heads", base.n_heads),
            d_ff=overrides.get("d_ff", base.d_ff),
            dropout=overrides.get("dropout", base.dropout),
            tie_embeddings=overrides.get("tie_embeddings", base.tie_embeddings),
        )
    return base


class DummyDataset(Dataset):
    """Placeholder dataset for testing - generates random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, size: int = 10000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.seed = seed
        # Pre-generate data for reproducibility
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (size, seq_len), generator=rng)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens, "labels": tokens}


def _get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _should_checkpoint(step: int, cfg: ResearchCfg) -> bool:
    """Determine if we should save a checkpoint at this step."""
    if not cfg.checkpoints.enabled:
        return False
    if step < cfg.checkpoints.early_until_step:
        return step % cfg.checkpoints.early_every_steps == 0
    return step % cfg.checkpoints.late_every_steps == 0


def _resolve_activation_phases(cfg: ResearchCfg, total_steps: int) -> Dict[str, int]:
    """Resolve activation phase boundaries, using fractions if specified.

    Fractions take precedence over absolute steps and scale with step_multiplier.
    """
    act = cfg.activations

    # Early phase boundary
    if act.early_until_fraction is not None:
        early_until = int(act.early_until_fraction * total_steps)
    else:
        early_until = act.early_until_step

    # Mid phase boundary
    if act.mid_until_fraction is not None:
        mid_until = int(act.mid_until_fraction * total_steps)
    else:
        mid_until = act.mid_until_step

    return {
        "early_until": early_until,
        "mid_until": mid_until,
        "early_every": act.early_every_steps,
        "mid_every": act.mid_every_steps,
        "late_every": act.late_every_steps,
    }


def _should_capture_activations(
    step: int, cfg: ResearchCfg, resolved_phases: Optional[Dict[str, int]] = None
) -> bool:
    """Determine if we should capture activations at this step."""
    if not cfg.activations.enabled:
        return False

    # Use resolved phases if provided, otherwise fall back to config values
    if resolved_phases:
        early_until = resolved_phases["early_until"]
        mid_until = resolved_phases["mid_until"]
        early_every = resolved_phases["early_every"]
        mid_every = resolved_phases["mid_every"]
        late_every = resolved_phases["late_every"]
    else:
        early_until = cfg.activations.early_until_step
        mid_until = cfg.activations.mid_until_step
        early_every = cfg.activations.early_every_steps
        mid_every = cfg.activations.mid_every_steps
        late_every = cfg.activations.late_every_steps

    if step < early_until:
        return step % early_every == 0
    if step < mid_until:
        return step % mid_every == 0
    return step % late_every == 0


def _get_milestone_steps(cfg: ResearchCfg) -> List[int]:
    """Convert milestone fractions to step numbers."""
    total_steps = cfg.training.steps
    return [int(f * total_steps) for f in cfg.probes.extended.milestone_fractions]


@torch.no_grad()
def _compute_grad_norms(model: nn.Module, per_block: bool = False) -> Dict[str, float]:
    """Compute gradient norms."""
    norms = {}

    # Global grad norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    norms["grad_norm_global"] = total_norm**0.5

    # Per-block norms if requested
    if per_block and hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            block_norm = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    block_norm += p.grad.data.norm(2).item() ** 2
            norms[f"grad_norm_block_{i}"] = block_norm**0.5

    return norms


@torch.no_grad()
def _eval_probe(
    model: ResearchTransformerLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate model on probe data."""
    model.eval()

    logits, loss = model(input_ids, targets)
    loss_val = float(loss.item()) if loss is not None else float("nan")

    # Token-level accuracy
    preds = logits.argmax(dim=-1)
    # Shift for next-token prediction
    shift_preds = preds[..., :-1]
    shift_targets = targets[..., 1:]
    mask = shift_targets != -100
    correct = ((shift_preds == shift_targets) & mask).float().sum()
    total = mask.float().sum()
    accuracy = float(correct / total) if total > 0 else 0.0

    return {"loss": loss_val, "accuracy": accuracy}


@torch.no_grad()
def _compute_probe_metrics(
    model: ResearchTransformerLM,
    input_ids: torch.Tensor,
    cfg: ResearchCfg,
) -> Dict[str, torch.Tensor]:
    """Compute detailed probe metrics including embeddings, residuals, etc."""
    model.eval()
    metrics = {}

    # Forward pass to get logits
    logits, _ = model(input_ids)

    # Output entropy
    if cfg.probes.metrics.entropy:
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        metrics["entropy_mean"] = entropy.mean()
        metrics["entropy_std"] = entropy.std()

    # Top-k mass
    if cfg.probes.metrics.top_k_mass:
        probs = torch.softmax(logits, dim=-1)
        for k in cfg.probes.metrics.top_k_values:
            topk_probs, _ = torch.topk(probs, k, dim=-1)
            mass = topk_probs.sum(dim=-1)
            metrics[f"top_{k}_mass_mean"] = mass.mean()

    return metrics


@torch.no_grad()
def _evaluate_validation(
    model: ResearchTransformerLM,
    val_dataset: Dataset,
    device: str,
    batch_size: int = 4,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Evaluate model on validation dataset.

    Args:
        model: Model to evaluate
        val_dataset: Validation dataset
        device: Device to use
        batch_size: Batch size for evaluation
        max_batches: Maximum batches to evaluate (for speed)

    Returns:
        Dict with val_loss and val_accuracy
    """
    model.eval()

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, loss = model(input_ids, labels)

        if loss is not None:
            total_loss += loss.item()

        # Token-level accuracy
        preds = logits.argmax(dim=-1)
        shift_preds = preds[..., :-1]
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        correct = ((shift_preds == shift_labels) & mask).float().sum().item()
        total_tokens += mask.float().sum().item()
        total_correct += correct

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return {"val_loss": avg_loss, "val_accuracy": accuracy}


@torch.no_grad()
def _capture_activations(
    model: ResearchTransformerLM,
    input_ids: torch.Tensor,
    run_id: str,
    step: int,
    cfg: ResearchCfg,
) -> None:
    """Capture activation tensors at specified layers."""
    model.eval()

    capture_cfg = cfg.activations.capture
    out_dir = paths.captures_dir(run_id) / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "step": step,
        "source": cfg.activations.source,
        "tensors": {},
    }

    # Token embeddings
    x = model.tok_emb(input_ids)
    if capture_cfg.x_in:
        torch.save(x.detach().cpu(), out_dir / "embed.pt")
        manifest["tensors"]["embed"] = {"path": "embed.pt", "shape": list(x.shape)}

    # Get relevant portion of precomputed values
    seq_len = input_ids.size(1)
    freqs_cis = model.freqs_cis[:seq_len]
    mask = model.causal_mask[:seq_len, :seq_len]

    # Capture per-layer activations
    for i, layer in enumerate(model.layers):
        # x_in: input to block
        if capture_cfg.x_in:
            fname = f"layer_{i:02d}_x_in.pt"
            torch.save(x.detach().cpu(), out_dir / fname)
            manifest["tensors"][f"layer_{i}_x_in"] = {"path": fname, "shape": list(x.shape)}

        # Forward through attention
        h = layer.attn_norm(x)
        attn_out = layer.attn(h, freqs_cis, mask)

        if capture_cfg.attn_out:
            fname = f"layer_{i:02d}_attn_out.pt"
            torch.save(attn_out.detach().cpu(), out_dir / fname)
            manifest["tensors"][f"layer_{i}_attn_out"] = {
                "path": fname,
                "shape": list(attn_out.shape),
            }

        # x_mid: after attention + residual, before MLP
        x_mid = x + attn_out
        if capture_cfg.x_mid:
            fname = f"layer_{i:02d}_x_mid.pt"
            torch.save(x_mid.detach().cpu(), out_dir / fname)
            manifest["tensors"][f"layer_{i}_x_mid"] = {"path": fname, "shape": list(x_mid.shape)}

        # MLP
        mlp_out = layer.mlp(layer.mlp_norm(x_mid))
        if capture_cfg.mlp_out:
            fname = f"layer_{i:02d}_mlp_out.pt"
            torch.save(mlp_out.detach().cpu(), out_dir / fname)
            manifest["tensors"][f"layer_{i}_mlp_out"] = {
                "path": fname,
                "shape": list(mlp_out.shape),
            }

        # x_out: final block output
        x = x_mid + mlp_out
        if capture_cfg.x_out:
            fname = f"layer_{i:02d}_x_out.pt"
            torch.save(x.detach().cpu(), out_dir / fname)
            manifest["tensors"][f"layer_{i}_x_out"] = {"path": fname, "shape": list(x.shape)}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _save_checkpoint(
    model: ResearchTransformerLM,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    run_id: str,
    cfg: ResearchCfg,
) -> Path:
    """Save model checkpoint.

    Uses two-stage saving when the data directory is on a slow filesystem:
    1. Save to fast local staging directory (SSD)
    2. Move to final destination in background thread

    This prevents slow I/O from blocking training.
    """
    final_ckpt_dir = paths.run_dir(run_id) / "checkpoints"
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = final_ckpt_dir / f"step_{step:06d}.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": asdict(model.cfg)
        if hasattr(model.cfg, "__dataclass_fields__")
        else model.cfg.__dict__,
    }

    if cfg.checkpoints.save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Check if we should use staging (when data dir is on a different filesystem)
    stage_dir = _CHECKPOINT_STAGE_DIR / run_id
    use_staging = not str(final_ckpt_dir).startswith("/tmp")

    if use_staging:
        # Save to fast local staging directory
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_path = stage_dir / f"step_{step:06d}.pt"
        torch.save(checkpoint, stage_path)

        # Move to final destination in background
        executor = _get_checkpoint_executor()
        executor.submit(_move_checkpoint_background, stage_path, final_ckpt_path)
    else:
        # Direct save (data dir is already on fast local storage)
        torch.save(checkpoint, final_ckpt_path)

    # Cleanup old checkpoints (in final location)
    if cfg.checkpoints.keep_last_n > 0:
        existing = sorted(final_ckpt_dir.glob("step_*.pt"))
        if len(existing) > cfg.checkpoints.keep_last_n:
            for old_ckpt in existing[: -cfg.checkpoints.keep_last_n]:
                try:
                    old_ckpt.unlink()
                except FileNotFoundError:
                    pass  # May not have been moved yet

    return final_ckpt_path


def run_research(
    config_path: Optional[str] = None,
    cfg: Optional[ResearchCfg] = None,
    seed_override: Optional[int] = None,
    step_multiplier_override: Optional[float] = None,
    quiet: bool = False,
) -> str:
    """Run research-grade training with comprehensive logging.

    Args:
        config_path: Path to YAML config file
        cfg: Pre-loaded config (alternative to config_path)
        seed_override: Override seed from config (useful for running same config with different seeds)
        step_multiplier_override: Override step_multiplier (for extension ladder experiments)
        quiet: Suppress verbose output (checkpoint saves, epoch validation prints)
    """
    from dataclasses import replace

    if cfg is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or cfg")
        cfg = load_research_config(config_path)

    # Apply seed override if provided
    if seed_override is not None:
        cfg = replace(cfg, seed=seed_override)
        print(f"Seed overridden to: {seed_override}")

    # Apply step_multiplier override if provided
    if step_multiplier_override is not None:
        cfg = replace(cfg, training=replace(cfg.training, step_multiplier=step_multiplier_override))
        print(f"Step multiplier overridden to: {step_multiplier_override}")

    set_seed(cfg.seed)

    device = _pick_device(cfg.device)
    print(f"Using device: {device}")

    # Generate run ID
    run_id = cfg.run_id if cfg.run_id else make_run_id(cfg.run_name, cfg.seed)
    print(f"Run ID: {run_id}")

    # Model setup
    model_cfg = _get_model_config(cfg)
    print(f"Model params: {model_cfg.param_count():,}")

    model = ResearchTransformerLM(model_cfg).to(device)

    # Dataset setup (before optimizer/scheduler so we can compute total_steps from epochs)
    tokenizer = None
    val_dataset = None
    val_family_dataset = None

    if cfg.data.split_dir:
        # Use split directory (train.jsonl, val_random.jsonl, val_family.jsonl)
        split_dir = Path(cfg.data.split_dir)
        tokenizer = get_tokenizer(model_cfg.vocab_size)
        dataset = SplitDataset(
            split_dir=split_dir,
            tokenizer=tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            split_type="train",
            max_samples=cfg.data.max_samples,
            shuffle_seed=cfg.data.shuffle_seed,
        )
        # Also load validation sets
        val_dataset = SplitDataset(
            split_dir=split_dir,
            tokenizer=tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            split_type="val_random",
            shuffle_seed=cfg.data.shuffle_seed,
        )
        val_family_dataset = SplitDataset(
            split_dir=split_dir,
            tokenizer=tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            split_type="val_family",
            shuffle_seed=cfg.data.shuffle_seed,
        )
        print(f"Using split dataset from {split_dir}")
    elif cfg.data.family_data_dir:
        # Use real family data (legacy mode)
        family_dir = Path(cfg.data.family_data_dir)
        tokenizer = get_tokenizer(model_cfg.vocab_size)
        dataset = FamilyDataset(
            family_dir=family_dir,
            tokenizer=tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            families=cfg.data.families,
            max_samples=cfg.data.max_samples,
            shuffle_seed=cfg.data.shuffle_seed,
        )
        print(f"Using family dataset from {family_dir}")
    else:
        # Fallback to dummy dataset for testing
        dataset = DummyDataset(
            vocab_size=model_cfg.vocab_size,
            seq_len=model_cfg.max_seq_len,
            size=cfg.training.steps
            * cfg.training.batch_size
            * cfg.training.gradient_accumulation_steps,
            seed=cfg.seed,
        )
        print("Using dummy dataset (no split_dir or family_data_dir specified)")

    # Compute total_steps with precedence: total_steps > epochs > steps (fallback)
    effective_batch_size = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    steps_per_epoch = max(1, len(dataset) // effective_batch_size)

    # Step budget resolution
    if cfg.training.total_steps is not None:
        # PRIMARY: explicit total_steps
        base_steps = cfg.training.total_steps
        budget_source = "total_steps"
    elif cfg.training.epochs is not None:
        # Convenience: epochs converted to steps
        base_steps = cfg.training.epochs * steps_per_epoch
        budget_source = f"{cfg.training.epochs} epochs"
    else:
        # Fallback: legacy steps field
        base_steps = cfg.training.steps
        budget_source = "steps (fallback)"

    # Apply step_multiplier for extension ladder
    total_steps = int(base_steps * cfg.training.step_multiplier)
    n_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0

    print(f"Step budget: {base_steps} ({budget_source}) × {cfg.training.step_multiplier} = {total_steps} steps")
    print(f"  = {n_epochs:.2f} epochs ({steps_per_epoch} steps/epoch)")

    # Compute warmup_steps with guardrail (fraction takes precedence)
    if cfg.training.warmup_fraction is not None:
        warmup_steps = int(cfg.training.warmup_fraction * total_steps)
    else:
        warmup_steps = cfg.training.warmup_steps
    # Apply minimum warmup guardrail
    warmup_steps = max(cfg.training.min_warmup_steps, warmup_steps)

    # Compute floor_steps (for cosine schedule)
    floor_steps = cfg.training.min_floor_steps

    # Guardrail: prevent impossible schedules where warmup + floor >= total
    schedule_clamped = False
    schedule_clamp_reason = None
    if warmup_steps + floor_steps >= total_steps:
        schedule_clamped = True
        schedule_clamp_reason = "warmup+floor>=total"
        # Shrink floor first (keep warmup intact for stable early training)
        floor_steps = max(1, total_steps - warmup_steps - 1)
        print(f"[WARN] Schedule clamped: {schedule_clamp_reason}, floor_steps reduced to {floor_steps}")

    # Compute phase boundaries for event analysis
    # Phases: warmup (0 -> warmup_end), high_lr (warmup_end -> decay_start), decay (decay_start -> floor_start), floor
    warmup_end_step = warmup_steps
    if cfg.training.lr_schedule == "cosine":
        # Cosine: decay starts after warmup, reaches floor near the end
        decay_start_step = warmup_steps
        # Floor phase: where LR is within ~5% of min_lr (roughly last 10% of schedule)
        floor_start_step = total_steps - floor_steps
    else:
        # Constant: no decay phase
        decay_start_step = total_steps
        floor_start_step = total_steps

    phase_boundaries = {
        "warmup_end": warmup_end_step,
        "decay_start": decay_start_step,
        "floor_start": floor_start_step,
        "total_steps": total_steps,
    }

    print(
        f"LR schedule: {cfg.training.lr_schedule}, warmup={warmup_steps} steps ({100 * warmup_steps / total_steps:.1f}%)"
    )
    print(f"Phase boundaries: warmup_end={warmup_end_step}, floor_start={floor_start_step}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler
    if cfg.training.lr_schedule == "cosine":
        scheduler = _get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
            cfg.training.min_lr_ratio,
        )
    else:
        # Constant or linear - for now just use constant
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Setup curriculum sampler if enabled
    curriculum_sampler: Optional[CurriculumSampler] = None
    curriculum_spec: Optional[CurriculumSpec] = None

    if cfg.curriculum.enabled and cfg.curriculum.spec_path:
        spec_path = Path(cfg.curriculum.spec_path)
        candidates = [spec_path]  # Track what we searched

        if not spec_path.is_absolute():
            # Try to resolve relative paths in order of priority
            candidates = []

            # 1. Relative to config file directory
            if config_path:
                config_dir = Path(config_path).resolve().parent
                candidates.append(config_dir / cfg.curriculum.spec_path)

            # 2. Relative to squiggle-experiments package root
            package_root = Path(__file__).resolve().parent.parent.parent.parent
            candidates.append(package_root / cfg.curriculum.spec_path)

            # 3. Relative to current working directory
            candidates.append(Path(cfg.curriculum.spec_path))

            # Find first existing path
            for candidate in candidates:
                if candidate.exists():
                    spec_path = candidate
                    break

        if not spec_path.exists():
            print(f"Warning: Curriculum spec not found: {spec_path}")
            print(f"  Searched: {[str(c) for c in candidates]}")
            print("  Falling back to shuffle")
        else:
            curriculum_spec = CurriculumSpec.from_yaml(spec_path)
            family_index = dataset.build_family_index()
            curriculum_sampler = CurriculumSampler(
                spec=curriculum_spec,
                family_index=family_index,
                total_steps=total_steps,
                batch_size=cfg.training.batch_size,
                seed=cfg.seed,
            )
            print(
                f"Curriculum enabled: {curriculum_spec.name} ({len(curriculum_spec.phases)} phases)"
            )

    # Setup trace writer if enabled (for attribution analysis)
    trace_writer: Optional[SampleTraceWriter] = None
    if cfg.curriculum.trace_enabled and curriculum_sampler is not None:
        trace_path = paths.run_dir(run_id) / "sample_trace.jsonl"
        trace_writer = SampleTraceWriter(
            output_path=trace_path,
            run_id=run_id,
            seed=cfg.seed,
            split="train",
            sampler_mode=curriculum_spec.default_sampling_mode if curriculum_spec else "unknown",
        )
        trace_writer.__enter__()
        print(f"Trace enabled: {trace_path}")

    # Note: when using curriculum sampler, we sample batches manually in the loop
    # So we create DataLoader with shuffle for fallback, but use sampler directly when available
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(curriculum_sampler is None),  # Only shuffle when not using curriculum
        num_workers=0,
        pin_memory=True if device.startswith("cuda") else False,
    )
    data_iter = iter(dataloader)

    # Setup probe data
    probe_fixed = None
    probe_extended = None

    if cfg.probes.fixed.enabled:
        set_seed(cfg.probes.fixed.seed)
        probe_fixed = torch.randint(
            0, model_cfg.vocab_size, (cfg.probes.fixed.n_examples, model_cfg.max_seq_len)
        ).to(device)
        set_seed(cfg.seed)

    if cfg.probes.extended.enabled:
        set_seed(cfg.probes.extended.seed)
        probe_extended = torch.randint(
            0, model_cfg.vocab_size, (cfg.probes.extended.n_examples, model_cfg.max_seq_len)
        ).to(device)
        set_seed(cfg.seed)

    milestone_steps = _get_milestone_steps(cfg)

    # Resolve activation capture phases (handles fraction-based boundaries)
    activation_phases = _resolve_activation_phases(cfg, total_steps)
    if cfg.activations.enabled:
        print(f"Activation capture: early<{activation_phases['early_until']}, "
              f"mid<{activation_phases['mid_until']}, late>={activation_phases['mid_until']}")

    # Write meta.json
    meta_path = paths.run_dir(run_id) / "meta.json"

    # Track step multiplier sources for auditability
    step_multiplier_config = cfg.training.step_multiplier
    step_multiplier_requested = step_multiplier_override  # From CLI, may be None
    step_multiplier_effective = cfg.training.step_multiplier  # After any override applied

    write_meta_json(
        meta_path,
        {
            "run_id": run_id,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            # Config file info
            "config_path": str(Path(config_path).resolve()) if config_path else None,
            "config_filename": Path(config_path).name if config_path else None,
            # Step budget - canonical fields
            "training": {
                "total_steps_base": base_steps,
                "step_multiplier_config": step_multiplier_config,
                "step_multiplier_requested": step_multiplier_requested,
                "step_multiplier_effective": step_multiplier_effective,
                "total_steps_effective": total_steps,
                "step_budget_source": budget_source,
                "steps_per_epoch": steps_per_epoch,
                "epochs_effective": n_epochs,
                # Schedule
                "warmup_steps": warmup_steps,
                "warmup_fraction": cfg.training.warmup_fraction,
                "min_warmup_steps": cfg.training.min_warmup_steps,
                "floor_steps": floor_steps,
                "min_floor_steps": cfg.training.min_floor_steps,
                "schedule_clamped": schedule_clamped,
                "schedule_clamp_reason": schedule_clamp_reason,
                # Optimizer
                "lr": cfg.training.lr,
                "lr_schedule": cfg.training.lr_schedule,
                "min_lr_ratio": cfg.training.min_lr_ratio,
                "batch_size": cfg.training.batch_size,
                "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
            },
            "phase_boundaries": phase_boundaries,
            "loss_milestones_config": {
                "enabled": cfg.training.loss_milestones.enabled,
                "mode": cfg.training.loss_milestones.mode,
                "relative_fractions": list(cfg.training.loss_milestones.relative_fractions)
                if cfg.training.loss_milestones.mode == "relative"
                else None,
                "absolute_values": list(cfg.training.loss_milestones.absolute_values)
                if cfg.training.loss_milestones.absolute_values
                else None,
            },
            "device": device,
            "model": {
                "size": cfg.model.size,
                "d_model": model_cfg.d_model,
                "n_layers": model_cfg.n_layers,
                "n_heads": model_cfg.n_heads,
                "d_ff": model_cfg.effective_d_ff,
                "vocab_size": model_cfg.vocab_size,
                "max_seq_len": model_cfg.max_seq_len,
                "param_count": model_cfg.param_count(),
            },
            "data": {
                "split_dir": cfg.data.split_dir,
                "family_data_dir": cfg.data.family_data_dir,
                "train_size": len(dataset),
                "val_random_size": len(val_dataset) if val_dataset else 0,
                "val_family_size": len(val_family_dataset) if val_family_dataset else 0,
            },
            "curriculum": {
                "enabled": curriculum_sampler is not None,
                "spec_path": cfg.curriculum.spec_path if curriculum_sampler else None,
                "spec_name": curriculum_spec.name if curriculum_spec else None,
                "spec_hash": curriculum_spec.yaml_hash[:16] if curriculum_spec else None,
                "phases": len(curriculum_spec.phases) if curriculum_spec else 0,
                "trace_enabled": trace_writer is not None,
                "trace_path": "sample_trace.jsonl" if trace_writer else None,
            },
            "activations": {
                "enabled": cfg.activations.enabled,
                "early_until_step_config": cfg.activations.early_until_step,
                "early_until_fraction_config": cfg.activations.early_until_fraction,
                "mid_until_step_config": cfg.activations.mid_until_step,
                "mid_until_fraction_config": cfg.activations.mid_until_fraction,
                "early_until_resolved": activation_phases["early_until"],
                "mid_until_resolved": activation_phases["mid_until"],
                "early_every_steps": activation_phases["early_every"],
                "mid_every_steps": activation_phases["mid_every"],
                "late_every_steps": activation_phases["late_every"],
            },
        },
    )

    # Write curriculum manifest if enabled
    if curriculum_sampler is not None and curriculum_spec is not None:
        family_index = dataset.build_family_index()
        family_counts = {f: len(indices) for f, indices in family_index.items()}
        manifest = curriculum_spec.to_manifest(
            total_steps=total_steps,
            available_families=list(family_index.keys()),
            family_counts=family_counts,
            seed=cfg.seed,
        )
        manifest_path = paths.run_dir(run_id) / "curriculum_manifest.json"
        write_curriculum_manifest(manifest, manifest_path)
        print(f"Curriculum manifest written to {manifest_path}")

    # Training loop
    scalar_rows: List[Dict] = []
    pbar = tqdm(range(total_steps), desc=f"Research[{run_id}] ({device})")

    model.train()
    accumulated_loss = 0.0
    accumulation_steps = 0

    # Loss delta tracking
    start_loss: Optional[float] = None
    epoch_start_loss: Optional[float] = None
    epoch_losses: Dict[int, Dict[str, float]] = {}  # epoch -> {start, end, delta}

    # Loss milestone tracking (for analysis, not termination)
    # Stable definition: milestone at p means loss <= baseline * (1 - p)
    # Baseline computed from first K steps after warmup for stability
    best_loss: Optional[float] = None
    loss_milestones_achieved: List[Dict] = []  # List of milestone records
    milestone_targets_remaining: List[float] = []  # Milestones not yet achieved

    # Baseline tracking for relative milestones
    baseline_steps_k = 10  # Number of steps to average for baseline
    baseline_losses: List[float] = []  # Collect losses for baseline computation
    baseline_loss: Optional[float] = None  # Computed baseline
    milestone_absolute_thresholds: Dict[float, float] = {}  # p -> absolute loss threshold

    # EMA smoothing for current loss (alpha=0.1 for stability)
    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.1

    if cfg.training.loss_milestones.enabled:
        if cfg.training.loss_milestones.mode == "relative":
            # Will compute thresholds once baseline is established
            milestone_targets_remaining = sorted(
                cfg.training.loss_milestones.relative_fractions
            )  # Start from smallest (easiest)
        elif cfg.training.loss_milestones.absolute_values:
            milestone_targets_remaining = sorted(
                cfg.training.loss_milestones.absolute_values, reverse=True
            )  # Start from highest (easiest)

    for step in pbar:
        # Update curriculum sampler step if enabled
        if curriculum_sampler is not None:
            curriculum_sampler.set_step(step)

        # Gradient accumulation loop
        for accum_idx in range(cfg.training.gradient_accumulation_steps):
            if curriculum_sampler is not None:
                # Use curriculum sampler to get batch indices
                batch_indices = curriculum_sampler.sample_batch()
                # Manually collate batch from dataset
                batch_items = [dataset[idx] for idx in batch_indices]
                input_ids = torch.stack([item["input_ids"] for item in batch_items]).to(device)
                labels = torch.stack([item["labels"] for item in batch_items]).to(device)

                # Write trace if enabled (log ALL microbatches for complete attribution)
                if trace_writer is not None:
                    # Get family and item IDs for trace
                    family_ids = [dataset.family_ids[idx] for idx in batch_indices]
                    # Use raw item's id field if available, otherwise use index
                    item_ids = []
                    for idx in batch_indices:
                        raw_item = dataset.items[idx]
                        item_id = raw_item.get("id") or raw_item.get("item_id") or str(idx)
                        item_ids.append(item_id)
                    current_phase = curriculum_sampler._get_current_phase()
                    phase_name = current_phase.name if current_phase else "unknown"
                    phase_idx = curriculum_sampler._get_current_phase_idx()
                    trace_writer.write_batch(
                        step=step,
                        micro=accum_idx,
                        phase_name=phase_name,
                        phase_idx=phase_idx,
                        family_ids=family_ids,
                        item_ids=item_ids,
                    )
            else:
                # Standard dataloader iteration
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

            _, loss = model(input_ids, labels)
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            accumulation_steps += 1

        # Gradient clipping
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # Compute grad norms before optimizer step
        grad_norms = {}
        if cfg.scalars.grad_norm_global:
            grad_norms = _compute_grad_norms(model, cfg.scalars.grad_norm_per_block)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        current_lr = scheduler.get_last_lr()[0]
        avg_loss = accumulated_loss
        accumulated_loss = 0.0

        # Loss delta tracking
        if start_loss is None:
            start_loss = avg_loss
        if epoch_start_loss is None:
            epoch_start_loss = avg_loss

        # Track best loss
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss

        # Update loss EMA for smoothed milestone checking
        if loss_ema is None:
            loss_ema = avg_loss
        else:
            loss_ema = loss_ema_alpha * avg_loss + (1 - loss_ema_alpha) * loss_ema

        # Loss milestone tracking
        if cfg.training.loss_milestones.enabled:
            # Collect baseline losses (first K steps after warmup)
            if baseline_loss is None and step >= warmup_steps:
                baseline_losses.append(avg_loss)
                if len(baseline_losses) >= baseline_steps_k:
                    baseline_loss = sum(baseline_losses) / len(baseline_losses)
                    # Compute absolute thresholds for relative milestones
                    if cfg.training.loss_milestones.mode == "relative":
                        for p in milestone_targets_remaining:
                            # milestone at p means loss <= baseline * (1 - p)
                            milestone_absolute_thresholds[p] = baseline_loss * (1 - p)
                        print(f"\n[Milestones] Baseline loss: {baseline_loss:.4f} (from steps {warmup_steps}-{step})")
                        print(f"[Milestones] Thresholds: {', '.join(f'{p*100:.0f}%={t:.4f}' for p, t in sorted(milestone_absolute_thresholds.items()))}")

            # Check milestones once baseline is established
            if baseline_loss is not None and milestone_targets_remaining:
                if cfg.training.loss_milestones.mode == "relative":
                    # Check each milestone using smoothed loss
                    for milestone_frac in list(milestone_targets_remaining):
                        target_loss = milestone_absolute_thresholds[milestone_frac]
                        if loss_ema <= target_loss:
                            loss_milestones_achieved.append({
                                "step": step,
                                "loss_raw": avg_loss,
                                "loss_smooth": loss_ema,
                                "milestone_value": milestone_frac,
                                "milestone_type": "relative",
                                "target_loss": target_loss,
                                "baseline_loss": baseline_loss,
                                "lr": current_lr,
                                "epoch_float": step / steps_per_epoch if steps_per_epoch > 0 else 0,
                            })
                            milestone_targets_remaining.remove(milestone_frac)
                else:
                    # Absolute mode: check if we've crossed each absolute threshold
                    for target_loss in list(milestone_targets_remaining):
                        if loss_ema <= target_loss:
                            loss_milestones_achieved.append({
                                "step": step,
                                "loss_raw": avg_loss,
                                "loss_smooth": loss_ema,
                                "milestone_value": target_loss,
                                "milestone_type": "absolute",
                                "target_loss": target_loss,
                                "lr": current_lr,
                                "epoch_float": step / steps_per_epoch if steps_per_epoch > 0 else 0,
                            })
                            milestone_targets_remaining.remove(target_loss)

        # Track epoch boundary (start of new epoch)
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        prev_epoch = (step - 1) // steps_per_epoch if step > 0 and steps_per_epoch > 0 else 0

        if step > 0 and current_epoch != prev_epoch:
            # New epoch starting - store previous epoch's end loss
            prev_epoch_num = prev_epoch
            if prev_epoch_num not in epoch_losses:
                epoch_losses[prev_epoch_num] = {"start": epoch_start_loss, "end": avg_loss}
                epoch_losses[prev_epoch_num]["delta"] = (
                    epoch_losses[prev_epoch_num]["start"] - epoch_losses[prev_epoch_num]["end"]
                )

            # Reset epoch tracking for new epoch
            epoch_start_loss = avg_loss

        # Build scalar row
        row = {
            "run_id": run_id,
            "step": step,
            "loss": avg_loss,
            "lr": current_lr,
        }
        row.update(grad_norms)

        # Add curriculum phase info if enabled
        if curriculum_sampler is not None:
            current_phase = curriculum_sampler._get_current_phase()
            if current_phase is not None:
                row["curriculum_phase"] = current_phase.name

            # Log distribution periodically
            if (
                cfg.curriculum.log_distribution_every > 0
                and step % cfg.curriculum.log_distribution_every == 0
                and step > 0
            ):
                if current_phase is not None:
                    weights = curriculum_sampler._get_current_weights(current_phase)
                    print(f"Step {step} - Phase: {current_phase.name}, Weights: {weights}")

        # Fixed probe evaluation
        if cfg.probes.fixed.enabled and step % cfg.probes.fixed_every_steps == 0:
            if probe_fixed is not None:
                probe_metrics = _eval_probe(model, probe_fixed, probe_fixed)
                row["probe_fixed_loss"] = probe_metrics["loss"]
                row["probe_fixed_acc"] = probe_metrics["accuracy"]
            model.train()

        # Extended probe at milestones
        if cfg.probes.extended.enabled and step in milestone_steps:
            if probe_extended is not None:
                extended_metrics = _eval_probe(model, probe_extended, probe_extended)
                row["probe_extended_loss"] = extended_metrics["loss"]
                row["probe_extended_acc"] = extended_metrics["accuracy"]

                # Detailed metrics
                detailed = _compute_probe_metrics(model, probe_extended[:256], cfg)
                for k, v in detailed.items():
                    row[f"probe_{k}"] = float(v)
            model.train()

        # Validation at epoch boundaries (or step intervals)
        run_val = False
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

        if cfg.training.epochs is not None and cfg.training.val_every_epoch:
            # Run validation at the end of each epoch
            if step > 0 and (step + 1) % steps_per_epoch == 0:
                run_val = True
        elif cfg.training.val_every_steps is not None:
            # Run validation every N steps
            if step > 0 and step % cfg.training.val_every_steps == 0:
                run_val = True

        if run_val and val_dataset is not None:
            val_metrics = _evaluate_validation(model, val_dataset, device)
            row["val_random_loss"] = val_metrics["val_loss"]
            row["val_random_acc"] = val_metrics["val_accuracy"]

            if val_family_dataset is not None:
                val_family_metrics = _evaluate_validation(model, val_family_dataset, device)
                row["val_family_loss"] = val_family_metrics["val_loss"]
                row["val_family_acc"] = val_family_metrics["val_accuracy"]

            # Show loss delta for this epoch
            epoch_delta_str = ""
            if current_epoch in epoch_losses:
                delta = epoch_losses[current_epoch]["delta"]
                epoch_delta_str = f", delta={delta:.4f}"

            if not quiet:
                print(
                    f"\n[Epoch {current_epoch + 1}] Val-random: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_accuracy']:.4f}{epoch_delta_str}"
                )
                if val_family_dataset is not None:
                    print(
                        f"           Val-family: loss={val_family_metrics['val_loss']:.4f}, acc={val_family_metrics['val_accuracy']:.4f}"
                    )
            model.train()

        scalar_rows.append(row)

        # Progress bar
        if step % 10 == 0:
            postfix = {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
            if "probe_fixed_acc" in row:
                postfix["acc"] = f"{row['probe_fixed_acc']:.4f}"
            pbar.set_postfix(**postfix)

        # Checkpointing
        if _should_checkpoint(step, cfg):
            ckpt_path = _save_checkpoint(model, optimizer, scheduler, step, run_id, cfg)
            if not quiet:
                print(f"\n[Checkpoint] Saved: {ckpt_path}")

        # Activation capture (uses resolved phase boundaries)
        if _should_capture_activations(step, cfg, activation_phases):
            capture_input = (
                probe_fixed[: cfg.activations.probe_n_examples]
                if probe_fixed is not None
                else input_ids
            )
            _capture_activations(model, capture_input, run_id, step, cfg)

    # Final checkpoint
    _save_checkpoint(model, optimizer, scheduler, total_steps, run_id, cfg)

    # Record final epoch loss delta
    final_epoch = total_steps // steps_per_epoch if steps_per_epoch > 0 else 0
    if final_epoch not in epoch_losses and epoch_start_loss is not None:
        epoch_losses[final_epoch] = {
            "start": epoch_start_loss,
            "end": avg_loss,
            "delta": epoch_start_loss - avg_loss,
        }

    # Write scalars
    df = pd.DataFrame(scalar_rows)

    scalar_path = paths.metrics_scalar_path(run_id)
    scalar_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scalar_path, index=False)

    # Write loss milestones if any were achieved
    milestones_path = None
    if loss_milestones_achieved:
        milestones_df = pd.DataFrame(loss_milestones_achieved)
        milestones_df["run_id"] = run_id
        milestones_df["seed"] = cfg.seed
        # Reorder columns for clarity
        col_order = ["run_id", "seed", "step", "epoch_float", "milestone_value", "milestone_type",
                     "loss_raw", "loss_smooth", "target_loss", "baseline_loss", "lr"]
        milestones_df = milestones_df[[c for c in col_order if c in milestones_df.columns]]
        milestones_path = paths.run_dir(run_id) / "milestones.parquet"
        milestones_df.to_parquet(milestones_path, index=False)

    # Update meta.json with milestone summary
    meta_data = json.loads(meta_path.read_text())
    meta_data["loss_milestones_summary"] = {
        "achieved_count": len(loss_milestones_achieved),
        "milestones": loss_milestones_achieved,
        # Baseline tracking
        "baseline_loss": baseline_loss,
        "baseline_steps_used": baseline_steps_k,
        "baseline_warmup_offset": warmup_steps,
        "smoothing": {"method": "ema", "alpha": loss_ema_alpha},
        # Absolute thresholds (computed from baseline, useful for comparison)
        "thresholds_absolute": milestone_absolute_thresholds if baseline_loss else None,
        # Raw loss stats
        "start_loss": start_loss,
        "best_loss": best_loss,
        "final_loss": avg_loss,
        "final_loss_smooth": loss_ema,
    }
    meta_path.write_text(json.dumps(meta_data, indent=2))

    # Loss delta summary
    end_loss = avg_loss
    total_delta = (start_loss - end_loss) if start_loss is not None else 0.0

    print(f"\n[✓] Research run complete: {run_id}")
    print("\n--- Loss Summary ---")
    print(f"Start loss:    {start_loss:.4f}" if start_loss else "Start loss:    N/A")
    print(f"Baseline loss: {baseline_loss:.4f}" if baseline_loss else "Baseline loss: N/A")
    print(f"Best loss:     {best_loss:.4f}" if best_loss else "Best loss:     N/A")
    print(f"End loss:      {end_loss:.4f} (smooth: {loss_ema:.4f})" if loss_ema else f"End loss:      {end_loss:.4f}")
    print(f"Total delta:   {total_delta:.4f}")

    if loss_milestones_achieved:
        print("\n--- Loss Milestones (fractional reduction from baseline) ---")
        for m in loss_milestones_achieved:
            if m["milestone_type"] == "relative":
                pct = m['milestone_value'] * 100
                print(f"  {pct:.0f}% reduction at step {m['step']} (smooth={m['loss_smooth']:.4f}, target={m['target_loss']:.4f})")
            else:
                print(f"  Loss {m['milestone_value']:.4f} at step {m['step']} (smooth={m['loss_smooth']:.4f})")
    if epoch_losses:
        print("\nPer-epoch deltas:")
        for epoch_num in sorted(epoch_losses.keys()):
            ep_data = epoch_losses[epoch_num]
            print(
                f"  Epoch {epoch_num + 1}: {ep_data['start']:.4f} -> {ep_data['end']:.4f} (delta={ep_data['delta']:.4f})"
            )

        # Check validity: epoch 1 should have biggest delta
        if len(epoch_losses) > 1:
            epoch1_delta = epoch_losses.get(0, {}).get("delta", 0)
            later_deltas = [epoch_losses[e]["delta"] for e in epoch_losses if e > 0]
            if later_deltas and epoch1_delta > 0:
                if epoch1_delta > max(later_deltas):
                    print("  [OK] Epoch 1 has largest delta - healthy learning")
                else:
                    print("  [WARN] Epoch 1 delta smaller than later epochs - check LR")
    print(f"    Meta: {meta_path}")
    print(f"    Scalars: {scalar_path}")
    if milestones_path:
        print(f"    Milestones: {milestones_path}")
    print(f"    Checkpoints: {paths.run_dir(run_id) / 'checkpoints'}")
    print(f"    Captures: {paths.captures_dir(run_id)}")

    # Close trace writer if enabled
    if trace_writer is not None:
        trace_writer.__exit__(None, None, None)
        print(f"    Trace: {paths.run_dir(run_id) / 'sample_trace.jsonl'}")

    # Wait for any pending checkpoint moves to complete and cleanup staging
    if _checkpoint_move_executor is not None:
        _checkpoint_move_executor.shutdown(wait=True)
        # Cleanup staging directory for this run
        stage_dir = _CHECKPOINT_STAGE_DIR / run_id
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-grade training")
    parser.add_argument("--config", help="Path to research YAML config")
    parser.add_argument(
        "--preset",
        choices=["debug", "default"],
        help="Use a preset config instead of YAML file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output (checkpoint saves, epoch validation)",
    )
    args = parser.parse_args()

    if args.preset:
        if args.preset == "debug":
            cfg = get_research_debug_config()
        else:
            cfg = get_research_default_config()
        run_research(cfg=cfg, quiet=args.quiet)
    elif args.config:
        run_research(config_path=args.config, quiet=args.quiet)
    else:
        parser.error("Must specify either --config or --preset")


if __name__ == "__main__":
    main()
