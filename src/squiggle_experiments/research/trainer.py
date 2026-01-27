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
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from squiggle_core import paths
from squiggle_experiments.models.research_transformer import (
    ResearchTransformerConfig,
    ResearchTransformerLM,
    get_research_config_350m,
    get_research_config_1b,
    get_research_config_debug,
)
from squiggle_experiments.utils.logging import write_meta_json
from squiggle_experiments.utils.run_id import make_run_id
from squiggle_experiments.utils.seed import set_seed

from .config import (
    ResearchCfg,
    load_research_config,
    get_research_default_config,
    get_research_debug_config,
)
from .data import FamilyDataset, get_tokenizer


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


def _should_capture_activations(step: int, cfg: ResearchCfg) -> bool:
    """Determine if we should capture activations at this step."""
    if not cfg.activations.enabled:
        return False
    if step < cfg.activations.early_until_step:
        return step % cfg.activations.early_every_steps == 0
    if step < cfg.activations.mid_until_step:
        return step % cfg.activations.mid_every_steps == 0
    return step % cfg.activations.late_every_steps == 0


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
    norms["grad_norm_global"] = total_norm ** 0.5

    # Per-block norms if requested
    if per_block and hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            block_norm = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    block_norm += p.grad.data.norm(2).item() ** 2
            norms[f"grad_norm_block_{i}"] = block_norm ** 0.5

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
            manifest["tensors"][f"layer_{i}_attn_out"] = {"path": fname, "shape": list(attn_out.shape)}

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
            manifest["tensors"][f"layer_{i}_mlp_out"] = {"path": fname, "shape": list(mlp_out.shape)}

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
    """Save model checkpoint."""
    ckpt_dir = paths.run_dir(run_id) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"step_{step:06d}.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": asdict(model.cfg) if hasattr(model.cfg, "__dataclass_fields__") else model.cfg.__dict__,
    }

    if cfg.checkpoints.save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)

    # Cleanup old checkpoints
    if cfg.checkpoints.keep_last_n > 0:
        existing = sorted(ckpt_dir.glob("step_*.pt"))
        if len(existing) > cfg.checkpoints.keep_last_n:
            for old_ckpt in existing[: -cfg.checkpoints.keep_last_n]:
                old_ckpt.unlink()

    return ckpt_path


def run_research(config_path: Optional[str] = None, cfg: Optional[ResearchCfg] = None) -> str:
    """Run research-grade training with comprehensive logging."""

    if cfg is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or cfg")
        cfg = load_research_config(config_path)

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

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler
    total_steps = cfg.training.steps
    if cfg.training.lr_schedule == "cosine":
        scheduler = _get_cosine_schedule_with_warmup(
            optimizer,
            cfg.training.warmup_steps,
            total_steps,
            cfg.training.min_lr_ratio,
        )
    else:
        # Constant or linear - for now just use constant
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Dataset setup
    tokenizer = None
    if cfg.data.family_data_dir:
        # Use real family data
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
            size=cfg.training.steps * cfg.training.batch_size * cfg.training.gradient_accumulation_steps,
            seed=cfg.seed,
        )
        print("Using dummy dataset (no family_data_dir specified)")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
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

    # Write meta.json
    meta_path = paths.run_dir(run_id) / "meta.json"
    write_meta_json(
        meta_path,
        {
            "run_id": run_id,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "steps": cfg.training.steps,
            "batch_size": cfg.training.batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "lr": cfg.training.lr,
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
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )

    # Training loop
    scalar_rows: List[Dict] = []
    pbar = tqdm(range(total_steps), desc=f"Research[{run_id}] ({device})")

    model.train()
    accumulated_loss = 0.0
    accumulation_steps = 0

    for step in pbar:
        # Gradient accumulation loop
        for _ in range(cfg.training.gradient_accumulation_steps):
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

        # Build scalar row
        row = {
            "run_id": run_id,
            "step": step,
            "loss": avg_loss,
            "lr": current_lr,
        }
        row.update(grad_norms)

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
            print(f"\n[Checkpoint] Saved: {ckpt_path}")

        # Activation capture
        if _should_capture_activations(step, cfg):
            capture_input = probe_fixed[:cfg.activations.probe_n_examples] if probe_fixed is not None else input_ids
            _capture_activations(model, capture_input, run_id, step, cfg)

    # Final checkpoint
    _save_checkpoint(model, optimizer, scheduler, total_steps, run_id, cfg)

    # Write scalars
    df = pd.DataFrame(scalar_rows)

    scalar_path = paths.metrics_scalar_path(run_id)
    scalar_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scalar_path, index=False)

    print(f"\n[✓] Research run complete: {run_id}")
    print(f"    Meta: {meta_path}")
    print(f"    Scalars: {scalar_path}")
    print(f"    Checkpoints: {paths.run_dir(run_id) / 'checkpoints'}")
    print(f"    Captures: {paths.captures_dir(run_id)}")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-grade training")
    parser.add_argument("--config", help="Path to research YAML config")
    parser.add_argument(
        "--preset",
        choices=["debug", "default"],
        help="Use a preset config instead of YAML file",
    )
    args = parser.parse_args()

    if args.preset:
        if args.preset == "debug":
            cfg = get_research_debug_config()
        else:
            cfg = get_research_default_config()
        run_research(cfg=cfg)
    elif args.config:
        run_research(config_path=args.config)
    else:
        parser.error("Must specify either --config or --preset")


if __name__ == "__main__":
    main()
