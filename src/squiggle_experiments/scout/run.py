from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from squiggle_core import paths

from squiggle_experiments.models.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from squiggle_experiments.tasks.addition_mod import AdditionModTask
from squiggle_experiments.utils.logging import write_meta_json
from squiggle_experiments.utils.run_id import make_run_id
from squiggle_experiments.utils.seed import set_seed

from .config import load_scout_config


def _pick_device(device_setting: str) -> str:
    if device_setting == "cpu":
        return "cpu"
    if device_setting == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _capture_step(
    run_id: str,
    step: int,
    model: TinyTransformerLM,
    input_ids: torch.Tensor,
    layers_to_capture: List[int],
    capture_embeddings: bool,
    capture_residuals: bool,
) -> None:
    """
    Minimal capture for thin slice:
      - embeddings: token+pos embedding output (B,T,D)
      - residual proxy: output of each transformer block (B,T,D) for selected layers
    """
    model.eval()

    # Build embeddings manually so we can save them
    b, t = input_ids.shape
    pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)

    tok = model.tok_emb(input_ids)
    pos_emb = model.pos_emb(pos)
    x = tok + pos_emb

    out_dir = paths.samples_dir(run_id) / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if capture_embeddings:
        torch.save(x.detach().cpu(), out_dir / "embed.pt")

    if capture_residuals:
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i in layers_to_capture:
                torch.save(x.detach().cpu(), out_dir / f"resid_layer_{i:02d}.pt")


def run_scout(config_path: str) -> str:
    cfg = load_scout_config(config_path)
    set_seed(cfg.seed)

    device = _pick_device(cfg.device)

    task = AdditionModTask(p=cfg.task.p)

    model_cfg = TinyTransformerConfig(
        vocab_size=task.vocab_size,
        seq_len=task.seq_len,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
    )

    model = TinyTransformerLM(model_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    run_id = make_run_id(cfg.run_name, cfg.seed)

    # Meta.json
    meta_path = paths.run_dir(run_id) / "meta.json"
    write_meta_json(
        meta_path,
        {
            "run_id": run_id,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "device": device,
            "task": {"type": "addition_mod", "p": cfg.task.p},
            "model": {
                "d_model": cfg.model.d_model,
                "n_layers": cfg.model.n_layers,
                "n_heads": cfg.model.n_heads,
                "d_ff": cfg.model.d_ff,
                "dropout": cfg.model.dropout,
                "vocab_size": task.vocab_size,
                "seq_len": task.seq_len,
            },
            "capture": {
                "every_steps": cfg.capture.every_steps,
                "layers": cfg.capture.layers,
                "embeddings": cfg.capture.embeddings,
                "residuals": cfg.capture.residuals,
            },
            "config_path": str(Path(config_path).resolve()),
        },
    )

    # Training loop + scalar logging
    rows = []
    pbar = tqdm(range(cfg.steps), desc=f"Scout[{run_id}] ({device})")

    model.train()
    for step in pbar:
        x, y = task.sample_batch(cfg.batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]

        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "lr": float(lr),
            }
        )

        if (step % 10) == 0:
            pbar.set_postfix(loss=f"{rows[-1]['loss']:.4f}")

        if step % cfg.capture.every_steps == 0:
            _capture_step(
                run_id=run_id,
                step=step,
                model=model,
                input_ids=x,
                layers_to_capture=cfg.capture.layers,
                capture_embeddings=cfg.capture.embeddings,
                capture_residuals=cfg.capture.residuals,
            )

    # Write scalars
    out_path = paths.metrics_scalar_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)

    print(f"[✓] Scout run complete: {run_id}")
    print(f"    Meta:     {paths.run_dir(run_id) / 'meta.json'}")
    print(f"    Scalars:  {paths.metrics_scalar_path(run_id)}")
    print(f"    Samples:  {paths.samples_dir(run_id)}")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run Scout thin-slice training")
    parser.add_argument("--config", required=True, help="Path to scout YAML config")
    args = parser.parse_args()
    run_scout(args.config)


if __name__ == "__main__":
    main()
