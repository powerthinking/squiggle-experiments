from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import json

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
def _eval_probe(
    model: TinyTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    """
    Returns probe_loss and probe_acc for next-token prediction.

    Assumptions (common for LM training):
      - x: (B, T) token ids
      - y: (B, T) target token ids (or same shape as x)
      - model(x) returns logits (B, T, V)
      - model.loss(x, y) returns scalar loss
    """
    model.eval()

    probe_loss = float(model.loss(x, y).detach().cpu().item())

    # Try to get logits
    logits = model(x)  # expects (B,T,V)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    preds = logits.argmax(dim=-1)  # (B,T)
    correct = (preds == y).float().mean()  # token-accuracy
    probe_acc = float(correct.detach().cpu().item())

    return {"probe_loss": probe_loss, "probe_acc": probe_acc}

class TriggerManager:
    def __init__(self, rules: list[dict]):
        self.rules = rules
        self._fired = set()  # optional: prevent refiring same rule constantly
        self._loss_history: list[tuple[int, float]] = []  # (step, loss)

    def update(self, step: int, metrics: dict) -> list[dict]:
        """
        Return list of trigger events:
          {"name": "...", "step": step, "reason": "..."}
        """
        events = []

        loss = metrics.get("loss")
        probe_acc = metrics.get("probe_acc")

        if isinstance(loss, (int, float)):
            self._loss_history.append((step, float(loss)))
            # keep history bounded
            if len(self._loss_history) > 5000:
                self._loss_history = self._loss_history[-2000:]

        for i, r in enumerate(self.rules):
            rtype = r.get("type", "")
            rule_id = f"{rtype}:{i}"

            # 1) probe_acc crosses threshold upward
            if rtype == "probe_acc_crosses":
                thr = float(r.get("threshold", 0.0))
                if probe_acc is not None and probe_acc >= thr and rule_id not in self._fired:
                    events.append({"name": "probe_acc_crosses", "step": step, "reason": f"probe_acc>={thr}"})
                    self._fired.add(rule_id)

            # 2) loss drops by min_drop within a rolling window
            if rtype == "loss_drop":
                min_drop = float(r.get("min_drop", 0.0))
                window = int(r.get("window_steps", 200))
                if loss is not None:
                    # find loss from ~window steps ago (nearest)
                    past = [v for (s, v) in self._loss_history if s <= step - window]
                    if past:
                        past_loss = past[-1]
                        if (past_loss - float(loss)) >= min_drop:
                            # allow refire but rate-limit a bit
                            key = (rule_id, step // window)
                            if key not in self._fired:
                                events.append(
                                    {"name": "loss_drop", "step": step, "reason": f"loss dropped {past_loss - float(loss):.3f} over {window} steps"}
                                )
                                self._fired.add(key)

        return events

@torch.no_grad()
def _capture_step(
    run_id: str,
    step: int,
    model: TinyTransformerLM,
    input_ids: torch.Tensor,
    layers_to_capture: List[int],
    capture_embeddings: bool,
    capture_residuals: bool,
    source: str,
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
    (out_dir / "sample_meta.json").write_text(
        json.dumps({"source": source}, indent=2)
    )

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

    run_id = make_run_id(cfg.run_name, cfg.seed)

    task = AdditionModTask(p=cfg.task.p)

    probe_x = None
    probe_y = None

    trigger_mgr = None
    if getattr(cfg, "triggers", None) and cfg.triggers.enabled:
        # convert dataclass rules to dicts if needed
        rules = []
        for r in cfg.triggers.rules:    
            rules.append(r.__dict__ if hasattr(r, "__dict__") else dict(r))
        trigger_mgr = TriggerManager(rules)


    if cfg.probes.fixed.enabled:
        set_seed(cfg.probes.fixed.seed)

        probe_x, probe_y = task.sample_batch(cfg.probes.fixed.n_examples, device=device)

        probe_path = paths.run_dir(run_id) / "probe_fixed.pt"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": probe_x.detach().cpu(), "y": probe_y.detach().cpu()}, probe_path)

        # Restore training seed
        set_seed(cfg.seed)

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

        probe_loss = None
        probe_acc = None

        if (probe_x is not None) and (step % cfg.probe_eval.every_steps == 0):
            m = _eval_probe(model, probe_x, probe_y)
            probe_loss = m["probe_loss"]
            probe_acc = m["probe_acc"]

        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "lr": float(lr),
                "probe_loss": probe_loss,
                "probe_acc": probe_acc,
            }
        )



        if trigger_mgr is not None:
            metrics = {"loss": float(rows[-1]["loss"]), "probe_acc": probe_acc}
            trig_events = trigger_mgr.update(step, metrics)

            for ev in trig_events:
                # Triggered capture (same capture_ids you already use for v0)
                is_probe = probe_x is not None
                capture_ids = probe_x if is_probe else x

                _capture_step(
                    run_id=run_id,
                    step=step,
                    model=model,
                    input_ids=capture_ids,
                    layers_to_capture=cfg.capture.layers,
                    capture_embeddings=cfg.capture.embeddings,
                    capture_residuals=cfg.capture.residuals,
                    source=f"trigger:{ev['name']}",
                )


        if (step % 10) == 0:
            postfix = {"loss": f"{rows[-1]['loss']:.4f}"}
            if probe_acc is not None:
                postfix["probe_acc"] = f"{probe_acc:.3f}"
            pbar.set_postfix(**postfix) 

        if step % cfg.capture.every_steps == 0:
            is_probe = probe_x is not None
            source = "probe_fixed" if is_probe else "train_batch"
            capture_ids = probe_x if is_probe else x

            _capture_step(
                run_id=run_id,
                step=step,
                model=model,
                input_ids=capture_ids,
                layers_to_capture=cfg.capture.layers,
                capture_embeddings=cfg.capture.embeddings,
                capture_residuals=cfg.capture.residuals,
                source=source,
            )


    # Write scalars
    out_path = paths.metrics_scalar_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)

    print(f"[✓] Scout run complete: {run_id}")
    print(f"    Meta:     {paths.run_dir(run_id) / 'meta.json'}")
    print(f"    Scalars:  {paths.metrics_scalar_path(run_id)}")
    print(f"    Samples:  {paths.samples_dir(run_id)}")
    df = pd.DataFrame(rows)
    last_probe = df.dropna(subset=["probe_acc"]).iloc[-1]
    print(f"[probe] step={int(last_probe.step)} acc={last_probe.probe_acc:.4f} loss={last_probe.probe_loss:.6f}")
    
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run Scout thin-slice training")
    parser.add_argument("--config", required=True, help="Path to scout YAML config")
    args = parser.parse_args()
    run_scout(args.config)


if __name__ == "__main__":
    main()
