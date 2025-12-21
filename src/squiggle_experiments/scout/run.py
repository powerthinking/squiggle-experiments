from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

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
      - y: (B, T) target token ids (same shape as x)
      - model(x) returns logits (B, T, V)
      - model.loss(x, y) returns scalar loss
    """
    model.eval()

    probe_loss = float(model.loss(x, y).detach().cpu().item())

    logits = model(x)  # (B,T,V)
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
        events: list[dict] = []

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
                    past = [v for (s, v) in self._loss_history if s <= step - window]
                    if past:
                        past_loss = past[-1]
                        if (past_loss - float(loss)) >= min_drop:
                            key = (rule_id, step // window)  # rate-limit
                            if key not in self._fired:
                                events.append(
                                    {
                                        "name": "loss_drop",
                                        "step": step,
                                        "reason": f"loss dropped {past_loss - float(loss):.3f} over {window} steps",
                                    }
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
    x = tok + pos_emb  # (B,T,D)

    out_dir = paths.samples_dir(run_id) / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sample_meta.json").write_text(json.dumps({"source": source}, indent=2))

    manifest = {
        "run_id": run_id,
        "step": step,
        "source": source,
        "tensors": {},
    }

    # Save tensors and record them in manifest
    if capture_embeddings:
        torch.save(x.detach().cpu(), out_dir / "embed.pt")
        manifest["tensors"]["embed"] = {
            "path": "embed.pt",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
        }

    if capture_residuals:
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i in layers_to_capture:
                fname = f"resid_layer_{i:02d}.pt"
                torch.save(x.detach().cpu(), out_dir / fname)
                manifest["tensors"][f"resid_layer_{i:02d}"] = {
                    "path": fname,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_scout(config_path: str) -> str:
    cfg = load_scout_config(config_path)
    set_seed(cfg.seed)

    device = _pick_device(cfg.device)

    run_id = make_run_id(cfg.run_name, cfg.seed)

    task = AdditionModTask(p=cfg.task.p)

    # ---- probes (A=fixed, B=holdout) ----
    probe_x_A = probe_y_A = None
    probe_x_B = probe_y_B = None

    if getattr(cfg, "probes", None) and getattr(cfg.probes, "fixed", None) and cfg.probes.fixed.enabled:
        set_seed(cfg.probes.fixed.seed)
        probe_x_A, probe_y_A = task.sample_batch(cfg.probes.fixed.n_examples, device=device)

        probe_path = paths.run_dir(run_id) / "probe_fixed_A.pt"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": probe_x_A.detach().cpu(), "y": probe_y_A.detach().cpu()}, probe_path)

        set_seed(cfg.seed)  # restore training seed

    if getattr(cfg, "probes", None) and getattr(cfg.probes, "holdout", None) and cfg.probes.holdout.enabled:
        set_seed(cfg.probes.holdout.seed)
        probe_x_B, probe_y_B = task.sample_batch(cfg.probes.holdout.n_examples, device=device)

        probe_path = paths.run_dir(run_id) / "probe_fixed_B.pt"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": probe_x_B.detach().cpu(), "y": probe_y_B.detach().cpu()}, probe_path)

        set_seed(cfg.seed)  # restore training seed

    # ---- triggers ----
    trigger_mgr = None
    if getattr(cfg, "triggers", None) and cfg.triggers.enabled:
        rules = []
        for r in cfg.triggers.rules:
            rules.append(r.__dict__ if hasattr(r, "__dict__") else dict(r))
        trigger_mgr = TriggerManager(rules)

    # ---- model ----
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

    # ---- meta.json ----
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
                "source": getattr(cfg.capture, "source", "probe_fixed"),
            },
            "probes": {
                "fixed_A": {
                    "enabled": bool(probe_x_A is not None),
                    "n_examples": getattr(getattr(cfg.probes, "fixed", None), "n_examples", None),
                    "seed": getattr(getattr(cfg.probes, "fixed", None), "seed", None),
                    "path": str((paths.run_dir(run_id) / "probe_fixed_A.pt").resolve()),
                },
                "holdout_B": {
                    "enabled": bool(probe_x_B is not None),
                    "n_examples": getattr(getattr(cfg.probes, "holdout", None), "n_examples", None),
                    "seed": getattr(getattr(cfg.probes, "holdout", None), "seed", None),
                    "path": str((paths.run_dir(run_id) / "probe_fixed_B.pt").resolve()),
                },
            },
            "config_path": str(Path(config_path).resolve()),
        },
    )

    # ---- training loop + scalar logging ----
    rows: list[dict] = []
    pbar = tqdm(range(cfg.steps), desc=f"Scout[{run_id}] ({device})")

    model.train()
    for step in pbar:
        x, y = task.sample_batch(cfg.batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()

        # grad norm (every step) before optimizer.step()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        grad_norm = float(total_norm.detach().cpu().item())

        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]

        # probes (on cadence)
        probe_loss_A = probe_acc_A = None
        probe_loss_B = probe_acc_B = None

        if step % cfg.probe_eval.every_steps == 0:
            if probe_x_A is not None:
                mA = _eval_probe(model, probe_x_A, probe_y_A)
                probe_loss_A, probe_acc_A = mA["probe_loss"], mA["probe_acc"]
            if probe_x_B is not None:
                mB = _eval_probe(model, probe_x_B, probe_y_B)
                probe_loss_B, probe_acc_B = mB["probe_loss"], mB["probe_acc"]

        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "lr": float(lr),
                "grad_norm": grad_norm,
                "probe_loss_A": probe_loss_A,
                "probe_acc_A": probe_acc_A,
                "probe_loss_B": probe_loss_B,
                "probe_acc_B": probe_acc_B,
            }
        )

        # triggers (use probe A as primary signal)
        if trigger_mgr is not None:
            metrics = {"loss": float(rows[-1]["loss"]), "probe_acc": probe_acc_A}
            trig_events = trigger_mgr.update(step, metrics)

            for ev in trig_events:
                # Triggered capture: default to probe A if available, else train batch
                capture_ids = probe_x_A if probe_x_A is not None else x
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

        # progress bar
        if (step % 10) == 0:
            postfix = {"loss": f"{rows[-1]['loss']:.2e}"}
            if probe_acc_A is not None:
                postfix["probeA_acc"] = f"{probe_acc_A:.4f}"
            if probe_acc_B is not None:
                postfix["probeB_acc"] = f"{probe_acc_B:.4f}"
            pbar.set_postfix(**postfix)

        # periodic capture (configurable source policy)
        if step % cfg.capture.every_steps == 0:
            capture_source = getattr(cfg.capture, "source", "probe_fixed")

            if capture_source == "train_batch":
                capture_ids = x
                source = "train_batch"

            elif capture_source == "mixed":
                # alternate between probe A (if available) and train batches
                if (step // cfg.capture.every_steps) % 2 == 0 and probe_x_A is not None:
                    capture_ids = probe_x_A
                    source = "probe_fixed_A"
                else:
                    capture_ids = x
                    source = "train_batch"

            else:
                # default: probe A if available, else train batch
                if probe_x_A is not None:
                    capture_ids = probe_x_A
                    source = "probe_fixed_A"
                else:
                    capture_ids = x
                    source = "train_batch"

            _capture_step(
                run_id=run_id,
                step=step,
                model=model,
                input_ids=capture_ids,
                layers_to_capture=cfg.capture.layers,
                capture_embeddings=cfg.capture.embeddings,
                capture_residuals=cfg.capture.residuals,
                source=capture_source,
            )

    # Write scalars
    out_path = paths.metrics_scalar_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)

    print(f"[✓] Scout run complete: {run_id}")
    print(f"    Meta:     {paths.run_dir(run_id) / 'meta.json'}")
    print(f"    Scalars:  {paths.metrics_scalar_path(run_id)}")
    print(f"    Samples:  {paths.samples_dir(run_id)}")

    if "probe_acc_A" in df.columns and df["probe_acc_A"].notna().any():
        lastA = df.dropna(subset=["probe_acc_A", "probe_loss_A"]).iloc[-1]
        print(f"[probeA] step={int(lastA.step)} acc={lastA.probe_acc_A:.4f} loss={lastA.probe_loss_A:.6f}")
    else:
        print("[probeA] no probe rows logged")

    if "probe_acc_B" in df.columns and df["probe_acc_B"].notna().any():
        lastB = df.dropna(subset=["probe_acc_B", "probe_loss_B"]).iloc[-1]
        print(f"[probeB] step={int(lastB.step)} acc={lastB.probe_acc_B:.4f} loss={lastB.probe_loss_B:.6f}")
    else:
        print("[probeB] no probe rows logged")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scout thin-slice training")
    parser.add_argument("--config", required=True, help="Path to scout YAML config")
    args = parser.parse_args()
    run_scout(args.config)


if __name__ == "__main__":
    main()
