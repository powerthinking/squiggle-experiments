from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

@dataclass(frozen=True)
class FixedProbeCfg:
    enabled: bool = True
    n_examples: int = 256
    seed: int = 123


@dataclass(frozen=True)
class ProbesCfg:
    fixed: FixedProbeCfg = field(default_factory=FixedProbeCfg)


@dataclass(frozen=True)
class InstrumentationCfg:
    include_embedding: bool = True
    layers: List[int] = field(default_factory=lambda: [0, 3])

@dataclass(frozen=True)
class ScoutModelCfg:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class ScoutTaskCfg:
    p: int = 97


@dataclass(frozen=True)
class ScoutCaptureCfg:
    every_steps: int = 50
    layers: List[int] = None  # layers to save
    embeddings: bool = True
    residuals: bool = True


@dataclass(frozen=True)
class ScoutCfg:
    run_name: str = "scout_tiny"
    seed: int = 1337
    steps: int = 500
    batch_size: int = 64
    lr: float = 3e-4
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    model: ScoutModelCfg = ScoutModelCfg()
    task: ScoutTaskCfg = ScoutTaskCfg()
    capture: ScoutCaptureCfg = ScoutCaptureCfg(layers=[0, 3])
    probes: ProbesCfg = field(default_factory=ProbesCfg)
    instrumentation: InstrumentationCfg = field(default_factory=InstrumentationCfg)


def _coerce_model(d: Dict[str, Any]) -> ScoutModelCfg:
    return ScoutModelCfg(**d)


def _coerce_task(d: Dict[str, Any]) -> ScoutTaskCfg:
    return ScoutTaskCfg(**d)


def _coerce_capture(d: Dict[str, Any]) -> ScoutCaptureCfg:
    if "layers" not in d or d["layers"] is None:
        d["layers"] = [0]
    return ScoutCaptureCfg(**d)


def load_scout_config(path: str | Path) -> ScoutCfg:
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    model = _coerce_model(raw.get("model", {}))
    task = _coerce_task(raw.get("task", {}))
    capture = _coerce_capture(raw.get("capture", {}))

    return ScoutCfg(
        run_name=raw.get("run_name", "scout_tiny"),
        seed=int(raw.get("seed", 1337)),
        steps=int(raw.get("steps", 500)),
        batch_size=int(raw.get("batch_size", 64)),
        lr=float(raw.get("lr", 3e-4)),
        device=str(raw.get("device", "auto")),
        model=model,
        task=task,
        capture=capture,
    )
