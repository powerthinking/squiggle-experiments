from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# -------------------------
# Probe config
# -------------------------

@dataclass(frozen=True)
class FixedProbeCfg:
    enabled: bool = True
    n_examples: int = 256
    seed: int = 123


@dataclass(frozen=True)
class HoldoutProbeCfg:
    enabled: bool = False
    n_examples: int = 256
    seed: int = 456


@dataclass(frozen=True)
class ProbesCfg:
    fixed: FixedProbeCfg = field(default_factory=FixedProbeCfg)
    holdout: HoldoutProbeCfg = field(default_factory=HoldoutProbeCfg)


# -------------------------
# Instrumentation / eval / triggers
# -------------------------

@dataclass(frozen=True)
class InstrumentationCfg:
    include_embedding: bool = True
    layers: List[int] = field(default_factory=lambda: [0, 3])


@dataclass(frozen=True)
class ProbeEvalCfg:
    every_steps: int = 50


@dataclass(frozen=True)
class TriggerRuleCfg:
    type: str
    threshold: Optional[float] = None
    min_drop: Optional[float] = None
    window_steps: Optional[int] = None


@dataclass(frozen=True)
class TriggersCfg:
    enabled: bool = False
    rules: List[TriggerRuleCfg] = field(default_factory=list)


# -------------------------
# Model / task / capture
# -------------------------

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
    layers: List[int] = field(default_factory=lambda: [0, 3])
    embeddings: bool = True
    residuals: bool = True
    # "probe_fixed" | "train_batch" | "mixed"
    source: str = "probe_fixed"


# -------------------------
# Root config
# -------------------------

@dataclass(frozen=True)
class ScoutCfg:
    run_name: str = "scout_tiny"
    seed: int = 1337
    steps: int = 500
    batch_size: int = 64
    lr: float = 3e-4
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    model: ScoutModelCfg = field(default_factory=ScoutModelCfg)
    task: ScoutTaskCfg = field(default_factory=ScoutTaskCfg)
    capture: ScoutCaptureCfg = field(default_factory=ScoutCaptureCfg)

    probes: ProbesCfg = field(default_factory=ProbesCfg)
    instrumentation: InstrumentationCfg = field(default_factory=InstrumentationCfg)
    probe_eval: ProbeEvalCfg = field(default_factory=ProbeEvalCfg)
    triggers: TriggersCfg = field(default_factory=TriggersCfg)


# -------------------------
# Coercion helpers
# -------------------------

def _coerce_model(d: Dict[str, Any]) -> ScoutModelCfg:
    return ScoutModelCfg(**d)


def _coerce_task(d: Dict[str, Any]) -> ScoutTaskCfg:
    return ScoutTaskCfg(**d)


def _coerce_capture(d: Dict[str, Any]) -> ScoutCaptureCfg:
    dd = dict(d or {})
    # Layers default if missing/None
    if dd.get("layers") is None:
        dd["layers"] = [0, 3]
    # Source default if missing/None
    if dd.get("source") is None:
        dd["source"] = "probe_fixed"
    return ScoutCaptureCfg(**dd)


def _coerce_fixed_probe(d: Dict[str, Any]) -> FixedProbeCfg:
    return FixedProbeCfg(**(d or {}))


def _coerce_holdout_probe(d: Dict[str, Any]) -> HoldoutProbeCfg:
    return HoldoutProbeCfg(**(d or {}))


def _coerce_probes(d: Dict[str, Any]) -> ProbesCfg:
    dd = dict(d or {})
    fixed = _coerce_fixed_probe(dd.get("fixed", {}))
    holdout = _coerce_holdout_probe(dd.get("holdout", {}))
    return ProbesCfg(fixed=fixed, holdout=holdout)


def _coerce_instrumentation(d: Dict[str, Any]) -> InstrumentationCfg:
    dd = dict(d or {})
    if dd.get("layers") is None:
        dd["layers"] = [0, 3]
    return InstrumentationCfg(**dd)


def _coerce_probe_eval(d: Dict[str, Any]) -> ProbeEvalCfg:
    return ProbeEvalCfg(**(d or {}))


def _coerce_trigger_rule(d: Dict[str, Any]) -> TriggerRuleCfg:
    return TriggerRuleCfg(**d)


def _coerce_triggers(d: Dict[str, Any]) -> TriggersCfg:
    dd = dict(d or {})
    enabled = bool(dd.get("enabled", False))
    rules_raw = dd.get("rules", []) or []
    rules = [_coerce_trigger_rule(r) for r in rules_raw]
    return TriggersCfg(enabled=enabled, rules=rules)


# -------------------------
# Loader
# -------------------------

def load_scout_config(path: str | Path) -> ScoutCfg:
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}

    model = _coerce_model(raw.get("model", {}))
    task = _coerce_task(raw.get("task", {}))
    capture = _coerce_capture(raw.get("capture", {}))

    probes = _coerce_probes(raw.get("probes", {}))
    instrumentation = _coerce_instrumentation(raw.get("instrumentation", {}))
    probe_eval = _coerce_probe_eval(raw.get("probe_eval", {}))
    triggers = _coerce_triggers(raw.get("triggers", {}))

    return ScoutCfg(
        run_name=str(raw.get("run_name", "scout_tiny")),
        seed=int(raw.get("seed", 1337)),
        steps=int(raw.get("steps", 500)),
        batch_size=int(raw.get("batch_size", 64)),
        lr=float(raw.get("lr", 3e-4)),
        device=str(raw.get("device", "auto")),

        model=model,
        task=task,
        capture=capture,

        probes=probes,
        instrumentation=instrumentation,
        probe_eval=probe_eval,
        triggers=triggers,
    )
