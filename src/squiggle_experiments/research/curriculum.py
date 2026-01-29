"""Curriculum-driven training support.

This module implements time-varying sampling policies over classified problem families.
A curriculum maps training time t (step or epoch fraction) to a distribution over families.

Key components:
- CurriculumSpec: Parses and validates curriculum YAML, resolves phase boundaries
- CurriculumSampler: PyTorch Sampler that produces curriculum-aware batch indices
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

import yaml


@dataclass
class CurriculumPhase:
    """A single phase in a curriculum specification.

    Phases define time spans with specific family inclusion/exclusion rules
    and sampling behavior.
    """

    name: str
    start_frac: float  # 0.0-1.0 fraction of total steps
    end_frac: float
    families_include: Union[List[str], Literal["*"]]  # "*" means all families
    families_exclude: List[str] = field(default_factory=list)
    weights_type: Literal["uniform", "proportional", "explicit", "ramp"] = "uniform"
    weights_explicit: Optional[Dict[str, float]] = None
    weights_ramp_from: Optional[Dict[str, float]] = None
    weights_ramp_to: Optional[Dict[str, float]] = None
    sampling_mode: Optional[Literal["balanced_family", "proportional_family", "uniform_item"]] = (
        None
    )


@dataclass
class ResolvedPhase:
    """A phase with absolute step boundaries (resolved from fractions)."""

    name: str
    start_step: int
    end_step: int  # exclusive
    families: List[str]  # resolved list of included families
    weights: Dict[str, float]  # normalized weights per family
    sampling_mode: Literal["balanced_family", "proportional_family", "uniform_item"]


@dataclass
class CurriculumSpec:
    """Parsed curriculum specification from YAML.

    Handles YAML parsing, validation, and phase resolution.
    """

    name: str
    version: int
    time_unit: Literal["steps", "epochs"]
    phases: List[CurriculumPhase]
    default_sampling_mode: Literal["balanced_family", "proportional_family", "uniform_item"] = (
        "balanced_family"
    )
    default_replacement: bool = True
    yaml_hash: str = ""  # SHA256 of source YAML for reproducibility

    @classmethod
    def from_yaml(cls, path: Path) -> "CurriculumSpec":
        """Load and parse a curriculum specification from YAML file."""
        yaml_content = path.read_text()
        yaml_hash = hashlib.sha256(yaml_content.encode()).hexdigest()

        data = yaml.safe_load(yaml_content)

        # Parse defaults
        defaults = data.get("defaults", {})
        sampling_defaults = defaults.get("sampling", {})
        default_mode = sampling_defaults.get("mode", "balanced_family")
        default_replacement = sampling_defaults.get("replacement", True)

        # Parse phases
        phases = []
        for phase_data in data.get("phases", []):
            families_data = phase_data.get("families", {})
            families_include = families_data.get("include", "*")
            families_exclude = families_data.get("exclude", [])

            weights_data = phase_data.get("weights", {})
            weights_type = weights_data.get("type", "uniform")
            weights_explicit = weights_data.get("explicit")
            ramp_data = weights_data.get("ramp", {})

            phase_sampling = phase_data.get("sampling", {})
            phase_mode = phase_sampling.get("mode")

            phase = CurriculumPhase(
                name=phase_data["name"],
                start_frac=float(phase_data["start"]),
                end_frac=float(phase_data["end"]),
                families_include=families_include,
                families_exclude=families_exclude if families_exclude else [],
                weights_type=weights_type,
                weights_explicit=weights_explicit,
                weights_ramp_from=ramp_data.get("from"),
                weights_ramp_to=ramp_data.get("to"),
                sampling_mode=phase_mode,
            )
            phases.append(phase)

        return cls(
            name=data.get("name", path.stem),
            version=data.get("version", 1),
            time_unit=data.get("time_unit", "steps"),
            phases=phases,
            default_sampling_mode=default_mode,
            default_replacement=default_replacement,
            yaml_hash=yaml_hash,
        )

    def resolve_phases(
        self,
        total_steps: int,
        available_families: List[str],
        family_counts: Dict[str, int],
    ) -> List[ResolvedPhase]:
        """Convert fractional phase boundaries to absolute steps.

        Args:
            total_steps: Total training steps
            available_families: All families available in the dataset
            family_counts: Number of items per family (for proportional weights)

        Returns:
            List of ResolvedPhase with absolute step boundaries
        """
        resolved = []

        for phase in self.phases:
            start_step = int(phase.start_frac * total_steps)
            end_step = int(phase.end_frac * total_steps)

            # Resolve families
            if phase.families_include == "*":
                families = [f for f in available_families if f not in phase.families_exclude]
            else:
                families = [
                    f
                    for f in phase.families_include
                    if f in available_families and f not in phase.families_exclude
                ]

            # Resolve weights
            if phase.weights_type == "uniform":
                weights = {f: 1.0 for f in families}
            elif phase.weights_type == "proportional":
                weights = {f: float(family_counts.get(f, 0)) for f in families}
            elif phase.weights_type == "explicit":
                weights = {f: phase.weights_explicit.get(f, 0.0) for f in families}
            elif phase.weights_type == "ramp":
                # For ramp, we store the start weights; actual interpolation happens at sample time
                weights = {f: phase.weights_ramp_from.get(f, 0.0) for f in families}
            else:
                weights = {f: 1.0 for f in families}

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {f: w / total_weight for f, w in weights.items()}

            # Resolve sampling mode
            sampling_mode = phase.sampling_mode or self.default_sampling_mode

            resolved.append(
                ResolvedPhase(
                    name=phase.name,
                    start_step=start_step,
                    end_step=end_step,
                    families=families,
                    weights=weights,
                    sampling_mode=sampling_mode,
                )
            )

        return resolved

    def phase_at_step(
        self,
        step: int,
        total_steps: int,
        available_families: List[str],
        family_counts: Dict[str, int],
    ) -> Optional[ResolvedPhase]:
        """Get the resolved phase for a given step.

        Args:
            step: Current training step
            total_steps: Total training steps
            available_families: All families available
            family_counts: Items per family

        Returns:
            ResolvedPhase or None if step is outside all phases
        """
        resolved = self.resolve_phases(total_steps, available_families, family_counts)

        for phase in resolved:
            if phase.start_step <= step < phase.end_step:
                return phase

        return None

    def get_weights_at_step(
        self,
        step: int,
        phase: CurriculumPhase,
        resolved_phase: ResolvedPhase,
        total_steps: int,
    ) -> Dict[str, float]:
        """Get interpolated weights for ramp phases.

        For non-ramp phases, returns the resolved weights.
        For ramp phases, linearly interpolates between from/to weights.
        """
        if phase.weights_type != "ramp":
            return resolved_phase.weights

        # Linear interpolation within phase
        phase_start = int(phase.start_frac * total_steps)
        phase_end = int(phase.end_frac * total_steps)
        phase_progress = (step - phase_start) / max(1, phase_end - phase_start)
        phase_progress = max(0.0, min(1.0, phase_progress))

        weights = {}
        for family in resolved_phase.families:
            w_from = phase.weights_ramp_from.get(family, 0.0)
            w_to = phase.weights_ramp_to.get(family, w_from)
            weights[family] = w_from + phase_progress * (w_to - w_from)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {f: w / total for f, w in weights.items()}

        return weights

    def to_manifest(
        self,
        total_steps: int,
        available_families: List[str],
        family_counts: Dict[str, int],
        seed: int,
    ) -> Dict[str, Any]:
        """Generate a curriculum manifest for reproducibility.

        Args:
            total_steps: Total training steps
            available_families: Available families
            family_counts: Items per family
            seed: RNG seed used

        Returns:
            Dict suitable for JSON serialization
        """
        resolved = self.resolve_phases(total_steps, available_families, family_counts)

        return {
            "curriculum_name": self.name,
            "curriculum_version": self.version,
            "yaml_hash": self.yaml_hash,
            "total_steps": total_steps,
            "seed": seed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_phases": [
                {
                    "name": p.name,
                    "start_step": p.start_step,
                    "end_step": p.end_step,
                    "families": p.families,
                    "weights": p.weights,
                    "sampling_mode": p.sampling_mode,
                }
                for p in resolved
            ],
        }


class CurriculumSampler:
    """Curriculum-aware sampler for PyTorch DataLoader.

    This sampler produces indices that respect curriculum phase rules,
    selecting items based on family weights that vary over training time.

    Unlike a standard PyTorch Sampler, this one is step-aware and must be
    updated with the current step before each batch is sampled.
    """

    def __init__(
        self,
        spec: CurriculumSpec,
        family_index: Dict[str, List[int]],  # family_id -> list of dataset indices
        total_steps: int,
        batch_size: int,
        seed: int,
    ):
        """Initialize the curriculum sampler.

        Args:
            spec: Parsed curriculum specification
            family_index: Mapping from family_id to dataset indices
            total_steps: Total training steps
            batch_size: Batch size per sample() call
            seed: Random seed for reproducibility
        """
        self.spec = spec
        self.family_index = family_index
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)

        # Compute available families and counts
        self.available_families = list(family_index.keys())
        self.family_counts = {f: len(indices) for f, indices in family_index.items()}

        # Resolve phases once
        self.resolved_phases = spec.resolve_phases(
            total_steps, self.available_families, self.family_counts
        )

        # Current step (updated externally)
        self._current_step = 0

        # Validate that all phases have eligible items
        self._validate_phases()

    def _validate_phases(self) -> None:
        """Ensure all phases have at least one eligible family with items."""
        for phase in self.resolved_phases:
            eligible_families = [f for f in phase.families if self.family_counts.get(f, 0) > 0]
            if not eligible_families:
                raise ValueError(
                    f"Phase '{phase.name}' (steps {phase.start_step}-{phase.end_step}) "
                    f"has no eligible families with items. "
                    f"Requested families: {phase.families}"
                )

    def set_step(self, step: int) -> None:
        """Update the current training step.

        Call this before sampling each batch to ensure correct phase selection.
        """
        self._current_step = step

    def _get_current_phase(self) -> Optional[ResolvedPhase]:
        """Get the phase for the current step."""
        for phase in self.resolved_phases:
            if phase.start_step <= self._current_step < phase.end_step:
                return phase
        return None

    def _get_current_phase_idx(self) -> int:
        """Get the index of the current phase (0-indexed, -1 if no phase)."""
        for idx, phase in enumerate(self.resolved_phases):
            if phase.start_step <= self._current_step < phase.end_step:
                return idx
        return -1

    def _get_current_weights(self, phase: ResolvedPhase) -> Dict[str, float]:
        """Get weights for current step, handling ramp interpolation."""
        # Find the original phase definition for ramp handling
        for orig_phase in self.spec.phases:
            if orig_phase.name == phase.name:
                return self.spec.get_weights_at_step(
                    self._current_step, orig_phase, phase, self.total_steps
                )
        return phase.weights

    def sample_batch(self) -> List[int]:
        """Sample a batch of dataset indices according to current phase rules.

        Returns:
            List of dataset indices (length = batch_size)
        """
        phase = self._get_current_phase()
        if phase is None:
            # Fallback: uniform sampling from all families
            all_indices = []
            for indices in self.family_index.values():
                all_indices.extend(indices)
            return [self.rng.choice(all_indices) for _ in range(self.batch_size)]

        weights = self._get_current_weights(phase)

        # Filter to families with positive weight and items
        eligible_families = [
            f for f in phase.families if weights.get(f, 0) > 0 and self.family_counts.get(f, 0) > 0
        ]

        if not eligible_families:
            # Fallback: uniform from phase families
            eligible_families = [f for f in phase.families if self.family_counts.get(f, 0) > 0]

        if not eligible_families:
            raise ValueError(
                f"No eligible families for phase '{phase.name}' at step {self._current_step}"
            )

        batch_indices = []

        if phase.sampling_mode == "balanced_family":
            # Choose family uniformly, then item uniformly within family
            for _ in range(self.batch_size):
                family = self.rng.choice(eligible_families)
                idx = self.rng.choice(self.family_index[family])
                batch_indices.append(idx)

        elif phase.sampling_mode == "proportional_family":
            # Choose family proportional to item count, then item uniformly
            family_weights = [self.family_counts[f] for f in eligible_families]
            for _ in range(self.batch_size):
                family = self.rng.choices(eligible_families, weights=family_weights)[0]
                idx = self.rng.choice(self.family_index[family])
                batch_indices.append(idx)

        elif phase.sampling_mode == "uniform_item":
            # Choose uniformly from all eligible items (ignores family structure)
            all_eligible = []
            for f in eligible_families:
                all_eligible.extend(self.family_index[f])
            batch_indices = [self.rng.choice(all_eligible) for _ in range(self.batch_size)]

        else:
            # Default: use explicit weights for family selection
            family_weights = [weights.get(f, 0) for f in eligible_families]
            for _ in range(self.batch_size):
                family = self.rng.choices(eligible_families, weights=family_weights)[0]
                idx = self.rng.choice(self.family_index[family])
                batch_indices.append(idx)

        return batch_indices

    def __len__(self) -> int:
        """Return total number of samples (for DataLoader compatibility)."""
        return self.total_steps * self.batch_size

    def __iter__(self) -> Iterator[int]:
        """Yield indices for one epoch.

        Note: For curriculum training, prefer using sample_batch() directly
        with step tracking. This iterator is provided for DataLoader compatibility
        but yields indices without phase awareness.
        """
        # For compatibility, yield indices for all steps
        for step in range(self.total_steps):
            self.set_step(step)
            for idx in self.sample_batch():
                yield idx


class SampleTraceWriter:
    """Writes sample trace for deterministic replay and attribution analysis.

    Records which items were sampled at each step, enabling:
    - Exact reproduction of training
    - Attribution analysis (which items preceded events)
    - Counterfactual experiments (replace items within family)

    Schema v2 fields:
    - step: optimizer step number
    - micro: microbatch index within step (0 to accumulation_steps-1)
    - pos: position within microbatch (0 to batch_size-1)
    - phase: curriculum phase name
    - phase_idx: curriculum phase index (stable across name changes)
    - family_id: problem family
    - item_id: unique item identifier
    - split: dataset split (train/val_random/val_family)
    - sampler_mode: sampling mode (balanced_family/proportional_family/uniform_item)
    """

    def __init__(
        self,
        output_path: Path,
        run_id: str,
        seed: int,
        split: str = "train",
        sampler_mode: str = "balanced_family",
    ):
        """Initialize trace writer.

        Args:
            output_path: Path to output file (JSONL format)
            run_id: Run identifier for provenance
            seed: Random seed for provenance
            split: Dataset split name
            sampler_mode: Sampling mode name
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

        # Provenance fields (written to meta file)
        self.run_id = run_id
        self.seed = seed
        self.split = split
        self.sampler_mode = sampler_mode

    def __enter__(self) -> "SampleTraceWriter":
        self._file = self.output_path.open("w")
        return self

    def __exit__(self, *args) -> None:
        if self._file:
            self._file.close()
            self._file = None

        # Write companion meta file with provenance
        meta_path = self.output_path.with_suffix(".meta.json")
        meta = {
            "schema_version": 2,
            "run_id": self.run_id,
            "seed": self.seed,
            "split": self.split,
            "sampler_mode": self.sampler_mode,
            "trace_file": self.output_path.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    def write_batch(
        self,
        step: int,
        micro: int,
        phase_name: str,
        phase_idx: int,
        family_ids: List[str],
        item_ids: List[str],
    ) -> None:
        """Record a sampled microbatch.

        Args:
            step: Optimizer step number
            micro: Microbatch index within step (0 to accumulation_steps-1)
            phase_name: Current curriculum phase name
            phase_idx: Current curriculum phase index
            family_ids: Family IDs for each item in batch
            item_ids: Item IDs for each item in batch
        """
        if not self._file:
            return

        for pos, (family_id, item_id) in enumerate(zip(family_ids, item_ids, strict=True)):
            record = {
                "step": step,
                "micro": micro,
                "pos": pos,
                "phase": phase_name,
                "phase_idx": phase_idx,
                "family_id": family_id,
                "item_id": item_id,
            }
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_curriculum_manifest(
    manifest: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write curriculum manifest to file.

    Args:
        manifest: Manifest dict from CurriculumSpec.to_manifest()
        output_path: Path to write manifest JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
