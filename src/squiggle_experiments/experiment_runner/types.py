"""Type definitions for experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..test_runner.types import TestConfig


@dataclass
class ArmResult:
    """Result from running a single arm of an experiment."""

    arm_name: str
    """Name of this arm (e.g., 'iid', 'blocked')."""

    test_config: TestConfig | None = None
    """The TestConfig produced by the test runner for this arm."""

    consensus_event_count: int = 0
    """Number of consensus events detected across seeds."""

    test_id: str | None = None
    """ID of the test that was run for this arm."""

    error: str | None = None
    """Error message if this arm failed."""


@dataclass
class ExperimentResult:
    """Result from running a complete experiment."""

    exp_id: str
    """Experiment identifier."""

    spec_hash: str
    """Hash of the experiment spec for versioning."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the experiment started."""

    completed_at: datetime | None = None
    """When the experiment completed (None if still running)."""

    arms: dict[str, ArmResult] = field(default_factory=dict)
    """Results for each arm, keyed by arm name."""

    comparison_report_path: Path | None = None
    """Path to the generated comparison report."""

    error: str | None = None
    """Overall experiment error, if any."""

    @property
    def succeeded(self) -> bool:
        """True if all arms completed successfully."""
        return all(arm.error is None for arm in self.arms.values())

    @property
    def arm_names(self) -> list[str]:
        """Names of all arms."""
        return list(self.arms.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exp_id": self.exp_id,
            "spec_hash": self.spec_hash,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "arms": {
                name: {
                    "arm_name": arm.arm_name,
                    "test_id": arm.test_id,
                    "consensus_event_count": arm.consensus_event_count,
                    "error": arm.error,
                }
                for name, arm in self.arms.items()
            },
            "comparison_report_path": str(self.comparison_report_path)
            if self.comparison_report_path
            else None,
            "error": self.error,
            "succeeded": self.succeeded,
        }
