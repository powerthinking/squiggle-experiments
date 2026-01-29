"""Type definitions for the test runner pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class RunStatus(Enum):
    """Status of a single training run within a test."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ANALYSIS_FAILED = "analysis_failed"  # Training succeeded but analysis failed


class ConfidenceLevel(Enum):
    """Confidence level for test results."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class DetectionParams:
    """Event detection parameters."""

    warmup_fraction: float = 0.1
    max_pre_warmup: int = 1
    peak_suppression_radius: int = 15
    max_events_per_series: int = 5
    adaptive_k: float = 2.5
    step_tolerance: int = 5

    def to_dict(self) -> dict:
        return {
            "warmup_fraction": self.warmup_fraction,
            "max_pre_warmup": self.max_pre_warmup,
            "peak_suppression_radius": self.peak_suppression_radius,
            "max_events_per_series": self.max_events_per_series,
            "adaptive_k": self.adaptive_k,
            "step_tolerance": self.step_tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DetectionParams:
        return cls(
            warmup_fraction=d.get("warmup_fraction", 0.1),
            max_pre_warmup=d.get("max_pre_warmup", 1),
            peak_suppression_radius=d.get("peak_suppression_radius", 15),
            max_events_per_series=d.get("max_events_per_series", 5),
            adaptive_k=d.get("adaptive_k", 2.5),
            step_tolerance=d.get("step_tolerance", 5),
        )


@dataclass
class RunResult:
    """Result of a single training run within a test."""

    seed: int
    run_id: Optional[str] = None
    status: RunStatus = RunStatus.PENDING
    analysis_id: Optional[str] = None
    error_message: Optional[str] = None
    final_loss: Optional[float] = None
    event_count: Optional[int] = None
    retries: int = 0

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "run_id": self.run_id,
            "status": self.status.value,
            "analysis_id": self.analysis_id,
            "error_message": self.error_message,
            "final_loss": self.final_loss,
            "event_count": self.event_count,
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(
            seed=d["seed"],
            run_id=d.get("run_id"),
            status=RunStatus(d.get("status", "pending")),
            analysis_id=d.get("analysis_id"),
            error_message=d.get("error_message"),
            final_loss=d.get("final_loss"),
            event_count=d.get("event_count"),
            retries=d.get("retries", 0),
        )


@dataclass
class TestSummary:
    """Summary statistics for a completed test."""

    total_seeds: int
    successful: int
    failed: int
    consensus_events: int = 0
    jaccard_similarity: float = 0.0
    mean_correlation: float = 0.0
    confidence: ConfidenceLevel = ConfidenceLevel.LOW

    def to_dict(self) -> dict:
        return {
            "total_seeds": self.total_seeds,
            "successful": self.successful,
            "failed": self.failed,
            "consensus_events": self.consensus_events,
            "jaccard_similarity": self.jaccard_similarity,
            "mean_correlation": self.mean_correlation,
            "confidence": self.confidence.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestSummary:
        return cls(
            total_seeds=d["total_seeds"],
            successful=d["successful"],
            failed=d["failed"],
            consensus_events=d.get("consensus_events", 0),
            jaccard_similarity=d.get("jaccard_similarity", 0.0),
            mean_correlation=d.get("mean_correlation", 0.0),
            confidence=ConfidenceLevel(d.get("confidence", "low")),
        )


@dataclass
class TestConfig:
    """Configuration for a test (multi-seed run)."""

    test_id: str
    config_path: Path
    config_hash: str
    seeds: list[int]
    created_at: datetime
    detection_params: DetectionParams
    runs: list[RunResult] = field(default_factory=list)
    summary: Optional[TestSummary] = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "config_path": str(self.config_path),
            "config_hash": self.config_hash,
            "seeds": self.seeds,
            "created_at": self.created_at.isoformat(),
            "detection_params": self.detection_params.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "summary": self.summary.to_dict() if self.summary else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestConfig:
        return cls(
            test_id=d["test_id"],
            config_path=Path(d["config_path"]),
            config_hash=d["config_hash"],
            seeds=d["seeds"],
            created_at=datetime.fromisoformat(d["created_at"]),
            detection_params=DetectionParams.from_dict(d["detection_params"]),
            runs=[RunResult.from_dict(r) for r in d.get("runs", [])],
            summary=TestSummary.from_dict(d["summary"]) if d.get("summary") else None,
        )

    def successful_runs(self) -> list[RunResult]:
        """Return runs with status SUCCESS."""
        return [r for r in self.runs if r.status == RunStatus.SUCCESS]

    def failed_runs(self) -> list[RunResult]:
        """Return runs with status FAILED or ANALYSIS_FAILED."""
        return [
            r for r in self.runs if r.status in (RunStatus.FAILED, RunStatus.ANALYSIS_FAILED)
        ]


@dataclass
class ConsensusEvent:
    """A seed-invariant event appearing across multiple runs."""

    layer: int
    metric: str
    mean_step: float
    step_spread: float
    seed_count: int
    seed_fraction: float
    mean_score: float
    direction: str  # "increase" or "decrease"
    constituent_run_ids: list[str]
    constituent_event_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "metric": self.metric,
            "mean_step": self.mean_step,
            "step_spread": self.step_spread,
            "seed_count": self.seed_count,
            "seed_fraction": self.seed_fraction,
            "mean_score": self.mean_score,
            "direction": self.direction,
            "constituent_run_ids": self.constituent_run_ids,
            "constituent_event_ids": self.constituent_event_ids,
        }
