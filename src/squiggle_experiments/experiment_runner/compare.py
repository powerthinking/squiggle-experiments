"""Cross-arm comparison for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from squiggle_core import paths
from squiggle_core.schemas.experiment import ExperimentSpec

if TYPE_CHECKING:
    from .types import ArmResult


@dataclass
class ArmMetrics:
    """Computed metrics for a single arm."""

    arm_name: str
    consensus_event_count: int = 0
    event_yield_per_step: float = 0.0
    total_steps: int = 0
    seeds_succeeded: int = 0
    seeds_failed: int = 0
    mean_final_loss: float | None = None
    event_types: dict[str, int] = field(default_factory=dict)


@dataclass
class ComparisonResults:
    """Results from comparing multiple arms."""

    arm_metrics: dict[str, ArmMetrics] = field(default_factory=dict)
    """Metrics for each arm."""

    outcome_rankings: dict[str, list[str]] = field(default_factory=dict)
    """For each outcome, arms ranked best to worst."""

    event_overlap: dict[tuple[str, str], float] = field(default_factory=dict)
    """Jaccard similarity of event types between arm pairs."""

    winner_by_outcome: dict[str, str] = field(default_factory=dict)
    """Best arm for each primary outcome."""


def compare_arms(
    arm_results: dict[str, "ArmResult"],
    spec: ExperimentSpec,
) -> ComparisonResults:
    """Compare results across experiment arms.

    Args:
        arm_results: Mapping of arm name to ArmResult (only successful arms)
        spec: Experiment specification

    Returns:
        ComparisonResults with metrics and rankings
    """
    results = ComparisonResults()

    # Compute metrics for each arm
    for arm_name, arm_result in arm_results.items():
        metrics = _compute_arm_metrics(arm_name, arm_result)
        results.arm_metrics[arm_name] = metrics

    # Rank arms by each primary outcome
    for outcome in spec.primary_outcomes:
        ranking = _rank_by_outcome(results.arm_metrics, outcome)
        results.outcome_rankings[outcome] = ranking
        if ranking:
            results.winner_by_outcome[outcome] = ranking[0]

    # Compute event overlap between arm pairs
    arm_names = list(results.arm_metrics.keys())
    for i, arm1 in enumerate(arm_names):
        for arm2 in arm_names[i + 1 :]:
            overlap = _compute_event_overlap(
                results.arm_metrics[arm1].event_types,
                results.arm_metrics[arm2].event_types,
            )
            results.event_overlap[(arm1, arm2)] = overlap

    return results


def _compute_arm_metrics(arm_name: str, arm_result: "ArmResult") -> ArmMetrics:
    """Compute metrics for a single arm from its results."""
    metrics = ArmMetrics(arm_name=arm_name)

    if arm_result.test_config is None:
        return metrics

    test_config = arm_result.test_config

    # Count successes/failures
    for run in test_config.runs:
        if run.run_id is not None:
            metrics.seeds_succeeded += 1
        else:
            metrics.seeds_failed += 1

    # Get consensus event count
    metrics.consensus_event_count = arm_result.consensus_event_count

    # Get total steps (from first successful run)
    for run in test_config.runs:
        if run.run_id is not None:
            try:
                meta_path = paths.run_meta_path(run.run_id)
                if meta_path.exists():
                    import json

                    with open(meta_path) as f:
                        meta = json.load(f)
                    if "training" in meta and "total_steps" in meta["training"]:
                        metrics.total_steps = meta["training"]["total_steps"]
                        break
            except Exception:
                pass

    # Compute event yield
    if metrics.total_steps > 0:
        metrics.event_yield_per_step = metrics.consensus_event_count / metrics.total_steps

    # Get mean final loss
    losses = []
    for run in test_config.runs:
        if run.run_id is not None and run.final_loss is not None:
            losses.append(run.final_loss)
    if losses:
        metrics.mean_final_loss = sum(losses) / len(losses)

    # Get event types from consensus events
    if test_config.consensus_events is not None:
        for event in test_config.consensus_events:
            event_type = f"{event.metric}@L{event.layer}"
            metrics.event_types[event_type] = metrics.event_types.get(event_type, 0) + 1

    return metrics


def _rank_by_outcome(
    arm_metrics: dict[str, ArmMetrics],
    outcome: str,
) -> list[str]:
    """Rank arms by a specific outcome (higher is better)."""
    if outcome == "consensus_event_count":
        return sorted(
            arm_metrics.keys(),
            key=lambda a: arm_metrics[a].consensus_event_count,
            reverse=True,
        )
    elif outcome == "event_yield_per_step":
        return sorted(
            arm_metrics.keys(),
            key=lambda a: arm_metrics[a].event_yield_per_step,
            reverse=True,
        )
    elif outcome == "mean_final_loss":
        # Lower is better for loss
        valid = [a for a in arm_metrics if arm_metrics[a].mean_final_loss is not None]
        return sorted(
            valid,
            key=lambda a: arm_metrics[a].mean_final_loss or float("inf"),
        )
    else:
        # Unknown outcome - return empty ranking
        return []


def _compute_event_overlap(
    events1: dict[str, int],
    events2: dict[str, int],
) -> float:
    """Compute Jaccard similarity of event type sets."""
    set1 = set(events1.keys())
    set2 = set(events2.keys())

    if not set1 and not set2:
        return 1.0  # Both empty = identical

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def generate_comparison_report(
    comparison_results: ComparisonResults,
    spec: ExperimentSpec,
    output_path: Path,
) -> None:
    """Generate markdown comparison report.

    Args:
        comparison_results: Results from compare_arms()
        spec: Experiment specification
        output_path: Where to write the report
    """
    lines = []

    # Header
    lines.append(f"# Experiment Comparison: {spec.exp_id}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Hypothesis
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(f"> {spec.hypothesis}")
    lines.append("")

    # Design
    lines.append("## Design")
    lines.append("")
    lines.append(f"**Isolates**: {spec.isolates}")
    lines.append("")
    lines.append("**Invariants**:")
    for inv in spec.invariants:
        lines.append(f"- {inv}")
    lines.append("")

    # Arms summary table
    lines.append("## Arm Summary")
    lines.append("")
    lines.append("| Arm | Seeds OK | Consensus Events | Event Yield | Mean Loss |")
    lines.append("|-----|----------|------------------|-------------|-----------|")

    for arm_name, metrics in comparison_results.arm_metrics.items():
        seeds_ok = f"{metrics.seeds_succeeded}/{metrics.seeds_succeeded + metrics.seeds_failed}"
        events = str(metrics.consensus_event_count)
        yield_val = f"{metrics.event_yield_per_step:.4f}"
        loss = f"{metrics.mean_final_loss:.4f}" if metrics.mean_final_loss else "N/A"
        lines.append(f"| {arm_name} | {seeds_ok} | {events} | {yield_val} | {loss} |")
    lines.append("")

    # Rankings by outcome
    lines.append("## Rankings by Outcome")
    lines.append("")

    for outcome, ranking in comparison_results.outcome_rankings.items():
        lines.append(f"### {outcome}")
        lines.append("")
        for i, arm_name in enumerate(ranking, 1):
            winner = " (winner)" if i == 1 else ""
            lines.append(f"{i}. **{arm_name}**{winner}")
        lines.append("")

    # Event overlap matrix
    if comparison_results.event_overlap:
        lines.append("## Event Type Overlap (Jaccard)")
        lines.append("")

        arm_names = list(comparison_results.arm_metrics.keys())
        lines.append("| " + " | ".join([""] + arm_names) + " |")
        lines.append("|" + "|".join(["---"] * (len(arm_names) + 1)) + "|")

        for arm1 in arm_names:
            row = [arm1]
            for arm2 in arm_names:
                if arm1 == arm2:
                    row.append("1.00")
                else:
                    key = (arm1, arm2) if (arm1, arm2) in comparison_results.event_overlap else (arm2, arm1)
                    overlap = comparison_results.event_overlap.get(key, 0.0)
                    row.append(f"{overlap:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Conclusion placeholder
    lines.append("## Conclusion")
    lines.append("")

    if comparison_results.winner_by_outcome:
        winners = set(comparison_results.winner_by_outcome.values())
        if len(winners) == 1:
            winner = list(winners)[0]
            lines.append(f"**{winner}** wins on all primary outcomes.")
        else:
            lines.append("Mixed results - winners vary by outcome:")
            for outcome, winner in comparison_results.winner_by_outcome.items():
                lines.append(f"- {outcome}: **{winner}**")
    else:
        lines.append("No primary outcome rankings available.")
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
