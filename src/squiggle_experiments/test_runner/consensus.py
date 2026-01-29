"""Consensus event extraction and report generation for multi-seed tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .types import TestConfig


def _match_events_across_seeds(
    run_events: dict[str, pd.DataFrame],
    step_tolerance: int = 5,
) -> list[dict]:
    """Match events across runs within step tolerance.

    For each unique (layer, metric, direction) combination, finds events
    that occur at similar steps across runs.

    Returns list of matched event groups, where each group contains events
    from multiple runs that are considered the "same" event.
    """
    all_events = []
    for run_id, df in run_events.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            # Determine direction from polarity or delta
            direction = "increase"
            if "polarity" in row and pd.notna(row["polarity"]):
                direction = "increase" if row["polarity"] > 0 else "decrease"
            elif "delta" in row and pd.notna(row["delta"]):
                direction = "increase" if row["delta"] > 0 else "decrease"

            all_events.append({
                "run_id": run_id,
                "event_id": row.get("event_id", ""),
                "layer": int(row["layer"]),
                "metric": str(row["metric"]),
                "step": int(row["step"]),
                "score": float(row.get("score", 0)),
                "direction": direction,
            })

    if not all_events:
        return []

    # Group by (layer, metric, direction)
    groups: dict[tuple, list[dict]] = {}
    for ev in all_events:
        key = (ev["layer"], ev["metric"], ev["direction"])
        if key not in groups:
            groups[key] = []
        groups[key].append(ev)

    matched_groups = []

    # For each (layer, metric, direction) group, cluster events by step
    for (layer, metric, direction), events in groups.items():
        # Sort by step
        events = sorted(events, key=lambda x: x["step"])

        # Greedy clustering: events within step_tolerance of cluster center
        clusters = []
        for ev in events:
            added = False
            for cluster in clusters:
                # Check if event is within tolerance of cluster mean step
                mean_step = np.mean([e["step"] for e in cluster])
                if abs(ev["step"] - mean_step) <= step_tolerance:
                    cluster.append(ev)
                    added = True
                    break

            if not added:
                clusters.append([ev])

        # Only keep clusters with events from multiple runs
        for cluster in clusters:
            run_ids = set(e["run_id"] for e in cluster)
            if len(run_ids) >= 2:
                matched_groups.append({
                    "layer": layer,
                    "metric": metric,
                    "direction": direction,
                    "events": cluster,
                    "run_ids": list(run_ids),
                })

    return matched_groups


def extract_consensus_events(
    run_events: dict[str, pd.DataFrame],
    step_tolerance: int = 5,
    min_seed_fraction: float = 1.0,
    test_id: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Extract consensus (seed-invariant) events from multiple runs.

    Args:
        run_events: Dict mapping run_id to events DataFrame
        step_tolerance: Maximum step difference for matching events
        min_seed_fraction: Minimum fraction of seeds required (1.0 = all seeds)
        test_id: Test identifier for the output

    Returns:
        Tuple of (consensus_events_df, metrics_dict)
    """
    n_runs = len(run_events)
    min_run_count = max(2, int(n_runs * min_seed_fraction))

    matched_groups = _match_events_across_seeds(run_events, step_tolerance)

    # Filter to groups appearing in enough runs
    consensus_groups = [
        g for g in matched_groups if len(g["run_ids"]) >= min_run_count
    ]

    # Build consensus events DataFrame
    rows = []
    for group in consensus_groups:
        events = group["events"]
        steps = [e["step"] for e in events]
        scores = [e["score"] for e in events]

        rows.append({
            "test_id": test_id,
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc),
            "consensus_event_id": str(uuid.uuid4())[:8],
            "layer": group["layer"],
            "metric": group["metric"],
            "mean_step": np.mean(steps),
            "step_spread": np.std(steps) if len(steps) > 1 else 0.0,
            "seed_count": len(group["run_ids"]),
            "seed_fraction": len(group["run_ids"]) / n_runs,
            "mean_score": np.mean(scores),
            "direction": group["direction"],
            "constituent_run_ids": group["run_ids"],
            "constituent_event_ids": [e["event_id"] for e in events],
        })

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["layer", "metric", "mean_step"]).reset_index(drop=True)
    else:
        # Empty DataFrame with correct schema
        df = pd.DataFrame(columns=[
            "test_id", "schema_version", "created_at_utc", "consensus_event_id",
            "layer", "metric", "mean_step", "step_spread", "seed_count",
            "seed_fraction", "mean_score", "direction",
            "constituent_run_ids", "constituent_event_ids",
        ])

    # Compute metrics
    metrics = _compute_consensus_metrics(run_events, consensus_groups, n_runs)

    return df, metrics


def _compute_consensus_metrics(
    run_events: dict[str, pd.DataFrame],
    consensus_groups: list[dict],
    n_runs: int,
) -> dict:
    """Compute consensus metrics (Jaccard, correlation, etc.)."""
    # Get all unique events per run (as (layer, metric, step_bucket) tuples)
    def bucket_step(step: int, bucket_size: int = 10) -> int:
        return (step // bucket_size) * bucket_size

    run_event_sets = {}
    for run_id, df in run_events.items():
        if df.empty:
            run_event_sets[run_id] = set()
            continue

        events = set()
        for _, row in df.iterrows():
            events.add((
                int(row["layer"]),
                str(row["metric"]),
                bucket_step(int(row["step"])),
            ))
        run_event_sets[run_id] = events

    # Compute pairwise Jaccard similarities
    run_ids = list(run_event_sets.keys())
    jaccard_scores = []
    for i in range(len(run_ids)):
        for j in range(i + 1, len(run_ids)):
            set_a = run_event_sets[run_ids[i]]
            set_b = run_event_sets[run_ids[j]]
            if set_a or set_b:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_scores.append(jaccard)

    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0

    # Count total unique events across all runs
    all_events = set()
    for events in run_event_sets.values():
        all_events.update(events)

    # Consensus events (appearing in all runs)
    if run_event_sets:
        consensus_events = set.intersection(*run_event_sets.values()) if run_event_sets else set()
    else:
        consensus_events = set()

    return {
        "jaccard_similarity": float(mean_jaccard),
        "mean_correlation": 0.0,  # Placeholder - would need geometry data
        "total_unique_events": len(all_events),
        "consensus_event_count": len(consensus_events),
        "n_runs": n_runs,
    }


def _compute_partial_events(
    run_events: dict[str, pd.DataFrame],
    consensus_groups: list[dict],
    step_tolerance: int = 5,
) -> list[dict]:
    """Find events that appear in some but not all runs."""
    n_runs = len(run_events)
    matched_groups = _match_events_across_seeds(run_events, step_tolerance)

    # Events in some runs but not all
    partial = []
    for group in matched_groups:
        fraction = len(group["run_ids"]) / n_runs
        if 0 < fraction < 1.0:
            events = group["events"]
            partial.append({
                "layer": group["layer"],
                "metric": group["metric"],
                "step": int(np.mean([e["step"] for e in events])),
                "seeds_present": len(group["run_ids"]),
                "total_seeds": n_runs,
                "run_ids": group["run_ids"],
            })

    return sorted(partial, key=lambda x: (-x["seeds_present"], x["layer"], x["metric"]))


def generate_consensus_report(
    test_config: "TestConfig",
    consensus_df: pd.DataFrame,
    metrics: dict,
    run_events: dict[str, pd.DataFrame],
) -> str:
    """Generate markdown consensus report.

    Args:
        test_config: Test configuration
        consensus_df: Consensus events DataFrame
        metrics: Consensus metrics dict
        run_events: Dict mapping run_id to events DataFrame

    Returns:
        Markdown report content
    """
    lines = []

    # Header
    lines.append(f"# Test Report: {test_config.test_id}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    n_consensus = len(consensus_df)
    n_successful = len(test_config.successful_runs())
    confidence = test_config.summary.confidence.value if test_config.summary else "unknown"

    if n_consensus > 0:
        lines.append(
            f"Found **{n_consensus} seed-invariant events** across {n_successful} successful runs. "
            f"Jaccard similarity: {metrics.get('jaccard_similarity', 0):.1%}."
        )
    else:
        lines.append(
            f"No seed-invariant events found across {n_successful} runs. "
            f"Events may be seed-dependent or detection parameters need adjustment."
        )

    lines.append(f"**Confidence Level:** {confidence.upper()}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append(f"- Config: `{test_config.config_path}`")
    lines.append(f"- Config hash: `{test_config.config_hash}`")
    lines.append(f"- Seeds: {test_config.seeds}")

    dp = test_config.detection_params
    lines.append(
        f"- Detection: warmup={dp.warmup_fraction}, "
        f"suppression={dp.peak_suppression_radius}, "
        f"k={dp.adaptive_k}"
    )
    lines.append("")

    # Seed-Invariant Events
    lines.append("## Seed-Invariant Events (High Confidence)")
    lines.append(f"_These events appear consistently across ALL {n_successful} seeds._")
    lines.append("")

    if not consensus_df.empty:
        # Build table
        table_rows = []
        for _, row in consensus_df.iterrows():
            spread_str = f"±{row['step_spread']:.0f}" if row['step_spread'] > 0 else "—"
            table_rows.append({
                "Layer": int(row["layer"]),
                "Metric": row["metric"],
                "Mean Step": f"{row['mean_step']:.0f}",
                "Step Spread": spread_str,
                "Mean Score": f"{row['mean_score']:.3f}",
                "Direction": row["direction"],
            })

        if table_rows:
            table_df = pd.DataFrame(table_rows)
            lines.append(table_df.to_markdown(index=False))
    else:
        lines.append("_No seed-invariant events detected._")
    lines.append("")

    # Trajectory Consistency
    lines.append("## Trajectory Consistency")
    lines.append(f"- Mean pairwise Jaccard similarity: **{metrics.get('jaccard_similarity', 0):.1%}**")
    lines.append(f"- Total unique events across seeds: {metrics.get('total_unique_events', 0)}")
    lines.append("")

    # Seed-Sensitive Observations
    lines.append("## Seed-Sensitive Observations")
    lines.append("_These events appeared in some but not all seeds. Treat with caution._")
    lines.append("")

    partial = _compute_partial_events(run_events, [], test_config.detection_params.step_tolerance)
    if partial:
        partial_rows = []
        for p in partial[:20]:  # Limit to top 20
            partial_rows.append({
                "Layer": p["layer"],
                "Metric": p["metric"],
                "Step": p["step"],
                "Seeds Present": f"{p['seeds_present']}/{p['total_seeds']}",
            })

        if partial_rows:
            partial_df = pd.DataFrame(partial_rows)
            lines.append(partial_df.to_markdown(index=False))

        if len(partial) > 20:
            lines.append(f"\n_...and {len(partial) - 20} more partial events._")
    else:
        lines.append("_No partial events detected._")
    lines.append("")

    # Per-Run Summary
    lines.append("## Per-Run Summary")
    run_rows = []
    for run in test_config.runs:
        status_emoji = "✓" if run.status.value == "success" else "✗"
        run_rows.append({
            "Seed": run.seed,
            "Run ID": run.run_id or "—",
            "Status": f"{status_emoji} {run.status.value}",
            "Events": run.event_count if run.event_count is not None else "—",
            "Final Loss": f"{run.final_loss:.4f}" if run.final_loss is not None else "—",
        })

    run_df = pd.DataFrame(run_rows)
    lines.append(run_df.to_markdown(index=False))
    lines.append("")

    # Artifacts
    lines.append("## Artifacts")
    from squiggle_core import paths as p
    lines.append(f"- Consensus events: `{p.events_consensus_path(test_config.test_id)}`")
    lines.append(f"- Test manifest: `{p.test_manifest_path(test_config.test_id)}`")

    for run in test_config.successful_runs():
        lines.append(
            f"- {run.run_id} report: `{p.report_md_path(run.run_id, run.analysis_id)}`"
        )

    lines.append("")

    return "\n".join(lines)
