"""CLI for the experiment runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from squiggle_core.schemas.experiment import ExperimentSpec

from ..test_runner import DetectionParams
from .orchestrator import ExperimentOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-arm experiments with cross-arm comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml

  # Run specific arms only
  squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml \\
    --arms iid blocked

  # Override seeds for quick testing
  squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml \\
    --seeds 42 123

  # Dry run to see plan
  squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml --dry-run
        """,
    )

    # Required: spec file
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to experiment.yaml spec file",
    )

    # Optional: filter arms
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        metavar="ARM",
        help="Only run these arms (default: all)",
    )

    # Optional: override seeds
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="Override seeds from spec",
    )

    # Training override
    parser.add_argument(
        "--step-multiplier",
        type=float,
        help="Override step_multiplier for all arms",
    )

    # Retry behavior
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry attempts per failed seed (default: 2)",
    )

    # Analysis options
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis phase (just run training)",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip cross-arm comparison phase",
    )

    # Detection parameter overrides
    parser.add_argument(
        "--step-tolerance",
        type=int,
        help="Step tolerance for matching events across seeds (default: from spec)",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing",
    )

    # Validate spec only
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the spec file, don't run",
    )

    args = parser.parse_args()

    # Check spec exists
    if not args.spec.exists():
        print(f"Spec file not found: {args.spec}")
        sys.exit(1)

    # Load and validate spec
    try:
        spec = ExperimentSpec.from_yaml(args.spec)
    except Exception as e:
        print(f"Error loading spec: {e}")
        sys.exit(1)

    errors = spec.validate()
    if errors:
        print("Spec validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Validate-only mode
    if args.validate:
        print(f"Spec is valid: {args.spec}")
        print(f"  Exp ID: {spec.exp_id}")
        print(f"  Arms: {list(spec.arms.keys())}")
        print(f"  Seeds: {spec.seeds}")
        sys.exit(0)

    # Validate arm names if specified
    if args.arms:
        for arm_name in args.arms:
            if arm_name not in spec.arms:
                print(f"Unknown arm: {arm_name}")
                print(f"Available arms: {list(spec.arms.keys())}")
                sys.exit(1)

    # Build detection params
    detection_params = None
    if args.step_tolerance is not None:
        detection_params = DetectionParams(
            step_tolerance=args.step_tolerance,
        )

    # Dry run output
    if args.dry_run:
        arms = args.arms if args.arms else list(spec.arms.keys())
        seeds = args.seeds if args.seeds else spec.seeds

        print("[dry-run] Experiment configuration:")
        print(f"  Spec: {args.spec}")
        print(f"  Exp ID: {spec.exp_id}")
        print(f"  Hypothesis: {spec.hypothesis}")
        print(f"  Isolates: {spec.isolates}")
        print(f"  Invariants:")
        for inv in spec.invariants:
            print(f"    - {inv}")
        print(f"  Arms to run: {arms}")
        print(f"  Seeds: {seeds}")
        print(f"  Primary outcomes: {spec.primary_outcomes}")
        print(f"  Event consensus rules:")
        print(f"    step_tolerance: {spec.event_consensus_rules.step_tolerance}")
        print(f"    min_seed_fraction: {spec.event_consensus_rules.min_seed_fraction}")
        if args.step_multiplier:
            print(f"  Step multiplier override: {args.step_multiplier}")
        print(f"  Run analysis: {not args.no_analysis}")
        print(f"  Run comparison: {not args.no_comparison}")
        print(f"  Max retries: {args.max_retries}")

        print("\nArm configs:")
        for arm_name in arms:
            arm = spec.arms[arm_name]
            print(f"  {arm_name}:")
            print(f"    test_config: {arm.test_config}")
            if arm.curriculum:
                print(f"    curriculum: {arm.curriculum}")
            if arm.description:
                print(f"    description: {arm.description}")

        return

    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(
        spec_path=args.spec,
        seeds=args.seeds,
        arms=args.arms,
        detection_params=detection_params,
        max_retries=args.max_retries,
        run_analysis=not args.no_analysis,
        run_comparison=not args.no_comparison,
        step_multiplier=args.step_multiplier,
    )

    try:
        result = orchestrator.run()
        # Exit with error code if any arms failed
        if not result.succeeded:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[experiment] Interrupted - partial results may be saved")
        sys.exit(130)
    except Exception as e:
        print(f"\n[experiment] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
