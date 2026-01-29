"""CLI for the test runner pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import TestOrchestrator, resume_test
from .types import DetectionParams


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed tests with consensus analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5 seeds starting from config's base seed
  squiggle-test --config configs/scout_tiny.yaml --seed-count 5

  # Run specific seeds
  squiggle-test --config configs/scout_tiny.yaml --seeds 42 123 456

  # With analysis overrides
  squiggle-test --config configs/scout_tiny.yaml --seed-count 3 \\
    --warmup-fraction 0.15 --max-pre-warmup 0

  # Resume a failed test
  squiggle-test --continue-from test_scout_tiny_20250129
        """,
    )

    # Main mode: config-based or resume
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=Path,
        help="Path to training config (Scout or Research YAML)",
    )
    group.add_argument(
        "--continue-from",
        type=str,
        metavar="TEST_ID",
        help="Resume a previously started test",
    )

    # Seed specification (one required when using --config)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-count",
        type=int,
        metavar="N",
        help="Run N seeds starting from base seed in config",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="Explicit list of seeds to run",
    )

    # Test identification
    parser.add_argument(
        "--test-id",
        type=str,
        help="Override test_id (auto-generated if not set)",
    )

    # Retry behavior
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry attempts per failed seed (default: 2)",
    )

    # Training override
    parser.add_argument(
        "--step-multiplier",
        type=float,
        help="Override step_multiplier for extension ladder (e.g., 2.0 for 2x training)",
    )

    # Analysis options
    parser.add_argument(
        "--analysis-id",
        type=str,
        help="Override analysis_id (auto-generated from detection params if not set)",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis phase (just run training)",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip consensus/comparison phase",
    )

    # Detection parameter overrides
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        help="Fraction of training to treat as warmup (default: 0.1)",
    )
    parser.add_argument(
        "--max-pre-warmup",
        type=int,
        help="Max pre-warmup peaks per series (default: 1)",
    )
    parser.add_argument(
        "--suppression-radius",
        type=int,
        help="Step distance for peak suppression (default: 15)",
    )
    parser.add_argument(
        "--max-events-per-series",
        type=int,
        help="Maximum events per (layer, metric) series (default: 5)",
    )
    parser.add_argument(
        "--adaptive-k",
        type=float,
        help="Adaptive threshold multiplier (default: 2.5)",
    )
    parser.add_argument(
        "--step-tolerance",
        type=int,
        help="Step tolerance for matching events across seeds (default: 5)",
    )

    # LLM analysis options
    parser.add_argument(
        "--llm-analysis",
        action="store_true",
        help="Enable LLM analysis on consensus events",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o",
        help="Model to use for LLM analysis (default: gpt-4o)",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing",
    )

    args = parser.parse_args()

    # Handle resume mode
    if args.continue_from:
        if args.dry_run:
            print(f"[dry-run] Would resume test: {args.continue_from}")
            return

        try:
            test_config = resume_test(args.continue_from)
            sys.exit(0 if test_config.summary and test_config.summary.failed == 0 else 1)
        except Exception as e:
            print(f"Error resuming test: {e}")
            sys.exit(1)

    # Config-based mode: validate seed specification
    if not args.seed_count and not args.seeds:
        parser.error("--config requires either --seed-count or --seeds")

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        # Generate seeds starting from 42 (or could read from config)
        base_seed = 42
        seeds = list(range(base_seed, base_seed + args.seed_count))

    # Build detection params
    detection_params = DetectionParams(
        warmup_fraction=args.warmup_fraction if args.warmup_fraction is not None else 0.1,
        max_pre_warmup=args.max_pre_warmup if args.max_pre_warmup is not None else 1,
        peak_suppression_radius=args.suppression_radius if args.suppression_radius is not None else 15,
        max_events_per_series=args.max_events_per_series if args.max_events_per_series is not None else 5,
        adaptive_k=args.adaptive_k if args.adaptive_k is not None else 2.5,
        step_tolerance=args.step_tolerance if args.step_tolerance is not None else 5,
    )

    # Dry run output
    if args.dry_run:
        print("[dry-run] Test configuration:")
        print(f"  Config: {args.config}")
        print(f"  Seeds: {seeds}")
        print(f"  Test ID: {args.test_id or '(auto-generated)'}")
        if args.step_multiplier:
            print(f"  Step multiplier: {args.step_multiplier}")
        print("  Detection params:")
        print(f"    warmup_fraction: {detection_params.warmup_fraction}")
        print(f"    max_pre_warmup: {detection_params.max_pre_warmup}")
        print(f"    peak_suppression_radius: {detection_params.peak_suppression_radius}")
        print(f"    max_events_per_series: {detection_params.max_events_per_series}")
        print(f"    adaptive_k: {detection_params.adaptive_k}")
        print(f"    step_tolerance: {detection_params.step_tolerance}")
        print(f"  Run analysis: {not args.no_analysis}")
        print(f"  Run comparison: {not args.no_comparison}")
        print(f"  LLM analysis: {args.llm_analysis}")
        print(f"  Max retries: {args.max_retries}")
        return

    # Create and run orchestrator
    orchestrator = TestOrchestrator(
        config_path=args.config,
        seeds=seeds,
        test_id=args.test_id,
        detection_params=detection_params,
        max_retries=args.max_retries,
        run_analysis=not args.no_analysis,
        run_comparison=not args.no_comparison,
        llm_analysis=args.llm_analysis,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        step_multiplier=args.step_multiplier,
    )

    try:
        test_config = orchestrator.run()
        # Exit with error code if any runs failed
        if test_config.summary and test_config.summary.failed > 0:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[test_runner] Interrupted - partial results saved to manifest")
        sys.exit(130)
    except Exception as e:
        print(f"\n[test_runner] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
