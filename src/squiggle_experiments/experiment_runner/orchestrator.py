"""Experiment orchestrator for running multi-arm experiments."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from squiggle_core import paths
from squiggle_core.schemas.experiment import ExperimentSpec

from ..test_runner import DetectionParams, TestOrchestrator
from .compare import compare_arms, generate_comparison_report
from .types import ArmResult, ExperimentResult

logger = logging.getLogger(__name__)


def _experiments_root() -> Path:
    """Get the experiments root directory."""
    return paths.data_root() / "experiments"


def _experiment_dir(exp_id: str) -> Path:
    """Get directory for a specific experiment."""
    return _experiments_root() / exp_id


class ExperimentOrchestrator:
    """Orchestrates multi-arm experiments.

    An Experiment runs multiple Tests (one per arm), each Test being a multi-seed
    training run with identical config except seed.

    Usage:
        orchestrator = ExperimentOrchestrator(spec_path)
        result = orchestrator.run()
    """

    def __init__(
        self,
        spec_path: Path | str,
        seeds: list[int] | None = None,
        arms: list[str] | None = None,
        detection_params: DetectionParams | None = None,
        max_retries: int = 2,
        run_analysis: bool = True,
        run_comparison: bool = True,
        step_multiplier: float | None = None,
    ):
        """Initialize experiment orchestrator.

        Args:
            spec_path: Path to experiment.yaml spec file
            seeds: Override seeds from spec (for quick testing)
            arms: Only run these arms (for quick testing)
            detection_params: Override detection params
            max_retries: Retry attempts per failed seed
            run_analysis: Whether to run analysis after training
            run_comparison: Whether to run cross-arm comparison
            step_multiplier: Override step_multiplier for all arms
        """
        self.spec_path = Path(spec_path).resolve()
        self.spec = ExperimentSpec.from_yaml(self.spec_path)

        # Validate spec
        errors = self.spec.validate()
        if errors:
            raise ValueError(f"Invalid experiment spec: {'; '.join(errors)}")

        # Override seeds if provided
        self.seeds = seeds if seeds is not None else self.spec.seeds

        # Filter arms if specified
        self.arm_names = arms if arms is not None else list(self.spec.arms.keys())
        for arm_name in self.arm_names:
            if arm_name not in self.spec.arms:
                raise ValueError(f"Unknown arm: {arm_name}")

        # Detection params from spec or override
        if detection_params is not None:
            self.detection_params = detection_params
        else:
            self.detection_params = DetectionParams(
                step_tolerance=self.spec.event_consensus_rules.step_tolerance,
            )

        self.max_retries = max_retries
        self.run_analysis = run_analysis
        self.run_comparison = run_comparison
        self.step_multiplier = step_multiplier

        # Setup directories
        self.exp_dir = _experiment_dir(self.spec.exp_id)
        self.outputs_dir = self.exp_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize result
        self.result = ExperimentResult(
            exp_id=self.spec.exp_id,
            spec_hash=self.spec.compute_spec_hash(),
        )

    def run(self) -> ExperimentResult:
        """Run the complete experiment.

        Returns:
            ExperimentResult with all arm results and comparison report
        """
        logger.info(f"Starting experiment: {self.spec.exp_id}")
        logger.info(f"  Hypothesis: {self.spec.hypothesis}")
        logger.info(f"  Arms: {self.arm_names}")
        logger.info(f"  Seeds: {self.seeds}")

        print(f"\n{'='*60}")
        print(f"Experiment: {self.spec.exp_id}")
        print(f"Hypothesis: {self.spec.hypothesis}")
        print(f"Arms: {', '.join(self.arm_names)}")
        print(f"Seeds per arm: {self.seeds}")
        print(f"{'='*60}\n")

        # Run each arm
        for i, arm_name in enumerate(self.arm_names, 1):
            print(f"\n[{i}/{len(self.arm_names)}] Running arm: {arm_name}")
            print("-" * 40)

            arm_result = self._run_arm(arm_name)
            self.result.arms[arm_name] = arm_result

            if arm_result.error:
                print(f"  ERROR: {arm_result.error}")
            else:
                print(f"  Test ID: {arm_result.test_id}")
                print(f"  Consensus events: {arm_result.consensus_event_count}")

        # Run comparison if requested and we have results
        if self.run_comparison and len(self.result.arms) >= 2:
            print(f"\n{'='*60}")
            print("Running cross-arm comparison...")
            print(f"{'='*60}\n")

            comparison_path = self._run_comparison()
            self.result.comparison_report_path = comparison_path

        # Mark complete
        self.result.completed_at = datetime.now()

        # Save result manifest
        self._save_manifest()

        # Print summary
        self._print_summary()

        return self.result

    def _run_arm(self, arm_name: str) -> ArmResult:
        """Run a single arm of the experiment."""
        arm_spec = self.spec.arms[arm_name]
        spec_dir = self.spec_path.parent

        # Resolve test config path
        test_config_path = spec_dir / arm_spec.test_config
        if not test_config_path.exists():
            return ArmResult(
                arm_name=arm_name,
                error=f"Test config not found: {test_config_path}",
            )

        # Generate test_id for this arm
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_id = f"{self.spec.exp_id}_{arm_name}_{timestamp}"

        try:
            # Create test orchestrator
            orchestrator = TestOrchestrator(
                config_path=test_config_path,
                seeds=self.seeds,
                test_id=test_id,
                detection_params=self.detection_params,
                max_retries=self.max_retries,
                run_analysis=self.run_analysis,
                run_comparison=True,  # We need consensus for comparison
                step_multiplier=self.step_multiplier,
            )

            # Run the test
            test_config = orchestrator.run()

            # Get consensus event count
            consensus_count = 0
            if test_config.consensus_events is not None:
                consensus_count = len(test_config.consensus_events)

            return ArmResult(
                arm_name=arm_name,
                test_config=test_config,
                test_id=test_config.test_id,
                consensus_event_count=consensus_count,
            )

        except Exception as e:
            logger.exception(f"Error running arm {arm_name}")
            return ArmResult(
                arm_name=arm_name,
                error=str(e),
            )

    def _run_comparison(self) -> Path | None:
        """Run cross-arm comparison and generate report."""
        try:
            # Collect arm data for comparison
            arm_data = {}
            for arm_name, arm_result in self.result.arms.items():
                if arm_result.error is None and arm_result.test_config is not None:
                    arm_data[arm_name] = arm_result

            if len(arm_data) < 2:
                logger.warning("Need at least 2 successful arms for comparison")
                return None

            # Generate comparison
            comparison_results = compare_arms(arm_data, self.spec)

            # Generate report
            report_path = self.outputs_dir / "compare.md"
            generate_comparison_report(
                comparison_results=comparison_results,
                spec=self.spec,
                output_path=report_path,
            )

            print(f"Comparison report: {report_path}")
            return report_path

        except Exception as e:
            logger.exception("Error generating comparison")
            print(f"ERROR generating comparison: {e}")
            return None

    def _save_manifest(self) -> None:
        """Save experiment result manifest."""
        manifest_path = self.outputs_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)
        logger.info(f"Saved manifest: {manifest_path}")

    def _print_summary(self) -> None:
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {self.spec.exp_id}")
        print(f"{'='*60}")

        print("\nArm Summary:")
        for arm_name, arm_result in self.result.arms.items():
            status = "ERROR" if arm_result.error else "OK"
            events = arm_result.consensus_event_count
            print(f"  {arm_name}: {status} ({events} consensus events)")

        if self.result.comparison_report_path:
            print(f"\nComparison report: {self.result.comparison_report_path}")

        print(f"\nOutput directory: {self.outputs_dir}")


def run_experiment(
    spec_path: Path | str,
    seeds: list[int] | None = None,
    arms: list[str] | None = None,
    dry_run: bool = False,
) -> ExperimentResult | None:
    """Convenience function to run an experiment.

    Args:
        spec_path: Path to experiment.yaml
        seeds: Override seeds (for testing)
        arms: Only run these arms (for testing)
        dry_run: Just print plan, don't execute

    Returns:
        ExperimentResult, or None if dry_run
    """
    spec = ExperimentSpec.from_yaml(spec_path)

    if dry_run:
        print("[dry-run] Experiment configuration:")
        print(f"  Spec: {spec_path}")
        print(f"  Exp ID: {spec.exp_id}")
        print(f"  Hypothesis: {spec.hypothesis}")
        print(f"  Isolates: {spec.isolates}")
        print(f"  Invariants: {spec.invariants}")
        print(f"  Arms: {list(spec.arms.keys())}")
        print(f"  Seeds: {seeds if seeds else spec.seeds}")
        print(f"  Primary outcomes: {spec.primary_outcomes}")
        return None

    orchestrator = ExperimentOrchestrator(
        spec_path=spec_path,
        seeds=seeds,
        arms=arms,
    )
    return orchestrator.run()
