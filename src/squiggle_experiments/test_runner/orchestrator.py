"""Test orchestrator for multi-seed training runs."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from squiggle_core import paths

from .consensus import extract_consensus_events, generate_consensus_report
from .types import (
    ConfidenceLevel,
    DetectionParams,
    RunResult,
    RunStatus,
    TestConfig,
    TestSummary,
)

logger = logging.getLogger(__name__)


def _compute_config_hash(config_path: Path) -> str:
    """Compute SHA256 hash of config file contents."""
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _generate_test_id(config_path: Path) -> str:
    """Generate a test ID from config name and timestamp."""
    config_name = config_path.stem  # e.g., "scout_tiny"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"test_{config_name}_{timestamp}"


def _parse_run_id_from_output(stdout: str) -> Optional[str]:
    """Parse run_id from training subprocess output."""
    # Match: "[✓] Scout run complete: {run_id}" or similar
    patterns = [
        r"run complete: (\S+)",
        r"Run complete: (\S+)",
        r"run_id[=:]\s*(\S+)",
        r"Run ID: (\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            return match.group(1)
    return None


def _find_latest_run_id(config_name: str, seed: int, before_time: datetime) -> Optional[str]:
    """Find the most recent run_id matching config name and seed.

    Looks for runs created after before_time.
    """
    runs_dir = paths.runs_root() / "runs"
    if not runs_dir.exists():
        return None

    # Pattern: YYYYMMDD_HHMMSS_{config_name}_s{seed}
    pattern = re.compile(rf"\d{{8}}_\d{{6}}_.*_s{seed}$")

    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not pattern.match(run_dir.name):
            continue

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            # Check creation time
            mtime = datetime.fromtimestamp(meta_path.stat().st_mtime)
            if mtime > before_time:
                candidates.append((mtime, run_dir.name))

    if candidates:
        # Return most recent
        candidates.sort(reverse=True)
        return candidates[0][1]

    return None


def _get_final_loss(run_id: str) -> Optional[float]:
    """Get final loss from run's metrics."""
    try:
        wide_path = paths.metrics_wide_path(run_id)
        if wide_path.exists():
            df = pd.read_parquet(wide_path)
            if "loss" in df.columns:
                return float(df["loss"].iloc[-1])
    except Exception:
        pass
    return None


def _get_event_count(run_id: str, analysis_id: str) -> Optional[int]:
    """Get event count from run's events file."""
    try:
        events_path = paths.events_candidates_path(run_id, analysis_id)
        if events_path.exists():
            df = pd.read_parquet(events_path)
            return len(df)
    except Exception:
        pass
    return None


class TestOrchestrator:
    """Orchestrates multi-seed training runs and analysis."""

    def __init__(
        self,
        config_path: Path,
        seeds: list[int],
        test_id: Optional[str] = None,
        detection_params: Optional[DetectionParams] = None,
        max_retries: int = 2,
        run_analysis: bool = True,
        run_comparison: bool = True,
        llm_analysis: bool = False,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o",
        step_multiplier: Optional[float] = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.seeds = seeds
        self.test_id = test_id or _generate_test_id(self.config_path)
        self.detection_params = detection_params or DetectionParams()
        self.max_retries = max_retries
        self.run_analysis = run_analysis
        self.run_comparison = run_comparison
        self.llm_analysis = llm_analysis
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.step_multiplier = step_multiplier

        # Analysis ID generated from detection params
        self.analysis_id = paths.generate_analysis_id(
            warmup_fraction=self.detection_params.warmup_fraction,
            max_pre_warmup=self.detection_params.max_pre_warmup,
            peak_suppression_radius=self.detection_params.peak_suppression_radius,
            max_events_per_series=self.detection_params.max_events_per_series,
            adaptive_k=self.detection_params.adaptive_k,
        )

        # Initialize test config
        self.test_config = TestConfig(
            test_id=self.test_id,
            config_path=self.config_path,
            config_hash=_compute_config_hash(self.config_path),
            seeds=seeds,
            created_at=datetime.now(),
            detection_params=self.detection_params,
            runs=[RunResult(seed=s) for s in seeds],
        )

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging to test log file."""
        log_path = paths.test_log_path(self.test_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _run_training_subprocess(self, seed: int) -> tuple[Optional[str], Optional[str]]:
        """Run training subprocess for a single seed.

        Lets output go directly to terminal so progress bars work properly.
        Finds run_id from filesystem after completion.

        Returns:
            (run_id, error_message) - run_id if successful, error_message if failed
        """
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            "-m",
            "squiggle_experiments",
            "--config",
            str(self.config_path),
            "--seed",
            str(seed),
            "--quiet",  # Suppress checkpoint/epoch prints for cleaner test output
        ]

        # Add step_multiplier if set
        if self.step_multiplier is not None:
            cmd.extend(["--step-multiplier", str(self.step_multiplier)])

        logger.info(f"Running training: {' '.join(cmd)}")
        start_time = datetime.now()

        try:
            # Let both stdout and stderr go directly to terminal
            # This allows tqdm progress bars to work properly
            result = subprocess.run(
                cmd,
                timeout=7200,  # 2 hour timeout (research runs can take 60-90 min)
            )

            if result.returncode != 0:
                error_msg = f"Training failed with return code {result.returncode}"
                logger.error(error_msg)
                return None, error_msg

            # Find run_id from filesystem (look for runs created after start_time)
            run_id = _find_latest_run_id(
                config_name=self.config_path.stem,
                seed=seed,
                before_time=start_time,
            )

            if run_id:
                logger.info(f"Training completed successfully: {run_id}")
                return run_id, None
            else:
                error_msg = "Could not find run_id in filesystem"
                logger.error(error_msg)
                return None, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Training timed out after 1 hour"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Training subprocess error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _run_analysis_subprocess(self, run_id: str) -> Optional[str]:
        """Run analysis on a completed training run.

        Returns:
            error_message if failed, None if successful
        """
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            "-m",
            "squiggle_analysis",
            "--run-id",
            run_id,
            "--analysis-id",
            self.analysis_id,
            "--warmup-fraction",
            str(self.detection_params.warmup_fraction),
            "--max-pre-warmup",
            str(self.detection_params.max_pre_warmup),
            "--suppression-radius",
            str(self.detection_params.peak_suppression_radius),
            "--max-events-per-series",
            str(self.detection_params.max_events_per_series),
        ]

        logger.info(f"Running analysis: {' '.join(cmd)}")

        try:
            # Let output go directly to terminal
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                error_msg = f"Analysis failed with return code {result.returncode}"
                logger.error(error_msg)
                return error_msg

            logger.info(f"Analysis completed successfully for {run_id}")
            return None

        except subprocess.TimeoutExpired:
            error_msg = "Analysis timed out after 10 minutes"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Analysis subprocess error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run_single_seed(self, run_result: RunResult) -> None:
        """Run training and optional analysis for a single seed."""
        seed = run_result.seed
        run_result.status = RunStatus.RUNNING

        for attempt in range(self.max_retries + 1):
            run_result.retries = attempt

            # Run training
            run_id, error = self._run_training_subprocess(seed)

            if run_id:
                run_result.run_id = run_id
                run_result.final_loss = _get_final_loss(run_id)

                # Run analysis if enabled
                if self.run_analysis:
                    analysis_error = self._run_analysis_subprocess(run_id)
                    if analysis_error:
                        run_result.status = RunStatus.ANALYSIS_FAILED
                        run_result.error_message = analysis_error
                        logger.warning(f"Analysis failed for seed {seed}: {analysis_error}")
                    else:
                        run_result.status = RunStatus.SUCCESS
                        run_result.analysis_id = self.analysis_id
                        run_result.event_count = _get_event_count(run_id, self.analysis_id)
                else:
                    run_result.status = RunStatus.SUCCESS

                return

            # Training failed - retry with backoff
            run_result.error_message = error
            if attempt < self.max_retries:
                backoff = 2**attempt
                logger.info(f"Retrying seed {seed} in {backoff}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(backoff)

        # All retries exhausted
        run_result.status = RunStatus.FAILED
        logger.error(f"All retries exhausted for seed {seed}")

    def _compute_summary(self) -> TestSummary:
        """Compute test summary statistics."""
        successful = self.test_config.successful_runs()
        failed = self.test_config.failed_runs()

        summary = TestSummary(
            total_seeds=len(self.seeds),
            successful=len(successful),
            failed=len(failed),
        )

        return summary

    def _classify_confidence(self, summary: TestSummary) -> ConfidenceLevel:
        """Classify confidence level based on summary stats."""
        if (
            summary.successful >= 5
            and summary.jaccard_similarity > 0.50
            and summary.mean_correlation > 0.95
            and summary.consensus_events >= 10
        ):
            return ConfidenceLevel.HIGH

        if (
            summary.successful >= 3
            and summary.jaccard_similarity > 0.25
            and summary.mean_correlation > 0.80
        ):
            return ConfidenceLevel.MODERATE

        return ConfidenceLevel.LOW

    def _write_manifest(self) -> None:
        """Write test manifest to YAML file."""
        manifest_path = paths.test_manifest_path(self.test_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as f:
            yaml.dump(self.test_config.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Manifest written to {manifest_path}")

    def run(self) -> TestConfig:
        """Execute the full test pipeline.

        Returns:
            TestConfig with results
        """
        print(f"[test_runner] Starting test: {self.test_id}")
        print(f"[test_runner] Config: {self.config_path}")
        print(f"[test_runner] Seeds: {self.seeds}")
        print(f"[test_runner] Analysis ID: {self.analysis_id}")

        logger.info(f"Starting test {self.test_id} with {len(self.seeds)} seeds")

        # Run training for each seed sequentially
        for i, run_result in enumerate(self.test_config.runs):
            print(f"\n[test_runner] Running seed {run_result.seed} ({i + 1}/{len(self.seeds)})")
            self._run_single_seed(run_result)

            status_emoji = "✓" if run_result.status == RunStatus.SUCCESS else "✗"
            print(f"[test_runner] Seed {run_result.seed}: [{status_emoji}] {run_result.status.value}")

            # Write manifest after each run (for recovery)
            self._write_manifest()

        # Compute summary
        summary = self._compute_summary()
        self.test_config.summary = summary

        # Run comparison if enabled and we have at least 2 successful runs
        successful_runs = self.test_config.successful_runs()
        if self.run_comparison and len(successful_runs) >= 2:
            print(f"\n[test_runner] Running consensus analysis on {len(successful_runs)} successful runs...")

            try:
                # Load events from all successful runs
                run_events = {}
                for run_result in successful_runs:
                    events_path = paths.events_candidates_path(
                        run_result.run_id, run_result.analysis_id
                    )
                    if events_path.exists():
                        run_events[run_result.run_id] = pd.read_parquet(events_path)

                if len(run_events) >= 2:
                    # Extract consensus events
                    consensus_df, metrics = extract_consensus_events(
                        run_events,
                        step_tolerance=self.detection_params.step_tolerance,
                        test_id=self.test_id,
                    )

                    # Update summary with consensus metrics
                    summary.consensus_events = len(consensus_df)
                    summary.jaccard_similarity = metrics.get("jaccard_similarity", 0.0)
                    summary.mean_correlation = metrics.get("mean_correlation", 0.0)
                    summary.confidence = self._classify_confidence(summary)

                    # Write consensus events
                    consensus_path = paths.events_consensus_path(self.test_id)
                    consensus_path.parent.mkdir(parents=True, exist_ok=True)
                    consensus_df.to_parquet(consensus_path, index=False)
                    print(f"[test_runner] Consensus events written to {consensus_path}")

                    # Generate consensus report
                    report_content = generate_consensus_report(
                        test_config=self.test_config,
                        consensus_df=consensus_df,
                        metrics=metrics,
                        run_events=run_events,
                    )

                    report_path = paths.consensus_report_path(self.test_id)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(report_content)
                    print(f"[test_runner] Consensus report written to {report_path}")

                    # Optional LLM analysis
                    if self.llm_analysis:
                        self._run_llm_analysis(consensus_df, report_content)

            except Exception as e:
                logger.error(f"Consensus analysis failed: {e}")
                print(f"[test_runner] Warning: Consensus analysis failed: {e}")

        elif len(successful_runs) < 2:
            print(f"\n[test_runner] Skipping consensus analysis (need >= 2 successful runs, have {len(successful_runs)})")

        # Final manifest write
        self.test_config.summary = summary
        self._write_manifest()

        # Print summary
        print(f"\n[test_runner] Test complete: {self.test_id}")
        print(f"    Successful: {summary.successful}/{summary.total_seeds}")
        print(f"    Failed: {summary.failed}/{summary.total_seeds}")
        if summary.consensus_events > 0:
            print(f"    Consensus events: {summary.consensus_events}")
            print(f"    Jaccard similarity: {summary.jaccard_similarity:.2%}")
            print(f"    Confidence: {summary.confidence.value}")
        print(f"    Manifest: {paths.test_manifest_path(self.test_id)}")

        return self.test_config

    def _run_llm_analysis(self, consensus_df: pd.DataFrame, report_content: str) -> None:
        """Run LLM analysis on consensus events."""
        try:
            from squiggle_analysis.llm_analysis.analyzer import (
                AnalysisRequest,
                analyze_report,
                write_analysis_result,
            )

            print(f"[test_runner] Running LLM analysis with {self.llm_model}...")

            run_context = {
                "analysis_mode": "test_consensus",
                "test_id": self.test_id,
                "seeds": self.seeds,
                "successful_runs": len(self.test_config.successful_runs()),
                "consensus_events": len(consensus_df),
                "detection_config": self.detection_params.to_dict(),
            }

            request = AnalysisRequest(
                run_context=run_context,
                primary_report=report_content,
                compare_report=None,
                artifacts=[],
                user_question=None,
            )

            result = analyze_report(
                request,
                backend=self.llm_backend,
                model=self.llm_model,
            )

            analysis_path = paths.consensus_llm_analysis_path(self.test_id)
            write_analysis_result(result, analysis_path)
            print(f"[test_runner] LLM analysis written to {analysis_path}")

        except ImportError:
            logger.warning("LLM analysis not available - install squiggle-analysis[llm]")
            print("[test_runner] Warning: LLM analysis not available")
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            print(f"[test_runner] Warning: LLM analysis failed: {e}")


def resume_test(test_id: str) -> TestConfig:
    """Resume a previously started test.

    Args:
        test_id: The test ID to resume

    Returns:
        Updated TestConfig
    """
    manifest_path = paths.test_manifest_path(test_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    test_config = TestConfig.from_dict(data)

    # Find pending runs
    pending = [r for r in test_config.runs if r.status == RunStatus.PENDING]
    if not pending:
        print(f"[test_runner] No pending runs for test {test_id}")
        return test_config

    print(f"[test_runner] Resuming test {test_id} with {len(pending)} pending runs")

    # Create orchestrator with existing config
    orchestrator = TestOrchestrator(
        config_path=test_config.config_path,
        seeds=[r.seed for r in pending],
        test_id=test_id,
        detection_params=test_config.detection_params,
    )

    # Replace test config with loaded one
    orchestrator.test_config = test_config

    return orchestrator.run()
