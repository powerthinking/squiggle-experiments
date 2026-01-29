"""Experiment runner pipeline for multi-arm experiments with cross-arm comparison.

An Experiment is a structured comparison of 2+ Tests, where each Test is a multi-seed
run with identical config except seed (as defined in design_and_contracts.md).

Hierarchy:
    Experiment
      └── Arms (2+ Tests)
            └── Test (multi-seed runs)
                  └── Runs (individual training executions)
"""

from .compare import compare_arms, generate_comparison_report
from .orchestrator import ExperimentOrchestrator
from .types import ArmResult, ExperimentResult

__all__ = [
    "ArmResult",
    "ExperimentOrchestrator",
    "ExperimentResult",
    "compare_arms",
    "generate_comparison_report",
]
