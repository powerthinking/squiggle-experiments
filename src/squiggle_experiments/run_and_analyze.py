from __future__ import annotations

import argparse
import sys

from squiggle_experiments.scout.run import run_scout
from squiggle_analysis.run import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scout + analysis end-to-end")
    parser.add_argument("--config", required=True, help="Path to scout YAML config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute geometry/events/report even if outputs exist",
    )
    args = parser.parse_args()

    # 1) Run scout
    run_id = run_scout(args.config)

    # 2) Run analysis
    run_analysis(run_id=run_id, force=args.force)

    print(f"[✓] run_and_analyze complete: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
