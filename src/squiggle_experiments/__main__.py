"""Main entry point for squiggle-experiments.

Auto-detects config type (Scout vs Research) based on config contents.
"""

import argparse
import sys
from pathlib import Path

import yaml


def _is_research_config(config_path: Path) -> bool:
    """Check if config is for Research trainer (has model.size)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_cfg = raw.get("model", {})
    # Research configs have "size" (e.g., "350m"), Scout configs have "n_layers"
    return "size" in model_cfg


def main():
    parser = argparse.ArgumentParser(description="Run squiggle training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    if _is_research_config(config_path):
        # Research trainer
        from squiggle_experiments.research.trainer import run_research

        run_research(config_path=str(config_path), seed_override=args.seed)
    else:
        # Scout trainer
        from squiggle_experiments.scout.run import run_scout
        run_scout(config_path, seed_override=args.seed)


if __name__ == "__main__":
    main()
