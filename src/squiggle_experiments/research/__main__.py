"""Entry point for running research training as a module.

Usage:
    python -m squiggle_experiments.research --config path/to/config.yaml
    python -m squiggle_experiments.research --preset debug
    python -m squiggle_experiments.research --preset default
"""

from .trainer import main

if __name__ == "__main__":
    main()
