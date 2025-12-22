# squiggle-experiments

Training and instrumentation code for Squiggle experiments.

This repo is responsible for:
- Model training
- Probe evaluation
- Activation capture
- Writing raw artifacts for analysis

## Scout (v0)

The `scout` pipeline is a *thin-slice training run* designed to:

- Validate instrumentation correctness
- Produce geometry-rich activations
- Reach task mastery quickly
- Minimize experimental surface area

Scout runs are **not** meant to be benchmarks.

## What Scout Produces

During training, Scout writes:

- Scalar metrics (loss, lr, grad_norm)
- Fixed probe A metrics
- Periodic activation captures
- Trigger-based captures (optional)

All outputs conform to `squiggle-core` schemas.

## Configuration

Scout runs are fully config-driven via YAML:

```yaml
model:
task:
capture:
probes:
triggers:
```

The goal is to make experiment intent explicit and auditable.

## What This Repo Does NOT Do

- Geometry computation
- Event detection
- Reporting
- Cross-run analysis

Those belong downstream.

## Philosophy

Training is noisy.
Instrumentation must not be.

This repo prioritizes:

- Determinism
- Traceability
- Minimal cleverness