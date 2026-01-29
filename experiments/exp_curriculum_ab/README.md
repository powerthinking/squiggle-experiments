# Experiment: Curriculum A/B (Event Yield)

**Exp ID**: `exp_curriculum_ab`

## Hypothesis

Different curricula produce different event yields. Blocked and ramp curricula yield more consensus events than iid/balanced baselines.

## Design

### Invariants (What Stays Constant)
- Model config: 350M parameter research transformer
- Total steps: 374 (~17 epochs)
- Dataset: split_v1_s42 (1408 training items, 16 families)
- Optimizer: AdamW, lr=3e-4
- Batch size: 64 effective (2 × 32 accumulation)
- Seed set: [42, 123, 456] (identical across all arms)

### Differences Between Arms

| Arm | Curriculum | Description |
|-----|------------|-------------|
| `iid` | `iid_baseline.yaml` | Proportional sampling, no structure |
| `balanced` | `balanced_baseline.yaml` | Family-balanced sampling |
| `blocked` | `family_blocked.yaml` | One target family at a time, then mix |
| `ramp` | `family_ramp.yaml` | Targets-only → ramp controls → full mix |

### Seed Plan
- Seeds: 42, 123, 456
- Each arm runs as a Test with 3 seeds
- Total runs: 4 arms × 3 seeds = 12 runs

## Primary Outcomes

1. **consensus_event_count**: Number of seed-invariant events per arm
2. **event_yield_per_step**: Events / total_steps ratio
3. **mean_final_loss**: Average final loss across seeds

## Event Consensus Rules

```yaml
step_tolerance: 5
min_seed_fraction: 1.0  # Event must appear in all seeds
```

## Usage

```bash
# Dry run to see plan
squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml --dry-run

# Run full experiment (all 4 arms, 3 seeds each)
squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml

# Run subset for testing
squiggle-experiment --spec experiments/exp_curriculum_ab/spec/experiment.yaml \
  --arms iid blocked --seeds 42 123
```

## Outputs

After running:
```
experiments/exp_curriculum_ab/
  outputs/
    manifest.json       # Experiment result metadata
    compare.md          # Cross-arm comparison report
```

## Produced Run IDs

(Will be populated after experiment runs)

| Arm | Seed | Run ID |
|-----|------|--------|
| iid | 42 | |
| iid | 123 | |
| iid | 456 | |
| balanced | 42 | |
| ... | ... | ... |

## Links to Reports

- [Comparison Report](outputs/compare.md)
