# Squiggle Experiments  
**A/B Research Paths for Studying Learning Geometry and Event Structure**

This repository contains a structured set of **controlled A/B experiment paths** designed to probe how transformer models acquire arithmetic and symbolic structure over training.

Rather than optimizing for final accuracy, these experiments are explicitly designed to surface:

- **Geometric phase transitions**
- **Subspace specialization and collapse**
- **Bifurcations, repairs, and integration events**
- **Temporal correspondences across runs (“squiggle matching”)**

Each path is constructed to isolate *one primary causal factor* while holding others fixed. All paths are intended to be run in two stages:

1. **Scout runs** — lightweight sanity checks to confirm expected behavior  
2. **Research runs** — deep logging with full trajectory capture for geometric analysis

---

## How to Read These Paths

Each path below specifies:

- **What it isolates** — the primary experimental variable  
- **A/B variants** — the exact comparison  
- **Hypothesis** — expected *squiggle-level* behavior (not just loss curves)

These hypotheses are intentionally falsifiable. A “null” outcome (no meaningful geometric difference) is considered a valid and informative result.

---

## Path 0 — Baseline Sanity (Anchor)

**Purpose:** Control condition for all other experiments.

- **Tokenization:** Digit tokens (e.g. `"123"` → `"1 2 3"`)
- **Curriculum:** Low → high difficulty
- **DDAR / traces:** Off
- **Proof families:** None

**Hypothesis:**  
This setting should produce the *cleanest and most interpretable* phase transitions (e.g. carry, long multiplication, division with remainder), since no structure is injected beyond the raw task. These runs serve as alignment anchors for squiggle matching across all other paths.

---

## Path 1 — Curriculum Reversal (A/B)

**Isolates:** Ordering effects in curriculum design

- **A:** Low → high difficulty  
- **B:** High → low difficulty (start hard, then “relax”)

**Hypothesis:**  
Reversed curricula often induce **early brittle shortcuts or collapsed representations**, followed by later *repair squiggles* when simpler structure arrives. Expect sharper collapses and delayed reintegration compared to the baseline.

---

## Path 2 — Interleaved vs Staged Tasks (A/B)

**Isolates:** Mixture vs staging effects

- **A:** Staged blocks (add → sub → mul → div)
- **B:** Interleaved from step 1 (small % of future tasks introduced early)

**Hypothesis:**  
Interleaving should produce **smoother representational drift**, while staged curricula should generate **punctuated geometric events** at task boundaries. This path is particularly valuable for validating event-detection methods.

---

## Path 3 — Tokenization Granularity (A/B)

**Isolates:** Representational granularity

- **A:** Digit tokens (`"123"` → `"1 2 3"`)
- **B:** Integer tokens (`"123"` as atomic token), with range-bucket vocab (e.g. `0–9999`)

**Hypothesis:**  
Integer-token models may develop **table-lookup-like behavior** early, while digit-token models should show **stronger algorithmic squiggles**, especially around carry and place-value mechanics.

---

## Path 4 — Hybrid Tokenization (Exploratory)

**Isolates:** Compositionality vs memorization pressure

- **Hybrid:** Integers appear both digitized and atomic, with explicit delimiters

**Hypothesis:**  
This setting may induce **bifurcated internal mechanisms**—parallel pathways for symbolic composition and memorization—that later align or collapse. This path is a prime target for squiggle matching across subspaces.

---

## Path 5 — DDAR Traces as Supervision (A/B)

**Isolates:** Effect of intermediate structure

- **A:** Final answer only  
- **B:** Answer + DDAR-style traces (no external tools at inference)

**Hypothesis:**  
Trace supervision often leads to **earlier subspace specialization** and fewer late “surgery” events. If squiggle events still occur, they should shift earlier in training—a strong test of whether events are timing-dependent or structurally inevitable.

---

## Path 6 — DDAR as Data Generator (Not Teacher)

**Isolates:** Data geometry vs step-by-step supervision

- **A:** Vanilla arithmetic distribution  
- **B:** DDAR-generated problem distribution (edge cases, adversarial carry chains), final answer only

**Hypothesis:**  
If geometric events are driven primarily by **data structure**, this path should reproduce similar event timing to Path 5—but with distinct internal signatures. Divergence here would strongly suggest that *how* structure is introduced matters, not just *what* appears in the data.

---

## Path 7 — Proof Families as Probes

**Isolates:** Interaction between proof reasoning and arithmetic geometry

- **Base curriculum:** Path 0  
- **Intervention:** Short proof-family bursts (≈1–5% of steps), injected at controlled times

**Hypothesis:**  
Proof bursts may induce **localized geometric perturbations** that either:

1. Remain isolated  
2. Integrate with arithmetic representations  
3. Overwrite existing structure  

All three outcomes are informative and directly support the broader research narrative.

---

## Path 8 — Proof-First vs Arithmetic-First (A/B)

**Isolates:** Priming direction

- **A:** Arithmetic warmup → proofs  
- **B:** Proof warmup → arithmetic

**Hypothesis:**  
Proof-first training may create **symbolic scaffolding** that alters the downstream arithmetic learning trajectory. If present, this would manifest as persistent geometric differences long after the proof data disappears.

---

## Path 9 — Minimal-Pair Micro-Curricula

**Isolates:** Onset of specific mechanisms

Ultra-targeted curricula designed to trigger *one* mechanism at a time:

- Addition without carry → add carry  
- Addition with carry from start  
- Borrowing  
- Long multiplication  
- Division with remainder

**Hypothesis:**  
These runs are optimized for **precise event localization** and for generating **clean, matchable squiggle signatures**. They are the strongest tools for mapping mechanism-specific phase transitions.

---

## Closing Notes

This repository is not a leaderboard-oriented benchmark suite. It is an **instrumented experimental framework** for studying how learning unfolds internally.

Null results, unexpected stability, or absence of events are all considered valid outcomes—often more informative than dramatic effects.

If squiggle matching is going to work as a general interpretability primitive, it must survive experiments like these.
