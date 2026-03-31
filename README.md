# FunSearch-inspired NAS prototype

This repository contains a **practical prototype** showing how to adapt ideas from:

- Nature paper: *Mathematical discoveries from program search with large language models* (doi: `10.1038/s41586-023-06924-6`).
- Reference code: `google-deepmind/funsearch` implementation directory (especially `funsearch.py`, `programs_database.py`, `evaluator.py`, `sampler.py`).

## What to transfer from FunSearch to NAS

FunSearch’s key design pattern is to evolve **programs/functions** that drive decision making, instead of searching for raw outputs. For NAS, this maps naturally to evolving:

1. **Architecture scoring functions** (rank candidate networks), or
2. **Operator selection functions** (choose block type/expansion/attention), or
3. **Training hyper-heuristics** (learning rate, augmentation policy, regularization schedule).

This prototype implements (1): it evolves a policy function that scores candidate architectures.

## Prototype design

`nas_funsearch_prototype.py` includes:

- **Skeleton + critical function search**: fixed search loop, only policy coefficients evolve.
- **Best-shot sampling analogue**: mutation starts from top-K members in each island.
- **Island-based diversity**: multiple sub-populations with periodic migration.
- **Evaluator guardrail**: every candidate is executed and scored via a reproducible evaluator.

## Quickstart

```bash
python nas_funsearch_prototype.py
```

Expected output: best discovered policy and score.

## How to adapt this to your real NAS stack

Replace proxy objective (`_objective`) with your training loop:

1. Sample architecture from your search space (e.g., NB201, OFA, custom transformer search space).
2. Train partially (few epochs, shared weights, or supernet subnet eval).
3. Return a scalar objective (e.g., `val_acc - λ * log(FLOPs) - μ * latency_ms`).

Then add a model-backed proposer:

- Current code uses Gaussian mutation.
- In production, generate code/function candidates from an LLM prompt seeded by top programs (FunSearch-style best-shot prompting).

## Suggested NAS-specific upgrades

- Multi-objective Pareto archive (accuracy/latency/memory).
- Constraint-aware evaluator (hardware or deployment SLO constraints).
- Asynchronous workers for scalable evaluation.
- Novelty score in selection pressure to avoid architecture collapse.
- Validation on held-out tasks for transfer robustness.

## Minimal API

- `NASFunSearch.run()` → returns best `Candidate`.
- `format_candidate(best)` → pretty-print policy coefficients.

---

If you want, next step I can extend this into a **real trainer-backed pipeline** (e.g., PyTorch + CIFAR-10 + latency-aware objective).
