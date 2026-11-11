# CUDA depth, pruning, and graph validation — 2026-11-11

The complete suite passed with CUDA enabled: **123 tests and 25 subtests passed**,
with **zero failures, errors, or skips**, in 71.93 seconds. All six CUDA example
variants passed, and their saved champions reloaded and ran on CPU. No library
implementation fix was needed. Expanded persistent CUDA tests and this report
record the validation of the new features.

## Hardware and source

- NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes usable memory,
  compute capability 8.9; driver 555.97.
- PyTorch 2.12.1+cu126, CUDA build 12.6; Python 3.11.9 on Windows.
- Package version `0.1.0a6`, with the working-tree MLP depth/pruning and dynamic
  accelerated PPO changes. This validates the modified source, not the earlier
  published alpha-6 artifacts.
- Starting commit, source/test SHA-256 hashes, exact hardware context, test
  summaries, and example outputs are in the
  [machine-readable report](cuda_depth_validation_2026-11-11.json).

## Coverage

- Hidden-layer insertion and removal, lifetime width shrinking, and pruning to
  zero hidden layers, with both regular and accelerated PPO on CUDA.
- Mixed-depth actor/critic outputs, PPO losses and gradients against separate
  networks; inactive slots have zero gradients. Independent Adam state exports
  to CPU and restores correctly to padded CUDA storage.
- Structural changes retain parameter storage, peer parameters/optimizer state,
  and live environments. Removed slots clear before regrowth. New layers learn.
- Accelerated population chunks and member reordering preserve each member's
  seeded training stream and learned parameters within numerical tolerances.
- Exact generation-boundary checkpoint resume with dynamic depth, both PPO paths,
  and both inheritance modes. Checks cover native tensor environments with
  discrete actions and hybrid Python environments with continuous actions;
  the hybrid checks also compare optimizer moments exactly.
- Frozen mixed-depth evaluation for discrete/continuous actions and native/hybrid
  environments, without changing genomes.
- Sparse graph CUDA learning and evaluation, plus complete discrete/continuous
  graph examples with portable CPU champions.
- Existing CUDA width-growth, inference, distributions, clipping, independent
  Adam, terminal/truncation bootstrapping, checkpoint, and SB3 PPO/DQN checks.

`DepthCUDA` now repeats the detailed variable-depth PPO invariants on GPU, in
addition to the explicit CUDA growth/pruning smoke test. The expanded depth suite
passed **26 tests and 4 subtests**. Core-only depth validation passed its seven
applicable tests, with 19 PyTorch-dependent tests skipped in that separate
environment. The full environment had no skips.

## CUDA examples

Every MLP example below exercised width growth, width shrinking, layer addition,
and layer removal. Every saved champion passed a frozen CPU evaluation.

| Example | PPO | Actions | Inheritance | Chunk size |
|---|---|---|---|---|
| MLP dynamic depth | Regular | Discrete | Lamarckian | — |
| MLP dynamic depth | Regular | Continuous | Baldwinian | — |
| MLP dynamic depth | Accelerated | Discrete | Lamarckian | 2 |
| MLP dynamic depth | Accelerated | Continuous | Baldwinian | 2 |
| Sparse graph | Regular | Discrete | Lamarckian | — |
| Sparse graph | Regular | Continuous | Baldwinian | — |

Reproduce from the repository with the CUDA-capable project environment:

```sh
python -m pytest -q -ra
python -m pytest tests/test_depth.py -q -ra
python examples/neural_growth.py --device cuda --dynamic-depth
python examples/neural_growth.py --device cuda --dynamic-depth --continuous --inheritance baldwinian
python examples/neural_growth.py --device cuda --dynamic-depth --accelerated --chunk-size 2
python examples/neural_growth.py --device cuda --dynamic-depth --accelerated --chunk-size 2 --continuous --inheritance baldwinian
python examples/graph_brain.py --device cuda
python examples/graph_brain.py --device cuda --continuous --inheritance baldwinian
```

The full suite emitted one expected SB3 warning that MLP PPO may be slower on GPU;
its tests passed. These are correctness checks on this hardware/software
combination, not new performance benchmarks or a promise of cross-device bitwise
identity. Earlier [CUDA validation](cuda_validation.md) and
[accelerated benchmarks](accelerated_ppo_benchmarks_2026-11-05.md) retain their
original measurement context.
