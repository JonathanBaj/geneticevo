# CUDA hardware validation - 2026-10-30

CUDA execution passed on the local NVIDIA GeForce RTX 4060 Laptop GPU. The full
suite completed with **61 tests and 21 subtests passed**, zero failures/errors,
and **zero skips**, in 32.27 seconds. All seven CUDA example variants passed.
The persistent hardware tests are in `tests/test_cuda.py`, alongside the existing
tensor and growth CUDA tests. No library implementation fix was needed.

## Hardware and source context

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8.00 GiB usable device memory;
  compute capability 8.9; one CUDA device.
- NVIDIA driver: 555.97; PyTorch 2.12.1+cu126, CUDA build 12.6.
- Python 3.11.9 on Windows-10-10.0.26300-SP0.
- Source version: `0.1.0a3`, including the uncommitted neural-growth feature.
  Installed editable-distribution metadata still reports `0.1.0a2`; the tests execute
  the current source tree. The starting Git commit and source SHA-256 hashes are
  recorded in [cuda_validation.json](cuda_validation.json).

## Correctness coverage

- CUDA actor/critic outputs, discrete/continuous distributions, PPO losses and
  gradients, per-individual clipping, and three Adam steps compared with CPU.
- Actual CUDA parameter placement, export to NumPy, and CPU-portable optimizer state.
- Native CUDA environment transitions, inactive-lane masks, partial resets,
  terminal/truncation bootstrapping, independent CUDA RNG streams, and absence of
  gradients between individuals. An out-of-range CUDA device is rejected.
- Frozen native and hybrid GPU-policy evaluation compared with serial CPU
  evaluation, including mixed network widths and a partial final episode batch.
- Exact generation-boundary checkpoint resume in all **16 combinations** of
  discrete/continuous actions, native/hybrid environments, fixed/growing
  networks, and Lamarckian/Baldwinian inheritance. Portable champion save/load
  and CPU inference were checked. Exact comparisons apply to this same device
  and software configuration, rather than promising cross-device identity.
- CUDA SB3 PPO (discrete/continuous) and DQN: actual training, caller RNG
  isolation, fresh optimizer/replay state, resumed learning, and portable CPU
  champion loading. SB3 emitted its expected MLP-PPO GPU performance warning;
  it did not prevent training or cause a test failure.
- Scheduled and value-loss-gated neural growth, inherited sizes, mutation,
  changing architectures, and the complete CUDA examples below.

Run the hardware tests and full suite:

```sh
python -m pytest -q tests/test_cuda.py tests/test_tensor.py tests/test_growth.py
python -m pytest -q -ra
```

## Successful CUDA examples

```sh
python examples/native_ppo.py --device cuda
python examples/native_ppo.py --device cuda --continuous
python examples/neural_growth.py --device cuda
python examples/neural_growth.py --device cuda --inheritance baldwinian
python examples/neural_growth.py --device cuda --continuous
python examples/neural_growth.py --device cuda --continuous --inheritance baldwinian
python examples/neural_growth.py --device cuda --continuous --inheritance baldwinian --value-loss-threshold 0.5
```

## Complete-run benchmark

`equal_work.py` completed **72 runs**: populations 4/16, environment copies
1/4, three seeds, and six execution modes. All 12 matched groups had equal
interaction counts and matching fitness (absolute tolerance 1e-5). Runs include
initialization, reproduction, frozen evaluation, serial archive reference
comparisons, and result copies. Imports and warm-up are excluded. CUDA timings
are explicitly synchronized. CPU PyTorch threads were fixed at one.

Representative configuration: **16 agents, 4 environments per agent**, four
selection episodes, two generations, horizon 16, hidden widths (16,16), and
2,240 total evaluation interactions including archive comparisons. Times are
medians of three runs; all configurations are retained in the JSON snapshot.

| Mode | Complete seconds | Maximum allocated CUDA MiB |
|---|---:|---:|
| `batched_native_cpu` | 0.1169 | - |
| `batched_native_cuda` | 0.2114 | 8.19 |
| `batched_python_cpu` | 0.1385 | - |
| `hybrid_cuda` | 0.2287 | 8.19 |
| `parallel_cpu` | 1.6306 | - |
| `serial_cpu` | 0.2456 | - |

The GPU was slower than native CPU for these small workloads. These results
verify execution and expose overhead; they do not establish a general GPU
speedup or identify the workload size where one would occur.

## PPO rollout and update benchmark

The PPO benchmark now accepts `--device cpu` or `--device cuda`, retaining CPU
as the default. CUDA timings synchronize before and after learning. Each row
below uses population batching, 64 training transitions per agent, two PPO
epochs, and minibatches of 16. Counts match between CPU/CUDA and independent/
batched routes. These are single timing samples; stochastic training trajectories
differ across routes and devices. Frozen test results are recorded but are not
an algorithm comparison.

| Agents | Environments each | CPU seconds | CUDA seconds | Training transitions | Individual optimizer updates |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.0804 | 0.2026 | 256 | 32 |
| 4 | 4 | 0.0585 | 0.1131 | 256 | 32 |
| 16 | 1 | 0.1040 | 0.4402 | 1024 | 128 |
| 16 | 4 | 0.0732 | 0.2188 | 1024 | 128 |

The maximum allocated PyTorch CUDA memory across the PPO runs was 16.55 MiB.
This measures live allocations reported by PyTorch, rather than total driver
memory, allocator reservation, or peak host/child-process memory. Transfer-only
timing, profiler traces, compilation, mixed precision, and multi-GPU execution
were not measured by this validation.

Reproduce the benchmark configurations:

```sh
python benchmarks/equal_work.py --populations 4 16 --copies 1 4 --repeats 3
python benchmarks/ppo_updates.py --device cuda
python benchmarks/ppo_updates.py --device cpu
```

Raw hardware information, exact example output, command logs, JUnit reports,
and benchmark JSON are in `results/cuda_validation/2026-10-30/` (ignored local results).
The compact tracked snapshot is [cuda_validation.json](cuda_validation.json).
The older alpha release/benchmark records keep their original dates and results.
