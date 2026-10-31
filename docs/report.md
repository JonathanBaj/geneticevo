# Delivery report — 2026-10-20

This workspace now contains an installable `src/geneticevo` package, runnable
examples, meaningful tests, bounded benchmarks, and extension documentation.
Nothing was published, deployed, or pushed to a remote repository.
The current local package is alpha `0.1.0a2`, licensed under MIT. Current release
metadata, installation, example, and GPU-validation status are recorded in
[release.md](release.md); the timings below retain their historical build context.

## What works

- CPU vector and fixed-architecture neural evolution, pure evolution without RL,
  tournament/rank/custom selection, elitism, Gaussian/custom mutation, optional
  custom crossover, immigrants, lineage, independent archives, and stopping caps.
- Separate training, rotating selection, fixed archive, and explicit final-test
  distributions; frozen evaluation; shifted-task adaptation on fresh copies;
  multiple-seed summaries; custom task metrics and penalties.
- Atomic schema-versioned generation-boundary checkpoints, including deterministic
  CPU resume, IDs, budgets, archive context, and genetic versus learning state.
- Optional SB3 PPO and DQN with compatible native SB3 policies, Lamarckian and
  Baldwinian inheritance, fresh child optimizers/replay/targets/counters, and
  resumable unchanged elites.
- Process CPU evaluation and explicit population tensor networks. Native PPO
  batches independent losses, gradients, clipping, and Adam, including separate
  step counters for older elites and new children. CPU tensor execution is tested.
- Matching Gymnasium/tensor Reach1D environments, categorical and bounded
  tanh-Gaussian actions, proper terminal/truncation targets, and actual budget
  accounting.
- PettingZoo Parallel protocol adaptation, an official-API-tested reference
  environment, individual/team selection, partner/role rotation, held-out
  partners, deterministic evolved communication, and fixed-message SB3 learning
  against frozen partners.
- JSON metadata/configuration, JSONL metrics, lineage, timing/cost records,
  optional saved plotting, inference saves, and local wheel packaging.

## How to run

Follow installation in [README](../README.md). Start with
`python examples/vector.py` or `python examples/neural.py`. Optional learning:
`python examples/sb3_learning.py --algorithm PPO` (or `DQN`) and
`python examples/native_ppo.py --device cpu`. Shared interaction:
`python examples/shared_world.py`, optionally with `--rl`.
`python examples/generalization.py` demonstrates separate shifted tests and
task-specific budgeted adaptation. All these paths were smoke-tested.

Run `python -m pytest -q` or `python -m unittest discover -s tests -v` for
validation. The latter works without pytest. Optional cases skip when extras are
absent. Windows CPU worker tests need an ordinary terminal or permission to
create worker pipes outside the execution sandbox.

## Validation

The full suite passed **43 tests**, with **one CUDA-only test skipped**. A clean
wheel installation containing none of PyTorch, SB3, PettingZoo, or Matplotlib
passed all **13 core tests** and ran both vector and neural examples. Dependency
checks passed in both the full and core-only project environments. Test coverage
and methodology are described in [validation.md](validation.md). Wheel build and
editable installation completed successfully. The plotting helper generated a
saved fitness plot.

Tested environment: Python 3.11.9, NumPy 2.4.4, Gymnasium 1.2.3, PyTorch
2.12.1+cu126 running on CPU, SB3 2.9.0, PettingZoo 1.25.0, Matplotlib 3.10.6,
pytest 9.1.1, setuptools 81.0.0. The host reports 16 logical CPUs and an AMD64
Family 25 processor. CUDA reported no available devices.

## Measured performance

Three complete runs per cell, two evaluated generations, four selection episodes,
16-step horizon, hidden widths (16,16), identical seed/work per mode, one archive
entry. Median wall seconds:

| Population | Copies per individual | Serial CPU | Two CPU workers | Batched Python CPU | Native tensor CPU | Actual transitions per run |
|---|---|---|---|---|---|---|
| 4 | 1 | 0.1298 | 1.4478 | 0.1490 | 0.1632 | 704 |
| 4 | 4 | 0.1248 | 1.4157 | 0.0917 | 0.0948 | 704 |
| 16 | 1 | 0.3134 | 1.5125 | 0.2081 | 0.1897 | 2,240 |
| 16 | 4 | 0.3133 | 1.5318 | 0.1563 | 0.1264 | 2,240 |

The tiny environment makes process startup expensive. Batching helps some
configurations and has overhead in others. These three-sample timings do not
establish general speedups. GPU/hybrid GPU performance was not measured.
Timings include initialization, evolution, frozen/archive evaluation, and result
copies; imports and explicit warm-up are separate. No compilation was used.

Native PPO smoke timing includes policy/optimizer setup, rollout collection, and
updates. Each individual receives 64 training transitions and eight optimizer
updates in either route:

| Population | Copies | Independent seconds | Batched seconds | Total training transitions |
|---|---|---|---|---|
| 4 | 1 | 0.3846 | 0.0887 | 256 |
| 4 | 4 | 0.2076 | 0.0662 | 256 |
| 16 | 1 | 1.3980 | 0.1288 | 1,024 |
| 16 | 4 | 0.8018 | 0.0760 | 1,024 |

These are single measurements. Stochastic training trajectories differ between
the benchmark routes; frozen outcomes are recorded, and batching did not
consistently improve those outcomes. Numerical update equivalence is established
by separate independent-reference tests, not by this timing experiment.

Three-seed Reach1D frozen-test smoke results (higher is better):

| Method | Mean ± sample std | Lifetime-learning transitions per seed | Evaluation transitions per seed, including final test |
|---|---|---|---|
| Random | -14.738 ± 1.743 | 0 | 256 |
| Evolution only | -3.602 ± 4.925 | 0 | 1,344 |
| PPO only | -1.306 ± 0.879 | 512 | 576 |
| Evolution plus PPO | -0.895 ± 0.164 | 512 | 1,344 |

Learning budgets match for PPO-only and evolution-plus-PPO, but evaluation
budgets differ. These short runs are insufficient to establish superiority.
Separate fixed-policy SignalWorld controls scored 3.521 ± 0.072 with messages
and 2.083 ± 0.289 without them. That task is deliberately constructed to make
another agent's private bit useful; this demonstrates routing and evaluation,
not general communication benefits.

The reproducible scripts and full compact data snapshot are in
[benchmark_snapshot.json](benchmark_snapshot.json). Raw local records are in
`results/benchmarks`. Records include outcomes, actual interactions, component
times, and post-run parent RSS. RSS is not peak host memory and excludes child
process memory. Transfer-only timing, host peak memory, profiler traces, and
CUDA measurements were added on 2026-10-30 in [CUDA validation](cuda_validation.md);
the original snapshot above remains unchanged.

## Unsupported combinations and remaining limitations

- CUDA hardware validation was completed on 2026-10-30 for the configurations
  documented in [CUDA validation](cuda_validation.md). The earlier timings above
  retain their original CPU-only context. Unavailable devices still fail clearly.
- Native PPO supports homogeneous fixed architectures/schedules and synchronous
  stepping. Random filling retains a short population loop; collection uses a
  Python time loop and ordinary tensor allocations. Compilation, mixed precision,
  distributed execution, multi-GPU, and heterogeneous architecture grouping are
  not implemented.
- SB3 objects do not become a batched native learner. SB3 training is sequential
  across individuals, with optional CPU process fitness evaluation. Vector
  genomes and custom shared evaluators do not run through the tensor backend.
- Supported neural spaces are flat Box observations, zero-start Discrete
  actions, and finite bounded flat Box actions. DQN is discrete only. Other
  spaces and normalization wrappers need explicit tested adapters.
- Learned joint stochastic message/task likelihoods, simultaneous independent
  multi-agent RL, delayed message queues, shared experience, and parameter
  averaging are not implemented. The tested learning integration fixes messages
  and partner policies. Evolved message heads work with pure evolution.
- Checkpoints are at evaluated generation boundaries, reset simulations on
  resume, and use trusted Python pickle. Custom factories should be stateless
  given their configuration and seeds. Hidden globals/closures/external state
  cannot be validated or resumed automatically. Cross-platform/device/version
  bitwise equality and exact mid-episode resume are not promised.
- Archive shortlisting is based on current selection fitness; only shortlisted
  candidates and old snapshots receive the fixed archive comparison. This is
  not exhaustive reevaluation of every historical genotype. Fixed validation
  can still be overfit; final tests must remain untouched.
- Time and interaction caps finish the current generation; learning also rounds
  to algorithm boundaries. Shared-world learning budgets count focal steps,
  while actual interaction totals include partner transitions and separately
  report world steps.
- Adaptive mutation, novelty search, islands, multiple objectives, architecture
  evolution, automatic curricula, rich experiment dashboards, and cloud
  infrastructure are deferred. Existing mutation/evaluator/selection hooks are
  extension points, not claims those algorithms already exist.
