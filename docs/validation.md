# Correctness and performance validation

Tests are standard-library `unittest` cases, also collected by pytest. Core tests
run without extras. Optional integration cases skip when their dependency is
absent; CUDA cases run only when hardware is present. Process evaluation requires
permission to create Windows worker pipes, which this execution sandbox blocks
by default; the recorded full suite was also run outside that sandbox.

The tests cover selection, rank/tournament direction, elite preservation, random
immigrants, parameter mutation, population/generation counts, parent-child and
archive independence, frozen fitness, distribution separation, terminal versus
truncated episodes, caps, scenario immutability, champion serialization,
generation-boundary resume, and incompatible configuration/provenance rejection.
An isolated interpreter actively blocks PyTorch, SB3, PettingZoo, and Matplotlib
imports while running core evolution.

Lifetime-learning tests distinguish Lamarckian from Baldwinian inheritance,
ensure adaptation tasks start fresh, check SB3 PPO's actual rounded budget,
validate DQN replay/optimizer/target reset rules, and compare continuing/restored
SB3 and native PPO state. Continuous and discrete algorithms are exercised.
Normalization and unsupported space/backend combinations raise errors.

Tensor tests compare serial NumPy and stacked PyTorch outputs; CPU and tensor
environment transitions; termination/truncation advantage targets; independent
random streams; and gradients that cannot cross individuals. PPO losses,
gradients, per-individual clipping, and three Adam updates are checked against
independent `nn.Sequential`, `torch.distributions.Categorical`, and
`torch.optim.Adam` implementations. Continuous likelihood is checked against
PyTorch's transformed distribution. Resume includes older elites alongside fresh
children, with independent optimizer counters.

The hardware-only `tests/test_cuda.py` additionally compares CUDA inference,
likelihoods, PPO gradients, clipping, and Adam against CPU results; checks masked
tensor transitions, bootstrap targets, independent CUDA generators and gradients;
and checks frozen native/hybrid evaluation with heterogeneous widths. It runs
checkpoint/resume across discrete/continuous actions, native/hybrid environments,
fixed/growing networks, and both inheritance modes. SB3 CUDA PPO/DQN tests cover
training, caller RNG isolation, fresh children, resumed state, and CPU-portable
agent saves. These cases skip cleanly without CUDA; see [hardware results](cuda_validation.md).

Shared-world tests cover routing, sender identity, one-step message timing,
reset clearing, communication cost, recipient masks, inactive agents, task/team
reward selection, role/partner rotation, held-out fixed partner pools, evolved
messages, final observations for truncation, and fixed-message PPO against frozen
partners. PettingZoo's own `parallel_api_test` validates the bundled real
`ParallelEnv` subclass.

## Reproducing measurements

```powershell
python -m pytest -q
python benchmarks/equal_work.py --repeats 3
python benchmarks/ppo_updates.py
python benchmarks/ppo_updates.py --device cuda
python benchmarks/baselines.py --seeds 1 2 3
```

`equal_work.py` uses populations 4 and 16, environment copies 1 and 4, four frozen
selection episodes per individual, two evaluated generations, a fixed horizon of
16, hidden widths (16,16), and one archive entry. It reports actual interactions
including archive comparisons. Every mode in a matched group performs the same
number of transitions. Modes include NumPy serial CPU, two process workers,
batched CPU policies with Python environments, and native tensor CPU simulation.
When CUDA is present, hybrid and native CUDA modes are added with explicit
synchronization and peak device allocation. Startup/warm-up is separate; no
compilation or mixed precision is used.

Complete-run timings include initialization, reproduction, frozen evaluation,
archive evaluation, and result copies. Python imports and warm-up are excluded.
Per-component timings are retained. Process workers are started each generation;
this intentionally exposes startup costs in these small runs. Parent RSS is
sampled after the run, not as a peak and not including child RSS. Host peak memory,
transfer-only timing, compilation, and profiler traces remain unverified.
CUDA complete-run timing and peak allocated device memory were measured on
2026-10-30; see [CUDA validation](cuda_validation.md). Missing measurements are
explicitly labeled rather than inferred.

`ppo_updates.py` compares native and opt-in accelerated PPO on CPU and CUDA at
populations 16, 64, 256, and 1,024, with configurable widths, copies, and chunks.
It warms every configuration and reports repeat medians for rollout, PPO update,
full generation time, throughput, and peak allocated CUDA memory. Raw samples,
actual interaction/update counts, ablations, and optional profiler traces are
retained. Training RNG trajectories differ between routes; this is an equal-work
benchmark, not an algorithm superiority study. GPU validation on 2026-11-05
passed 50 tests and 21 subtests with no skips. See the
[measured results](accelerated_ppo_benchmarks_2026-11-05.md) and
[commands and methodology](accelerated_ppo.md).

`baselines.py` uses three independent seeds and reports frozen random behavior,
evolution only, RL only, evolution plus RL, and fixed-policy communication
ablations. RL-only and evolution-plus-RL spend the same lifetime-learning budget;
evaluation budgets differ and are disclosed. The communication control uses a
different task, so its scores must not be compared with Reach1D scores. Normal
95% intervals are descriptive small-sample summaries, not robust confidence
claims. Training-seed variability is separate from episode count.

Raw JSON lives under ignored `results/benchmarks/`. The compact measured snapshot
and environment details are retained in [report.md](report.md).
