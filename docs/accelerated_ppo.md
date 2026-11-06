# Accelerated population PPO

Enable the experimental path with `NativePPO(accelerated=True, device="cuda")`.
It uses the existing PPO loss, GAE, per-agent gradient clipping, independent
Adam, growth rules, inheritance, and generation-boundary checkpoints. The
default `accelerated=False` path remains the reference for comparison.

GPU measurements and correctness checks were completed on 2026-11-05 using an
RTX 4060 Laptop GPU. The [measured report](accelerated_ppo_benchmarks_2026-11-05.md)
contains the 96-configuration CPU/CUDA matrix, targeted ablations, growth,
chunking, a four-thread CPU check, and profiler analysis. CUDA helps larger
networks; small networks often favor CPU. Results apply to the documented
TensorReach1D workload and hardware.

```python
learner = NativePPO(
    accelerated=True, device="cuda", env_copies=4,
    rollout_steps=32, epochs=2, minibatch_size=32,
    population_chunk_size=256,  # None means the entire compatible group
)
```

Pass this learner to `Evolution` with `backend="batched"`, `device="cuda"`,
`environment_device="cuda"`, `tensor_env_factory=TensorReach1D`, and matching
`config.env_copies`. `GymTensorBridge` also works, with CPU simulation and GPU
policies. Custom tensor environments must implement `set_member_seeds(seeds)`
for this path. An optional `prepare_rollout(steps)` can prefill reset randomness.

## Changes and invariants

- Sampling reuses the actor output to calculate the action likelihood. The
  critic evaluates trajectory states together after collection. Adjacent states
  share bootstrap values, while truncations use their final pre-reset
  observations. True terminals cut bootstrapping and both types cut GAE.
- Rollout tensors allocate once per training group and are overwritten in place
  on later rollouts. PPO minibatches view the flattened storage before gathering.
- Action noise and tensor reset noise fill once per agent per rollout, with
  independent PyTorch generators. Sampling uses slices of these draws. Shuffle
  randomness has a separate seed domain. The small per-agent generator loop
  remains; no custom random-number kernel or global generator is introduced.
- The runner passes stable numeric individual IDs to training. Direct calls to
  `learn_population(..., member_ids=[...])` can do the same. Default IDs are list
  positions. Streams remain reproducible under chunking or regrouping when IDs,
  generation, device, environment copies, and rollout/update schedules match.
  Different schedules and CPU versus CUDA need not produce identical streams.
- Every loss and normalization leads with the agent dimension. Losses are
  summed for backward; gradients, gradient norms, Adam moments, bias correction,
  and step counters remain independent. Adam bias corrections are computed once
  per optimizer update. Device-to-host moment exports transfer stacked tensors
  once before constructing individual CPU states.
- Generation initialization leaves policy construction to the training chunks,
  avoiding an extra GPU policy that was built solely for validation and discarded.
- Growing networks batch by compatible observation/action spaces and hidden
  layer count. Each layer reserves its configured maximum width. Boolean masks
  zero inactive activations, so inactive parameters receive zero gradients.
  `growth_capacity=(128, 128)` can specify larger per-layer bounds; every bound
  must cover the corresponding configured growth maximum and initial widths.
- At each existing growth boundary, `NeuralGrowth.after_update` still chooses
  the layer and initializes new slots using the individual's growth RNG. The
  device policy imports only new incoming connections and zero outgoing rows;
  trained overlapping weights and parameter storage remain intact. Policy and
  value outputs are preserved at activation. The entire Adam state of each
  changed agent resets, as on the reference path; peers retain their state.
- Environments and observations continue across rollouts and growth boundaries.
  Training calls start fresh environments at each generation, matching the
  reference behavior. Fresh children reset training state; elites retain it.
  Baldwinian runs inherit acquired sizes while retaining birth weights.
- `population_chunk_size` bounds both training and frozen evaluation. Every
  agent receives its full rounded budget and all fitness results finish before
  selection. Chunking never subsamples agents, shortens rollouts, or reduces
  PPO epochs. It is explicit rather than an automatic OOM retry.
- Weights and Adam remain on device across all rollouts in a training call,
  including growth. Exports crop padded state to active shapes. Checkpoints
  contain CPU arrays/tensors using the existing schema; no CUDA handles or
  learner-owned population caches are needed to resume.

## Commands for the GPU machine

Run from the repository root in the environment with the project's torch extra.
The default matrix compares both paths on CPU and CUDA, using the same initial
genes, simulator, budgets, PPO schedules, and frozen evaluation scenarios:

```powershell
python benchmarks/ppo_updates.py --devices cpu cuda --populations 16 64 256 1024 --widths 16 64 128 --copies 1 4 --warmups 1 --repeats 5 --output results/benchmarks/ppo_matrix.json
```

Profile the representative large case separately, then measure the contribution
of each rollout optimization by disabling it from the accelerated path:

```powershell
python benchmarks/ppo_updates.py --devices cuda --populations 1024 --widths 64 --copies 4 --warmups 1 --repeats 5 --profile --output results/benchmarks/ppo_large_profile.json
python benchmarks/ppo_updates.py --devices cpu cuda --populations 256 1024 --widths 64 --copies 4 --variants native accelerated no-reuse no-buffers no-batch-rng --warmups 1 --repeats 5 --output results/benchmarks/ppo_ablations.json
```

If the whole population does not fit, set a chunk limit. The native baseline
continues using its original grouping, and OOM configurations are recorded:

```powershell
python benchmarks/ppo_updates.py --devices cpu cuda --populations 1024 --widths 128 --copies 4 --chunk-size 256 --warmups 1 --repeats 5 --output results/benchmarks/ppo_chunks.json
```

Measure growth with a common capacity and with continuous actions:

```powershell
python benchmarks/ppo_updates.py --devices cpu cuda --populations 16 64 256 1024 --widths 32 --copies 4 --growth-max 64 --growth-every 128 --growth-step 4 --warmups 1 --repeats 5 --output results/benchmarks/ppo_growth.json
python benchmarks/ppo_updates.py --devices cpu cuda --populations 256 1024 --widths 64 --copies 4 --continuous --warmups 1 --repeats 5 --output results/benchmarks/ppo_continuous.json
```

GPU correctness and existing regression checks:

```powershell
python -m pytest -q tests/test_accelerated_ppo.py tests/test_tensor.py tests/test_growth.py tests/test_cuda.py
python examples/native_ppo.py --device cuda --accelerated --chunk-size 256
python examples/neural_growth.py --device cuda --accelerated --inheritance baldwinian
```

`AcceleratedCUDA` repeats the CPU invariants on the GPU: independent gradients
and Adam against separate networks, inactive masks, output-preserving growth,
truncation bootstrapping, chunks and stable IDs, tensor/hybrid environments, and
exact generation-checkpoint resume in both inheritance modes. CUDA tests skip
when hardware is missing.

## Reading the measurements

JSON retains every raw sample and the median of each metric. Times are seconds:

| Field | Work included |
| --- | --- |
| `rollout_seconds` | Noise, simulation, action sampling, storage, old critic values |
| `update_seconds` | GAE, advantage normalization, shuffles, PPO forward/backward, clipping, Adam |
| `setup_seconds`, `growth_seconds`, `export_seconds` | Training initialization, growth handling, portable state export |
| `learning_seconds` | Complete training wall time across all chunks, per generation |
| `full_generation_seconds` | Complete run wall time divided by generations, including initialization, breeding, frozen validation, CPU archive scoring, final result copies |
| `transitions_per_second` | Actual training transitions divided by learning wall time |
| `generation_training_transitions_per_second` | Training transitions divided by complete run wall time |
| `total_interactions_per_second` | Training plus evaluation interactions divided by complete run wall time |
| `cuda_peak_allocated_bytes` | Peak live CUDA allocation during the complete run |
| `cuda_peak_reserved_bytes` | Allocator reservation peak, potentially including previously cached blocks |

Imports and complete warmup runs are excluded. Each timed phase synchronizes CUDA
at its boundaries, as does the complete run. This makes phase wall times
readable but adds synchronization overhead; production runs leave
`measure_phases=False`. Profiles are separate runs with phase synchronization
disabled and their own warmup. Shape/memory recording is optional with
`--profile-detailed`; it can add substantial post-processing cost for large
populations with many small operations. Peak CUDA allocation is measured in the
timing runs regardless of this flag. No checkpoint or
result-file I/O is included in generation timing. The summary file is saved
after each configuration, including OOM/skipped statuses.

`--profile` writes Chrome traces directly. Add `--profile-table` to also build
PyTorch's operator table; aggregating a large trace can take minutes and several
GiB of RAM. Raw traces can be analyzed separately without holding the complete
event list in memory.

The script verifies equal actual training interactions, evaluation interactions,
and optimizer updates across successful compared routes. `comparisons` reports
training and full-generation speedups versus native on the same device, native
CPU, and the same variant on CPU. `observed_crossovers` gives the first sampled
population where CUDA beats CPU for training and for complete generations,
separately. A null entry means no observed win in the measured range. `ablations`
reports ratios for enabling reuse, buffers, and batched action randomness. These
are disable-one comparisons, not additive percentages; stochastic training
streams can differ, particularly for batched normal draws. Fitness is recorded
as context, not evidence of learning superiority.

## Profiling results and remaining work

Chrome traces and optional operator tables are written under
`results/benchmarks/ppo_profiles`. Ranges distinguish `ppo.setup`, `ppo.rollout`,
`ppo.update`, `ppo.growth`, `ppo.export`, and generation initialization, reproduction, evaluation,
and archive work. Inspect device time, CPU submission gaps, copy operations, and
peak allocation together with full generation speedups.

The large-run traces show rand events decreasing from 65,536 to 6,144 and bmm
events from 1,600 to 652. Per-agent shuffles remain at 8,192 randperm events;
population/state construction still produces about 49,500 clone events.
The native trace records shapes/memory and the accelerated trace uses lighter
instrumentation, so profiler durations are not compared. Synchronized timing
runs establish the speedups; operation counts identify remaining work.

At 1,024 agents, width 128, and four copies, accelerated CUDA spends about 53.5%
of complete-run time outside learning and another 24.9% in setup/export.
Policy packing for evaluation, reproduction, and portable CPU state construction
now limit full-generation gains. Archive scoring is small in these traces.
Batched action randomness has the clearest rollout gain; preallocation has no
demonstrated complete-generation gain in the measured ablation repeats.

One bounded capacity per layer is implemented. Capacity buckets should follow
only if profiles show padding is expensive. GPU state residency across
generations should follow only if setup/export/reproduction transfers contribute
meaningfully to generation time; it would require explicit handling of elites,
fresh children, mutation, archives, and portable checkpoints. Compilation and
CUDA Graphs are deferred until a batched GPU profile identifies launch overhead
as a limiting cost. This avoids claiming gains from unmeasured changes.
