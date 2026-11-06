# geneticevo

**Alpha preview.** APIs and checkpoint compatibility may change.

A Python library for reproducible evolutionary experiments, with optional
lifetime learning and population tensor execution. Supply a problem and an agent
factory; the runner handles selection, mutation, elites, lineage, evaluation,
metrics, archives, and checkpoints.

Pure evolution requires only NumPy and Gymnasium. PyTorch, Stable-Baselines3
(SB3), PettingZoo, and plotting are optional. The package and Python import name
are `geneticevo`. Install published releases from [PyPI](https://pypi.org/project/geneticevo/).

[Documentation](https://github.com/JonathanBaj/geneticevo/blob/main/README.md) |
[Source code](https://github.com/JonathanBaj/geneticevo) |
[Issue tracker](https://github.com/JonathanBaj/geneticevo/issues) |
[Releases](https://github.com/JonathanBaj/geneticevo/releases)

Evolution, learning, communication, and GPU execution do not guarantee better
results or general intelligence. These are research tools, not deployment safety
guarantees.

## Installation

Install from PyPI:

```sh
pip install geneticevo
```

To upgrade to the latest alpha release:

```sh
python -m pip install --upgrade --pre geneticevo
```

Choose optional features as needed:

```sh
python -m pip install "geneticevo[sb3]"    # PPO and DQN; includes PyTorch
python -m pip install "geneticevo[torch]"  # Native population tensors and PPO
python -m pip install "geneticevo[shared]" # Official PettingZoo integration
python -m pip install "geneticevo[plot]"   # Saved plots
```

Python 3.10+ is declared; Python 3.11.9 was tested on Windows. A virtual
environment is recommended. For source development, clone the repository:

```sh
git clone https://github.com/JonathanBaj/geneticevo.git
cd geneticevo
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install development tools:

```sh
python -m pip install -e ".[dev,release]"
```

Core evolution does not install PyTorch, SB3, PettingZoo, or Matplotlib. Optional
extras have their own upstream Python/platform requirements; for example, the
tested PettingZoo 1.25.0 declares Python <3.13. Only Python 3.11.9 on Windows has
been exercised here; the minimum declared Python version is not a tested matrix.

## CPU and GPU requirements

- Ordinary CPU execution needs Python, NumPy, and Gymnasium. CUDA and a GPU are
  unnecessary for vector optimization and NumPy neural evolution.
- Native tensor inference/PPO requires the `torch` extra and works on CPU.
  SB3 learning requires the `sb3` extra and also works on CPU.
- The CUDA backend supports batched MLP inference, native PPO, and compatible
  tensor environments. It requires an NVIDIA GPU, a compatible driver, and a
  CUDA-enabled PyTorch build. Follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
  before requesting CUDA; a CPU-only PyTorch build cannot use the GPU.
- Python environments keep running on the CPU. GPU policies with Python
  environments are hybrid execution; native tensor simulation requires a
  compatible tensor environment. There is no general GPU speedup claim.
- An explicitly requested unavailable device raises an actionable error.
  Automatic CPU fallback is not used.

Declared ranges and tested versions:

| Dependency | Declared range | Tested |
|---|---|---|
| NumPy | >=1.26,<3 | 2.4.4 |
| Gymnasium | >=1.2,<1.3 | 1.2.3 |
| PyTorch (optional) | >=2.12,<2.13 | 2.12.1+cu126, CPU and CUDA execution |
| SB3 (optional) | >=2.9,<2.10 | 2.9.0 |
| PettingZoo (optional) | >=1.24,<1.26 | 1.25.0 |
| Matplotlib (optional) | >=3.10,<3.11 | 3.10.6 |

Only the listed versions were exercised; the entire allowed ranges were not
tested. CPU and CUDA execution were validated, including native/hybrid inference,
native PPO, optional SB3 PPO/DQN, neural growth, and checkpoint resume. The CUDA
checks ran on an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch 2.12.1+cu126
on 2026-10-30. CUDA tests skip when no usable GPU is available. Measured results,
hardware details, and reproduction commands are in [CUDA validation](docs/cuda_validation.md).
See also [validation](https://github.com/JonathanBaj/geneticevo/blob/main/docs/validation.md).

## Complete quickstart

This environment is included and uses Gymnasium's reset/step conventions:

```python
from geneticevo import Evolution, EvolutionConfig, GaussianMutation
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D

experiment = Evolution(
    env_factory=Reach1D,
    agent_factory=MLPAgentFactory(hidden_sizes=(16, 16)),
    config=EvolutionConfig(
        population_size=12,
        generations=5,
        elite_count=2,
        selection="tournament",
        evaluation_episodes=3,
        seed=42,
        output_dir="results/quickstart",
    ),
    mutation=GaussianMutation(parameter_probability=0.1, std=0.1),
    learner=None,
    backend="serial",
    device="cpu",
)
result = experiment.run()
result.best_agent.save("results/quickstart/champion")
print(result.best_fitness)
```

The NumPy MLP has separate actor and critic networks. Its inherited vector
contains all actor weights/biases, critic weights/biases, and, for continuous
actions, log standard deviations. Critic genes matter when lifetime learning is
enabled. Pure evolution does not use the critic for action selection.

## Runnable examples

Run these from the repository after installation:

```powershell
python examples/vector.py
python examples/neural.py
python examples/custom_genome.py
python examples/custom_learner.py
python examples/generalization.py
python examples/sb3_learning.py --algorithm PPO
python examples/sb3_learning.py --algorithm DQN --inheritance baldwinian
python examples/native_ppo.py --device cpu
python examples/native_ppo.py --continuous
python examples/neural_growth.py --device cpu
python examples/neural_growth.py --inheritance baldwinian
python examples/shared_world.py
python examples/shared_world.py --rl
```

Run native PPO on a CUDA device from the source checkout:

```sh
python -c "import torch; assert torch.cuda.is_available(), 'No usable CUDA device found'"
python examples/native_ppo.py --device cuda
```

The example sets both policy and tensor-environment devices to CUDA. An ordinary
Python environment remains on the CPU when paired with a CUDA policy.

## Configuration and lifecycle

`generations=5` evaluates populations **0 through 4**. There is no unused final
offspring population. A generation initializes/restores, optionally measures
pre-learning performance, learns, evaluates frozen policies, updates the
archive, records statistics, and checkpoints. Reproduction happens only if
another generation will be evaluated.

These settings have different meanings:

| Setting | Meaning |
|---|---|
| `population_size` | Number of independently inherited individuals |
| `generations` | Number of evaluated populations |
| `env_copies` | Environment copies **per individual**, used by learners and tensor evaluation |
| `learning_budget` | Requested training transitions **per individual per generation** |
| `evaluation_episodes` | Frozen selection episodes per individual; also sizes the archive reference suite |
| Shared world's `possible_agents` | Roles acting together in one world; independent of population size |

Serial evaluation visits episodes sequentially. CPU parallelism evaluates
individuals in spawned processes. It requires pickleable top-level factories
and an `if __name__ == "__main__":` entry point. Shared population evaluation is
currently serial. SB3 learning runs individuals sequentially, even when fitness
evaluation uses CPU workers.

Mutation probability applies **per scalar parameter**. `std` controls its
magnitude. Tournament and rank selection honor `maximize`; custom selection,
mutation, and optional crossover are supported. Crossover is disabled by default.
`immigrant_count` injects fresh agents. `diversity_pairs` enables sampled parameter
distances without an all-pairs calculation.

Target fitness and stagnation use fixed archive-reference fitness. Time and
interaction caps stop at **complete generation boundaries**, so they can
overshoot by a generation. The first population is always evaluated. Counts
include archive and optional pre-learning evaluation. Custom vector objectives
consume zero environment interactions.

## Learning and inheritance

Use `learner=None` and `learning_budget=0` for evolution only.

- **Lamarckian:** learned parameters replace inherited genes before reproduction.
- **Baldwinian:** train a fresh copy; fitness and the deployed archive snapshot
  use that copy, while reproduction uses the original genes.
- Unchanged elites continue optimizer/counter/replay state by default and learn
  again (`elite_learning=True`). Set it false to skip further elite learning with
  serial SB3/custom learners. Native batching requires it true.
- Mutated children and immigrants start fresh optimizer, replay/rollout,
  exploration, counter, and random-action state. DQN targets are synchronized to
  the child's mutated online network. Unchanged elites preserve DQN targets.
- Environments reset at generation boundaries. Mid-episode state is not resumed.

SB3's factory builds the algorithm's own compatible `MlpPolicy`; arbitrary NumPy
or PyTorch modules cannot attach to it unchanged. PPO evolves its full online
policy, including value and distribution parameters. DQN evolves its online Q
network; target networks are algorithm state. Normalization wrappers are rejected
until a frozen-statistics serialization adapter exists.

PPO rounds a budget up to `rollout_steps * env_copies` (SB3 uses `n_steps`).
Updates may occur within unfinished episodes. DQN's default step frequency is
one; custom frequencies may overshoot. Requested and actual counts are distinct.
With SB3, set the factory's and learner's `env_copies` to match the config.

Native PPO batches independent policies, losses, gradients, clipping, and Adam.
Adam has separate moments **and step counters** for each individual, so a fresh
child can learn alongside an older elite. Advantages normalize per individual
over a rollout. Discrete categorical and continuous tanh-squashed Gaussian
likelihoods are implemented. Continuous entropy regularization uses the base
normal's entropy; its default coefficient is zero.

For large populations, opt in to the accelerated native path:

```python
learner = NativePPO(
    device="cuda", accelerated=True,
    env_copies=4, rollout_steps=32, epochs=2, minibatch_size=32,
    population_chunk_size=256,  # Optional limit on agents resident at once
)
```

Use `backend="batched"`, matching policy/environment devices, and
`tensor_env_factory=TensorReach1D` to keep simulation on the GPU. This path
reuses rollout storage and actor outputs, batches critic evaluation and random
draws over a rollout, and trains growing networks together using fixed capacity
and per-agent masks. It also runs on CPU. Weights and Adam state remain on the
device throughout learning; portable CPU state is exported at generation
boundaries. All training and evaluation chunks finish before selection.

The path is experimental and remains opt-in. On the tested RTX 4060 Laptop GPU,
the 1,024-agent, 128-by-128, four-copy case ran 1.69 times faster per complete
generation than native CUDA and 1.48 times faster than accelerated four-thread
CPU. Small networks often favor CPU. [Measured results](docs/accelerated_ppo_benchmarks_2026-11-05.md)
include the full matrix, ablations, memory, growth, and profiling.
[Benchmark commands and limitations](docs/accelerated_ppo.md) explain reproduction.

## Optional lifetime network growth

Enable growth on the built-in MLP factory; `growth=None` (the default) retains
fixed-size networks and the existing behavior:

```python
from geneticevo import NeuralGrowth
from geneticevo.agents import MLPAgentFactory

factory = MLPAgentFactory(
    hidden_sizes=(4, 4),
    growth=NeuralGrowth(
        check_every=32,             # Actual training transitions per individual
        neurons_per_growth=2,       # Added to one eligible hidden layer per event
        max_hidden_size=32,         # Hard cap per hidden layer
        min_hidden_size=2,          # Lower bound for offspring size mutations
        value_loss_threshold=None, # Scheduled growth; use a number to gate on value loss
        mutation_probability=0.1,   # Probability per layer at reproduction
        mutation_step=2,            # Add or remove 1..2 neurons on a mutation
    ),
)
```

Pass this factory to an `Evolution` with `NativePPO`, `backend="batched"`, and a
positive `learning_budget`. Install the `torch` extra for PPO. A complete runnable
example is [examples/neural_growth.py](examples/neural_growth.py):

```sh
python examples/neural_growth.py --device cpu
python examples/neural_growth.py --continuous --inheritance baldwinian
python examples/neural_growth.py --value-loss-threshold 0.5
```

During training, growth is checked after completed PPO rollouts. An eligible
hidden layer is chosen randomly and widened in both actor and critic networks.
Checks round up to a rollout boundary and perform at most one growth event at a
time. Existing parameters are retained; new neurons receive random incoming
weights and zero outgoing weights into existing neurons, preserving the current
policy and value outputs at the moment of growth. Later updates can learn to
use them. The environment continues without resetting when the network grows.
Adam moments reset after a size change; growth counters continue.

By default, growth is scheduled. Set `value_loss_threshold` to require the mean
value loss across that rollout's PPO updates to meet the threshold before
adding capacity. This is a learning-error heuristic, not a measurement of
novelty, intelligence, or a biological model. Larger networks need more memory
and may perform worse; choose a suitable width cap.

The agent's final **hidden-layer sizes** become inherited traits in both modes:
Lamarckian offspring inherit learned weights and sizes; Baldwinian offspring
inherit sizes while retaining the original genes in overlapping neurons and
initializing additional capacity afresh. Each layer may then mutate up or down
within its bounds, followed by normal parameter mutation. Lifetime growth only
adds neurons; shrinking happens through offspring mutations. Layer count,
observation size, and action size stay fixed. The growth settings themselves
are configuration, not mutated traits. Elites keep their sizes; immigrants use
the factory's starting sizes. Fresh children reset growth and optimizer counters.

Growth-enabled agents learn individually, with their environment copies still
batched on the selected CPU/CUDA device. Frozen evaluation groups agents by
architecture. Fixed-size agents retain population-batched learning. This avoids
padding every policy to the maximum size, but may reduce population throughput.
Growth and size mutations, saving/loading, archives, and generation-boundary
resume are supported. Native and hybrid CUDA execution, growth, and resume were
checked on the hardware listed in [CUDA validation](docs/cuda_validation.md).

Automatic lifetime growth currently supports the built-in MLP with native PPO.
Custom MLP learners can opt in with `supports_neural_growth=True` and call
`agent.growth.after_update(agent, interactions, value_loss, rng)` at safe update
boundaries. SB3 policies and communicating task-policy wrappers do not support
this feature. `crossover` must remain `None`. With `learner=None`, offspring size
mutations still work, but networks do not grow during frozen evaluation.

Metrics record `network_hidden_sizes`, `network_parameter_counts`, and learner
`growth_events` (mean per trained individual per generation). Lineage records
architecture sizes before/after each offspring mutation. Agent folders retain
sizes and growth configuration; training counters require a full checkpoint.
For different widths, sampled parameter distances align parameters by layer and
neuron index, treating missing neurons as zeros.

## Supported combinations

| Agents / learner | Serial CPU | CPU process evaluation | Batched policies, Python env | Tensor environment |
|---|---|---|---|---|
| Vector objective / none | Yes | Yes | No | No |
| NumPy MLP / none | Yes | Yes | Yes | Yes |
| SB3 PPO or DQN / SB3Learner | Yes | Yes; learning stays sequential | No | No |
| MLP / NativePPO | No | No | Yes | Yes |
| Evolved message head / none | Shared serial | No | No | No |
| SB3 task policy / fixed messages | Shared serial | No | No | No |

Neural observations must be flat `Box`; actions must be zero-start `Discrete` or
flat, finite, strictly bounded `Box`. DQN supports `Discrete` only. Shared worlds
use the PettingZoo Parallel protocol with flat `Box` observations. Other spaces
need explicit tested adapters.

`backend="batched"` groups compatible MLP architectures for evaluation. Fixed-size
learning uses population batches; growth-enabled policies learn individually by
default and use masked population batches with `NativePPO(accelerated=True)`.
Tensor computation also runs on CPU. Omitting `tensor_env_factory` uses `GymTensorBridge`: Python simulation
stays on CPU even if policies run on CUDA. This is labeled hybrid execution.
Setting a device never migrates arbitrary Python simulation code. Native tensor
environments require matching policy/environment devices. The included
`TensorReach1D` has a matching CPU reference. Archives always use the serial CPU
reference evaluator, including for tensor runs.

## Shared worlds and communication

Communication is disabled by default. `MessageChannel` uses bounded fixed-size
messages and broadcasts to other active agents. Each sender owns an identified
inbox slot plus a presence bit. Messages sent with a task action become visible
in the **next** observation. Reset clears inboxes. Optional L1 message cost is
subtracted from sender reward. Recipient and transform hooks allow custom
routing/noise/dropout; queued delays need a custom channel implementation.

Policies receive only their permitted local observation and prior inbox. Pure
evolution can change both task and message heads. `SharedEvaluator` rotates focal
roles and current-population partners for selection; archive comparisons use a
fixed partner pool. Use another evaluator with unfamiliar partners for final
testing. `reward_mode` chooses focal-individual or mean-team reward. World steps
and all active-agent transitions are counted separately; inactive roles do not
act. Explicit focal-to-role mappings are retained in evaluation details.

The tested multi-agent learning integration is PPO/DQN on `SharedFocalEnv` against
frozen partners with **fixed message rules**. The learner trains task actions;
messages have no trainable distribution. End-to-end learned stochastic messages
and simultaneous RL of distinct communicating policies are unsupported.
Experience sharing, parameter averaging, imitation, and global memory are not
part of communication.

## Results, checkpoints, and generalization

`output_dir` writes `metadata.json`, incremental `metrics.jsonl`, `lineage.json`,
and atomic `latest.checkpoint`. Metrics include fitness quantiles, task metrics,
episode lengths, training/evaluation costs, learner reports, population and
mutation settings, component timings, throughput, and cumulative wall time.
Final-info success/failure fields are retained when environments provide them.
Trajectories and every network are not saved by default.

The archive shortlists the best `archive_size` current candidates, then compares
them with old archive snapshots on common, fixed reference scenarios. It does
not compare incomparable rotating selection scores across generations.
Snapshots are independent copies with their evaluation context.

Resume at an evaluated generation boundary:

```python
# Re-create experiment with the same components/configuration; generations may increase.
result = experiment.run(resume_from="results/quickstart/latest.checkpoint")
```

Checkpoint schema 1 stores population/archive genes, resumable learning state,
RNG state, IDs, lineage, and counters. Scalar settings and component descriptions
are checked on restore. Custom components should expose configuration as public
fields on stable classes; arbitrary hidden global or closure state cannot be
verified. Load only trusted checkpoints: they use Python pickle. Agent folders
save inference genomes; continuing optimizer/replay state requires a checkpoint.
Bitwise reproducibility across devices, versions, and platforms is not promised.

Training uses a separate `training_env_factory` (defaults to `env_factory`).
Selection scenarios rotate by generation, with a common suite for each
population. Archive scenarios stay fixed. `validation_sampler` and
`archive_sampler` accept `(rng, count, generation)` and return `Scenario` objects.
`Scenario.options` reaches Gymnasium reset, enabling procedural distributions.

Final testing is explicit and never drives the runner:

```python
from geneticevo.core import scenarios_for
from geneticevo.evaluation import GymEvaluator
test = result.evaluate(GymEvaluator(Reach1D), scenarios_for(2026, "test", 20))
```

Use separate test scenarios and partners, including shifted distributions.
`adaptation_evaluation` gives every task a fresh training-state copy and records
pre/post performance and costs. It does not modify the champion. Repeated
training seeds and `uncertainty_summary` describe uncertainty across runs;
additional episodes do not replace independent training seeds. No mechanism
prevents memorization or invalid experimental design.

## Development and evidence

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
python -m build --wheel --no-isolation
python -m twine check --strict dist/*
python benchmarks/equal_work.py --repeats 3
python benchmarks/baselines.py --seeds 1 2 3
```

Windows sandbox restrictions may prevent spawned worker pipes; use an ordinary
local terminal for process tests. [Architecture and extensions](https://github.com/JonathanBaj/geneticevo/blob/main/docs/architecture.md)
explain custom genomes/environments/learners. [Validation](https://github.com/JonathanBaj/geneticevo/blob/main/docs/validation.md)
documents correctness checks and benchmark methodology. [Delivery report](https://github.com/JonathanBaj/geneticevo/blob/main/docs/report.md)
records measured results and remaining limitations.
The [alpha release checklist](https://github.com/JonathanBaj/geneticevo/blob/main/docs/release.md) records packaging checks and the
current release checks and hardware validation status.

For a saved plot: `from geneticevo.plotting import plot_history`, then call
`plot_history("results/neural/metrics.jsonl", "results/neural/fitness.png")`.

## License

geneticevo is licensed under the [MIT License](https://github.com/JonathanBaj/geneticevo/blob/main/LICENSE). Copyright notices and
the license text ship with the source distribution and wheel. Dependencies
retain their own licenses.
