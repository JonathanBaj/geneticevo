# geneticevo

**Alpha preview.** APIs and checkpoint compatibility may change.

A Python library for reproducible evolutionary experiments, with optional
lifetime learning and population tensor execution. Supply a problem and an agent
factory; the runner handles selection, mutation, elites, lineage, evaluation,
metrics, archives, and checkpoints.

Pure evolution requires only NumPy and Gymnasium. Rust inference, PyTorch,
PettingZoo, and plotting are optional. The package and Python import name
are `geneticevo`. Install published releases from [PyPI](https://pypi.org/project/geneticevo/).

[Documentation](https://github.com/JonathanBaj/geneticevo/blob/main/README.md) |
[Source code](https://github.com/JonathanBaj/geneticevo) |
[Issue tracker](https://github.com/JonathanBaj/geneticevo/issues) |
[Releases](https://github.com/JonathanBaj/geneticevo/releases)

Evolution, learning, communication, and GPU execution do not guarantee better
results or general intelligence. These are research tools, not deployment safety
guarantees.

## What's new in 0.1.0a9

Evolution uses selection and mutation to optimize policies. The built-in PPO and
DQN learners and the Stable-Baselines3 integration were removed. Rust and PyTorch
can accelerate frozen policy evaluation. Width, depth, and graph topology remain
heritable, and the custom learner interface accepts user-supplied algorithms.
See [release notes](docs/release.md) for migration details.

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
python -m pip install "geneticevo[torch]"  # CPU/CUDA population inference
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

Core evolution installs NumPy and Gymnasium. Optional
extras have their own upstream Python/platform requirements; for example, the
tested PettingZoo 1.25.0 declares Python <3.13. Only Python 3.11.9 on Windows has
been exercised here; the minimum declared Python version is not a tested matrix.

## CPU and GPU requirements

Serial NumPy and process-based evaluation run on CPU. The optional Rust backend
batches frozen policies on CPU; see [accelerated evaluation](#accelerated-evaluation)
for installation. The `torch` extra enables batched MLP and graph inference on CPU
or CUDA and supports compatible tensor environments.

CUDA needs a supported NVIDIA GPU, driver, and CUDA-enabled PyTorch build.
See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
With Gymnasium environments, policies can use CUDA while simulation runs on CPU.
A compatible tensor environment can keep simulation and inference on the same device.

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

```sh
python examples/vector.py
python examples/neural.py
python examples/custom_genome.py
python examples/custom_learner.py
python examples/generalization.py
python examples/neural_growth.py --dynamic-depth
python examples/graph_brain.py --continuous
python examples/shared_world.py
```

Use `--backend batched --device cuda` on the growth or graph example for CUDA
inference with Gymnasium simulation. Use `--backend rust` for native CPU inference.

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
currently serial.

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

## Custom learning and inheritance

The default run evolves policies with selection, mutation, and frozen evaluation.
To supply your own lifetime adaptation, pass a learner implementing the
`Learner` protocol and set a positive per-individual `learning_budget`. The
[custom learner example](examples/custom_learner.py) uses budgeted random search.

- **Lamarckian** inheritance passes the adapted genome to reproduction.
- **Baldwinian** inheritance evaluates an adapted copy and breeds from the
  original weights. Structural edits can still be inherited.
- Unchanged elites can continue their per-agent state. Set `elite_learning=False`
  to skip further adaptation of those elites. New offspring start with fresh
  learner state.

The learner reports actual interactions and updates. Requested learning budgets
and actual costs are recorded separately.

## Accelerated evaluation

The default backend is `"serial"`. Use `"parallel"` to evaluate independent
individuals in spawned processes, or use one of the batched neural backends:

- `backend="batched"` uses PyTorch. It groups agents by compatible spaces and
  architecture. With `tensor_env_factory=TensorReach1D`, simulation and inference
  can use the same CPU or CUDA device. Gymnasium environments run on CPU.
- `backend="rust"` uses the optional `geneticevo-core` wheel for CPU inference
  with Gymnasium. A Rust batch can include mixed widths, depths, and graph topologies.
  Set `RAYON_NUM_THREADS` before launch to control the worker count.

Archives evaluate with the serial CPU reference. A compatible core wheel
installs with `python -m pip install --pre "geneticevo[rust]==0.1.0a9"` once
the matching releases are on PyPI. For a source checkout with a Rust toolchain:

```sh
python -m pip install . ./rust
python examples/neural_growth.py --backend rust --dynamic-depth
```

## Optional network growth

```python
from geneticevo import NeuralGrowth
from geneticevo.agents import MLPAgentFactory

factory = MLPAgentFactory(
    hidden_sizes=(4, 4),
    growth=NeuralGrowth(
        min_hidden_size=2, max_hidden_size=32,
        mutation_probability=0.10, mutation_step=2,
        min_hidden_layers=0, max_hidden_layers=4,
        layer_mutation_probability=0.25, new_layer_size=4,
    ),
)
```

With `growth=NeuralGrowth(...)`, offspring can mutate hidden widths and add or
remove a hidden layer before scalar gene mutation. Stable layer IDs preserve
surviving genes as indices shift. Expanding a layer initializes new incoming
weights and zeros the new outgoing connections to existing neurons, preserving
policy and value outputs at first. Shrinking or changing depth can change
outputs. Sizes, IDs, lineage, and parameter counts are tracked in the saved data.

A custom learner can opt into lifetime edits with `supports_neural_growth=True`
and call `agent.growth.after_update(agent, interactions, value_loss, rng)` at an
update boundary. `check_every` counts actual interactions and an optional
`value_loss_threshold` uses the loss supplied by the learner. Keep
`crossover=None` for growth-enabled agents.

## Sparse graph brains

`GraphAgentFactory` adds a brain with arbitrary directed connectivity instead of
hidden layers. Connections can skip between any computational neurons, form
cycles or self connections, or leave neurons disconnected. Only existing edges
have weight genes; a neuron may have zero, one, or many incoming/outgoing edges.

```python
from geneticevo import GraphAgentFactory, GraphGrowth

factory = GraphAgentFactory(
    initial_neurons=8,       # Hidden neurons; action/value readouts are additional
    initial_connections=24,
    settling_steps=4,
    growth=GraphGrowth(
        check_every=32,     # Interaction interval for a custom learner
        min_neurons=0,
        max_neurons=64,
        max_connections=512,
        node_add_probability=0.5,
        node_remove_probability=0.25,
        connection_add_probability=1.0,
        connection_remove_probability=0.25,
        mutation_probability=0.25,
    ),
)
```

Use this factory with `Evolution` to evolve connectivity at reproduction:

```sh
python examples/graph_brain.py
python examples/graph_brain.py --continuous --backend rust
```

At reproduction, independent random proposals add or remove hidden neurons and
connections. Custom learners can also call the lifetime growth hook. `neurons_per_event` (default 1) and
`connections_per_event` (default 2) bound each proposal. Pruning removes incident
edges; additions initialize new edge weights randomly. Surviving genes retain
their neuron/edge identities. Structural edits can change behavior immediately,
reset that individual's Adam state, and leave its environment running. Neurons
may initially be isolated and gain connections in later updates. Growth checks
round up to completed rollouts; evaluation never edits a brain.

The final connectivity and neuron identities are inherited in both modes.
Lamarckian inheritance also retains learned weights. Baldwinian inheritance
keeps birth weights on surviving genes and initializes newly inherited genes
afresh. At reproduction, `mutation_probability` applies independently to each
of the four structural proposals before ordinary weight mutation. Fitness
selection favors successful offspring; random growth/pruning alone does not
guarantee improvement or find an optimal size. To favor smaller brains explicitly,
use a custom evaluator with a neuron/connection cost in its fitness. Observation
inputs and action/value readouts stay fixed; only hidden neurons can be removed.
`growth=None` fixes topology, while `learner=None` still allows birth mutations
with a growth configuration. Elites retain their brains; immigrants start fresh.

This model uses a shared graph for the actor and critic. Each observation starts
from zero neuron activations and runs `settling_steps` synchronous message passes:
inputs are supplied each pass, hidden neurons apply tanh, and readouts are linear.
Cycles therefore have bounded computation, and paths longer than the settling
budget cannot carry that observation to a readout. There is no memory across
environment steps. This models flexible connectivity, not biological development
or a connectome reconstruction.

Custom learners can opt into lifetime graph edits with `supports_graph_growth=True`
and call `agent.growth.after_update(...)` at safe update boundaries. Keep
`crossover=None` for graph agents. Generation records include neuron, connection,
and parameter counts. Lineage records added and removed neuron IDs and edges.

## Supported combinations

| Agent kind | Serial CPU | CPU process evaluation | Torch batches | Rust batches |
| --- | --- | --- | --- | --- |
| Vector objective | Yes | Yes |  |  |
| MLP with fixed or mutable topology | Yes | Yes | Yes | Yes |
| Sparse graph | Yes | Yes | Yes | Yes |
| Evolved message head | Shared serial |  |  |  |

Neural observations use a flat `Box`. Actions use zero-start `Discrete` or a flat,
finite, strictly bounded `Box`. Shared worlds use the PettingZoo Parallel protocol
with flat `Box` observations.

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

`SharedFocalEnv` exposes a single focal agent against frozen partners through
Gymnasium. A supplied fixed message rule handles outgoing messages in that view.

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
save inference genomes; continuing custom learner state requires a checkpoint.
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
documents correctness checks and benchmark methodology.
The [alpha release checklist](https://github.com/JonathanBaj/geneticevo/blob/main/docs/release.md) records packaging checks and the
current release checks and hardware validation status.

For a saved plot: `from geneticevo.plotting import plot_history`, then call
`plot_history("results/neural/metrics.jsonl", "results/neural/fitness.png")`.

## License

geneticevo is licensed under the [MIT License](https://github.com/JonathanBaj/geneticevo/blob/main/LICENSE). Copyright notices and
the license text ship with the source distribution and wheel. Dependencies
retain their own licenses.
