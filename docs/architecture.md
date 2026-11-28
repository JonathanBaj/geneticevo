# Architecture and extension contracts

The modules are deliberately small in number. `core` holds typed protocols and
data; `agents` supplies NumPy genomes/policies; `environments` supplies a CPU
reference; `evaluation` handles frozen objectives and episodes; `evolution`
implements selection and mutation; `execution` provides serial/process routes;
`runner` orchestrates lifecycle; `results` handles records and atomic checkpoints.
Optional tensor, Rust, plotting, and shared-world modules have explicit imports.

## Custom environments and scenarios

A factory must return a new Gymnasium environment on every call. It exposes
`observation_space`, `action_space`, `reset(seed=..., options=...)`, and
`step(action) -> observation, reward, terminated, truncated, info`. `close` releases
resources. `Reach1D` is a complete example. Render through your environment's
`render_mode` configuration; the runner intentionally does not open GUI windows.

Keep termination (MDP terminal state) separate from external truncation. Both end
frozen evaluation episodes. Auto-reset wrappers
that replace final observations are unsupported unless an explicit adapter
restores those observations. Tensor `step` must return final observations before
`reset_done(mask)`. Inactive tensor lanes must not move, receive rewards, or add
interactions. Custom tensor factories are responsible for matching the CPU
reference's dynamics, spaces, and reset options.

For procedural validation, define a top-level sampler:

```python
from geneticevo import Scenario

def shifted_validation(rng, count, generation):
    radius = 1.0 + 0.1 * generation
    return [Scenario(int(rng.integers(2**31)), {"radius": radius}, "validation")
            for _ in range(count)]
```

Use a separately configured training factory and a separately constructed final
test suite. Never use final scores to adjust selection, stopping, or a curriculum.
Archive scenarios are another fixed validation suite, not a final test.

## Genomes, agents, factories, and variation

`Genome` requires `clone()`, `vector()` (an independent flat NumPy array), and
`assign(values)`. It need not represent a network. A custom discrete genotype
can accept this projection while using its own mutation operator; see
`examples/custom_genome.py`. The default Gaussian mutation assumes continuous
parameters and is not appropriate for every genotype.

An agent exposes `genome`, `act(observation, rng, deterministic=True)`,
`clone(training_state=False)`, and `save(path)`. Objective-only agents need not
implement environmental actions. A factory is called as
`factory(rng, observation_space, action_space, device)` and must produce an
independent agent. All mutable genetic and non-genetic state must be copied;
fresh children discard per-agent learner state.

Mutation hooks have signature `(genome, rng) -> metadata_dict` and modify the
provided child genome. Selection hooks have signature
`(population, rng, maximize) -> Individual` and return a member of the population.
Crossover hooks accept cloned parent genomes and an RNG and return a compatible
genome. There is no implicit neural parameter averaging.

The included parameter layout uses separate actor and critic tanh MLPs.
`MLPAgentFactory(growth=NeuralGrowth(...))` optionally enables bounded width/depth
edits during life and at reproduction. Architecture lives on `hidden_sizes`,
`layer_ids`, and the monotonic `next_layer_id`; the
parameter vector reflects its current layout. `resize_hidden_sizes` preserves
overlapping genes, initializes new incoming connections randomly, and zeroes
new outgoing connections into old neurons so widening preserves actor/critic
outputs. `insert_hidden_layer` and `remove_hidden_layer` edit any position;
`resize_hidden_sizes` appends/removes trailing layers by default and also accepts
explicit IDs. Output parameters map to a permanent output role, rather than an
old numeric index. New layers use identity-shaped input matrices and zero bias;
the extra tanh and changed adjacency can change outputs. Removing neurons crops
their connections. Shapes change atomically;
incompatible Adam state is discarded, while growth counters are retained.

Acquired widths, depths, and layer identities are inherited in both modes.
Baldwinian inheritance keeps birth genes on surviving layers and initializes
new capacity without copying learned weights. Fresh offspring mutate widths and
optionally depth before scalar parameter mutation. Architecture mutations appear
in lineage; generation metrics include widths/counts and learner `hidden_layers`.
Agent save files include growth configuration, layer IDs and the next ID without
changing schema 1. Legacy files/checkpoints receive sequential layer IDs. Checkpoints
retain architecture and per-individual learner state. Custom crossover is
currently rejected for growth-enabled agents. No network or optimizer state is
shared between individuals.

## Frozen population evaluation

The Torch backend groups compatible neural architectures and batches independent
policies on CPU or CUDA. A tensor environment can run on the same device as the
policies. Gymnasium environments use the CPU bridge. Both routes preserve the
population order even when architectures have different widths or depths.

The optional Rust CPU backend passes each compatible group of dense and graph
agents to the PyO3 extension. Each member has its own active float32 gene vector.
Rayon distributes inference after observations have been copied to Rust-owned
storage and the GIL is released. Gymnasium environments, selection, mutations,
lineage, archives, and checkpointing stay in Python.

Archives use the same serial CPU reference evaluator across all backends.

## Sparse graph topology

`GraphAgent` stores stable computational neuron IDs and sorted directed edges
separately from the numeric genome. Observation sources use negative IDs; action
readouts have IDs `0..action_dim-1`, and the value readout has ID `action_dim`.
Readout neurons can also send signals, including feedback connections. Other
neurons may be disconnected, recurrent, or self connected. The genome contains
one scalar per edge, one bias per computational neuron, and continuous-action
log standard deviations. A shared graph supplies actor and critic outputs.

Forward evaluation starts with zero activations on every observation and injects
the observation at each synchronous settling step. Sparse edge gather and
scatter-add implement weighted messages; hidden neurons use tanh, readouts stay
linear. The fixed settling budget makes cycles finite. NumPy, Torch, and Rust
implement the same computation.

`GraphGrowth` independently proposes adding/removing nodes and edges at learning
boundaries, within configured bounds. All action/value neurons are permanent.
Neuron removal deletes incident edges. Surviving weights and biases are mapped
by IDs, so removing a middle neuron never shifts another neuron's genes. IDs
are allocated monotonically within a lineage. Edge additions are sampled without
constructing a dense adjacency matrix. Edits may immediately change behavior;
any structural change clears Adam state even when parameter counts stay equal.
Reproduction clones the acquired topology before structural and scalar weight
mutation. Baldwinian learning copies acquired topology back to the birth agent,
retaining its surviving birth genes and initializing only new genes. New children
discard training counters; elites preserve theirs in Lamarckian mode. Graph
architecture is included in archives, portable agent files, and checkpoints.
Parameter diversity aligns weights by edge IDs and biases by neuron IDs, treating
missing genes as zeros. Graph crossover is rejected because numeric position
alone does not identify homologous graph genes. Population metrics record neuron,
edge, and parameter counts; lineage includes specific structural changes.

## Custom evaluators and learners

An evaluator implements
`evaluate(agent, scenarios, action_seed) -> Evaluation`. `fitness` is the scalar
used for selection. Keep task metrics and optional descriptors separate.
`GymEvaluator.aggregate` customizes episodic aggregation; its optional penalty
is subtracted from fitness. For minimization, choose the sign accordingly or
return a fully composed fitness in a custom evaluator. These hooks do not
guarantee constraint satisfaction or deployment safety.

Learners implement the `Learner` protocol in `core.py`; `GenomeLearner` supplies
copy/export/state helpers. See the executable `examples/custom_learner.py` for a
bounded random-search adapter. A custom learner implements these responsibilities.

1. Validate algorithm/space/backend compatibility before training.
2. Prepare per-agent training state; do not keep it on the shared learner object.
3. Learn with a specified transition budget and report actual interactions,
   update counts, and metrics. Account for vector stepping and rounding.
4. Export inherited parameters and distinguish a fresh child from a continuing
   unchanged individual.
5. Save/restore all training state needed for boundary resume.

The runner calls `learn` for each eligible individual before frozen evaluation.
Use the environment factory to open and close isolated environments. Report
actual interactions and updates through `LearnReport`.

## Shared interaction

`ParallelWorldAdapter` accepts PettingZoo Parallel-compatible environments.
Every action dictionary maps exactly the active world IDs. A focal evaluator
retains its explicit ID mapping; population membership and world role count are
independent. Selection pools rotate partners; archive and final evaluators use
explicit frozen partner pools. Environments define cooperative or competitive
rewards; evaluator `reward_mode` determines individual or mean-team selection.

`CommunicatingAgentFactory` evolves an extra deterministic tanh message head.
Messages depend only on the policy's permitted observation (including earlier
inboxes). `SharedFocalEnv` presents a single focal policy against a fixed
partner pool through a Gymnasium environment. Its supplied message rule sets
the outgoing message.

## Official integration references

Implementation was checked against the official
[Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/),
[PyTorch population/model vectorization tutorial](https://docs.pytorch.org/tutorials/intermediate/ensembling.html),
and [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/).

