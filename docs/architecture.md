# Architecture and extension contracts

The modules are deliberately small in number. `core` holds typed protocols and
data; `agents` supplies NumPy genomes/policies; `environments` supplies a CPU
reference; `evaluation` handles frozen objectives and episodes; `evolution`
implements selection and mutation; `execution` provides serial/process routes;
`runner` orchestrates lifecycle; `results` handles records and atomic checkpoints.
Optional tensor/PPO, SB3, plotting, and shared-world modules have explicit imports.

## Custom environments and scenarios

A factory must return a new Gymnasium environment on every call. It exposes
`observation_space`, `action_space`, `reset(seed=..., options=...)`, and
`step(action) -> observation, reward, terminated, truncated, info`. `close` releases
resources. `Reach1D` is a complete example. Render through your environment's
`render_mode` configuration; the runner intentionally does not open GUI windows.

Keep termination (MDP terminal state) separate from external truncation. Both end
evaluation episodes; learning bootstraps only a truncation. Auto-reset wrappers
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
fresh children must discard optimizers, replay, counters, and normalization.

Mutation hooks have signature `(genome, rng) -> metadata_dict` and modify the
provided child genome. Selection hooks have signature
`(population, rng, maximize) -> Individual` and return a member of the population.
Crossover hooks accept cloned parent genomes and an RNG and return a compatible
genome. There is no implicit neural parameter averaging.

The included parameter layout uses separate actor and critic tanh MLPs.
`MLPAgentFactory(growth=NeuralGrowth(...))` optionally enables bounded width
growth and size mutations. Architecture lives on `MLPAgent.hidden_sizes`; the
parameter vector reflects its current layout. `resize_hidden_sizes` preserves
overlapping genes, initializes new incoming connections randomly, and zeroes
new outgoing connections into old neurons so widening preserves actor/critic
outputs. Removing neurons crops their connections. Shapes change atomically;
incompatible Adam state is discarded, while growth counters are retained.

Native PPO checks growth at completed rollout/update boundaries without resetting
environments. The default path trains growing agents individually and rebuilds
policy/optimizer tensors at growth. Fixed-size agents use architecture groups.
The opt-in accelerated path groups by spaces and depth, allocates one bounded
capacity per layer, and masks each individual's active hidden prefix. Growth
copies only newly initialized slots and clears only that individual's Adam
moments and step counter. Existing device weights remain in place. Portable
exports crop parameters and moments to active shapes; generation checkpoints
retain the existing CPU schema.

Accelerated action, shuffle, environment, and growth streams use separate seeds
derived from generation and individual identity. The runner passes numeric
individual IDs; direct learn_population calls can pass member_ids explicitly.
TensorReach1D and GymTensorBridge implement set_member_seeds(seeds) so chunking
and regrouping preserve each individual's environment stream as well. Custom
tensor environments need this method for accelerated training and may implement
prepare_rollout(steps) to batch reset randomness. Fixed rollout shapes and work
schedules are required for reproducibility; CPU/CUDA bitwise equivalence is not
promised. Evaluation preserves original population order and can also be chunked.
All chunks complete before selection. See [accelerated PPO](accelerated_ppo.md).

Acquired widths are inherited in both weight-inheritance modes. Baldwinian
inheritance expands the original genome with fresh, disconnected neurons rather
than copying learned weights. Fresh offspring mutate widths before scalar
parameter mutations; hidden layer count stays fixed. Architecture mutations
appear in lineage; generation metrics include widths/counts. Agent save files
optionally include growth configuration without changing schema 1. Checkpoints
retain architecture and per-individual counters/RNG state. Custom crossover is
currently rejected for growth-enabled agents. No network or optimizer state is
shared between individuals.

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
bounded random-search adapter, demonstrating that the interface is not tied to
PPO rollouts. Required behavior:

1. Validate algorithm/space/backend compatibility before training.
2. Prepare per-agent training state; do not keep it on the shared learner object.
3. Learn with a specified transition budget and report actual interactions,
   update counts, and metrics. Account for vector stepping and rounding.
4. Export inherited parameters and distinguish a fresh child from a continuing
   unchanged individual.
5. Save/restore all training state needed for boundary resume.

Runner dispatches population learning to `learn_population` only when a learner
implements it. Native PPO requires common rollout/minibatch schedules. Its
loss is a sum of per-individual means, preserving independent loss scaling.
Per-individual norms clip gradients; Adam moments and counters have a population
dimension. Sampling and shuffling use independent generators. Random filling
uses a small population loop; computation, simulation, clipping, and updates are
batched. Rollout collection still uses a Python time loop.

SB3 RNG use is scoped to each individual and the caller's global state is
restored. SB3 state serialization includes policy/optimizer, counters, replay
when applicable, targets, exploration schedule, and per-agent action RNG state.
PPO buffers are discarded at completed-rollout boundaries. Learning environments
are closed, reset, and reconstructed at boundaries rather than serialized.
Normalization wrappers are explicitly rejected.

## Shared interaction

`ParallelWorldAdapter` accepts PettingZoo Parallel-compatible environments.
Every action dictionary maps exactly the active world IDs. A focal evaluator
retains its explicit ID mapping; population membership and world role count are
independent. Selection pools rotate partners; archive and final evaluators use
explicit frozen partner pools. Environments define cooperative or competitive
rewards; evaluator `reward_mode` determines individual or mean-team selection.

`CommunicatingAgentFactory` evolves an extra deterministic tanh message head.
Messages depend only on the policy's permitted observation (including earlier
inboxes). Fixed-rule task learners on `SharedFocalEnv` optimize only task action
likelihood. Joint stochastic task/message training is deferred, rather than
treated as ordinary SB3 PPO.

## Official integration references

Implementation was checked against the official
[Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/),
[PyTorch population/model vectorization tutorial](https://docs.pytorch.org/tutorials/intermediate/ensembling.html),
[SB3 PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
[SB3 DQN source/API](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/dqn/dqn.html),
and [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/).
Local installed SB3 source was also exercised by the integration tests.

