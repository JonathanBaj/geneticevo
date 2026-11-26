"""Optional independent CPU PPO over native, variable-topology policies."""
from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
import time

import numpy as np

from .agents import MLPAgent, validate_training_environment
from .core import Evaluation, LearnReport, seed_for
from .graph import GraphAgent, GraphGrowth
from .growth import NeuralGrowth
from .integrations_validation import reject_normalization
from .learners import GenomeLearner


METRICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm")
STATE_KEY = "rust_ppo"
STATE_VERSION = 1


def require_extension():
    try:
        from ._rust import Population
    except ImportError as error:
        raise ImportError(
            "Rust execution requires geneticevo-core. Install geneticevo[rust] with a compatible "
            "wheel, or build the core with 'python -m pip install ./rust' from the source checkout."
        ) from error
    return Population


def topology(agent):
    if isinstance(agent, GraphAgent):
        sources, targets = agent.edge_indices()
        return (), (len(agent.node_ids), list(zip(sources.tolist(), targets.tolist())), agent.settling_steps)
    return agent.hidden_sizes, None


def _policy_groups(agents, chunk_size):
    groups = {}
    for i, agent in enumerate(agents):
        if not isinstance(agent, (MLPAgent, GraphAgent)):
            raise ValueError("Rust execution requires MLPAgentFactory or GraphAgentFactory agents")
        key = (agent.observation_dim, agent.action_dim, agent.discrete, tuple(agent.low), tuple(agent.high))
        groups.setdefault(key, []).append(i)
    for group in groups.values():
        size = chunk_size or len(group)
        for start in range(0, len(group), size):
            yield group[start:start + size]


def _population(agents, settings, seed=0, generation=0, member_ids=None):
    first = agents[0]
    core = require_extension()(first.observation_dim, first.discrete, first.low.tolist(), first.high.tolist(), settings)
    for index, agent in enumerate(agents):
        saved = None if member_ids is None else getattr(agent, "training_state", {}).get(STATE_KEY)
        adam = None
        if saved is not None:
            version, step, moment1, moment2, rng = saved
            if version != STATE_VERSION:
                raise ValueError("Unsupported Rust PPO checkpoint state version")
            adam = step, moment1, moment2
        else:
            member_id = 0 if member_ids is None else member_ids[index]
            rng = (seed_for(seed, "actions", generation, member_id), 0,
                   seed_for(seed, "actions", generation, member_id, 1), 0)
        core.add(topology(agent), agent.genome.values, rng, adam)
    return core


class _GymPopulation:
    def __init__(self, environments, members, copies, first):
        self.envs = environments
        self.copies = copies
        self.discrete = first.discrete
        self.action_dim = first.action_dim
        self.obs = np.empty((members, copies, first.observation_dim), np.float32)
        self.rewards = np.empty((members, copies), np.float32)
        self.terminated = np.empty((members, copies), bool)
        self.truncated = np.empty_like(self.terminated)
        self._obs = self.obs.reshape(members * copies, first.observation_dim)
        self._rewards = self.rewards.ravel()
        self._terminated = self.terminated.ravel()
        self._truncated = self.truncated.ravel()

    def reset(self, *, seeds=None, scenarios=None):
        for i, env in enumerate(self.envs):
            member, slot = divmod(i, self.copies)
            if scenarios is None:
                obs, _ = env.reset(seed=(seeds[member] + slot) % 2**32)
            else:
                scenario = scenarios[slot]
                obs, _ = env.reset(seed=scenario.seed, options=copy.deepcopy(scenario.options))
            self._obs[i] = obs

    def step(self, actions, active=None):
        flat = actions.ravel() if self.discrete else actions.reshape(-1, self.action_dim)
        # Box actions have their own backing array. A Gym environment can retain
        # an action after step() and it will not be overwritten by the next call.
        rewards, terminated, truncated = self._rewards, self._terminated, self._truncated
        observations = self._obs
        if active is None:
            slots = range(len(self.envs))
        else:
            slots = np.flatnonzero(active)
        for i in slots:
            action = int(flat[i]) if self.discrete else flat[i]
            observations[i], rewards[i], terminated[i], truncated[i], _ = self.envs[i].step(action)

    def reset_done(self):
        for i in np.flatnonzero(self._terminated | self._truncated):
            self._obs[i], _ = self.envs[i].reset()


@contextmanager
def _gym_population(factory, agents, copies):
    with ExitStack() as stack:
        environments, identities = [], set()
        for _ in range(len(agents) * copies):
            env = factory()
            identity = id(env.unwrapped)
            if identity in identities:
                raise ValueError("env_factory must return independent environments")
            stack.callback(env.close)
            reject_normalization(env)
            identities.add(identity)
            environments.append(env)
        yield _GymPopulation(environments, len(agents), copies, agents[0])


@contextmanager
def _phase(name, timings):
    if timings is None:
        yield
    else:
        start = time.perf_counter()
        yield
        key = name + "_seconds"
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


class _GrowthChecks:
    def __init__(self, agent, seed):
        self.agent = agent
        growth = agent.growth
        graph = isinstance(agent, GraphAgent)
        self.key = "graph_growth" if graph else "neural_growth"
        self.counters = agent.training_state.setdefault(self.key, {"interactions": 0, "last_check": 0, "events": 0})
        self.initial = self.counters.copy()
        self.pending = 0
        self.rng = np.random.default_rng(seed)
        # Subclasses can implement a different check schedule or hook signature.
        self.builtin = type(growth) is (GraphGrowth if graph else NeuralGrowth)

    def after_update(self, core, index, transitions, value_loss):
        self.pending += transitions
        growth = self.agent.growth
        if self.builtin and self.counters["interactions"] + self.pending - self.counters["last_check"] < growth.check_every:
            return

        def sync_genes():
            self.agent.genome.assign(core.get_genes(index))

        if self.builtin:
            changed = growth.after_update(self.agent, self.pending, value_loss, self.rng, before_edit=sync_genes)
        else:
            sync_genes()
            changed = growth.after_update(self.agent, self.pending, value_loss, self.rng)
            self.counters = self.agent.training_state[self.key]
        self.pending = 0
        if changed:
            core.replace_topology(index, topology(self.agent), self.agent.genome.values)

    def report(self, metrics):
        self.counters["interactions"] += self.pending
        agent = self.agent
        metrics.update(growth_events=self.counters["events"] - self.initial["events"],
                       parameters=agent.genome.values.size)
        if isinstance(agent, GraphAgent):
            metrics.update(hidden_neurons=agent.hidden_neurons, connections=len(agent.edges))
        else:
            metrics.update(hidden_neurons=sum(agent.hidden_sizes), hidden_layers=len(agent.hidden_sizes))
            for source, label in (("grow", "width_growth_events"), ("shrink", "width_shrink_events"),
                                  ("add_layer", "layers_added"), ("remove_layer", "layers_removed")):
                metrics[label] = self.counters.get(source, 0) - self.initial.get(source, 0)


@dataclass(frozen=True)
class RustNativePPO(GenomeLearner):
    """Independent PPO for dense and graph agents on CPU, using Gymnasium environments."""

    supports_neural_growth = True
    supports_graph_growth = True
    execution_backend = "rust"

    env_copies: int = 1
    rollout_steps: int = 32
    epochs: int = 2
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    device: str = "cpu"
    population_chunk_size: int | None = None
    measure_phases: bool = False

    def __post_init__(self):
        for name in ("env_copies", "rollout_steps", "epochs", "minibatch_size", "population_chunk_size"):
            value = getattr(self, name)
            if value is None and name == "population_chunk_size":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.env_copies * self.rollout_steps < 2:
            raise ValueError("PPO needs at least two transitions per person per rollout")
        if not isinstance(self.measure_phases, bool):
            raise ValueError("measure_phases must be a bool")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.gamma, self.gae_lambda, self.clip,
                   self.value_coefficient, self.entropy_coefficient, self.max_gradient_norm)) or not (
                0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.learning_rate > 0 and
                self.clip > 0 and self.max_gradient_norm > 0):
            raise ValueError("Invalid PPO hyperparameters")
        if self.device != "cpu":
            raise ValueError("RustNativePPO requires device='cpu'")
        require_extension()

    def validate(self, agent, env_factory, backend):
        self.prepare(agent, 0)
        if backend not in ("rust", "serial", "parallel"):
            raise ValueError("RustNativePPO requires backend='rust', 'serial', or 'parallel'")
        validate_training_environment(agent, env_factory)

    def prepare(self, agent, seed):
        if not isinstance(agent, (MLPAgent, GraphAgent)):
            raise ValueError("RustNativePPO requires MLPAgentFactory or GraphAgentFactory agents")

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        return self.learn_population([agent], env_factory, None, budget, environment_seed, action_seed)[0]

    def learn_population(self, agents, env_factory, tensor_env_factory, budget, seed, generation, *,
                         member_ids=None, timings=None):
        if tensor_env_factory is not None:
            raise ValueError("RustNativePPO uses Gymnasium environments via training_env_factory")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("Learning budget must be a positive integer")
        ids = list(range(len(agents))) if member_ids is None else list(member_ids)
        if len(ids) != len(agents) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids) or (
                len(set(ids)) != len(ids)):
            raise ValueError("member_ids must contain one unique nonnegative integer per agent")
        reports = [None] * len(agents)
        for indices in _policy_groups(agents, self.population_chunk_size):
            chunk = self._learn_chunk([agents[i] for i in indices], env_factory, budget, seed, generation,
                                      [ids[i] for i in indices], timings)
            for i, report in zip(indices, chunk):
                reports[i] = report
        return reports

    def _learn_chunk(self, agents, factory, budget, seed, generation, ids, timings):
        transitions = self.env_copies * self.rollout_steps
        rollouts = (budget + transitions - 1) // transitions
        updates = rollouts * self.epochs * ((transitions + self.minibatch_size - 1) // self.minibatch_size)
        totals = np.zeros((len(agents), len(METRICS)), dtype=np.float64)
        with ExitStack() as stack:
            with _phase("setup", timings):
                core = _population(agents, self, seed, generation, ids)
                env = stack.enter_context(_gym_population(factory, agents, self.env_copies))
                env.reset(seeds=[seed_for(seed, "training", generation, index) for index in ids])
                growth = {}
                for i, (agent, index) in enumerate(zip(agents, ids)):
                    if not hasattr(agent, "training_state"):
                        agent.training_state = {}
                    if agent.growth is not None:
                        growth[i] = _GrowthChecks(agent, seed_for(seed, "growth", generation, index))
            for rollout in range(rollouts):
                with _phase("rollout", timings):
                    for step in range(self.rollout_steps):
                        actions = core.sample(env.obs)
                        env.step(actions)
                        core.record(env.rewards, env.terminated, env.truncated, env.obs)
                        if rollout + 1 < rollouts or step + 1 < self.rollout_steps:
                            env.reset_done()
                with _phase("update", timings):
                    losses = core.update()
                    totals += losses
                if growth:
                    with _phase("growth", timings):
                        for i, checker in growth.items():
                            checker.after_update(core, i, transitions, float(losses[i, 1]))
            with _phase("export", timings):
                states = core.export()
                for agent, (genes, step, first, second, rng) in zip(agents, states):
                    agent.genome.assign(genes)
                    agent.training_state.pop("adam", None)
                    agent.training_state[STATE_KEY] = (STATE_VERSION, step, first, second, rng)
        reports = []
        for i, scores in enumerate(totals / rollouts):
            metrics = dict(zip(METRICS, map(float, scores)))
            if i in growth:
                growth[i].report(metrics)
            reports.append(LearnReport(transitions * rollouts, updates, budget, metrics))
        return reports


def evaluate_rust(agents, scenarios, factory, copies, max_steps, settings=None):
    if settings is None:
        settings = RustNativePPO()
    scores = [None] * len(agents)
    for indices in _policy_groups(agents, settings.population_chunk_size):
        group = [agents[i] for i in indices]
        core = _population(group, settings)
        returns = np.zeros((len(group), len(scenarios)), dtype=np.float32)
        lengths = np.zeros_like(returns, dtype=np.int64)
        for start in range(0, len(scenarios), copies):
            chunk = scenarios[start:start + copies]
            span = slice(start, start + len(chunk))
            total, steps = returns[:, span], lengths[:, span]
            with _gym_population(factory, group, len(chunk)) as env:
                env.reset(scenarios=chunk)
                active = np.ones(total.shape, dtype=bool)
                for _ in range(max_steps):
                    env.step(core.act(env.obs), active)
                    np.add(total, env.rewards, out=total, where=active)
                    steps += active
                    active &= ~(env.terminated | env.truncated)
                    if not active.any():
                        break
        for local, i in enumerate(indices):
            scores[i] = Evaluation(float(returns[local].mean()), returns[local].tolist(),
                                   lengths[local].tolist(), int(lengths[local].sum()), context=list(scenarios))
    return scores
