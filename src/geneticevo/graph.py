"""Sparse, heritable directed brains with bounded lifetime structural plasticity.

Inputs have negative IDs; all other neurons (including action/value readouts)
can connect in either direction, to themselves, or remain disconnected. Cycles
are evaluated by a fixed number of synchronous settling steps per observation.
This is an engineering model inspired by connectivity, not a biological model.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .agents import validate_spaces
from .core import VectorGenome


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _new_edges(nodes, observation_dim, existing, count, rng):
    """Uniform missing edges without constructing a dense adjacency matrix."""
    sources = list(range(-observation_dim, 0)) + list(nodes)
    possible = len(sources) * len(nodes)
    count = min(count, possible - len(existing))
    if not count:
        return set()
    # At most len(existing) sampled pairs can already exist. Taking the first
    # count missing pairs gives a uniform sample with storage proportional to E.
    indices = rng.choice(possible, count + len(existing), replace=False)
    added = set()
    for index in indices:
        edge = (sources[int(index) // len(nodes)], nodes[int(index) % len(nodes)])
        if edge not in existing:
            added.add(edge)
            if len(added) == count:
                break
    return added


@dataclass(frozen=True)
class GraphGrowth:
    """Check after learning updates; independently propose additions and removals.

    Fitness selection chooses among offspring with different structures. These
    stochastic edits do not themselves guarantee improved fitness. Bounds apply
    to hidden neurons; action/value neurons are permanent. Zero edges is valid.
    """
    check_every: int = 256
    min_neurons: int = 0
    max_neurons: int = 128
    max_connections: int = 2048
    neurons_per_event: int = 1
    connections_per_event: int = 2
    node_add_probability: float = 0.5
    node_remove_probability: float = 0.25
    connection_add_probability: float = 1.0
    connection_remove_probability: float = 0.25
    mutation_probability: float = 0.25

    def __post_init__(self):
        for name in ('check_every', 'neurons_per_event', 'connections_per_event'):
            _integer(name, getattr(self, name), 1)
        for name in ('min_neurons', 'max_neurons', 'max_connections'):
            _integer(name, getattr(self, name))
        if self.min_neurons > self.max_neurons:
            raise ValueError('min_neurons must not exceed max_neurons')
        for name in ('node_add_probability', 'node_remove_probability',
                     'connection_add_probability', 'connection_remove_probability', 'mutation_probability'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')

    def validate(self, agent):
        if not self.min_neurons <= agent.hidden_neurons <= self.max_neurons:
            raise ValueError('Graph hidden neuron count must be within growth bounds')
        if len(agent.edges) > self.max_connections:
            raise ValueError('Graph connections exceed max_connections')

    def _edit(self, agent, rng, probabilities):
        self.validate(agent)
        before = agent.structure
        add_node, remove_node, add_edge, remove_edge = probabilities
        nodes, edges = list(agent.node_ids), set(agent.edges)
        next_id = agent.next_node_id
        if rng.random() < remove_node:
            hidden = nodes[agent.output_count:]
            count = min(self.neurons_per_event, len(hidden) - self.min_neurons)
            if count:
                removed = set(int(x) for x in rng.choice(hidden, count, replace=False))
                nodes = [x for x in nodes if x not in removed]
                edges = {e for e in edges if not removed.intersection(e)}
        if rng.random() < add_node:
            count = min(self.neurons_per_event, self.max_neurons - (len(nodes) - agent.output_count))
            nodes.extend(range(next_id, next_id + count))
            next_id += count
        if rng.random() < remove_edge and edges:
            ordered = sorted(edges)
            indices = rng.choice(len(ordered), min(self.connections_per_event, len(ordered)), replace=False)
            edges.difference_update(ordered[int(i)] for i in indices)
        if rng.random() < add_edge:
            count = min(self.connections_per_event, self.max_connections - len(edges))
            edges.update(_new_edges(nodes, agent.observation_dim, edges, count, rng))
        agent.restructure(tuple(nodes), tuple(sorted(edges)), next_id, rng)
        return before != agent.structure

    def after_update(self, agent, interactions, value_loss, rng):
        _integer('interaction count', interactions, 1)
        self.validate(agent)
        if not hasattr(agent, 'training_state'):
            agent.training_state = {}
        state = agent.training_state.setdefault('graph_growth',
            {'interactions': 0, 'last_check': 0, 'events': 0})
        state['interactions'] += interactions
        if state['interactions'] - state['last_check'] < self.check_every:
            return False
        state['last_check'] = state['interactions']
        changed = self._edit(agent, rng, (self.node_add_probability, self.node_remove_probability,
            self.connection_add_probability, self.connection_remove_probability))
        state['events'] += int(changed)
        return changed

    def mutate(self, agent, rng):
        before = {'neurons': agent.hidden_neurons, 'connections': len(agent.edges)}
        nodes, edges = set(agent.node_ids), set(agent.edges)
        # Independent birth edits allow growth, shrinkage, and rewiring.
        self._edit(agent, rng, (self.mutation_probability,) * 4)
        return {'kind': 'graph_structure', 'before': before,
                'after': {'neurons': agent.hidden_neurons, 'connections': len(agent.edges)},
                'nodes_added': sorted(set(agent.node_ids) - nodes),
                'nodes_removed': sorted(nodes - set(agent.node_ids)),
                'connections_added': sorted(set(agent.edges) - edges),
                'connections_removed': sorted(edges - set(agent.edges))}


@dataclass
class GraphAgent:
    genome: VectorGenome
    observation_dim: int
    action_dim: int
    discrete: bool
    low: np.ndarray
    high: np.ndarray
    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    next_node_id: int
    settling_steps: int = 4
    growth: GraphGrowth | None = None

    @property
    def output_count(self):
        return self.action_dim + 1  # action readouts followed by the value readout

    @property
    def hidden_neurons(self):
        return len(self.node_ids) - self.output_count

    @property
    def structure(self):
        return self.node_ids, self.edges

    def layout(self):
        result = [('weights', (len(self.edges),)), ('biases', (len(self.node_ids),))]
        if not self.discrete:
            result.append(('log_std', (self.action_dim,)))
        return result

    def parameters(self):
        result, offset = {}, 0
        for name, shape in self.layout():
            count = shape[0]
            result[name] = self.genome.values[offset:offset + count].reshape(shape)
            offset += count
        return result

    def edge_indices(self):
        lookup = {node: self.observation_dim + i for i, node in enumerate(self.node_ids)}
        lookup.update({node: node + self.observation_dim for node in range(-self.observation_dim, 0)})
        return (np.array([lookup[a] for a, _ in self.edges], dtype=np.int64),
                np.array([lookup[b] - self.observation_dim for _, b in self.edges], dtype=np.int64))

    def forward(self, observation, prefix='actor'):
        observation = np.asarray(observation, dtype=np.float32)
        state = np.zeros(observation.shape[:-1] + (len(self.node_ids),), dtype=np.float32)
        params = self.parameters()
        sources, targets = self.edge_indices()
        for _ in range(self.settling_steps):
            values = np.concatenate((observation, state), axis=-1)
            updated = np.broadcast_to(params['biases'], state.shape).copy()
            # add.at handles multiple incoming edges without dropping duplicates.
            np.add.at(updated, (..., targets), values[..., sources] * params['weights'])
            state = np.tanh(updated)
            state[..., :self.output_count] = updated[..., :self.output_count]
        return state[..., :self.action_dim] if prefix == 'actor' else state[..., self.action_dim:self.output_count]

    def act(self, observation, rng, deterministic=True):
        output = self.forward(observation)
        if self.discrete:
            if deterministic:
                return int(np.argmax(output))
            probabilities = np.exp(output - output.max())
            return int(rng.choice(self.action_dim, p=probabilities / probabilities.sum()))
        if not deterministic:
            output = output + rng.normal(size=self.action_dim) * np.exp(self.parameters()['log_std'].clip(-5, 2))
        return (self.low + (np.tanh(output) + 1) * 0.5 * (self.high - self.low)).astype(np.float32)

    def restructure(self, node_ids, edges, next_node_id, rng):
        """Retain surviving genes by stable neuron/edge identity, not array offset."""
        node_ids, edges = tuple(node_ids), tuple(sorted(edges))
        if (node_ids[:self.output_count] != tuple(range(self.output_count)) or
                len(set(node_ids)) != len(node_ids) or any(type(x) is not int or x < 0 for x in node_ids)):
            raise ValueError('Graph requires unique nonnegative neuron IDs and permanent readouts')
        _integer('next_node_id', next_node_id, max(node_ids) + 1)
        valid_sources = set(node_ids) | set(range(-self.observation_dim, 0))
        if len(set(edges)) != len(edges) or any(a not in valid_sources or b not in node_ids for a, b in edges):
            raise ValueError('Invalid or duplicate graph connections')
        if self.growth and (not self.growth.min_neurons <= len(node_ids) - self.output_count <= self.growth.max_neurons
                            or len(edges) > self.growth.max_connections):
            raise ValueError('Graph structure exceeds growth bounds')
        if (node_ids, edges) == self.structure:
            self.next_node_id = max(self.next_node_id, next_node_id)
            return False
        old = self.parameters()
        weights = dict(zip(self.edges, old['weights']))
        biases = dict(zip(self.node_ids, old['biases']))
        parts = [np.array([weights[e] if e in weights else rng.normal(0, 0.2) for e in edges], np.float32),
                 np.array([biases.get(node, 0) for node in node_ids], np.float32)]
        if not self.discrete:
            parts.append(old['log_std'].copy())
        self.genome = VectorGenome(np.concatenate(parts))
        self.node_ids, self.edges = node_ids, edges
        self.next_node_id = max(self.next_node_id, next_node_id)
        if hasattr(self, 'training_state'):
            self.training_state.pop('adam', None)
        return True

    def inherit_structure(self, learned, rng):
        self.restructure(learned.node_ids, learned.edges, learned.next_node_id, rng)

    def clone(self, training_state=False):
        cloned = copy.deepcopy(self)
        if not training_state and hasattr(cloned, 'training_state'):
            del cloned.training_state
        return cloned

    def save(self, path):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / 'genome.npz', values=self.genome.vector())
        spec = {name: getattr(self, name) for name in ('observation_dim', 'action_dim', 'discrete',
            'node_ids', 'edges', 'next_node_id', 'settling_steps')}
        spec.update(kind='graph', schema=1, low=self.low.tolist(), high=self.high.tolist(),
                    growth=asdict(self.growth) if self.growth else None)
        (folder / 'agent.json').write_text(json.dumps(spec), encoding='utf-8')


@dataclass(frozen=True)
class GraphAgentFactory:
    initial_neurons: int = 8
    initial_connections: int = 24
    settling_steps: int = 4
    growth: GraphGrowth | None = field(default_factory=GraphGrowth)

    def __post_init__(self):
        _integer('initial_neurons', self.initial_neurons)
        _integer('initial_connections', self.initial_connections)
        _integer('settling_steps', self.settling_steps, 1)
        if self.growth is not None:
            if not isinstance(self.growth, GraphGrowth):
                raise ValueError('growth must be GraphGrowth or None')
            if not self.growth.min_neurons <= self.initial_neurons <= self.growth.max_neurons:
                raise ValueError('initial_neurons must be within growth bounds')
            if self.initial_connections > self.growth.max_connections:
                raise ValueError('initial_connections exceeds max_connections')

    def __call__(self, rng, observation_space=None, action_space=None, device='cpu'):
        validate_spaces(observation_space, action_space)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = int(action_space.n if discrete else action_space.shape[0])
        nodes = tuple(range(action_dim + 1 + self.initial_neurons))
        possible = (observation_space.shape[0] + len(nodes)) * len(nodes)
        if self.initial_connections > possible:
            raise ValueError('initial_connections exceeds the number of possible directed connections')
        edges = tuple(sorted(_new_edges(nodes, observation_space.shape[0], set(), self.initial_connections, rng)))
        parts = [rng.normal(0, 0.2, len(edges)), np.zeros(len(nodes))]
        if not discrete:
            parts.append(np.zeros(action_dim))
        return GraphAgent(VectorGenome(np.concatenate(parts)), observation_space.shape[0], action_dim, discrete,
            np.zeros(action_dim, np.float32) if discrete else action_space.low.copy(),
            np.ones(action_dim, np.float32) if discrete else action_space.high.copy(),
            nodes, edges, len(nodes), self.settling_steps, self.growth)
