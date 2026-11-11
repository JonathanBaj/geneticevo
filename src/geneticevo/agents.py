"""NumPy vector and MLP agents; importing these never imports PyTorch."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from .core import VectorGenome
from .growth import NeuralGrowth


def validate_spaces(observation_space: Any, action_space: Any) -> None:
    if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1 or observation_space.shape[0] < 1:
        raise ValueError("Neural agents require a flat Box observation space")
    if isinstance(action_space, spaces.Discrete):
        if action_space.start != 0:
            raise ValueError("Discrete actions must start at zero")
    elif isinstance(action_space, spaces.Box):
        if len(action_space.shape) != 1 or action_space.shape[0] < 1 or not (np.isfinite(action_space.low).all()
                                               and np.isfinite(action_space.high).all()
                                               and (action_space.high > action_space.low).all()):
            raise ValueError("Box actions must be flat, finite and strictly bounded")
    else:
        raise ValueError("Supported actions: Discrete or finite flat Box")


@dataclass
class VectorAgent:
    genome: VectorGenome

    def clone(self, training_state: bool = False) -> VectorAgent:
        cloned = copy.deepcopy(self)
        if not training_state and hasattr(cloned, "training_state"):
            del cloned.training_state
        return cloned

    def act(self, observation, rng, deterministic=True):
        raise TypeError("VectorAgent is for an objective evaluator, not an environment policy")

    def save(self, path: str) -> None:
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "genome.npz", values=self.genome.vector())
        (folder / "agent.json").write_text(json.dumps({"kind": "vector", "schema": 1}), encoding="utf-8")


@dataclass(frozen=True)
class VectorAgentFactory:
    dimension: int
    initial_std: float = 1.0

    def __post_init__(self):
        if self.dimension < 1 or self.initial_std < 0:
            raise ValueError("Invalid vector initialization")

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        if device != "cpu":
            raise ValueError("VectorAgent uses CPU NumPy arrays")
        return VectorAgent(VectorGenome(rng.normal(0, self.initial_std, self.dimension)))


@dataclass
class MLPAgent:
    genome: VectorGenome
    observation_dim: int
    hidden_sizes: tuple[int, ...]
    action_dim: int
    discrete: bool
    low: np.ndarray
    high: np.ndarray
    growth: NeuralGrowth | None = None
    layer_ids: tuple[int, ...] | None = None
    next_layer_id: int = 0

    def __post_init__(self):
        if self.layer_ids is None:
            self.layer_ids = tuple(range(len(self.hidden_sizes)))
        if (len(self.layer_ids) != len(self.hidden_sizes) or len(set(self.layer_ids)) != len(self.layer_ids)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.layer_ids)):
            raise ValueError('Hidden layers require unique nonnegative integer IDs')
        self.next_layer_id = max(self.next_layer_id, max(self.layer_ids, default=-1) + 1)

    def __setstate__(self, state):
        # Older generation checkpoints predate explicit layer identities.
        self.__dict__.update(state)
        self.__post_init__()

    def resize_hidden_sizes(self, hidden_sizes, rng, *, layer_ids=None):
        """Resize width/depth, preserving surviving layers and the output head.

        New neurons have random incoming weights and zero outgoing weights into
        existing neurons. New layers start with an identity-shaped input matrix
        and zero bias. Depth edits and shrinking can change behavior because of
        tanh and changed adjacency. Any architecture edit clears Adam moments.
        Without explicit IDs, depth edits append/remove layers at the end.
        """
        sizes = tuple(hidden_sizes)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in sizes):
            raise ValueError('Resize requires positive integer widths')
        if self.growth is not None:
            self.growth.validate_sizes(sizes)
        if layer_ids is None:
            retained = self.layer_ids[:len(sizes)]
            layer_ids = retained + tuple(range(self.next_layer_id, self.next_layer_id + len(sizes) - len(retained)))
        ids = tuple(layer_ids)
        if len(ids) != len(sizes) or len(set(ids)) != len(ids) or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ids):
            raise ValueError('Hidden layers require unique nonnegative integer IDs')
        if sizes == self.hidden_sizes and ids == self.layer_ids:
            return False
        old = self.parameters()
        old_positions = {identity: i for i, identity in enumerate(self.layer_ids)}
        # Compute the new layout without changing the agent until all arrays exist.
        resized = copy.copy(self)
        resized.hidden_sizes = sizes
        parts = []
        for name, shape in resized.layout():
            if name == 'log_std':
                parts.append(old[name].copy())
                continue
            prefix, suffix = name.split('_')
            kind, index = suffix[0], int(suffix[1:])
            old_index = (len(self.hidden_sizes) if index == len(sizes) else old_positions.get(ids[index]))
            if old_index is None:
                value = np.eye(*shape, dtype=np.float32) if kind == 'w' else np.zeros(shape, np.float32)
                parts.append(value.reshape(-1))
                continue
            source = old[f'{prefix}_{kind}{old_index}']
            value = np.zeros(shape, dtype=np.float32)
            overlap = tuple(slice(0, min(a, b)) for a, b in zip(shape, source.shape))
            value[overlap] = source[overlap]
            if '_w' in name and shape[1] > source.shape[1]:
                existing_inputs = min(shape[0], source.shape[0])
                value[:existing_inputs, source.shape[1]:] = rng.normal(
                    0, np.sqrt(1 / shape[0]), (existing_inputs, shape[1] - source.shape[1]))
            parts.append(value.reshape(-1))
        genome = VectorGenome(np.concatenate(parts))
        self.hidden_sizes, self.genome, self.layer_ids = sizes, genome, ids
        self.next_layer_id = max(self.next_layer_id, max(ids, default=-1) + 1)
        if hasattr(self, 'training_state'):
            self.training_state.pop('adam', None)
        return True

    def insert_hidden_layer(self, index, width, rng):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(self.hidden_sizes):
            raise ValueError('Invalid hidden-layer insertion index')
        sizes = self.hidden_sizes[:index] + (width,) + self.hidden_sizes[index:]
        ids = self.layer_ids[:index] + (self.next_layer_id,) + self.layer_ids[index:]
        return self.resize_hidden_sizes(sizes, rng, layer_ids=ids)

    def remove_hidden_layer(self, index, rng):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.hidden_sizes):
            raise ValueError('Invalid hidden-layer removal index')
        return self.resize_hidden_sizes(self.hidden_sizes[:index] + self.hidden_sizes[index + 1:], rng,
            layer_ids=self.layer_ids[:index] + self.layer_ids[index + 1:])

    def inherit_structure(self, learned, rng):
        self.resize_hidden_sizes(learned.hidden_sizes, rng, layer_ids=learned.layer_ids)
        self.next_layer_id = max(self.next_layer_id, learned.next_layer_id)

    def named_genes(self):
        """Parameters keyed by layer identity, with a distinct permanent readout."""
        result = {}
        params = self.parameters()
        for prefix in ('actor', 'critic'):
            for i, identity in enumerate((*self.layer_ids, 'output')):
                for kind in ('w', 'b'):
                    result[(prefix, kind, identity)] = params[f'{prefix}_{kind}{i}']
        if not self.discrete:
            result[('log_std',)] = params['log_std']
        return result

    def layout(self):
        result = []
        for prefix, output in (("actor", self.action_dim), ("critic", 1)):
            sizes = (self.observation_dim, *self.hidden_sizes, output)
            for i, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
                result.extend([(f"{prefix}_w{i}", (left, right)), (f"{prefix}_b{i}", (right,))])
        if not self.discrete:
            result.append(("log_std", (self.action_dim,)))
        return result

    def parameters(self) -> dict[str, np.ndarray]:
        params, offset = {}, 0
        for name, shape in self.layout():
            count = int(np.prod(shape))
            params[name] = self.genome.values[offset:offset + count].reshape(shape)
            offset += count
        return params

    def forward(self, observation, prefix="actor"):
        value = np.asarray(observation, dtype=np.float32)
        params = self.parameters()
        for i in range(len(self.hidden_sizes) + 1):
            value = value @ params[f"{prefix}_w{i}"] + params[f"{prefix}_b{i}"]
            if i < len(self.hidden_sizes):
                value = np.tanh(value)
        return value

    def act(self, observation, rng, deterministic=True):
        output = self.forward(observation)
        if self.discrete:
            if deterministic:
                return int(np.argmax(output))
            probabilities = np.exp(output - output.max())
            return int(rng.choice(self.action_dim, p=probabilities / probabilities.sum()))
        if not deterministic:
            output = output + rng.normal(size=self.action_dim) * np.exp(self.parameters()["log_std"].clip(-5, 2))
        return (self.low + (np.tanh(output) + 1) * 0.5 * (self.high - self.low)).astype(np.float32)

    def clone(self, training_state=False):
        cloned = copy.deepcopy(self)
        if not training_state and hasattr(cloned, "training_state"):
            del cloned.training_state
        return cloned

    def save(self, path):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "genome.npz", values=self.genome.vector())
        spec = {"kind": "mlp", "schema": 1, "observation_dim": self.observation_dim,
                "hidden_sizes": self.hidden_sizes, "action_dim": self.action_dim,
                "discrete": self.discrete, "low": self.low.tolist(), "high": self.high.tolist(),
                "growth": asdict(self.growth) if self.growth is not None else None}
        spec.update(layer_ids=self.layer_ids, next_layer_id=self.next_layer_id)
        (folder / "agent.json").write_text(json.dumps(spec), encoding="utf-8")


@dataclass(frozen=True)
class MLPAgentFactory:
    hidden_sizes: tuple[int, ...] = (64, 64)
    growth: NeuralGrowth | None = None

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.hidden_sizes):
            raise ValueError("Hidden sizes must be positive integers")
        if self.growth is not None:
            if not isinstance(self.growth, NeuralGrowth):
                raise ValueError('growth must be a NeuralGrowth configuration or None')
            self.growth.validate_sizes(self.hidden_sizes)

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        validate_spaces(observation_space, action_space)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = int(action_space.n if discrete else action_space.shape[0])
        agent = MLPAgent(VectorGenome(np.zeros(1)), observation_space.shape[0], tuple(self.hidden_sizes),
                         action_dim, discrete, np.zeros(action_dim, dtype=np.float32) if discrete else action_space.low.copy(),
                         np.ones(action_dim, dtype=np.float32) if discrete else action_space.high.copy(), self.growth)
        parts = []
        for name, shape in agent.layout():
            value = rng.normal(0, np.sqrt(1 / shape[0]), shape) if "_w" in name else np.zeros(shape)
            if name.startswith("actor_w") and name.endswith(str(len(self.hidden_sizes))):
                value *= 0.1
            parts.append(value.reshape(-1))
        agent.genome = VectorGenome(np.concatenate(parts))
        return agent


def load_agent(path: str):
    folder = Path(path)
    spec = json.loads((folder / "agent.json").read_text(encoding="utf-8"))
    if spec.get("schema") != 1:
        raise ValueError("Unsupported agent schema")
    if spec["kind"] == "sb3":
        from .integrations.sb3 import SB3Agent
        return SB3Agent.load(folder)
    if spec["kind"] == "communicating":
        from .shared import CommunicatingAgent
        with np.load(folder / "genome.npz", allow_pickle=False) as data:
            return CommunicatingAgent(load_agent(folder / "task"), VectorGenome(data["values"]),
                                      spec["observation_dim"], spec["message_dimension"])
    with np.load(folder / "genome.npz", allow_pickle=False) as data:
        genome = VectorGenome(data["values"])
    if spec["kind"] == "vector":
        return VectorAgent(genome)
    if spec["kind"] == "graph":
        from .graph import GraphAgent, GraphGrowth
        return GraphAgent(genome, spec['observation_dim'], spec['action_dim'], spec['discrete'],
            np.array(spec['low'], np.float32), np.array(spec['high'], np.float32),
            tuple(spec['node_ids']), tuple(tuple(edge) for edge in spec['edges']), spec['next_node_id'],
            spec['settling_steps'], GraphGrowth(**spec['growth']) if spec.get('growth') is not None else None)
    if spec["kind"] != "mlp":
        raise ValueError("Unknown agent kind")
    return MLPAgent(genome, spec["observation_dim"], tuple(spec["hidden_sizes"]), spec["action_dim"],
                    spec["discrete"], np.array(spec["low"], np.float32), np.array(spec["high"], np.float32),
                    NeuralGrowth(**spec["growth"]) if spec.get("growth") is not None else None,
                    tuple(spec['layer_ids']) if 'layer_ids' in spec else None, spec.get('next_layer_id', 0))


def architecture_groups(agents):
    """Population indices grouped by compatible parameter shapes and spaces."""
    from .graph import GraphAgent
    if not agents or any(not isinstance(agent, (MLPAgent, GraphAgent)) for agent in agents):
        raise ValueError('Batched policies require MLPAgentFactory or GraphAgentFactory agents')
    groups = {}
    for i, agent in enumerate(agents):
        architecture = ((len(agent.node_ids), len(agent.edges), agent.settling_steps)
                        if isinstance(agent, GraphAgent) else agent.hidden_sizes)
        key = (type(agent), agent.observation_dim, architecture, agent.action_dim, agent.discrete,
               tuple(agent.low), tuple(agent.high))
        groups.setdefault(key, []).append(i)
    return list(groups.values())
