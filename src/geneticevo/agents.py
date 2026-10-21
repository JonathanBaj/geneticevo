"""NumPy vector and MLP agents; importing these never imports PyTorch."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from .core import VectorGenome


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
                "discrete": self.discrete, "low": self.low.tolist(), "high": self.high.tolist()}
        (folder / "agent.json").write_text(json.dumps(spec), encoding="utf-8")


@dataclass(frozen=True)
class MLPAgentFactory:
    hidden_sizes: tuple[int, ...] = (64, 64)

    def __post_init__(self):
        if any(x < 1 for x in self.hidden_sizes):
            raise ValueError("Hidden sizes must be positive")

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        validate_spaces(observation_space, action_space)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = int(action_space.n if discrete else action_space.shape[0])
        agent = MLPAgent(VectorGenome(np.zeros(1)), observation_space.shape[0], tuple(self.hidden_sizes),
                         action_dim, discrete, np.zeros(action_dim, dtype=np.float32) if discrete else action_space.low.copy(),
                         np.ones(action_dim, dtype=np.float32) if discrete else action_space.high.copy())
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
    if spec["kind"] != "mlp":
        raise ValueError("Unknown agent kind")
    return MLPAgent(genome, spec["observation_dim"], tuple(spec["hidden_sizes"]), spec["action_dim"],
                    spec["discrete"], np.array(spec["low"], np.float32), np.array(spec["high"], np.float32))

