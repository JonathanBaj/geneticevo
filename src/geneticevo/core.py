"""Small extension contracts, genetic data, configuration, and random streams."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class Genome(Protocol):
    def clone(self) -> Genome: ...
    def vector(self) -> np.ndarray: ...
    def assign(self, values: np.ndarray) -> None: ...


class Agent(Protocol):
    genome: Genome
    def act(self, observation: Any, rng: np.random.Generator, deterministic: bool = True) -> Any: ...
    def clone(self, training_state: bool = False) -> Agent: ...
    def save(self, path: str) -> None: ...


class AgentFactory(Protocol):
    def __call__(self, rng: np.random.Generator, observation_space: Any = None,
                 action_space: Any = None, device: str = "cpu") -> Agent: ...


@dataclass
class VectorGenome:
    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32).reshape(-1).copy()
        if not np.isfinite(self.values).all():
            raise ValueError("Genome must contain finite values")

    def clone(self) -> VectorGenome:
        return copy.deepcopy(self)

    def vector(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != self.values.shape or not np.isfinite(values).all():
            raise ValueError("Incompatible or non-finite genome")
        self.values[:] = values


@dataclass(frozen=True)
class Scenario:
    seed: int
    options: dict[str, Any] = field(default_factory=dict)
    distribution: str = "validation"


@dataclass
class Evaluation:
    fitness: float
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    interactions: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    descriptor: list[float] = field(default_factory=list)
    context: list[Scenario] = field(default_factory=list)
    world_steps: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fitness):
            raise ValueError("Fitness must be finite")


class Evaluator(Protocol):
    def evaluate(self, agent: Agent, scenarios: Sequence[Scenario], action_seed: int) -> Evaluation: ...


@dataclass
class LearnReport:
    interactions: int = 0
    updates: int = 0
    requested: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class Learner(Protocol):
    """An algorithm owns its state on the agent, never on a shared learner object.

    prepare/learn/export/clone/state_dict/load_state_dict are algorithm-neutral:
    replay-based learners need not expose a rollout or an epoch.
    """
    def validate(self, agent: Agent, env_factory: Callable, backend: str) -> None: ...
    def prepare(self, agent: Agent, seed: int) -> None: ...
    def learn(self, agent: Agent, env_factory: Callable, budget: int,
              environment_seed: int, action_seed: int) -> LearnReport: ...
    def export(self, agent: Agent) -> Genome: ...
    def clone(self, agent: Agent, training_state: bool = False) -> Agent: ...
    def state_dict(self, agent: Agent) -> dict: ...
    def load_state_dict(self, agent: Agent, state: dict) -> None: ...


@dataclass
class Individual:
    id: str
    agent: Agent
    born: int
    parents: tuple[str, ...] = ()
    mutation: dict[str, Any] = field(default_factory=dict)
    evaluation: Evaluation | None = None


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 32
    generations: int = 100
    elite_count: int = 2
    immigrant_count: int = 0
    selection: str = "tournament"
    tournament_size: int = 3
    maximize: bool = True
    evaluation_episodes: int = 5
    archive_size: int = 3
    max_episode_steps: int = 1000
    seed: int = 42
    inheritance: str = "lamarckian"
    learning_budget: int = 0
    env_copies: int = 1
    measure_pre_learning: bool = False
    elite_learning: bool = True
    target_fitness: float | None = None
    stagnation_generations: int | None = None
    time_limit_seconds: float | None = None
    max_interactions: int | None = None
    workers: int = 2
    diversity_pairs: int = 0
    output_dir: str | None = None
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        for name in ("population_size", "generations", "tournament_size", "evaluation_episodes",
                     "archive_size", "max_episode_steps", "env_copies", "workers", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("elite_count", "immigrant_count", "learning_budget", "diversity_pairs", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.elite_count + self.immigrant_count > self.population_size:
            raise ValueError("Elites and immigrants exceed population size")
        if self.selection not in ("tournament", "rank"):
            raise ValueError("selection must be tournament or rank; pass selection_hook for custom selection")
        if self.inheritance not in ("lamarckian", "baldwinian"):
            raise ValueError("inheritance must be lamarckian or baldwinian")
        for name in ("stagnation_generations", "max_interactions"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.target_fitness is not None and not np.isfinite(self.target_fitness):
            raise ValueError("target_fitness must be finite")

    def as_dict(self) -> dict:
        return asdict(self)


# Names are stable, and adding a stream never changes existing stream seeds.
STREAMS = {"initialization": 1, "mutation": 2, "training": 3, "actions": 4,
           "validation": 5, "test": 6, "archive": 7, "selection": 8, "diversity": 9, "growth": 10}


def seed_for(seed: int, stream: str, *coordinates: int) -> int:
    return int(np.random.SeedSequence([seed, STREAMS[stream], *coordinates]).generate_state(1)[0])


def scenarios_for(seed: int, stream: str, count: int, generation: int = 0,
                  sampler: Callable | None = None) -> list[Scenario]:
    rng = np.random.default_rng(seed_for(seed, stream, generation))
    if sampler is not None:
        result = list(sampler(rng, count, generation))
        if len(result) != count or any(not isinstance(x, Scenario) for x in result):
            raise ValueError("Scenario sampler must return the requested number of Scenario objects")
        return result
    return [Scenario(int(rng.integers(0, 2**31)), distribution=stream) for _ in range(count)]

