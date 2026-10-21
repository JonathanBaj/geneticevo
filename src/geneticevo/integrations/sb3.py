"""Independent SB3 PPO/DQN policies, fresh children, and resumable elites.

No VecNormalize or normalization wrappers: rejecting them avoids silently
losing running statistics. Evaluation uses a separate raw Gymnasium env.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..agents import validate_spaces
from ..core import LearnReport


@contextmanager
def isolated_rng(seed=None, state=None):
    """SB3 uses global RNGs; scope them to one individual and restore callers."""
    previous = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if state is None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.random.set_rng_state(state[2])
            if len(state) > 3 and state[3] is not None:
                torch.cuda.set_rng_state_all(state[3])
        yield
    finally:
        random.setstate(previous[0])
        np.random.set_state(previous[1])
        torch.random.set_rng_state(previous[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


class SpaceEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = copy.deepcopy(observation_space)
        self.action_space = copy.deepcopy(action_space)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("SpaceEnv only constructs policies; attach a real environment before training")


def algorithm(name):
    if name not in ("PPO", "DQN"):
        raise ValueError("Implemented SB3 algorithms: PPO, DQN")
    return {"PPO": PPO, "DQN": DQN}[name]


class SB3Genome:
    def __init__(self, agent):
        self.agent = agent

    def _parameters(self):
        # DQN target network is algorithm state, not an independently mutated gene.
        return [p for name, p in self.agent.model.policy.named_parameters() if "q_net_target" not in name]

    def vector(self):
        return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in self._parameters()]).copy()

    def assign(self, values):
        values = np.asarray(values, dtype=np.float32)
        parameters = self._parameters()
        if values.shape != (sum(p.numel() for p in parameters),) or not np.isfinite(values).all():
            raise ValueError("Incompatible or non-finite SB3 genome")
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(torch.as_tensor(values[offset:offset + parameter.numel()].reshape(parameter.shape), device=parameter.device))
                offset += parameter.numel()
        if self.agent.algorithm_name == "DQN":
            self.agent.model.q_net_target.load_state_dict(self.agent.model.q_net.state_dict())

    def clone(self):
        from ..core import VectorGenome
        return VectorGenome(self.vector())


class SB3Agent:
    def __init__(self, algorithm_name, observation_space, action_space, kwargs, seed, device="cpu", env_copies=1):
        from ..tensor import resolve_device
        resolve_device(device)
        validate_spaces(observation_space, action_space)
        if algorithm_name == "DQN" and not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("SB3 DQN requires Discrete actions")
        self.algorithm_name, self.kwargs = algorithm_name, copy.deepcopy(kwargs)
        self.observation_space, self.action_space = observation_space, action_space
        self.seed, self.device = seed, device
        self.env_copies = env_copies
        with isolated_rng(seed):
            env = DummyVecEnv([lambda: SpaceEnv(observation_space, action_space) for _ in range(env_copies)])
            self.model = algorithm(algorithm_name)("MlpPolicy", env,
                                                    seed=seed, device=device, verbose=0, **kwargs)
        self.rng_state = None

    @property
    def genome(self):
        return SB3Genome(self)

    def act(self, observation, rng, deterministic=True):
        if not deterministic:
            with isolated_rng(int(rng.integers(0, 2**31))):
                action, _ = self.model.predict(observation, deterministic=False)
        else:
            action, _ = self.model.predict(observation, deterministic=True)
        return int(action) if isinstance(self.action_space, gym.spaces.Discrete) else action

    def clone(self, training_state=False):
        if training_state:
            agent = object.__new__(SB3Agent)
            agent.__setstate__(self.__getstate__())
            return agent
        agent = SB3Agent(self.algorithm_name, self.observation_space, self.action_space, self.kwargs, self.seed, self.device, self.env_copies)
        agent.genome.assign(self.genome.vector())
        return agent

    def __getstate__(self):
        stream = io.BytesIO()
        self.model.save(stream)
        return {"algorithm_name": self.algorithm_name, "kwargs": self.kwargs, "seed": self.seed,
                "device": self.device, "observation_space": self.observation_space, "action_space": self.action_space,
                "env_copies": self.env_copies,
                "model_bytes": stream.getvalue(), "replay_buffer": copy.deepcopy(getattr(self.model, "replay_buffer", None)),
                "rng_state": copy.deepcopy(self.rng_state)}

    def __setstate__(self, state):
        state = copy.deepcopy(state)
        model_bytes, replay = state.pop("model_bytes"), state.pop("replay_buffer")
        self.__dict__.update(state)
        with isolated_rng(self.seed):
            self.model = algorithm(self.algorithm_name).load(io.BytesIO(model_bytes), device=self.device, force_reset=True)
        if replay is not None:
            self.model.replay_buffer = replay

    def save(self, path):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / "model.zip")
        (folder / "agent.json").write_text(json.dumps({"kind": "sb3", "schema": 1,
            "algorithm_name": self.algorithm_name, "kwargs": self.kwargs, "seed": self.seed, "env_copies": self.env_copies}), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        spec = json.loads((folder / "agent.json").read_text(encoding="utf-8"))
        agent = object.__new__(cls)
        agent.algorithm_name, agent.kwargs, agent.seed, agent.device = spec["algorithm_name"], spec["kwargs"], spec["seed"], "cpu"
        agent.env_copies = spec["env_copies"]
        with isolated_rng(agent.seed):
            agent.model = algorithm(agent.algorithm_name).load(folder / "model.zip", device="cpu")
        agent.observation_space, agent.action_space = agent.model.observation_space, agent.model.action_space
        agent.rng_state = None
        return agent


@dataclass(frozen=True)
class SB3AgentFactory:
    algorithm: str = "PPO"
    kwargs: dict = field(default_factory=dict)
    env_copies: int = 1

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        kwargs = ({"n_steps": 32, "batch_size": 32, "n_epochs": 2} if self.algorithm == "PPO" else
                  {"buffer_size": 10000, "learning_starts": 32, "batch_size": 32, "train_freq": 1}) | self.kwargs
        if kwargs.get("use_sde", False):
            raise ValueError("State-dependent exploration is not currently supported")
        return SB3Agent(self.algorithm, observation_space, action_space, kwargs, int(rng.integers(0, 2**31)), device, self.env_copies)


def check_raw_environment(env):
    current = env
    while current is not None:
        if isinstance(current, VecNormalize) or type(current).__name__ in ("NormalizeObservation", "NormalizeReward"):
            raise ValueError("Normalization wrappers are unsupported; supply an explicit frozen-statistics adapter")
        current = getattr(current, "env", None)


@dataclass(frozen=True)
class SB3Learner:
    env_copies: int = 1

    def __post_init__(self):
        if self.env_copies < 1:
            raise ValueError("env_copies must be positive")

    def validate(self, agent, env_factory, backend):
        if not isinstance(agent, SB3Agent) or backend not in ("serial", "parallel"):
            raise ValueError("SB3 learning requires SB3AgentFactory and serial/parallel evaluation")
        if agent.env_copies != self.env_copies:
            raise ValueError("SB3AgentFactory.env_copies must match learner.env_copies")
        env = env_factory()
        try:
            check_raw_environment(env)
            if env.observation_space != agent.observation_space or env.action_space != agent.action_space:
                raise ValueError("Training and policy spaces differ")
        finally:
            env.close()

    def prepare(self, agent, seed):
        if not isinstance(agent, SB3Agent):
            raise ValueError("SB3Learner requires SB3Agent")

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        self.validate(agent, env_factory, "serial")
        self.prepare(agent, action_seed)
        environments = DummyVecEnv([env_factory for _ in range(self.env_copies)])
        if len({id(x.unwrapped) for x in environments.envs}) != self.env_copies:
            environments.close()
            raise ValueError("env_factory must create independent environments")
        environments.seed(environment_seed)
        model = agent.model
        before, updates = model.num_timesteps, model._n_updates
        try:
            model.set_env(environments, force_reset=True)
            with isolated_rng(action_seed, agent.rng_state):
                model.learn(total_timesteps=budget, reset_num_timesteps=False)
                agent.rng_state = rng_snapshot()
            metrics = {key.removeprefix("train/"): float(value) for key, value in model.logger.name_to_value.items()
                       if key.startswith("train/") and isinstance(value, (float, int, np.number)) and np.isfinite(value)}
            actual = model.num_timesteps - before
            if all(hasattr(env.unwrapped, "total_agent_transitions") for env in environments.envs):
                metrics["focal_interactions"] = actual
                metrics["world_steps"] = sum(env.unwrapped.total_world_steps for env in environments.envs)
                actual = sum(env.unwrapped.total_agent_transitions for env in environments.envs)
                metrics["all_agent_transitions"] = actual
            return LearnReport(actual, model._n_updates - updates, budget, metrics)
        finally:
            environments.close()
            # Checkpoints never retain live simulations or pretend to resume their state.
            model.env = None
            model._last_obs = None

    def export(self, agent):
        return agent.genome.clone()

    def clone(self, agent, training_state=False):
        return agent.clone(training_state)

    def state_dict(self, agent):
        return agent.__getstate__()

    def load_state_dict(self, agent, state):
        agent.__setstate__(state)

