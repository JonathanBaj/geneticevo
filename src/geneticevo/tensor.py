"""Optional population inference and batched environments."""
from __future__ import annotations

import copy
from typing import Protocol

import numpy as np
import torch
from torch import nn

from .agents import MLPAgent, architecture_groups
from .core import Evaluation


def resolve_device(device):
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid PyTorch device {device!r}") from error
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Supported policy devices are cpu and cuda")
    if resolved.type == "cuda" and (not torch.cuda.is_available() or
            (resolved.index is not None and resolved.index >= torch.cuda.device_count())):
        raise ValueError("CUDA was requested but is unavailable. Install a compatible CUDA PyTorch build and driver, or use device='cpu'.")
    return resolved


class BatchedPolicy(nn.Module):
    """Weights [P, input, output], observations [P, E, observation_dim]."""
    def __init__(self, agents, device="cpu"):
        super().__init__()
        if not agents or any(not isinstance(x, MLPAgent) for x in agents):
            raise ValueError("Batched policies require MLPAgentFactory agents")
        first = agents[0]
        signature = (first.observation_dim, first.hidden_sizes, first.action_dim, first.discrete)
        for agent in agents:
            if (agent.observation_dim, agent.hidden_sizes, agent.action_dim, agent.discrete) != signature or not (
                    np.array_equal(first.low, agent.low) and np.array_equal(first.high, agent.high)):
                raise ValueError("Batched individuals must share architecture and action bounds")
        self.device = resolve_device(device)
        self.population_size, self.hidden_sizes = len(agents), first.hidden_sizes
        self.observation_dim, self.action_dim, self.discrete = first.observation_dim, first.action_dim, first.discrete
        self.params = nn.ParameterDict({key: nn.Parameter(torch.tensor(np.stack([x.parameters()[key] for x in agents]), device=self.device))
                                        for key, _ in first.layout()})
        self.register_buffer("low", torch.tensor(first.low, device=self.device))
        self.register_buffer("high", torch.tensor(first.high, device=self.device))

    def forward(self, observations, prefix="actor"):
        value = observations
        for layer in range(len(self.hidden_sizes) + 1):
            value = torch.bmm(value, self.params[f"{prefix}_w{layer}"]) + self.params[f"{prefix}_b{layer}"][:, None, :]
            if layer < len(self.hidden_sizes):
                value = torch.tanh(value)
        return value

    def values(self, observations):
        return self(observations, "critic").squeeze(-1)

    def deterministic_actions(self, observations):
        output = self(observations)
        return output.argmax(-1) if self.discrete else self.low + (torch.tanh(output) + 1) * (self.high - self.low) / 2

    def export(self, agents):
        arrays = {key: value.detach().cpu().numpy() for key, value in self.params.items()}
        for index, agent in enumerate(agents):
            slots = self.parameter_slots(agent, index)
            agent.genome.assign(np.concatenate([arrays[slots[key][0]][slots[key][1]].reshape(-1)
                                                for key, _ in agent.layout()]))

    def parameter_slots(self, agent, index):
        """Map portable parameter names to population tensor slots."""
        return {name: (name, (index, *[slice(0, n) for n in shape])) for name, shape in agent.layout()}


class GraphBatchedPolicy(BatchedPolicy):
    """Sparse edge gathers/scatter-adds, differentiable through settling steps.

    Only existing edges have weights and gradients. Members may have different
    connectivity while sharing node/edge counts and spaces.
    """
    def __init__(self, agents, device='cpu'):
        nn.Module.__init__(self)
        from .graph import GraphAgent
        if any(not isinstance(a, GraphAgent) for a in agents) or len(architecture_groups(agents)) != 1:
            raise ValueError('Graph policies require compatible GraphAgent instances')
        first = agents[0]
        self.device = resolve_device(device)
        self.population_size = len(agents)
        self.observation_dim, self.action_dim, self.discrete = first.observation_dim, first.action_dim, first.discrete
        self.node_count, self.settling_steps = len(first.node_ids), first.settling_steps
        self.params = nn.ParameterDict({key: nn.Parameter(torch.tensor(
            np.stack([a.parameters()[key] for a in agents]), device=self.device)) for key, _ in first.layout()})
        self.register_buffer('low', torch.tensor(first.low, device=self.device))
        self.register_buffer('high', torch.tensor(first.high, device=self.device))
        indices = [a.edge_indices() for a in agents]
        self.register_buffer('sources', torch.tensor(np.stack([x[0] for x in indices]), device=self.device))
        self.register_buffer('targets', torch.tensor(np.stack([x[1] for x in indices]), device=self.device))

    def forward(self, observations, prefix='actor'):
        state = observations.new_zeros((*observations.shape[:-1], self.node_count))
        source = self.sources[:, None, :].expand(-1, observations.shape[1], -1)
        target = self.targets[:, None, :].expand_as(source)
        for _ in range(self.settling_steps):
            values = torch.cat((observations, state), dim=-1)
            messages = values.gather(-1, source) * self.params['weights'][:, None, :]
            updated = self.params['biases'][:, None, :].expand_as(state).scatter_add(-1, target, messages)
            # Linear readouts keep the critic unbounded; hidden neurons use tanh.
            state = torch.cat((updated[..., :self.action_dim + 1],
                               torch.tanh(updated[..., self.action_dim + 1:])), dim=-1)
        return state[..., :self.action_dim] if prefix == 'actor' else state[..., self.action_dim:self.action_dim + 1]


def build_policy(agents, device='cpu'):
    from .graph import GraphAgent
    return GraphBatchedPolicy(agents, device) if isinstance(agents[0], GraphAgent) else BatchedPolicy(agents, device)


class TensorEnvironment(Protocol):
    """step returns *pre-reset* next observations. reset_done performs resets.

    All tensors lead with [P,E]. Inactive lanes must not step or earn rewards.
    Implementations also expose horizon, discrete, observation_dim, action_dim.
    """
    def reset(self, scenarios=None): ...
    def step(self, actions, active=None): ...
    def reset_done(self, mask): ...
    def close(self): ...


class TensorReach1D:
    def __init__(self, population_size, env_copies, device="cpu", seed=0, horizon=32, continuous=False, radius=1.0):
        self.device = resolve_device(device)
        self.population_size, self.env_copies, self.horizon = population_size, env_copies, horizon
        self.discrete, self.action_dim, self.observation_dim = not continuous, 3 if not continuous else 1, 1
        self.radius = radius
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.position = torch.zeros((population_size, env_copies, 1), device=self.device)
        self.t = torch.zeros((population_size, env_copies), device=self.device, dtype=torch.long)
    def reset(self, scenarios=None):
        if scenarios is not None:
            if len(scenarios) != self.env_copies:
                raise ValueError("One scenario per environment copy is required")
            positions = []
            for scenario in scenarios:
                if set(scenario.options) - {"position", "radius"}:
                    raise ValueError("Unsupported TensorReach1D scenario option")
                radius = scenario.options.get("radius", 1.0)
                random_position = np.random.default_rng(scenario.seed).uniform(-radius, radius)
                positions.append(scenario.options.get("position", random_position))
            self.position.copy_(torch.tensor(positions, dtype=torch.float32, device=self.device)[None, :, None])
        else:
            self.position.uniform_(-self.radius, self.radius, generator=self.generator)
        self.t.zero_()
        return self.position.clone()

    def step(self, actions, active=None):
        if active is None:
            active = torch.ones_like(self.t, dtype=torch.bool)
        move = actions.float().unsqueeze(-1) - 1 if self.discrete else actions
        updated = (self.position + 0.1 * move).clamp(-10, 10)
        self.position.copy_(torch.where(active.unsqueeze(-1), updated, self.position))
        self.t.add_(active.long())
        rewards = -(self.position.squeeze(-1)**2) * active
        terminated = torch.zeros_like(active)
        truncated = (self.t >= self.horizon) & active
        return self.position.clone(), rewards, terminated, truncated

    def reset_done(self, mask):
        replacement = torch.empty_like(self.position).uniform_(-self.radius, self.radius, generator=self.generator)
        self.position.copy_(torch.where(mask.unsqueeze(-1), replacement, self.position))
        self.t.masked_fill_(mask, 0)
        return self.position.clone()

    def close(self):
        pass


class GymTensorBridge:
    """Hybrid execution: Python simulations stay on CPU; policies may use CUDA."""
    def __init__(self, env_factory, population_size, env_copies, device="cpu", seed=0):
        self.device = resolve_device(device)
        self.population_size, self.env_copies = population_size, env_copies
        self.envs = [[env_factory() for _ in range(env_copies)] for _ in range(population_size)]
        if len({id(env.unwrapped) for row in self.envs for env in row}) != population_size * env_copies:
            self.close()
            raise ValueError("env_factory must return independent environments")
        env = self.envs[0][0]
        from gymnasium import spaces
        from .agents import validate_spaces
        validate_spaces(env.observation_space, env.action_space)
        self.discrete = isinstance(env.action_space, spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.horizon = None
        self.seed = seed
        self.observations = np.zeros((population_size, env_copies, self.observation_dim), dtype=np.float32)

    def reset(self, scenarios=None):
        for p, row in enumerate(self.envs):
            for e, env in enumerate(row):
                scenario = scenarios[e] if scenarios else None
                member_seed = self.seed + p * self.env_copies
                self.observations[p, e], _ = env.reset(seed=scenario.seed if scenario else (member_seed + e) % 2**32,
                                                     options=copy.deepcopy(scenario.options) if scenario else None)
        return torch.tensor(self.observations, device=self.device)

    def step(self, actions, active=None):
        actions = actions.detach().cpu().numpy()
        mask = active.cpu().numpy() if active is not None else np.ones((self.population_size, self.env_copies), bool)
        rewards = np.zeros(mask.shape, np.float32)
        terminated, truncated = np.zeros_like(mask), np.zeros_like(mask)
        for p, row in enumerate(self.envs):
            for e, env in enumerate(row):
                if mask[p, e]:
                    action = int(actions[p, e]) if self.discrete else actions[p, e]
                    self.observations[p, e], rewards[p, e], terminated[p, e], truncated[p, e], _ = env.step(action)
        return (torch.tensor(self.observations, device=self.device), torch.tensor(rewards, device=self.device),
                torch.tensor(terminated, device=self.device), torch.tensor(truncated, device=self.device))

    def reset_done(self, mask):
        mask = mask.cpu().numpy()
        for p, row in enumerate(self.envs):
            for e, env in enumerate(row):
                if mask[p, e]:
                    self.observations[p, e], _ = env.reset()
        return torch.tensor(self.observations, device=self.device)

    def close(self):
        for row in self.envs:
            for env in row:
                env.close()


def make_tensor_environment(env_factory, tensor_env_factory, agents, copies, device, seed):
    env = (tensor_env_factory(len(agents), copies, device, seed) if tensor_env_factory else
           GymTensorBridge(env_factory, len(agents), copies, device, seed))
    first = agents[0]
    if (env.discrete, env.observation_dim, env.action_dim) != (first.discrete, first.observation_dim, first.action_dim):
        env.close()
        raise ValueError("Tensor environment and policy spaces differ")
    return env


@torch.no_grad()
def evaluate_batched(agents, scenarios, env_factory, tensor_env_factory=None, device="cpu", env_copies=1, max_steps=1000,
                     population_chunk_size=None):
    if population_chunk_size is not None:
        if isinstance(population_chunk_size, bool) or not isinstance(population_chunk_size, int) or population_chunk_size < 1:
            raise ValueError("population_chunk_size must be a positive integer")
        if len(agents) > population_chunk_size:
            return [score for start in range(0, len(agents), population_chunk_size)
                    for score in evaluate_batched(agents[start:start + population_chunk_size], scenarios,
                        env_factory, tensor_env_factory, device, env_copies, max_steps)]
    groups = architecture_groups(agents)
    if len(groups) > 1:
        evaluations = [None] * len(agents)
        for indices in groups:
            scores = evaluate_batched([agents[i] for i in indices], scenarios, env_factory,
                                       tensor_env_factory, device, env_copies, max_steps)
            for i, score in zip(indices, scores):
                evaluations[i] = score
        return evaluations
    policy = build_policy(agents, device)
    returns, lengths = [], []
    for start in range(0, len(scenarios), env_copies):
        chunk = scenarios[start:start + env_copies]
        env = make_tensor_environment(env_factory, tensor_env_factory, agents, len(chunk), device, chunk[0].seed)
        try:
            observation = env.reset(chunk)
            active = torch.ones((len(agents), len(chunk)), dtype=torch.bool, device=policy.device)
            total = torch.zeros_like(active, dtype=torch.float32)
            steps = torch.zeros_like(active, dtype=torch.long)
            # Fixed shape native loop avoids device scalar reads on every step.
            limit = min(max_steps, env.horizon) if env.horizon is not None else max_steps
            for _ in range(limit):
                if isinstance(env, GymTensorBridge) and not bool(active.any()):
                    break
                observation, rewards, terminated, truncated = env.step(policy.deterministic_actions(observation), active)
                total += rewards * active
                steps += active.long()
                active &= ~(terminated | truncated)
            returns.append(total.cpu().numpy())
            lengths.append(steps.cpu().numpy())
        finally:
            env.close()
    returns, lengths = np.concatenate(returns, axis=1), np.concatenate(lengths, axis=1)
    return [Evaluation(float(returns[p].mean()), returns[p].tolist(), lengths[p].tolist(),
                       int(lengths[p].sum()), context=list(scenarios)) for p in range(len(agents))]
