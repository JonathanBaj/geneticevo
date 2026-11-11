"""Optional population tensors, independent distributions, and batched environments."""
from __future__ import annotations

import math
import copy
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


class IndependentRNG:
    """Independent generator state per individual, including minibatch shuffles.

    Only random filling uses a short population loop; forward/backward,
    simulation, losses, clipping, and optimizer operations are batched.
    """
    def __init__(self, seeds, device):
        self.device = resolve_device(device)
        self.generators = [torch.Generator(device=self.device).manual_seed(int(x)) for x in seeds]

    def uniform(self, shape):
        return torch.stack([torch.rand(shape, device=self.device, generator=g) for g in self.generators])

    def normal(self, shape):
        return torch.stack([torch.randn(shape, device=self.device, generator=g) for g in self.generators])

    def permutations(self, size):
        return torch.stack([torch.randperm(size, device=self.device, generator=g) for g in self.generators])


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

    def sample(self, observations, rng, deterministic=False, *, reuse_output=False, compute_values=True):
        output = self(observations)
        if self.discrete:
            probabilities = output.softmax(-1)
            actions = output.argmax(-1) if deterministic else (rng.uniform(output.shape[1:2]).unsqueeze(-1) >= probabilities.cumsum(-1)).sum(-1).clamp_max(self.action_dim - 1)
            raw = actions
        else:
            std = self.params["log_std"].clamp(-5, 2).exp()[:, None, :]
            raw = output if deterministic else output + std * rng.normal(output.shape[1:])
            actions = self.low + (torch.tanh(raw) + 1) * (self.high - self.low) / 2
        log_prob, _ = self.log_prob_from_output(output, raw) if reuse_output else self.log_prob(observations, raw)
        return actions, raw, log_prob, self.values(observations) if compute_values else None

    def log_prob(self, observations, raw_actions):
        return self.log_prob_from_output(self(observations), raw_actions)

    def log_prob_from_output(self, output, raw_actions):
        if self.discrete:
            log_probs = output.log_softmax(-1)
            log_prob = log_probs.gather(-1, raw_actions.long().unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs.exp() * log_probs).sum(-1)
        else:
            log_std = self.params["log_std"].clamp(-5, 2)[:, None, :]
            normal_log_prob = -0.5 * (((raw_actions - output) / log_std.exp())**2 + 2 * log_std + math.log(2 * math.pi))
            log_jacobian = 2 * (math.log(2) - raw_actions - F.softplus(-2 * raw_actions))
            log_prob = (normal_log_prob - log_jacobian - torch.log((self.high - self.low) / 2)).sum(-1)
            # Analytic base-normal entropy regularization, explicitly not tanh entropy.
            entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1).expand_as(log_prob)
        return log_prob, entropy

    def export(self, agents):
        arrays = {key: value.detach().cpu().numpy() for key, value in self.params.items()}
        for index, agent in enumerate(agents):
            slots = self.parameter_slots(agent, index)
            agent.genome.assign(np.concatenate([arrays[slots[key][0]][slots[key][1]].reshape(-1)
                                                for key, _ in agent.layout()]))

    def parameter_slots(self, agent, index):
        """Map portable active parameter names to padded tensor storage."""
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


class MaskedBatchedPolicy(BatchedPolicy):
    """Fixed capacity per layer; each individual's active prefix can grow in place.

    Inactive neurons have zero activations and gradients. Portable agents and
    Adam exports contain only active parameters, so checkpoints stay compatible.
    """
    def __init__(self, agents, device="cpu", capacity=None):
        if not agents:
            raise ValueError("Batched policies require agents")
        depth = len(agents[0].hidden_sizes)
        if any(len(a.hidden_sizes) != depth for a in agents):
            raise ValueError("Masked policies require a common hidden layer count")
        required = tuple(max(a.growth.max_hidden_size if a.growth else a.hidden_sizes[i]
                             for a in agents) for i in range(depth))
        capacity = required if capacity is None else tuple(capacity)
        if len(capacity) != depth or any(isinstance(c, bool) or not isinstance(c, int) or c < r
                                       for c, r in zip(capacity, required)):
            raise ValueError("growth_capacity must cover every layer's configured maximum width")
        padded = []
        for agent in agents:
            proxy = copy.copy(agent)
            proxy.hidden_sizes = capacity
            source = agent.parameters()
            parts = []
            for name, shape in proxy.layout():
                values = np.zeros(shape, np.float32)
                values[tuple(slice(0, n) for n in source[name].shape)] = source[name]
                parts.append(values.reshape(-1))
            from .core import VectorGenome
            proxy.genome = VectorGenome(np.concatenate(parts))
            padded.append(proxy)
        super().__init__(padded, device)
        for layer, size in enumerate(capacity):
            widths = torch.tensor([a.hidden_sizes[layer] for a in agents], device=self.device)
            self.register_buffer(f"mask_{layer}", (torch.arange(size, device=self.device)[None, :] < widths[:, None])[:, None, :])

    def forward(self, observations, prefix="actor"):
        value = observations
        for layer in range(len(self.hidden_sizes) + 1):
            value = torch.bmm(value, self.params[f"{prefix}_w{layer}"]) + self.params[f"{prefix}_b{layer}"][:, None, :]
            if layer < len(self.hidden_sizes):
                value = torch.tanh(value) * getattr(self, f"mask_{layer}")
        return value

    @torch.no_grad()
    def activate(self, index, agent, previous_sizes):
        """Import only newly initialized slots from resize_hidden_sizes.

        The CPU agent's overlapping genes may be stale during a rollout; the
        trained overlap stays on device. New outgoing rows are zero, exactly as
        in MLPAgent.resize_hidden_sizes. No parameter objects are replaced.
        """
        old = copy.copy(agent)
        old.hidden_sizes = previous_sizes
        old_shapes = dict(old.layout())
        for name, value in agent.parameters().items():
            before = old_shapes[name]
            target = self.params[name][index]
            if value.ndim == 2:
                if value.shape[0] > before[0]:
                    target[before[0]:value.shape[0], :value.shape[1]].copy_(
                        torch.as_tensor(value[before[0]:, :].copy(), device=self.device))
                if value.shape[1] > before[1]:
                    target[:before[0], before[1]:value.shape[1]].copy_(
                        torch.as_tensor(value[:before[0], before[1]:].copy(), device=self.device))
            elif value.shape[0] > before[0]:
                target[before[0]:value.shape[0]].zero_()
        for layer, width in enumerate(agent.hidden_sizes):
            getattr(self, f"mask_{layer}")[index, 0].copy_(torch.arange(self.hidden_sizes[layer], device=self.device) < width)

    def export(self, agents):
        arrays = {key: value.detach().cpu().numpy() for key, value in self.params.items()}
        for index, agent in enumerate(agents):
            agent.genome.assign(np.concatenate([arrays[key][(index, *[slice(0, n) for n in shape])].reshape(-1)
                                                for key, shape in agent.layout()]))


def required_mlp_capacity(agents):
    """Cover reachable depth and widths for a bounded training/evaluation batch."""
    depths = [max(len(a.hidden_sizes), a.growth.max_hidden_layers
                  if a.growth and (a.growth.layer_add_probability or a.growth.layer_mutation_probability) else 0)
              for a in agents]
    return tuple(max(a.growth.max_hidden_size if a.growth and layer < depth else
                     a.hidden_sizes[layer] if layer < len(a.hidden_sizes) else 1
                     for a, depth in zip(agents, depths)) for layer in range(max(depths)))


class DynamicBatchedPolicy(BatchedPolicy):
    """Masked hidden slots plus a separate output head for each individual.

    Members select their readout at their own active depth, including depth zero.
    Padding has zero gradients. Depth changes import just one member into the
    same fixed tensors; peers' weights/moments and live environments stay intact.
    """
    def __init__(self, agents, device='cpu', capacity=None):
        nn.Module.__init__(self)
        if len(masked_architecture_groups(agents)) != 1:
            raise ValueError('Dynamic policies require matching spaces and action bounds')
        required = required_mlp_capacity(agents)
        capacity = required if capacity is None else tuple(capacity)
        if len(capacity) < len(required) or any(isinstance(c, bool) or not isinstance(c, int) or c < 1
                for c in capacity) or any(c < r for c, r in zip(capacity, required)):
            raise ValueError('growth_capacity must cover reachable hidden depth and widths')
        first = agents[0]
        self.device = resolve_device(device)
        self.population_size, self.hidden_sizes = len(agents), capacity
        self.observation_dim, self.action_dim, self.discrete = first.observation_dim, first.action_dim, first.discrete
        self.feature_size = max((self.observation_dim, *capacity))
        layout = {}
        for prefix, output in (('actor', self.action_dim), ('critic', 1)):
            for layer, (left, right) in enumerate(zip((self.observation_dim, *capacity), capacity)):
                layout[f'{prefix}_w{layer}'] = (left, right)
                layout[f'{prefix}_b{layer}'] = (right,)
            layout[f'{prefix}_out_w'] = (self.feature_size, output)
            layout[f'{prefix}_out_b'] = (output,)
        if not self.discrete:
            layout['log_std'] = (self.action_dim,)
        self.params = nn.ParameterDict({name: nn.Parameter(torch.zeros((len(agents), *shape), device=self.device))
                                        for name, shape in layout.items()})
        self.register_buffer('low', torch.tensor(first.low, device=self.device))
        self.register_buffer('high', torch.tensor(first.high, device=self.device))
        self.register_buffer('depths', torch.zeros(len(agents), dtype=torch.long, device=self.device))
        for layer, width in enumerate(capacity):
            self.register_buffer(f'mask_{layer}', torch.zeros((len(agents), 1, width), device=self.device))
        for index, agent in enumerate(agents):
            self.activate(index, agent, None)

    def parameter_slots(self, agent, index):
        slots = {}
        for name, shape in agent.layout():
            key = name
            if name != 'log_std':
                prefix, suffix = name.split('_')
                if int(suffix[1:]) == len(agent.hidden_sizes):
                    key = f'{prefix}_out_{suffix[0]}'
            slots[name] = (key, (index, *[slice(0, n) for n in shape]))
        return slots

    @torch.no_grad()
    def activate(self, index, agent, previous_sizes):
        for parameter in self.params.values():
            parameter[index].zero_()
        slots = self.parameter_slots(agent, index)
        for name, value in agent.parameters().items():
            key, slices = slots[name]
            self.params[key][slices].copy_(torch.as_tensor(value.copy(), device=self.device))
        self.depths[index] = len(agent.hidden_sizes)
        for layer, capacity in enumerate(self.hidden_sizes):
            width = agent.hidden_sizes[layer] if layer < len(agent.hidden_sizes) else 0
            getattr(self, f'mask_{layer}')[index, 0].copy_(torch.arange(capacity, device=self.device) < width)

    def forward(self, observations, prefix='actor'):
        value = observations
        selected = F.pad(observations, (0, self.feature_size - self.observation_dim))
        for layer, width in enumerate(self.hidden_sizes):
            value = torch.tanh(torch.bmm(value, self.params[f'{prefix}_w{layer}']) +
                               self.params[f'{prefix}_b{layer}'][:, None, :]) * getattr(self, f'mask_{layer}')
            selected = torch.where((self.depths == layer + 1)[:, None, None],
                F.pad(value, (0, self.feature_size - width)), selected)
        return torch.bmm(selected, self.params[f'{prefix}_out_w']) + self.params[f'{prefix}_out_b'][:, None, :]


def build_masked_policy(agents, device='cpu', capacity=None):
    dynamic = (len({len(a.hidden_sizes) for a in agents}) > 1 or
               (capacity is not None and len(capacity) != len(agents[0].hidden_sizes)) or
               any(a.growth and a.growth.dynamic_architecture for a in agents))
    return (DynamicBatchedPolicy(agents, device, capacity) if dynamic else MaskedBatchedPolicy(agents, device, capacity))


class RolloutRandomness:
    """Fill action noise once per individual per rollout, then serve views."""
    def __init__(self, rng, steps, copies, action_dim, discrete):
        shape = (steps, copies) if discrete else (steps, copies, action_dim)
        self.noise = rng.uniform(shape) if discrete else rng.normal(shape)
        self.step = 0

    def uniform(self, shape):
        result = self.noise[:, self.step]
        self.step += 1
        return result

    normal = uniform


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
        self.member_rng = None
        self.reset_noise = None

    def set_member_seeds(self, seeds):
        """Opt-in streams whose identity does not depend on population layout."""
        if len(seeds) != self.population_size:
            raise ValueError("One environment seed per individual is required")
        self.member_rng = IndependentRNG(seeds, self.device)

    def prepare_rollout(self, steps):
        if self.member_rng is not None:
            self.reset_noise = self.member_rng.uniform((steps, self.env_copies, 1))
            self.reset_noise.mul_(2 * self.radius).sub_(self.radius)
            self.reset_noise_step = 0

    def reset(self, scenarios=None):
        self.reset_noise = None
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
        elif self.member_rng is not None:
            self.position.copy_(self.member_rng.uniform((self.env_copies, 1)) * (2 * self.radius) - self.radius)
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
        if self.reset_noise is not None:
            replacement = self.reset_noise[:, self.reset_noise_step]
            self.reset_noise_step += 1
        elif self.member_rng is not None:
            replacement = self.member_rng.uniform((self.env_copies, 1)) * (2 * self.radius) - self.radius
        else:
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
        self.member_seeds = None
        self.observations = np.zeros((population_size, env_copies, self.observation_dim), dtype=np.float32)

    def set_member_seeds(self, seeds):
        if len(seeds) != self.population_size:
            raise ValueError("One environment seed per individual is required")
        self.member_seeds = seeds

    def reset(self, scenarios=None):
        for p, row in enumerate(self.envs):
            for e, env in enumerate(row):
                scenario = scenarios[e] if scenarios else None
                member_seed = self.member_seeds[p] if self.member_seeds is not None else self.seed + p * self.env_copies
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


def make_tensor_environment(env_factory, tensor_env_factory, agents, copies, device, seed, member_seeds=None):
    env = (tensor_env_factory(len(agents), copies, device, seed) if tensor_env_factory else
           GymTensorBridge(env_factory, len(agents), copies, device, seed))
    first = agents[0]
    if (env.discrete, env.observation_dim, env.action_dim) != (first.discrete, first.observation_dim, first.action_dim):
        env.close()
        raise ValueError("Tensor environment and policy spaces differ")
    if member_seeds is not None:
        if not hasattr(env, "set_member_seeds"):
            env.close()
            raise ValueError("Accelerated PPO tensor environments must implement set_member_seeds(seeds) for chunk-independent streams")
        try:
            env.set_member_seeds(member_seeds)
        except Exception:
            env.close()
            raise
    return env


@torch.no_grad()
def evaluate_batched(agents, scenarios, env_factory, tensor_env_factory=None, device="cpu", env_copies=1, max_steps=1000,
                     population_chunk_size=None, masked=False, growth_capacity=None):
    if population_chunk_size is not None:
        if isinstance(population_chunk_size, bool) or not isinstance(population_chunk_size, int) or population_chunk_size < 1:
            raise ValueError("population_chunk_size must be a positive integer")
        if len(agents) > population_chunk_size:
            return [score for start in range(0, len(agents), population_chunk_size)
                    for score in evaluate_batched(agents[start:start + population_chunk_size], scenarios,
                        env_factory, tensor_env_factory, device, env_copies, max_steps,
                        masked=masked, growth_capacity=growth_capacity)]
    groups = architecture_groups(agents)
    if masked:
        groups = masked_architecture_groups(agents)
    if len(groups) > 1:
        evaluations = [None] * len(agents)
        for indices in groups:
            scores = evaluate_batched([agents[i] for i in indices], scenarios, env_factory,
                                       tensor_env_factory, device, env_copies, max_steps,
                                       masked=masked, growth_capacity=growth_capacity)
            for i, score in zip(indices, scores):
                evaluations[i] = score
        return evaluations
    policy = build_masked_policy(agents, device, growth_capacity) if masked else build_policy(agents, device)
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
                actions = (policy.deterministic_actions(observation) if masked else
                           policy.sample(observation, None, deterministic=True)[0])
                observation, rewards, terminated, truncated = env.step(actions, active)
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


def masked_architecture_groups(agents):
    """Compatible spaces; active hidden width and depth may differ."""
    if not agents or any(not isinstance(agent, MLPAgent) for agent in agents):
        raise ValueError("Batched policies require MLPAgentFactory agents")
    groups = {}
    for i, agent in enumerate(agents):
        key = (agent.observation_dim, agent.action_dim, agent.discrete,
               tuple(agent.low), tuple(agent.high))
        groups.setdefault(key, []).append(i)
    return list(groups.values())
