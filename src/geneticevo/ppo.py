"""Native independent PPO with stacked parameters and per-individual Adam state."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .agents import MLPAgent, architecture_groups
from .graph import GraphAgent
from .core import LearnReport, seed_for
from .learners import GenomeLearner
from .tensor import (MaskedBatchedPolicy, IndependentRNG, RolloutRandomness,
                     TensorReach1D, masked_architecture_groups, make_tensor_environment, resolve_device, build_policy)


@contextmanager
def training_phase(name, device, timings=None):
    """Profiler ranges always; synchronized wall timings only when requested."""
    if timings is not None:
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
    with torch.profiler.record_function(f"ppo.{name}"):
        yield
    if timings is not None:
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        key = name + "_seconds"
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - started


class RolloutBuffer:
    """Reusable device storage [P,T,E,...], shared by collection and PPO views."""
    def __init__(self, policy, steps, copies, deferred_values=False):
        shape = (policy.population_size, steps, copies)
        specs = {"observations": (shape + (policy.observation_dim,), torch.float32),
                 "actions": (shape if policy.discrete else shape + (policy.action_dim,),
                             torch.long if policy.discrete else torch.float32)}
        for key in ("log_probs", "values", "next_values", "rewards", "terminated", "truncated"):
            specs[key] = (shape, torch.bool if key in ("terminated", "truncated") else torch.float32)
        if deferred_values:
            specs["next_observations"] = specs["observations"]
        self.data = {key: torch.empty(size, dtype=dtype, device=policy.device) for key, (size, dtype) in specs.items()}


@torch.no_grad()
def rollout_values(policy, buffer, env):
    """One critic pass for trajectory states; correct pre-reset truncations.

    Interior next-values reuse the following state's value. Only truncations
    need a different bootstrap observation. Reach1D has a known upper bound on
    their count; generic environments need one scalar read per rollout.
    """
    observations = buffer["observations"]
    p, steps, copies, _ = observations.shape
    all_values = policy.values(torch.cat((observations.flatten(1, 2), buffer["next_observations"][:, -1]), dim=1))
    buffer["values"].copy_(all_values[:, :steps * copies].reshape(p, steps, copies))
    buffer["next_values"].copy_(all_values[:, copies:].reshape(p, steps, copies))
    mask = (buffer["truncated"] & ~buffer["terminated"]).clone()
    mask[:, -1] = False  # The final pre-reset observation is already the tail.
    mask = mask.flatten(1, 2)
    slots = (min(steps * copies, math.ceil(steps / env.horizon) * copies) if type(env) is TensorReach1D
             else int(mask.sum(1).max().cpu()))
    if slots:
        indices = mask.to(torch.int32).topk(slots, dim=1).indices
        rows = torch.arange(p, device=policy.device)[:, None]
        final = buffer["next_observations"].flatten(1, 2)[rows, indices]
        values = policy.values(final)
        next_values = buffer["next_values"].flatten(1, 2)
        next_values.scatter_(1, indices, torch.where(mask[rows, indices], values, next_values[rows, indices]))


def generalized_advantages(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Inputs [P,T,E]. Bootstrap truncations, cut recurrence at every reset."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + gamma * (~terminated[:, t]) * next_values[:, t] - values[:, t]
        carry = delta + gamma * gae_lambda * (~(terminated[:, t] | truncated[:, t])) * carry
        advantage[:, t] = carry
    return advantage, advantage + values


def ppo_losses(policy, observations, actions, old_log_probs, advantages, returns,
               clip=0.2, value_coefficient=0.5, entropy_coefficient=0.0):
    """Return one mean loss per individual. Sum these for independent gradients."""
    log_probs, entropy = policy.log_prob(observations, actions)
    values = policy.values(observations)
    ratio = (log_probs - old_log_probs).exp()
    policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean(1)
    value_loss = 0.5 * (values - returns).square().mean(1)
    entropy_mean = entropy.mean(1)
    total = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
    with torch.no_grad():
        log_ratio = log_probs - old_log_probs
        metrics = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy_mean,
                   "approx_kl": ((ratio - 1) - log_ratio).mean(1),
                   "clip_fraction": ((ratio - 1).abs() > clip).float().mean(1)}
    return total, metrics


def clip_individual_gradients(parameters, maximum):
    parameters = list(parameters)
    norm_squared = torch.zeros(parameters[0].shape[0], device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            norm_squared += parameter.grad.flatten(1).square().sum(1)
    norms = norm_squared.sqrt()
    scale = (maximum / (norms + 1e-6)).clamp_max(1)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale.reshape((-1,) + (1,) * (parameter.ndim - 1)))
    return norms


class IndependentAdam:
    """Adam moments *and step counters* lead with P, including fresh children.

    Elites may have more updates than newly mutated offspring. A conventional
    scalar counter over stacked tensors would give children wrong corrections.
    """
    def __init__(self, policy, learning_rate=3e-4, epsilon=1e-5, agents=None, optimized=False):
        self.params, self.lr, self.epsilon = policy.params, learning_rate, epsilon
        self.optimized = optimized
        self.steps = torch.zeros(policy.population_size, device=policy.device)
        self.first = {name: torch.zeros_like(param) for name, param in self.params.items()}
        self.second = {name: torch.zeros_like(param) for name, param in self.params.items()}
        if agents is not None:
            for index, agent in enumerate(agents):
                state = getattr(agent, "training_state", {}).get("adam")
                if state:
                    self.steps[index] = state["step"]
                    for name in self.params:
                        shape = dict(agent.layout())[name]
                        for moments, source in ((self.first, state["first"]), (self.second, state["second"])):
                            if tuple(source[name].shape) != shape:
                                raise ValueError("Adam state shape differs from active agent parameters")
                            moments[name][(index, *[slice(0, n) for n in shape])].copy_(source[name].to(policy.device))

    @torch.no_grad()
    def step(self):
        self.steps += 1
        if self.optimized:
            correction1, correction2 = 1 - 0.9**self.steps, (1 - 0.999**self.steps).sqrt()
        for name, param in self.params.items():
            if param.grad is None:
                continue
            first, second = self.first[name], self.second[name]
            first.mul_(0.9).add_(param.grad, alpha=0.1)
            second.mul_(0.999).addcmul_(param.grad, param.grad, value=0.001)
            shape = (-1,) + (1,) * (param.ndim - 1)
            bias1 = correction1.reshape(shape) if self.optimized else (1 - 0.9**self.steps).reshape(shape)
            root_bias2 = correction2.reshape(shape) if self.optimized else (1 - 0.999**self.steps).reshape(shape).sqrt()
            denominator = second.sqrt() / root_bias2 + self.epsilon
            param.add_(-(self.lr / bias1) * first / denominator)

    @torch.no_grad()
    def reset_members(self, indices):
        self.steps[indices] = 0
        for name in self.params:
            self.first[name][indices] = 0
            self.second[name][indices] = 0

    def export(self, agents):
        # Accelerated exports transfer each stacked tensor once, rather than
        # issuing P small device-to-host copies per tensor.
        first = {name: value.detach().cpu() for name, value in self.first.items()} if self.optimized else self.first
        second = {name: value.detach().cpu() for name, value in self.second.items()} if self.optimized else self.second
        steps = self.steps.cpu() if self.optimized else self.steps
        for index, agent in enumerate(agents):
            if not hasattr(agent, 'training_state'):
                agent.training_state = {}
            slices = {name: (index, *[slice(0, n) for n in shape]) for name, shape in agent.layout()}
            agent.training_state['adam'] = {"step": float(steps[index].cpu()),
                "first": {name: value[slices[name]].detach().cpu().clone() for name, value in first.items()},
                "second": {name: value[slices[name]].detach().cpu().clone() for name, value in second.items()}}


@dataclass(frozen=True)
class NativePPO(GenomeLearner):
    supports_neural_growth = True
    supports_graph_growth = True

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
    accelerated: bool = False
    population_chunk_size: int | None = None
    growth_capacity: tuple[int, ...] | None = None
    # Ablation switches for benchmarks; apply only to the accelerated path.
    reuse_rollout_values: bool = True
    preallocate_rollouts: bool = True
    batch_randomness: bool = True
    measure_phases: bool = False

    def __post_init__(self):
        for name in ("accelerated", "reuse_rollout_values", "preallocate_rollouts", "batch_randomness", "measure_phases"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.population_chunk_size is not None and (isinstance(self.population_chunk_size, bool) or
                not isinstance(self.population_chunk_size, int) or self.population_chunk_size < 1):
            raise ValueError("population_chunk_size must be a positive integer")
        if self.growth_capacity is not None and (not isinstance(self.growth_capacity, tuple) or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.growth_capacity)):
            raise ValueError("growth_capacity must be a tuple of positive integers")
        if not self.accelerated and (self.population_chunk_size is not None or self.growth_capacity is not None):
            raise ValueError("Population chunks and growth_capacity require accelerated=True")
        for name in ("env_copies", "rollout_steps", "epochs", "minibatch_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rollout_steps * self.env_copies < 2:
            raise ValueError("PPO needs at least two transitions per individual per rollout")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.clip > 0 and
                self.learning_rate > 0 and self.max_gradient_norm > 0):
            raise ValueError("Invalid PPO hyperparameters")
        resolve_device(self.device)

    def validate(self, agent, env_factory, backend):
        if not isinstance(agent, (MLPAgent, GraphAgent)) or backend != "batched":
            raise ValueError("NativePPO requires MLPAgentFactory or GraphAgentFactory and backend='batched'")
        if isinstance(agent, GraphAgent) and self.accelerated:
            raise ValueError("Graph brains require NativePPO(accelerated=False); CPU and CUDA are supported")
        from .agents import validate_spaces
        env = env_factory()
        try:
            validate_spaces(env.observation_space, env.action_space)
            if env.observation_space.shape != (agent.observation_dim,):
                raise ValueError("Training observation space differs")
            from gymnasium import spaces
            if isinstance(env.action_space, spaces.Discrete) != agent.discrete or (
                env.action_space.n if agent.discrete else env.action_space.shape[0]) != agent.action_dim:
                raise ValueError("Training action space differs")
            if not agent.discrete and not (np.array_equal(env.action_space.low, agent.low) and np.array_equal(env.action_space.high, agent.high)):
                raise ValueError("Training action bounds differ")
            from .integrations_validation import reject_normalization
            reject_normalization(env)
        finally:
            env.close()

    def prepare(self, agent, seed):
        if not isinstance(agent, (MLPAgent, GraphAgent)):
            raise ValueError("NativePPO requires MLPAgent or GraphAgent")

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        # This also enables bounded adaptation of one champion through the same implementation.
        return self.learn_population([agent], env_factory, None, budget, environment_seed, action_seed)[0]

    def learn_population(self, agents, env_factory, tensor_env_factory, budget, seed, generation,
                         *, member_ids=None, timings=None):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("Learning budget must be a positive integer")
        if self.accelerated and any(isinstance(a, GraphAgent) for a in agents):
            raise ValueError("Graph brains require NativePPO(accelerated=False); CPU and CUDA are supported")
        member_ids = list(range(len(agents))) if member_ids is None else list(member_ids)
        if len(member_ids) != len(agents) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in member_ids) or len(set(member_ids)) != len(member_ids):
            raise ValueError("member_ids must contain one unique nonnegative integer per agent")
        groups = masked_architecture_groups(agents) if self.accelerated else architecture_groups(agents)
        batches = []
        for group in groups:
            if self.accelerated:
                depth = len(agents[group[0]].hidden_sizes)
                capacity = self.growth_capacity if self.growth_capacity is not None else tuple(max(
                    agents[i].growth.max_hidden_size if agents[i].growth else agents[i].hidden_sizes[layer]
                    for i in group) for layer in range(depth))
                chunk = self.population_chunk_size or len(group)
                batches.extend((group[start:start + chunk], capacity) for start in range(0, len(group), chunk))
                continue
            fixed = [i for i in group if agents[i].growth is None]
            if fixed:
                batches.append((fixed, None))
            batches.extend(([i], None) for i in group if agents[i].growth is not None)
        reports = [None] * len(agents)
        for indices, capacity in batches:
            group_reports = self._learn_group([agents[i] for i in indices], env_factory,
                tensor_env_factory, budget, seed, generation, [member_ids[i] for i in indices],
                split=len(batches) > 1 or any(a.growth is not None for a in agents),
                capacity=capacity, timings=timings)
            for i, report in zip(indices, group_reports):
                reports[i] = report
        return reports

    def _learn_group(self, agents, env_factory, tensor_env_factory, budget, seed, generation, member_indices,
                     split, capacity=None, timings=None):
        deferred = self.accelerated and self.reuse_rollout_values
        preallocated = self.accelerated and self.preallocate_rollouts
        has_growth = any(a.growth is not None for a in agents)
        with training_phase("setup", self.device, timings):
            masked = self.accelerated and (has_growth or capacity != agents[0].hidden_sizes or
                                           any(a.hidden_sizes != agents[0].hidden_sizes for a in agents))
            policy = (MaskedBatchedPolicy(agents, self.device, capacity) if masked else build_policy(agents, self.device))
            optimizer = IndependentAdam(policy, self.learning_rate, agents=agents, optimized=self.accelerated)
            rng = IndependentRNG([seed_for(seed, "actions", generation, i) for i in member_indices], self.device)
            # Separate domains keep action streams independent of shuffle work.
            shuffle_rng = (IndependentRNG([seed_for(seed, "actions", generation, i, 1) for i in member_indices], self.device)
                           if self.accelerated else rng)
            growth_rngs = [np.random.default_rng(seed_for(seed, "growth", generation, i)) if a.growth else None
                           for a, i in zip(agents, member_indices)]
            environment_seed = (seed_for(seed, 'training', generation, member_indices[0]) if split else
                                seed_for(seed, 'training', generation))
            storage = RolloutBuffer(policy, self.rollout_steps, self.env_copies, deferred).data if preallocated else None
            rows = torch.arange(len(agents), device=policy.device)[:, None]
            env = make_tensor_environment(env_factory, tensor_env_factory, agents, self.env_copies,
                self.device, environment_seed, member_seeds=[seed_for(seed, "training", generation, i)
                    for i in member_indices] if self.accelerated else None)
        rollout_count = math.ceil(budget / (self.env_copies * self.rollout_steps))
        metric_totals, updates = {}, 0
        growth_events = [0] * len(agents)
        try:
            with training_phase("setup", self.device, timings):
                observations = env.reset()
            for _ in range(rollout_count):
                with training_phase("rollout", self.device, timings), torch.no_grad():
                    keys = ("observations", "actions", "log_probs", "values", "next_values", "rewards", "terminated", "truncated")
                    if deferred:
                        keys += ("next_observations",)
                    buffer = storage if preallocated else {key: [] for key in keys}
                    action_rng = (RolloutRandomness(rng, self.rollout_steps, self.env_copies, policy.action_dim, policy.discrete)
                                  if self.accelerated and self.batch_randomness else rng)
                    if self.accelerated and hasattr(env, "prepare_rollout"):
                        env.prepare_rollout(self.rollout_steps)
                    for t in range(self.rollout_steps):
                        actions, raw, log_probs, values = policy.sample(observations, action_rng,
                            reuse_output=deferred, compute_values=not deferred)
                        next_observations, rewards, terminated, truncated = env.step(actions)
                        items = {"observations": observations, "actions": raw, "log_probs": log_probs,
                                 "rewards": rewards, "terminated": terminated, "truncated": truncated}
                        if deferred:
                            items["next_observations"] = next_observations
                        else:
                            items.update(values=values, next_values=policy.values(next_observations))
                        for key, value in items.items():
                            if preallocated:
                                buffer[key][:, t].copy_(value)
                            else:
                                # Snapshot before reset_done can mutate environment storage.
                                buffer[key].append(value.clone() if self.accelerated else value)
                        observations = env.reset_done(terminated | truncated)
                    if not preallocated:
                        buffer = {key: torch.stack(value, dim=1) for key, value in buffer.items() if value}
                        if deferred:
                            buffer["values"] = torch.empty_like(buffer["rewards"])
                            buffer["next_values"] = torch.empty_like(buffer["rewards"])
                    if deferred:
                        rollout_values(policy, buffer, env)
                with training_phase("update", self.device, timings):
                    advantages, returns = generalized_advantages(buffer["rewards"], buffer["values"], buffer["next_values"],
                        buffer["terminated"], buffer["truncated"], self.gamma, self.gae_lambda)
                    batch = {"observations": buffer["observations"].flatten(1, 2), "actions": buffer["actions"].flatten(1, 2),
                             "old_log_probs": buffer["log_probs"].flatten(1, 2),
                             "advantages": advantages.flatten(1, 2), "returns": returns.flatten(1, 2)}
                    adv = batch["advantages"]
                    batch["advantages"] = (adv - adv.mean(1, keepdim=True)) / (adv.std(1, unbiased=False, keepdim=True) + 1e-8)
                    rollout_value_loss = torch.zeros(len(agents), device=policy.device) if has_growth else None
                    rollout_updates = 0
                    count = self.rollout_steps * self.env_copies
                    for _ in range(self.epochs):
                        order = shuffle_rng.permutations(count)
                        for start in range(0, count, self.minibatch_size):
                            indices = order[:, start:start + self.minibatch_size]
                            mini = {key: value[rows, indices] for key, value in batch.items()}
                            policy.zero_grad(set_to_none=True)
                            loss, metrics = ppo_losses(policy, **mini, clip=self.clip,
                                value_coefficient=self.value_coefficient, entropy_coefficient=self.entropy_coefficient)
                            loss.sum().backward()
                            metrics["gradient_norm"] = clip_individual_gradients(policy.parameters(), self.max_gradient_norm)
                            optimizer.step()
                            updates += 1
                            if has_growth:
                                rollout_updates += 1
                                rollout_value_loss += metrics['value_loss'].detach()
                            for key, value in metrics.items():
                                if key not in metric_totals:
                                    metric_totals[key] = torch.zeros_like(value)
                                metric_totals[key].add_(value.detach())
                if has_growth:
                    with training_phase("growth", self.device, timings):
                        losses = (rollout_value_loss / rollout_updates).detach().cpu().tolist()
                        if not self.accelerated:
                            policy.export(agents)
                            optimizer.export(agents)
                        changed_indices = []
                        for i, agent in enumerate(agents):
                            before = getattr(agent, 'hidden_sizes', None)
                            if agent.growth and agent.growth.after_update(agent, count, losses[i], growth_rngs[i]):
                                growth_events[i] += 1
                                changed_indices.append(i)
                                if self.accelerated:
                                    policy.activate(i, agent, before)
                        if changed_indices:
                            if self.accelerated:
                                optimizer.reset_members(changed_indices)
                            else:
                                policy = build_policy(agents, self.device)
                                optimizer = IndependentAdam(policy, self.learning_rate, agents=agents)
            with training_phase("export", self.device, timings):
                policy.export(agents)
                optimizer.export(agents)
                if self.accelerated:
                    metric_totals = {key: value.cpu() for key, value in metric_totals.items()}
        finally:
            env.close()
        actual = rollout_count * self.env_copies * self.rollout_steps
        reports = [LearnReport(actual, updates, budget,
                    {key: float(value[i].cpu()) / updates for key, value in metric_totals.items()})
                   for i in range(len(agents))]
        for i, agent in enumerate(agents):
            if agent.growth is not None:
                reports[i].metrics.update(growth_events=growth_events[i],
                    hidden_neurons=agent.hidden_neurons if isinstance(agent, GraphAgent) else sum(agent.hidden_sizes),
                    parameters=agent.genome.values.size)
                if isinstance(agent, GraphAgent):
                    reports[i].metrics['connections'] = len(agent.edges)
        return reports

