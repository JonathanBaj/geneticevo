"""Native independent PPO with stacked parameters and per-individual Adam state."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
import torch

from .agents import MLPAgent, architecture_groups
from .core import LearnReport, seed_for
from .learners import GenomeLearner
from .tensor import BatchedPolicy, IndependentRNG, make_tensor_environment, resolve_device


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
    def __init__(self, policy, learning_rate=3e-4, epsilon=1e-5, agents=None):
        self.params, self.lr, self.epsilon = policy.params, learning_rate, epsilon
        self.steps = torch.zeros(policy.population_size, device=policy.device)
        self.first = {name: torch.zeros_like(param) for name, param in self.params.items()}
        self.second = {name: torch.zeros_like(param) for name, param in self.params.items()}
        if agents is not None:
            for index, agent in enumerate(agents):
                state = getattr(agent, "training_state", {}).get("adam")
                if state:
                    self.steps[index] = state["step"]
                    for name in self.params:
                        self.first[name][index].copy_(state["first"][name].to(policy.device))
                        self.second[name][index].copy_(state["second"][name].to(policy.device))

    @torch.no_grad()
    def step(self):
        self.steps += 1
        for name, param in self.params.items():
            if param.grad is None:
                continue
            first, second = self.first[name], self.second[name]
            first.mul_(0.9).add_(param.grad, alpha=0.1)
            second.mul_(0.999).addcmul_(param.grad, param.grad, value=0.001)
            shape = (-1,) + (1,) * (param.ndim - 1)
            bias1 = (1 - 0.9**self.steps).reshape(shape)
            bias2 = (1 - 0.999**self.steps).reshape(shape)
            denominator = second.sqrt() / bias2.sqrt() + self.epsilon
            param.add_(-(self.lr / bias1) * first / denominator)

    def export(self, agents):
        for index, agent in enumerate(agents):
            if not hasattr(agent, 'training_state'):
                agent.training_state = {}
            agent.training_state['adam'] = {"step": float(self.steps[index].cpu()),
                "first": {name: value[index].detach().cpu().clone() for name, value in self.first.items()},
                "second": {name: value[index].detach().cpu().clone() for name, value in self.second.items()}}


@dataclass(frozen=True)
class NativePPO(GenomeLearner):
    supports_neural_growth = True

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

    def __post_init__(self):
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
        if not isinstance(agent, MLPAgent) or backend != "batched":
            raise ValueError("NativePPO requires MLPAgentFactory and backend='batched'")
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
        if not isinstance(agent, MLPAgent):
            raise ValueError("NativePPO requires MLPAgent")

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        # This also enables bounded adaptation of one champion through the same implementation.
        return self.learn_population([agent], env_factory, None, budget, environment_seed, action_seed)[0]

    def learn_population(self, agents, env_factory, tensor_env_factory, budget, seed, generation):
        # Growth-enabled individuals learn separately because widths can diverge
        # at any rollout. Environment copies still execute as tensors on device.
        # Non-growing individuals retain batching within each compatible shape.
        groups = architecture_groups(agents)
        batches = []
        for group in groups:
            fixed = [i for i in group if agents[i].growth is None]
            if fixed:
                batches.append(fixed)
            batches.extend([i] for i in group if agents[i].growth is not None)
        reports = [None] * len(agents)
        for indices in batches:
            group_reports = self._learn_group([agents[i] for i in indices], env_factory,
                tensor_env_factory, budget, seed, generation, indices,
                split=len(batches) > 1 or any(a.growth is not None for a in agents))
            for i, report in zip(indices, group_reports):
                reports[i] = report
        return reports

    def _learn_group(self, agents, env_factory, tensor_env_factory, budget, seed, generation, member_indices, split):
        if budget < 1:
            raise ValueError("Learning budget must be positive")
        policy = BatchedPolicy(agents, self.device)
        optimizer = IndependentAdam(policy, self.learning_rate, agents=agents)
        rng = IndependentRNG([seed_for(seed, "actions", generation, i) for i in member_indices], self.device)
        growth = agents[0].growth
        growth_rng = np.random.default_rng(seed_for(seed, 'growth', generation, member_indices[0])) if growth else None
        environment_seed = (seed_for(seed, 'training', generation, member_indices[0]) if split else
                            seed_for(seed, 'training', generation))
        env = make_tensor_environment(env_factory, tensor_env_factory, agents, self.env_copies,
                                      self.device, environment_seed)
        rollout_count = math.ceil(budget / (self.env_copies * self.rollout_steps))
        metric_totals, updates, growth_events = {}, 0, 0
        try:
            observations = env.reset()
            for _ in range(rollout_count):
                buffer = {key: [] for key in ("observations", "actions", "log_probs", "values", "next_values", "rewards", "terminated", "truncated")}
                with torch.no_grad():
                    for _ in range(self.rollout_steps):
                        actions, raw, log_probs, values = policy.sample(observations, rng)
                        next_observations, rewards, terminated, truncated = env.step(actions)
                        # Value the *final* observation before resetting any ended episode.
                        next_values = policy.values(next_observations)
                        for key, value in (("observations", observations), ("actions", raw), ("log_probs", log_probs),
                                           ("values", values), ("next_values", next_values), ("rewards", rewards),
                                           ("terminated", terminated), ("truncated", truncated)):
                            buffer[key].append(value)
                        observations = env.reset_done(terminated | truncated)
                buffer = {key: torch.stack(value, dim=1) for key, value in buffer.items()}
                advantages, returns = generalized_advantages(buffer["rewards"], buffer["values"], buffer["next_values"],
                                                             buffer["terminated"], buffer["truncated"], self.gamma, self.gae_lambda)
                batch = {"observations": buffer["observations"].flatten(1, 2), "actions": buffer["actions"].flatten(1, 2),
                         "old_log_probs": buffer["log_probs"].flatten(1, 2),
                         "advantages": advantages.flatten(1, 2), "returns": returns.flatten(1, 2)}
                adv = batch["advantages"]
                batch["advantages"] = (adv - adv.mean(1, keepdim=True)) / (adv.std(1, unbiased=False, keepdim=True) + 1e-8)
                rollout_value_loss = torch.zeros(len(agents), device=policy.device) if growth is not None else None
                rollout_updates = 0
                count = self.rollout_steps * self.env_copies
                rows = torch.arange(len(agents), device=policy.device)[:, None]
                for _ in range(self.epochs):
                    order = rng.permutations(count)
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
                        if growth is not None:
                            rollout_updates += 1
                            rollout_value_loss += metrics['value_loss'].detach()
                        for key, value in metrics.items():
                            metric_totals[key] = metric_totals.get(key, torch.zeros_like(value)) + value.detach()
                if growth is not None:
                    policy.export(agents)
                    optimizer.export(agents)
                    changed = growth.after_update(agents[0], count,
                        float((rollout_value_loss[0] / rollout_updates).cpu()), growth_rng)
                    if changed:
                        growth_events += 1
                        # Observations and environments continue across resizing.
                        # Resize clears incompatible Adam state; rebuild tensors.
                        policy = BatchedPolicy(agents, self.device)
                        optimizer = IndependentAdam(policy, self.learning_rate, agents=agents)
            policy.export(agents)
            optimizer.export(agents)
        finally:
            env.close()
        actual = rollout_count * self.env_copies * self.rollout_steps
        reports = [LearnReport(actual, updates, budget,
                    {key: float(value[i].cpu()) / updates for key, value in metric_totals.items()})
                   for i in range(len(agents))]
        if growth is not None:
            reports[0].metrics.update(growth_events=growth_events,
                hidden_neurons=sum(agents[0].hidden_sizes), parameters=agents[0].genome.values.size)
        return reports

