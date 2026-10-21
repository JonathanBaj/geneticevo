"""PettingZoo Parallel protocol adapter and synchronous local communication.

The adapter is dependency-free; real PettingZoo environments may be supplied
when their optional packages are installed. Sender slots preserve identity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import Evaluation, VectorGenome


@dataclass
class CommunicationAction:
    task_action: object
    message: np.ndarray


@dataclass(frozen=True)
class MessageChannel:
    dimension: int = 1
    cost: float = 0.0
    recipient_rule: Callable | None = None
    transform: Callable | None = None

    def __post_init__(self):
        if self.dimension < 1 or self.cost < 0:
            raise ValueError("Message dimension must be positive and cost nonnegative")

    def route(self, messages, sender_ids, recipient_ids, possible_ids, rng):
        inboxes = {recipient: np.zeros((len(possible_ids), self.dimension + 1), np.float32)
                   for recipient in recipient_ids}
        for sender in sender_ids:
            message = np.asarray(messages.get(sender, np.zeros(self.dimension)), dtype=np.float32)
            if message.shape != (self.dimension,) or not np.isfinite(message).all():
                raise ValueError("Message must be a finite vector with the configured dimension")
            message = np.clip(message, -1, 1)
            if self.transform:
                message = np.asarray(self.transform(sender, message.copy(), rng), dtype=np.float32)
                if message.shape != (self.dimension,) or not np.isfinite(message).all():
                    raise ValueError("Message transform returned an invalid packet")
                message = np.clip(message, -1, 1)
            slot = possible_ids.index(sender)
            for recipient in recipient_ids:
                if sender != recipient and (self.recipient_rule is None or self.recipient_rule(sender, recipient)):
                    inboxes[recipient][slot, :self.dimension] = message
                    inboxes[recipient][slot, self.dimension] = 1  # sender-present mask
        return {recipient: inbox.reshape(-1) for recipient, inbox in inboxes.items()}


class ParallelWorldAdapter:
    """All sends use observations available before the step.

    New messages appear after that step in recipients' next observations. Reset
    clears all slots. Inactive IDs neither act nor earn communication costs.
    Fixed message rules receive only the sender's own augmented observation.
    """
    def __init__(self, env, channel=None, fixed_message=None):
        self.env, self.channel, self.fixed_message = env, channel, fixed_message
        self.possible_ids = list(env.possible_agents)
        if len(set(self.possible_ids)) != len(self.possible_ids):
            raise ValueError("possible_agents must be unique")
        self.inbox_size = len(self.possible_ids) * (channel.dimension + 1) if channel else 0
        self.observations = {}

    def observation_space(self, agent_id):
        raw = self.env.observation_space(agent_id)
        if not isinstance(raw, spaces.Box) or len(raw.shape) != 1:
            raise ValueError("Shared adapter requires flat Box observations")
        if not self.channel:
            return raw
        return spaces.Box(np.concatenate((raw.low, -np.ones(self.inbox_size, np.float32))),
                          np.concatenate((raw.high, np.ones(self.inbox_size, np.float32))), dtype=np.float32)

    def reset(self, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=copy.deepcopy(options))
        self.observations = {agent_id: np.concatenate((np.asarray(raw[agent_id], np.float32), np.zeros(self.inbox_size, np.float32)))
                             for agent_id in self.env.agents}
        self.world_steps, self.agent_transitions, self.message_usage = 0, 0, 0.0
        return self.observations, info

    def step(self, actions, rng):
        active = list(self.env.agents)
        if set(actions) != set(active):
            raise ValueError("Actions must map exactly to the world's active agent IDs")
        tasks, messages = {}, {}
        for agent_id in active:
            action = actions[agent_id]
            if isinstance(action, CommunicationAction):
                if self.channel is None:
                    raise ValueError("CommunicationAction requires an enabled channel")
                tasks[agent_id], messages[agent_id] = action.task_action, action.message
            else:
                tasks[agent_id] = action
                messages[agent_id] = self.fixed_message(self.observations[agent_id].copy()) if self.channel and self.fixed_message else np.zeros(self.channel.dimension if self.channel else 0)
        # Validate and route before changing the simulation; filter recipients afterwards.
        inboxes = self.channel.route(messages, active, active, self.possible_ids, rng) if self.channel else {}
        raw, rewards, terminated, truncated, infos = self.env.step(tasks)
        self.world_steps += 1
        self.agent_transitions += len(active)
        if self.channel:
            for sender in active:
                usage = float(np.abs(np.clip(messages[sender], -1, 1)).sum())
                rewards[sender] = float(rewards.get(sender, 0)) - self.channel.cost * usage
                self.message_usage += usage
        self.final_observations = {agent_id: np.concatenate((np.asarray(value, np.float32),
                                   inboxes.get(agent_id, np.zeros(self.inbox_size, np.float32)))) for agent_id, value in raw.items()}
        self.observations = {agent_id: self.final_observations[agent_id] for agent_id in self.env.agents}
        return self.observations, rewards, terminated, truncated, infos

    def close(self):
        self.env.close()


@dataclass
class SharedEvaluator:
    env_factory: Callable
    partner_pool: tuple
    channel: MessageChannel | None = None
    fixed_message: Callable | None = None
    reward_mode: str = "individual"
    max_steps: int = 1000

    def __post_init__(self):
        self.partner_pool = tuple(agent.clone(False) for agent in self.partner_pool)
        if self.reward_mode not in ("individual", "team") or not self.partner_pool:
            raise ValueError("Shared evaluation requires a fixed partner pool and individual/team reward mode")

    def _evaluate(self, focal, pool, scenarios, action_seed):
        world = ParallelWorldAdapter(self.env_factory(), self.channel, self.fixed_message)
        returns, lengths, mappings = [], [], []
        interactions = world_steps = 0
        usage = 0.0
        try:
            for episode, scenario in enumerate(scenarios):
                observations, _ = world.reset(scenario.seed, scenario.options)
                # Rotate roles and partners using common scenario seeds, independently of focal identity.
                role = world.possible_ids[episode % len(world.possible_ids)]
                mapping, partner_indices = {role: focal}, {}
                for slot, agent_id in enumerate(x for x in world.possible_ids if x != role):
                    index = (scenario.seed + episode + slot) % len(pool)
                    mapping[agent_id] = pool[index]
                    partner_indices[agent_id] = index
                rng = np.random.default_rng(action_seed + episode)
                total = 0.0
                for step in range(1, self.max_steps + 1):
                    if role not in world.env.agents:
                        break
                    actions = {agent_id: mapping[agent_id].act(observations[agent_id], rng, True) for agent_id in world.env.agents}
                    observations, rewards, terminated, truncated, _ = world.step(actions, rng)
                    total += float(np.mean(list(rewards.values()))) if self.reward_mode == "team" else float(rewards.get(role, 0))
                    if not world.env.agents or terminated.get(role, False) or truncated.get(role, False):
                        break
                returns.append(total)
                lengths.append(world.world_steps)
                interactions += world.agent_transitions
                world_steps += world.world_steps
                usage += world.message_usage
                mappings.append({"focal_id": role, "partners": partner_indices})
            return Evaluation(float(np.mean(returns)), returns, lengths, interactions,
                              {"message_usage": usage}, context=list(scenarios), world_steps=world_steps,
                              details={"mappings": mappings, "reward_mode": self.reward_mode})
        finally:
            world.close()

    def evaluate(self, agent, scenarios, action_seed=0):
        """Fixed/held-out partners, used for archive comparisons and final tests."""
        return self._evaluate(agent, self.partner_pool, scenarios, action_seed)

    def evaluate_population(self, agents, scenarios, action_seed=0):
        """Selection rotates teammates/opponents from the current population."""
        return [self._evaluate(agent, [other for i, other in enumerate(agents) if i != index] or list(self.partner_pool),
                               scenarios, action_seed) for index, agent in enumerate(agents)]


class CommunicationGenome:
    def __init__(self, agent):
        self.agent = agent

    def vector(self):
        return np.concatenate((self.agent.task_agent.genome.vector(), self.agent.message_genome.vector()))

    def assign(self, values):
        count = self.agent.task_agent.genome.vector().size
        self.agent.task_agent.genome.assign(values[:count])
        self.agent.message_genome.assign(values[count:])

    def clone(self):
        return VectorGenome(self.vector())


@dataclass
class CommunicatingAgent:
    task_agent: object
    message_genome: VectorGenome
    observation_dim: int
    message_dimension: int

    @property
    def genome(self):
        return CommunicationGenome(self)

    def act(self, observation, rng, deterministic=True):
        weights = self.message_genome.values.reshape(self.observation_dim + 1, self.message_dimension)
        message = np.tanh(np.append(observation, 1) @ weights).astype(np.float32)
        return CommunicationAction(self.task_agent.act(observation, rng, deterministic), message)

    def clone(self, training_state=False):
        result = copy.deepcopy(self)
        result.task_agent = self.task_agent.clone(training_state)
        return result

    def save(self, path):
        from pathlib import Path
        import json
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        self.task_agent.save(folder / "task")
        np.savez(folder / "genome.npz", values=self.message_genome.vector())
        (folder / "agent.json").write_text(json.dumps({"kind": "communicating", "schema": 1,
            "observation_dim": self.observation_dim, "message_dimension": self.message_dimension}), encoding="utf-8")


@dataclass(frozen=True)
class CommunicatingAgentFactory:
    task_factory: object
    message_dimension: int = 1

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        if self.message_dimension < 1 or device != "cpu":
            raise ValueError("Evolved communication currently uses CPU and a positive message dimension")
        task_agent = self.task_factory(rng, observation_space, action_space, device)
        dimension = observation_space.shape[0]
        return CommunicatingAgent(task_agent, VectorGenome(rng.normal(0, 0.1, (dimension + 1) * self.message_dimension)),
                                  dimension, self.message_dimension)


def observed_signal_message(observation):
    """Fixed one-dimensional message for the bundled SignalWorld."""
    return np.asarray(observation[:1], dtype=np.float32)


class SignalWorld:
    """Two roles privately see their own bit and predict the other role's bit.

    Small Parallel-API reference, with stationary hidden bits per episode.
    External horizon causes truncation; communication reveals bits next step.
    This implements the protocol without importing PettingZoo.
    """
    metadata = {"name": "geneticevo_signal_v0", "render_modes": [], "is_parallelizable": True}
    possible_agents = ["left", "right"]

    def __init__(self, horizon=4):
        self.horizon = horizon
        self.agents = []
        self._observation_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self._action_space = spaces.Discrete(2)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.agents, self.t = self.possible_agents.copy(), 0
        self.bits = rng.integers(0, 2, 2)
        if options and "bits" in options:
            self.bits = np.asarray(options["bits"])
        return {agent: np.array([self.bits[i] * 2 - 1], np.float32) for i, agent in enumerate(self.agents)}, {x: {} for x in self.agents}

    def step(self, actions):
        active = self.agents.copy()
        if set(actions) != set(active):
            raise ValueError("Supply one task action per active agent")
        rewards = {agent: float(int(actions[agent]) == self.bits[1 - i]) for i, agent in enumerate(active)}
        self.t += 1
        observations = {agent: np.array([self.bits[i] * 2 - 1], np.float32) for i, agent in enumerate(active)}
        terminated = {agent: False for agent in active}
        truncated = {agent: self.t >= self.horizon for agent in active}
        if self.t >= self.horizon:
            self.agents = []
        return observations, rewards, terminated, truncated, {agent: {} for agent in active}

    def close(self):
        pass


class SignalPartner:
    """Fixed task policy reads the other sender's slot when present."""
    def __init__(self, guess=0):
        self.guess = guess

    def act(self, observation, rng, deterministic=True):
        inbox = np.asarray(observation[1:]).reshape(-1, 2)
        present = inbox[:, 1] > 0
        return int(inbox[present, 0].mean() > 0) if present.any() else self.guess

    def clone(self, training_state=False):
        return copy.deepcopy(self)


class SharedFocalEnv(gym.Env):
    """A focal task policy learns against frozen partners in a Parallel world.

    Messages follow fixed_message, so the learner's likelihood covers the full
    *trainable* action. Arbitrary evolved message heads are rejected by SB3.
    Each episode owns independent partner copies and a fresh channel.
    """
    metadata = {"render_modes": []}

    def __init__(self, world_factory=SignalWorld, partner_pool=None, channel=None,
                 fixed_message=None, focal_id=None, reward_mode="individual"):
        self.world = ParallelWorldAdapter(world_factory(), channel, fixed_message)
        self.focal_id = focal_id or self.world.possible_ids[0]
        if self.focal_id not in self.world.possible_ids or reward_mode not in ("individual", "team"):
            raise ValueError("Invalid focal role or reward mode")
        self.partner_pool = tuple(partner_pool or (SignalPartner(),))
        self.reward_mode = reward_mode
        self.observation_space = self.world.observation_space(self.focal_id)
        self.action_space = self.world.env.action_space(self.focal_id)
        self.total_agent_transitions = self.total_world_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.world.reset(seed, options)
        self.partners = {agent_id: self.partner_pool[int(self.np_random.integers(len(self.partner_pool)))].clone(False)
                         for agent_id in self.world.possible_ids if agent_id != self.focal_id}
        return observations[self.focal_id], infos.get(self.focal_id, {})

    def step(self, action):
        active = list(self.world.env.agents)
        actions = {agent_id: action if agent_id == self.focal_id else self.partners[agent_id].act(self.world.observations[agent_id], self.np_random, True)
                   for agent_id in active}
        observation, rewards, terminated, truncated, infos = self.world.step(actions, self.np_random)
        self.total_world_steps += 1
        self.total_agent_transitions += len(active)
        reward = float(np.mean(list(rewards.values()))) if self.reward_mode == "team" else float(rewards[self.focal_id])
        # Final observation is the permitted focal observation plus the last inbox.
        result = self.world.final_observations.get(self.focal_id)
        if result is None:
            if truncated.get(self.focal_id, False):
                raise ValueError("A truncated focal agent must have a final observation for bootstrapping")
            result = np.zeros(self.observation_space.shape, np.float32)
        return result, reward, terminated.get(self.focal_id, False), truncated.get(self.focal_id, False), infos.get(self.focal_id, {})

    def close(self):
        self.world.close()
