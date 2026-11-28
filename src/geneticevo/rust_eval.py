"""Frozen population evaluation with optional native CPU policy inference."""
from __future__ import annotations

import copy
from contextlib import ExitStack

import numpy as np

from .agents import MLPAgent
from .core import Evaluation
from .graph import GraphAgent
from .integrations_validation import reject_normalization


def require_extension():
    try:
        from ._rust import Population
    except ImportError as error:
        raise ImportError(
            "Rust evaluation requires the matching geneticevo-core wheel. Install geneticevo[rust] "
            "or build the core with 'python -m pip install ./rust' from the source checkout."
        ) from error
    return Population


def _policy_groups(agents):
    groups = {}
    for i, agent in enumerate(agents):
        if not isinstance(agent, (MLPAgent, GraphAgent)):
            raise ValueError("Rust evaluation requires MLPAgentFactory or GraphAgentFactory agents")
        key = (agent.observation_dim, agent.action_dim, agent.discrete, tuple(agent.low), tuple(agent.high))
        groups.setdefault(key, []).append(i)
    return groups.values()


def _population(agents):
    first = agents[0]
    core = require_extension()(first.observation_dim, first.discrete, first.low.tolist(), first.high.tolist())
    for agent in agents:
        if isinstance(agent, GraphAgent):
            sources, targets = agent.edge_indices()
            spec = (), (len(agent.node_ids), list(zip(sources.tolist(), targets.tolist())), agent.settling_steps)
        else:
            spec = agent.hidden_sizes, None
        core.add(spec, agent.genome.values)
    return core


def _evaluate_chunk(core, first, factory, scenarios, total, steps, max_steps):
    members, copies = total.shape
    obs = np.empty((members, copies, first.observation_dim), np.float32)
    active = np.ones(total.shape, dtype=bool)
    rewards = np.zeros_like(total)
    done = np.zeros_like(active)
    with ExitStack() as stack:
        envs, identities = [], set()
        for index in np.ndindex(total.shape):
            env = factory()
            identity = id(env.unwrapped)
            if identity in identities:
                raise ValueError("env_factory must return independent environments")
            identities.add(identity)
            stack.callback(env.close)
            reject_normalization(env)
            scenario = scenarios[index[1]]
            obs[index], _ = env.reset(seed=scenario.seed, options=copy.deepcopy(scenario.options))
            envs.append(env)
        for _ in range(max_steps):
            actions = core.act(obs)
            for lane in np.flatnonzero(active):
                index = divmod(int(lane), copies)
                action = int(actions[index]) if first.discrete else actions[index]
                obs[index], rewards[index], terminated, truncated, _ = envs[lane].step(action)
                done[index] = terminated or truncated
            np.add(total, rewards, out=total, where=active)
            steps += active
            active &= ~done
            if not active.any():
                break


def evaluate_rust(agents, scenarios, factory, copies, max_steps):
    scores = [None] * len(agents)
    for indices in _policy_groups(agents):
        group = [agents[i] for i in indices]
        core = _population(group)
        returns = np.zeros((len(group), len(scenarios)), dtype=np.float32)
        lengths = np.zeros_like(returns, dtype=np.int64)
        for start in range(0, len(scenarios), copies):
            stop = min(start + copies, len(scenarios))
            span = slice(start, stop)
            _evaluate_chunk(core, group[0], factory, scenarios[span], returns[:, span], lengths[:, span], max_steps)
        for local, i in enumerate(indices):
            scores[i] = Evaluation(float(returns[local].mean()), returns[local].tolist(),
                                   lengths[local].tolist(), int(lengths[local].sum()), context=list(scenarios))
    return scores
