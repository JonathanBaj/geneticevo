"""Frozen evaluation, disjoint scenarios, and adaptation on fresh copies."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Callable

import numpy as np

from .core import Evaluation, Scenario, seed_for


@dataclass
class ObjectiveEvaluator:
    objective: Callable
    penalty: Callable | None = None

    def evaluate(self, agent, scenarios, action_seed=0):
        values, metrics, descriptors = [], {}, []
        for scenario in scenarios:
            output = self.objective(agent.genome.vector(), scenario)
            score, task = output if isinstance(output, tuple) else (output, {})
            if self.penalty:
                score -= self.penalty(agent.genome.vector(), scenario)
            values.append(float(score))
            for key, value in task.items():
                metrics.setdefault(key, []).append(float(value))
        return Evaluation(float(np.mean(values)), returns=values,
                          metrics={k: float(np.mean(v)) for k, v in metrics.items()}, context=list(scenarios))


@dataclass
class GymEvaluator:
    env_factory: Callable
    max_steps: int = 1000
    aggregate: Callable = np.mean
    penalty: Callable | None = None

    def evaluate(self, agent, scenarios, action_seed=0):
        env = self.env_factory()
        from .integrations_validation import reject_normalization
        returns, lengths, metrics = [], [], {}
        try:
            reject_normalization(env)
            for index, scenario in enumerate(scenarios):
                # Evaluation never calls a learner or updates normalization.
                rng = np.random.default_rng(seed_for(action_seed, "actions", index))
                observation, _ = env.reset(seed=scenario.seed, options=copy.deepcopy(scenario.options))
                total, info = 0.0, {}
                for step in range(1, self.max_steps + 1):
                    action = agent.act(observation, rng, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                returns.append(total)
                lengths.append(step)
                for key, value in info.items():
                    if isinstance(value, (int, float, np.number)):
                        metrics.setdefault(key, []).append(float(value))
            fitness = float(self.aggregate(returns))
            if self.penalty is not None:
                fitness -= float(self.penalty(agent, returns, metrics))
            return Evaluation(fitness, returns, lengths, sum(lengths),
                              {k: float(np.mean(v)) for k, v in metrics.items()}, context=list(scenarios))
        finally:
            env.close()


@dataclass(frozen=True)
class AdaptationTask:
    training_env_factory: Callable
    scenarios: list[Scenario]
    label: str = "task"


def adaptation_evaluation(agent, learner, env_factory, evaluator, tasks, budget, seed=0):
    """One fresh copy and fresh training state per task; champion stays frozen."""
    records = []
    for index, task in enumerate(tasks):
        scenarios = task.scenarios if isinstance(task, AdaptationTask) else task
        training_factory = task.training_env_factory if isinstance(task, AdaptationTask) else env_factory
        copy = learner.clone(agent, training_state=False)
        before = evaluator.evaluate(copy, scenarios, seed_for(seed, "test", index))
        learner.validate(copy, training_factory, "batched" if hasattr(learner, "learn_population") else "serial")
        report = learner.learn(copy, training_factory, budget, seed_for(seed, "training", index),
                               seed_for(seed, "actions", index))
        after = evaluator.evaluate(copy, scenarios, seed_for(seed, "test", index))
        records.append({"before": before, "after": after, "learning": report})
    return records

