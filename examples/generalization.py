"""Frozen shifted-distribution testing and independent budgeted adaptation."""
from functools import partial

from geneticevo import Evolution, EvolutionConfig, Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.core import scenarios_for
from geneticevo.environments import Reach1D
from geneticevo.evaluation import AdaptationTask, GymEvaluator, adaptation_evaluation
from custom_learner import RandomSearchLearner


class ShiftedReach(Reach1D):
    def __init__(self, radius=2):
        super().__init__()
        self.radius = radius

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options={"radius": self.radius} | (options or {}))


if __name__ == "__main__":
    result = Evolution(env_factory=Reach1D, agent_factory=MLPAgentFactory((8,)),
        config=EvolutionConfig(population_size=6, generations=3, elite_count=1, evaluation_episodes=2)).run()
    evaluator = GymEvaluator(Reach1D)
    frozen = result.evaluate(evaluator, scenarios_for(2026, "test", 8))
    shifted_scenarios = [Scenario(seed, {"radius": 2}, "test") for seed in range(100, 108)]
    shifted = result.evaluate(evaluator, shifted_scenarios)
    tasks = [AdaptationTask(partial(ShiftedReach, radius=radius),
                [Scenario(seed, {"radius": radius}, "test") for seed in range(200, 204)], f"radius={radius}")
             for radius in (2, 3)]
    adapted = adaptation_evaluation(result.best_agent, RandomSearchLearner(), Reach1D, evaluator,
                                    tasks, budget=64, seed=2026)
    print(f"Frozen={frozen.fitness:.4f}; shifted frozen={shifted.fitness:.4f}")
    for task, record in zip(tasks, adapted):
        evaluation_cost = record["before"].interactions + record["after"].interactions
        print(f"{task.label}: {record['before'].fitness:.4f} -> {record['after'].fitness:.4f}; "
              f"adaptation transitions={record['learning'].interactions}; evaluation transitions={evaluation_cost}")
