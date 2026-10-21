"""A non-PPO learner extension: budgeted random-search lifetime adaptation."""
from dataclasses import dataclass
import numpy as np

from geneticevo import Evolution, EvolutionConfig, LearnReport
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.core import Scenario
from geneticevo.evaluation import GymEvaluator
from geneticevo.learners import GenomeLearner


@dataclass(frozen=True)
class RandomSearchLearner(GenomeLearner):
    std: float = 0.01

    def validate(self, agent, env_factory, backend):
        if backend != "serial":
            raise ValueError("This example learner uses serial Gymnasium evaluation")

    def prepare(self, agent, seed):
        pass

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        self.validate(agent, env_factory, "serial")
        if budget < 2:
            raise ValueError("Two candidate evaluations require at least two transitions")
        rng = np.random.default_rng(action_seed)
        candidate = agent.clone(False)
        candidate.genome.assign(candidate.genome.vector() + rng.normal(0, self.std, candidate.genome.vector().size))
        # Cap each candidate evaluation so actual interactions cannot exceed the budget.
        evaluator = GymEvaluator(env_factory, max_steps=budget // 2)
        scenarios = [Scenario(environment_seed, distribution="training")]
        before, after = (evaluator.evaluate(policy, scenarios) for policy in (agent, candidate))
        if after.fitness > before.fitness:
            agent.genome.assign(candidate.genome.vector())
        return LearnReport(before.interactions + after.interactions, 1, budget,
                           {"accepted": float(after.fitness > before.fitness)})


if __name__ == "__main__":
    result = Evolution(env_factory=Reach1D, agent_factory=MLPAgentFactory((8,)), learner=RandomSearchLearner(),
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1, learning_budget=64,
                               evaluation_episodes=2)).run()
    print(f"Random-search adaptation plus evolution: {result.best_fitness:.4f}")

