"""Run: python examples/neural.py; no torch needed."""
from geneticevo import Evolution, EvolutionConfig, GaussianMutation
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.core import scenarios_for
from geneticevo.evaluation import GymEvaluator


if __name__ == "__main__":
    result = Evolution(env_factory=Reach1D, agent_factory=MLPAgentFactory(hidden_sizes=(16, 16)),
                       config=EvolutionConfig(population_size=12, generations=5, evaluation_episodes=3,
                                              output_dir="results/neural"),
                       mutation=GaussianMutation(0.1, 0.1), learner=None).run()
    result.best_agent.save("results/neural/champion")
    test = result.evaluate(GymEvaluator(Reach1D), scenarios_for(2026, "test", 10))
    print(f"Reference fitness: {result.best_fitness:.4f}; untouched final test: {test.fitness:.4f}")
