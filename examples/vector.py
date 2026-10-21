"""Run: python examples/vector.py"""
from geneticevo import Evolution, EvolutionConfig, GaussianMutation
from geneticevo.agents import VectorAgentFactory
from geneticevo.evaluation import ObjectiveEvaluator


def sphere(vector, scenario):
    return float((vector**2).sum()), {"norm_squared": float((vector**2).sum())}


if __name__ == "__main__":
    result = Evolution(agent_factory=VectorAgentFactory(8), evaluator=ObjectiveEvaluator(sphere),
                       config=EvolutionConfig(population_size=24, generations=12, maximize=False,
                                              evaluation_episodes=1, output_dir="results/vector"),
                       mutation=GaussianMutation(parameter_probability=0.5, std=0.15)).run()
    result.best_agent.save("results/vector/champion")
    print(f"Best squared distance: {result.best_fitness:.6f}; evaluated generations: {len(result.history)}")

