"""Optional PPO/DQN: python examples/sb3_learning.py --algorithm PPO"""
import argparse

import torch

from geneticevo import Evolution, EvolutionConfig, GaussianMutation
from geneticevo.environments import Reach1D
from geneticevo.integrations.sb3 import SB3AgentFactory, SB3Learner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("PPO", "DQN"), default="PPO")
    parser.add_argument("--inheritance", choices=("lamarckian", "baldwinian"), default="lamarckian")
    options = parser.parse_args()
    torch.set_num_threads(1)
    result = Evolution(env_factory=Reach1D,
        agent_factory=SB3AgentFactory(options.algorithm, {"policy_kwargs": {"net_arch": [16, 16]}}),
        learner=SB3Learner(env_copies=1), mutation=GaussianMutation(0.05, 0.02),
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1, learning_budget=64,
            env_copies=1, inheritance=options.inheritance, evaluation_episodes=2,
            measure_pre_learning=True, output_dir=f"results/sb3_{options.algorithm.lower()}")).run()
    result.best_agent.save(f"results/sb3_{options.algorithm.lower()}/champion")
    print(f"{options.algorithm}: fitness={result.best_fitness:.4f}; training interactions={result.history[-1]['training_interactions']}")

