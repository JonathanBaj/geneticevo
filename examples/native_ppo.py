"""Tensor PPO on CPU by default: python examples/native_ppo.py --device cpu"""
import argparse
from functools import partial
import torch

from geneticevo import Evolution, EvolutionConfig
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.ppo import NativePPO
from geneticevo.tensor import TensorReach1D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--accelerated", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = Evolution(env_factory=partial(Reach1D, continuous=args.continuous),
        tensor_env_factory=partial(TensorReach1D, continuous=args.continuous),
        agent_factory=MLPAgentFactory((16, 16)), backend="batched", device=args.device,
        environment_device=args.device,
        learner=NativePPO(env_copies=2, rollout_steps=16, epochs=2, minibatch_size=16, device=args.device,
            accelerated=args.accelerated, population_chunk_size=args.chunk_size),
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1,
            env_copies=2, learning_budget=64, evaluation_episodes=2, output_dir="results/native_ppo")).run()
    print(f"Native PPO fitness={result.best_fitness:.4f}; interactions={result.history[-1]['training_interactions']}")

