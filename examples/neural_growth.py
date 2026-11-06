"""Optional heritable MLP growth: python examples/neural_growth.py --device cpu"""
import argparse
from functools import partial

import torch

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.ppo import NativePPO
from geneticevo.tensor import TensorReach1D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--inheritance', choices=('lamarckian', 'baldwinian'), default='lamarckian')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--accelerated', action='store_true')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--value-loss-threshold', type=float, default=None,
                        help='Only grow when mean PPO value loss meets this threshold')
    args = parser.parse_args()
    torch.set_num_threads(1)
    growth = NeuralGrowth(check_every=32, neurons_per_growth=2, max_hidden_size=16,
                          value_loss_threshold=args.value_loss_threshold,
                          mutation_probability=0.25, mutation_step=2)
    result = Evolution(
        env_factory=partial(Reach1D, continuous=args.continuous),
        tensor_env_factory=partial(TensorReach1D, continuous=args.continuous),
        agent_factory=MLPAgentFactory(hidden_sizes=(4, 4), growth=growth),
        learner=NativePPO(env_copies=2, rollout_steps=8, epochs=2, minibatch_size=16, device=args.device,
            accelerated=args.accelerated, population_chunk_size=args.chunk_size),
        backend='batched', device=args.device, environment_device=args.device,
        config=EvolutionConfig(population_size=4, generations=3, elite_count=1,
            learning_budget=64, env_copies=2, evaluation_episodes=2,
            inheritance=args.inheritance, output_dir='results/neural_growth'),
    ).run()
    for record in result.history:
        print(f"Generation {record['generation']}: hidden sizes={record['network_hidden_sizes']}, "
              f"mean growth events={record['learner_metrics']['growth_events']:.2f}")
    result.best_agent.save('results/neural_growth/champion')
    print(f'Champion hidden sizes={result.best_agent.hidden_sizes}; fitness={result.best_fitness:.4f}')
