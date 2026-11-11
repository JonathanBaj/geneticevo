"""Evolve sparse brains: python examples/graph_brain.py --device cpu"""
import argparse
from functools import partial

import torch

from geneticevo import Evolution, EvolutionConfig, GraphAgentFactory, GraphGrowth
from geneticevo.environments import Reach1D
from geneticevo.ppo import NativePPO
from geneticevo.tensor import TensorReach1D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--inheritance', choices=('lamarckian', 'baldwinian'), default='lamarckian')
    parser.add_argument('--continuous', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    factory = GraphAgentFactory(initial_neurons=8, initial_connections=24, settling_steps=4,
        growth=GraphGrowth(check_every=16, min_neurons=0, max_neurons=24,
            max_connections=128, mutation_probability=0.3))
    result = Evolution(
        agent_factory=factory,
        env_factory=partial(Reach1D, continuous=args.continuous),
        tensor_env_factory=partial(TensorReach1D, continuous=args.continuous),
        learner=NativePPO(rollout_steps=8, epochs=2, minibatch_size=16, device=args.device),
        backend='batched', device=args.device, environment_device=args.device,
        config=EvolutionConfig(population_size=4, generations=3, elite_count=1,
            learning_budget=64, evaluation_episodes=2, inheritance=args.inheritance,
            output_dir='results/graph_brain'),
    ).run()
    for record in result.history:
        print(f"Generation {record['generation']}: neurons={record['network_neuron_counts']}, "
              f"connections={record['network_connection_counts']}, fitness={record['mean']:.3f}")
    result.best_agent.save('results/graph_brain/champion')
    print(f'Champion saved; fitness={result.best_fitness:.4f}')
