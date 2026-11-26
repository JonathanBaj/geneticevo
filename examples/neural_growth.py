"""Optional heritable MLP growth with Rust CPU PPO by default."""
import argparse
from functools import partial

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D

if __package__:
    from ._ppo_backend import add_backend_arguments, ppo_options
else:
    from _ppo_backend import add_backend_arguments, ppo_options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser, allow_accelerated=True)
    parser.add_argument('--inheritance', choices=('lamarckian', 'baldwinian'), default='lamarckian')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--dynamic-depth', action='store_true',
                        help='Enable lifetime shrinking and hidden-layer addition/removal')
    parser.add_argument('--value-loss-threshold', type=float, default=None,
                        help='Only grow when mean PPO value loss meets this threshold')
    args = parser.parse_args()
    structural = (dict(shrink_probability=0.2, layer_add_probability=0.25, layer_remove_probability=0.25,
                       layer_mutation_probability=0.25, min_hidden_layers=0, max_hidden_layers=4,
                       new_layer_size=4) if args.dynamic_depth else {})
    growth = NeuralGrowth(check_every=32, neurons_per_growth=2, max_hidden_size=16,
                          value_loss_threshold=args.value_loss_threshold,
                          mutation_probability=0.25, mutation_step=2, **structural)
    result = Evolution(
        env_factory=partial(Reach1D, continuous=args.continuous),
        agent_factory=MLPAgentFactory(hidden_sizes=(4, 4), growth=growth),
        device=args.device, environment_device=args.device,
        config=EvolutionConfig(population_size=4, generations=3, elite_count=1,
            learning_budget=64, env_copies=2, evaluation_episodes=2,
            inheritance=args.inheritance, output_dir='results/neural_growth'),
        **ppo_options(parser, args, env_copies=2, rollout_steps=8, epochs=2, minibatch_size=16),
    ).run()
    for record in result.history:
        print(f"Generation {record['generation']}: hidden sizes={record['network_hidden_sizes']}, "
              f"mean growth events={record['learner_metrics']['growth_events']:.2f}")
    result.best_agent.save('results/neural_growth/champion')
    print(f'Champion hidden sizes={result.best_agent.hidden_sizes}; fitness={result.best_fitness:.4f}')
