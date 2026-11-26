"""Train an MLP with Rust CPU PPO by default."""
import argparse
from functools import partial

from geneticevo import Evolution, EvolutionConfig
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D

if __package__:
    from ._ppo_backend import add_backend_arguments, ppo_options
else:
    from _ppo_backend import add_backend_arguments, ppo_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser, allow_accelerated=True)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    result = Evolution(env_factory=partial(Reach1D, continuous=args.continuous),
        agent_factory=MLPAgentFactory((16, 16)), device=args.device,
        environment_device=args.device,
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1,
            env_copies=2, learning_budget=64, evaluation_episodes=2, output_dir="results/native_ppo"),
        **ppo_options(parser, args, env_copies=2, rollout_steps=16, epochs=2, minibatch_size=16),
    ).run()
    print(f"Native PPO fitness={result.best_fitness:.4f}; interactions={result.history[-1]['training_interactions']}")

