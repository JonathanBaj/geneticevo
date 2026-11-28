"""Evolve MLPs whose hidden widths and depths can mutate at reproduction."""
import argparse
from functools import partial

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--dynamic-depth", action="store_true", help="Allow adding or removing hidden layers")
    parser.add_argument("--backend", choices=("serial", "parallel", "batched", "rust"), default="serial")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    growth = NeuralGrowth(max_hidden_size=16, mutation_probability=0.5, mutation_step=2,
                          layer_mutation_probability=0.5 if args.dynamic_depth else 0,
                          min_hidden_layers=0, max_hidden_layers=4, new_layer_size=4)
    result = Evolution(
        env_factory=partial(Reach1D, continuous=args.continuous),
        agent_factory=MLPAgentFactory(hidden_sizes=(4, 4), growth=growth),
        backend=args.backend, device=args.device,
        config=EvolutionConfig(population_size=8, generations=3, elite_count=1,
                               evaluation_episodes=2, output_dir="results/neural_growth"),
    ).run()
    for record in result.history:
        print(f"Generation {record['generation']}: hidden sizes={record['network_hidden_sizes']}")
    result.best_agent.save("results/neural_growth/champion")
    print(f"Champion hidden sizes={result.best_agent.hidden_sizes}; fitness={result.best_fitness:.4f}")
