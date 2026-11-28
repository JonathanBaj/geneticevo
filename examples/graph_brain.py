"""Evolve sparse graph policies with neuron and connection mutations."""
import argparse
from functools import partial

from geneticevo import Evolution, EvolutionConfig, GraphAgentFactory, GraphGrowth
from geneticevo.environments import Reach1D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--backend", choices=("serial", "parallel", "batched", "rust"), default="serial")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    factory = GraphAgentFactory(initial_neurons=8, initial_connections=24, settling_steps=4,
        growth=GraphGrowth(min_neurons=0, max_neurons=24, max_connections=128, mutation_probability=0.3))
    result = Evolution(
        agent_factory=factory,
        env_factory=partial(Reach1D, continuous=args.continuous),
        backend=args.backend, device=args.device,
        config=EvolutionConfig(population_size=8, generations=3, elite_count=1,
                               evaluation_episodes=2, output_dir="results/graph_brain"),
    ).run()
    for record in result.history:
        print(f"Generation {record['generation']}: neurons={record['network_neuron_counts']}, "
              f"connections={record['network_connection_counts']}, fitness={record['mean']:.3f}")
    result.best_agent.save("results/graph_brain/champion")
    print(f"Champion saved; fitness={result.best_fitness:.4f}")
