"""Compare equal-work CPU PPO on growing networks using complete Evolution runs."""
from __future__ import annotations

import argparse
import cProfile
from functools import partial
from itertools import product
import os
import platform
import pstats
from statistics import median
import time

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth, RustNativePPO
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.results import software_metadata, write_json


def make_experiment(args, variant, population, width, seed, *, measure=False):
    gym = partial(Reach1D, horizon=args.horizon, continuous=args.continuous)
    learner_options = dict(env_copies=args.copies, rollout_steps=args.rollout_steps, epochs=args.epochs,
                           minibatch_size=args.minibatch_size, measure_phases=measure)
    tensor = None
    if variant == "rust":
        backend = "rust"
        learner = RustNativePPO(**learner_options)
    else:
        from geneticevo.ppo import NativePPO
        backend = "batched"
        learner = NativePPO(accelerated=variant != "native", **learner_options)
        if variant == "tensor":
            from geneticevo.tensor import TensorReach1D
            tensor = partial(TensorReach1D, horizon=args.horizon, continuous=args.continuous)
    growth = None if args.fixed else NeuralGrowth(
                          check_every=args.growth_every, neurons_per_growth=2, max_hidden_size=width + 16,
                          min_hidden_layers=0, max_hidden_layers=max(4, args.layers), new_layer_size=4,
                          shrink_probability=0.2, layer_add_probability=0.25, layer_remove_probability=0.25,
                          layer_mutation_probability=0.25, mutation_probability=0.25)
    return Evolution(env_factory=gym, tensor_env_factory=tensor, backend=backend, learner=learner,
                     agent_factory=MLPAgentFactory((width,) * args.layers, growth),
                     config=EvolutionConfig(population_size=population, generations=args.generations,
                         elite_count=1, archive_size=1, seed=seed, learning_budget=args.budget,
                         env_copies=args.copies, evaluation_episodes=2, max_episode_steps=args.horizon))


def sample(args, variant, population, width, seed):
    experiment = make_experiment(args, variant, population, width, seed)
    start = time.perf_counter()
    result = experiment.run()
    elapsed = time.perf_counter() - start
    history = result.history
    return dict(seed=seed, generation_seconds=elapsed / len(history),
                learning_seconds=sum(row["timings"]["learning_seconds"] for row in history) / len(history),
                training_interactions=history[-1]["training_interactions"],
                evaluation_interactions=history[-1]["evaluation_interactions"],
                learning_updates=sum(row["learning_updates"] for row in history),
                best_fitness=result.best_fitness, final_mean_fitness=history[-1]["mean"])


def run_profile(args, variant, population, width):
    profile = cProfile.Profile()
    result = profile.runcall(make_experiment(args, variant, population, width, args.seeds[0], measure=True).run)
    print({key: round(sum(row["timings"][key] for row in result.history), 5)
           for key in result.history[0]["timings"]})
    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    stats.print_stats(28)
    stats.print_stats("backward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, nargs="+", default=[16, 128, 1024])
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--variants", nargs="+", choices=["rust", "native", "accelerated", "tensor"],
                        default=["rust", "accelerated"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    for name, default in dict(generations=2, budget=64, copies=2, layers=2, rollout_steps=8,
                              epochs=2, minibatch_size=16, horizon=32, growth_every=32, threads=4).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--fixed", action="store_true", help="Use fixed widths and depths")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--machine", default=os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()))
    parser.add_argument("--output", default="results/benchmarks/rust_vs_python.json")
    args = parser.parse_args()
    numeric = (args.populations + args.widths + [args.generations, args.budget, args.copies, args.layers,
        args.rollout_steps, args.epochs, args.minibatch_size, args.horizon, args.growth_every, args.threads])
    if min(numeric) < 1 or args.copies * args.rollout_steps < 2:
        parser.error("Sizes, budgets, and thread counts must be positive, with at least two transitions per rollout")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    if any(variant != "rust" for variant in args.variants):
        import torch
        torch.set_num_threads(args.threads)
    records = []
    for population, width in product(args.populations, args.widths):
        for variant in args.variants:
            make_experiment(args, variant, population, width, args.seeds[0]).run()
            samples = [sample(args, variant, population, width, seed) for seed in args.seeds]
            metrics = {key: median(row[key] for row in samples) for key in samples[0] if key != "seed"}
            record = dict(population=population, width=width, variant=variant, median=metrics, samples=samples)
            records.append(record)
            print(f"P={population:4} W={width:3} {variant:11} {metrics['generation_seconds']:.4f}s/generation "
                  f"{metrics['learning_seconds']:.4f}s learning  fitness={metrics['best_fitness']:.4f}", flush=True)
            write_json(args.output, dict(machine=args.machine, software=software_metadata(), settings=vars(args),
                                         records=records))
            if args.profile:
                run_profile(args, variant, population, width)


if __name__ == "__main__":
    main()
