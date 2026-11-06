"""Equal-work native/accelerated PPO matrix, complete generations, and profiles."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import statistics
import time

import torch

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.ppo import NativePPO
from geneticevo.results import write_json, software_metadata
from geneticevo.tensor import TensorReach1D


VARIANTS = {
    "native": {"accelerated": False},
    "accelerated": {"accelerated": True},
    "no-reuse": {"accelerated": True, "reuse_rollout_values": False},
    "no-buffers": {"accelerated": True, "preallocate_rollouts": False},
    "no-batch-rng": {"accelerated": True, "batch_randomness": False},
}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def make_experiment(args, population, width, copies, device, variant, seed, measure=True):
    growth = (NeuralGrowth(check_every=args.growth_every, max_hidden_size=args.growth_max,
                          neurons_per_growth=args.growth_step, mutation_probability=0)
              if args.growth_max is not None else None)
    learner = NativePPO(env_copies=copies, rollout_steps=args.rollout_steps,
        epochs=args.epochs, minibatch_size=args.minibatch_size, device=device,
        measure_phases=measure, **VARIANTS[variant])
    if learner.accelerated:
        learner = replace(learner, population_chunk_size=args.chunk_size)
    return Evolution(agent_factory=MLPAgentFactory((width,) * args.layers, growth),
        env_factory=partial(Reach1D, horizon=args.horizon, continuous=args.continuous),
        tensor_env_factory=partial(TensorReach1D, horizon=args.horizon, continuous=args.continuous),
        backend="batched", device=device, environment_device=device, learner=learner,
        config=EvolutionConfig(population_size=population, generations=args.generations,
            elite_count=1, archive_size=1, evaluation_episodes=args.evaluation_episodes,
            learning_budget=args.budget, env_copies=copies, seed=seed, diversity_pairs=0,
            max_episode_steps=args.horizon))


def run_sample(args, population, width, copies, device, variant, seed):
    experiment = make_experiment(args, population, width, copies, device, variant, seed)
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = experiment.run()
    synchronize(device)
    elapsed = time.perf_counter() - started
    sample = {name + "_seconds": sum(row["timings"].get("learning_" + name + "_seconds", 0)
        for row in result.history) / args.generations for name in ("setup", "rollout", "update", "growth", "export")}
    training_seconds = sum(row["timings"]["learning_seconds"] for row in result.history)
    training = result.history[-1]["training_interactions"]
    evaluation = result.history[-1]["evaluation_interactions"]
    sample.update(full_generation_seconds=elapsed / args.generations,
        learning_seconds=training_seconds / args.generations,
        transitions_per_second=training / training_seconds,
        generation_training_transitions_per_second=training / elapsed,
        total_interactions_per_second=(training + evaluation) / elapsed,
        training_interactions=training, evaluation_interactions=evaluation,
        optimizer_updates_all_individuals=sum(row["learning_updates"] for row in result.history),
        mean_fitness=result.history[-1]["mean"],
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)
    return sample


def summarize(records):
    """Paired speedups and the first measured CPU/CUDA crossover; no estimates."""
    indexed = {(r["population"], r["width"], r["env_copies"], r["device"], r["variant"]): r
               for r in records if r.get("status") == "ok"}
    comparisons, crossovers, ablations = [], [], []
    for key, record in indexed.items():
        population, width, copies, device, variant = key
        identity = dict(population=population, width=width, env_copies=copies, device=device, variant=variant)
        for ref_device, ref_variant, label in (
                (device, "native", "versus_native_same_device"),
                ("cpu", "native", "versus_native_cpu"),
                ("cpu", variant, "versus_same_variant_cpu")):
            reference = indexed.get((population, width, copies, ref_device, ref_variant))
            if reference:
                comparisons.append({**identity, "comparison": label,
                    "training_speedup": reference["median"]["learning_seconds"] / record["median"]["learning_seconds"],
                    "generation_speedup": reference["median"]["full_generation_seconds"] / record["median"]["full_generation_seconds"]})
        optimized = indexed.get((population, width, copies, device, "accelerated"))
        if optimized and variant in ("no-reuse", "no-buffers", "no-batch-rng"):
            ablations.append({**identity,
                "training_speedup_when_enabled": record["median"]["learning_seconds"] / optimized["median"]["learning_seconds"],
                "rollout_speedup_when_enabled": record["median"]["rollout_seconds"] / optimized["median"]["rollout_seconds"]})
    for width, copies, variant in sorted({(k[1], k[2], k[4]) for k in indexed}):
        paired = []
        for population in sorted({k[0] for k in indexed}):
            cpu = indexed.get((population, width, copies, "cpu", variant))
            gpu = indexed.get((population, width, copies, "cuda", variant))
            if cpu and gpu:
                paired.append((population, cpu, gpu))
        if paired:
            crossovers.append({"width": width, "env_copies": copies, "variant": variant,
                "first_training_win_population": next((p for p, c, g in paired
                    if g["median"]["learning_seconds"] < c["median"]["learning_seconds"]), None),
                "first_generation_win_population": next((p for p, c, g in paired
                    if g["median"]["full_generation_seconds"] < c["median"]["full_generation_seconds"]), None)})
    return {"comparisons": comparisons, "observed_crossovers": crossovers, "ablations": ablations}


def profile_run(args, device, variant):
    population = args.profile_population or max(args.populations)
    width, copies = max(args.widths), max(args.copies)
    for _ in range(args.warmups):
        make_experiment(args, population, width, copies, device, variant, args.seed, measure=False).run()
    experiment = make_experiment(args, population, width, copies, device, variant, args.seed, measure=False)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, record_shapes=args.profile_detailed,
                                profile_memory=args.profile_detailed) as profiler:
        with torch.profiler.record_function("complete_evolution_run"):
            experiment.run()
        synchronize(device)
    directory = Path(args.profile_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{device}_{variant}_p{population}_w{width}_e{copies}"
    trace = directory / (stem + ".trace.json")
    profiler.export_chrome_trace(str(trace))
    print(f"Profile: {trace}", flush=True)
    # Aggregating millions of CPU events can dwarf the capture itself and use
    # several GiB of RAM. Keep the raw trace as the scalable default.
    if args.profile_table:
        table = profiler.key_averages().table(sort_by="self_device_time_total" if device == "cuda" else "self_cpu_time_total", row_limit=40)
        (directory / (stem + ".txt")).write_text(table, encoding="utf-8")
        print(table, flush=True)
    return {"device": device, "variant": variant, "population": population,
            "width": width, "env_copies": copies, "trace": str(trace)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", "--device", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--populations", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--widths", type=int, nargs="+", default=[64])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--copies", "--env-copies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["native", "accelerated"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--growth-max", type=int)
    parser.add_argument("--growth-every", type=int, default=128)
    parser.add_argument("--growth-step", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-detailed", action="store_true",
                        help="Also record shapes and memory; can be expensive for populations with many small operations")
    parser.add_argument("--profile-table", action="store_true",
                        help="Also aggregate an operator table; can take minutes and several GiB for large traces")
    parser.add_argument("--profile-population", type=int)
    parser.add_argument("--profile-dir", default="results/benchmarks/ppo_profiles")
    parser.add_argument("--output", default="results/benchmarks/ppo_updates.json")
    args = parser.parse_args()
    for name in ("populations", "widths", "copies"):
        if any(n < 1 for n in getattr(args, name)):
            parser.error(f"--{name} must be positive")
    for name in ("layers", "rollout_steps", "epochs", "minibatch_size", "budget", "horizon", "generations",
                 "evaluation_episodes", "warmups", "repeats", "threads", "growth_every", "growth_step"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("chunk_size", "growth_max", "profile_population"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.growth_max is not None and args.growth_max < max(args.widths):
        parser.error("--growth-max must cover all initial widths")
    if "cuda" in args.devices and not torch.cuda.is_available() and args.devices == ["cuda"]:
        parser.error("CUDA was requested but is unavailable")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    records, profiles = [], []
    metadata = software_metadata()
    metadata.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        threads=torch.get_num_threads(), arguments=vars(args))
    notes = ["TensorReach1D simulation and policy run on the selected device. Warmups and profiling are excluded.",
        "Each sample runs complete Evolution generations, including initialization, selection/reproduction, frozen validation, CPU archive evaluation, and final result copies; wall time is divided by generation count.",
        "transitions_per_second = actual training interactions / learning wall time; generation_training_transitions_per_second includes complete generation overhead.",
        "Phase wall times synchronize CUDA at both boundaries; synchronization perturbs overlap. Production learning does not synchronize phases.",
        "Initial genes, budgets, rollout lengths, minibatches, epochs, and evaluation scenarios match. Native and accelerated random streams differ; this measures equal work, not identical trajectories.",
        "CUDA crossovers are observations at sampled populations. OOM is reported, never retried with reduced work."]

    def save():
        write_json(args.output, {"records": records, "profiles": profiles, "metadata": metadata,
                               "notes": notes, **summarize(records)})

    for population in args.populations:
        for width in args.widths:
            for copies in args.copies:
                expected_work = None
                for device in args.devices:
                    for variant in args.variants:
                        record = {"population": population, "width": width, "hidden_sizes": [width] * args.layers,
                            "env_copies": copies, "device": device, "variant": variant,
                            "chunk_size": args.chunk_size if variant != "native" else None}
                        if device == "cuda" and not torch.cuda.is_available():
                            record.update(status="skipped", reason="CUDA unavailable")
                        else:
                            try:
                                print(f"Running p={population} w={width} e={copies} {device}/{variant}", flush=True)
                                for _ in range(args.warmups):
                                    run_sample(args, population, width, copies, device, variant, args.seed)
                                samples = [run_sample(args, population, width, copies, device, variant, args.seed)
                                           for _ in range(args.repeats)]
                                work = {tuple(s[k] for k in ("training_interactions", "evaluation_interactions", "optimizer_updates_all_individuals")) for s in samples}
                                if len(work) != 1 or (expected_work is not None and next(iter(work)) != expected_work):
                                    raise RuntimeError("Unequal environment interactions or PPO updates across compared routes")
                                expected_work = next(iter(work))
                                record.update(status="ok", samples=samples, median={key: statistics.median(s[key] for s in samples)
                                    if samples[0][key] is not None else None for key in samples[0]})
                                print({**{k: record[k] for k in ("population", "width", "env_copies", "device", "variant")},
                                       **record["median"]}, flush=True)
                            except torch.cuda.OutOfMemoryError:
                                record.update(status="out_of_memory", reason="Reduce --chunk-size for accelerated variants")
                                torch.cuda.empty_cache()
                        records.append(record)
                        save()
    if args.profile:
        for device in args.devices:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for variant in args.variants:
                try:
                    profiles.append(profile_run(args, device, variant))
                except torch.cuda.OutOfMemoryError:
                    profiles.append({"device": device, "variant": variant, "status": "out_of_memory"})
                    torch.cuda.empty_cache()
                save()
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
