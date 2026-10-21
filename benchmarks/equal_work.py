"""Bounded complete-run benchmark; run from the repo after installing geneticevo.

python benchmarks/equal_work.py --populations 4 16 --copies 1 4 --repeats 3
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from geneticevo import Evolution, EvolutionConfig
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.results import software_metadata, write_json


def synchronize(device):
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def run_one(mode, population, copies, episodes, generations, seed):
    from geneticevo.tensor import TensorReach1D
    device = "cuda" if mode.endswith("cuda") else "cpu"
    backend = "parallel" if mode == "parallel_cpu" else "serial" if mode == "serial_cpu" else "batched"
    native = "native" in mode
    config = EvolutionConfig(population_size=population, generations=generations, elite_count=1,
                             env_copies=copies, evaluation_episodes=episodes, seed=seed, archive_size=1)
    synchronize(device)
    started = time.perf_counter()
    experiment = Evolution(env_factory=partial(Reach1D, horizon=16), agent_factory=MLPAgentFactory((16, 16)),
        config=config, backend=backend, device=device, environment_device=device if native else "cpu",
        tensor_env_factory=partial(TensorReach1D, horizon=16) if native else None)
    result = experiment.run()
    synchronize(device)
    elapsed = time.perf_counter() - started
    interactions = result.history[-1]["evaluation_interactions"]
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    return {"mode": mode, "population": population, "env_copies": copies, "episodes": episodes,
            "generations": generations, "seed": seed, "complete_seconds": elapsed,
            "interactions": interactions, "interactions_per_second": interactions / elapsed,
            "best_fitness": result.best_fitness, "rss_after_bytes": rss,
            "component_seconds": {key: sum(row["timings"][key] for row in result.history) for key in result.history[0]["timings"]},
            "compilation_seconds": 0.0, "transfer_seconds": None if mode.startswith("hybrid") else 0.0,
            "cuda_peak_allocated_bytes": __import__("torch").cuda.max_memory_allocated() if device == "cuda" else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--populations", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default="results/benchmarks/equal_work.json")
    args = parser.parse_args()
    import torch
    from geneticevo.tensor import BatchedPolicy
    torch.set_num_threads(1)
    warmup_started = time.perf_counter()
    env = Reach1D()
    agent = MLPAgentFactory((16, 16))(np.random.default_rng(0), env.observation_space, env.action_space)
    for _ in range(5):
        BatchedPolicy([agent])(torch.zeros(1, 1, 1))
    modes = ["serial_cpu", "parallel_cpu", "batched_python_cpu", "batched_native_cpu"]
    if torch.cuda.is_available():
        modes += ["hybrid_cuda", "batched_native_cuda"]
        for _ in range(5):
            BatchedPolicy([agent], "cuda")(torch.zeros(1, 1, 1, device="cuda"))
        torch.cuda.synchronize()
    warmup = time.perf_counter() - warmup_started
    records = []
    for population in args.populations:
        for copies in args.copies:
            for repeat in range(args.repeats):
                # Each mode receives identical P, episodes, horizons, generations and seed.
                for mode in modes:
                    if mode.endswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    record = run_one(mode, population, copies, 4, 2, 42 + repeat)
                    records.append(record)
                    print(f"{mode:22} P={population:3} E={copies} {record['complete_seconds']:.4f}s {record['interactions']} transitions", flush=True)
    write_json(args.output, {"metadata": software_metadata(), "warmup_seconds": warmup,
        "cuda_verified": torch.cuda.is_available(), "records": records,
        "notes": "Timing includes initialization, archive reevaluation and result copies; imports/warmup excluded. RSS is parent RSS after run, not peak or child-process memory. Hybrid transfer time is not separately instrumented. No compilation or mixed precision."})

