"""Equal-work independent versus population PPO smoke benchmark."""
from functools import partial
import argparse
import time

import numpy as np
import torch

from geneticevo.agents import MLPAgentFactory
from geneticevo.core import scenarios_for
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator
from geneticevo.ppo import NativePPO
from geneticevo.results import write_json, software_metadata
from geneticevo.tensor import TensorReach1D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    device = args.device
    torch.set_num_threads(1)
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable')

    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()

    env_factory = partial(Reach1D, horizon=16)
    tensor_factory = partial(TensorReach1D, horizon=16)
    env = env_factory()
    factory = MLPAgentFactory((16, 16))
    template = factory(np.random.default_rng(0), env.observation_space, env.action_space)
    NativePPO(rollout_steps=8, epochs=1, minibatch_size=8, device=device).learn_population([template], env_factory, tensor_factory, 8, 0, 0)
    synchronize()
    env.close()
    records = []
    for population in (4, 16):
        for copies in (1, 4):
            originals = [factory(np.random.default_rng(i), env.observation_space, env.action_space) for i in range(population)]
            learner = NativePPO(env_copies=copies, rollout_steps=16, epochs=2, minibatch_size=16, device=device)
            for mode in ("independent", "batched"):
                agents = [agent.clone(False) for agent in originals]
                synchronize()
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                if mode == "batched":
                    reports = learner.learn_population(agents, env_factory, tensor_factory, 64, 42, 0)
                else:
                    reports = [learner.learn_population([agent], env_factory, tensor_factory, 64, 42 + index, 0)[0]
                               for index, agent in enumerate(agents)]
                synchronize()
                elapsed = time.perf_counter() - start
                evaluations = [GymEvaluator(env_factory).evaluate(agent, scenarios_for(2026, "test", 2)) for agent in agents]
                record = {"mode": mode, "device": device, "population": population, "env_copies": copies,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                    "rollout_and_update_seconds": elapsed, "training_interactions": sum(r.interactions for r in reports),
                    "optimizer_updates_all_individuals": sum(r.updates for r in reports),
                    "frozen_evaluation_interactions": sum(r.interactions for r in evaluations),
                    "mean_test_fitness": float(np.mean([r.fitness for r in evaluations]))}
                records.append(record)
                print(record, flush=True)
    output = args.output or ('results/benchmarks/ppo_updates_cuda.json' if device == 'cuda' else
                             'results/benchmarks/ppo_updates.json')
    write_json(output, {"records": records, "metadata": software_metadata(), "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == 'cuda' else None,
        "notes": "Identical architectures, initial genes, budgets and schedules; stochastic training scenarios/action streams differ between routes. Frozen test suite is common. One timing per configuration; no superiority claim. CUDA timings are synchronized; imports and warm-up are excluded."})
