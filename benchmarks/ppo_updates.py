"""Equal-work independent versus population PPO smoke benchmark."""
from functools import partial
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
    torch.set_num_threads(1)
    env_factory = partial(Reach1D, horizon=16)
    tensor_factory = partial(TensorReach1D, horizon=16)
    env = env_factory()
    factory = MLPAgentFactory((16, 16))
    template = factory(np.random.default_rng(0), env.observation_space, env.action_space)
    NativePPO(rollout_steps=8, epochs=1, minibatch_size=8).learn_population([template], env_factory, tensor_factory, 8, 0, 0)
    records = []
    for population in (4, 16):
        for copies in (1, 4):
            originals = [factory(np.random.default_rng(i), env.observation_space, env.action_space) for i in range(population)]
            learner = NativePPO(env_copies=copies, rollout_steps=16, epochs=2, minibatch_size=16)
            for mode in ("independent", "batched"):
                agents = [agent.clone(False) for agent in originals]
                start = time.perf_counter()
                if mode == "batched":
                    reports = learner.learn_population(agents, env_factory, tensor_factory, 64, 42, 0)
                else:
                    reports = [learner.learn_population([agent], env_factory, tensor_factory, 64, 42 + index, 0)[0]
                               for index, agent in enumerate(agents)]
                elapsed = time.perf_counter() - start
                evaluations = [GymEvaluator(env_factory).evaluate(agent, scenarios_for(2026, "test", 2)) for agent in agents]
                record = {"mode": mode, "population": population, "env_copies": copies,
                    "rollout_and_update_seconds": elapsed, "training_interactions": sum(r.interactions for r in reports),
                    "optimizer_updates_all_individuals": sum(r.updates for r in reports),
                    "frozen_evaluation_interactions": sum(r.interactions for r in evaluations),
                    "mean_test_fitness": float(np.mean([r.fitness for r in evaluations]))}
                records.append(record)
                print(record, flush=True)
    write_json("results/benchmarks/ppo_updates.json", {"records": records, "metadata": software_metadata(),
        "notes": "Identical architectures, initial genes, budgets and schedules; stochastic training scenarios/action streams differ between routes. Frozen test suite is common. One timing per configuration; no superiority claim. CPU only."})
