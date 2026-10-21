"""Small, multi-seed comparisons with explicit actual costs; not a research claim.

python benchmarks/baselines.py --seeds 1 2 3
"""
import argparse
from functools import partial
import time

import numpy as np

from geneticevo import Evolution, EvolutionConfig, Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.core import scenarios_for
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator
from geneticevo.results import uncertainty_summary, write_json
from geneticevo.shared import SharedEvaluator, SignalWorld, SignalPartner, MessageChannel, observed_signal_message


class RandomAgent:
    def act(self, observation, rng, deterministic=True):
        return int(rng.integers(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    if len(args.seeds) < 2:
        parser.error("Use at least two independent training seeds for uncertainty summaries")
    import torch
    from geneticevo.integrations.sb3 import SB3AgentFactory, SB3Learner
    torch.set_num_threads(1)
    records = []
    final_scenarios = scenarios_for(2026, "test", 8)
    evaluator = GymEvaluator(Reach1D)
    for seed in args.seeds:
        for mode in ("random", "evolution", "rl_only", "evolution_rl"):
            start = time.perf_counter()
            if mode == "random":
                result = evaluator.evaluate(RandomAgent(), final_scenarios, seed)
                record = {"mode": mode, "seed": seed, "test_fitness": result.fitness,
                          "training_interactions": 0, "evaluation_interactions": result.interactions}
            else:
                learning = mode in ("rl_only", "evolution_rl")
                size = 1 if mode == "rl_only" else 4
                experiment = Evolution(env_factory=Reach1D,
                    agent_factory=SB3AgentFactory("PPO", {"policy_kwargs": {"net_arch": [16, 16]}}) if learning else MLPAgentFactory((16, 16)),
                    learner=SB3Learner() if learning else None,
                    config=EvolutionConfig(population_size=size, generations=2, elite_count=1, seed=seed,
                        evaluation_episodes=2, learning_budget=256 if mode == "rl_only" else 64 if learning else 0))
                result = experiment.run()
                final = result.evaluate(evaluator, final_scenarios, seed)
                record = {"mode": mode, "seed": seed, "test_fitness": final.fitness,
                          "training_interactions": result.history[-1]["training_interactions"],
                          "evaluation_interactions": result.history[-1]["evaluation_interactions"] + final.interactions}
            record["wall_seconds"] = time.perf_counter() - start
            records.append(record)
            print(record, flush=True)
        # Communication ablation is a separate shared task with fixed policies.
        for enabled in (False, True):
            start = time.perf_counter()
            shared = SharedEvaluator(SignalWorld, (SignalPartner(),), MessageChannel() if enabled else None,
                                     observed_signal_message if enabled else None)
            evaluation = shared.evaluate(SignalPartner(), scenarios_for(seed, "test", 16), seed)
            records.append({"mode": "communication" if enabled else "no_communication", "seed": seed,
                "test_fitness": evaluation.fitness, "training_interactions": 0,
                "evaluation_interactions": evaluation.interactions, "world_steps": evaluation.world_steps,
                "wall_seconds": time.perf_counter() - start})
    summaries = {mode: uncertainty_summary([r["test_fitness"] for r in records if r["mode"] == mode]) for mode in {r["mode"] for r in records}}
    write_json("results/benchmarks/baselines.json", {"records": records, "summaries": summaries,
        "notes": "RL-only and evolution+RL match lifetime-learning interactions; their evaluation budgets differ and are reported. Evolution-only uses no RL interactions. Random and communication ablations are frozen controls. Small smoke runs cannot establish superiority."})
