"""Small, multi-seed comparisons with explicit actual costs; not a research claim.

python benchmarks/baselines.py --seeds 1 2 3
"""
import argparse
import time

import numpy as np

from geneticevo import Evolution, EvolutionConfig
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
        parser.error("Use at least two independent seeds for uncertainty summaries")
    records = []
    final_scenarios = scenarios_for(2026, "test", 8)
    evaluator = GymEvaluator(Reach1D)
    for seed in args.seeds:
        for mode in ("random", "evolution"):
            start = time.perf_counter()
            if mode == "random":
                result = evaluator.evaluate(RandomAgent(), final_scenarios, seed)
                record = {"mode": mode, "seed": seed, "test_fitness": result.fitness,
                          "training_interactions": 0, "evaluation_interactions": result.interactions}
            else:
                experiment = Evolution(env_factory=Reach1D,
                    agent_factory=MLPAgentFactory((16, 16)),
                    config=EvolutionConfig(population_size=4, generations=2, elite_count=1, seed=seed,
                        evaluation_episodes=2))
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
        "notes": "Random and evolved policies share a final test suite. Evolution also spends interactions on selection and archive evaluation. Communication uses a separate shared task. These small runs are descriptive."})
