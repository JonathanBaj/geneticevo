"""Serial reference and process-based CPU evaluation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _evaluate(payload):
    evaluator, agent, scenarios, action_seed = payload
    return evaluator.evaluate(agent, scenarios, action_seed)


def evaluate_population(evaluator, agents, scenarios, action_seed, backend="serial", workers=2):
    payloads = [(evaluator, agent, scenarios, action_seed) for agent in agents]
    if backend == "serial":
        return [_evaluate(x) for x in payloads]
    if backend != "parallel":
        raise ValueError(f"Unsupported evaluation backend: {backend}")
    # spawn is explicit, so Linux and Windows have the same factory requirements.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_evaluate, payloads))

