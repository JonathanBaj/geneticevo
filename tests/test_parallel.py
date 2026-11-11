import unittest
from functools import partial
import numpy as np

from geneticevo.agents import VectorAgentFactory
from geneticevo.core import Scenario
from geneticevo.evaluation import ObjectiveEvaluator
from geneticevo.execution import evaluate_population
from geneticevo import GraphAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator


def objective(vector, scenario):
    return float(np.square(vector).sum()) + scenario.seed / 1000


class ParallelTests(unittest.TestCase):
    def test_graph_process_evaluation_matches_serial(self):
        env = Reach1D()
        try:
            agents = [GraphAgentFactory(3, 8)(np.random.default_rng(i),
                env.observation_space, env.action_space) for i in range(3)]
        finally:
            env.close()
        evaluator = GymEvaluator(partial(Reach1D, horizon=3))
        scenarios = [Scenario(1), Scenario(2)]
        serial = evaluate_population(evaluator, agents, scenarios, 1)
        parallel = evaluate_population(evaluator, agents, scenarios, 1, 'parallel', 2)
        self.assertEqual([x.returns for x in serial], [x.returns for x in parallel])

    def test_process_evaluation_matches_serial(self):
        agents = [VectorAgentFactory(3)(np.random.default_rng(i)) for i in range(4)]
        evaluator = ObjectiveEvaluator(objective)
        scenarios = [Scenario(1), Scenario(2)]
        serial = evaluate_population(evaluator, agents, scenarios, 1)
        parallel = evaluate_population(evaluator, agents, scenarios, 1, "parallel", 2)
        self.assertEqual([x.fitness for x in serial], [x.fitness for x in parallel])


if __name__ == "__main__":
    unittest.main()
