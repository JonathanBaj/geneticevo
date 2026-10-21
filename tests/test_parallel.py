import unittest
import numpy as np

from geneticevo.agents import VectorAgentFactory
from geneticevo.core import Scenario
from geneticevo.evaluation import ObjectiveEvaluator
from geneticevo.execution import evaluate_population


def objective(vector, scenario):
    return float(np.square(vector).sum()) + scenario.seed / 1000


class ParallelTests(unittest.TestCase):
    def test_process_evaluation_matches_serial(self):
        agents = [VectorAgentFactory(3)(np.random.default_rng(i)) for i in range(4)]
        evaluator = ObjectiveEvaluator(objective)
        scenarios = [Scenario(1), Scenario(2)]
        serial = evaluate_population(evaluator, agents, scenarios, 1)
        parallel = evaluate_population(evaluator, agents, scenarios, 1, "parallel", 2)
        self.assertEqual([x.fitness for x in serial], [x.fitness for x in parallel])


if __name__ == "__main__":
    unittest.main()
