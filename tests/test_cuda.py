"""Hardware validation for CUDA execution; skipped without a usable GPU."""
from functools import partial
import unittest

import numpy as np

from geneticevo import Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator

try:
    import torch
    from geneticevo.tensor import TensorReach1D, evaluate_batched
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def agents(continuous=False, widths=((4, 4),) * 3):
    env = Reach1D(continuous=continuous)
    try:
        return [MLPAgentFactory(size)(np.random.default_rng(i), env.observation_space, env.action_space)
                for i, size in enumerate(widths)]
    finally:
        env.close()


@unittest.skipUnless(HAS_CUDA, 'CUDA hardware unavailable')
class CUDATests(unittest.TestCase):
    def test_frozen_evaluation_native_and_hybrid_variable_widths(self):
        for continuous in (False, True):
            population = agents(continuous, ((3,), (5,), (3,)))
            before = [a.genome.vector() for a in population]
            scenarios = [Scenario(1), Scenario(2), Scenario(3)]
            factory = partial(Reach1D, horizon=5, continuous=continuous)
            for tensor_factory in (None, partial(TensorReach1D, horizon=5, continuous=continuous)):
                actual = evaluate_batched(population, scenarios, factory, tensor_factory, device='cuda', env_copies=2)
                for agent, score, genome in zip(population, actual, before):
                    expected = GymEvaluator(factory).evaluate(agent, scenarios)
                    np.testing.assert_allclose(score.returns, expected.returns, atol=3e-5, rtol=3e-5)
                    self.assertEqual(score.lengths, expected.lengths)
                    self.assertEqual(score.interactions, 15)
                    np.testing.assert_array_equal(genome, agent.genome.vector())


if __name__ == "__main__":
    unittest.main()
