from functools import partial
import unittest

import numpy as np

try:
    import torch
    from geneticevo.tensor import BatchedPolicy, TensorReach1D, evaluate_batched, resolve_device
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from geneticevo import Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator


@unittest.skipUnless(HAS_TORCH, "torch extra is not installed")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agents(self, count=3, continuous=False):
        env = Reach1D(continuous=continuous)
        return [MLPAgentFactory((4,))(np.random.default_rng(i), env.observation_space, env.action_space) for i in range(count)]

    def test_serial_batched_outputs_and_inference(self):
        for continuous in (False, True):
            agents = self.agents(continuous=continuous)
            observations = torch.linspace(-1, 1, 15).reshape(3, 5, 1)
            policy = BatchedPolicy(agents)
            actual = policy(observations).detach().numpy()
            expected = np.stack([agent.forward(observations[i].numpy()) for i, agent in enumerate(agents)])
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            scenarios = [Scenario(1), Scenario(2), Scenario(3)]
            env_factory = partial(Reach1D, horizon=5, continuous=continuous)
            factory = partial(TensorReach1D, horizon=5, continuous=continuous)
            evaluations = evaluate_batched(agents, scenarios, env_factory, factory, env_copies=2, max_steps=10)
            for agent, evaluation in zip(agents, evaluations):
                reference = GymEvaluator(env_factory).evaluate(agent, scenarios)
                np.testing.assert_allclose(evaluation.returns, reference.returns, atol=1e-6)
                self.assertEqual(evaluation.interactions, 15)

    def test_environment_transition_equivalence_and_mask(self):
        for continuous in (False, True):
            cpu = Reach1D(horizon=4, continuous=continuous)
            tensor = TensorReach1D(1, 1, horizon=4, continuous=continuous)
            observation, _ = cpu.reset(seed=2, options={"position": 0.4})
            np.testing.assert_array_equal(observation, tensor.reset([Scenario(2, {"position": 0.4})])[0, 0].numpy())
            for value in (0, 2, 1, 0):
                action = np.array([value / 2 - 0.5], np.float32) if continuous else value
                t_action = torch.tensor(action).reshape(1, 1, 1) if continuous else torch.tensor([[value]])
                o, r, term, trunc, _ = cpu.step(action)
                to, tr, tt, tx = tensor.step(t_action)
                np.testing.assert_allclose(o, to[0, 0].numpy(), atol=1e-7)
                self.assertAlmostEqual(r, tr.item(), places=6)
                self.assertEqual((term, trunc), (tt.item(), tx.item()))
            saved = tensor.position.clone()
            tensor.step(t_action, torch.zeros((1, 1), dtype=torch.bool))
            torch.testing.assert_close(saved, tensor.position)

    def test_no_cross_individual_gradients(self):
        policy = BatchedPolicy(self.agents())
        observations = torch.ones((3, 5, 1))
        policy(observations)[0].sum().backward()
        for parameter in policy.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad[1:], torch.zeros_like(parameter.grad[1:]))

    def test_reject_incompatible_configurations(self):
        agents = self.agents()
        incompatible = self.agents(1, continuous=True)
        with self.assertRaises(ValueError):
            BatchedPolicy([agents[0], incompatible[0]])
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
