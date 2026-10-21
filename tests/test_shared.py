from functools import partial
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, Scenario, load_agent
from geneticevo.agents import MLPAgentFactory
from geneticevo.shared import (MessageChannel, ParallelWorldAdapter, CommunicationAction, SignalWorld,
    SignalPartner, SharedFocalEnv, SharedEvaluator, CommunicatingAgentFactory, observed_signal_message)


def shared_env():
    return SharedFocalEnv(channel=MessageChannel(), fixed_message=observed_signal_message)


class SharedTests(unittest.TestCase):
    def test_routing_identity_timing_reset_and_cost(self):
        world = ParallelWorldAdapter(SignalWorld(horizon=3), MessageChannel(cost=0.1), observed_signal_message)
        observation, _ = world.reset(1, {"bits": [1, 0]})
        np.testing.assert_array_equal(observation["left"][1:], np.zeros(4))
        observation, rewards, _, _, _ = world.step({"left": 0, "right": 1}, np.random.default_rng(0))
        np.testing.assert_array_equal(observation["left"][1:], [0, 0, -1, 1])
        np.testing.assert_array_equal(observation["right"][1:], [1, 1, 0, 0])
        self.assertAlmostEqual(rewards["left"], 0.9)
        self.assertEqual(world.agent_transitions, 2)
        self.assertEqual(world.world_steps, 1)
        observation, _ = world.reset(2)
        np.testing.assert_array_equal(observation["left"][1:], np.zeros(4))

    def test_recipient_rule_and_invalid_messages(self):
        channel = MessageChannel(recipient_rule=lambda sender, recipient: sender == "a")
        routed = channel.route({"a": np.array([0.2]), "b": np.array([0.5])}, ["a", "b"], ["a", "b"], ["a", "b"], np.random.default_rng(0))
        np.testing.assert_array_equal(routed["a"], [0, 0, 0, 0])
        np.testing.assert_allclose(routed["b"], [0.2, 1, 0, 0])
        with self.assertRaises(ValueError):
            channel.route({"a": np.array([np.nan])}, ["a"], ["b"], ["a", "b"], np.random.default_rng(0))

    def test_shared_selection_team_partner_rotation_and_counts(self):
        agents = [SignalPartner(0), SignalPartner(1), SignalPartner(0)]
        scenarios = [Scenario(1), Scenario(2)]
        evaluator = SharedEvaluator(SignalWorld, (SignalPartner(),), MessageChannel(), observed_signal_message)
        results = evaluator.evaluate_population(agents, scenarios)
        self.assertEqual(results[0].interactions, 16)
        self.assertEqual(results[0].world_steps, 8)
        self.assertEqual([x["focal_id"] for x in results[0].details["mappings"]], ["left", "right"])
        self.assertGreaterEqual(results[0].fitness, 3)
        team = SharedEvaluator(SignalWorld, (SignalPartner(),), MessageChannel(), observed_signal_message, "team")
        self.assertTrue(np.isfinite(team.evaluate(agents[0], scenarios).fitness))
        no_messages = SharedEvaluator(SignalWorld, (SignalPartner(),))
        with_messages = evaluator.evaluate(agents[0], [Scenario(i) for i in range(20)])
        without_messages = no_messages.evaluate(agents[0], [Scenario(i) for i in range(20)])
        self.assertGreater(with_messages.fitness, without_messages.fitness)

    def test_evolved_communication_end_to_end_and_serialization(self):
        config = EvolutionConfig(population_size=4, generations=2, elite_count=1, evaluation_episodes=2)
        evaluator = SharedEvaluator(SignalWorld, (SignalPartner(),), MessageChannel(), observed_signal_message)
        result = Evolution(env_factory=shared_env, agent_factory=CommunicatingAgentFactory(MLPAgentFactory((4,))),
                           evaluator=evaluator, config=config).run()
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history[0]["world_steps"], 32)
        self.assertGreater(result.history[-1]["evaluation_interactions"], 0)
        agent = result.best_agent
        observation, _ = shared_env().reset(seed=1)
        self.assertIsInstance(agent.act(observation, np.random.default_rng(0)), CommunicationAction)
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            loaded = load_agent(directory)
            np.testing.assert_array_equal(agent.genome.vector(), loaded.genome.vector())

    def test_focal_final_observation_and_independent_resets(self):
        env = SharedFocalEnv(world_factory=partial(SignalWorld, horizon=1), channel=MessageChannel(), fixed_message=observed_signal_message)
        initial, _ = env.reset(seed=0, options={"bits": [1, 0]})
        final, reward, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        np.testing.assert_array_equal(final, [1, 0, 0, -1, 1])

    def test_inactive_agent_masks(self):
        class DisappearingWorld(SignalWorld):
            def step(self, actions):
                result = super().step(actions)
                if self.t == 1:
                    self.agents = ["right"]
                return result
        world = ParallelWorldAdapter(DisappearingWorld(), MessageChannel(), observed_signal_message)
        world.reset(0)
        world.step({"left": 0, "right": 1}, np.random.default_rng(0))
        self.assertEqual(list(world.observations), ["right"])
        world.step({"right": 0}, np.random.default_rng(0))
        self.assertEqual(world.agent_transitions, 3)
        self.assertEqual(world.world_steps, 2)
        with self.assertRaises(ValueError):
            world.step({"left": 0, "right": 0}, np.random.default_rng(0))

    def test_fixed_communication_sb3_learning(self):
        try:
            import torch
            from geneticevo.integrations.sb3 import SB3AgentFactory, SB3Learner
        except ImportError:
            self.skipTest("SB3 unavailable")
        torch.set_num_threads(1)
        evaluator = SharedEvaluator(SignalWorld, (SignalPartner(),), MessageChannel(), observed_signal_message)
        result = Evolution(env_factory=shared_env, evaluator=evaluator,
            agent_factory=SB3AgentFactory("PPO", {"n_steps": 8, "batch_size": 8, "n_epochs": 1,
                                                  "policy_kwargs": {"net_arch": [4]}}),
            learner=SB3Learner(), config=EvolutionConfig(population_size=2, generations=1, elite_count=1,
                learning_budget=8, evaluation_episodes=1)).run()
        self.assertEqual(result.history[0]["training_interactions"], 32)
        self.assertEqual(result.history[0]["learner_metrics"]["focal_interactions"], 8)
        self.assertEqual(result.history[0]["learner_metrics"]["world_steps"], 8)


if __name__ == "__main__":
    unittest.main()
