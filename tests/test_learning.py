from functools import partial
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, GaussianMutation, LearnReport, Scenario, load_agent
from geneticevo.agents import VectorAgentFactory
from geneticevo.evaluation import ObjectiveEvaluator, GymEvaluator, adaptation_evaluation, AdaptationTask
from geneticevo.environments import Reach1D
from geneticevo.learners import GenomeLearner


class AddOneLearner(GenomeLearner):
    def validate(self, agent, env_factory, backend):
        pass

    def prepare(self, agent, seed):
        pass

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        agent.genome.assign(agent.genome.vector() + 1)
        agent.training_state = {"updates": agent.training_state.get("updates", 0) + 1} if hasattr(agent, "training_state") else {"updates": 1}
        return LearnReport(budget, 1, budget)

    def __repr__(self):
        return "AddOneLearner()"


class InheritanceTests(unittest.TestCase):
    def experiment(self, mode):
        return Evolution(agent_factory=VectorAgentFactory(2, initial_std=0), env_factory=Reach1D,
                         evaluator=ObjectiveEvaluator(lambda v, s: float(v.sum())),
                         learner=AddOneLearner(), mutation=GaussianMutation(0, 0),
                         config=EvolutionConfig(population_size=2, generations=3, elite_count=1,
                         evaluation_episodes=1, inheritance=mode, learning_budget=5, measure_pre_learning=True))

    def test_lamarckian_and_fresh_children(self):
        result = self.experiment("lamarckian").run()
        self.assertEqual([x["mean"] for x in result.history], [2, 4, 6])
        self.assertEqual(result.population[0].agent.training_state["updates"], 3)
        self.assertEqual(result.population[1].agent.training_state["updates"], 1)
        self.assertEqual(result.history[-1]["training_interactions"], 30)

    def test_baldwinian_trains_a_copy(self):
        result = self.experiment("baldwinian").run()
        self.assertEqual([x["mean"] for x in result.history], [2, 2, 2])
        for individual in result.population:
            np.testing.assert_array_equal(individual.agent.genome.vector(), [0, 0])
            self.assertFalse(hasattr(individual.agent, "training_state"))
        np.testing.assert_array_equal(result.best_agent.genome.vector(), [1, 1])

    def test_adaptation_resets_between_tasks(self):
        agent = VectorAgentFactory(2, 0)(np.random.default_rng(0))
        evaluator = ObjectiveEvaluator(lambda v, s: float(v.sum()))
        records = adaptation_evaluation(agent, AddOneLearner(), Reach1D, evaluator, [[Scenario(1)], [Scenario(2)]], 4)
        self.assertEqual([x["after"].fitness for x in records], [2, 2])
        np.testing.assert_array_equal(agent.genome.vector(), [0, 0])

    def test_adaptation_uses_task_specific_training_environments(self):
        class TaskLearner(AddOneLearner):
            def learn(self, agent, env_factory, budget, environment_seed, action_seed):
                report = super().learn(agent, env_factory, budget, environment_seed, action_seed)
                env = env_factory()
                report.metrics["training_horizon"] = env.horizon
                env.close()
                return report
        tasks = [AdaptationTask(partial(Reach1D, horizon=horizon), [Scenario(horizon)]) for horizon in (3, 7)]
        agent = VectorAgentFactory(2, 0)(np.random.default_rng(0))
        evaluator = ObjectiveEvaluator(lambda v, s: float(v.sum()))
        records = adaptation_evaluation(agent, TaskLearner(), Reach1D, evaluator, tasks, 4)
        self.assertEqual([r["learning"].metrics["training_horizon"] for r in records], [3, 7])


try:
    import torch
    from geneticevo.integrations.sb3 import SB3AgentFactory, SB3Learner
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


@unittest.skipUnless(HAS_SB3, "SB3 extra is not installed")
class SB3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agent(self, algorithm="PPO", copies=1):
        env = Reach1D(horizon=4)
        options = {"n_steps": 8, "batch_size": 8, "n_epochs": 1, "policy_kwargs": {"net_arch": [8]}} if algorithm == "PPO" else {
            "buffer_size": 100, "learning_starts": 0, "batch_size": 4, "policy_kwargs": {"net_arch": [8]}}
        return SB3AgentFactory(algorithm, options, copies)(np.random.default_rng(1), env.observation_space, env.action_space)

    def test_ppo_budget_frozen_eval_and_save(self):
        agent = self.agent()
        learner = SB3Learner()
        before = agent.genome.vector()
        report = learner.learn(agent, partial(Reach1D, horizon=4), 9, 1, 2)
        self.assertEqual(report.interactions, 16)
        self.assertGreater(report.updates, 0)
        self.assertFalse(np.array_equal(before, agent.genome.vector()))
        frozen = agent.genome.vector()
        GymEvaluator(Reach1D).evaluate(agent, [Scenario(10)])
        np.testing.assert_array_equal(frozen, agent.genome.vector())
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = load_agent(directory)
            np.testing.assert_array_equal(frozen, restored.genome.vector())

    def test_dqn_replay_target_and_child_reset(self):
        agent = self.agent("DQN")
        learner = SB3Learner()
        learner.learn(agent, Reach1D, 8, 3, 4)
        self.assertGreater(agent.model.replay_buffer.size(), 0)
        child = learner.clone(agent, False)
        self.assertEqual(child.model.replay_buffer.size(), 0)
        self.assertEqual(child.model.num_timesteps, 0)
        self.assertEqual(child.model._n_updates, 0)
        self.assertEqual(len(child.model.policy.optimizer.state), 0)
        GaussianMutation(1, 0.1)(child.genome, np.random.default_rng(3))
        for online, target in zip(child.model.q_net.parameters(), child.model.q_net_target.parameters()):
            torch.testing.assert_close(online, target)
        resumed = learner.clone(agent, True)
        self.assertEqual(resumed.model.replay_buffer.size(), agent.model.replay_buffer.size())

    def test_multiple_copies_and_resume(self):
        agent = self.agent(copies=2)
        learner = SB3Learner(2)
        self.assertEqual(learner.learn(agent, Reach1D, 5, 1, 2).interactions, 16)
        clone = learner.clone(agent, True)
        report1 = learner.learn(agent, Reach1D, 16, 3, 4)
        report2 = learner.learn(clone, Reach1D, 16, 3, 4)
        np.testing.assert_array_equal(agent.genome.vector(), clone.genome.vector())
        self.assertEqual(report1.interactions, report2.interactions)

    def test_reject_spaces_backends_and_normalization(self):
        env = Reach1D(continuous=True)
        with self.assertRaises(ValueError):
            SB3AgentFactory("DQN")(np.random.default_rng(0), env.observation_space, env.action_space)
        learner = SB3Learner()
        with self.assertRaises(ValueError):
            learner.validate(self.agent(), Reach1D, "batched")
        from gymnasium.wrappers import NormalizeObservation
        with self.assertRaises(ValueError):
            learner.validate(self.agent(), lambda: NormalizeObservation(Reach1D()), "serial")

    def test_evolution_sb3_resume_and_continuous_ppo(self):
        def experiment(generations):
            return Evolution(env_factory=Reach1D, learner=SB3Learner(),
                agent_factory=SB3AgentFactory("PPO", {"n_steps": 8, "batch_size": 8, "n_epochs": 1,
                                                     "policy_kwargs": {"net_arch": [4]}}),
                config=EvolutionConfig(population_size=2, generations=generations, elite_count=1,
                                       evaluation_episodes=1, learning_budget=8))
        complete = experiment(2).run()
        with tempfile.TemporaryDirectory() as directory:
            first = experiment(1)
            first.run()
            path = Path(directory) / "sb3.checkpoint"
            first.checkpoint(path)
            resumed = experiment(2).run(path)
        for left, right in zip(complete.population, resumed.population):
            np.testing.assert_array_equal(left.agent.genome.vector(), right.agent.genome.vector())
        env = Reach1D(continuous=True)
        agent = SB3AgentFactory("PPO", {"n_steps": 4, "batch_size": 4, "n_epochs": 1,
            "policy_kwargs": {"net_arch": [4]}})(np.random.default_rng(0), env.observation_space, env.action_space)
        report = SB3Learner().learn(agent, partial(Reach1D, continuous=True), 4, 0, 1)
        self.assertEqual(report.interactions, 4)


if __name__ == "__main__":
    unittest.main()
