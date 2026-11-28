from functools import partial
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, GaussianMutation, LearnReport, Scenario
from geneticevo.agents import MLPAgentFactory, VectorAgentFactory
from geneticevo.evaluation import ObjectiveEvaluator, adaptation_evaluation, AdaptationTask
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
    def test_neural_lifetime_learning_requires_an_explicit_learner(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit custom learner"):
            Evolution(agent_factory=MLPAgentFactory((4,)), env_factory=Reach1D,
                      config=EvolutionConfig(learning_budget=4))

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


if __name__ == "__main__":
    unittest.main()
