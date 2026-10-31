import importlib.abc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, GaussianMutation, Scenario, load_agent
from geneticevo.agents import VectorAgentFactory, MLPAgentFactory
from geneticevo.core import VectorGenome
from geneticevo.environments import Reach1D
from geneticevo.evaluation import ObjectiveEvaluator, GymEvaluator


def sphere(vector, scenario):
    return float(np.square(vector).sum()), {"norm": float(np.linalg.norm(vector))}


def make_run(generations=4, **options):
    config = EvolutionConfig(population_size=8, generations=generations, elite_count=2,
                             maximize=False, evaluation_episodes=2, seed=10, **options)
    return Evolution(agent_factory=VectorAgentFactory(4), evaluator=ObjectiveEvaluator(sphere),
                     config=config, mutation=GaussianMutation(0.8, 0.2))


class CoreTests(unittest.TestCase):
    def test_generation_count_and_elitism(self):
        result = make_run().run()
        self.assertEqual([x["generation"] for x in result.history], [0, 1, 2, 3])
        self.assertEqual(len(result.population), 8)
        self.assertEqual(len(result.lineage), 8 + 3 * 6)
        best = [x["best"] for x in result.history]
        self.assertEqual(best, sorted(best, reverse=True))

    def test_maximize_rank_and_immigrants(self):
        result = Evolution(agent_factory=VectorAgentFactory(2), evaluator=ObjectiveEvaluator(sphere),
                           config=EvolutionConfig(population_size=6, generations=3, elite_count=1,
                                                  selection="rank", immigrant_count=2)).run()
        self.assertGreaterEqual(result.history[-1]["best"], result.history[0]["best"])
        self.assertEqual(sum(x["mutation"].get("kind") == "immigrant" for x in result.lineage), 4)

    def test_mutation_independence_and_probability(self):
        parent = VectorGenome(np.ones(50))
        child = parent.clone()
        GaussianMutation(0, 10)(child, np.random.default_rng(2))
        np.testing.assert_array_equal(child.values, parent.values)
        GaussianMutation(1, 1)(child, np.random.default_rng(2))
        self.assertFalse(np.array_equal(child.values, parent.values))
        np.testing.assert_array_equal(parent.values, np.ones(50))

    def test_archive_is_independent_and_contextual(self):
        result = make_run().run()
        snapshot = result.archive[0].agent.genome.vector()
        for individual in result.population:
            individual.agent.genome.assign(np.zeros(4))
        result.best_agent.genome.assign(np.zeros(4))
        np.testing.assert_array_equal(snapshot, result.archive[0].agent.genome.vector())
        self.assertTrue(all(x.distribution == "archive" for x in result.archive[0].evaluation.context))

    def test_resume_matches_uninterrupted(self):
        complete = make_run(5).run()
        with tempfile.TemporaryDirectory() as directory:
            experiment = make_run(2)
            experiment.run()
            path = Path(directory) / "state.checkpoint"
            experiment.checkpoint(path)
            resumed = make_run(5).run(path)
        for left, right in zip(complete.population, resumed.population):
            self.assertEqual(left.id, right.id)
            np.testing.assert_array_equal(left.agent.genome.vector(), right.agent.genome.vector())
        self.assertEqual(complete.lineage, resumed.lineage)
        self.assertEqual([x["mean"] for x in complete.history], [x["mean"] for x in resumed.history])

    def test_stop_conditions(self):
        self.assertEqual(make_run(target_fitness=100).run().stop_reason, "target_fitness")
        self.assertEqual(make_run(time_limit_seconds=1e-12).run().stop_reason, "time_limit")
        result = Evolution(env_factory=Reach1D, agent_factory=MLPAgentFactory((4,)),
                           config=EvolutionConfig(population_size=2, generations=5, elite_count=1,
                                                  evaluation_episodes=1, max_interactions=1)).run()
        self.assertEqual(result.stop_reason, "interaction_budget")
        self.assertEqual(len(result.history), 1)
        result = Evolution(agent_factory=VectorAgentFactory(2), evaluator=ObjectiveEvaluator(sphere),
                           mutation=GaussianMutation(0, 0), config=EvolutionConfig(population_size=1, generations=10,
                           elite_count=1, stagnation_generations=2)).run()
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.stop_reason, "stagnation")

    def test_scenarios_and_frozen_evaluation(self):
        factory = MLPAgentFactory((4,))
        env = Reach1D(horizon=3)
        agent = factory(np.random.default_rng(1), env.observation_space, env.action_space)
        snapshot = agent.genome.vector()
        evaluation = GymEvaluator(lambda: Reach1D(horizon=3)).evaluate(agent, [Scenario(1), Scenario(2)])
        self.assertEqual(evaluation.lengths, [3, 3])
        self.assertEqual(evaluation.interactions, 6)
        np.testing.assert_array_equal(snapshot, agent.genome.vector())

    def test_save_load_and_initialization(self):
        agent = make_run(1).run().best_agent
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = load_agent(directory)
            np.testing.assert_array_equal(agent.genome.vector(), restored.genome.vector())
            run = make_run(1)
            run.initial_agents = (restored,)
            result = run.run()
            np.testing.assert_array_equal(result.population[0].agent.genome.vector(), restored.genome.vector())

    def test_optional_dependency_isolation(self):
        script = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3', 'pettingzoo', 'matplotlib'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from geneticevo import Evolution, EvolutionConfig
from geneticevo.agents import VectorAgentFactory
from geneticevo.evaluation import ObjectiveEvaluator
result = Evolution(agent_factory=VectorAgentFactory(3), evaluator=ObjectiveEvaluator(lambda v,s: float(v.sum())), config=EvolutionConfig(population_size=2, elite_count=1, generations=2)).run()
assert len(result.history) == 2
from geneticevo import NeuralGrowth
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from functools import partial
result = Evolution(agent_factory=MLPAgentFactory((4,), NeuralGrowth(mutation_probability=1)),
    env_factory=partial(Reach1D, horizon=2),
    config=EvolutionConfig(population_size=3, elite_count=1, generations=2, evaluation_episodes=1)).run()
assert result.history[0]['network_hidden_sizes'] == [[4], [4], [4]]
assert all('architecture' in row['mutation'] for row in result.lineage if row['parents'])
assert 'torch' not in sys.modules
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        process = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_invalid_configuration(self):
        for changes in ({"population_size": 0}, {"elite_count": 100}, {"learning_budget": -1},
                        {"evaluation_episodes": 0}, {"inheritance": "unknown"}):
            with self.assertRaises(ValueError):
                EvolutionConfig(**changes)
        with self.assertRaises(ValueError):
            make_run().config = EvolutionConfig(population_size=1)
        with self.assertRaises(ValueError):
            Evolution(agent_factory=VectorAgentFactory(2), evaluator=ObjectiveEvaluator(sphere),
                      config=EvolutionConfig(learning_budget=10))

    def test_changed_components_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = make_run(1)
            experiment.run()
            path = Path(directory) / "state.checkpoint"
            experiment.checkpoint(path)
            original = make_run(2)
            changed = Evolution(agent_factory=VectorAgentFactory(5), evaluator=ObjectiveEvaluator(sphere),
                                config=original.config, mutation=original.mutation)
            with self.assertRaisesRegex(ValueError, "factories"):
                changed.run(path)

    def test_options_are_not_mutated_by_environments(self):
        class MutatingEnv(Reach1D):
            def reset(self, *, seed=None, options=None):
                observation = super().reset(seed=seed, options=options)
                options["position"] = 100
                return observation
        scenario = Scenario(1, {"position": 0.2})
        env = Reach1D()
        agent = MLPAgentFactory((4,))(np.random.default_rng(0), env.observation_space, env.action_space)
        GymEvaluator(MutatingEnv).evaluate(agent, [scenario])
        self.assertEqual(scenario.options["position"], 0.2)

    def test_terminated_and_truncated_both_end_evaluation(self):
        class TerminalEnv(Reach1D):
            def step(self, action):
                observation, reward, _, _, info = super().step(action)
                return observation, reward, True, False, info
        env = Reach1D()
        agent = MLPAgentFactory((4,))(np.random.default_rng(0), env.observation_space, env.action_space)
        self.assertEqual(GymEvaluator(TerminalEnv).evaluate(agent, [Scenario(1)]).lengths, [1])
        self.assertEqual(GymEvaluator(lambda: Reach1D(horizon=2)).evaluate(agent, [Scenario(1)]).lengths, [2])


if __name__ == "__main__":
    unittest.main()

