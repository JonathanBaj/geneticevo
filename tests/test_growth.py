"""Lifetime capacity, genetic inheritance, and optional PPO integration."""
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np

from geneticevo import Evolution, EvolutionConfig, GaussianMutation, LearnReport, NeuralGrowth, Scenario, load_agent
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator, ObjectiveEvaluator
from geneticevo.evolution import parameter_distance
from geneticevo.learners import GenomeLearner


def make_agent(sizes=(3, 4), growth=None, continuous=False, seed=0):
    env = Reach1D(continuous=continuous)
    try:
        return MLPAgentFactory(sizes, growth)(np.random.default_rng(seed), env.observation_space, env.action_space)
    finally:
        env.close()


class GrowingLearner(GenomeLearner):
    """Deterministic learning makes inherited versus acquired weights observable."""
    supports_neural_growth = True

    def validate(self, agent, env_factory, backend):
        pass

    def prepare(self, agent, seed):
        pass

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        agent.genome.assign(agent.genome.vector() + 1)
        agent.growth.after_update(agent, budget, 1.0, np.random.default_rng(action_seed))
        return LearnReport(budget, 1, budget)

    def __repr__(self):
        return 'GrowingLearner()'


class GrowthTests(unittest.TestCase):
    def test_resize_preserves_policy_value_and_existing_genes(self):
        for continuous in (False, True):
            agent = make_agent(growth=NeuralGrowth(max_hidden_size=12), continuous=continuous)
            # Exercise nonzero biases and continuous distribution parameters too.
            agent.genome.assign(np.random.default_rng(9).normal(size=agent.genome.values.size))
            original = agent.clone()
            observation = np.linspace(-2, 2, 31, dtype=np.float32)[:, None]
            expected = [agent.forward(observation, prefix) for prefix in ('actor', 'critic')]
            agent.training_state = {'adam': {'old': True}, 'neural_growth': {'events': 2}}
            self.assertTrue(agent.resize_hidden_sizes((6, 7), np.random.default_rng(1)))
            for prefix, output in zip(('actor', 'critic'), expected):
                np.testing.assert_allclose(agent.forward(observation, prefix), output, atol=1e-6)
            for name, values in original.parameters().items():
                slices = tuple(slice(0, x) for x in values.shape)
                np.testing.assert_array_equal(agent.parameters()[name][slices], values)
            self.assertNotIn('adam', agent.training_state)
            self.assertEqual(agent.training_state['neural_growth'], {'events': 2})
            # Existing behavior remains intact while new incoming pathways are active.
            self.assertGreater(np.linalg.norm(agent.parameters()['actor_w0'][:, 3:]), 0)
            np.testing.assert_array_equal(agent.parameters()['actor_w2'][4:], 0)
            self.assertFalse(agent.resize_hidden_sizes((6, 7), np.random.default_rng(2)))
            child = agent.clone(False)
            self.assertEqual(child.hidden_sizes, (6, 7))
            self.assertEqual(child.growth, agent.growth)
            self.assertFalse(hasattr(child, 'training_state'))
            child.resize_hidden_sizes((2, 3), np.random.default_rng(3))
            self.assertEqual(agent.hidden_sizes, (6, 7))
            self.assertTrue(np.isfinite(child.forward(observation)).all())
            for name, values in child.parameters().items():
                slices = tuple(slice(0, x) for x in values.shape)
                np.testing.assert_array_equal(values, agent.parameters()[name][slices])

    def test_trigger_threshold_interval_cap_and_frozen_evaluation(self):
        growth = NeuralGrowth(check_every=4, neurons_per_growth=2, max_hidden_size=6,
                              value_loss_threshold=0.5)
        agent = make_agent((3,), growth)
        rng = np.random.default_rng(2)
        self.assertFalse(growth.after_update(agent, 2, 1.0, rng))
        self.assertFalse(growth.after_update(agent, 2, 0.1, rng))
        self.assertFalse(growth.after_update(agent, 2, 1.0, rng))
        self.assertTrue(growth.after_update(agent, 2, 0.5, rng))
        self.assertEqual(agent.hidden_sizes, (5,))
        self.assertTrue(growth.after_update(agent, 4, 1.0, rng))
        self.assertEqual(agent.hidden_sizes, (6,))
        self.assertFalse(growth.after_update(agent, 4, 10.0, rng))
        self.assertEqual(agent.training_state['neural_growth']['events'], 2)
        state = dict(agent.training_state['neural_growth'])
        frozen = agent.genome.vector()
        GymEvaluator(partial(Reach1D, horizon=3)).evaluate(agent, [Scenario(1)])
        np.testing.assert_array_equal(agent.genome.vector(), frozen)
        self.assertEqual(agent.training_state['neural_growth'], state)

    def test_size_mutations_are_bounded_reproducible_and_independent(self):
        growth = NeuralGrowth(min_hidden_size=2, max_hidden_size=6, mutation_probability=1, mutation_step=2)
        parent = make_agent((4, 4), growth)
        seen = set()
        for seed in range(20):
            child, other = parent.clone(), parent.clone()
            metadata = growth.mutate(child, np.random.default_rng(seed))
            self.assertEqual(metadata, growth.mutate(other, np.random.default_rng(seed)))
            np.testing.assert_array_equal(child.genome.vector(), other.genome.vector())
            self.assertTrue(all(2 <= x <= 6 for x in child.hidden_sizes))
            seen.update(child.hidden_sizes)
        self.assertTrue(any(x < 4 for x in seen))
        self.assertTrue(any(x > 4 for x in seen))
        self.assertEqual(parent.hidden_sizes, (4, 4))
        disabled = make_agent((3,), replace(growth, mutation_probability=0))
        before = disabled.genome.vector()
        disabled.growth.mutate(disabled, np.random.default_rng(1))
        np.testing.assert_array_equal(disabled.genome.vector(), before)

    def test_size_is_inherited_in_both_weight_inheritance_modes(self):
        growth = NeuralGrowth(check_every=1, neurons_per_growth=1, max_hidden_size=8, mutation_probability=0)
        factory = MLPAgentFactory((2,), growth)
        for mode in ('lamarckian', 'baldwinian'):
            experiment = Evolution(agent_factory=factory, env_factory=Reach1D, learner=GrowingLearner(),
                evaluator=ObjectiveEvaluator(lambda v, s: float(v.sum())), mutation=GaussianMutation(0, 0),
                config=EvolutionConfig(population_size=2, elite_count=1, generations=2,
                    inheritance=mode, learning_budget=1, evaluation_episodes=1))
            # Capture birth genes before training to check retention exactly.
            births = [experiment._factory_agent(i) for i in range(2)]
            result = experiment.run()
            self.assertEqual(result.history[0]['network_hidden_sizes'], [[3], [3]])
            self.assertEqual(result.history[1]['network_hidden_sizes'], [[4], [4]])
            for individual in result.population:
                self.assertEqual(individual.agent.hidden_sizes, (4,))
            child = next(x for x in result.population if x.born == 1)
            self.assertEqual(child.mutation['architecture']['before'], [3])
            self.assertEqual(child.mutation['architecture']['after'], [3])
            if mode == 'baldwinian':
                original_individual = next(x for x in result.population if x.born == 0)
                birth = births[int(original_individual.id[1:])]
                for name, values in birth.parameters().items():
                    slices = tuple(slice(0, x) for x in values.shape)
                    np.testing.assert_array_equal(original_individual.agent.parameters()[name][slices], values)
                self.assertFalse(hasattr(original_individual.agent, 'training_state'))
                self.assertFalse(np.array_equal(result.best_agent.parameters()['actor_w0'][:, :2],
                                                birth.parameters()['actor_w0']))
            else:
                self.assertTrue(all(hasattr(x.agent, 'training_state') for x in result.population))

    def test_pure_evolution_size_mutation_and_variable_diversity(self):
        result = Evolution(env_factory=partial(Reach1D, horizon=3),
            agent_factory=MLPAgentFactory((4, 4), NeuralGrowth(max_hidden_size=8, mutation_probability=1)),
            config=EvolutionConfig(population_size=5, generations=3, elite_count=1,
                immigrant_count=1, evaluation_episodes=1, diversity_pairs=6)).run()
        self.assertEqual(result.history[0]['network_hidden_sizes'], [[4, 4]] * 5)
        children = [x for x in result.lineage if x['born'] > 0 and x['parents']]
        self.assertTrue(all('architecture' in x['mutation'] for x in children))
        self.assertTrue(all(np.isfinite(x['sampled_parameter_distance']) for x in result.history))
        agent = make_agent((3, 4))
        self.assertEqual(parameter_distance(agent, agent.clone()), 0)
        other = agent.clone()
        other.resize_hidden_sizes((4, 5), np.random.default_rng(5))
        self.assertAlmostEqual(parameter_distance(agent, other), parameter_distance(other, agent))

    def test_save_load_grown_agent_initialization_and_legacy_format(self):
        agent = make_agent((3,), NeuralGrowth(max_hidden_size=8))
        agent.resize_hidden_sizes((6,), np.random.default_rng(0))
        agent.training_state = {'neural_growth': {'events': 1}}
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = load_agent(directory)
            self.assertEqual(restored.hidden_sizes, (6,))
            self.assertEqual(restored.growth, agent.growth)
            np.testing.assert_array_equal(restored.genome.vector(), agent.genome.vector())
            self.assertFalse(hasattr(restored, 'training_state'))
            result = Evolution(agent_factory=MLPAgentFactory((3,), agent.growth),
                env_factory=partial(Reach1D, horizon=2), initial_agents=(restored,),
                config=EvolutionConfig(population_size=2, elite_count=1, generations=1,
                    evaluation_episodes=1)).run()
            self.assertEqual(result.population[0].agent.hidden_sizes, (6,))
            np.testing.assert_array_equal(result.population[0].agent.genome.vector(), agent.genome.vector())
            # Legacy schema-1 MLPs without a growth field still load unchanged.
            spec_path = Path(directory) / 'agent.json'
            spec = json.loads(spec_path.read_text())
            spec.pop('growth')
            spec_path.write_text(json.dumps(spec))
            self.assertIsNone(load_agent(directory).growth)

    def test_invalid_configuration_and_unsupported_combinations(self):
        for changes in ({'check_every': 0}, {'neurons_per_growth': True}, {'max_hidden_size': 0},
                        {'min_hidden_size': 5, 'max_hidden_size': 4}, {'mutation_step': -1},
                        {'mutation_probability': -0.1}, {'mutation_probability': float('nan')},
                        {'value_loss_threshold': -1}, {'value_loss_threshold': float('inf')}):
            with self.assertRaises(ValueError):
                NeuralGrowth(**changes)
        for sizes in ((), (0,), (1.5,), (True,), (129,)):
            with self.assertRaises(ValueError):
                MLPAgentFactory(sizes, NeuralGrowth())
        self.assertIsNone(make_agent().growth)
        agent = make_agent((3,), NeuralGrowth())
        for interactions in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'interaction count'):
                agent.growth.after_update(agent, interactions, 1.0, np.random.default_rng(1))
        for sizes in ((2,) * 9, (0,), (1.5,), (129,)):
            with self.assertRaises(ValueError):
                agent.resize_hidden_sizes(sizes, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, 'crossover=None'):
            Evolution(agent_factory=MLPAgentFactory((3,), NeuralGrowth()), env_factory=Reach1D,
                      crossover=lambda a, b, rng: a).run()
        class Unsupported(GrowingLearner):
            supports_neural_growth = False
        with self.assertRaisesRegex(ValueError, 'requires NativePPO'):
            Evolution(agent_factory=MLPAgentFactory((3,), NeuralGrowth()), env_factory=Reach1D,
                      learner=Unsupported(), config=EvolutionConfig(learning_budget=1)).run()


try:
    import torch
    from geneticevo.ppo import NativePPO
    from geneticevo.tensor import TensorReach1D, evaluate_batched
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, 'torch extra is not installed')
class GrowthPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def experiment(self, generations, mode='lamarckian', output_dir=None, tensor=True):
        return Evolution(env_factory=partial(Reach1D, horizon=7),
            tensor_env_factory=partial(TensorReach1D, horizon=7) if tensor else None,
            agent_factory=MLPAgentFactory((3, 4), NeuralGrowth(check_every=8,
                neurons_per_growth=1, max_hidden_size=9, mutation_probability=1, mutation_step=1)),
            backend='batched', learner=NativePPO(env_copies=2, rollout_steps=4, epochs=1, minibatch_size=4),
            config=EvolutionConfig(population_size=3, generations=generations, elite_count=1,
                immigrant_count=1, evaluation_episodes=2, learning_budget=9, env_copies=2,
                diversity_pairs=5, measure_pre_learning=True, inheritance=mode, output_dir=output_dir))

    def test_ppo_growth_budget_state_and_checkpoint_resume(self):
        for mode in ('lamarckian', 'baldwinian'):
            complete = self.experiment(3, mode).run()
            self.assertEqual(complete.history[0]['learning_actual_per_individual'], [16] * 3)
            self.assertEqual(complete.history[-1]['training_interactions'], 144)
            self.assertEqual(complete.history[0]['learner_metrics']['growth_events'], 2)
            self.assertEqual([sum(s) for s in complete.history[0]['network_hidden_sizes']], [9] * 3)
            self.assertGreater(len({tuple(s) for s in complete.history[-1]['network_hidden_sizes']}), 1)
            with tempfile.TemporaryDirectory() as directory:
                first = self.experiment(1, mode, directory)
                first.run()
                resumed = self.experiment(3, mode).run(Path(directory) / 'latest.checkpoint')
                for left, right in zip(complete.population, resumed.population):
                    self.assertEqual(left.id, right.id)
                    self.assertEqual(left.agent.hidden_sizes, right.agent.hidden_sizes)
                    np.testing.assert_array_equal(left.agent.genome.vector(), right.agent.genome.vector())
                    if mode == 'lamarckian':
                        self.assertEqual(left.agent.training_state['neural_growth'],
                                         right.agent.training_state['neural_growth'])
                        for name, parameter in left.agent.parameters().items():
                            self.assertEqual(tuple(left.agent.training_state['adam']['first'][name].shape), parameter.shape)
                self.assertEqual(complete.lineage, resumed.lineage)
                self.assertEqual([x['mean'] for x in complete.history], [x['mean'] for x in resumed.history])
                self.assertEqual([x['network_hidden_sizes'] for x in complete.history],
                                 [x['network_hidden_sizes'] for x in resumed.history])

    def test_ppo_learns_new_connections_and_keeps_environment_alive(self):
        for continuous in (False, True):
            growth = NeuralGrowth(check_every=4, neurons_per_growth=2, max_hidden_size=5, mutation_probability=0)
            agent = make_agent((3,), growth, continuous)
            instances = []
            class CountingEnv(Reach1D):
                def __init__(self):
                    super().__init__(horizon=50, continuous=continuous)
                    self.resets = 0
                    instances.append(self)
                def reset(self, **kwargs):
                    self.resets += 1
                    return super().reset(**kwargs)
            learner = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4)
            report = learner.learn_population([agent], CountingEnv, None, 12, 0, 0)[0]
            self.assertEqual(agent.hidden_sizes, (5,))
            self.assertEqual(report.interactions, 12)
            self.assertEqual(report.updates, 3)
            self.assertEqual(report.metrics['growth_events'], 1)
            self.assertEqual(agent.training_state['neural_growth']['interactions'], 12)
            self.assertEqual(agent.training_state['adam']['step'], 2)
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].resets, 1)
            self.assertGreater(np.linalg.norm(agent.parameters()['actor_w1'][3:]), 0)
            self.assertGreater(np.linalg.norm(agent.parameters()['critic_w1'][3:]), 0)

    def test_variable_width_evaluation_matches_serial_and_is_frozen(self):
        for continuous in (False, True):
            agents = [make_agent(sizes, NeuralGrowth(), continuous, i)
                      for i, sizes in enumerate(((3,), (5,), (3,)))]
            snapshots = [a.genome.vector() for a in agents]
            scenarios = [Scenario(1), Scenario(2), Scenario(3)]
            factory = partial(Reach1D, horizon=5, continuous=continuous)
            for tensor_factory in (None, partial(TensorReach1D, horizon=5, continuous=continuous)):
                evaluations = evaluate_batched(agents, scenarios, factory, tensor_factory, env_copies=2)
                for agent, snapshot, evaluation in zip(agents, snapshots, evaluations):
                    reference = GymEvaluator(factory).evaluate(agent, scenarios)
                    np.testing.assert_allclose(evaluation.returns, reference.returns, atol=1e-6)
                    np.testing.assert_array_equal(agent.genome.vector(), snapshot)
                    self.assertEqual(evaluation.interactions, 15)
                    self.assertFalse(hasattr(agent, 'training_state'))

    def test_native_ppo_loss_gate_and_python_bridge(self):
        agent = make_agent((3,), NeuralGrowth(check_every=4, value_loss_threshold=1e20))
        report = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4).learn(agent, Reach1D, 8, 1, 2)
        self.assertEqual(agent.hidden_sizes, (3,))
        self.assertEqual(report.metrics['growth_events'], 0)
        result = self.experiment(2, tensor=False).run()
        self.assertEqual(result.history[-1]['training_interactions'], 96)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_growth_and_variable_widths(self):
        agents = [make_agent((width,), NeuralGrowth(check_every=4, max_hidden_size=8)) for width in (3, 5)]
        reports = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4, device='cuda').learn_population(
            agents, Reach1D, TensorReach1D, 8, 0, 0)
        self.assertEqual([r.interactions for r in reports], [8, 8])
        self.assertTrue(all(r.metrics['growth_events'] > 0 for r in reports))
        evaluations = evaluate_batched(agents, [Scenario(1)], Reach1D, TensorReach1D, device='cuda')
        self.assertTrue(all(np.isfinite(x.fitness) for x in evaluations))


if __name__ == '__main__':
    unittest.main()
