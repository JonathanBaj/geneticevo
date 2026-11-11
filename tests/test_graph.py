"""Sparse connectivity, structural plasticity, inheritance and PPO integration."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geneticevo import (Evolution, EvolutionConfig, GaussianMutation, GraphAgentFactory,
                        GraphGrowth, LearnReport, Scenario, load_agent)
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator
from geneticevo.evolution import parameter_distance
from geneticevo.learners import GenomeLearner


def make_agent(growth=None, continuous=False, seed=0, neurons=3, connections=8):
    env = Reach1D(continuous=continuous)
    try:
        return GraphAgentFactory(neurons, connections, growth=growth)(
            np.random.default_rng(seed), env.observation_space, env.action_space)
    finally:
        env.close()


class StructuralLearner(GenomeLearner):
    supports_graph_growth = True

    def validate(self, agent, env_factory, backend):
        pass

    def learn(self, agent, env_factory, budget, environment_seed, action_seed):
        agent.genome.values += 5
        agent.growth.after_update(agent, budget, 0, np.random.default_rng(action_seed))
        return LearnReport(budget, 1, budget)


class GraphTests(unittest.TestCase):
    def test_cycles_self_connections_isolation_and_identity_preserving_pruning(self):
        agent = make_agent()
        rng = np.random.default_rng(1)
        # A cycle 4 <-> 5, self edge, an input skip to output, and isolated 6.
        edges = ((-1, 0), (-1, 4), (4, 5), (5, 4), (5, 5), (5, 1))
        agent.restructure(agent.node_ids, edges, agent.next_node_id, rng)
        agent.parameters()['weights'][:] = np.arange(1, 7) / 10
        agent.parameters()['biases'][:] = np.arange(7) / 100
        before = {e: v for e, v in zip(agent.edges, agent.parameters()['weights'])}
        biases = dict(zip(agent.node_ids, agent.parameters()['biases']))
        self.assertTrue(np.isfinite(agent.forward([[0.5], [-0.2]])).all())
        nodes = tuple(n for n in agent.node_ids if n != 4)
        surviving = tuple(e for e in agent.edges if 4 not in e)
        agent.training_state = {'adam': {}, 'other': 3}
        agent.restructure(nodes, surviving, agent.next_node_id, rng)
        for e, value in zip(agent.edges, agent.parameters()['weights']):
            self.assertEqual(value, before[e])
        for n, value in zip(agent.node_ids, agent.parameters()['biases']):
            self.assertEqual(value, biases[n])
        self.assertNotIn('adam', agent.training_state)
        self.assertEqual(agent.training_state['other'], 3)
        self.assertTrue(np.isfinite(agent.forward([0.5])).all())

    def test_lifetime_grow_prune_caps_counters_and_frozen_evaluation(self):
        growth = GraphGrowth(check_every=4, min_neurons=1, max_neurons=5, max_connections=12,
            node_add_probability=1, node_remove_probability=0, connection_remove_probability=0)
        agent = make_agent(growth)
        rng = np.random.default_rng(8)
        self.assertFalse(growth.after_update(agent, 2, 0, rng))
        self.assertTrue(growth.after_update(agent, 2, 0, rng))
        for _ in range(10):
            growth.after_update(agent, 4, 0, rng)
        self.assertEqual(agent.hidden_neurons, 5)
        self.assertEqual(len(agent.edges), 12)
        prune = replace(growth, node_add_probability=0, node_remove_probability=1,
                        connection_add_probability=0, connection_remove_probability=1)
        agent.growth = prune
        for _ in range(12):
            prune.after_update(agent, 4, 0, rng)
        self.assertEqual(agent.hidden_neurons, 1)
        self.assertEqual(len(agent.edges), 0)
        snapshot, state = agent.genome.vector(), agent.clone(True).training_state
        GymEvaluator(partial(Reach1D, horizon=3)).evaluate(agent, [Scenario(1)])
        np.testing.assert_array_equal(agent.genome.vector(), snapshot)
        self.assertEqual(agent.training_state, state)

    def test_birth_mutations_reproducible_bounded_both_directions_and_no_aliases(self):
        growth = GraphGrowth(max_neurons=5, max_connections=12, mutation_probability=0.5)
        parent = make_agent(growth)
        sizes = set()
        for seed in range(60):
            child, other = parent.clone(), parent.clone()
            self.assertEqual(growth.mutate(child, np.random.default_rng(seed)),
                             growth.mutate(other, np.random.default_rng(seed)))
            self.assertEqual(child.structure, other.structure)
            np.testing.assert_array_equal(child.genome.vector(), other.genome.vector())
            self.assertLessEqual(len(child.edges), 12)
            self.assertLessEqual(child.hidden_neurons, 5)
            sizes.add(child.hidden_neurons)
            child.genome.values += 1
        self.assertTrue(any(x < 3 for x in sizes))
        self.assertTrue(any(x > 3 for x in sizes))
        self.assertEqual(parent.hidden_neurons, 3)

    def test_lifetime_topology_inherited_in_both_weight_modes(self):
        growth = GraphGrowth(check_every=1, max_neurons=8, node_add_probability=1,
            node_remove_probability=0, connection_remove_probability=0, mutation_probability=0)
        for mode in ('lamarckian', 'baldwinian'):
            experiment = Evolution(agent_factory=GraphAgentFactory(3, 8, growth=growth),
                env_factory=partial(Reach1D, horizon=2), learner=StructuralLearner(),
                mutation=GaussianMutation(0, 0), config=EvolutionConfig(population_size=2,
                    elite_count=1, generations=2, learning_budget=1, evaluation_episodes=1, inheritance=mode))
            births = [experiment._factory_agent(i) for i in range(2)]
            result = experiment.run()
            self.assertEqual(result.history[0]['network_neuron_counts'], [8, 8])
            self.assertEqual(result.history[1]['network_neuron_counts'], [9, 9])
            self.assertTrue(all(a.agent.hidden_neurons == 5 for a in result.population))
            survivor = next(x for x in result.population if x.born == 0)
            birth = births[int(survivor.id[1:])]
            inherited = dict(zip(survivor.agent.edges, survivor.agent.parameters()['weights']))
            for edge, value in zip(birth.edges, birth.parameters()['weights']):
                self.assertAlmostEqual(inherited[edge], value + (10 if mode == 'lamarckian' else 0), places=5)
            self.assertIn('architecture', next(x for x in result.lineage if x['parents'])['mutation'])

    def test_save_load_initial_agent_and_diversity_align_by_identity(self):
        agent = make_agent(GraphGrowth(max_neurons=8))
        agent.restructure(agent.node_ids + (7,), agent.edges + ((7, 0),), 8, np.random.default_rng(3))
        agent.training_state = {'graph_growth': {'events': 1}}
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = load_agent(directory)
            self.assertEqual(restored.structure, agent.structure)
            self.assertEqual(restored.growth, agent.growth)
            self.assertEqual(restored.next_node_id, agent.next_node_id)
            np.testing.assert_array_equal(restored.genome.vector(), agent.genome.vector())
            np.testing.assert_array_equal(restored.forward([0.5]), agent.forward([0.5]))
            self.assertFalse(hasattr(restored, 'training_state'))
            result = Evolution(agent_factory=GraphAgentFactory(3, 8, growth=agent.growth),
                initial_agents=(restored,), env_factory=partial(Reach1D, horizon=2),
                config=EvolutionConfig(population_size=2, elite_count=1, generations=1,
                                       evaluation_episodes=1, diversity_pairs=3)).run()
            self.assertEqual(result.population[0].agent.structure, restored.structure)
        self.assertEqual(parameter_distance(agent, restored), 0)
        other = agent.clone()
        other.restructure(other.node_ids, other.edges + ((-1, 7),), 8, np.random.default_rng(2))
        self.assertAlmostEqual(parameter_distance(agent, other), parameter_distance(other, agent))
        self.assertGreater(parameter_distance(agent, other), 0)

    def test_validation_and_unsupported_combinations(self):
        for changes in ({'check_every': 0}, {'min_neurons': -1}, {'max_neurons': True},
                        {'min_neurons': 5, 'max_neurons': 4}, {'max_connections': -1},
                        {'node_add_probability': float('nan')}, {'node_remove_probability': 2}):
            with self.assertRaises(ValueError):
                GraphGrowth(**changes)
        for changes in ({'initial_neurons': -1}, {'settling_steps': 0}, {'initial_connections': True},
                        {'growth': 'wrong'}, {'initial_neurons': 129}):
            with self.assertRaises(ValueError):
                GraphAgentFactory(**changes)
        with self.assertRaisesRegex(ValueError, 'crossover=None'):
            Evolution(agent_factory=GraphAgentFactory(), env_factory=Reach1D, crossover=lambda a, b, rng: a).run()
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.restructure(agent.node_ids, ((999, 0),), agent.next_node_id, np.random.default_rng(0))

    def test_evolution_without_learning_mutates_structure(self):
        result = Evolution(agent_factory=GraphAgentFactory(3, 8,
                growth=GraphGrowth(mutation_probability=0.5, max_neurons=8)),
            env_factory=partial(Reach1D, horizon=3), config=EvolutionConfig(population_size=5,
                elite_count=1, generations=3, diversity_pairs=4, evaluation_episodes=1)).run()
        self.assertEqual(result.history[0]['network_neuron_counts'], [7] * 5)
        self.assertTrue(any(x['mutation']['architecture']['nodes_added'] or
                            x['mutation']['architecture']['nodes_removed'] or
                            x['mutation']['architecture']['connections_added'] or
                            x['mutation']['architecture']['connections_removed']
                            for x in result.lineage if x['parents']))
        self.assertTrue(all(np.isfinite(x['sampled_parameter_distance']) for x in result.history))
        self.assertEqual(result.history[-1]['training_interactions'], 0)

    def test_structure_inheritance_retains_allocation_history(self):
        agent = make_agent()
        learned = agent.clone()
        learned.next_node_id += 5  # Neurons created and subsequently removed.
        agent.inherit_structure(learned, np.random.default_rng(0))
        self.assertEqual(agent.next_node_id, learned.next_node_id)


try:
    import torch
    from geneticevo.ppo import NativePPO
    from geneticevo.tensor import GraphBatchedPolicy, TensorReach1D, evaluate_batched
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, 'torch extra is not installed')
class GraphPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numpy_torch_parity_gradients_and_missing_edges(self):
        for continuous in (False, True):
            agents = [make_agent(continuous=continuous, seed=i) for i in range(2)]
            policy = GraphBatchedPolicy(agents)
            observation = torch.tensor([[[0.3], [0.8]], [[-0.2], [0.6]]])
            for prefix in ('actor', 'critic'):
                expected = np.stack([a.forward(o.numpy(), prefix) for a, o in zip(agents, observation)])
                np.testing.assert_allclose(policy(observation, prefix).detach().numpy(), expected, atol=1e-6)
            policy(observation).sum().backward()
            self.assertEqual(policy.params['weights'].numel(), sum(len(a.edges) for a in agents))
            self.assertTrue(torch.isfinite(policy.params['weights'].grad).all())

    def test_grown_connections_learn_and_environment_is_not_restarted(self):
        for continuous in (False, True):
            growth = GraphGrowth(check_every=4, max_neurons=4, max_connections=20,
                node_add_probability=1, node_remove_probability=0, connection_remove_probability=0,
                connections_per_event=20)
            agent = make_agent(growth, continuous=continuous, connections=0)
            instances = []
            class CountingEnv(Reach1D):
                def __init__(self):
                    super().__init__(horizon=50, continuous=continuous)
                    self.resets = 0
                    instances.append(self)
                def reset(self, **kwargs):
                    self.resets += 1
                    return super().reset(**kwargs)
            # Snapshot newly created weights at the first completed update.
            created = {}
            original = agent.restructure
            def record(*args):
                changed = original(*args)
                if not created and agent.edges:
                    created.update(zip(agent.edges, agent.parameters()['weights'].copy()))
                return changed
            agent.restructure = record
            report = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4).learn(
                agent, CountingEnv, 12, 1, 2)
            self.assertEqual(report.interactions, 12)
            self.assertEqual(report.updates, 3)
            self.assertEqual(agent.hidden_neurons, 4)
            self.assertEqual(len(agent.edges), 20)
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].resets, 1)
            self.assertTrue(any(v != created[e] for e, v in zip(agent.edges, agent.parameters()['weights'])))

    def test_graph_pruning_during_ppo_and_variable_shape_frozen_evaluation(self):
        growth = GraphGrowth(check_every=4, node_add_probability=0, node_remove_probability=1,
            connection_add_probability=0, connection_remove_probability=1)
        for continuous in (False, True):
            agents = [make_agent(growth, continuous, i, neurons=i + 2, connections=8) for i in range(2)]
            reports = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4).learn_population(
                agents, partial(Reach1D, continuous=continuous), None, 8, 1, 0)
            self.assertEqual([a.hidden_neurons for a in agents], [0, 1])
            self.assertTrue(all(r.metrics['growth_events'] == 2 for r in reports))
            snapshots = [a.genome.vector() for a in agents]
            scenarios = [Scenario(1), Scenario(2)]
            factory = partial(Reach1D, horizon=3, continuous=continuous)
            for tensor in (None, partial(TensorReach1D, horizon=3, continuous=continuous)):
                scores = evaluate_batched(agents, scenarios, factory, tensor, env_copies=2)
                for agent, snapshot, score in zip(agents, snapshots, scores):
                    reference = GymEvaluator(factory).evaluate(agent, scenarios)
                    np.testing.assert_allclose(score.returns, reference.returns, atol=1e-6)
                    np.testing.assert_array_equal(agent.genome.vector(), snapshot)

    def experiment(self, generations, mode='lamarckian', output_dir=None):
        growth = GraphGrowth(check_every=4, max_neurons=7, max_connections=24, mutation_probability=0.5)
        return Evolution(agent_factory=GraphAgentFactory(3, 8, growth=growth),
            env_factory=partial(Reach1D, horizon=5), tensor_env_factory=partial(TensorReach1D, horizon=5),
            learner=NativePPO(rollout_steps=4, epochs=1, minibatch_size=4), backend='batched',
            config=EvolutionConfig(population_size=3, elite_count=1, immigrant_count=1,
                generations=generations, learning_budget=8, evaluation_episodes=2,
                inheritance=mode, diversity_pairs=3, output_dir=output_dir))

    def test_evolution_checkpoint_resume_exactly_matches_uninterrupted(self):
        for mode in ('lamarckian', 'baldwinian'):
            complete = self.experiment(3, mode).run()
            with tempfile.TemporaryDirectory() as directory:
                self.experiment(1, mode, directory).run()
                resumed = self.experiment(3, mode).run(Path(directory) / 'latest.checkpoint')
            for a, b in zip(complete.population, resumed.population):
                self.assertEqual(a.agent.structure, b.agent.structure)
                self.assertEqual(a.agent.next_node_id, b.agent.next_node_id)
                np.testing.assert_array_equal(a.agent.genome.vector(), b.agent.genome.vector())
            self.assertEqual(complete.lineage, resumed.lineage)
            self.assertEqual([x['mean'] for x in complete.history], [x['mean'] for x in resumed.history])

    def test_accelerated_mode_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, 'accelerated=False'):
            NativePPO(accelerated=True).learn(make_agent(), Reach1D, 8, 0, 0)

    def test_fixed_topology_ppo_batches_independent_genomes(self):
        agents = [make_agent(seed=i) for i in range(2)]
        snapshots = [a.genome.vector() for a in agents]
        topology = [a.structure for a in agents]
        reports = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4).learn_population(
            agents, Reach1D, None, 8, 1, 0)
        for agent, snapshot, structure, report in zip(agents, snapshots, topology, reports):
            self.assertEqual(agent.structure, structure)
            self.assertFalse(np.array_equal(agent.genome.vector(), snapshot))
            self.assertEqual(report.interactions, 8)
            self.assertEqual(agent.training_state['adam']['step'], 2)
        self.assertFalse(np.shares_memory(agents[0].genome.values, agents[1].genome.values))

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_learning_and_evaluation(self):
        agent = make_agent(GraphGrowth(check_every=4))
        report = NativePPO(device='cuda', rollout_steps=4, epochs=1, minibatch_size=4).learn_population(
            [agent], Reach1D, TensorReach1D, 8, 0, 0)[0]
        self.assertEqual(report.interactions, 8)
        score = evaluate_batched([agent], [Scenario(1)], Reach1D, TensorReach1D, device='cuda')[0]
        self.assertTrue(np.isfinite(score.fitness))


if __name__ == '__main__':
    unittest.main()
