"""Sparse connectivity, structural plasticity, inheritance, and frozen inference."""
from dataclasses import replace
from functools import partial
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
    from geneticevo.tensor import GraphBatchedPolicy
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, 'torch extra is not installed')
class GraphInferenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
