"""Variable-depth MLP genetics."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import pickle
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth, load_agent
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evolution import parameter_distance


def make(sizes=(3, 4), growth=None, continuous=False, seed=0):
    env = Reach1D(continuous=continuous)
    try:
        return MLPAgentFactory(sizes, growth)(np.random.default_rng(seed), env.observation_space, env.action_space)
    finally:
        env.close()


def plastic(**changes):
    return replace(NeuralGrowth(check_every=4, min_hidden_layers=0, max_hidden_layers=3,
        min_hidden_size=2, max_hidden_size=6, new_layer_size=3, mutation_probability=0,
        layer_mutation_probability=0.5, shrink_probability=0.2,
        layer_add_probability=0.3, layer_remove_probability=0.3), **changes)


class DepthTests(unittest.TestCase):
    def test_insert_remove_at_every_position_preserves_surviving_layers_and_readout(self):
        for continuous in (False, True):
            original = make(continuous=continuous)
            for index in range(3):
                agent = original.clone()
                before = {key: value.copy() for key, value in agent.named_genes().items()}
                agent.training_state = {'adam': {}, 'other': 1}
                agent.insert_hidden_layer(index, 4, np.random.default_rng(index))
                self.assertEqual(len(agent.hidden_sizes), 3)
                self.assertEqual(agent.layer_ids[index], 2)
                for key, old in before.items():
                    current = agent.named_genes()[key]
                    overlap = tuple(slice(0, min(x, y)) for x, y in zip(old.shape, current.shape))
                    np.testing.assert_array_equal(old[overlap], current[overlap])
                self.assertNotIn('adam', agent.training_state)
                self.assertEqual(agent.training_state['other'], 1)
                self.assertTrue(np.isfinite(agent.forward([0.3])).all())
                agent.remove_hidden_layer(index, np.random.default_rng(index))
                self.assertEqual(agent.hidden_sizes, original.hidden_sizes)
                self.assertEqual(agent.layer_ids, original.layer_ids)
                # Insertion wider than its predecessor preserves the overlap,
                # so removing it restores all original genes exactly.
                np.testing.assert_array_equal(agent.genome.vector(), original.genome.vector())

    def test_resize_depth_keeps_output_role_and_allows_linear_policy(self):
        agent = make((3,), plastic())
        bias = agent.parameters()['actor_b1'].copy()
        agent.resize_hidden_sizes((3, 5), np.random.default_rng(0))
        np.testing.assert_array_equal(agent.parameters()['actor_b2'], bias)
        agent.resize_hidden_sizes((), np.random.default_rng(1))
        self.assertEqual(agent.layer_ids, ())
        self.assertEqual(agent.parameters()['actor_w0'].shape, (1, 3))
        self.assertTrue(np.isfinite(agent.forward([0.3])).all())
        agent.insert_hidden_layer(0, 2, np.random.default_rng(1))
        self.assertGreater(agent.layer_ids[0], 1)

    def test_lifetime_grow_shrink_and_remove_depth_respect_bounds(self):
        rng = np.random.default_rng(0)
        growth = plastic(layer_add_probability=1, layer_remove_probability=0, shrink_probability=0)
        agent = make((3,), growth)
        self.assertFalse(growth.after_update(agent, 2, 1, rng))
        self.assertTrue(growth.after_update(agent, 2, 1, rng))
        for _ in range(5):
            growth.after_update(agent, 4, 1, rng)
        self.assertEqual(len(agent.hidden_sizes), 3)
        shrink = plastic(layer_add_probability=0, layer_remove_probability=0, shrink_probability=1)
        agent.growth = shrink
        for _ in range(5):
            shrink.after_update(agent, 4, 1, rng)
        self.assertEqual(agent.hidden_sizes, (2, 2, 2))
        remove = plastic(layer_add_probability=0, layer_remove_probability=1, shrink_probability=0)
        agent.growth = remove
        for _ in range(5):
            remove.after_update(agent, 4, 1, rng)
        self.assertEqual(agent.hidden_sizes, ())
        self.assertEqual(agent.training_state['neural_growth']['remove_layer'], 3)
        self.assertEqual(agent.training_state['neural_growth']['add_layer'], 2)
        self.assertEqual(agent.training_state['neural_growth']['shrink'], 3)

    def test_birth_mutations_add_and_remove_layers_reproducibly(self):
        growth = plastic(layer_mutation_probability=1)
        parent = make((3,), growth)
        depths = set()
        for seed in range(30):
            a, b = parent.clone(), parent.clone()
            self.assertEqual(growth.mutate(a, np.random.default_rng(seed)), growth.mutate(b, np.random.default_rng(seed)))
            np.testing.assert_array_equal(a.genome.vector(), b.genome.vector())
            self.assertEqual(a.layer_ids, b.layer_ids)
            depths.add(len(a.hidden_sizes))
        self.assertEqual(depths, {0, 2})
        self.assertEqual(parent.hidden_sizes, (3,))

    def test_save_initial_agent_and_legacy_loading(self):
        agent = make((3,), plastic())
        agent.insert_hidden_layer(0, 4, np.random.default_rng(0))
        agent.remove_hidden_layer(1, np.random.default_rng(1))
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = load_agent(directory)
            self.assertEqual(restored.layer_ids, agent.layer_ids)
            self.assertEqual(restored.next_layer_id, agent.next_layer_id)
            self.assertEqual(restored.growth, agent.growth)
            np.testing.assert_array_equal(restored.genome.vector(), agent.genome.vector())
            result = Evolution(agent_factory=MLPAgentFactory((3,), agent.growth), initial_agents=(restored,),
                env_factory=partial(Reach1D, horizon=2), config=EvolutionConfig(population_size=2,
                    elite_count=1, generations=1, evaluation_episodes=1)).run()
            self.assertEqual(result.population[0].agent.layer_ids, restored.layer_ids)
            path = Path(directory) / 'agent.json'
            spec = json.loads(path.read_text())
            spec.pop('layer_ids')
            spec.pop('next_layer_id')
            path.write_text(json.dumps(spec))
            self.assertEqual(load_agent(directory).layer_ids, (0,))
        legacy = make()
        del legacy.layer_ids
        del legacy.next_layer_id
        recovered = pickle.loads(pickle.dumps(legacy))
        self.assertEqual(recovered.layer_ids, (0, 1))
        self.assertEqual(recovered.next_layer_id, 2)

    def test_baldwinian_structure_transfer_uses_identity_without_learned_weights(self):
        birth = make((3, 4), plastic())
        learned = birth.clone()
        learned.genome.values += 10
        learned.insert_hidden_layer(0, 3, np.random.default_rng(2))
        learned.remove_hidden_layer(2, np.random.default_rng(2))
        original = {k: v.copy() for k, v in birth.named_genes().items()}
        birth.inherit_structure(learned, np.random.default_rng(3))
        self.assertEqual(birth.layer_ids, learned.layer_ids)
        for key, value in birth.named_genes().items():
            if key in original:
                overlap = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, original[key].shape))
                np.testing.assert_array_equal(value[overlap], original[key][overlap])
        self.assertGreater(parameter_distance(birth, learned), 0)
        self.assertAlmostEqual(parameter_distance(birth, learned), parameter_distance(learned, birth))

    def test_invalid_configuration_and_atomic_edits(self):
        for changes in ({'shrink_probability': -1}, {'layer_add_probability': float('nan')},
                        {'layer_mutation_probability': 2}, {'min_hidden_layers': True},
                        {'min_hidden_layers': 4, 'max_hidden_layers': 3}, {'max_hidden_layers': -1},
                        {'layer_add_probability': 0.8, 'layer_remove_probability': 0.8},
                        {'new_layer_size': 7}):
            with self.assertRaises(ValueError):
                plastic(**changes)
        agent = make((3,), plastic(max_hidden_layers=1))
        before = agent.genome.vector()
        with self.assertRaises(ValueError):
            agent.insert_hidden_layer(0, 3, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            agent.resize_hidden_sizes((3,), np.random.default_rng(0), layer_ids=(True,))
        np.testing.assert_array_equal(agent.genome.vector(), before)


if __name__ == "__main__":
    unittest.main()
