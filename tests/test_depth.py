"""Variable-depth MLP genetics and CPU/CUDA checks for both PPO paths."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import pickle
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth, Scenario, load_agent
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator
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


try:
    import torch
    from geneticevo.ppo import NativePPO, IndependentAdam, ppo_losses
    from geneticevo.tensor import (BatchedPolicy, DynamicBatchedPolicy, IndependentRNG,
                                  TensorReach1D, evaluate_batched)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, 'torch extra is not installed')
class DepthPPOTests(unittest.TestCase):
    device = 'cpu'

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixed_depth_forward_gradients_adam_and_export_match_individual_policies(self):
        for continuous in (False, True):
            agents = [make(s, plastic(), continuous, i) for i, s in enumerate(((), (3,), (4, 2)))]
            policy = DynamicBatchedPolicy(agents, self.device)
            references = [BatchedPolicy([a], self.device) for a in agents]
            observation = torch.linspace(-1, 1, 12, device=self.device).reshape(3, 4, 1)
            self.assertTrue(all(p.device.type == self.device for p in policy.parameters()))
            for prefix in ('actor', 'critic'):
                torch.testing.assert_close(policy(observation, prefix),
                    torch.cat([p(observation[i:i+1], prefix) for i, p in enumerate(references)]))
            with torch.no_grad():
                actions = policy.sample(observation, IndependentRNG([1, 2, 3], self.device))[1]
            zeros, ones = torch.zeros(3, 4, device=self.device), torch.ones(3, 4, device=self.device)
            adam = IndependentAdam(policy, optimized=True)
            loss, _ = ppo_losses(policy, observation, actions, zeros, ones, ones)
            loss.sum().backward()
            for i, reference in enumerate(references):
                expected, _ = ppo_losses(reference, observation[i:i+1], actions[i:i+1],
                                       zeros[i:i+1], ones[i:i+1], ones[i:i+1])
                torch.testing.assert_close(loss[i], expected[0])
                expected.sum().backward()
                slots = policy.parameter_slots(agents[i], i)
                for name, (key, slices) in slots.items():
                    torch.testing.assert_close(policy.params[key].grad[slices], reference.params[name].grad[0])
                for key, parameter in policy.params.items():
                    active = torch.zeros_like(parameter[i], dtype=torch.bool)
                    for slot_key, slices in slots.values():
                        if slot_key == key:
                            active[slices[1:]] = True
                    torch.testing.assert_close(parameter.grad[i][~active], torch.zeros_like(parameter.grad[i][~active]))
            adam.step()
            policy.export(agents)
            adam.export(agents)
            restored = IndependentAdam(DynamicBatchedPolicy(agents, self.device), agents=agents, optimized=True)
            torch.testing.assert_close(restored.steps, adam.steps)
            for key in adam.first:
                torch.testing.assert_close(restored.first[key], adam.first[key])
                torch.testing.assert_close(restored.second[key], adam.second[key])

    def test_depth_edits_keep_tensors_peers_and_environment_alive(self):
        for accelerated in (False, True):
            agents = [make((3,), plastic(layer_add_probability=1, layer_remove_probability=0,
                shrink_probability=0, layer_mutation_probability=0)), make((3,), None, seed=1)]
            instances = []
            class CountingEnv(TensorReach1D):
                def __init__(self, *args):
                    super().__init__(*args, horizon=50)
                    self.resets = 0
                    instances.append(self)
                def reset(self, scenarios=None):
                    self.resets += 1
                    return super().reset(scenarios)
            learner = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4, accelerated=accelerated, device=self.device)
            reports = learner.learn_population(agents, Reach1D, CountingEnv, 12, 42, 0)
            self.assertEqual(len(agents[0].hidden_sizes), 3)
            self.assertEqual(agents[0].training_state['adam']['step'], 1)
            self.assertEqual(agents[1].training_state['adam']['step'], 3)
            self.assertEqual(reports[0].metrics['growth_events'], 2)
            new_positions = [i for i, identity in enumerate(agents[0].layer_ids) if identity > 0]
            self.assertTrue(any(not np.array_equal(agents[0].parameters()[f'actor_w{i}'],
                np.eye(*agents[0].parameters()[f'actor_w{i}'].shape, dtype=np.float32)) for i in new_positions))
            self.assertTrue(all(env.resets == 1 for env in instances))
            if accelerated:
                self.assertEqual(len(instances), 1)
        # Direct structural activation keeps parameter objects and peers intact.
        policy = DynamicBatchedPolicy(agents, self.device)
        pointers = [p.data_ptr() for p in policy.parameters()]
        peers = {k: v[1].detach().clone() for k, v in policy.params.items()}
        agents[0].remove_hidden_layer(0, np.random.default_rng(1))
        policy.activate(0, agents[0], None)
        self.assertEqual([p.data_ptr() for p in policy.parameters()], pointers)
        for key, value in peers.items():
            torch.testing.assert_close(policy.params[key][1], value, atol=0, rtol=0)

    def test_lifetime_pruning_to_zero_depth_and_frozen_mixed_depth_evaluation(self):
        for accelerated in (False, True):
            for continuous in (False, True):
                growth = plastic(layer_remove_probability=1, layer_add_probability=0, shrink_probability=0)
                agents = [make((3, 4), growth, continuous), make((3,), None, continuous, 1)]
                factory = partial(Reach1D, continuous=continuous, horizon=3)
                NativePPO(rollout_steps=4, epochs=1, minibatch_size=4, accelerated=accelerated, device=self.device).learn_population(
                    agents, factory, None, 12, 42, 0)
                self.assertEqual(agents[0].hidden_sizes, ())
                snapshots = [a.genome.vector() for a in agents]
                for tensor in (None, partial(TensorReach1D, continuous=continuous, horizon=3)):
                    scores = evaluate_batched(agents, [Scenario(1), Scenario(2)], factory, tensor,
                        env_copies=2, masked=accelerated, device=self.device)
                    for agent, snapshot, score in zip(agents, snapshots, scores):
                        expected = GymEvaluator(factory).evaluate(agent, [Scenario(1), Scenario(2)])
                        np.testing.assert_allclose(score.returns, expected.returns, atol=1e-6)
                        np.testing.assert_array_equal(agent.genome.vector(), snapshot)

    def test_accelerated_chunks_and_member_order_preserve_training(self):
        originals = [make(s, plastic(), seed=i) for i, s in enumerate(((), (3,), (4, 2)))]
        learner = NativePPO(accelerated=True, rollout_steps=4, epochs=1, minibatch_size=4, device=self.device)
        full = [a.clone() for a in originals]
        learner.learn_population(full, Reach1D, TensorReach1D, 20, 42, 0, member_ids=[11, 5, 23])
        order = [2, 0, 1]
        chunked = [originals[i].clone() for i in order]
        replace(learner, population_chunk_size=2).learn_population(chunked, Reach1D, TensorReach1D,
            20, 42, 0, member_ids=[[11, 5, 23][i] for i in order])
        for i, agent in enumerate(full):
            other = chunked[order.index(i)]
            self.assertEqual(agent.hidden_sizes, other.hidden_sizes)
            self.assertEqual(agent.layer_ids, other.layer_ids)
            np.testing.assert_allclose(agent.genome.vector(), other.genome.vector(), atol=2e-6, rtol=2e-5)

    def test_checkpoint_resume_both_ppo_paths_and_weight_inheritance_modes(self):
        for accelerated in (False, True):
            for mode in ('lamarckian', 'baldwinian'):
                def experiment(generations, output=None):
                    return Evolution(agent_factory=MLPAgentFactory((3,), plastic()),
                        env_factory=partial(Reach1D, horizon=3), tensor_env_factory=partial(TensorReach1D, horizon=3),
                        backend='batched', device=self.device, environment_device=self.device,
                        learner=NativePPO(rollout_steps=4, epochs=1, minibatch_size=4,
                            accelerated=accelerated, device=self.device), config=EvolutionConfig(population_size=3,
                            elite_count=1, generations=generations, learning_budget=12, evaluation_episodes=1,
                            inheritance=mode, diversity_pairs=3, output_dir=output))
                complete = experiment(3).run()
                with tempfile.TemporaryDirectory() as directory:
                    experiment(1, directory).run()
                    resumed = experiment(3).run(Path(directory) / 'latest.checkpoint')
                self.assertEqual(complete.lineage, resumed.lineage)
                self.assertEqual([x['mean'] for x in complete.history], [x['mean'] for x in resumed.history])
                for a, b in zip(complete.population, resumed.population):
                    self.assertEqual(a.agent.layer_ids, b.agent.layer_ids)
                    self.assertEqual(a.agent.next_layer_id, b.agent.next_layer_id)
                    self.assertEqual(a.agent.hidden_sizes, b.agent.hidden_sizes)
                    np.testing.assert_array_equal(a.agent.genome.vector(), b.agent.genome.vector())

    def test_capacity_covers_depth_bound(self):
        with self.assertRaisesRegex(ValueError, 'depth'):
            DynamicBatchedPolicy([make((3,), plastic())], self.device, capacity=(6,))

    def test_fixed_mixed_depth_chunks_share_capacity_safely(self):
        agents = [make(s, seed=i) for i, s in enumerate(((), (3,), (4, 2)))]
        reports = NativePPO(accelerated=True, population_chunk_size=1, rollout_steps=4, device=self.device,
            epochs=1, minibatch_size=4).learn_population(agents, Reach1D, TensorReach1D, 4, 1, 0)
        self.assertEqual([r.interactions for r in reports], [4, 4, 4])

    def test_width_shrink_regrow_clears_removed_slots(self):
        growth = plastic(shrink_probability=1, layer_add_probability=0, layer_remove_probability=0)
        agent = make((5, 4), growth)
        policy = DynamicBatchedPolicy([agent], self.device)
        agent.resize_hidden_sizes((2, 2), np.random.default_rng(0))
        policy.activate(0, agent, (5, 4))
        for name, parameter in policy.params.items():
            active = torch.zeros_like(parameter[0], dtype=torch.bool)
            for key, slices in policy.parameter_slots(agent, 0).values():
                if key == name:
                    active[slices[1:]] = True
            self.assertTrue(torch.count_nonzero(parameter[0][~active]) == 0)
        expected = agent.forward([0.3])
        agent.resize_hidden_sizes((5, 4), np.random.default_rng(1))
        policy.activate(0, agent, (2, 2))
        np.testing.assert_allclose(policy(torch.tensor([[[0.3]]], device=self.device)).detach().cpu().numpy()[0, 0], expected, atol=1e-6)
        agent = make((5, 4), growth)
        report = NativePPO(accelerated=True, rollout_steps=4, epochs=1, minibatch_size=4, device=self.device).learn(
            agent, Reach1D, 8, 1, 2)
        self.assertEqual(agent.hidden_sizes, (2, 2))
        self.assertEqual(report.metrics['width_shrink_events'], 2)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_depth_growth_pruning_and_inference(self):
        for accelerated in (False, True):
            for continuous in (False, True):
                growth = plastic(shrink_probability=0, layer_add_probability=1,
                                 layer_remove_probability=0, layer_mutation_probability=0)
                agents = [make((3,), growth, continuous), make((3, 2), None, continuous, 1)]
                factory = partial(Reach1D, horizon=3, continuous=continuous)
                tensor = partial(TensorReach1D, horizon=3, continuous=continuous)
                learner = NativePPO(accelerated=accelerated, device='cuda', rollout_steps=4,
                                    epochs=1, minibatch_size=4)
                learner.learn_population(agents, factory, tensor, 8, 1, 0)
                self.assertEqual(len(agents[0].hidden_sizes), 3)
                agents[0].growth = replace(growth, layer_add_probability=0, layer_remove_probability=1)
                learner.learn_population(agents, factory, tensor, 12, 1, 1)
                self.assertEqual(agents[0].hidden_sizes, ())
                policy = DynamicBatchedPolicy(agents, 'cuda')
                observation = torch.full((2, 1, 1), 0.3, device='cuda')
                for prefix in ('actor', 'critic'):
                    expected = np.stack([a.forward([[0.3]], prefix) for a in agents])
                    np.testing.assert_allclose(policy(observation, prefix).detach().cpu().numpy(), expected,
                                               atol=1e-5, rtol=1e-5)
                scores = evaluate_batched(agents, [Scenario(1)], factory, tensor, device='cuda', masked=accelerated)
                self.assertTrue(all(np.isfinite(s.fitness) for s in scores))


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA hardware unavailable')
class DepthCUDA(DepthPPOTests):
    device = 'cuda'

    def test_dynamic_hybrid_continuous_checkpoint_resume(self):
        for accelerated in (False, True):
            for mode in ('lamarckian', 'baldwinian'):
                with self.subTest(accelerated=accelerated, inheritance=mode):
                    def experiment(generations):
                        return Evolution(agent_factory=MLPAgentFactory((3,), plastic()),
                            env_factory=partial(Reach1D, horizon=3, continuous=True), backend='batched',
                            device=self.device, learner=NativePPO(device=self.device, rollout_steps=4,
                                epochs=1, minibatch_size=4, accelerated=accelerated),
                            config=EvolutionConfig(population_size=3, elite_count=1, generations=generations,
                                learning_budget=12, evaluation_episodes=1, inheritance=mode, diversity_pairs=3))
                    complete = experiment(3).run()
                    with tempfile.TemporaryDirectory() as directory:
                        initial = experiment(1)
                        initial.run()
                        path = Path(directory) / 'state.checkpoint'
                        initial.checkpoint(path)
                        resumed = experiment(3).run(path)
                    self.assertEqual(complete.lineage, resumed.lineage)
                    self.assertEqual([r['mean'] for r in complete.history], [r['mean'] for r in resumed.history])
                    for a, b in zip(complete.population, resumed.population):
                        self.assertEqual(a.agent.hidden_sizes, b.agent.hidden_sizes)
                        self.assertEqual(a.agent.layer_ids, b.agent.layer_ids)
                        self.assertEqual(a.agent.next_layer_id, b.agent.next_layer_id)
                        np.testing.assert_array_equal(a.agent.genome.vector(), b.agent.genome.vector())
                        if mode == 'lamarckian':
                            self.assertEqual(a.agent.training_state['neural_growth'], b.agent.training_state['neural_growth'])
                            self.assertEqual(a.agent.training_state['adam']['step'], b.agent.training_state['adam']['step'])
                            for name, _ in a.agent.layout():
                                for moment in ('first', 'second'):
                                    torch.testing.assert_close(a.agent.training_state['adam'][moment][name],
                                        b.agent.training_state['adam'][moment][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
