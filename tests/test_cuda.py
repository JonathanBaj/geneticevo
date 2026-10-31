"""Hardware validation for CUDA execution; skipped without a usable GPU."""
from functools import partial
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth, Scenario, load_agent
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator

try:
    import torch
    from geneticevo.ppo import NativePPO, IndependentAdam, clip_individual_gradients, ppo_losses, generalized_advantages
    from geneticevo.tensor import BatchedPolicy, IndependentRNG, TensorReach1D, evaluate_batched, resolve_device
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

try:
    from geneticevo.integrations.sb3 import SB3AgentFactory, SB3Learner
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def agents(continuous=False, widths=((4, 4),) * 3):
    env = Reach1D(continuous=continuous)
    try:
        return [MLPAgentFactory(size)(np.random.default_rng(i), env.observation_space, env.action_space)
                for i, size in enumerate(widths)]
    finally:
        env.close()


@unittest.skipUnless(HAS_CUDA, 'CUDA hardware unavailable')
class CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inference_distributions_losses_gradients_clipping_and_adam(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                population = agents(continuous)
                cpu, gpu = BatchedPolicy(population), BatchedPolicy(population, 'cuda:0')
                self.assertTrue(all(p.device.type == 'cuda' for p in gpu.parameters()))
                observation = torch.linspace(-1, 1, 21).reshape(3, 7, 1)
                action = torch.linspace(-0.8, 0.8, 21).reshape(3, 7, 1) if continuous else torch.arange(21).reshape(3, 7) % 3
                old = torch.linspace(-2, -0.1, 21).reshape(3, 7)
                advantage = torch.linspace(-1, 1, 21).reshape(3, 7)
                returns = advantage.square()
                for prefix in ('actor', 'critic'):
                    torch.testing.assert_close(gpu(observation.cuda(), prefix).cpu(), cpu(observation, prefix), atol=2e-6, rtol=2e-5)
                for output_cpu, output_gpu in zip(cpu.sample(observation, None, deterministic=True),
                                                   gpu.sample(observation.cuda(), None, deterministic=True)):
                    torch.testing.assert_close(output_gpu.cpu(), output_cpu, atol=2e-6, rtol=2e-5)
                optimizers = (IndependentAdam(cpu), IndependentAdam(gpu))
                for _ in range(3):
                    norms = []
                    for policy, optimizer in zip((cpu, gpu), optimizers):
                        batch = [value.to(policy.device) for value in (observation, action, old, advantage, returns)]
                        policy.zero_grad(set_to_none=True)
                        loss, _ = ppo_losses(policy, *batch, entropy_coefficient=0.01)
                        loss.sum().backward()
                        norms.append(clip_individual_gradients(policy.parameters(), 0.5).cpu())
                        optimizer.step()
                    torch.testing.assert_close(norms[0], norms[1], atol=3e-6, rtol=3e-5)
                    for name in cpu.params:
                        torch.testing.assert_close(gpu.params[name].grad.cpu(), cpu.params[name].grad, atol=3e-6, rtol=3e-5)
                        torch.testing.assert_close(gpu.params[name].cpu(), cpu.params[name], atol=3e-6, rtol=3e-5)
                # GPU training exports portable NumPy genomes and CPU optimizer state.
                gpu.export(population)
                optimizers[1].export(population)
                for agent in population:
                    self.assertTrue(np.isfinite(agent.genome.vector()).all())
                    self.assertTrue(all(x.device.type == 'cpu' for x in agent.training_state['adam']['first'].values()))

    def test_environment_transitions_masks_bootstrapping_and_random_streams(self):
        for continuous in (False, True):
            env = TensorReach1D(2, 2, 'cuda', horizon=3, continuous=continuous)
            reference = TensorReach1D(2, 2, 'cpu', horizon=3, continuous=continuous)
            scenarios = [Scenario(1, {'position': 0.4}), Scenario(2, {'position': -0.3})]
            torch.testing.assert_close(env.reset(scenarios).cpu(), reference.reset(scenarios))
            for step in range(3):
                action = torch.full((2, 2, 1) if continuous else (2, 2), 0.25 if continuous else 2)
                active = torch.tensor([[True, False], [True, True]])
                for actual, expected in zip(env.step(action.cuda(), active.cuda()), reference.step(action, active)):
                    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(env.t[0, 1].item(), 0)
            self.assertAlmostEqual(env.position[0, 1].item(), -0.3, places=6)
            before = env.position.clone()
            mask = torch.tensor([[True, False], [False, True]], device='cuda')
            env.reset_done(mask)
            torch.testing.assert_close(env.position[~mask], before[~mask])
            self.assertTrue((env.t[mask] == 0).all())
            env.close()
            reference.close()
        rewards = torch.ones((1, 3, 1), device='cuda')
        values = torch.zeros_like(rewards)
        term = torch.tensor([[[True], [False], [False]]], device='cuda')
        trunc = torch.tensor([[[False], [True], [False]]], device='cuda')
        advantage, _ = generalized_advantages(rewards, values, torch.full_like(rewards, 10), term, trunc, 0.9, 1)
        torch.testing.assert_close(advantage.cpu(), torch.tensor([[[1.0], [10.0], [10.0]]]))
        one, many = IndependentRNG([2], 'cuda'), IndependentRNG([2, 100, 200], 'cuda')
        for _ in range(3):
            torch.testing.assert_close(one.normal((5, 2))[0], many.normal((5, 2))[0])
            torch.testing.assert_close(one.uniform((5,))[0], many.uniform((5,))[0])
            torch.testing.assert_close(one.permutations(8)[0], many.permutations(8)[0])
        policy = BatchedPolicy(agents(), 'cuda')
        policy(torch.ones((3, 5, 1), device='cuda'))[0].sum().backward()
        for parameter in policy.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad[1:], torch.zeros_like(parameter.grad[1:]))
        with self.assertRaisesRegex(ValueError, 'CUDA was requested'):
            resolve_device(f'cuda:{torch.cuda.device_count()}')

    def test_frozen_evaluation_native_and_hybrid_variable_widths(self):
        for continuous in (False, True):
            population = agents(continuous, ((3,), (5,), (3,)))
            before = [a.genome.vector() for a in population]
            scenarios = [Scenario(1), Scenario(2), Scenario(3)]
            factory = partial(Reach1D, horizon=5, continuous=continuous)
            for tensor_factory in (None, partial(TensorReach1D, horizon=5, continuous=continuous)):
                actual = evaluate_batched(population, scenarios, factory, tensor_factory, device='cuda', env_copies=2)
                for agent, score, genome in zip(population, actual, before):
                    expected = GymEvaluator(factory).evaluate(agent, scenarios)
                    np.testing.assert_allclose(score.returns, expected.returns, atol=3e-5, rtol=3e-5)
                    self.assertEqual(score.lengths, expected.lengths)
                    self.assertEqual(score.interactions, 15)
                    np.testing.assert_array_equal(genome, agent.genome.vector())

    def test_ppo_checkpoint_resume_fixed_and_growing_native_and_hybrid(self):
        for continuous in (False, True):
            for native in (False, True):
                for growing in (False, True):
                    for mode in ('lamarckian', 'baldwinian'):
                        with self.subTest(continuous=continuous, native=native, growing=growing, inheritance=mode):
                            def experiment(generations):
                                growth = NeuralGrowth(check_every=8, neurons_per_growth=1, max_hidden_size=9,
                                                      mutation_probability=1, mutation_step=1) if growing else None
                                return Evolution(env_factory=partial(Reach1D, horizon=5, continuous=continuous),
                                    tensor_env_factory=partial(TensorReach1D, horizon=5, continuous=continuous) if native else None,
                                    agent_factory=MLPAgentFactory((3, 4), growth),
                                    learner=NativePPO(env_copies=2, rollout_steps=4, epochs=1, minibatch_size=4, device='cuda'),
                                    backend='batched', device='cuda', environment_device='cuda' if native else 'cpu',
                                    config=EvolutionConfig(population_size=3, generations=generations, elite_count=1,
                                        immigrant_count=1, evaluation_episodes=2, learning_budget=8, env_copies=2,
                                        inheritance=mode, diversity_pairs=3, measure_pre_learning=True))
                            complete = experiment(2).run()
                            self.assertEqual(complete.history[-1]['training_interactions'], 48)
                            if growing:
                                self.assertEqual(complete.history[0]['learner_metrics']['growth_events'], 1)
                            with tempfile.TemporaryDirectory() as directory:
                                path = Path(directory)
                                first = experiment(1)
                                first.run()
                                first.checkpoint(path / 'state.checkpoint')
                                resumed = experiment(2).run(path / 'state.checkpoint')
                                self.assertEqual(complete.lineage, resumed.lineage)
                                for left, right in zip(complete.population, resumed.population):
                                    self.assertEqual(left.agent.hidden_sizes, right.agent.hidden_sizes)
                                    np.testing.assert_array_equal(left.agent.genome.vector(), right.agent.genome.vector())
                                self.assertEqual([x['mean'] for x in complete.history], [x['mean'] for x in resumed.history])
                                complete.best_agent.save(path / 'champion')
                                restored = load_agent(path / 'champion')
                                self.assertEqual(restored.hidden_sizes, complete.best_agent.hidden_sizes)
                                np.testing.assert_array_equal(restored.genome.vector(), complete.best_agent.genome.vector())
                                self.assertTrue(np.isfinite(GymEvaluator(partial(Reach1D, horizon=3, continuous=continuous))
                                                            .evaluate(restored, [Scenario(9)]).fitness))

    @unittest.skipUnless(HAS_SB3, 'SB3 extra is not installed')
    def test_sb3_cuda_ppo_dqn_state_reset_resume_and_portable_save(self):
        for algorithm, continuous in (('PPO', False), ('PPO', True), ('DQN', False)):
            with self.subTest(algorithm=algorithm, continuous=continuous):
                factory = partial(Reach1D, horizon=4, continuous=continuous)
                env = factory()
                kwargs = {'n_steps': 4, 'batch_size': 4, 'n_epochs': 1} if algorithm == 'PPO' else {
                    'buffer_size': 100, 'learning_starts': 0, 'batch_size': 4}
                kwargs['policy_kwargs'] = {'net_arch': [8]}
                caller_rng = torch.cuda.get_rng_state_all()
                agent = SB3AgentFactory(algorithm, kwargs)(np.random.default_rng(1),
                    env.observation_space, env.action_space, 'cuda')
                env.close()
                self.assertTrue(all(p.device.type == 'cuda' for p in agent.model.policy.parameters()))
                learner = SB3Learner()
                before = agent.genome.vector()
                report = learner.learn(agent, factory, 8, 1, 2)
                self.assertEqual(report.interactions, 8)
                self.assertGreater(report.updates, 0)
                self.assertFalse(np.array_equal(before, agent.genome.vector()))
                for current, previous in zip(torch.cuda.get_rng_state_all(), caller_rng):
                    torch.testing.assert_close(current, previous)
                continuing, fresh = agent.clone(True), agent.clone(False)
                self.assertEqual(fresh.model.num_timesteps, 0)
                self.assertEqual(len(fresh.model.policy.optimizer.state), 0)
                if algorithm == 'DQN':
                    self.assertEqual(fresh.model.replay_buffer.size(), 0)
                learner.learn(agent, factory, 4, 3, 4)
                learner.learn(continuing, factory, 4, 3, 4)
                np.testing.assert_array_equal(agent.genome.vector(), continuing.genome.vector())
                with tempfile.TemporaryDirectory() as directory:
                    agent.save(directory)
                    portable = load_agent(directory)
                    self.assertTrue(all(p.device.type == 'cpu' for p in portable.model.policy.parameters()))
                    np.testing.assert_array_equal(agent.genome.vector(), portable.genome.vector())
                    self.assertTrue(np.isfinite(GymEvaluator(factory).evaluate(portable, [Scenario(1)]).fitness))


if __name__ == '__main__':
    unittest.main()
