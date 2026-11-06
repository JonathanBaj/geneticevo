"""Accelerated PPO invariants; the same checks run on CUDA when available."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geneticevo import Evolution, EvolutionConfig, NeuralGrowth, Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D

try:
    import torch
    from geneticevo.ppo import NativePPO, IndependentAdam, RolloutBuffer, rollout_values, generalized_advantages, ppo_losses
    from geneticevo.tensor import BatchedPolicy, MaskedBatchedPolicy, IndependentRNG, TensorReach1D, evaluate_batched
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch extra is not installed")
class AcceleratedPPOTests(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agents(self, widths=((3, 4), (5, 3), (4, 5)), continuous=False, growing=True):
        env = Reach1D(continuous=continuous)
        growth = NeuralGrowth(check_every=8, max_hidden_size=7, neurons_per_growth=2, mutation_probability=0) if growing else None
        try:
            return [MLPAgentFactory(sizes, growth)(np.random.default_rng(i), env.observation_space, env.action_space)
                    for i, sizes in enumerate(widths)]
        finally:
            env.close()

    def assert_agents_close(self, left, right, exact=False):
        for a, b in zip(left, right):
            self.assertEqual(a.hidden_sizes, b.hidden_sizes)
            if exact:
                np.testing.assert_array_equal(a.genome.vector(), b.genome.vector())
            else:
                np.testing.assert_allclose(a.genome.vector(), b.genome.vector(), atol=2e-6, rtol=2e-5)
            self.assertEqual(a.training_state.get("neural_growth"), b.training_state.get("neural_growth"))
            self.assertEqual(a.training_state["adam"]["step"], b.training_state["adam"]["step"])
            for kind in ("first", "second"):
                for name, values in a.training_state["adam"][kind].items():
                    torch.testing.assert_close(values, b.training_state["adam"][kind][name], atol=2e-6, rtol=2e-5)

    def test_reused_actor_output_matches_existing_distributions(self):
        for continuous in (False, True):
            policy = BatchedPolicy(self.agents(((3,),) * 3, continuous, False), self.device)
            observation = torch.ones(3, 2, 1, device=self.device)
            original = policy.sample(observation, IndependentRNG([1, 2, 3], self.device))
            reused = policy.sample(observation, IndependentRNG([1, 2, 3], self.device), reuse_output=True)
            for a, b in zip(original, reused):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_deferred_values_bootstrap_asynchronous_resets_and_truncations(self):
        torch.manual_seed(3)
        policy = BatchedPolicy(self.agents(((3,),) * 3, growing=False), self.device)
        buffer = RolloutBuffer(policy, 5, 2, True).data
        buffer["observations"].normal_()
        buffer["next_observations"][:, :-1].copy_(buffer["observations"][:, 1:])
        buffer["next_observations"][:, -1].normal_()
        buffer["rewards"].normal_()
        buffer["terminated"].zero_()
        buffer["truncated"].zero_()
        for p, t, e, terminal in ((0, 1, 0, True), (1, 2, 1, False), (2, 0, 0, False), (2, 3, 1, False), (0, 4, 1, False)):
            buffer["terminated" if terminal else "truncated"][p, t, e] = True
            buffer["next_observations"][p, t, e] = 9.0
        with torch.no_grad():
            values = policy.values(buffer["observations"].flatten(1, 2)).reshape(3, 5, 2)
            next_values = policy.values(buffer["next_observations"].flatten(1, 2)).reshape(3, 5, 2)
            expected = generalized_advantages(buffer["rewards"], values, next_values,
                buffer["terminated"], buffer["truncated"], 0.9, 0.95)
        # Generic environments use the observed max truncations, once/rollout.
        rollout_values(policy, buffer, object())
        actual = generalized_advantages(buffer["rewards"], buffer["values"], buffer["next_values"],
            buffer["terminated"], buffer["truncated"], 0.9, 0.95)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_masked_gradients_adam_independence_and_portable_state(self):
        for continuous in (False, True):
            agents = self.agents(continuous=continuous)
            policy = MaskedBatchedPolicy(agents, self.device)
            adam = IndependentAdam(policy, optimized=True)
            observation = torch.ones(3, 6, 1, device=self.device)
            with torch.no_grad():
                actions = policy.sample(observation, IndependentRNG([1, 2, 3], self.device))[1]
            before = {name: value.detach().clone() for name, value in policy.params.items()}
            losses, _ = ppo_losses(policy, observation, actions,
                torch.zeros(3, 6, device=self.device), torch.ones(3, 6, device=self.device),
                torch.ones(3, 6, device=self.device))
            losses[0].backward()
            for name, parameter in policy.params.items():
                torch.testing.assert_close(parameter.grad[1:], torch.zeros_like(parameter.grad[1:]), atol=0, rtol=0)
                shape = dict(agents[0].layout())[name]
                active = torch.zeros_like(parameter[0], dtype=torch.bool)
                active[tuple(slice(0, n) for n in shape)] = True
                torch.testing.assert_close(parameter.grad[0][~active], torch.zeros_like(parameter.grad[0][~active]), atol=0, rtol=0)
            adam.step()
            for name in policy.params:
                torch.testing.assert_close(policy.params[name][1:], before[name][1:], atol=0, rtol=0)
                torch.testing.assert_close(adam.first[name][1:], torch.zeros_like(adam.first[name][1:]), atol=0, rtol=0)
                torch.testing.assert_close(adam.second[name][1:], torch.zeros_like(adam.second[name][1:]), atol=0, rtol=0)
            untouched = {name: value[0].clone() for name, value in adam.first.items()}
            adam.reset_members([1])
            self.assertEqual(adam.steps.tolist(), [1, 0, 1])
            for name in untouched:
                torch.testing.assert_close(adam.first[name][0], untouched[name], atol=0, rtol=0)
            policy.export(agents)
            adam.export(agents)
            restored = IndependentAdam(MaskedBatchedPolicy(agents, self.device), agents=agents, optimized=True)
            torch.testing.assert_close(restored.steps, adam.steps, atol=0, rtol=0)
            for name in adam.first:
                torch.testing.assert_close(restored.first[name], adam.first[name], atol=0, rtol=0)
                torch.testing.assert_close(restored.second[name], adam.second[name], atol=0, rtol=0)
            for agent in agents:
                for name, shape in agent.layout():
                    self.assertEqual(tuple(agent.training_state["adam"]["first"][name].shape), shape)

    def test_growth_preserves_outputs_and_existing_trained_weights(self):
        for continuous in (False, True):
            agents = self.agents(continuous=continuous)
            policy = MaskedBatchedPolicy(agents, self.device)
            observations = torch.ones(3, 5, 1, device=self.device)
            # Simulate GPU training without exporting the overlap to the agent.
            with torch.no_grad():
                policy.params["actor_b0"][0, :3].add_(0.25)
                policy.params["critic_w1"][0, :3, :4].add_(0.5)
                expected = [policy(observations, prefix).clone() for prefix in ("actor", "critic")]
                pointers = [p.data_ptr() for p in policy.parameters()]
            old = agents[0].hidden_sizes
            agents[0].resize_hidden_sizes((5, 6), np.random.default_rng(22))
            policy.activate(0, agents[0], old)
            self.assertEqual([p.data_ptr() for p in policy.parameters()], pointers)
            for prefix, output in zip(("actor", "critic"), expected):
                torch.testing.assert_close(policy(observations, prefix), output, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(policy.params["actor_w2"][0, 4:6], torch.zeros_like(policy.params["actor_w2"][0, 4:6]))

    def test_masked_losses_gradients_and_adam_match_separate_networks(self):
        for continuous in (False, True):
            agents = self.agents(continuous=continuous)
            policy = MaskedBatchedPolicy(agents, self.device)
            optimizer = IndependentAdam(policy, optimized=True)
            references = [BatchedPolicy([a], self.device) for a in agents]
            optimizers = [IndependentAdam(p) for p in references]
            observations = torch.linspace(-1, 1, 18, device=self.device).reshape(3, 6, 1)
            with torch.no_grad():
                actions = policy.sample(observations, IndependentRNG([1, 2, 3], self.device))[1]
            zeros, ones = torch.zeros(3, 6, device=self.device), torch.ones(3, 6, device=self.device)
            for _ in range(2):
                policy.zero_grad()
                loss, _ = ppo_losses(policy, observations, actions, zeros, ones, ones)
                loss.sum().backward()
                for i, reference in enumerate(references):
                    reference.zero_grad()
                    expected, _ = ppo_losses(reference, observations[i:i+1], actions[i:i+1],
                                            zeros[i:i+1], ones[i:i+1], ones[i:i+1])
                    torch.testing.assert_close(loss[i], expected[0])
                    expected.sum().backward()
                    for name, shape in agents[i].layout():
                        slices = (i, *[slice(0, n) for n in shape])
                        torch.testing.assert_close(policy.params[name].grad[slices], reference.params[name].grad[0])
                    optimizers[i].step()
                optimizer.step()
                for i, reference in enumerate(references):
                    for name, shape in agents[i].layout():
                        slices = (i, *[slice(0, n) for n in shape])
                        torch.testing.assert_close(policy.params[name][slices], reference.params[name][0])
                        torch.testing.assert_close(optimizer.first[name][slices], optimizers[i].first[name][0])
                        torch.testing.assert_close(optimizer.second[name][slices], optimizers[i].second[name][0])

    def test_chunks_and_regrouping_preserve_streams_and_training(self):
        for continuous in (False, True):
            originals = self.agents(continuous=continuous)
            learner = NativePPO(env_copies=2, rollout_steps=4, epochs=2, minibatch_size=5,
                                device=self.device, accelerated=True)
            factory = partial(Reach1D, horizon=3, continuous=continuous)
            tensor = partial(TensorReach1D, horizon=3, continuous=continuous)
            full = [a.clone() for a in originals]
            reports = learner.learn_population(full, factory, tensor, 17, 42, 0, member_ids=[11, 5, 23])
            self.assertEqual([r.interactions for r in reports], [24] * 3)
            self.assertEqual([r.updates for r in reports], [12] * 3)
            order = [2, 0, 1]
            regrouped = [originals[i].clone() for i in order]
            replace(learner, population_chunk_size=2).learn_population(regrouped, factory, tensor, 17, 42, 0,
                member_ids=[[11, 5, 23][i] for i in order])
            self.assert_agents_close(full, [regrouped[order.index(i)] for i in range(3)])
            scenarios = [Scenario(1), Scenario(2)]
            all_scores = evaluate_batched(full, scenarios, factory, tensor, self.device, masked=True)
            chunks = evaluate_batched(full, scenarios, factory, tensor, self.device, population_chunk_size=1, masked=True)
            for a, b in zip(all_scores, chunks):
                np.testing.assert_allclose(a.returns, b.returns, atol=1e-6)

    def test_reused_values_and_buffers_match_per_step_reference(self):
        for continuous in (False, True):
            originals = self.agents(((4,),) * 3, continuous, False)
            learner = NativePPO(env_copies=2, rollout_steps=4, epochs=1, minibatch_size=4,
                                accelerated=True, device=self.device)
            factory = partial(Reach1D, horizon=3, continuous=continuous)
            tensor = partial(TensorReach1D, horizon=3, continuous=continuous)
            reference = [a.clone() for a in originals]
            replace(learner, reuse_rollout_values=False).learn_population(reference, factory, tensor, 16, 42, 0)
            for preallocate in (False, True):
                agents = [a.clone() for a in originals]
                replace(learner, preallocate_rollouts=preallocate).learn_population(agents, factory, tensor, 16, 42, 0)
                self.assert_agents_close(agents, reference)

    def test_growth_keeps_environment_alive_and_resets_only_changed_members(self):
        agents = self.agents(((3,), (7,)))
        agents[0].growth = replace(agents[0].growth, check_every=4)
        instances = []
        class CountingEnv(TensorReach1D):
            def __init__(self, *args):
                super().__init__(*args, horizon=50)
                self.resets = 0
                instances.append(self)
            def reset(self, scenarios=None):
                self.resets += 1
                return super().reset(scenarios)
        NativePPO(rollout_steps=4, epochs=1, minibatch_size=4, accelerated=True,
            device=self.device).learn_population(agents, Reach1D, CountingEnv, 12, 42, 0)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].resets, 1)
        self.assertEqual(agents[0].hidden_sizes, (7,))
        self.assertEqual(agents[0].training_state["adam"]["step"], 1)
        self.assertEqual(agents[1].training_state["adam"]["step"], 3)
        self.assertGreater(np.linalg.norm(agents[0].parameters()["actor_w1"][5:]), 0)

    def test_checkpoint_resume_both_inheritance_modes_and_environment_paths(self):
        for inheritance in ("lamarckian", "baldwinian"):
            for tensor in (False, True):
                def experiment(generations):
                    return Evolution(agent_factory=MLPAgentFactory((3,), NeuralGrowth(check_every=4,
                        neurons_per_growth=1, max_hidden_size=6, mutation_probability=1)),
                        env_factory=partial(Reach1D, horizon=5),
                        tensor_env_factory=partial(TensorReach1D, horizon=5) if tensor else None,
                        backend="batched", device=self.device, environment_device=self.device if tensor else "cpu",
                        learner=NativePPO(rollout_steps=4, epochs=1, minibatch_size=4,
                            accelerated=True, population_chunk_size=2, device=self.device),
                        config=EvolutionConfig(population_size=3, generations=generations, elite_count=1,
                            learning_budget=8, evaluation_episodes=1, inheritance=inheritance))
                complete = experiment(3).run()
                with tempfile.TemporaryDirectory() as directory:
                    short = experiment(1)
                    short.run()
                    path = Path(directory) / "state.checkpoint"
                    short.checkpoint(path)
                    resumed = experiment(3).run(path)
                self.assertEqual(complete.lineage, resumed.lineage)
                self.assertEqual([x["mean"] for x in complete.history], [x["mean"] for x in resumed.history])
                if inheritance == "lamarckian":
                    self.assert_agents_close([x.agent for x in complete.population], [x.agent for x in resumed.population], exact=True)
                else:
                    for a, b in zip(complete.population, resumed.population):
                        np.testing.assert_array_equal(a.agent.genome.vector(), b.agent.genome.vector())
                        self.assertFalse(hasattr(a.agent, "training_state"))

    def test_invalid_capacity_chunks_and_stream_ids(self):
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                NativePPO(accelerated=True, population_chunk_size=chunk)
        with self.assertRaises(ValueError):
            NativePPO(population_chunk_size=2)
        with self.assertRaisesRegex(ValueError, "maximum width"):
            MaskedBatchedPolicy(self.agents(), self.device, (5, 5))
        with self.assertRaisesRegex(ValueError, "member_ids"):
            NativePPO(accelerated=True, device=self.device).learn_population(self.agents(), Reach1D,
                TensorReach1D, 4, 42, 0, member_ids=[1, 1, 2])


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA hardware unavailable")
class AcceleratedCUDA(AcceleratedPPOTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
