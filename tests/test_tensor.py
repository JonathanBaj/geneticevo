from functools import partial
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
    from torch import nn
    from geneticevo.tensor import BatchedPolicy, IndependentRNG, TensorReach1D, evaluate_batched, resolve_device
    from geneticevo.ppo import NativePPO, IndependentAdam, generalized_advantages, ppo_losses, clip_individual_gradients
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from geneticevo import Evolution, EvolutionConfig, Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator


@unittest.skipUnless(HAS_TORCH, "torch extra is not installed")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agents(self, count=3, continuous=False):
        env = Reach1D(continuous=continuous)
        return [MLPAgentFactory((4,))(np.random.default_rng(i), env.observation_space, env.action_space) for i in range(count)]

    def test_serial_batched_outputs_and_inference(self):
        for continuous in (False, True):
            agents = self.agents(continuous=continuous)
            observations = torch.linspace(-1, 1, 15).reshape(3, 5, 1)
            policy = BatchedPolicy(agents)
            actual = policy(observations).detach().numpy()
            expected = np.stack([agent.forward(observations[i].numpy()) for i, agent in enumerate(agents)])
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            scenarios = [Scenario(1), Scenario(2), Scenario(3)]
            env_factory = partial(Reach1D, horizon=5, continuous=continuous)
            factory = partial(TensorReach1D, horizon=5, continuous=continuous)
            evaluations = evaluate_batched(agents, scenarios, env_factory, factory, env_copies=2, max_steps=10)
            for agent, evaluation in zip(agents, evaluations):
                reference = GymEvaluator(env_factory).evaluate(agent, scenarios)
                np.testing.assert_allclose(evaluation.returns, reference.returns, atol=1e-6)
                self.assertEqual(evaluation.interactions, 15)

    def test_environment_transition_equivalence_and_mask(self):
        for continuous in (False, True):
            cpu = Reach1D(horizon=4, continuous=continuous)
            tensor = TensorReach1D(1, 1, horizon=4, continuous=continuous)
            observation, _ = cpu.reset(seed=2, options={"position": 0.4})
            np.testing.assert_array_equal(observation, tensor.reset([Scenario(2, {"position": 0.4})])[0, 0].numpy())
            for value in (0, 2, 1, 0):
                action = np.array([value / 2 - 0.5], np.float32) if continuous else value
                t_action = torch.tensor(action).reshape(1, 1, 1) if continuous else torch.tensor([[value]])
                o, r, term, trunc, _ = cpu.step(action)
                to, tr, tt, tx = tensor.step(t_action)
                np.testing.assert_allclose(o, to[0, 0].numpy(), atol=1e-7)
                self.assertAlmostEqual(r, tr.item(), places=6)
                self.assertEqual((term, trunc), (tt.item(), tx.item()))
            saved = tensor.position.clone()
            tensor.step(t_action, torch.zeros((1, 1), dtype=torch.bool))
            torch.testing.assert_close(saved, tensor.position)

    def test_terminal_and_truncation_bootstrap(self):
        rewards = torch.ones((1, 3, 1))
        values = torch.zeros_like(rewards)
        next_values = torch.full_like(rewards, 10)
        term = torch.tensor([[[True], [False], [False]]])
        trunc = torch.tensor([[[False], [True], [False]]])
        advantage, returns = generalized_advantages(rewards, values, next_values, term, trunc, 0.9, 1)
        torch.testing.assert_close(advantage, torch.tensor([[[1.0], [10.0], [10.0]]]))

    def reference_network(self, agent, prefix):
        sizes = (agent.observation_dim, *agent.hidden_sizes, agent.action_dim if prefix == "actor" else 1)
        modules = []
        for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
            linear = nn.Linear(left, right)
            params = agent.parameters()
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(params[f"{prefix}_w{index}"].T.copy()))
                linear.bias.copy_(torch.from_numpy(params[f"{prefix}_b{index}"]))
            modules.append(linear)
            if index < len(sizes) - 2:
                modules.append(nn.Tanh())
        return nn.Sequential(*modules)

    def test_ppo_losses_gradients_clipping_and_adam_against_independent_reference(self):
        torch.manual_seed(9)
        agents = self.agents()
        policy = BatchedPolicy(agents)
        references = [(self.reference_network(a, "actor"), self.reference_network(a, "critic")) for a in agents]
        optimizers = [torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4, eps=1e-5) for actor, critic in references]
        adam = IndependentAdam(policy)
        observations = torch.randn(3, 7, 1)
        actions = torch.randint(0, 3, (3, 7))
        old_log_probs = torch.randn(3, 7) - 1
        advantages, returns = torch.randn(3, 7), torch.randn(3, 7)
        for _ in range(3):
            policy.zero_grad()
            losses, metrics = ppo_losses(policy, observations, actions, old_log_probs, advantages, returns, entropy_coefficient=0.01)
            losses.sum().backward()
            for p, (actor, critic) in enumerate(references):
                optimizers[p].zero_grad()
                distribution = torch.distributions.Categorical(logits=actor(observations[p]))
                ratio = torch.exp(distribution.log_prob(actions[p]) - old_log_probs[p])
                objective = -torch.min(ratio * advantages[p], torch.clamp(ratio, 0.8, 1.2) * advantages[p]).mean()
                value_error = torch.mean((critic(observations[p]).squeeze(-1) - returns[p])**2)
                reference_loss = objective + 0.25 * value_error - 0.01 * distribution.entropy().mean()
                torch.testing.assert_close(losses[p], reference_loss)
                reference_loss.backward()
                for prefix, network in (("actor", actor), ("critic", critic)):
                    for layer, linear in enumerate(x for x in network if isinstance(x, nn.Linear)):
                        torch.testing.assert_close(policy.params[f"{prefix}_w{layer}"].grad[p], linear.weight.grad.T)
                        torch.testing.assert_close(policy.params[f"{prefix}_b{layer}"].grad[p], linear.bias.grad)
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            clip_individual_gradients(policy.parameters(), 0.5)
            adam.step()
            for p, (actor, critic) in enumerate(references):
                optimizers[p].step()
                for prefix, network in (("actor", actor), ("critic", critic)):
                    for layer, linear in enumerate(x for x in network if isinstance(x, nn.Linear)):
                        torch.testing.assert_close(policy.params[f"{prefix}_w{layer}"][p], linear.weight.T, atol=3e-7, rtol=1e-5)

    def test_no_cross_individual_gradients(self):
        policy = BatchedPolicy(self.agents())
        observations = torch.ones((3, 5, 1))
        policy(observations)[0].sum().backward()
        for parameter in policy.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad[1:], torch.zeros_like(parameter.grad[1:]))

    def test_categorical_zero_uniform_never_selects_zero_probability(self):
        class ZeroRNG:
            def uniform(self, shape):
                return torch.zeros((1, *shape))
        policy = BatchedPolicy(self.agents(1))
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
            policy.params["actor_b1"][0].copy_(torch.tensor([-1000.0, 0.0, -1000.0]))
        actions, _, log_probs, _ = policy.sample(torch.ones(1, 3, 1), ZeroRNG())
        torch.testing.assert_close(actions, torch.ones_like(actions))
        torch.testing.assert_close(log_probs, torch.zeros_like(log_probs))

    def test_continuous_likelihood_matches_transformed_distribution(self):
        policy = BatchedPolicy(self.agents(continuous=True))
        observation = torch.ones(3, 5, 1)
        raw = torch.linspace(-1, 1, 15).reshape(3, 5, 1)
        log_prob, _ = policy.log_prob(observation, raw)
        base = torch.distributions.Normal(policy(observation), policy.params["log_std"][:, None, :].exp())
        transformed = torch.distributions.TransformedDistribution(base, [torch.distributions.TanhTransform(),
            torch.distributions.AffineTransform((policy.low + policy.high) / 2, (policy.high - policy.low) / 2)])
        torch.testing.assert_close(log_prob, transformed.log_prob(torch.tanh(raw)).sum(-1), atol=1e-6, rtol=1e-5)

    def test_independent_random_streams(self):
        one = IndependentRNG([2], "cpu")
        many = IndependentRNG([2, 100, 200], "cpu")
        for _ in range(3):
            torch.testing.assert_close(one.normal((5, 2))[0], many.normal((5, 2))[0])

    def test_continuous_ppo_reference_gradients(self):
        torch.manual_seed(3)
        agents = self.agents(continuous=True)
        policy = BatchedPolicy(agents)
        observations = torch.randn(3, 6, 1)
        raw = torch.randn(3, 6, 1) * 0.3
        old = torch.randn(3, 6) - 1
        advantages, returns = torch.randn(3, 6), torch.randn(3, 6)
        loss, _ = ppo_losses(policy, observations, raw, old, advantages, returns)
        loss.sum().backward()
        for index, agent in enumerate(agents):
            actor, critic = self.reference_network(agent, "actor"), self.reference_network(agent, "critic")
            log_std = nn.Parameter(torch.zeros(1))
            distribution = torch.distributions.TransformedDistribution(
                torch.distributions.Normal(actor(observations[index]), log_std.exp()),
                [torch.distributions.TanhTransform()])
            log_prob = distribution.log_prob(torch.tanh(raw[index])).sum(-1)
            ratio = (log_prob - old[index]).exp()
            objective = -torch.minimum(ratio * advantages[index], ratio.clamp(0.8, 1.2) * advantages[index]).mean()
            reference = objective + 0.25 * (critic(observations[index]).squeeze(-1) - returns[index]).square().mean()
            reference.backward()
            torch.testing.assert_close(loss[index], reference, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(policy.params["actor_w0"].grad[index], actor[0].weight.grad.T, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(policy.params["log_std"].grad[index], log_std.grad, atol=1e-6, rtol=1e-5)

    def test_native_ppo_budgets_and_state_resume(self):
        def experiment(generations):
            return Evolution(env_factory=partial(Reach1D, horizon=5), agent_factory=MLPAgentFactory((4,)),
                tensor_env_factory=partial(TensorReach1D, horizon=5), backend="batched",
                learner=NativePPO(env_copies=2, rollout_steps=4, epochs=1, minibatch_size=4),
                config=EvolutionConfig(population_size=3, generations=generations, elite_count=1,
                    evaluation_episodes=2, learning_budget=9, env_copies=2))
        complete = experiment(3).run()
        self.assertEqual(complete.history[0]["learning_actual_per_individual"], [16, 16, 16])
        with tempfile.TemporaryDirectory() as directory:
            short = experiment(1)
            short.run()
            path = Path(directory) / "state.checkpoint"
            short.checkpoint(path)
            resumed = experiment(3).run(path)
        for left, right in zip(complete.population, resumed.population):
            np.testing.assert_array_equal(left.agent.genome.vector(), right.agent.genome.vector())
        # Continuing elite has more Adam updates; children reset their counters.
        self.assertEqual(complete.population[0].agent.training_state["adam"]["step"], 12)
        self.assertEqual(complete.population[1].agent.training_state["adam"]["step"], 4)

    def test_continuous_native_update(self):
        agents = self.agents(continuous=True)
        learner = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4)
        before = agents[0].genome.vector()
        reports = learner.learn_population(agents, partial(Reach1D, continuous=True),
                                            partial(TensorReach1D, continuous=True), 4, 0, 0)
        self.assertEqual(reports[0].interactions, 4)
        self.assertFalse(np.array_equal(before, agents[0].genome.vector()))

    def test_reject_incompatible_configurations(self):
        agents = self.agents()
        incompatible = self.agents(1, continuous=True)
        with self.assertRaises(ValueError):
            BatchedPolicy([agents[0], incompatible[0]])
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_cuda_update(self):
        agents = self.agents()
        reports = NativePPO(rollout_steps=4, epochs=1, minibatch_size=4, device="cuda").learn_population(
            agents, Reach1D, TensorReach1D, 4, 0, 0)
        self.assertEqual(reports[0].interactions, 4)


if __name__ == "__main__":
    unittest.main()
