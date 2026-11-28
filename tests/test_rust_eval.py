from functools import partial
import sys

from gymnasium import spaces
import numpy as np
import pytest

from geneticevo import Evolution, EvolutionConfig, GraphAgentFactory, NeuralGrowth, Scenario
from geneticevo.agents import MLPAgentFactory
from geneticevo.environments import Reach1D
from geneticevo.evaluation import GymEvaluator
from geneticevo.rust_eval import _population, evaluate_rust


@pytest.fixture
def rust_available():
    try:
        from geneticevo._rust import Population
    except ImportError:
        pytest.skip("Matching Rust inference extension unavailable")
    return Population


def test_missing_rust_extension_has_an_install_message(monkeypatch):
    monkeypatch.delitem(sys.modules, "geneticevo._rust", raising=False)
    monkeypatch.setitem(sys.modules, "_geneticevo_rust", None)
    with pytest.raises(ImportError, match="geneticevo-core"):
        Evolution(agent_factory=MLPAgentFactory(), env_factory=Reach1D, backend="rust")


@pytest.mark.parametrize("continuous", [False, True])
def test_mixed_topology_forward_matches_numpy(rust_available, continuous):
    observation = spaces.Box(-1, 1, (4,), np.float32)
    action = (spaces.Box(np.array([-2, 0.1, -0.5], np.float32), np.array([4, 1, 0.5], np.float32))
              if continuous else spaces.Discrete(3))
    factories = [MLPAgentFactory(()), MLPAgentFactory((4, 6)), MLPAgentFactory((96, 192, 144)),
                 GraphAgentFactory(initial_neurons=4, settling_steps=4)]
    agents = [factory(np.random.default_rng(i), observation, action) for i, factory in enumerate(factories)]
    obs = np.random.default_rng(5).normal(size=(len(agents), 17, 4)).astype(np.float32)
    core = _population(agents)
    for critic in (False, True):
        prefix = "critic" if critic else "actor"
        reference = np.stack([agent.forward(inputs, prefix) for agent, inputs in zip(agents, obs)])
        np.testing.assert_allclose(core.forward(obs, critic), reference, atol=2e-6, rtol=2e-5)
    actions = [[agent.act(row, None) for row in inputs] for agent, inputs in zip(agents, obs)]
    np.testing.assert_allclose(core.act(obs), actions, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("continuous", [False, True])
def test_frozen_gymnasium_evaluation(rust_available, continuous):
    env_factory = partial(Reach1D, continuous=continuous, horizon=4)
    env = env_factory()
    try:
        factories = [MLPAgentFactory((4,)), GraphAgentFactory(), MLPAgentFactory((5, 6))]
        agents = [f(np.random.default_rng(i), env.observation_space, env.action_space) for i, f in enumerate(factories)]
    finally:
        env.close()
    before = [agent.genome.vector() for agent in agents]
    scenarios = [Scenario(1), Scenario(2, {"position": 0.25}), Scenario(3)]
    scores = evaluate_rust(agents, scenarios, env_factory, 2, 6)
    for agent, score, genes in zip(agents, scores, before):
        expected = GymEvaluator(env_factory).evaluate(agent, scenarios)
        np.testing.assert_allclose(score.returns, expected.returns, atol=2e-6)
        assert score.lengths == expected.lengths and score.interactions == 12
        np.testing.assert_array_equal(agent.genome.values, genes)


def test_evolution_with_rust_evaluation_and_topology_mutations(rust_available):
    experiment = Evolution(
        agent_factory=MLPAgentFactory((4,), NeuralGrowth(mutation_probability=1, layer_mutation_probability=1)),
        env_factory=partial(Reach1D, horizon=4), backend="rust",
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1,
                              env_copies=2, evaluation_episodes=3),
    )
    result = experiment.run()
    assert result.history[-1]["training_interactions"] == 0
    assert len({agent.agent.hidden_sizes for agent in result.population}) > 1
