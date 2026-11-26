from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys

from gymnasium import Env, spaces
import numpy as np
import pytest

from geneticevo import Evolution, EvolutionConfig, GraphAgentFactory, GraphGrowth, NeuralGrowth, RustNativePPO
from geneticevo.agents import MLPAgentFactory, load_agent
from geneticevo.environments import Reach1D
from geneticevo.evaluation import AdaptationTask, GymEvaluator, adaptation_evaluation
from geneticevo.core import Scenario
from geneticevo.rust_ppo import _population, topology


@pytest.fixture
def rust():
    pytest.importorskip("_geneticevo_rust")
    return RustNativePPO(env_copies=2, rollout_steps=3, epochs=1, minibatch_size=4)


def test_rust_request_explains_missing_extension(monkeypatch):
    monkeypatch.delitem(sys.modules, "geneticevo._rust", raising=False)
    monkeypatch.setitem(sys.modules, "_geneticevo_rust", None)
    with pytest.raises(ImportError, match="geneticevo-core"):
        RustNativePPO()


@pytest.mark.parametrize("factory", [MLPAgentFactory((4,)), GraphAgentFactory()])
def test_cpu_neural_learning_budget_defaults_to_rust(rust, factory):
    experiment = Evolution(agent_factory=factory, env_factory=partial(Reach1D, horizon=4),
                           config=EvolutionConfig(population_size=3, generations=1, elite_count=1,
                                                  env_copies=2, learning_budget=5))
    assert experiment.backend == "rust"
    assert isinstance(experiment.learner, RustNativePPO)
    assert experiment.learner.env_copies == 2
    assert experiment.run().history[0]["training_interactions"] == 192


def run_ppo_example(tmp_path, example, *args):
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    python_path = (str(Path(entry).resolve()) for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry)
    environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), *python_path))
    subprocess.run([sys.executable, str(repo / "examples" / f"{example}.py"), *args], cwd=tmp_path,
                   env=environment, check=True)
    return json.loads((tmp_path / "results" / example / "metadata.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("example", ["neural_growth", "native_ppo", "graph_brain"])
def test_ppo_examples_use_rust_by_default(rust, tmp_path, example):
    assert run_ppo_example(tmp_path, example)["backend"] == "rust"


@pytest.mark.parametrize("option", ["--no-rust", "--accelerated"])
def test_ppo_example_explicitly_selects_torch(tmp_path, option):
    pytest.importorskip("torch")
    assert run_ppo_example(tmp_path, "neural_growth", option)["backend"] == "batched"


def make_agents(factories, continuous):
    obs = spaces.Box(-1, 1, (4,), np.float32)
    acts = (spaces.Box(np.array([-2, 0.1, -0.5], np.float32),
                       np.array([4, 1, 0.5], np.float32)) if continuous else spaces.Discrete(3))
    return [f(np.random.default_rng(i), obs, acts) for i, f in enumerate(factories)]


@pytest.mark.parametrize("continuous", [False, True])
def test_mixed_topology_forward_and_deterministic_actions(rust, continuous):
    agents = make_agents([MLPAgentFactory(sizes) for sizes in [(), (4, 6), (96, 192, 144, 80)]]
                         + [GraphAgentFactory(initial_neurons=4, settling_steps=4)], continuous)
    obs = np.random.default_rng(5).normal(size=(len(agents), 17, 4)).astype(np.float32)
    core = _population(agents, rust)
    for critic in (False, True):
        prefix = "critic" if critic else "actor"
        reference = np.stack([agent.forward(inputs, prefix) for agent, inputs in zip(agents, obs)])
        np.testing.assert_allclose(core.forward(obs, critic), reference, atol=2e-6, rtol=2e-5)
    reference = [[agent.act(row, None) for row in inputs] for agent, inputs in zip(agents, obs)]
    np.testing.assert_allclose(core.act(obs), reference, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("continuous", [False, True])
def test_widening_preserves_rust_policy_and_values(rust, continuous):
    agent, = make_agents([MLPAgentFactory((4, 5), NeuralGrowth(check_every=6, max_hidden_size=12))], continuous)
    obs = np.random.default_rng(10).normal(size=(1, 16, 4)).astype(np.float32)
    core = _population([agent], rust)
    before = core.forward(obs), core.forward(obs, True)
    agent.resize_hidden_sizes((7, 5), np.random.default_rng(11))
    core.replace_topology(0, topology(agent), agent.genome.values)
    for actual, expected in zip((core.forward(obs), core.forward(obs, True)), before):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("widths", [(6, 7), (32, 80)])
def test_fixed_rollout_ppo_update_matches_torch(rust, continuous, widths):
    torch = pytest.importorskip("torch")
    from geneticevo.ppo import IndependentAdam, clip_individual_gradients, generalized_advantages, ppo_losses
    from geneticevo.tensor import build_policy

    rust = replace(rust, minibatch_size=6, entropy_coefficient=0.03)
    agent, = make_agents([MLPAgentFactory(widths)], continuous)
    core = _population([agent], rust)
    policy = build_policy([agent])
    optimizer = IndependentAdam(policy, rust.learning_rate)
    obs = np.random.default_rng(18).normal(size=(3, 1, 2, 4)).astype(np.float32)
    final_obs = np.concatenate((obs[1:], -obs[-1:])).copy()
    final_obs[0, 0, 0] *= 0.25  # truncation has a different pre-reset state
    rewards = np.array([[[0.1, 0.2]], [[-0.3, 0.4]], [[1.1, -0.6]]], dtype=np.float32)
    terminated = np.array([[[0, 0]], [[0, 1]], [[0, 0]]], dtype=bool)
    truncated = np.array([[[1, 0]], [[0, 0]], [[0, 0]]], dtype=bool)
    sampled = []
    for step in range(rust.rollout_steps):
        sampled.append(core.sample(obs[step]))
        core.record(rewards[step], terminated[step], truncated[step], final_obs[step])
    actual = core.update()[0]
    inputs = torch.from_numpy(obs.swapaxes(0, 1)).flatten(1, 2)
    bootstrap_obs = torch.from_numpy(final_obs.swapaxes(0, 1)).flatten(1, 2)
    actions = np.stack(sampled, axis=1).reshape((1, 6, 3) if continuous else (1, 6))
    if continuous:
        actions = np.arctanh(2 * (actions - agent.low) / (agent.high - agent.low) - 1)
    actions = torch.from_numpy(actions)
    with torch.no_grad():
        old, _ = policy.log_prob(inputs, actions)
        values, next_values = (policy.values(x).reshape(1, 3, 2) for x in (inputs, bootstrap_obs))
        advantages, returns = generalized_advantages(torch.from_numpy(rewards.swapaxes(0, 1)), values,
            next_values, torch.from_numpy(terminated.swapaxes(0, 1)), torch.from_numpy(truncated.swapaxes(0, 1)),
            rust.gamma, rust.gae_lambda)
        advantages = advantages.flatten(1)
        advantages = (advantages - advantages.mean(1, keepdim=True)) / (advantages.std(1, unbiased=False) + 1e-8)
    loss, metrics = ppo_losses(policy, inputs, actions, old, advantages, returns.flatten(1), rust.clip,
                                rust.value_coefficient, rust.entropy_coefficient)
    loss.sum().backward()
    norm = clip_individual_gradients(policy.parameters(), rust.max_gradient_norm)
    optimizer.step()
    policy.export([agent])
    np.testing.assert_allclose(actual, [float(metrics[k].detach()) for k in
        ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")] + [float(norm)],
        atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(core.get_genes(0), agent.genome.values, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("operation, metric, expected", [
    ({"neurons_per_growth": 2}, "width_growth_events", 6),
    ({"shrink_probability": 1}, "width_shrink_events", 1),
    ({"layer_add_probability": 1}, "layers_added", 2),
    ({"layer_remove_probability": 1}, "layers_removed", 0),
])
def test_growth_resets_individual_adam_and_preserves_rng(rust, operation, metric, expected):
    env = Reach1D()
    growth = NeuralGrowth(check_every=6, max_hidden_size=12, min_hidden_layers=0, **operation)
    factory = MLPAgentFactory((4,), growth)
    grower = factory(np.random.default_rng(0), env.observation_space, env.action_space)
    fixed = MLPAgentFactory((4,))(np.random.default_rng(1), env.observation_space, env.action_space)
    reports = rust.learn_population([grower, fixed], Reach1D, None, 5, 6, 0)
    actual = sum(grower.hidden_sizes) if metric.startswith("width") else len(grower.hidden_sizes)
    assert actual == expected
    assert reports[0].requested == 5
    assert reports[0].interactions == 6
    assert reports[0].metrics[metric] == 1
    assert grower.training_state["neural_growth"] == dict(interactions=6, last_check=6,
        events=1, **{dict(width_growth_events="grow", width_shrink_events="shrink",
                          layers_added="add_layer", layers_removed="remove_layer")[metric]: 1})
    _, step, first, second, rng = grower.training_state["rust_ppo"]
    assert step == 0 and first.size == second.size == 0
    assert fixed.training_state["rust_ppo"][1] == 2
    rust.learn_population([grower, fixed], Reach1D, None, 6, 6, 1)
    next_rng = grower.training_state["rust_ppo"][4]
    assert next_rng[::2] == rng[::2]
    assert next_rng[1] > rng[1] and next_rng[3] > rng[3]
    assert fixed.training_state["rust_ppo"][1] == 4


def test_chunking_and_scheduling_do_not_change_training(rust):
    env = Reach1D(continuous=True, horizon=2)
    growth = NeuralGrowth(check_every=10, max_hidden_size=12, shrink_probability=0.2,
                          layer_add_probability=0.5, layer_remove_probability=0.2, min_hidden_layers=0)
    agents = [MLPAgentFactory((i,), growth)(np.random.default_rng(i), env.observation_space, env.action_space)
              for i in range(3, growth.max_hidden_size + 1)]
    copies = [agent.clone() for agent in agents]
    ids = list(range(18, 18 + len(agents)))
    settings = (partial(Reach1D, continuous=True, horizon=2), None, 19, 102, 2)
    reports = rust.learn_population(agents, *settings, member_ids=ids)
    assert reports == replace(rust, population_chunk_size=1).learn_population(copies, *settings, member_ids=ids)
    for agent, other in zip(agents, copies):
        assert agent.hidden_sizes == other.hidden_sizes and agent.layer_ids == other.layer_ids
        _assert_state_equal(agent, other)


def test_growth_threshold_and_rounded_checks_survive_generations(rust):
    env = Reach1D()
    growth = NeuralGrowth(check_every=10, value_loss_threshold=1e9)
    agent = MLPAgentFactory((4,), growth)(np.random.default_rng(0), env.observation_space, env.action_space)
    for generation, last_check in enumerate([0, 12, 12]):
        report, = rust.learn_population([agent], Reach1D, None, 5, 6, generation)
        assert report.metrics["growth_events"] == 0 and agent.hidden_sizes == (4,)
        assert agent.training_state["neural_growth"] == dict(interactions=6 * (generation + 1),
                                                            last_check=last_check, events=0)
        assert agent.training_state["rust_ppo"][1] == 2 * (generation + 1)


def _assert_state_equal(a, b):
    np.testing.assert_array_equal(a.genome.values, b.genome.values)
    a_state, b_state = getattr(a, "training_state", {}), getattr(b, "training_state", {})
    assert a_state.keys() == b_state.keys()
    for key, value in a_state.items():
        if key == "rust_ppo":
            assert value[:2] == b_state[key][:2] and value[4] == b_state[key][4]
            for left, right in zip(value[2:4], b_state[key][2:4]):
                np.testing.assert_array_equal(left, right)
        else:
            assert value == b_state[key]


@pytest.mark.parametrize("mode", ["lamarckian", "baldwinian", "graph"])
def test_evolution_checkpoint_resume_and_portable_champion(rust, tmp_path, mode):
    factory = (GraphAgentFactory(growth=GraphGrowth(check_every=12, node_add_probability=1.0,
                connection_add_probability=1.0)) if mode == "graph" else MLPAgentFactory((4, 5),
                NeuralGrowth(check_every=12, max_hidden_size=12, min_hidden_layers=0,
                    layer_add_probability=0.5, layer_remove_probability=0.25, shrink_probability=0.25,
                    layer_mutation_probability=0.5)))
    cfg = EvolutionConfig(population_size=6, generations=3, elite_count=2, env_copies=2, learning_budget=17,
                          seed=33, inheritance="baldwinian" if mode == "baldwinian" else "lamarckian")

    def experiment(config):
        return Evolution(env_factory=Reach1D, agent_factory=factory, config=config, learner=rust, backend="rust")

    full = experiment(cfg)
    expected = full.run()
    short = experiment(replace(cfg, generations=1))
    short.run()
    checkpoint = tmp_path / "resume.checkpoint"
    short.checkpoint(checkpoint)
    resumed = experiment(cfg)
    actual = resumed.run(resume_from=checkpoint)
    assert actual.best_fitness == expected.best_fitness
    assert actual.lineage == expected.lineage
    for left, right in zip(full.population, resumed.population):
        _assert_state_equal(left.agent, right.agent)
    for row, same in zip(expected.history, actual.history):
        assert row["learner_metrics"] == same["learner_metrics"]
        assert row["training_interactions"] == same["training_interactions"]
    actual.best_agent.save(tmp_path / "champion")
    saved = load_agent(tmp_path / "champion")
    np.testing.assert_array_equal(saved.genome.values, actual.best_agent.genome.values)


def test_rust_elites_can_skip_learning_and_champions_can_adapt(rust):
    cfg = EvolutionConfig(population_size=3, elite_count=3, elite_learning=False, generations=2,
                          learning_budget=5, env_copies=2)
    experiment = Evolution(env_factory=Reach1D, agent_factory=MLPAgentFactory((4,)), config=cfg,
                           learner=rust, backend="rust")
    result = experiment.run()
    assert [row["training_interactions"] for row in result.history] == [18, 18]
    champion = result.best_agent
    vals = [Scenario(42)]
    reports = adaptation_evaluation(champion, rust, Reach1D, GymEvaluator(Reach1D),
                                    [AdaptationTask(Reach1D, vals, "reach")], 5)
    assert reports[0]["learning"].interactions == 6
    assert not hasattr(champion, "training_state")


def test_core_imports_without_torch(rust):
    code = (
        "import sys; sys.modules['torch'] = None; "
        "from geneticevo import RustNativePPO; RustNativePPO()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ppo_learns_an_observation_dependent_action(rust):
    class SignBandit(Env):
        observation_space = spaces.Box(-1, 1, (1,), np.float32)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.target = int(self.np_random.integers(2))
            self.obs = np.array([2 * self.target - 1], np.float32)
            return self.obs, {}

        def step(self, action):
            return self.obs, float(action == self.target), True, False, {}

    agent = MLPAgentFactory(())(np.random.default_rng(0), SignBandit.observation_space, SignBandit.action_space)
    agent.genome.values.fill(0)
    learner = replace(rust, rollout_steps=8, epochs=2, minibatch_size=8, learning_rate=0.02)
    learner.learn(agent, SignBandit, 128, 22, 23)
    assert [agent.act([sign], None) for sign in (-1, 1)] == [0, 1]
