"""Generation-boundary orchestration with frozen validation and independent archives."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import time
from typing import Callable

import numpy as np

from .core import EvolutionConfig, Individual, scenarios_for, seed_for
from .evaluation import GymEvaluator
from .evolution import GaussianMutation, select
from .execution import evaluate_population
from .results import Result, load_checkpoint, save_checkpoint, software_metadata, write_json, json_default, describe
import json


class Evolution:
    """Generation 0 is evaluated. `generations` counts evaluated populations.

    Stops and checkpoints happen after complete generations. Time/interaction
    caps may therefore overshoot by a generation, and requested learning budgets
    may round to backend update boundaries. All actual costs are recorded.
    """

    def __init__(self, *, agent_factory, env_factory=None, config=None, mutation=None,
                 learner=None, evaluator=None, backend="serial", device="cpu",
                 environment_device="cpu", tensor_env_factory=None, training_env_factory=None,
                 selection_hook=None, crossover=None, validation_sampler=None, archive_sampler=None,
                 callbacks=(), initial_agents=()):
        self.config = config or EvolutionConfig()
        self.agent_factory = agent_factory
        self.env_factory = env_factory
        self.training_env_factory = training_env_factory or env_factory
        self.mutation = mutation if mutation is not None else GaussianMutation()
        self.learner = learner
        self.evaluator = evaluator or (GymEvaluator(env_factory, self.config.max_episode_steps) if env_factory else None)
        self.backend, self.device, self.environment_device = backend, device, environment_device
        self.tensor_env_factory = tensor_env_factory
        self.selection_hook, self.crossover = selection_hook, crossover
        self.validation_sampler, self.archive_sampler = validation_sampler, archive_sampler
        self.callbacks = tuple(callbacks)
        self.initial_agents = tuple(initial_agents)
        self.population, self.archive, self.history, self.lineage = [], [], [], []
        self.completed, self.next_id, self.training_interactions, self.evaluation_interactions = 0, 0, 0, 0
        self.pre_learning_interactions, self.archive_interactions = 0, 0
        self.elapsed, self.stagnation = 0.0, 0
        self.best_seen = -np.inf if self.config.maximize else np.inf
        self.mutation_rng = np.random.default_rng(seed_for(self.config.seed, "mutation"))
        self.selection_rng = np.random.default_rng(seed_for(self.config.seed, "selection"))
        self.metadata = software_metadata()
        self.metadata.update({"config": self.config.as_dict(), "backend": backend, "policy_device": device,
                              "environment_device": environment_device, "learning": repr(learner),
                              "mutation": repr(self.mutation), "seeds": {k: seed_for(self.config.seed, k) for k in
                              ("initialization", "mutation", "selection", "training", "actions", "validation", "test", "archive")}})
        self.provenance = {key: describe(getattr(self, key)) for key in ("agent_factory", "env_factory", "training_env_factory",
            "tensor_env_factory", "evaluator", "mutation", "learner", "selection_hook", "crossover", "validation_sampler", "archive_sampler")}
        self.metadata["components"] = self.provenance
        self.metadata["execution_kind"] = ("native_tensor" if tensor_env_factory else
            "hybrid" if device != "cpu" else "cpu")
        self.metadata["archive_execution"] = "serial_cpu_reference"
        self._validate()

    def _validate(self):
        if self.evaluator is None:
            raise ValueError("Supply env_factory or a functional evaluator")
        if self.backend not in ("serial", "parallel", "batched"):
            raise ValueError("backend must be serial, parallel or batched")
        if self.config.learning_budget and self.learner is None:
            raise ValueError("learning_budget requires a learner")
        if self.learner is not None and (not self.config.learning_budget or not self.training_env_factory):
            raise ValueError("Learners require a positive per-individual learning_budget and training_env_factory")
        if len(self.initial_agents) > self.config.population_size:
            raise ValueError("Too many initial agents")
        if self.backend != "batched" and self.tensor_env_factory is not None:
            raise ValueError("tensor_env_factory requires backend='batched'")
        if self.environment_device != "cpu" and self.tensor_env_factory is None:
            raise ValueError("Python environments execute on CPU; a tensor_env_factory is required for an environment device")
        if self.backend == "batched" or self.device != "cpu":
            from .tensor import resolve_device
            resolve_device(self.device)
            resolve_device(self.environment_device)
        if self.backend == "batched":
            if type(self.evaluator) is not GymEvaluator:
                raise ValueError("Batched evaluation currently supports the built-in GymEvaluator")
            if self.tensor_env_factory and self.device != self.environment_device:
                raise ValueError("Native tensor environments currently require matching policy/environment devices")
            if self.evaluator.penalty or self.evaluator.aggregate is not np.mean:
                raise ValueError("Batched fitness supports mean return; use serial for custom aggregation/penalties")
        if self.backend == "parallel":
            try:
                pickle.dumps(self.evaluator)
            except Exception as error:
                raise ValueError("Parallel evaluators and factories must be pickleable top-level objects") from error
            if hasattr(self.evaluator, "evaluate_population"):
                raise ValueError("Shared population evaluation currently requires backend='serial'")
        if self.learner is not None and hasattr(self.learner, "env_copies"):
            if self.learner.env_copies != self.config.env_copies:
                raise ValueError("config.env_copies must match learner.env_copies")
        if self.learner is not None and hasattr(self.learner, "learn_population"):
            if not self.config.elite_learning:
                raise ValueError("Native batched learning requires elite_learning=True")
            if self.learner.device != self.device:
                raise ValueError("Learner device must match policy device")

    def _new(self, agent, generation, parents=(), mutation=None):
        individual = Individual(f"i{self.next_id:08d}", agent, generation, parents, mutation or {})
        self.next_id += 1
        self.lineage.append({"id": individual.id, "born": generation, "parents": parents, "mutation": mutation or {}})
        return individual

    def _factory_agent(self, coordinate):
        observation_space = action_space = None
        if self.env_factory:
            env = self.env_factory()
            try:
                observation_space, action_space = env.observation_space, env.action_space
            finally:
                env.close()
        return self.agent_factory(np.random.default_rng(seed_for(self.config.seed, "initialization", coordinate)),
                                  observation_space, action_space, self.device)

    def _initialize(self):
        for index in range(self.config.population_size):
            agent = self._factory_agent(index)
            if any(agent is individual.agent for individual in self.population):
                raise ValueError("agent_factory returned the same mutable agent for multiple individuals")
            if index < len(self.initial_agents):
                initial = self.initial_agents[index]
                if type(initial) is not type(agent) or initial.genome.vector().shape != agent.genome.vector().shape:
                    raise ValueError("Initial checkpoint agent is incompatible with the configured factory")
                if hasattr(initial, "layout") and initial.layout() != agent.layout():
                    raise ValueError("Initial neural architecture does not match")
                if hasattr(initial, "algorithm_name") and (initial.algorithm_name != agent.algorithm_name or
                        initial.observation_space != agent.observation_space or initial.action_space != agent.action_space or
                        describe(initial.kwargs) != describe(agent.kwargs)):
                    raise ValueError("Initial SB3 policy configuration does not match")
                agent.genome.assign(initial.genome.vector())
            if self.learner:
                self.learner.validate(agent, self.training_env_factory, self.backend)
            elif self.backend != "batched" and self.device != "cpu" and type(agent).__module__ == "geneticevo.agents":
                raise ValueError("NumPy policies use CPU; select backend='batched' for tensor policy execution")
            self.population.append(self._new(agent, 0))
        if self.backend == "batched":
            from .tensor import BatchedPolicy
            BatchedPolicy([x.agent for x in self.population], self.device)

    def _evaluate(self, agents, scenarios, generation):
        action_seed = seed_for(self.config.seed, "validation", generation)
        if self.backend == "batched":
            from .tensor import evaluate_batched
            return evaluate_batched(agents, scenarios, self.env_factory, self.tensor_env_factory,
                                    self.device, self.config.env_copies, self.config.max_episode_steps)
        if hasattr(self.evaluator, "evaluate_population"):
            return self.evaluator.evaluate_population(agents, scenarios, action_seed)
        return evaluate_population(self.evaluator, agents, scenarios, action_seed,
                                   self.backend, self.config.workers)

    def _learn(self, generation):
        agents, indices = [], []
        for index, individual in enumerate(self.population):
            agent = individual.agent
            if self.config.inheritance == "baldwinian":
                agent = self.learner.clone(agent, training_state=False) if self.learner else agent.clone(False)
            agents.append(agent)
            if self.learner and (self.config.elite_learning or individual.born == generation):
                indices.append(index)
        reports = []
        if self.learner:
            if hasattr(self.learner, "learn_population"):
                if len(indices) != len(agents):
                    raise ValueError("Native batched learning requires elite_learning=True")
                reports = self.learner.learn_population(agents, self.training_env_factory, self.tensor_env_factory,
                                                       self.config.learning_budget, self.config.seed, generation)
            else:
                for index in indices:
                    numeric_id = int(self.population[index].id[1:])
                    report = self.learner.learn(agents[index], self.training_env_factory, self.config.learning_budget,
                                                seed_for(self.config.seed, "training", generation, numeric_id),
                                                seed_for(self.config.seed, "actions", generation, numeric_id))
                    reports.append(report)
            self.training_interactions += sum(r.interactions for r in reports)
        return agents, reports

    def _update_archive(self, agents, evaluations, generation):
        order = sorted(range(len(agents)), key=lambda i: evaluations[i].fitness, reverse=self.config.maximize)
        candidates = [Individual(self.population[i].id, agents[i].clone(False), generation,
                                 self.population[i].parents, copy.deepcopy(self.population[i].mutation))
                      for i in order[:self.config.archive_size]]
        candidates += copy.deepcopy(self.archive)
        scenarios = scenarios_for(self.config.seed, "archive", self.config.evaluation_episodes,
                                  sampler=self.archive_sampler)
        # Every candidate is compared on these same fixed reference scenarios.
        # Shared evaluators use their fixed partner pool for archive comparisons.
        scores = [self.evaluator.evaluate(x.agent, scenarios, seed_for(self.config.seed, "archive")) for x in candidates]
        cost = sum(x.interactions for x in scores)
        self.archive_interactions += cost
        self.evaluation_interactions += cost
        for candidate, score in zip(candidates, scores):
            candidate.evaluation = score
        candidates.sort(key=lambda x: x.evaluation.fitness, reverse=self.config.maximize)
        # Identical genomes need not occupy the entire archive.
        self.archive = []
        for candidate in candidates:
            if not any(np.array_equal(candidate.agent.genome.vector(), x.agent.genome.vector()) for x in self.archive):
                self.archive.append(candidate)
            if len(self.archive) == self.config.archive_size:
                break

    def _breed(self, generation):
        ranked = sorted(self.population, key=lambda x: x.evaluation.fitness, reverse=self.config.maximize)
        population = [copy.deepcopy(x) for x in ranked[:self.config.elite_count]]
        for individual in population:
            individual.evaluation = None
        children_count = self.config.population_size - self.config.immigrant_count
        while len(population) < children_count:
            parent = self.selection_hook(self.population, self.selection_rng, self.config.maximize) if self.selection_hook else select(
                self.population, self.selection_rng, method=self.config.selection,
                maximize=self.config.maximize, tournament_size=self.config.tournament_size)
            if not any(parent is x for x in self.population):
                raise ValueError("selection_hook must return a member of the population")
            child = self.learner.clone(parent.agent, False) if self.learner else parent.agent.clone(False)
            parents = (parent.id,)
            if self.crossover:
                other = select(self.population, self.selection_rng, method=self.config.selection,
                               maximize=self.config.maximize, tournament_size=self.config.tournament_size)
                child.genome.assign(self.crossover(parent.agent.genome.clone(), other.agent.genome.clone(), self.mutation_rng).vector())
                parents += (other.id,)
            metadata = self.mutation(child.genome, self.mutation_rng)
            population.append(self._new(child, generation, parents, metadata))
        while len(population) < self.config.population_size:
            population.append(self._new(self._factory_agent(self.next_id), generation, mutation={"kind": "immigrant"}))
        self.population = population

    def _stop(self):
        cfg = self.config
        if cfg.target_fitness is not None and ((self.best_seen >= cfg.target_fitness) if cfg.maximize else (self.best_seen <= cfg.target_fitness)):
            return "target_fitness"
        if cfg.stagnation_generations is not None and self.stagnation >= cfg.stagnation_generations:
            return "stagnation"
        if cfg.max_interactions is not None and self.training_interactions + self.evaluation_interactions >= cfg.max_interactions:
            return "interaction_budget"
        if cfg.time_limit_seconds is not None and self.elapsed >= cfg.time_limit_seconds:
            return "time_limit"
        if self.completed >= cfg.generations:
            return "generations"
        return None

    def _state(self):
        names = ("population", "archive", "history", "lineage", "completed", "next_id", "training_interactions",
                 "evaluation_interactions", "pre_learning_interactions", "archive_interactions", "elapsed",
                 "stagnation", "best_seen", "mutation_rng", "selection_rng")
        return {"config": self.config.as_dict(), "provenance": self.provenance, "execution": (self.backend, self.device, self.environment_device),
                "learner": repr(self.learner), "mutation": repr(self.mutation),
                **{key: getattr(self, key) for key in names}}

    def checkpoint(self, path):
        save_checkpoint(path, self._state())

    def restore(self, path):
        state = load_checkpoint(path)
        old, new = state["config"].copy(), self.config.as_dict()
        for key in ("generations", "output_dir", "checkpoint_every", "time_limit_seconds", "max_interactions",
                    "target_fitness", "stagnation_generations"):
            old.pop(key, None)
            new.pop(key, None)
        if old != new or state["execution"] != (self.backend, self.device, self.environment_device):
            raise ValueError("Resume configuration is incompatible (only run length, output and stopping caps may change)")
        if state["learner"] != repr(self.learner) or state["mutation"] != repr(self.mutation):
            raise ValueError("Resume learner or mutation configuration differs")
        if state["provenance"] != self.provenance:
            raise ValueError("Resume factories, evaluators, samplers or hooks differ")
        for key, value in state.items():
            if key not in ("config", "provenance", "execution", "learner", "mutation"):
                setattr(self, key, value)
        if self.completed < 1:
            raise ValueError("Checkpoint must be at an evaluated generation boundary")

    def run(self, resume_from=None):
        wall_started = time.perf_counter()
        if resume_from:
            self.restore(resume_from)
        elif self.population:
            raise ValueError("Create a new Evolution or resume a generation-boundary checkpoint")
        else:
            self._initialize()
        elapsed_before_run = self.elapsed
        cfg = self.config
        output = Path(cfg.output_dir) if cfg.output_dir else None
        if output:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "metadata.json", self.metadata)
            # Rewrite on resume so metrics cannot be silently duplicated.
            (output / "metrics.jsonl").write_text("".join(json.dumps(x, default=json_default) + "\n" for x in self.history), encoding="utf-8")
        reason = self._stop() if self.completed else None
        while reason is None:
            started = time.perf_counter()
            generation = self.completed
            timings = {}
            part = time.perf_counter()
            if generation:
                self._breed(generation)
            timings["reproduction_seconds"] = time.perf_counter() - part
            scenarios = scenarios_for(cfg.seed, "validation", cfg.evaluation_episodes, generation, self.validation_sampler)
            before = None
            part = time.perf_counter()
            if self.learner and cfg.measure_pre_learning:
                before = self._evaluate([x.agent for x in self.population], scenarios, generation)
                cost = sum(x.interactions for x in before)
                self.evaluation_interactions += cost
                self.pre_learning_interactions += cost
            timings["pre_evaluation_seconds"] = time.perf_counter() - part
            part = time.perf_counter()
            agents, reports = self._learn(generation)
            timings["learning_seconds"] = time.perf_counter() - part
            part = time.perf_counter()
            evaluations = self._evaluate(agents, scenarios, generation)
            self.evaluation_interactions += sum(x.interactions for x in evaluations)
            for individual, evaluation in zip(self.population, evaluations):
                individual.evaluation = evaluation
            timings["evaluation_seconds"] = time.perf_counter() - part
            part = time.perf_counter()
            self._update_archive(agents, evaluations, generation)
            timings["archive_seconds"] = time.perf_counter() - part
            score = self.archive[0].evaluation.fitness
            better = score > self.best_seen if cfg.maximize else score < self.best_seen
            self.best_seen, self.stagnation = (score, 0) if better else (self.best_seen, self.stagnation + 1)
            fitness = np.array([x.fitness for x in evaluations])
            lengths = [length for x in evaluations for length in x.lengths]
            metrics = {key: float(np.mean([x.metrics[key] for x in evaluations if key in x.metrics]))
                       for key in {k for x in evaluations for k in x.metrics}}
            elapsed = time.perf_counter() - started
            self.elapsed = elapsed_before_run + time.perf_counter() - wall_started
            self.completed += 1
            record = {"generation": generation, "population_size": len(self.population),
                      "best": float(fitness.max() if cfg.maximize else fitness.min()), "mean": float(fitness.mean()),
                      "median": float(np.median(fitness)), "std": float(fitness.std()),
                      "quantiles": np.quantile(fitness, [0.1, 0.25, 0.75, 0.9]).tolist(), "archive_best": score,
                      "training_interactions": self.training_interactions, "evaluation_interactions": self.evaluation_interactions,
                      "pre_learning_interactions": self.pre_learning_interactions, "archive_interactions": self.archive_interactions,
                      "learning_updates": sum(x.updates for x in reports),
                      "learning_requested_per_individual": cfg.learning_budget,
                      "learning_actual_per_individual": [x.interactions for x in reports],
                      "mean_episode_length": float(np.mean(lengths)) if lengths else None,
                      "task_metrics": metrics, "timings": timings, "generation_seconds": elapsed, "total_seconds": self.elapsed,
                      "interactions_per_second": (self.training_interactions + self.evaluation_interactions) / max(self.elapsed, 1e-9),
                      "elites": cfg.elite_count if generation else 0, "immigrants": cfg.immigrant_count if generation else 0,
                      "mutation": repr(self.mutation), "world_steps": sum(x.world_steps for x in evaluations)}
            if before:
                record["pre_learning_mean"] = float(np.mean([x.fitness for x in before]))
                record["learning_gain"] = record["mean"] - record["pre_learning_mean"]
            if reports:
                record["learner_metrics"] = {key: float(np.mean([x.metrics[key] for x in reports if key in x.metrics]))
                                              for key in {k for x in reports for k in x.metrics}}
            if cfg.diversity_pairs:
                rng = np.random.default_rng(seed_for(cfg.seed, "diversity", generation))
                pairs = rng.integers(0, len(agents), (cfg.diversity_pairs, 2))
                record["sampled_parameter_distance"] = float(np.mean([np.linalg.norm(agents[i].genome.vector() - agents[j].genome.vector()) for i, j in pairs]))
            self.history.append(record)
            reason = self._stop()
            if output:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                write_json(output / "lineage.json", self.lineage)
                if self.completed % cfg.checkpoint_every == 0 or reason:
                    self.checkpoint(output / "latest.checkpoint")
            for callback in self.callbacks:
                callback(copy.deepcopy(record))
        self.elapsed = elapsed_before_run + time.perf_counter() - wall_started
        self.metadata["wall_seconds"] = self.elapsed
        self.metadata["stop_reason"] = reason
        if output:
            write_json(output / "metadata.json", self.metadata)
        return Result(self.archive[0].agent.clone(False), self.archive[0].evaluation.fitness,
                      copy.deepcopy(self.population), copy.deepcopy(self.archive), copy.deepcopy(self.history),
                      copy.deepcopy(self.lineage), copy.deepcopy(self.metadata), reason)
