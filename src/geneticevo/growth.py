"""Optional, bounded MLP width/depth plasticity and architecture mutations.

This is a capacity-growth model, not a biological model of brain development.
No optional dependencies are imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NeuralGrowth:
    """Edit width or depth at a completed learning-update boundary.

    check_every counts actual environment interactions per individual, rounded
    up to a completed PPO rollout. None for value_loss_threshold means scheduled
    growth; a threshold requires mean rollout-update value loss at least that
    large. Architecture is inherited independently of learned-weight inheritance.
    mutation_probability applies to each hidden width at birth; the separate
    layer_mutation_probability enables one depth edit. Lifetime shrink/add-layer/
    remove-layer probabilities reserve portions of the event distribution; the
    remainder widens a layer. Only eligible events participate at each check.
    New probabilities default to zero for compatibility with width-only runs.
    """
    check_every: int = 256
    neurons_per_growth: int = 4
    max_hidden_size: int = 128
    min_hidden_size: int = 1
    value_loss_threshold: float | None = None
    mutation_probability: float = 0.1
    mutation_step: int = 2
    shrink_probability: float = 0.0
    layer_add_probability: float = 0.0
    layer_remove_probability: float = 0.0
    layer_mutation_probability: float = 0.0
    min_hidden_layers: int = 1
    max_hidden_layers: int = 8
    new_layer_size: int | None = None

    def __post_init__(self):
        for name in ('check_every', 'neurons_per_growth', 'max_hidden_size',
                     'min_hidden_size', 'mutation_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.min_hidden_size > self.max_hidden_size:
            raise ValueError('min_hidden_size must not exceed max_hidden_size')
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError('mutation_probability must be in [0, 1]')
        if self.value_loss_threshold is not None and (
                not math.isfinite(self.value_loss_threshold) or self.value_loss_threshold < 0):
            raise ValueError('value_loss_threshold must be finite and nonnegative')
        for name in ('shrink_probability', 'layer_add_probability', 'layer_remove_probability',
                     'layer_mutation_probability'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.shrink_probability + self.layer_add_probability + self.layer_remove_probability > 1:
            raise ValueError('Lifetime shrink/layer probabilities must sum to at most 1')
        for name in ('min_hidden_layers', 'max_hidden_layers'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.min_hidden_layers > self.max_hidden_layers:
            raise ValueError('min_hidden_layers must not exceed max_hidden_layers')
        if self.new_layer_size is not None and (isinstance(self.new_layer_size, bool) or
                not isinstance(self.new_layer_size, int) or
                not self.min_hidden_size <= self.new_layer_size <= self.max_hidden_size):
            raise ValueError('new_layer_size must be an integer within width bounds')

    @property
    def dynamic_architecture(self):
        return bool(self.shrink_probability or self.layer_add_probability or
                    self.layer_remove_probability or self.layer_mutation_probability)

    def layer_size(self, agent, index):
        if self.new_layer_size is not None:
            return self.new_layer_size
        return (agent.hidden_sizes[min(index, len(agent.hidden_sizes) - 1)]
                if agent.hidden_sizes else self.min_hidden_size)

    def validate_sizes(self, sizes):
        if not self.min_hidden_layers <= len(sizes) <= self.max_hidden_layers or any(isinstance(x, bool) or not isinstance(x, int) or
                            not self.min_hidden_size <= x <= self.max_hidden_size for x in sizes):
            raise ValueError('NeuralGrowth requires integer hidden sizes and depth within its min/max bounds')

    def after_update(self, agent, interactions, value_loss, rng):
        """Called by NativePPO after an update; custom learners may also use it.

        Training counters belong to the individual and reset on fresh clones.
        Return True if the architecture changed. Evaluation never calls this.
        """
        self.validate_sizes(agent.hidden_sizes)
        if isinstance(interactions, bool) or not isinstance(interactions, int) or interactions < 1:
            raise ValueError('Growth updates require a positive integer interaction count')
        if not hasattr(agent, 'training_state'):
            agent.training_state = {}
        state = agent.training_state.setdefault('neural_growth',
                    {'interactions': 0, 'last_check': 0, 'events': 0})
        state['interactions'] += interactions
        if state['interactions'] - state['last_check'] < self.check_every:
            return False
        state['last_check'] = state['interactions']
        if self.value_loss_threshold is not None and (
                not math.isfinite(value_loss) or value_loss < self.value_loss_threshold):
            return False
        grow = [i for i, width in enumerate(agent.hidden_sizes) if width < self.max_hidden_size]
        shrink = [i for i, width in enumerate(agent.hidden_sizes) if width > self.min_hidden_size]
        events = [('grow', 1 - self.shrink_probability - self.layer_add_probability - self.layer_remove_probability, bool(grow)),
                  ('shrink', self.shrink_probability, bool(shrink)),
                  ('add_layer', self.layer_add_probability, len(agent.hidden_sizes) < self.max_hidden_layers),
                  ('remove_layer', self.layer_remove_probability, len(agent.hidden_sizes) > self.min_hidden_layers)]
        events = [(name, probability) for name, probability, eligible in events if eligible and probability > 0]
        if not events:
            return False
        if len(events) == 1:
            event = events[0][0]
        else:
            probabilities = np.array([p for _, p in events])
            event = events[int(rng.choice(len(events), p=probabilities / probabilities.sum()))][0]
        if event == 'add_layer':
            index = int(rng.integers(len(agent.hidden_sizes) + 1))
            agent.insert_hidden_layer(index, self.layer_size(agent, index), rng)
        elif event == 'remove_layer':
            agent.remove_hidden_layer(int(rng.integers(len(agent.hidden_sizes))), rng)
        else:
            sizes = list(agent.hidden_sizes)
            layer = int(rng.choice(grow if event == 'grow' else shrink))
            sizes[layer] = int(np.clip(sizes[layer] + (1 if event == 'grow' else -1) * self.neurons_per_growth,
                                       self.min_hidden_size, self.max_hidden_size))
            agent.resize_hidden_sizes(tuple(sizes), rng)
        state['events'] += 1
        state[event] = state.get(event, 0) + 1
        return True

    def mutate(self, agent, rng):
        """Mutate a child's inherited widths/depth, retaining surviving genes."""
        self.validate_sizes(agent.hidden_sizes)
        before = agent.hidden_sizes
        before_ids = agent.layer_ids
        sizes = list(before)
        for i, width in enumerate(sizes):
            if rng.random() < self.mutation_probability:
                delta = int(rng.integers(1, self.mutation_step + 1)) * int(rng.choice((-1, 1)))
                sizes[i] = int(np.clip(width + delta, self.min_hidden_size, self.max_hidden_size))
        agent.resize_hidden_sizes(tuple(sizes), rng)
        if self.layer_mutation_probability and rng.random() < self.layer_mutation_probability:
            operations = []
            if len(agent.hidden_sizes) < self.max_hidden_layers:
                operations.append('add')
            if len(agent.hidden_sizes) > self.min_hidden_layers:
                operations.append('remove')
            if operations:
                if rng.choice(operations) == 'add':
                    index = int(rng.integers(len(agent.hidden_sizes) + 1))
                    agent.insert_hidden_layer(index, self.layer_size(agent, index), rng)
                else:
                    agent.remove_hidden_layer(int(rng.integers(len(agent.hidden_sizes))), rng)
        return {'kind': 'neural_size', 'before': list(before), 'after': list(agent.hidden_sizes),
                'layer_ids_before': list(before_ids), 'layer_ids_after': list(agent.layer_ids)}
