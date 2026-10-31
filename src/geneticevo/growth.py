"""Optional, bounded MLP width growth and heritable architecture mutations.

This is a capacity-growth model, not a biological model of brain development.
No optional dependencies are imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NeuralGrowth:
    """Grow one eligible hidden layer at a completed learning-update boundary.

    check_every counts actual environment interactions per individual, rounded
    up to a completed PPO rollout. None for value_loss_threshold means scheduled
    growth; a threshold requires mean rollout-update value loss at least that
    large. Widths are inherited independently of learned-weight inheritance.
    Mutation probability applies separately to each hidden layer at birth.
    """
    check_every: int = 256
    neurons_per_growth: int = 4
    max_hidden_size: int = 128
    min_hidden_size: int = 1
    value_loss_threshold: float | None = None
    mutation_probability: float = 0.1
    mutation_step: int = 2

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

    def validate_sizes(self, sizes):
        if not sizes or any(isinstance(x, bool) or not isinstance(x, int) or
                            not self.min_hidden_size <= x <= self.max_hidden_size for x in sizes):
            raise ValueError('NeuralGrowth requires nonempty integer hidden sizes within its min/max bounds')

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
        eligible = [i for i, width in enumerate(agent.hidden_sizes) if width < self.max_hidden_size]
        if not eligible:
            return False
        sizes = list(agent.hidden_sizes)
        layer = int(rng.choice(eligible))
        sizes[layer] = min(self.max_hidden_size, sizes[layer] + self.neurons_per_growth)
        agent.resize_hidden_sizes(tuple(sizes), rng)
        state['events'] += 1
        return True

    def mutate(self, agent, rng):
        """Mutate a fresh child's inherited widths, retaining overlapping weights."""
        self.validate_sizes(agent.hidden_sizes)
        before = agent.hidden_sizes
        sizes = list(before)
        for i, width in enumerate(sizes):
            if rng.random() < self.mutation_probability:
                delta = int(rng.integers(1, self.mutation_step + 1)) * int(rng.choice((-1, 1)))
                sizes[i] = int(np.clip(width + delta, self.min_hidden_size, self.max_hidden_size))
        agent.resize_hidden_sizes(tuple(sizes), rng)
        return {'kind': 'neural_size', 'before': list(before), 'after': list(agent.hidden_sizes)}
