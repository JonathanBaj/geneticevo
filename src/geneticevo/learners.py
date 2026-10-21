"""Algorithm-neutral conveniences for external learner implementations."""
from __future__ import annotations

import copy
from .core import Learner, LearnReport


class GenomeLearner:
    """Base helpers for learners storing non-genetic state on each agent.

    Implement validate, prepare, and learn. Never store one individual's
    optimizer, replay buffer, normalization or counters on this shared object.
    """
    def export(self, agent):
        return agent.genome.clone()

    def clone(self, agent, training_state=False):
        return agent.clone(training_state)

    def state_dict(self, agent):
        return copy.deepcopy(getattr(agent, "training_state", {}))

    def load_state_dict(self, agent, state):
        agent.training_state = copy.deepcopy(state)
