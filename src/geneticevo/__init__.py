"""Evolutionary experiments. Optional learners and tensor modules import lazily."""
from .core import EvolutionConfig, Evaluation, Scenario, VectorGenome, LearnReport
from .evolution import GaussianMutation
from .runner import Evolution
from .agents import load_agent
from .growth import NeuralGrowth

__all__ = ["Evolution", "EvolutionConfig", "GaussianMutation", "Evaluation", "Scenario",
           "VectorGenome", "LearnReport", "load_agent", "NeuralGrowth"]
__version__ = "0.1.0a5"
