"""Evolutionary experiments. Optional learners and tensor modules import lazily."""
from .core import EvolutionConfig, Evaluation, Scenario, VectorGenome, LearnReport
from .evolution import GaussianMutation
from .runner import Evolution
from .agents import load_agent
from .growth import NeuralGrowth
from .graph import GraphAgent, GraphAgentFactory, GraphGrowth
from .rust_ppo import RustNativePPO

__all__ = ["Evolution", "EvolutionConfig", "GaussianMutation", "Evaluation", "Scenario",
           "VectorGenome", "LearnReport", "load_agent", "NeuralGrowth", "GraphAgent", "GraphAgentFactory", "GraphGrowth",
           "RustNativePPO"]
__version__ = "0.1.0a8"
