"""Selection and variation operate on genomes, independently of learning."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import Genome, Individual


@dataclass(frozen=True)
class GaussianMutation:
    """Probability is per scalar parameter; std is the Gaussian magnitude."""
    parameter_probability: float = 0.05
    std: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.parameter_probability <= 1 or not np.isfinite(self.std) or self.std < 0:
            raise ValueError("Mutation needs probability in [0, 1] and finite nonnegative std")

    def __call__(self, genome: Genome, rng: np.random.Generator) -> dict:
        values = genome.vector()
        mask = rng.random(values.size) < self.parameter_probability
        values[mask] += rng.normal(0, self.std, int(mask.sum()))
        genome.assign(values)
        return {"kind": "gaussian", "changed": int(mask.sum()),
                "parameter_probability": self.parameter_probability, "std": self.std}


def select(population: list[Individual], rng: np.random.Generator, *,
           method: str, maximize: bool, tournament_size: int) -> Individual:
    ranked = sorted(population, key=lambda x: x.evaluation.fitness, reverse=maximize)
    if method == "tournament":
        indices = rng.integers(0, len(ranked), size=tournament_size)
        return ranked[int(indices.min())]
    weights = np.arange(len(ranked), 0, -1, dtype=float)
    return ranked[int(rng.choice(len(ranked), p=weights / weights.sum()))]

