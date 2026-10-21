"""A non-neural, discrete genome with its own mutation: python examples/custom_genome.py"""
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from geneticevo import Evolution, EvolutionConfig
from geneticevo.evaluation import ObjectiveEvaluator


@dataclass
class BitGenome:
    bits: np.ndarray

    def clone(self):
        return copy.deepcopy(self)

    def vector(self):
        return self.bits.astype(np.float32).copy()

    def assign(self, values):
        values = np.asarray(values)
        if values.shape != self.bits.shape or not np.isin(values, [0, 1]).all():
            raise ValueError("Bit genomes require compatible binary values")
        self.bits[:] = values.astype(bool)


@dataclass
class BitAgent:
    genome: BitGenome

    def clone(self, training_state=False):
        return copy.deepcopy(self)

    def save(self, path):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "bits.npz", bits=self.genome.bits)


@dataclass(frozen=True)
class BitFactory:
    dimension: int = 16

    def __call__(self, rng, observation_space=None, action_space=None, device="cpu"):
        return BitAgent(BitGenome(rng.integers(0, 2, self.dimension).astype(bool)))


@dataclass(frozen=True)
class BitFlip:
    probability: float = 0.1

    def __call__(self, genome, rng):
        mask = rng.random(genome.bits.size) < self.probability
        genome.bits ^= mask
        return {"kind": "bit_flip", "changed": int(mask.sum())}


def count_bits(values, scenario):
    return float(values.sum())


if __name__ == "__main__":
    result = Evolution(agent_factory=BitFactory(), evaluator=ObjectiveEvaluator(count_bits), mutation=BitFlip(),
        config=EvolutionConfig(population_size=16, generations=8, elite_count=2, evaluation_episodes=1)).run()
    print(f"Custom genome: {result.best_fitness:g}/16 bits")
