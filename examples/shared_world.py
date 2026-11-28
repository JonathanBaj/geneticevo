"""Evolve a policy and its message head in a shared world."""
import argparse
from functools import partial
import numpy as np

from geneticevo import Evolution, EvolutionConfig
from geneticevo.agents import MLPAgentFactory
from geneticevo.shared import (SharedEvaluator, SharedFocalEnv, SignalWorld, SignalPartner,
    MessageChannel, CommunicatingAgentFactory, observed_signal_message, CommunicationAction)
from geneticevo.core import scenarios_for


class QuietPartner(SignalPartner):
    def act(self, observation, rng, deterministic=True):
        return CommunicationAction(super().act(observation, rng, deterministic), np.zeros(1, np.float32))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    channel = MessageChannel(dimension=1, cost=0.01)
    env_factory = partial(SharedFocalEnv, channel=channel, fixed_message=observed_signal_message)
    evaluator = SharedEvaluator(SignalWorld, (SignalPartner(0), SignalPartner(1)), channel,
                                observed_signal_message, reward_mode="individual")
    factory = CommunicatingAgentFactory(MLPAgentFactory((8,)))
    result = Evolution(env_factory=env_factory, agent_factory=factory, evaluator=evaluator,
        config=EvolutionConfig(population_size=4, generations=2, elite_count=1,
            evaluation_episodes=2, output_dir="results/shared")).run()
    print(f"Shared fitness={result.best_fitness:.4f}; evaluated world steps={sum(r['world_steps'] for r in result.history)}")
    held_out = SharedEvaluator(SignalWorld, (QuietPartner(),), channel, observed_signal_message)
    test = result.evaluate(held_out, scenarios_for(2026, "test", 8))
    print(f"Held-out quiet-partner test={test.fitness:.4f}; all-agent transitions={test.interactions}")

