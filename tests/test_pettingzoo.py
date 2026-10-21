import unittest


class PettingZooTests(unittest.TestCase):
    def test_official_parallel_api_and_adapter(self):
        try:
            from pettingzoo.test import parallel_api_test
            from geneticevo.integrations.pettingzoo import SignalParallelEnv
        except ImportError:
            self.skipTest("PettingZoo extra unavailable")
        from geneticevo.shared import SharedEvaluator, SignalPartner, MessageChannel, observed_signal_message
        from geneticevo import Scenario
        parallel_api_test(SignalParallelEnv(horizon=4), num_cycles=8)
        evaluator = SharedEvaluator(SignalParallelEnv, (SignalPartner(),), MessageChannel(), observed_signal_message)
        evaluation = evaluator.evaluate(SignalPartner(), [Scenario(3)])
        self.assertEqual(evaluation.interactions, 8)


if __name__ == "__main__":
    unittest.main()
