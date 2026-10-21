"""Optional real PettingZoo ParallelEnv for the bundled shared-world example."""
from pettingzoo import ParallelEnv

from ..shared import SignalWorld


class SignalParallelEnv(SignalWorld, ParallelEnv):
    render_mode = None

