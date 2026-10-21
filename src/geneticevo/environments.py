"""Small Gymnasium reference environment; tensor equivalent is optional."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Reach1D(gym.Env):
    """Move toward zero. Fixed episode horizon is an external truncation.

    reset options: position (exact), radius (initial distribution).
    No intrinsic terminal state; observation contains position only.
    """
    metadata = {"render_modes": []}

    def __init__(self, horizon=32, continuous=False):
        self.horizon = horizon
        self.continuous = continuous
        self.observation_space = spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32) if continuous else spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.position = np.float32(options.get("position", self.np_random.uniform(-options.get("radius", 1), options.get("radius", 1))))
        self.t = 0
        return np.array([self.position], np.float32), {}

    def step(self, action):
        move = float(np.asarray(action).reshape(-1)[0]) if self.continuous else int(action) - 1
        self.position = np.float32(np.clip(self.position + 0.1 * move, -10, 10))
        self.t += 1
        return np.array([self.position], np.float32), -float(self.position**2), False, self.t >= self.horizon, {"success": float(abs(self.position) < 0.1)}

