from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.jin_hilla_scenario_env import JinHillaScenarioEnv


class JinHillaScenarioGymEnv(gym.Env):
    """Gymnasium wrapper for JinHillaScenarioEnv adhering to SB3 standards."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 7) -> None:
        super().__init__()
        self._env = JinHillaScenarioEnv(seed=seed)
        self.action_space = spaces.Discrete(self._env.action_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._env.observation_size,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._env = JinHillaScenarioEnv(seed=seed)
        obs = self._env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return np.array(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info
