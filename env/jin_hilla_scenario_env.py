from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from env.jin_hilla_training_env import Action, JinHillaTrainingEnv


@dataclass(frozen=True)
class ScenarioConfig:
    altar_interval: int = 45
    hazard_interval: int = 12
    max_steps: int = 360
    lane_count: int = 7


class JinHillaScenarioEnv:
    """Seeded training scenarios layered over the M8 rules environment.

    Hazard lanes are observed before an action. A web hit occurs only when the
    player remains in that lane, so survival and altar choices are learnable.
    """

    action_size = len(Action)

    def __init__(self, config: ScenarioConfig = ScenarioConfig()) -> None:
        self.config = config
        self.core = JinHillaTrainingEnv(
            lane_count=config.lane_count,
            max_steps=config.max_steps,
        )
        self.rng = random.Random()
        self.steps = 0
        self.next_hazard_lane: int | None = None
        self.altar_spawns = 0
        self.cleanse_count = 0
        self.web_hits = 0

    @property
    def observation_size(self) -> int:
        return self.core.observation_size + 2

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        self.rng.seed(seed)
        self.steps = 0
        self.altar_spawns = self.cleanse_count = self.web_hits = 0
        self.next_hazard_lane = self._draw_hazard_lane()
        _, info = self.core.reset(options={"player_lane": self.config.lane_count // 2})
        return self.observation(), {**info, **self.metrics()}

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        state = self.core._require_state()
        old_red = state.souls.red
        hazard_now = self.next_hazard_lane
        _, reward, terminated, truncated, _ = self.core.step(action)
        self.steps += 1

        if hazard_now is not None and state.player_lane == hazard_now and not terminated:
            self.core.apply_web_hit()
            self.web_hits += 1
            reward -= self.core.reward_config.web_hit
            terminated = state.terminated
            if terminated:
                reward += self.core.reward_config.defeat
        if state.souls.red < old_red:
            self.cleanse_count += old_red - state.souls.red

        if self.steps % self.config.altar_interval == 0 and not state.altar_present:
            self.core.spawn_altar(self.rng.randrange(self.config.lane_count))
            self.altar_spawns += 1
        self.next_hazard_lane = self._draw_hazard_lane()
        truncated = truncated or self.steps >= self.config.max_steps
        return self.observation(), reward, terminated, truncated, {**self.core.info(), **self.metrics()}

    def observation(self) -> list[float]:
        hazard = -1.0 if self.next_hazard_lane is None else self.next_hazard_lane / (self.config.lane_count - 1)
        until_hazard = self.config.hazard_interval - (self.steps % self.config.hazard_interval)
        return self.core.observation() + [hazard, until_hazard / self.config.hazard_interval]

    def metrics(self) -> dict[str, int]:
        return {"scenario_steps": self.steps, "web_hits": self.web_hits, "cleanse_count": self.cleanse_count, "altar_spawns": self.altar_spawns}

    def _draw_hazard_lane(self) -> int | None:
        if self.steps % self.config.hazard_interval == 0:
            return self.rng.randrange(self.config.lane_count)
        return None
