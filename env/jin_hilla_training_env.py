from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from env.jin_hilla_environment_core import JinHillaState, ScytheCycle, ScythePhase, SoulState


class Action(IntEnum):
    LEFT = 0
    STAY = 1
    RIGHT = 2
    HARVEST = 3


@dataclass(frozen=True)
class RewardConfig:
    survive: float = 0.05
    cleanse_per_soul: float = 1.0
    web_hit: float = -1.0
    defeat: float = -25.0
    unsafe_scythe: float = -0.25


class JinHillaTrainingEnv:
    """Lightweight RL environment backed by the M8 Jin Hilla rule engine.

    `step()` follows the common `(observation, reward, terminated, truncated, info)`
    convention without requiring Gymnasium at import time.
    """

    observation_size = 9
    action_size = len(Action)

    def __init__(
        self,
        *,
        lane_count: int = 7,
        max_steps: int = 900,
        scythe_ticks: tuple[int, int, int] = (180, 30, 45),
        reward_config: RewardConfig = RewardConfig(),
    ) -> None:
        if lane_count < 2:
            raise ValueError("lane_count must be at least 2")
        self.lane_count = lane_count
        self.max_steps = max_steps
        self.scythe_ticks = scythe_ticks
        self.reward_config = reward_config
        self.state: JinHillaState | None = None
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[list[float], dict[str, Any]]:
        del seed
        options = options or {}
        green = int(options.get("green_skulls", 5))
        red = int(options.get("red_skulls", 0))
        player_lane = int(options.get("player_lane", self.lane_count // 2))
        player_lane = max(0, min(self.lane_count - 1, player_lane))
        countdown, warning, spread_art = self.scythe_ticks
        self.state = JinHillaState(
            souls=SoulState(green=green, red=red),
            scythe=ScytheCycle(countdown, warning, spread_art),
            player_lane=player_lane,
        )
        self.steps = 0
        return self.observation(), self.info()

    def spawn_altar(self, lane: int) -> None:
        state = self._require_state()
        state.altar_present = True
        state.altar_lane = max(0, min(self.lane_count - 1, lane))

    def apply_web_hit(self) -> None:
        self._require_state().apply_web_hit()

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        try:
            action = Action(action)
        except ValueError as exc:
            raise ValueError(f"unknown action: {action}") from exc
        if state.terminated:
            return self.observation(), 0.0, True, False, self.info()

        reward = self.reward_config.survive
        if action is Action.LEFT:
            state.player_lane = max(0, state.player_lane - 1)
        elif action is Action.RIGHT:
            state.player_lane = min(self.lane_count - 1, state.player_lane + 1)
        elif action is Action.HARVEST:
            cleaned = state.harvest_altar(presses=10)
            reward += cleaned * self.reward_config.cleanse_per_soul

        phase_change = state.tick()
        if phase_change is ScythePhase.SPREAD_ART and state.altar_present:
            reward += self.reward_config.unsafe_scythe

        self.steps += 1
        terminated = state.terminated
        if terminated:
            reward += self.reward_config.defeat
        truncated = self.steps >= self.max_steps
        return self.observation(), reward, terminated, truncated, self.info()

    def observation(self) -> list[float]:
        state = self._require_state()
        phase = state.scythe.phase
        phase_index = {
            ScythePhase.COUNTDOWN: 0.0,
            ScythePhase.WARNING: 1.0,
            ScythePhase.SPREAD_ART: 2.0,
        }[phase]
        altar_distance = 0.0
        if state.altar_present and state.altar_lane is not None:
            altar_distance = abs(state.player_lane - state.altar_lane) / (self.lane_count - 1)
        return [
            state.souls.green / 5.0,
            state.souls.red / 5.0,
            state.souls.danger_margin / 5.0,
            float(state.souls.defeated),
            phase_index / 2.0,
            state.scythe.remaining_ticks / max(self.scythe_ticks),
            float(state.altar_present),
            altar_distance,
            state.player_lane / (self.lane_count - 1),
        ]

    def info(self) -> dict[str, Any]:
        state = self._require_state()
        return {**state.observation(), "steps": self.steps, "lane_count": self.lane_count}

    def _require_state(self) -> JinHillaState:
        if self.state is None:
            raise RuntimeError("call reset() before interacting with the environment")
        return self.state


def smoke_test() -> None:
    env = JinHillaTrainingEnv(scythe_ticks=(2, 1, 1))
    observation, _ = env.reset(options={"green_skulls": 3, "red_skulls": 2, "player_lane": 3})
    assert len(observation) == env.observation_size
    env.spawn_altar(3)
    _, reward, terminated, _, info = env.step(Action.HARVEST)
    assert reward > 0 and not terminated and info["red_skulls"] == 1
    env.apply_web_hit()
    env.apply_web_hit()
    _, reward, terminated, _, _ = env.step(Action.STAY)
    assert terminated and reward < 0


if __name__ == "__main__":
    smoke_test()
    print("Jin Hilla training integration: OK")
