from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScythePhase(str, Enum):
    COUNTDOWN = "countdown"
    WARNING = "warning"
    SPREAD_ART = "spread_art"


@dataclass
class SoulState:
    green: int = 5
    red: int = 0

    @property
    def danger_margin(self) -> int:
        return self.green - self.red

    @property
    def defeated(self) -> bool:
        return self.red > self.green

    def web_hit(self) -> bool:
        if self.green <= 0:
            return True
        self.green -= 1
        self.red += 1
        return self.defeated

    def cleanse(self, count: int = 1) -> int:
        moved = min(max(count, 0), self.red)
        self.red -= moved
        self.green += moved
        return moved


@dataclass
class ScytheCycle:
    countdown_ticks: int
    warning_ticks: int
    spread_art_ticks: int
    phase: ScythePhase = ScythePhase.COUNTDOWN
    remaining_ticks: int | None = None

    def __post_init__(self) -> None:
        if self.remaining_ticks is None:
            self.remaining_ticks = self.countdown_ticks

    def reset(self) -> None:
        self.phase = ScythePhase.COUNTDOWN
        self.remaining_ticks = self.countdown_ticks

    def tick(self) -> ScythePhase | None:
        self.remaining_ticks -= 1
        if self.remaining_ticks > 0:
            return None
        if self.phase is ScythePhase.COUNTDOWN:
            self.phase = ScythePhase.WARNING
            self.remaining_ticks = self.warning_ticks
            return self.phase
        if self.phase is ScythePhase.WARNING:
            self.phase = ScythePhase.SPREAD_ART
            self.remaining_ticks = self.spread_art_ticks
            return self.phase
        self.reset()
        return self.phase


@dataclass
class JinHillaState:
    souls: SoulState
    scythe: ScytheCycle
    altar_present: bool = False
    altar_lane: int | None = None
    player_lane: int = 3
    terminated: bool = False
    termination_reason: str | None = None

    def apply_web_hit(self) -> None:
        if self.terminated:
            return
        if self.souls.web_hit():
            self.terminated = True
            self.termination_reason = "red_skulls_exceed_green_skulls"

    def harvest_altar(self, presses: int, presses_per_cleanse: int = 10) -> int:
        if self.terminated or not self.altar_present or self.altar_lane is None:
            return 0
        if abs(self.player_lane - self.altar_lane) > 1:
            return 0
        cleaned = self.souls.cleanse(presses // presses_per_cleanse)
        if self.souls.red == 0:
            self.altar_present = False
            self.altar_lane = None
        return cleaned

    def tick(self) -> ScythePhase | None:
        if self.terminated:
            return None
        return self.scythe.tick()

    def observation(self) -> dict:
        return {
            "green_skulls": self.souls.green,
            "red_skulls": self.souls.red,
            "danger_margin": self.souls.danger_margin,
            "immediate_defeat": self.souls.defeated,
            "scythe_phase": self.scythe.phase.value,
            "scythe_remaining_ticks": self.scythe.remaining_ticks,
            "altar_present": self.altar_present,
            "altar_lane": self.altar_lane,
            "player_lane": self.player_lane,
        }


def smoke_test() -> None:
    state = JinHillaState(SoulState(green=3, red=0), ScytheCycle(30, 15, 20))
    state.apply_web_hit()
    assert (state.souls.green, state.souls.red, state.terminated) == (2, 1, False)
    state.apply_web_hit()
    assert (state.souls.green, state.souls.red, state.terminated) == (1, 2, True)
    assert state.termination_reason == "red_skulls_exceed_green_skulls"

    state = JinHillaState(SoulState(green=3, red=2), ScytheCycle(2, 1, 1), altar_present=True, altar_lane=3)
    assert state.harvest_altar(20) == 2
    assert (state.souls.green, state.souls.red) == (5, 0)
    assert state.tick() is None
    assert state.tick() is ScythePhase.WARNING
    assert state.tick() is ScythePhase.SPREAD_ART


if __name__ == "__main__":
    smoke_test()
    print("Jin Hilla core rules: OK")
