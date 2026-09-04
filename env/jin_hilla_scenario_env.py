import random
import math
from typing import Any, Dict, Tuple

from .jin_hilla_environment_core import JinHillaState, ScytheCycle, SoulState, ScythePhase


class JinHillaScenarioEnv:
    """M8 Jin Hilla training scenario with numeric policy observations.

    Altar spawning rule (deviates intentionally from the real-game formula):
    - The real MapleStory formula is ceil(green_skulls / 2) hits to spawn an
      altar. Under this simulator's simplified defeat rule (defeated when
      red_skulls > green_skulls), that exact formula coincides with the lethal
      hit for every ODD starting skull count (5, 3, 1): the hit that spawns the
      altar is the SAME hit that ends the episode, making cleansing
      structurally impossible no matter how good the policy is.
    - To keep training meaningful, this simulator instead uses
      max(1, green_skulls // 2), which guarantees at least a one-hit buffer
      between the altar-spawning hit and the lethal hit for every starting
      skull count. This is a disclosed fixture deviation, not the real game
      rule.
    - If green_skulls <= 1 when the threshold is (re)activated, altar spawning
      is impossible until the next soul slash.
    - Each web hit during the cycle counts toward this threshold; once
      reached, a single altar is spawned.

    Hazard-avoidance shaping:
    - In addition to the sparse +1 dodge / -5 hit reward, a potential-based
      shaping term rewards keeping distance from the current hazard lane.
      Without this, the only signal a policy gets about the hazard is after
      the fact (already hit or already safe this step), which gave no
      gradient toward proactively moving away. This mirrors the existing
      altar-distance PBRS term.
    """

    observation_size = 7
    action_size = 4

    def __init__(
        self,
        seed: int = 7,
        hazard_interval: int = 16,
        altar_interval: int = 30,
        max_steps: int = 360,
    ) -> None:
        self.rng = random.Random(seed)
        self.hazard_interval = hazard_interval
        self.altar_interval = altar_interval
        self.max_steps = max_steps
        self.num_lanes = 7
        self.step_count = 0
        self.player_lane = 0
        self.hazard_lane = 0
        self.altar_lane = 0
        self.altar_active = False
        self.web_hits = 0
        self.cleanse_count = 0
        self.dodge_count = 0
        self.state: JinHillaState | None = None
        self.altar_hits_needed: int | None = None
        self.hits_since_soul_slash: int = 0
        self.can_spawn_altar: bool = True
        self._reset_state()

    def _reset_state(self) -> None:
        self.step_count = 0
        self.web_hits = 0
        self.cleanse_count = 0
        self.dodge_count = 0
        self.player_lane = self.rng.randint(0, self.num_lanes - 1)
        self.hazard_lane = self.rng.randint(0, self.num_lanes - 1)
        while self.hazard_lane == self.player_lane:
            self.hazard_lane = self.rng.randint(0, self.num_lanes - 1)
        self.altar_lane = self.rng.randint(0, self.num_lanes - 1)
        self.altar_active = False
        self.altar_hits_needed = None
        self.hits_since_soul_slash = 0
        self.can_spawn_altar = True
        self.state = JinHillaState(
            souls=SoulState(green=5, red=0),
            scythe=ScytheCycle(30, 15, 20),
            altar_present=False,
            altar_lane=None,
            player_lane=self.player_lane,
        )
        # Activate the first altar threshold immediately so episodes that end
        # before the scythe cycle's first SPREAD_ART tick (30 + 15 = 45 steps)
        # still get a chance to spawn an altar.
        self._on_soul_slash()

    def _require_state(self) -> JinHillaState:
        if self.state is None:
            raise RuntimeError("call reset() before interacting with the environment")
        return self.state

    def reset(self) -> list[float]:
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> list[float]:
        state = self._require_state()
        return [
            self.player_lane / (self.num_lanes - 1),
            self.hazard_lane / (self.num_lanes - 1),
            self.altar_lane / (self.num_lanes - 1),
            float(self.altar_active),
            state.souls.green / 5.0,
            state.souls.red / 5.0,
            self.step_count / self.max_steps,
        ]

    def _info(self) -> Dict[str, Any]:
        state = self._require_state()
        return {
            "player_lane": self.player_lane,
            "hazard_lane": self.hazard_lane,
            "altar_lane": self.altar_lane,
            "altar_active": self.altar_active,
            "step_count": self.step_count,
            "green_skulls": state.souls.green,
            "red_skulls": state.souls.red,
            "web_hits": self.web_hits,
            "cleanse_count": self.cleanse_count,
            "dodge_count": self.dodge_count,
            "altar_hits_needed": self.altar_hits_needed,
            "hits_since_soul_slash": self.hits_since_soul_slash,
        }

    def _altar_potential(self, player_lane: int, altar_lane: int, altar_active: bool) -> float:
        if not altar_active:
            return 0.0
        return 2.0 / (abs(player_lane - altar_lane) + 1)

    def _hazard_potential(self, player_lane: int, hazard_lane: int) -> float:
        """Higher (less negative) when farther from the hazard lane.

        This gives a dense, directional signal that rewards proactively
        moving away from danger, instead of only reacting after a hit
        already happened.
        """
        distance = abs(player_lane - hazard_lane)
        max_distance = self.num_lanes - 1
        return -1.0 * (max_distance - distance) / max_distance

    def _on_soul_slash(self) -> None:
        """Update altar spawning rules at the start of a soul slash cycle.

        Uses max(1, green // 2) instead of the real-game ceil(green / 2) to
        guarantee at least a one-hit buffer before the lethal hit under this
        simulator's simplified defeat rule (red_skulls > green_skulls). See
        the class docstring for the full rationale.
        """
        state = self._require_state()
        self.hits_since_soul_slash = 0
        green = state.souls.green
        if green <= 1:
            self.can_spawn_altar = False
            self.altar_hits_needed = None
            return
        self.can_spawn_altar = True
        self.altar_hits_needed = max(1, green // 2)

    def _register_hit_for_altar(self) -> None:
        """Register a web hit and spawn altar when the threshold is reached."""
        if not self.can_spawn_altar or self.altar_hits_needed is None or self.altar_active:
            return
        self.hits_since_soul_slash += 1
        if self.hits_since_soul_slash < self.altar_hits_needed:
            return
        # Spawn a single altar for this cycle.
        self.altar_lane = self.rng.randint(0, self.num_lanes - 1)
        self.altar_active = True
        state = self._require_state()
        state.altar_present = True
        state.altar_lane = self.altar_lane

    def step(self, action: int) -> Tuple[list[float], float, bool, bool, Dict[str, Any]]:
        if action not in (0, 1, 2, 3):
            raise ValueError(f"unknown action: {action}")
        state = self._require_state()
        reward = 0.0
        prev_altar_potential = self._altar_potential(self.player_lane, self.altar_lane, self.altar_active)
        prev_hazard_potential = self._hazard_potential(self.player_lane, self.hazard_lane)

        phase_change = state.tick()
        if phase_change is ScythePhase.SPREAD_ART:
            self._on_soul_slash()

        if action == 0:
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:
            self.player_lane = min(self.num_lanes - 1, self.player_lane + 1)
        state.player_lane = self.player_lane
        reward += 0.05

        if self.player_lane == self.hazard_lane:
            state.apply_web_hit()
            self.web_hits += 1
            reward -= 5.0
            self._register_hit_for_altar()
        else:
            self.dodge_count += 1
            reward += 1.0

        if action == 3:
            if self.altar_active and abs(self.player_lane - self.altar_lane) <= 1:
                cleaned = state.harvest_altar(presses=10)
                self.cleanse_count += cleaned
                reward += 10.0 * cleaned
            else:
                reward -= 0.1
            self.altar_active = state.altar_present
            if state.altar_lane is not None:
                self.altar_lane = state.altar_lane

        curr_altar_potential = self._altar_potential(self.player_lane, self.altar_lane, self.altar_active)
        curr_hazard_potential = self._hazard_potential(self.player_lane, self.hazard_lane)
        reward += 0.99 * curr_altar_potential - prev_altar_potential
        reward += 0.99 * curr_hazard_potential - prev_hazard_potential
        self.step_count += 1

        if self.step_count % self.hazard_interval == 0:
            self.hazard_lane = self.rng.randint(0, self.num_lanes - 1)
            while self.hazard_lane == self.player_lane:
                self.hazard_lane = self.rng.randint(0, self.num_lanes - 1)

        terminated = state.terminated
        truncated = self.step_count >= self.max_steps
        info = self._info()
        if terminated:
            info["termination_reason"] = state.termination_reason
        if truncated:
            info["truncation_reason"] = "max_steps"
        return self._get_observation(), reward, terminated, truncated, info
