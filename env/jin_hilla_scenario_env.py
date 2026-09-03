import random
from typing import Any, Dict, Tuple

from .jin_hilla_environment_core import JinHillaEnvironmentCore


class JinHillaScenarioEnv:
    """
    M8 진힐라 학습용 시나리오 환경 (PBRS 포함).
    
    - 7 레인 이산 공간
    - hazard_interval=12, altar_interval=45, max_steps=360
    - 위험 레인과 제단 레인은 재현 가능한 난수 시드로 생성
    - PBRS: 제단 접근 시 잠재 기반 보상 형태화
    """

    def __init__(
        self,
        seed: int = 7,
        hazard_interval: int = 12,
        altar_interval: int = 45,
        max_steps: int = 360,
    ):
        self.rng = random.Random(seed)
        self.hazard_interval = hazard_interval
        self.altar_interval = altar_interval
        self.max_steps = max_steps

        self.core = JinHillaEnvironmentCore()
        self.num_lanes = 7

        self.step_count = 0
        self.player_lane = 0
        self.hazard_lane = 0
        self.altar_lane = 0
        self.altar_active = False

        self._reset_state()

    def _reset_state(self):
        self.step_count = 0
        self.player_lane = self.rng.randint(0, self.num_lanes - 1)
        hazard_candidate = self.rng.randint(0, self.num_lanes - 1)
        while hazard_candidate == self.player_lane:
            hazard_candidate = self.rng.randint(0, self.num_lanes - 1)
        self.hazard_lane = hazard_candidate

        self.altar_lane = self.rng.randint(0, self.num_lanes - 1)
        self.altar_active = True

        self.core.reset()

    def reset(self) -> Dict[str, Any]:
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "player_lane": self.player_lane,
            "hazard_lane": self.hazard_lane,
            "altar_lane": self.altar_lane,
            "altar_active": self.altar_active,
            "step_count": self.step_count,
            "green_skulls": self.core.green_skulls,
            "red_skulls": self.core.red_skulls,
        }

    def _potential(self, player_lane: int, altar_lane: int, altar_active: bool) -> float:
        """PBRS 잠재 함수: 제단 접근도 기반."""
        if not altar_active:
            return 0.0
        return 2.0 / (abs(player_lane - altar_lane) + 1)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        action: 0=LEFT, 1=STAY, 2=RIGHT, 3=HARVEST
        """
        reward = 0.0
        info: Dict[str, Any] = {}

        # 이전 상태 저장 (PBRS 용)
        prev_potential = self._potential(self.player_lane, self.altar_lane, self.altar_active)

        # 1. 플레이어 이동 처리
        if action == 0:  # LEFT
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:  # RIGHT
            self.player_lane = min(self.num_lanes - 1, self.player_lane + 1)

        # 2. 한 스텝 생존 보상
        reward += 0.05

        # 3. 위험 레인 처리
        hazard_now = self.hazard_lane
        if hazard_now is not None:
            if self.player_lane == hazard_now:
                self.core.apply_web_hit()
                reward -= 5.0
            else:
                reward += 1.0

        # 4. 제단 정화 처리
        if action == 3:  # HARVEST
            if self.altar_active:
                if abs(self.player_lane - self.altar_lane) <= 1:
                    cleansed = self.core.attempt_cleanse()
                    if cleansed:
                        reward += 10.0
                else:
                    reward -= 0.1
            else:
                reward -= 0.1

        # 5. PBRS: 제단 접근 보상
        curr_potential = self._potential(self.player_lane, self.altar_lane, self.altar_active)
        pbrs_reward = 0.99 * curr_potential - prev_potential
        reward += pbrs_reward

        # 6. 스텝 카운트 증가
        self.step_count += 1

        # 7. 위험/제단 레인 업데이트
        if self.step_count % self.hazard_interval == 0:
            new_hazard = self.rng.randint(0, self.num_lanes - 1)
            while new_hazard == self.player_lane:
                new_hazard = self.rng.randint(0, self.num_lanes - 1)
            self.hazard_lane = new_hazard

        if self.step_count % self.altar_interval == 0:
            self.altar_lane = self.rng.randint(0, self.num_lanes - 1)
            self.altar_active = True

        # 8. 종료 조건
        terminated = False
        truncated = False

        if self.core.red_skulls > self.core.green_skulls:
            terminated = True
            info["termination_reason"] = "red_exceeds_green"

        if self.step_count >= self.max_steps:
            truncated = True
            info["truncation_reason"] = "max_steps"

        return self._get_observation(), reward, terminated, truncated, info
