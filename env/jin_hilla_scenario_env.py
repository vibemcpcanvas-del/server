import random
from typing import Any, Dict, Tuple

from .jin_hilla_environment_core import JinHillaEnvironmentCore, RewardConfig


class JinHillaScenarioEnv:
    """
    M8 진힐라 학습용 시나리오 환경.
    
    - 7 레인 이산 공간
    - hazard_interval=12, altar_interval=45, max_steps=360
    - 위험 레인과 제단 레인은 재현 가능한 난수 시드로 생성
    - 이 값들과 7-레인 모델은 실제 진힐라 규칙이 아니라 학습용 fixture
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

        # 7 레인
        self.num_lanes = 7

        # 상태
        self.step_count = 0
        self.player_lane = 0
        self.hazard_lane = 0
        self.altar_lane = 0
        self.altar_active = False

        self._reset_state()

    def _reset_state(self):
        self.step_count = 0
        self.player_lane = self.rng.randint(0, self.num_lanes - 1)
        # 위험 레인은 플레이어와 다르게
        hazard_candidate = self.rng.randint(0, self.num_lanes - 1)
        while hazard_candidate == self.player_lane:
            hazard_candidate = self.rng.randint(0, self.num_lanes - 1)
        self.hazard_lane = hazard_candidate

        # 제단 레인
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

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        action: 0=LEFT, 1=STAY, 2=RIGHT, 3=HARVEST
        """
        reward = 0.0
        info: Dict[str, Any] = {}

        # 1. 플레이어 이동 처리
        if action == 0:  # LEFT
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:  # RIGHT
            self.player_lane = min(self.num_lanes - 1, self.player_lane + 1)
        # STAY, HARVEST 는 레인 변경 없음

        # 2. 한 스텝 생존 보상 (소량)
        reward += 0.05

        # 3. 위험 레인 처리
        hazard_now = self.hazard_lane
        if hazard_now is not None:
            if self.player_lane == hazard_now:
                # 피격: 핵심 버그 수정 - 명시적 패널티
                self.core.apply_web_hit()
                reward -= 5.0
            else:
                # 회피 보상
                reward += 1.0

        # 4. 제단 정화 처리 (HARVEST 행동)
        if action == 3:  # HARVEST
            if self.altar_active:
                # 제단 레인에서 1 레인 이내면 정화 가능
                if abs(self.player_lane - self.altar_lane) <= 1:
                    cleansed = self.core.attempt_cleanse()
                    if cleansed:
                        reward += 10.0
                else:
                    # 정화 불가 상태의 HARVEST: 작은 패널티
                    reward -= 0.1
            else:
                # 제단이 없으면 HARVEST 는 무의미
                reward -= 0.1

        # 5. 스텝 카운트 증가
        self.step_count += 1

        # 6. 위험/제단 레인 업데이트 (주기적)
        if self.step_count % self.hazard_interval == 0:
            # 새 위험 레인
            new_hazard = self.rng.randint(0, self.num_lanes - 1)
            while new_hazard == self.player_lane:
                new_hazard = self.rng.randint(0, self.num_lanes - 1)
            self.hazard_lane = new_hazard

        if self.step_count % self.altar_interval == 0:
            # 제단 재배치
            self.altar_lane = self.rng.randint(0, self.num_lanes - 1)
            self.altar_active = True

        # 7. 종료 조건
        terminated = False
        truncated = False

        if self.core.red_skulls > self.core.green_skulls:
            terminated = True
            info["termination_reason"] = "red_exceeds_green"

        if self.step_count >= self.max_steps:
            truncated = True
            info["truncation_reason"] = "max_steps"

        return self._get_observation(), reward, terminated, truncated, info
