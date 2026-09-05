from __future__ import annotations

from enum import IntEnum
from core.vision_schema import BossBattleState


class Action(IntEnum):
    LEFT = 0
    STAY = 1
    RIGHT = 2
    HARVEST = 3


class PriorityArbiter:
    """3계층 아키텍처 명령어 충돌 방지 및 우선순위 중재기.

    우선순위:
    1. Layer 3 (특수 기믹): 빨강 해골 2개 이상 & 제단 활성화 시 ➔ 제단으로 이동 및 HARVEST
    2. Layer 2 (구석 갇힘 방지): 외곽 레인(0.0 또는 1.0)에 오래 체류 시 ➔ 중앙(0.5)으로 복귀 유도
    3. Layer 1 (범용 맵 회피): 평소에는 PPO 신경망의 무적 회피 예측 행동 채택
    """

    def __init__(self, danger_red_threshold: int = 2) -> None:
        self.danger_red_threshold = danger_red_threshold

    def arbitrate(self, state: BossBattleState, l1_dodge_action: int) -> int:
        # [Layer 3: 특수 기믹 개입]
        # 빨강 해골이 위험 수준이고 제단이 켜져 있다면, 회피를 잠시 중단하고 제단으로 개입
        if state.altar_active and state.red_skulls >= self.danger_red_threshold:
            player_lane_idx = int(round(state.player_lane * 6))
            altar_lane_idx = int(round(state.altar_lane * 6))

            # 제단 반경 1레인 이내 도착 시 정화(HARVEST) 수행
            if abs(player_lane_idx - altar_lane_idx) <= 1:
                return int(Action.HARVEST)
            # 제단보다 왼쪽에 있으면 오른쪽으로 이동
            elif player_lane_idx < altar_lane_idx:
                return int(Action.RIGHT)
            # 제단보다 오른쪽에 있으면 왼쪽으로 이동
            else:
                return int(Action.LEFT)

        # [Layer 2: 외곽 갇힘 방지 바이어스]
        # 실을 피하다가 맵 맨 끝(0번 또는 6번 레인)에 박히는 현상 완화
        if state.player_lane <= 0.05 and state.hazard_lane > 0.3:
            return int(Action.RIGHT)
        if state.player_lane >= 0.95 and state.hazard_lane < 0.7:
            return int(Action.LEFT)

        # [Layer 1: 기본 무적 회피]
        # 평소에는 M8에서 검증된 PPO 모델의 회피 액션을 전적으로 신뢰
        return int(l1_dodge_action)
