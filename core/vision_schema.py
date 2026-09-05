from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BossBattleState:
    """7차원 보스전 정규화 상태 텐서 규격."""
    player_lane: float         # 0.0 ~ 1.0 (플레이어 위치)
    hazard_lane: float         # 0.0 ~ 1.0 (맵 낙하물/실 위험 위치)
    altar_lane: float          # 0.0 ~ 1.0 (제단 위치)
    altar_active: bool         # 제단 활성화 여부
    green_skulls: int          # 초록 해골 수 (0~5)
    red_skulls: int            # 빨강 해골 수 (0~5)
    time_ratio: float = 0.0    # 전투 진행 시간 비율 (0.0 ~ 1.0)

    def to_observation_vector(self) -> list[float]:
        """Layer 1 PPO 회피 신경망 입력 텐서로 변환."""
        return [
            float(self.player_lane),
            float(self.hazard_lane),
            float(self.altar_lane),
            float(self.altar_active),
            float(self.green_skulls / 5.0),
            float(self.red_skulls / 5.0),
            float(self.time_ratio),
        ]


class VisionDetector(Protocol):
    """컴퓨터 비전(CV) 프레임 파서 인터페이스."""
    def extract_state(self, frame_bgr: bytes | any) -> BossBattleState:
        ...
