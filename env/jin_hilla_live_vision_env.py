from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.vision.screen_source import ScreenSource, SyntheticScreenSource
from core.vision.feature_extractor import ModernVisionFeatureExtractor


class JinHillaVisionObservationWrapper(gym.ObservationWrapper):
    """Farama Foundation Gymnasium 표준을 준수하는 실시간 화면 관측 Wrapper.

    기능:
    - 카메라/화면 소스로부터 실시간 RGB/BGR 버퍼를 읽음
    - FeatureExtractor를 거쳐 7차원 정규화 Box 텐서로 변환
    - M8 PPO 정책망이 즉각 입력받을 수 있는 Gymnasium Box(7,) 인터페이스 제공
    """

    def __init__(self, env: gym.Env, screen_source: ScreenSource | None = None) -> None:
        super().__init__(env)
        self.screen_source = screen_source or SyntheticScreenSource()
        self.extractor = ModernVisionFeatureExtractor()

        # Gymnasium 표준 Observation Space 재정의 (7차원 실수 벡터)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """내부 시뮬레이터 또는 화면 버퍼 관측을 7차원 정규화 텐서로 변환."""
        # 1. 화면 소스로부터 최신 프레임 캡처
        frame = self.screen_source.grab_frame()

        # 2. 특징점 추출기를 통해 BossBattleState 계산
        state = self.extractor.extract_state(frame)

        # 3. M8 PPO 신경망 규격의 7차원 ndarray 반환
        return np.array(state.to_observation_vector(), dtype=np.float32)
