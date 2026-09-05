from __future__ import annotations

import time
from typing import Protocol, Tuple
import numpy as np


class ScreenSource(Protocol):
    """실시간 화면 캡처 장치 추상 인터페이스."""
    def grab_frame(self) -> np.ndarray:
        """반환 형태: (Height, Width, 3) BGR uint8 넘파이 배열."""
        ...


class SyntheticScreenSource:
    """로컬 테스트 및 CI를 위한 결정론적 가상 화면 생성기."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._start_time = time.time()

    def grab_frame(self) -> np.ndarray:
        # 검은색 캔버스 (720, 1280, 3)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elapsed = time.time() - self._start_time

        # 1. 플레이어 위치 렌더링 (파란색 사각형, 좌우로 주기적 이동)
        player_x = int((np.sin(elapsed) * 0.4 + 0.5) * self.width)
        player_y = int(self.height * 0.8)
        frame[player_y - 20:player_y + 20, max(0, player_x - 15):min(self.width, player_x + 15)] = [255, 120, 0]

        # 2. 위험체 렌더링 (빨간색 실/낙하물)
        hazard_x = int((np.cos(elapsed * 1.5) * 0.4 + 0.5) * self.width)
        hazard_y = int(self.height * 0.7)
        frame[hazard_y - 10:hazard_y + 10, max(0, hazard_x - 10):min(self.width, hazard_x + 10)] = [0, 0, 255]

        # 3. 제단 렌더링 (보라색 사각형)
        altar_x = int(self.width * 0.5)
        altar_y = int(self.height * 0.8)
        frame[altar_y - 25:altar_y + 25, altar_x - 25:altar_x + 25] = [200, 0, 200]

        # 4. 상단 HUD 해골 스택 영역 (초록/빨강 게이지 시뮬레이션)
        # 좌측 상단 초록 3칸, 빨강 2칸
        frame[20:40, 50:110] = [0, 255, 0]   # 초록 해골
        frame[20:40, 110:150] = [0, 0, 255]  # 빨강 해골

        return frame


class DXCamScreenSource:
    """Windows Desktop Duplication API 기반 Zero-Copy 저지연 VRAM 캡처러."""

    def __init__(self, region: Tuple[int, int, int, int] | None = None) -> None:
        try:
            import dxcam  # type: ignore
            self.camera = dxcam.create(output_idx=0, output_color="BGR")
            self.region = region
            self.camera.start(target_fps=60, region=self.region)
        except Exception as e:
            raise RuntimeError(f"DXCam 초기화 실패 (Windows 환경이 아니거나 DirectX 미지원): {e}")

    def grab_frame(self) -> np.ndarray:
        frame = self.camera.get_latest_frame()
        while frame is None:
            time.sleep(0.001)
            frame = self.camera.get_latest_frame()
        return frame

    def close(self) -> None:
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.stop()
