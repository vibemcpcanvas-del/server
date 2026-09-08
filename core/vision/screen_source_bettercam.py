# -*- coding: utf-8 -*-
"""BetterCam(Desktop Duplication) 화면 캡처 소스 — v3 실기기 표준.

기존 DXCamScreenSource와 동일 인터페이스(grab/close)로 구현해
기존 라이브 파이프라인과 호환. Moonlight/Sunshine 원격 세션에서도
동일하게 동작한다(화면은 OS 컴포지터가 하나).
"""
from __future__ import annotations

import numpy as np

try:
    import bettercam
except ImportError:  # pragma: no cover
    bettercam = None


class BetterCamScreenSource:
    """전체 화면(또는 지정 영역) 캡처. 기존 screen_source와 drop-in 호환."""

    def __init__(self, region: tuple[int, int, int, int] | None = None,
                 target_fps: int = 30):
        if bettercam is None:
            raise RuntimeError("bettercam 미설치 — pip install bettercam")
        self.region = region
        self._cam = bettercam.create(output_color="BGR")
        self._started = False
        self._last: np.ndarray | None = None
        # None-safe grab 방식 (정적 화면도 마지막 프레임 재사용)

    def grab(self) -> np.ndarray:
        f = self._cam.grab(region=self.region) if self.region else self._cam.grab()
        if f is not None:
            self._last = f.copy()
        if self._last is None:
            raise RuntimeError("캡처 실패 — 화면 출력 확인 필요")
        return self._last

    def grab_frame(self) -> np.ndarray:
        """구 인터페이스(jin_hilla_live_vision_env) 호환 별칭."""
        return self.grab()

    def close(self):
        try:
            self._cam.release()
        except Exception:
            pass
