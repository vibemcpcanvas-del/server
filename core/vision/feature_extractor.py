from __future__ import annotations

import cv2
import numpy as np
from core.vision_schema import BossBattleState


class ModernVisionFeatureExtractor:
    """ROI 파티셔닝 및 공간 특징점 추출 기반 모던 비전 파서."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720) -> None:
        self.width = frame_width
        self.height = frame_height

    def extract_state(self, frame: np.ndarray, elapsed_ratio: float = 0.0) -> BossBattleState:
        """단일 BGR 프레임을 7차원 BossBattleState로 파싱."""
        h, w, _ = frame.shape

        # 1. ROI 파티셔닝
        # HUD 영역 (상단 15%)
        hud_roi = frame[0:int(h * 0.15), :]
        # 전투 플레이 영역 (하단 85%)
        battle_roi = frame[int(h * 0.15):, :]

        # 2. 플레이어 및 위험체 공간 좌표 추출 (HSV / Color Thresholding)
        # 플레이어 추출 (파란색/시안 바이어스)
        hsv_battle = cv2.cvtColor(battle_roi, cv2.COLOR_BGR2HSV)
        
        # 플레이어 마스크 (파란색 계열)
        lower_blue = np.array([90, 80, 80])
        upper_blue = np.array([130, 255, 255])
        mask_player = cv2.inRange(hsv_battle, lower_blue, upper_blue)
        player_lane = self._get_normalized_x_center(mask_player, default_val=0.5)

        # 위험체/실 마스크 (빨간색 계열)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_hazard = cv2.bitwise_or(
            cv2.inRange(hsv_battle, lower_red1, upper_red1),
            cv2.inRange(hsv_battle, lower_red2, upper_red2)
        )
        hazard_lane = self._get_normalized_x_center(mask_hazard, default_val=0.5)

        # 제단 마스크 (보라색 계열)
        lower_purple = np.array([135, 80, 80])
        upper_purple = np.array([165, 255, 255])
        mask_altar = cv2.inRange(hsv_battle, lower_purple, upper_purple)
        altar_active = bool(np.sum(mask_altar > 0) > 100)
        altar_lane = self._get_normalized_x_center(mask_altar, default_val=0.5) if altar_active else 0.5

        # 3. HUD 해골 수 통계적 추정
        # 상단 HUD 영역에서 초록색/빨간색 비율 계산
        hsv_hud = cv2.cvtColor(hud_roi, cv2.COLOR_BGR2HSV)
        mask_green_hud = cv2.inRange(hsv_hud, np.array([35, 80, 80]), np.array([85, 255, 255]))
        mask_red_hud = cv2.bitwise_or(
            cv2.inRange(hsv_hud, lower_red1, upper_red1),
            cv2.inRange(hsv_hud, lower_red2, upper_red2)
        )

        green_pixels = int(np.sum(mask_green_hud > 0))
        red_pixels = int(np.sum(mask_red_hud > 0))
        total_skull_pixels = max(1, green_pixels + red_pixels)

        if total_skull_pixels > 50:
            red_ratio = red_pixels / total_skull_pixels
            red_skulls = int(round(red_ratio * 5))
            green_skulls = max(0, 5 - red_skulls)
        else:
            green_skulls, red_skulls = 5, 0

        return BossBattleState(
            player_lane=float(np.clip(player_lane, 0.0, 1.0)),
            hazard_lane=float(np.clip(hazard_lane, 0.0, 1.0)),
            altar_lane=float(np.clip(altar_lane, 0.0, 1.0)),
            altar_active=altar_active,
            green_skulls=green_skulls,
            red_skulls=red_skulls,
            time_ratio=float(np.clip(elapsed_ratio, 0.0, 1.0)),
        )

    @staticmethod
    def _get_normalized_x_center(binary_mask: np.ndarray, default_val: float = 0.5) -> float:
        moments = cv2.moments(binary_mask)
        if moments["m00"] > 1e-4:
            cx = moments["m10"] / moments["m00"]
            return float(cx / binary_mask.shape[1])
        return default_val
