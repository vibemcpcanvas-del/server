# -*- coding: utf-8 -*-
"""실화면 → v7 관측 브리지 (60차원 프레임 스택).

v7 정책의 관측 형식(15차원 × 4프레임 스택)에 맞춰 실화면 파싱 결과를 변환.

매핑 (파서 제공 → 관측 필드):
  green_skulls      g/5                      ✅ 직접
  red_skulls        r/5                      ✅ 직접
  danger_margin     (g-r)/5                  ✅ 파생
  defeated          0 (관전 추정 — 부활창 감지 시 1)  ⚠️ 근사
  altar_can_interact ⚠️ 파서에 제단 판정 없음 → 0 (v7이 harvest 안 누르게 됨)
  altar_direction   ⚠️ 동일 → 0
  altar_present     ⚠️ 동일 → 0
  altar_distance    ⚠️ 동일 → 1.0 (최대거리 = 무관심)
  player_lane       플레이어 x → 레인 (카메라 앵커 필요 — 상대좌표 우선)
  boss_lane         (player_lane + boss_dir_lanes) 근사
  boss_hp_pct       보스 HP바 폭 측정 (미구현 → 100 고정, 페이즈 고정 부작용)
  candles_lit       촛불 아이콘 카운트 (미구현 → 0)
  soul_split_seconds 파서의 ss_timer 존재 → 임박도 근사 (미구현 → 150)
  bone_wave_active  bone_wave_danger에서 파생
  bone_wave_danger  파서 미검출 → 0

⚠️ 미구현 필드는 안전한 기본값(관전 모드 기준 판정 유지)으로 채운다.
   이 브리지가 완성되려면: 제단 검출 + HP바 폭 + 촛불 카운트 + 카메라 앵커.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FRAME_DIM = 15
FRAME_STACK = 4


def parse_to_frame_fields(parsed: dict) -> list[float] | None:
    """파서 결과 1프레임 → 15차원 필드. 보스전 아니면 None."""
    if not parsed.get("is_bossfight"):
        return None
    g = parsed.get("green_skulls", 5)
    r = parsed.get("red_skulls", 0)
    bone_danger = 0.0
    bone_active = 0.0
    # 보스/플레이어 검출 시 상대 레인을 레인 값으로 근사
    # (v7 관측의 player_lane/boss_lane은 정규화값 0..1)
    if parsed.get("player_found") and parsed.get("boss_found"):
        player_lane_n = 0.5   # 카메라 앵커 미해결 — 플레이어를 화면 중앙으로 가정
        boss_lane_n = float(np.clip(0.5 + parsed.get("boss_dir_lanes", 0.0) / 6.0, 0.0, 1.0))
    else:
        player_lane_n, boss_lane_n = 0.5, 0.5
    return [
        float(np.clip(g / 5.0, 0.0, 1.0)),                    # 0 green_skulls
        float(np.clip(r / 5.0, 0.0, 1.0)),                    # 1 red_skulls
        float(np.clip((g - r) / 5.0, 0.0, 1.0)),              # 2 danger_margin
        0.0,                                                  # 3 defeated
        0.0,                                                  # 4 altar_can_interact (미구현)
        0.0,                                                  # 5 altar_direction (미구현)
        0.0,                                                  # 6 altar_present (미구현)
        1.0,                                                  # 7 altar_distance (미구현=무관심)
        player_lane_n,                                        # 8 player_lane (근사)
        boss_lane_n,                                          # 9 boss_lane (근사)
        1.0,                                                  # 10 boss_hp_pct (미구현=100)
        0.0,                                                  # 11 candles_lit (미구현)
        float(np.clip(parsed.get("ss_timer_visible", False) and 0.2 or 1.0, 0.0, 1.0)),
        bone_active,                                          # 13 bone_wave_active
        bone_danger,                                          # 14 bone_wave_danger
    ]


class ObservationBridge:
    """실화면 → 60차원 관측. 프레임 스택 유지 (부족한 과거는 현재값으로 채움)."""

    def __init__(self):
        self._stack: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._primed = False

    def reset(self):
        self._stack.clear()
        self._primed = False

    def push(self, parsed: dict) -> np.ndarray | None:
        """파서 결과 추가 → 60차원 관측 반환. 보스전 아니면 None(스택 유지)."""
        fields = parse_to_frame_fields(parsed)
        if fields is None:
            return None
        frame = np.array(fields, dtype=np.float32)
        self._stack.append(frame)
        # 스택이 차오르지 않은 초기엔 현재 프레임으로 채움 (부팅 시퀀스)
        while len(self._stack) < FRAME_STACK:
            self._stack.appendleft(frame)
        self._primed = True
        return np.concatenate(list(self._stack))

    def is_ready(self) -> bool:
        return self._primed and len(self._stack) == FRAME_STACK
