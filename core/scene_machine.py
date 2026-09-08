# -*- coding: utf-8 -*-
"""장면 상태 머신 — 사냥/일퀘/보스전 등 게임 컨텍스트 구분 + 확장 가능 구조.

설계 (Game Programming Patterns의 HFSM 표준):
- GameState: 현재 장면 식별 (증상 기반 분류기)
- SceneState: 각 장면의 행동 정의 (enter/tick/exit)
- SceneMachine: 전환 관리 + 폴백(UNKNOWN 시 안전 정지)

확장 방법: SceneState 상속 클래스 하나 추가 + 감별자(특징 함수) 등록. 기존 코드 무수정.

감별 기준 (실측):
- BOSS_FIGHT: 보스 HP바(핑크 풀폭) 존재 + 해골 UI 존재
- BOSS_LOBBY: 미니맵 "고통의 미궁" 텍스트 or 입장 UI(NORMAL/HARD 버튼) — 추후 정밀화
- HUNTING_MAP: 일반 사냥필드 (미니맵 UI + 경험치 하단바) — 휴리스틱
- QUEST_DIALOG: 대화창(NPC 초상 + 진행 버튼) — 추후
- UNKNOWN: 위 어느 것도 아님 → 안전 정지

v0.1: BOSS_FIGHT / UNKNOWN 만 확정 구현 (실측 검증된 판별자만 등록).
      나머지는 플레이스홀더 — 사용자 플레이 녹화로 GT 수집 후 확정.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import cv2
import numpy as np

from core.vision.real_parser_v2 import parse_frame


class Scene(Enum):
    UNKNOWN = "unknown"
    BOSS_FIGHT = "boss_fight"
    BOSS_LOBBY = "boss_lobby"       # 입장 UI/로비 (미완 — 플레이스홀더)
    HUNTING_MAP = "hunting_map"     # 일반 사냥필드 (미완)
    QUEST_DIALOG = "quest_dialog"   # 일일퀘 대화창 (미완)


@dataclass
class SceneObservation:
    """한 프레임의 장면 관측 결과."""
    scene: Scene
    confidence: float
    parsed: dict = field(default_factory=dict)
    timestamp: float = 0.0


# ── 감별자 (장면 → 특징 함수) ──────────────────────────────────
# 각 감별자는 (프레임) -> (bool, confidence). 등록 순서 = 우선순위.

def detect_boss_fight(frame: np.ndarray) -> tuple[bool, float]:
    res = parse_frame(frame)
    if res.get("is_bossfight") and res.get("skulls_seen", 0) + res.get(
            "skulls_destroyed", 0) >= 2:
        return True, 0.95
    if res.get("is_bossfight"):
        return True, 0.7   # HP바만 있고 해골 UI 미검출 (전환 중)
    return False, 0.0


DETECTORS: list[tuple[Scene, Callable[[np.ndarray], tuple[bool, float]]]] = [
    (Scene.BOSS_FIGHT, detect_boss_fight),
    # BOSS_LOBBY / HUNTING_MAP / QUEST_DIALOG 감별자는
    # 사용자 플레이 녹화 GT 수집 후 추가 (플레이스홀더)
]


# ── 상태 정의 ────────────────────────────────────────────────
class SceneState:
    """장면 상태. 서브클래스가 행동을 정의한다."""
    scene: Scene = Scene.UNKNOWN

    def on_enter(self, ctx: dict):
        """장면 진입 시 1회."""

    def tick(self, frame: np.ndarray, parsed: dict, ctx: dict) -> int | None:
        """장면 유지 중 매 프레임. 반환: 행동(int) 또는 None(행동 없음)."""
        return None

    def on_exit(self, ctx: dict):
        """장면 이탈 시 1회."""


class BossFightState(SceneState):
    """보스전: v7 정책/휴리스틱으로 회피·정화 (관전 모드와 공유)."""
    scene = Scene.BOSS_FIGHT

    def on_enter(self, ctx: dict):
        print("[SCENE] ⚔️ 보스전 진입")
        ctx.setdefault("events", []).append(
            {"t": time.time(), "event": "boss_fight_enter"})

    def tick(self, frame, parsed, ctx):
        # 회피 정책 자리 — 현재는 안전 유지(STAY)
        return None

    def on_exit(self, ctx):
        print("[SCENE] 보스전 종료")
        ctx.setdefault("events", []).append(
            {"t": time.time(), "event": "boss_fight_exit"})


class UnknownState(SceneState):
    """알 수 없는 장면: 안전 원칙 — 어떤 입력도 내지 않는다."""
    scene = Scene.UNKNOWN


SCENE_STATES: dict[Scene, SceneState] = {
    Scene.BOSS_FIGHT: BossFightState(),
    Scene.UNKNOWN: UnknownState(),
}


class SceneMachine:
    """장면 상태 머신 — 감별 → 전환 → 현재 상태에 행동 위임.

    히스테리시스: 1프레임 오탐으로 흔들리지 않도록 N프레임 연속 감지 시 전환.
    """
    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = confirm_frames
        self.current: Scene = Scene.UNKNOWN
        self._candidate: Scene | None = None
        self._candidate_count = 0
        self.ctx: dict = {"events": []}

    def update(self, frame: np.ndarray) -> SceneObservation:
        detected, conf = Scene.UNKNOWN, 0.0
        for scene, detector in DETECTORS:
            ok, c = detector(frame)
            if ok:
                detected, conf = scene, c
                break
        # 히스테리시스
        if detected == self.current:
            self._candidate, self._candidate_count = None, 0
        elif detected == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = detected, 1
        if (self._candidate_count >= self.confirm_frames
                and self._candidate != self.current):
            self._transition(self._candidate)
        state = SCENE_STATES.get(self.current, UnknownState())
        if self.current == Scene.BOSS_FIGHT:
            state.tick(frame, {}, self.ctx)
        return SceneObservation(self.current, conf, {}, time.time())

    def _transition(self, new_scene: Scene):
        old_state = SCENE_STATES.get(self.current, UnknownState())
        old_state.on_exit(self.ctx)
        self.current = new_scene
        new_state = SCENE_STATES.get(new_scene, UnknownState())
        new_state.on_enter(self.ctx)
        self._candidate, self._candidate_count = None, 0
