"""보스 게임 상태 — 단일 진실 공급원(SSOT).

DIAMBRA Arena와 Gymnasium 7 Patterns의 표준 구현 패턴을 따른다:
- 모든 에피소드 상태가 이 데이터클래스에 소유됨
- Threat/Gimmick은 이 상태를 읽고/조작하는 순수 함수(뷰)로 동작
- env는 이 상태의 생성 + Threat/Gimmick 오케스트레이션 + 관측 인코딩만 담당

NG 1999 이론: potential shaping Φ(s)의 s가 이 클래스의 스냅샷.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BossGameState:
    """보스전 에피소드의 모든 상태. SSOT — 이 외에 상태를 두지 않는다."""

    # ── 플레이어 ──
    player_lane: int = 3

    # ── 보스 ──
    boss_lane: int = 3
    boss_hp_pct: float = 100.0
    phase: int = 1

    # ── 에피소드 ──
    steps: int = 0
    terminated: bool = False
    termination_reason: str | None = None

    # ── 기믹별 상태 (보스마다 다름 — 프로파일에서 결정) ──
    # 진힐라: 영혼석 + 제단
    green_skulls: int = 5
    red_skulls: int = 0
    candles_lit: int = 0
    altar_present: bool = False
    altar_lane: int | None = None
    altar_respawn_ticks: int = 0

    # ── 위협별 상태 ──
    # 뼈 파동
    bone_wave_active: bool = False
    bone_wave_aura: str | None = None       # "green" | "purple"
    bone_wave_danger_lanes: set[int] = field(default_factory=set)
    bone_wave_until_hit: int = 0
    bone_wave_duration: int = 0

    # 붉은 실 (활성/예고 레인 — 스케줄러 세부는 Threat 내부에 두되
    #            관측에 필요한 요약만 여기에 반영)
    thread_danger_lanes: set[int] = field(default_factory=set)
    thread_telegraph_lanes: set[int] = field(default_factory=set)
    # 유형별 서브레인 위험 (v2.1): zone → (type, {danger_sublanes})
    thread_sub_danger: dict[int, tuple] = field(default_factory=dict)
    thread_telegraph_types: dict[int, str] = field(default_factory=dict)
    # 플레이어 서브레인 오프셋 (0=좌, 1=중앙, 2=우) — v2.1 정밀 회피용
    player_sublane: int = 1
    # 뼈 파동 안전 레인
    bone_wave_safe_lane: int | None = None
    altar_pending_spawn: bool = False

    # ── Soul Split 타이머 (초 단위, M8 컨벤션 1스텝=2.5초로 감소) ──
    soul_split_seconds_remaining: float = 150.0

    # ── 통계 (info/로깅용) ──
    total_cleansed: int = 0
    total_web_hits: int = 0

    # ── 유틸리티 ──
    @property
    def danger_margin(self) -> int:
        return self.green_skulls - self.red_skulls

    @property
    def defeated(self) -> bool:
        """전 영혼 오염(초록 0) = 사망. 실게임 규칙 — 빨간 해골 수는 무관."""
        return self.green_skulls <= 0

    @property
    def altar_distance(self) -> float:
        """0.0 (제단 위) ~ 1.0 (최대 거리). 제단 없으면 1.0."""
        if not self.altar_present or self.altar_lane is None:
            return 1.0
        return abs(self.player_lane - self.altar_lane) / 6.0

    def snapshot_phi(self) -> float:
        """잠재 함수 Φ(s) — potential shaping용. 최적 정책 불변 보장."""
        return -0.2 * self.altar_distance

    def copy(self) -> "BossGameState":
        """깊은 복사 — 프레임 스택 스냅샷용."""
        return BossGameState(
            player_lane=self.player_lane,
            boss_lane=self.boss_lane,
            boss_hp_pct=self.boss_hp_pct,
            phase=self.phase,
            steps=self.steps,
            terminated=self.terminated,
            termination_reason=self.termination_reason,
            green_skulls=self.green_skulls,
            red_skulls=self.red_skulls,
            candles_lit=self.candles_lit,
            altar_present=self.altar_present,
            altar_lane=self.altar_lane,
            altar_respawn_ticks=self.altar_respawn_ticks,
            bone_wave_active=self.bone_wave_active,
            bone_wave_aura=self.bone_wave_aura,
            bone_wave_danger_lanes=set(self.bone_wave_danger_lanes),
            bone_wave_until_hit=self.bone_wave_until_hit,
            bone_wave_duration=self.bone_wave_duration,
            thread_danger_lanes=set(self.thread_danger_lanes),
            thread_telegraph_lanes=set(self.thread_telegraph_lanes),
            thread_sub_danger=dict(self.thread_sub_danger),
            thread_telegraph_types=dict(self.thread_telegraph_types),
            player_sublane=self.player_sublane,
            bone_wave_safe_lane=self.bone_wave_safe_lane,
            altar_pending_spawn=self.altar_pending_spawn,
            soul_split_seconds_remaining=self.soul_split_seconds_remaining,
            total_cleansed=self.total_cleansed,
            total_web_hits=self.total_web_hits,
        )
