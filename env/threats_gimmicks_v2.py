"""Threat/Gimmick v2.1 — 붉은 실 회피 가능성 수정.

결함 (커리큘럼 3 Phase 모두 생존 88스텝 고정, 정화 0.00):
  세트당 7영역이 0.25초 간격으로 순차 생성되어 1.75초 내 전면 커버.
  danger_lanes가 "영역 전체"를 위험 처리 → 어떤 레인에 있어도 피격 → 행동 무관 환경.

수정 (실게임 반영, verus_hilla_settings.json의 미사용 데이터 활용):
  1. safe_zones_no_C (C형 미출현 영역 2·4·7): 해당 영역은 C형 실이 없으므로
     항상 안전하지는 않지만, 서브레인 여유가 가장 큼 → 이 영역의 위험 서브레인 축소
  2. type_danger_subs (유형별 위험 서브레인):
     A형(직선) = 영역 내 서브레인 0~1만 위험 (좌측 240px 여유)
     B형(V자)  = 서브레인 1~2만 위험 (우측 200px 여유)
     C형(전체) = 서브레인 0~2 전부 위험
  3. 서브레인 모델: 각 영역(레인)을 3서브레인으로 분할.
     GameState는 정수 레인(0~6), 플레이어 위치는 "레인 + 서브레인 오프셋"으로
     내부 추적 → danger 판정 시 (레인, 서브레인) 정밀 비교.

하위 호환: danger_lanes(state) 시그니처 유지. 새 danger_sublanes(state) 추가.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from env.game_state import BossGameState
from env.boss_env_base import BossProfile


# ─────────────────────────────────────────────────────────────
# 위협 v2 — GameState 조작자
# ─────────────────────────────────────────────────────────────
class ThreatV2(ABC):
    """GameState를 읽고 위협을 진행하는 인터페이스. 상태는 GameState에 반영."""

    @abstractmethod
    def reset(self, state: BossGameState, rng: Any) -> None:
        """에피소드 시작. GameState 위협 필드 초기화."""

    @abstractmethod
    def tick(self, state: BossGameState, rng: Any) -> list[dict[str, Any]]:
        """1틱 진행. GameState의 위협 필드 갱신. 이벤트 목록 반환."""

    @abstractmethod
    def telegraph_lanes(self, state: BossGameState) -> set[int]:
        """예고 중인 레인."""

    @abstractmethod
    def danger_lanes(self, state: BossGameState) -> set[int]:
        """피격 가능한 레인."""

    @property
    @abstractmethod
    def name(self) -> str:
        """식별자."""


class GimmickV2(ABC):
    """GameState를 읽고 기믹을 진행하는 인터페이스."""

    @abstractmethod
    def reset(self, state: BossGameState, rng: Any) -> None:
        """에피소드 시작. GameState 기믹 필드 초기화."""

    @abstractmethod
    def tick(self, state: BossGameState, rng: Any) -> dict[str, Any]:
        """틱마다 진행. GameState 갱신."""

    @abstractmethod
    def can_interact(self, state: BossGameState) -> bool:
        """플레이어가 상호작용 가능한지."""

    @abstractmethod
    def interact(self, state: BossGameState) -> dict[str, Any]:
        """상호작용 실행. GameState 갱신."""

    @abstractmethod
    def on_hit(self, state: BossGameState) -> bool:
        """위협 피격 시 기믹 반응. True = 사망."""

    @abstractmethod
    def observation_fields(self, state: BossGameState) -> dict[str, float]:
        """관측 기여 필드."""

    @property
    @abstractmethod
    def name(self) -> str:
        """식별자."""


# ─────────────────────────────────────────────────────────────
# 붉은 실 v2.1 — 유형별 서브레인 위험 모델
# ─────────────────────────────────────────────────────────────
# 서브레인 의미 (영역 내 3칸):
#   sub 0 = 영역 좌측 1/3, sub 1 = 중앙, sub 2 = 우측 1/3
# 유형별 위험 (실제 게임의 여유 공간 반영):
#   A형(직선 낙하): 중앙 서브레인 1에 낙하 → 좌우 여유. 위험 = {1}
#     → 좌측 240px 여유 = sub 0은 안전
#   B형(V자 결속): V자 중심이 영역 우측 → 위험 = {1, 2}, sub 0(좌 240px) 안전
#     실제: "보스 앞쪽 안전, 전방 여유 240px" → 서브레인 좌측이 안전
#   C형(전면 커버): 전부 위험 = {0, 1, 2} — 단 C형은 영역 2·4·7에는 미출현
# safe_zones_no_C = [2, 4, 7] (1-based) → 0-based {1, 3, 6}
SUB_DANGER = {
    "A": {1},
    "B": {1, 2},
    "C": {0, 1, 2},
}
SAFE_NO_C_ZONES = {1, 3, 6}  # 0-based (설정값 2·4·7에서 1 감소)


class RedThreadThreatV2(ThreatV2):
    """진힐라 붉은 실 — 유형별 서브레인 위험 모델 적용.

    행동이 결과를 바꾸려면:
    - 실이 활성화된 영역에서도 특정 서브레인(=위치 미세 조정)은 생존 가능
    - C형이 오는 영역을 사전에 떠나야 함 (telegraph 1.32초 = 이동 여지)
    """

    def __init__(self, cfg: dict) -> None:
        self.telegraph_ticks = int(cfg.get("telegraph_sec", 1.32) * cfg.get("tps", 60))
        self.thread_active_ticks = int(cfg.get("thread_active_sec", 1.0) * cfg.get("tps", 60))
        self.order_interval = int(cfg.get("order_interval_sec", 0.25) * cfg.get("tps", 60))
        self.set_cooldown_range = tuple(cfg.get("set_cooldown_sec", [1, 3]))
        # 서브레인 위험 (설정 오버라이드 가능)
        rt = cfg.get("type_danger_subs", None)
        if rt:
            self.sub_danger = {k: set(v) for k, v in rt.items()}
        else:
            self.sub_danger = {k: set(v) for k, v in SUB_DANGER.items()}
        # 스케줄러 내부 상태
        self._active: list[dict] = []  # [{zone, type, telegraph, active}]
        self._current_set = 1
        self._current_order = 0
        self._cooldown = 0

    def reset(self, state: BossGameState, rng: Any) -> None:
        self._active.clear()
        self._current_set = int(rng.integers(1, 7))
        self._current_order = 0
        self._cooldown = int(rng.integers(self.set_cooldown_range[0], self.set_cooldown_range[1] + 1))
        state.thread_danger_lanes.clear()
        state.thread_telegraph_lanes.clear()
        # 유형별 위험 서브레인 맵 초기화
        if hasattr(state, "thread_sub_danger"):
            state.thread_sub_danger.clear()

    def tick(self, state: BossGameState, rng: Any) -> list[dict]:
        spawned = []
        still = []
        for t in self._active:
            if t["telegraph"] > 0:
                t["telegraph"] -= 1
                if t["telegraph"] == 0:
                    spawned.append({"type": "thread_active", "zone": t["zone"]})
                still.append(t)
            else:
                t["active"] -= 1
                if t["active"] > 0:
                    still.append(t)
        self._active = still

        if self._current_order == 0:
            if self._cooldown > 0:
                self._cooldown -= 1
            else:
                self._current_set = int(rng.integers(1, 7))
                self._current_order = 1

        if 0 < self._current_order <= 7:
            from env.verus_hilla_settings import SETTINGS
            pattern = SETTINGS()["red_thread_pattern"]["sets"][str(self._current_set)]
            for zone_1b, (order, ttype) in enumerate(pattern, start=1):
                if order == self._current_order:
                    self._active.append({
                        "zone": zone_1b - 1, "type": ttype, "order": order,
                        "telegraph": self.telegraph_ticks, "active": self.thread_active_ticks,
                    })
                    break
            self._current_order += 1
            if self._current_order > 7:
                self._current_order = 0
                self._cooldown = int(rng.integers(self.set_cooldown_range[0], self.set_cooldown_range[1] + 1))

        # GameState에 요약 반영 (SSOT) — 유형별 서브레인 포함
        state.thread_danger_lanes = {t["zone"] for t in self._active if t["telegraph"] <= 0 and t["active"] > 0}
        state.thread_telegraph_lanes = {t["zone"] for t in self._active if t["telegraph"] > 0}
        # zone → (type, danger_subs) 매핑 — env가 정밀 판정에 사용
        state.thread_sub_danger = {
            t["zone"]: (t["type"], self.sub_danger.get(t["type"], {0, 1, 2}))
            for t in self._active if t["telegraph"] <= 0 and t["active"] > 0
        }
        state.thread_telegraph_types = {
            t["zone"]: t["type"] for t in self._active if t["telegraph"] > 0
        }

        return spawned

    def telegraph_lanes(self, state: BossGameState) -> set[int]:
        return state.thread_telegraph_lanes

    def danger_lanes(self, state: BossGameState) -> set[int]:
        return state.thread_danger_lanes

    @property
    def name(self) -> str:
        return "red_thread"


class BoneWaveThreatV2(ThreatV2):
    """진힐라 뼈 파동 — GameState의 bone_wave_* 필드를 조작."""

    def __init__(self, cfg: dict) -> None:
        self.min_phase = int(cfg.get("min_phase", 2))
        self.first_hit_delay = int(cfg.get("first_hit_delay_sec", 2.25) * cfg.get("tps", 60))
        self.duration = int(cfg.get("duration_sec", 0.54) * cfg.get("tps", 60))

    def reset(self, state: BossGameState, rng: Any) -> None:
        state.bone_wave_active = False
        state.bone_wave_aura = None
        state.bone_wave_danger_lanes.clear()
        state.bone_wave_until_hit = 0
        state.bone_wave_duration = 0

    def trigger(self, state: BossGameState, rng: Any) -> bool:
        if state.bone_wave_active or state.phase < self.min_phase:
            return False
        if abs(state.player_lane - state.boss_lane) > 1:
            return False
        aura = rng.choice(["green", "purple"])
        boss = state.boss_lane
        # 초록(V자): 보스 앞(전방) 안전 — 전방 여유 240px, 후방 90px
        # 보라(∧자): 보스 뒤(후방) 안전 — 후방 여유 200px, 전방 50px
        # 7레인 추상화: 안전 레인을 명시적으로 보장
        if aura == "green":
            # 전방(플레이어 진행 +1 방향) 안전 보장
            danger = {boss, max(0, boss - 1), max(0, boss - 2)}
            safe_lane = min(6, boss + 1)
        else:
            # 후방(-1 방향) 안전 보장
            danger = {boss, min(6, boss + 1), min(6, boss + 2)}
            safe_lane = max(0, boss - 1)
        danger.discard(safe_lane)  # 안전 레인 명시 제거
        state.bone_wave_active = True
        state.bone_wave_aura = aura
        state.bone_wave_danger_lanes = danger
        state.bone_wave_safe_lane = safe_lane
        state.bone_wave_until_hit = self.first_hit_delay
        state.bone_wave_duration = self.duration
        return True

    def tick(self, state: BossGameState, rng: Any) -> list[dict]:
        if not state.bone_wave_active:
            return []
        state.bone_wave_until_hit -= 1
        if state.bone_wave_until_hit <= 0:
            state.bone_wave_duration -= 1
            if state.bone_wave_duration <= 0:
                state.bone_wave_active = False
            return [{"type": "bone_wave_active"}]
        return []

    def telegraph_lanes(self, state: BossGameState) -> set[int]:
        return set()

    def danger_lanes(self, state: BossGameState) -> set[int]:
        if state.bone_wave_active and state.bone_wave_until_hit <= 0:
            return state.bone_wave_danger_lanes
        return set()

    @property
    def name(self) -> str:
        return "bone_wave"


class SoulAltarGimmickV2(GimmickV2):
    """진힐라 영혼 제단 — GameState의 skull/altar/candle 필드를 조작."""

    def __init__(self, cfg: dict) -> None:
        self.soul_start = int(cfg.get("soul_start", 5))
        self.spawn_zones = cfg.get("spawn_zones", [2, 3, 4, 5, 6])
        self.respawn_sec = tuple(cfg.get("boss_interference_respawn_sec", [2, 3]))
        self.presses_per_cleanse = int(cfg.get("harvest_presses_per_cleanse", 10))
        # 제단 스폰에 필요한 촛불 수 (커리큘럼 완화용 — 실제 게임은 3)
        self.spawn_candles = int(cfg.get("altar_spawn_candles", 3))

    def reset(self, state: BossGameState, rng: Any) -> None:
        state.green_skulls = self.soul_start
        state.red_skulls = 0
        state.candles_lit = 0
        state.altar_present = False
        state.altar_lane = None
        state.altar_respawn_ticks = 0
        state.total_cleansed = 0
        state.total_web_hits = 0

    def tick(self, state: BossGameState, rng: Any) -> dict[str, Any]:
        events = {}
        # 촛불 조건 충족 → 제단 스폰 (on_hit에서 세팅된 pending 소비)
        if state.altar_pending_spawn and not state.altar_present:
            state.altar_pending_spawn = False
            state.altar_present = True
            state.altar_lane = int(rng.choice([z - 1 for z in self.spawn_zones]))
            events["altar_spawned"] = True
            return events
        if state.altar_respawn_ticks > 0:
            state.altar_respawn_ticks -= 1
            if state.altar_respawn_ticks == 0:
                state.altar_present = True
                state.altar_lane = int(rng.choice([z - 1 for z in self.spawn_zones]))
                events["altar_spawned"] = True
        return events

    def can_interact(self, state: BossGameState) -> bool:
        return (state.altar_present and state.altar_lane is not None
                and abs(state.player_lane - state.altar_lane) <= 1)

    def interact(self, state: BossGameState) -> dict[str, Any]:
        cleaned = min(30 // self.presses_per_cleanse, state.red_skulls)
        state.red_skulls -= cleaned
        state.green_skulls += cleaned
        state.total_cleansed += cleaned
        if state.red_skulls == 0:
            state.altar_present = False
            state.altar_lane = None
        return {"cleansed": cleaned}

    def on_hit(self, state: BossGameState) -> bool:
        """붉은 실 피격 — 초록 → 빨강, 촛불 점등. True = 사망.

        사망 규칙 (실게임 검증): 초록 해골이 0이 되면(전 영혼 오염) 사망.
        빨간 해골 수 자체는 사망과 무관 — 제단 정화로 복구 가능.
        """
        if state.green_skulls <= 0:
            state.terminated = True
            state.termination_reason = "soul_depletion"
            return True
        state.green_skulls -= 1
        state.red_skulls += 1
        state.total_web_hits += 1
        state.candles_lit = min(15, state.candles_lit + 1)
        self._maybe_spawn_altar(state)
        return False

    def _maybe_spawn_altar(self, state: BossGameState) -> None:
        # 촛불 조건 충족 시 다음 tick()에서 스폰 (v2.1 즉시 스폰 시맨틱 복원)
        if state.candles_lit >= self.spawn_candles and not state.altar_present:
            state.altar_pending_spawn = True

    def observation_fields(self, state: BossGameState) -> dict[str, float]:
        return {
            "green_skulls": state.green_skulls / self.soul_start,
            "red_skulls": state.red_skulls / self.soul_start,
            "danger_margin": state.danger_margin / self.soul_start,
            "defeated": float(state.defeated),
            "altar_present": float(state.altar_present),
            "altar_distance": state.altar_distance,
            "candles_lit": state.candles_lit / 15.0,
        }

    @property
    def name(self) -> str:
        return "soul_altar"
