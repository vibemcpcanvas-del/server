"""Verus Hilla v3 — GameState 중앙집중 훈련 환경.

SSOT 위반·위협 간 중복 피격·terminated 이중 권한 문제를 해결한
최종 설계. Gymnasium 7 Patterns 표준 준수.

관측: 60차원 (15차원 × 4프레임 스택)
행동: 4개 (LEFT/STAY/RIGHT/HARVEST)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any

from env.game_state import BossGameState
from env.boss_env_base import BossProfile, PotentialShaping, FrameStacker
from env.threats_gimmicks_v2 import (
    RedThreadThreatV2, BoneWaveThreatV2, SoulAltarGimmickV2,
)


class VerusHillaEnvV3(gym.Env):
    """진힐라 v3 — GameState 중앙집중 + Gymnasium 표준."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, profile: BossProfile, seed: int = 7, max_steps: int | None = None):
        super().__init__()
        self.profile = profile
        self.lane_count = profile.lane_count
        self.max_steps = max_steps or profile.max_steps
        self.tps = profile.ticks_per_second
        self.r_cfg = profile.rewards
        self.phi_k = float(profile.observation_spec.get("potential_phi", {}).get("k", 0.2))

        # 위협·기믹 인스턴스 (상태 없음 — GameState 조작만)
        threat_cfgs = {t["type"]: t for t in profile.threat_configs()}
        gimmick_cfgs = {g["type"]: g for g in profile.gimmick_configs()}
        self._red_thread = RedThreadThreatV2({**threat_cfgs.get("red_thread", {}), "tps": self.tps})
        self._bone_wave = BoneWaveThreatV2({**threat_cfgs.get("bone_wave", {}), "tps": self.tps})
        self._altar = SoulAltarGimmickV2({**gimmick_cfgs.get("soul_altar", {}), "tps": self.tps})

        self.action_space = spaces.Discrete(int(profile.data.get("action_size", 10)))
        fd = profile.frame_dim
        fs = profile.frame_stack
        self.observation_space = spaces.Box(0.0, 1.0, shape=(fd * fs,), dtype=np.float32)
        self.stacker = FrameStacker(size=fs, frame_dim=fd)

        self.state: BossGameState | None = None
        self._shaping = PotentialShaping(gamma=0.99)

        # 보상 분해 로깅 (Pattern 4: dense info)
        self._reward_breakdown: dict[str, float] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Pattern 3: np_random 기반 스토캐스틱 reset
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.state = BossGameState(
            player_lane=self.lane_count // 2,
            boss_lane=self.lane_count // 2,
            boss_hp_pct=100.0,
            phase=1,
            soul_split_seconds_remaining=150,  # 하드 입장 2:30
        )
        self.steps = 0
        self._shaping.reset()
        self._reward_breakdown = {}

        # 위협/기믹 초기화
        self._red_thread.reset(self.state, self.rng)
        self._bone_wave.reset(self.state, self.rng)
        self._altar.reset(self.state, self.rng)

        frame = self._encode_frame()
        self.stacker.reset_with(frame)
        return np.array(self.stacker.stacked(), dtype=np.float32), self._info()

    def step(self, action: int):
        st = self.state
        st.steps += 1
        self._reward_breakdown = {"survive": float(self.r_cfg.get("survive", 0.05))}
        reward = float(self.r_cfg.get("survive", 0.05))

        # 1. 행동 — 9행동 (레인 이동 × 서브레인 미세조정)
        # 0..2: 좌 레인 이동 (서브레인 0/1/2), 3..5: 유지, 6..8: 우 레인 이동
        # + 9: HARVEST
        act = int(action)
        if act == 9:
            if self._altar.can_interact(st):
                result = self._altar.interact(st)
                cleaned = result.get("cleansed", 0)
                cr = float(self.r_cfg.get("cleanse_per_soul", 1.0)) * cleaned
                reward += cr
                self._reward_breakdown["cleanse"] = cr
                # 행동 사슬 보상 — 제단 접근+입력 자체를 보상 (커리큘럼용,
                # 디폴트 0 = 기존 시맨틱 불변). 정화 성공 여부와 무관.
                ha = float(self.r_cfg.get("harvest_attempt", 0.0))
                if ha:
                    reward += ha
                    self._reward_breakdown["harvest_attempt"] = (
                        self._reward_breakdown.get("harvest_attempt", 0.0) + ha)
        else:
            d_lane, d_sub = divmod(act, 3)
            d_lane -= 1  # -1, 0, +1
            d_sub -= 1   # -1, 0, +1
            new_lane = max(0, min(self.lane_count - 1, st.player_lane + d_lane))
            new_sub = max(0, min(2, st.player_sublane + d_sub))
            st.player_lane = new_lane
            st.player_sublane = new_sub

        # 2. 보스 AI
        if self.rng.random() < 0.15:
            st.boss_lane = max(0, min(6, st.boss_lane + int(self.rng.choice([-1, 1]))))
        if st.altar_present and st.boss_lane == st.altar_lane:
            # 보스 방해 — 재등장 대기는 기믹 설정(보간 기본 2~3초, 커리큘럼 완화 가능)
            st.altar_present = False
            st.altar_lane = None
            lo, hi = self._altar.respawn_sec
            st.altar_respawn_ticks = int(self.rng.uniform(lo, hi) * self.tps)

        # 3. 뼈 파동 트리거
        self._bone_wave.trigger(st, self.rng)

        # 3.5 Soul Split (영혼 베기) — M8 컨벤션: 1 스텝 = 2.5초 경과
        # 150초(=60스텝)에 첫 시전, 이후 HP%별 주기(152/126/100초).
        # 발동: 빨간 해골 전부 영구 파괴. 커리큘럼 래퍼가 999999로 고정해 비활성 가능.
        if not st.terminated and st.soul_split_seconds_remaining < 900000:
            st.soul_split_seconds_remaining -= 2.5
            if st.soul_split_seconds_remaining <= 0:
                if st.red_skulls > 0:
                    st.green_skulls -= st.red_skulls
                    st.red_skulls = 0
                # 촛대 재계산 (남은 초록 해골 기준 — v2.1 시맨틱)
                st.candles_lit = min(15, max(0, self._altar.soul_start - st.green_skulls))
                if st.green_skulls <= 0:
                    st.terminated = True
                    st.termination_reason = "soul_split_death"
                    dp = float(self.r_cfg.get("defeat", -25.0))
                    reward += dp
                    self._reward_breakdown["defeat"] = self._reward_breakdown.get("defeat", 0.0) + dp
                st.soul_split_seconds_remaining = float(
                    self._soul_split_cycle(st.boss_hp_pct))

        # 4. 위협 틱 + 피격 처리 (통합 — 중복 피격 방지)
        all_events = []
        all_events += self._red_thread.tick(st, self.rng)
        all_events += self._bone_wave.tick(st, self.rng)

        # 정밀 피격 판정 (v2.1):
        #   붉은 실 — 플레이어 (레인, 서브레인)이 유형별 위험 서브레인에 포함될 때만 피격
        #   뼈 파동 — 레인 단위 (안전 레인 명시 보장)
        hit = False
        if st.player_lane in st.thread_danger_lanes:
            sub_info = st.thread_sub_danger.get(st.player_lane)
            if sub_info is not None:
                _, danger_subs = sub_info
                if st.player_sublane in danger_subs:
                    hit = True
            else:
                hit = True
        if not hit and st.bone_wave_active and st.bone_wave_until_hit <= 0:
            if st.player_lane in st.bone_wave_danger_lanes:
                hit = True

        if hit and not st.terminated:
            before_green = st.green_skulls
            died = self._altar.on_hit(st)
            hit_delta = before_green - st.green_skulls
            if hit_delta > 0:
                hr = float(self.r_cfg.get("web_hit", -1.0))
                reward += hr
                self._reward_breakdown["web_hit"] = hr
            if died:
                reward += float(self.r_cfg.get("defeat", -25.0))
                self._reward_breakdown["defeat"] = float(self.r_cfg.get("defeat", -25.0))

        # 5. 기믹 틱 (제단 재등장 등)
        self._altar.tick(st, self.rng)

        # 6. 페이즈 갱신
        for i, ph in enumerate(self.profile.data.get("phases", []), start=1):
            lo, hi = ph["hp_pct"]
            if lo >= st.boss_hp_pct > hi:
                st.phase = i
                break

        # 7. 종료 판정
        if st.terminated:
            pass  # 이미 reward에 반영됨
        elif st.steps >= self.max_steps:
            # Pattern 2: truncated로 분리 — TimeLimit 래퍼 권장이지만
            # 프로파일 max_steps를 유지하기 위해 수동 처리
            terminated = False
            truncated = True

        # Pattern 2: terminated vs truncated 정확 분리
        terminated = st.terminated
        truncated = st.steps >= self.max_steps and not st.terminated
        if truncated:
            st.terminated = True
            st.termination_reason = "time_limit"

        # 8. 프레임 스택 + 관측
        frame = self._encode_frame()
        self.stacker.push(frame)
        obs = self.stacker.stacked()

        # 9. 잠재 쉐이핑
        phi = self._compute_phi(st)
        reward = self._shaping.apply(reward, phi)
        self._reward_breakdown["shaping"] = reward - self._reward_breakdown.get("survive", 0) - sum(
            v for k, v in self._reward_breakdown.items() if k != "shaping" and k != "survive"
        )

        info = self._info()
        info["reward_breakdown"] = dict(self._reward_breakdown)

        return (np.array(obs, dtype=np.float32), reward,
                terminated, truncated, info)

    def _encode_frame(self) -> list[float]:
        st = self.state
        fields = self._altar.observation_fields(st)
        fields["player_lane"] = st.player_lane / max(1, self.lane_count - 1)
        fields["boss_lane"] = st.boss_lane / max(1, self.lane_count - 1)
        fields["boss_hp_pct"] = st.boss_hp_pct / 100.0
        fields["bone_wave_active"] = float(st.bone_wave_active)
        fields["bone_wave_danger"] = float(
            st.player_lane in st.bone_wave_danger_lanes and st.bone_wave_active
        )
        fields["soul_split_seconds"] = min(1.0, st.soul_split_seconds_remaining / 180.0)
        # 제단 방향(부호 있음) + 즉시 상호작용 가능 비트 — v2 PM 수정:
        # 기존 altar_distance는 크기만 있어 방향 탐색이 불가능 → 정화 학습 실패의
        # 구조적 원인. 죽은 상수 필드(scythe_*)를 대체, 60차원 유지.
        if st.altar_present and st.altar_lane is not None:
            fields["altar_direction"] = (st.altar_lane - st.player_lane) / 6.0
            fields["altar_can_interact"] = float(abs(st.player_lane - st.altar_lane) <= 1)
        else:
            fields["altar_direction"] = 0.0
            fields["altar_can_interact"] = 0.0
        fields.setdefault("scythe_phase", 0.0)
        fields.setdefault("scythe_remaining", 1.0)

        spec = self.profile.observation_spec.get("fields", [])
        return [float(fields.get(f, 0.0)) for f in spec]

    def _compute_phi(self, st: BossGameState) -> float:
        return -self.phi_k * st.altar_distance

    @staticmethod
    def _soul_split_cycle(hp_pct: float) -> int:
        """보스 HP%별 영혼 베기 재시전 주기(초) — 실측 데이터."""
        from env.verus_hilla_settings import soul_split_cycle
        return soul_split_cycle(hp_pct, "hard")

    def _info(self) -> dict[str, Any]:
        st = self.state
        return {
            "steps": st.steps,
            "boss": self.profile.name,
            "phase": st.phase,
            "termination_reason": st.termination_reason,
            "observation_size": self.observation_space.shape[0],
            "green_skulls": st.green_skulls,
            "red_skulls": st.red_skulls,
            "danger_margin": st.danger_margin,
            "candles_lit": st.candles_lit,
            "total_cleansed": st.total_cleansed,
            "total_web_hits": st.total_web_hits,
            "soul_split_seconds": st.soul_split_seconds_remaining,
            "altar_present": st.altar_present,
            "altar_lane": st.altar_lane,
            "bone_wave_active": st.bone_wave_active,
            "boss_lane": st.boss_lane,
            "player_lane": st.player_lane,
            "red_thread_danger": sorted(st.thread_danger_lanes),
            "red_thread_telegraph": sorted(st.thread_telegraph_lanes),
            "bone_wave_danger": sorted(st.bone_wave_danger_lanes) if st.bone_wave_active else [],
        }

    def render(self):
        if self.state:
            st = self.state
            print(f"[{st.steps}] P:{st.player_lane} B:{st.boss_lane} "
                  f"G:{st.green_skulls} R:{st.red_skulls} "
                  f"🕯:{st.candles_lit} ⏳:{st.soul_split_seconds_remaining}s "
                  f"Altar:{'Y' if st.altar_present else 'N'}@{st.altar_lane}")
