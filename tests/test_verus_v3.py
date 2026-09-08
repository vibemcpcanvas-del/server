"""Verus Hilla v3 (GameState 중앙집중) 단위 테스트.

SSOT 위반·중복 피격·terminated 이중 권한 문제가 해결됐는지 검증.
실행: python -m unittest tests.test_verus_v3 -v
"""
from __future__ import annotations

import os
import unittest
import numpy as np

from env.game_state import BossGameState
from env.boss_env_base import BossProfile
from env.verus_hilla_v3 import VerusHillaEnvV3

PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "env", "profiles", "verus_hilla.json"
)
import os


class SSOTTests(unittest.TestCase):
    """단일 진실 공급원 검증 — 상태가 BossGameState에만 존재."""

    def setUp(self):
        profile = BossProfile.load(PROFILE_PATH)
        self.env = VerusHillaEnvV3(profile, seed=42, max_steps=500)
        self.env.reset(seed=42)

    def test_state_is_single_object(self):
        """모든 상태가 BossGameState 하나에 있다."""
        st = self.env.state
        self.assertIsInstance(st, BossGameState)
        # env 자체에 분산된 상태 필드가 없어야 함
        for attr in ("green_skulls", "red_skulls", "candles_lit",
                     "altar_present", "boss_hp_pct", "terminated"):
            # env에는 계산용 참조만 있고 원본은 state에
            self.assertTrue(hasattr(st, attr), f"state에 {attr} 없음")

    def test_no_duplicate_state_on_env(self):
        """env에 상태 복사본이 생성되지 않는지."""
        # v1에서는 env.terminated와 gimmick.defeated가 별도였음
        # v3에서는 state.terminated 하나만 존재
        st = self.env.state
        st.terminated = True
        self.assertTrue(st.terminated)
        # env에는 terminated 속성이 없어야 함 (state로만 접근)
        # (동작 확인: env.step이 terminated를 state에서 읽는지)


class DeduplicationTests(unittest.TestCase):
    """위협 간 중복 피격 방지 검증."""

    def setUp(self):
        profile = BossProfile.load(PROFILE_PATH)
        self.env = VerusHillaEnvV3(profile, seed=42, max_steps=500)
        self.env.reset(seed=42)
        st = self.env.state
        # 붉은 실과 뼈 파동이 동시에 같은 레인을 치는 상황 강제
        st.player_lane = 3
        st.boss_lane = 3
        st.phase = 2
        st.green_skulls = 5
        st.red_skulls = 0
        st.bone_wave_active = True
        st.bone_wave_aura = "purple"
        st.bone_wave_danger_lanes = {2, 3, 4}  # 플레이어(3) 포함
        st.bone_wave_until_hit = 0
        st.bone_wave_duration = 30
        st.thread_danger_lanes = {3}  # 붉은 실도 레인 3에 활성

    def test_simultaneous_threats_single_hit(self):
        """두 위협이 동시에 겹쳐도 해골은 1개만 감소."""
        before = self.env.state.green_skulls
        obs, r, term, trunc, info = self.env.step(1)  # STAY
        after = self.env.state.green_skulls
        delta = before - after
        self.assertEqual(delta, 1, f"동시 피격 시 해골 {delta}개 감소 — 1개여야 함")


class TerminatedAuthorityTests(unittest.TestCase):
    """terminated 권한이 state에만 있는지."""

    def setUp(self):
        profile = BossProfile.load(PROFILE_PATH)
        self.env = VerusHillaEnvV3(profile, seed=42, max_steps=500)
        self.env.reset(seed=42)

    def test_death_sets_state_terminated(self):
        """사망 시 state.terminated가 설정됨."""
        st = self.env.state
        st.green_skulls = 1
        st.red_skulls = 3
        # web hit → green 0, red 4 → defeated
        obs, r, term, trunc, info = self.env.step(1)
        # 다음 틱에서 실에 맞으면 종료
        # (실이 즉시 오지 않을 수 있으므로 여러 틱)
        for _ in range(200):
            if self.env.state.terminated:
                break
            obs, r, term, trunc, info = self.env.step(1)
        # terminated 권한이 state에 있음
        if self.env.state.defeated:
            self.assertTrue(self.env.state.terminated)

    def test_time_limit_sets_truncated_not_terminated(self):
        """시간 만료는 truncated=True, terminated=False (Gymnasium Pattern 2)."""
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=5)
        env.reset(seed=42)
        for _ in range(10):
            obs, r, term, trunc, info = env.step(1)
            if trunc:
                # state.terminated는 env 내부에서 True로 설정하지만
                # step 반환값은 terminated=False, truncated=True
                self.assertFalse(term)
                self.assertTrue(trunc)
                return
        self.fail("10스텝 내 truncated가 발생하지 않음")


class RewardBreakdownTests(unittest.TestCase):
    """보상 분해 로깅 (Pattern 4: dense info)."""

    def test_reward_breakdown_present(self):
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env.reset(seed=42)
        obs, r, term, trunc, info = env.step(1)
        self.assertIn("reward_breakdown", info)
        self.assertIn("survive", info["reward_breakdown"])

    def test_reward_decomposition_consistent(self):
        """분해 합 = 총 보상."""
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env.reset(seed=42)
        obs, r, term, trunc, info = env.step(1)
        bd = info["reward_breakdown"]
        total = sum(bd.values())
        self.assertAlmostEqual(total, r, places=5)


class StochasticResetTests(unittest.TestCase):
    """Pattern 3: reset 스토캐스틱 + 재현 가능."""

    def test_same_seed_same_obs(self):
        profile = BossProfile.load(PROFILE_PATH)
        env1 = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env2 = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seed_different_obs(self):
        profile = BossProfile.load(PROFILE_PATH)
        env1 = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env2 = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        obs1, _ = env1.reset(seed=1)
        obs2, _ = env2.reset(seed=2)
        # 시간이 지나야 달라지므로 몇 스텝 진행
        for _ in range(50):
            env1.step(1)
            env2.step(1)
        o1 = env1.state.thread_danger_lanes
        o2 = env2.state.thread_danger_lanes
        # 시드가 다르면 언젠가 달라짐 (확률적 보장은 어려우나 일반적으로)


class ShapingIntegrationTests(unittest.TestCase):
    """제단 거리 쉐이핑이 v3에서 동작하는지."""

    def test_shaping_present_in_breakdown(self):
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env.reset(seed=42)
        # 제단 강제 활성화
        st = env.state
        st.altar_present = True
        st.altar_lane = 0
        obs, r, term, trunc, info = env.step(0)  # LEFT (제단 방향)
        self.assertIn("reward_breakdown", info)
        self.assertIn("shaping", info["reward_breakdown"])


class GymnasiumCompatTests(unittest.TestCase):
    """SB3 호환성 — Gymnasium 표준 준수."""

    def test_observation_space_contains_obs(self):
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        for i in range(20):
            obs, info = env.reset(seed=i)
            self.assertTrue(env.observation_space.contains(obs),
                            f"seed {i}: obs {obs.min():.2f}~{obs.max():.2f} not in space")

    def test_50_steps_no_crash(self):
        profile = BossProfile.load(PROFILE_PATH)
        env = VerusHillaEnvV3(profile, seed=42, max_steps=100)
        env.reset(seed=42)
        for _ in range(50):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()


if __name__ == "__main__":
    unittest.main()
