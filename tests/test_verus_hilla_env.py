"""Verus Hilla 훈련 환경 v2 단위 테스트 — 프레임 스태킹 + 실측 기믹.

실행: python -m unittest tests.test_verus_hilla_env -v
"""
from __future__ import annotations

import unittest

from env.jin_hilla_verus_env import VerusHillaTrainingEnv, Action
from env.verus_observation import FRAME_STACK, FRAME_DIM, OBSERVATION_SIZE


class ObservationShapeTests(unittest.TestCase):
    """관측 크기/구조 검증."""

    def setUp(self):
        self.env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=500)

    def test_observation_size_is_60(self):
        self.assertEqual(self.env.observation_size, 60)
        self.assertEqual(FRAME_STACK * FRAME_DIM, 60)

    def test_reset_returns_stacked_frames(self):
        obs, _ = self.env.reset(seed=42)
        self.assertEqual(len(obs), 60)
        # reset 시 4프레임이 동일 값으로 채워짐
        self.assertEqual(obs[:15], obs[15:30])

    def test_frames_diverge_after_steps(self):
        self.env.reset(seed=42)
        obs0 = self.env.observation()
        self.env.step(int(Action.RIGHT))
        obs1 = self.env.observation()
        # 플레이어 레인이 변했으면 최신 프레임이 달라져야 함
        self.assertNotEqual(obs0[-15:], obs1[-15:])
        # 오래된 프레임은 유지 (스택 왼쪽 시프트)
        self.assertEqual(obs0[-30:-15], obs1[-45:-30])


class FrameStackTests(unittest.TestCase):
    """프레임 스택 동작 검증."""

    def setUp(self):
        self.env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=500)
        self.env.reset(seed=42)

    def test_velocity_detection(self):
        """이동 후 프레임 간 플레이어 레인 변화가 감지되어야 함."""
        before = self.env.observation()
        lane_before = before[-15 + 8]  # 최신 프레임의 player_lane
        self.env.step(int(Action.RIGHT))
        after = self.env.observation()
        lane_after = after[-15 + 8]
        self.assertNotEqual(lane_before, lane_after)

    def test_max_frames_capped(self):
        for _ in range(10):
            self.env.step(int(Action.STAY))
        self.assertEqual(len(self.env.frame_stack.frames), FRAME_STACK)


class SoulSplitIntegrationTests(unittest.TestCase):
    """Soul Split이 환경에서 실제로 발동되는지 통합 검증."""

    def test_soul_split_fires_and_terminates(self):
        # 빨간 해골 3개 상태에서 Soul Split → green 5-3=2, 사망 안 함
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, soul_split_mode="hard")
        obs, _ = env.reset(seed=7, options={"green_skulls": 5, "red_skulls": 3})
        env.state.soul_split._next_at_sec = 1  # 즉시 발동
        env.state.soul_split.elapsed_ticks = 60  # 1초 경과
        env.step(int(Action.STAY))
        # 빨간 3개 파괴 → green 2
        self.assertEqual(env.state.souls.red, 0)
        self.assertEqual(env.state.souls.green, 2)
        self.assertFalse(env.state.terminated)

    def test_soul_split_can_kill(self):
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, soul_split_mode="hard")
        obs, _ = env.reset(seed=7, options={"green_skulls": 1, "red_skulls": 3})
        env.state.soul_split._next_at_sec = 1
        env.state.soul_split.elapsed_ticks = 60
        env.step(int(Action.STAY))
        self.assertTrue(env.state.terminated)
        self.assertEqual(env.state.termination_reason, "soul_split_death")


class AltarInterferenceTests(unittest.TestCase):
    """보스의 제단 방해 기믹."""

    def setUp(self):
        self.env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000)
        self.env.reset(seed=7)

    def test_boss_despawns_altar(self):
        st = self.env.state
        st.altar_present = True
        st.altar_lane = st.boss_lane  # 보스 위치에 제단
        st.boss_touch_altar()
        self.assertFalse(st.altar_present)
        self.assertGreater(st.altar_respawn_ticks, 0)

    def test_altar_respawns_in_valid_zone(self):
        st = self.env.state
        st.altar_present = True
        st.boss_touch_altar()
        for _ in range(st.altar_respawn_ticks + 1):
            st.tick()
        self.assertTrue(st.altar_present)
        # 재등장 레인은 1~5 (설정: 2~6구역 1기반 → 0기반 1~5)
        self.assertIn(st.altar_lane, [1, 2, 3, 4, 5])


class BoneWaveIntegrationTests(unittest.TestCase):
    """뼈 파동이 환경에서 발동되고 피해를 주는지."""

    def test_bone_wave_triggers_on_proximity(self):
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000)
        env.reset(seed=7)
        st = env.state
        st.phase = 2
        st.player_lane = 3
        st.boss_lane = 3
        wave = st.maybe_trigger_bone_wave()
        self.assertIsNotNone(wave)

    def test_bone_wave_damages_player_in_danger_lane(self):
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000)
        env.reset(seed=7)
        st = env.state
        st.phase = 2
        st.player_lane = 3
        st.boss_lane = 3
        st.souls = type(st.souls)(green=5, red=0)
        wave = st.maybe_trigger_bone_wave()
        assert wave is not None
        before_green = st.souls.green
        # 위험 레인에 강제 배치
        danger = wave.danger_lanes()
        st.player_lane = danger.pop()
        # 틱 진행 — 발동 프레임 통과
        for _ in range(wave.ticks_until_hit + 2):
            st.bone_wave.tick()
            if st.player_lane in wave.danger_lanes():
                st.apply_web_hit()
        self.assertLess(st.souls.green, before_green)


class M8CompatTests(unittest.TestCase):
    """M8 인터페이스 호환 — 기존 시나리오 이벤트 재현."""

    def setUp(self):
        self.env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000)
        self.env.reset(seed=7)

    def test_spawn_altar_event(self):
        self.env.spawn_altar(3)
        self.assertTrue(self.env.state.altar_present)
        self.assertEqual(self.env.state.altar_lane, 3)

    def test_apply_web_hit_event(self):
        before = self.env.state.souls.green
        self.env.apply_web_hit()
        self.assertEqual(self.env.state.souls.green, before - 1)

    def test_four_actions(self):
        self.assertEqual(self.env.action_size, 4)

    def test_info_contract(self):
        obs, info = self.env.reset(seed=7)
        for key in ("green_skulls", "red_skulls", "steps", "observation_size"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
