"""Verus Hilla (진힐라) 실측 코어 단위 테스트.

test_jin_hilla_m8.py의 M8 규약과의 호환성 + 진힐라 신규 기믹 검증.
실행: python -m unittest tests.test_verus_hilla_core -v
"""
from __future__ import annotations

import unittest

from env.jin_hilla_verus_core import (
    BoneWave,
    SoulState,
    SoulSplitTimer,
    VerusHillaState,
)
from env.verus_hilla_settings import SETTINGS, x_to_lane, lane_center_x, soul_split_cycle


class MapGeometryTests(unittest.TestCase):
    """실측 맵 데이터 검증 (assets/jinhilla_map_db.json)."""

    def test_lane_count_is_seven(self):
        self.assertEqual(SETTINGS()["map"]["lane_count"], 7)

    def test_map_width_matches_7_lanes(self):
        x_min, x_max = SETTINGS()["map"]["x_min"], SETTINGS()["map"]["x_max"]
        self.assertAlmostEqual(x_max - x_min, 1980, delta=1)

    def test_x_to_lane_round_trip(self):
        x_min = SETTINGS()["map"]["x_min"]
        w = (x_max := SETTINGS()["map"]["x_max"]) - x_min
        lane_w = w / 7
        # 각 레인 중심 → 레인 인덱스 복원
        for lane in range(7):
            cx = x_min + (lane + 0.5) * lane_w
            self.assertEqual(x_to_lane(cx), lane)

    def test_lane_clamping(self):
        self.assertEqual(x_to_lane(-99999), 0)
        self.assertEqual(x_to_lane(99999), 6)


class SoulSplitTests(unittest.TestCase):
    """영혼 베기(Soul Split) — 낫배기성 즉사 기믹."""

    def test_hard_first_timing(self):
        cfg = SETTINGS()["soul_split"]["hard"]
        self.assertEqual(cfg["first_at_sec"], 150)  # 입장 2:30

    def test_hard_cycle_by_hp(self):
        # 61.1% 초과 → 152초
        self.assertEqual(soul_split_cycle(70.0, "hard"), 152)
        # 31.1~61% → 126초
        self.assertEqual(soul_split_cycle(45.0, "hard"), 126)
        # 31% 이하 → 100초
        self.assertEqual(soul_split_cycle(20.0, "hard"), 100)

    def test_soul_split_destroys_all_red_skulls(self):
        st = VerusHillaState(souls=SoulState(green=3, red=2))
        st._apply_soul_split()
        self.assertEqual(st.souls.red, 0)
        self.assertEqual(st.souls.green, 1)  # 3 - 2

    def test_soul_split_can_kill_when_green_exhausted(self):
        st = VerusHillaState(souls=SoulState(green=1, red=1))
        st._apply_soul_split()
        self.assertTrue(st.terminated)
        self.assertEqual(st.termination_reason, "soul_split_death")

    def test_soul_split_noop_when_no_red(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        st._apply_soul_split()
        self.assertEqual(st.souls.green, 5)
        self.assertFalse(st.terminated)

    def test_timer_fires_after_scheduled_seconds(self):
        timer = SoulSplitTimer(mode="hard", ticks_per_second=60.0)
        timer._next_at_sec = 150
        fired = False
        for _ in range(150 * 60):
            if timer.tick():
                fired = True
                break
        self.assertTrue(fired)

    def test_timer_reschedules_by_hp(self):
        timer = SoulSplitTimer(mode="hard", boss_hp_pct=45.0, ticks_per_second=60.0)
        timer._next_at_sec = 150
        timer.elapsed_ticks = 150 * 60  # 150초 경과 상태로 점프
        self.assertTrue(timer.tick())   # 발동 → 재예약
        # 45% → 126초 주기
        self.assertEqual(timer.seconds_remaining, 126)


class AltarTests(unittest.TestCase):
    """제단 기믹 — M8 호환 + 보스 방해."""

    def test_m8_harvest_contract(self):
        """M8 규약: 제단 근처에서 수집키 연타 → 빨간 해골 정화."""
        st = VerusHillaState(souls=SoulState(green=3, red=2))
        st.altar_present = True
        st.altar_lane = 3
        st.player_lane = 3
        cleaned = st.harvest_altar(presses=20, presses_per_cleanse=10)
        self.assertEqual(cleaned, 2)
        self.assertEqual(st.souls.red, 0)
        self.assertFalse(st.altar_present)  # 전부 정화 → 제단 소멸

    def test_altar_spawn_zones_exclude_edges(self):
        zones = SETTINGS()["altar"]["spawn_zones"]
        self.assertNotIn(1, zones)
        self.assertNotIn(7, zones)
        self.assertEqual(sorted(zones), [2, 3, 4, 5, 6])

    def test_boss_interference_despawns_altar(self):
        st = VerusHillaState(souls=SoulState(green=3, red=2))
        st.altar_present = True
        st.altar_lane = 3
        st.boss_touch_altar()
        self.assertFalse(st.altar_present)
        self.assertGreater(st.altar_respawn_ticks, 0)

    def test_altar_respawns_after_interference(self):
        st = VerusHillaState(souls=SoulState(green=3, red=2))
        st.altar_present = True
        st.boss_touch_altar()
        # 재등장 대기 시간 소진
        for _ in range(st.altar_respawn_ticks + 1):
            st.tick()
        self.assertTrue(st.altar_present)
        self.assertIsNotNone(st.altar_lane)


class CandleTests(unittest.TestCase):
    """촛불 카운터 — 영혼 뺏길 때마다 점등, 전부 점등 시 제단 등장."""

    def test_web_hit_lights_candle(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0), candle_threshold=3)
        before = st.candles_lit
        st.apply_web_hit()
        self.assertEqual(st.candles_lit, before + 1)

    def test_altar_spawns_when_all_candles_lit(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0), candle_threshold=3)
        self.assertFalse(st.altar_present)
        st.apply_web_hit()
        st.apply_web_hit()
        st.apply_web_hit()
        self.assertTrue(st.altar_present)  # 3개 촛불 → 제단 등장


class BoneWaveTests(unittest.TestCase):
    """뼈 파동 — 초록(앞 안전) / 보라(뒤 안전)."""

    def test_green_safe_is_front(self):
        self.assertEqual(SETTINGS()["bone_wave"]["green"]["safe"], "front")

    def test_purple_safe_is_back(self):
        self.assertEqual(SETTINGS()["bone_wave"]["purple"]["safe"], "back")

    def test_green_danger_excludes_front_lane(self):
        bw = BoneWave(aura="green", boss_lane=3)
        danger = bw.danger_lanes()
        self.assertIn(3, danger)          # 보스 위치는 위험
        self.assertNotIn(4, danger)       # 보스 앞(오른쪽 진행방향) 안전
        self.assertIn(2, danger)          # 뒤는 위험

    def test_purple_danger_excludes_back_lane(self):
        bw = BoneWave(aura="purple", boss_lane=3)
        danger = bw.danger_lanes()
        self.assertIn(3, danger)
        self.assertNotIn(2, danger)       # 보스 뒤 안전
        self.assertIn(4, danger)          # 앞은 위험

    def test_bone_wave_requires_phase_2(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        st.player_lane = 3
        st.boss_lane = 3
        st.phase = 1
        self.assertIsNone(st.maybe_trigger_bone_wave())
        st.phase = 2
        self.assertIsNotNone(st.maybe_trigger_bone_wave())

    def test_bone_wave_requires_proximity(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        st.phase = 2
        st.player_lane = 0
        st.boss_lane = 6
        self.assertIsNone(st.maybe_trigger_bone_wave())


class PhaseTests(unittest.TestCase):
    """HP% → 페이즈 전환."""

    def test_phase_transitions(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        st.set_boss_hp(100.0)
        self.assertEqual(st.phase, 1)
        st.set_boss_hp(80.0)
        self.assertEqual(st.phase, 1)
        st.set_boss_hp(70.0)
        self.assertEqual(st.phase, 2)
        st.set_boss_hp(40.0)
        self.assertEqual(st.phase, 3)
        st.set_boss_hp(10.0)
        self.assertEqual(st.phase, 4)


class M8CompatibilityTests(unittest.TestCase):
    """M8 규약과의 호환성 — 기존 테스트 시나리오 재현."""

    def test_web_hit_changes_green_to_red(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        st.apply_web_hit()
        self.assertEqual((st.souls.green, st.souls.red), (4, 1))

    def test_termination_on_red_majority(self):
        st = VerusHillaState(souls=SoulState(green=2, red=3))
        st.apply_web_hit()  # green 1, red 4
        self.assertTrue(st.terminated)

    def test_observation_contract(self):
        st = VerusHillaState(souls=SoulState(green=5, red=0))
        obs = st.observation()
        for key in ("green_skulls", "red_skulls", "danger_margin",
                    "altar_present", "altar_lane", "player_lane",
                    "boss_lane", "boss_hp_pct", "phase", "candles_lit",
                    "soul_split_seconds", "bone_wave_active", "bone_wave_aura"):
            self.assertIn(key, obs)


if __name__ == "__main__":
    unittest.main()
