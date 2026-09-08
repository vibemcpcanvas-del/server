# -*- coding: utf-8 -*-
"""관측 브리지 단위 테스트 — v7 관측 형식 정합성."""
import sys
import unittest

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")
import numpy as np
from core.observation_bridge import (
    ObservationBridge, parse_to_frame_fields, FRAME_DIM, FRAME_STACK)


def parsed(g=5, r=0, d=0, boss=True, player=True, dir_lanes=0.0):
    return {
        "is_bossfight": True, "green_skulls": g, "red_skulls": r,
        "skulls_destroyed": d, "skulls_seen": g + r,
        "player_found": player, "boss_found": boss,
        "boss_dir_lanes": dir_lanes, "ss_timer_visible": False,
    }


class TestParseToFields(unittest.TestCase):
    def test_full_health(self):
        f = parse_to_frame_fields(parsed())
        self.assertEqual(len(f), FRAME_DIM)
        self.assertAlmostEqual(f[0], 1.0)   # green 5/5
        self.assertAlmostEqual(f[1], 0.0)   # red 0
        self.assertAlmostEqual(f[2], 1.0)   # margin (5-0)/5

    def test_damage(self):
        f = parse_to_frame_fields(parsed(g=3, r=2))
        self.assertAlmostEqual(f[0], 0.6)
        self.assertAlmostEqual(f[1], 0.4)
        self.assertAlmostEqual(f[2], 0.2)

    def test_boss_direction(self):
        f = parse_to_frame_fields(parsed(dir_lanes=2.0))
        # boss_lane = 0.5 + 2/6 = 0.833
        self.assertAlmostEqual(f[9], 0.5 + 2.0 / 6.0, places=3)

    def test_missing_entities(self):
        f = parse_to_frame_fields(parsed(player=False, boss=False))
        self.assertAlmostEqual(f[8], 0.5)
        self.assertAlmostEqual(f[9], 0.5)

    def test_not_bossfight_returns_none(self):
        self.assertIsNone(parse_to_frame_fields({"is_bossfight": False}))


class TestObservationBridge(unittest.TestCase):
    def test_shape_60(self):
        b = ObservationBridge()
        obs = b.push(parsed())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape, (FRAME_DIM * FRAME_STACK,))
        self.assertTrue(b.is_ready())

    def test_none_keeps_stack(self):
        b = ObservationBridge()
        first = b.push(parsed())
        none_obs = b.push({"is_bossfight": False})
        self.assertIsNone(none_obs)
        self.assertTrue(b.is_ready())  # 스택 유지

    def test_update_flow(self):
        b = ObservationBridge()
        b.push(parsed())
        obs = b.push(parsed(g=4, r=1))
        # 마지막 프레임이 새 값
        self.assertAlmostEqual(float(obs[-FRAME_DIM]), 0.8)
        self.assertAlmostEqual(float(obs[-FRAME_DIM + 1]), 0.2)


if __name__ == "__main__":
    unittest.main()
