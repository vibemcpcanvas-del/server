import unittest

from core.vision_schema import BossBattleState
from core.priority_arbiter import Action, PriorityArbiter


class PriorityArbiterTests(unittest.TestCase):
    def setUp(self):
        self.arbiter = PriorityArbiter(danger_red_threshold=2)

    def test_default_delegates_to_l1_dodge_when_safe(self):
        # 빨강 해골 0개, 제단 없음 -> PPO 회피 액션(LEFT) 그대로 채택
        state = BossBattleState(
            player_lane=0.5,
            hazard_lane=0.6,
            altar_lane=0.2,
            altar_active=False,
            green_skulls=5,
            red_skulls=0,
        )
        action = self.arbiter.arbitrate(state, l1_dodge_action=Action.LEFT)
        self.assertEqual(action, Action.LEFT)

    def test_l3_intervenes_when_red_skulls_critical_and_altar_active(self):
        # 빨강 해골 3개, 제단 활성, 플레이어(0.2)가 제단(0.8)보다 왼쪽 -> 회피 명령 무시하고 RIGHT로 제단 접근
        state = BossBattleState(
            player_lane=0.2,
            hazard_lane=0.5,
            altar_lane=0.8,
            altar_active=True,
            green_skulls=2,
            red_skulls=3,
        )
        action = self.arbiter.arbitrate(state, l1_dodge_action=Action.LEFT)
        self.assertEqual(action, Action.RIGHT)

    def test_l3_triggers_harvest_when_close_to_altar(self):
        # 플레이어가 제단 바로 옆 레인에 도착 -> HARVEST 발동
        state = BossBattleState(
            player_lane=0.5,
            hazard_lane=0.2,
            altar_lane=0.5,
            altar_active=True,
            green_skulls=2,
            red_skulls=3,
        )
        action = self.arbiter.arbitrate(state, l1_dodge_action=Action.STAY)
        self.assertEqual(action, Action.HARVEST)

    def test_l2_escapes_left_corner_trap(self):
        # 플레이어가 맨 왼쪽(0.0)에 몰렸고 오른쪽이 안전할 때 중앙 복귀 바이어스
        state = BossBattleState(
            player_lane=0.0,
            hazard_lane=0.5,
            altar_lane=0.5,
            altar_active=False,
            green_skulls=5,
            red_skulls=0,
        )
        action = self.arbiter.arbitrate(state, l1_dodge_action=Action.STAY)
        self.assertEqual(action, Action.RIGHT)


if __name__ == "__main__":
    unittest.main()
