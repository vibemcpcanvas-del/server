import unittest

from env.jin_hilla_scenario_env import JinHillaScenarioEnv


class TestJinHillaScenarioEnv(unittest.TestCase):
    """M8 시나리오 환경의 피격 보상 부호 회귀 테스트."""

    def test_web_hit_penalty_is_negative(self):
        """위험 레인에 있을 때 보상이 음수여야 한다."""
        env = JinHillaScenarioEnv(seed=42)
        obs = env.reset()

        # 플레이어와 같은 레인이 위험 레인이 되도록 강제 설정
        obs["player_lane"] = obs["hazard_lane"]
        env.player_lane = obs["player_lane"]

        # STAY 액션으로 한 스텝 진행 (위험 레인 유지)
        _, reward, terminated, truncated, info = env.step(1)

        # 피격 패널티 (-5.0) + 생존 보상 (+0.05) = -4.95 근방
        self.assertLess(reward, 0, "위험 레인 피격 시 보상은 음수여야 함")
        self.assertEqual(env.core.web_hits, 1, "피격 카운트가 1 이어야 함")

    def test_web_dodge_reward_is_positive(self):
        """위험 레인 밖에 있을 때 보상이 양수여야 한다."""
        env = JinHillaScenarioEnv(seed=42)
        obs = env.reset()

        # 플레이어를 위험 레인과 다른 레인으로 이동
        original_hazard = obs["hazard_lane"]
        new_player_lane = (original_hazard + 1) % env.num_lanes
        env.player_lane = new_player_lane

        # STAY 액션으로 한 스텝 진행 (위험 레인 아님)
        _, reward, terminated, truncated, info = env.step(1)

        # 회피 보상 (+1.0) + 생존 보상 (+0.05) = +1.05 근방
        self.assertGreater(reward, 0, "위험 레인 회피 시 보상은 양수여야 함")
        self.assertEqual(env.core.web_hits, 0, "피격 카운트가 0 이어야 함")


if __name__ == "__main__":
    unittest.main()
