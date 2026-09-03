import unittest

from env.jin_hilla_scenario_env import JinHillaScenarioEnv


class TestJinHillaScenarioEnv(unittest.TestCase):
    """Regression tests for M8 scenario-environment imports and rewards."""

    def test_import_and_reset_expose_soul_counts(self):
        env = JinHillaScenarioEnv(seed=42)
        obs = env.reset()
        self.assertEqual(obs["green_skulls"], 5)
        self.assertEqual(obs["red_skulls"], 0)

    def test_web_hit_penalty_is_negative(self):
        env = JinHillaScenarioEnv(seed=42)
        obs = env.reset()
        env.player_lane = obs["hazard_lane"]
        _, reward, _, _, _ = env.step(1)
        self.assertLess(reward, 0, "위험 레인 피격 시 보상은 음수여야 함")
        self.assertEqual(env.web_hits, 1, "피격 카운트가 1 이어야 함")

    def test_web_dodge_reward_is_positive(self):
        env = JinHillaScenarioEnv(seed=42)
        obs = env.reset()
        env.player_lane = (obs["hazard_lane"] + 1) % env.num_lanes
        _, reward, _, _, _ = env.step(1)
        self.assertGreater(reward, 0, "위험 레인 회피 시 보상은 양수여야 함")
        self.assertEqual(env.web_hits, 0, "피격 카운트가 0 이어야 함")


if __name__ == "__main__":
    unittest.main()
