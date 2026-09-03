import unittest

from env.jin_hilla_scenario_env import JinHillaScenarioEnv


class TestJinHillaScenarioEnv(unittest.TestCase):
    def assert_numeric_observation(self, observation, env):
        self.assertIsInstance(observation, list)
        self.assertEqual(len(observation), env.observation_size)
        self.assertTrue(all(isinstance(value, float) for value in observation))

    def test_reset_returns_numeric_policy_observation(self):
        env = JinHillaScenarioEnv(seed=42)
        self.assert_numeric_observation(env.reset(), env)

    def test_step_returns_numeric_policy_observation_and_evaluation_info(self):
        env = JinHillaScenarioEnv(seed=42)
        env.reset()
        observation, _, _, _, info = env.step(1)
        self.assert_numeric_observation(observation, env)
        self.assertTrue({"green_skulls", "red_skulls", "cleanse_count", "dodge_count"}.issubset(info))

    def test_web_hit_penalty_is_negative(self):
        env = JinHillaScenarioEnv(seed=42)
        env.reset()
        env.player_lane = env.hazard_lane
        _, reward, _, _, _ = env.step(1)
        self.assertLess(reward, 0)
        self.assertEqual(env.web_hits, 1)

    def test_web_dodge_reward_is_positive(self):
        env = JinHillaScenarioEnv(seed=42)
        env.reset()
        env.player_lane = (env.hazard_lane + 1) % env.num_lanes
        _, reward, _, _, _ = env.step(1)
        self.assertGreater(reward, 0)
        self.assertEqual(env.web_hits, 0)


if __name__ == "__main__":
    unittest.main()
