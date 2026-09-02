import unittest

from env.jin_hilla_environment_core import JinHillaState, ScytheCycle, ScythePhase, SoulState
from env.jin_hilla_training_env import Action, JinHillaTrainingEnv


class JinHillaCoreRulesTests(unittest.TestCase):
    def test_web_hit_converts_green_to_red_and_ends_when_red_exceeds_green(self):
        state = JinHillaState(SoulState(green=3, red=0), ScytheCycle(10, 2, 2))
        state.apply_web_hit()
        self.assertEqual((state.souls.green, state.souls.red), (2, 1))
        self.assertFalse(state.terminated)
        state.apply_web_hit()
        self.assertEqual((state.souls.green, state.souls.red), (1, 2))
        self.assertTrue(state.terminated)
        self.assertEqual(state.termination_reason, "red_skulls_exceed_green_skulls")

    def test_altar_requires_nearby_player_and_restores_red_skulls(self):
        state = JinHillaState(
            souls=SoulState(green=3, red=2),
            scythe=ScytheCycle(10, 2, 2),
            altar_present=True,
            altar_lane=5,
            player_lane=2,
        )
        self.assertEqual(state.harvest_altar(20), 0)
        self.assertEqual((state.souls.green, state.souls.red), (3, 2))
        state.player_lane = 4
        self.assertEqual(state.harvest_altar(20), 2)
        self.assertEqual((state.souls.green, state.souls.red), (5, 0))
        self.assertFalse(state.altar_present)

    def test_scythe_cycle_has_warning_before_spread_art(self):
        cycle = ScytheCycle(countdown_ticks=2, warning_ticks=1, spread_art_ticks=1)
        self.assertIsNone(cycle.tick())
        self.assertEqual(cycle.tick(), ScythePhase.WARNING)
        self.assertEqual(cycle.tick(), ScythePhase.SPREAD_ART)
        self.assertEqual(cycle.tick(), ScythePhase.COUNTDOWN)


class JinHillaTrainingEnvironmentTests(unittest.TestCase):
    def test_observation_actions_and_episode_limit(self):
        env = JinHillaTrainingEnv(max_steps=2, scythe_ticks=(3, 1, 1))
        observation, info = env.reset(options={"green_skulls": 5, "red_skulls": 0, "player_lane": 0})
        self.assertEqual(len(observation), env.observation_size)
        self.assertEqual(info["player_lane"], 0)
        _, _, terminated, truncated, info = env.step(Action.LEFT)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["player_lane"], 0)
        _, _, terminated, truncated, _ = env.step(Action.RIGHT)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_harvest_rewards_cleanse(self):
        env = JinHillaTrainingEnv()
        env.reset(options={"green_skulls": 3, "red_skulls": 2, "player_lane": 3})
        env.spawn_altar(3)
        _, reward, terminated, _, info = env.step(Action.HARVEST)
        self.assertGreater(reward, 0)
        self.assertFalse(terminated)
        self.assertEqual((info["green_skulls"], info["red_skulls"]), (4, 1))


if __name__ == "__main__":
    unittest.main()
