import unittest
import numpy as np

from env.jin_hilla_environment_core import JinHillaState, ScytheCycle, ScythePhase, SoulState
from env.jin_hilla_training_env import Action, JinHillaTrainingEnv
from env.jin_hilla_scenario_env import JinHillaScenarioEnv
from env.jin_hilla_gym_env import JinHillaScenarioGymEnv


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
        state = JinHillaState(\
            souls=SoulState(green=3, red=2),\
            scythe=ScytheCycle(10, 2, 2),\
            altar_present=True,\
            altar_lane=5,\
            player_lane=2,\
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


class JinHillaScenarioAltarRuleTests(unittest.TestCase):
    def _prepare_env(self, green_skulls: int) -> tuple[JinHillaScenarioEnv, JinHillaState]:
        env = JinHillaScenarioEnv()
        state = env._require_state()
        state.souls.green = green_skulls
        state.souls.red = max(0, 5 - green_skulls)
        env._on_soul_slash()
        return env, state

    def test_altar_hits_needed_mapping(self):
        env, _ = self._prepare_env(5)
        self.assertEqual(env.altar_hits_needed, 3)
        env, _ = self._prepare_env(4)
        self.assertEqual(env.altar_hits_needed, 2)
        env, _ = self._prepare_env(3)
        self.assertEqual(env.altar_hits_needed, 2)
        env, _ = self._prepare_env(2)
        self.assertEqual(env.altar_hits_needed, 1)

    def test_altar_spawns_after_required_hits_for_5_green(self):
        env, state = self._prepare_env(5)
        self.assertFalse(env.altar_active)
        for _ in range(2):
            env._register_hit_for_altar()
            self.assertFalse(env.altar_active)
        env._register_hit_for_altar()
        self.assertTrue(env.altar_active)
        self.assertTrue(state.altar_present)
        self.assertIsNotNone(state.altar_lane)


class JinHillaGymEnvTests(unittest.TestCase):
    def test_gym_interface_compliance(self):
        env = JinHillaScenarioGymEnv(seed=42)
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (7,))
        self.assertEqual(env.action_space.n, 4)

        next_obs, reward, terminated, truncated, info = env.step(1)
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


if __name__ == "__main__":
    unittest.main()
