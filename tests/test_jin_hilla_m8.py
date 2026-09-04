import unittest

from env.jin_hilla_environment_core import JinHillaState, ScytheCycle, ScythePhase, SoulState
from env.jin_hilla_training_env import Action, JinHillaTrainingEnv
from env.jin_hilla_scenario_env import JinHillaScenarioEnv


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


class JinHillaScenarioAltarRuleTests(unittest.TestCase):
    def _prepare_env(self, green_skulls: int) -> tuple[JinHillaScenarioEnv, JinHillaState]:
        env = JinHillaScenarioEnv()
        state = env._require_state()
        state.souls.green = green_skulls
        state.souls.red = max(0, 5 - green_skulls)
        env._on_soul_slash()
        return env, state

    def test_altar_hits_needed_mapping(self):
        # max(1, green // 2): a disclosed deviation from the real-game
        # ceil(green / 2) formula, needed to guarantee a one-hit buffer
        # before the lethal hit under this simulator's simplified defeat rule.
        env, _ = self._prepare_env(5)
        self.assertEqual(env.altar_hits_needed, 2)
        env, _ = self._prepare_env(4)
        self.assertEqual(env.altar_hits_needed, 2)
        env, _ = self._prepare_env(3)
        self.assertEqual(env.altar_hits_needed, 1)
        env, _ = self._prepare_env(2)
        self.assertEqual(env.altar_hits_needed, 1)

    def test_altar_spawns_before_lethal_hit_for_5_green(self):
        env, state = self._prepare_env(5)
        self.assertFalse(env.altar_active)
        env._register_hit_for_altar()
        self.assertFalse(env.altar_active)
        env._register_hit_for_altar()
        self.assertTrue(env.altar_active)
        self.assertFalse(state.terminated)
        self.assertTrue(state.altar_present)
        self.assertIsNotNone(state.altar_lane)

    def test_altar_spawns_before_lethal_hit_for_3_and_4_green(self):
        for green in (3, 4):
            env, state = self._prepare_env(green)
            needed = env.altar_hits_needed
            for _ in range(needed - 1):
                env._register_hit_for_altar()
                self.assertFalse(env.altar_active)
            env._register_hit_for_altar()
            self.assertTrue(env.altar_active)
            self.assertFalse(state.terminated)

    def test_altar_spawns_before_lethal_hit_for_2_green(self):
        env, state = self._prepare_env(2)
        self.assertFalse(env.altar_active)
        env._register_hit_for_altar()
        self.assertTrue(env.altar_active)
        self.assertFalse(state.terminated)

    def test_altar_never_spawns_for_one_green(self):
        env, state = self._prepare_env(1)
        self.assertIsNone(env.altar_hits_needed)
        self.assertFalse(env.can_spawn_altar)
        for _ in range(5):
            env._register_hit_for_altar()
        self.assertFalse(env.altar_active)
        self.assertFalse(state.altar_present)

    def test_altar_threshold_is_active_immediately_after_reset(self):
        env = JinHillaScenarioEnv()
        env.reset()
        self.assertIsNotNone(env.altar_hits_needed)
        self.assertEqual(env.altar_hits_needed, 2)  # max(1, 5 // 2)
        self.assertTrue(env.can_spawn_altar)

    def test_altar_spawns_with_margin_before_death_after_reset(self):
        """Regression test: the altar must spawn strictly before the lethal
        hit so a policy has at least one step of opportunity to harvest it."""
        env = JinHillaScenarioEnv()
        state = env.reset() and env._require_state()
        state = env._require_state()
        env._register_hit_for_altar()
        state.apply_web_hit()
        self.assertFalse(state.terminated)
        env._register_hit_for_altar()
        self.assertTrue(env.altar_active)


if __name__ == "__main__":
    unittest.main()
