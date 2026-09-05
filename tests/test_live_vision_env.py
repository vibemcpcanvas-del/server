import unittest
import numpy as np

from env.jin_hilla_gym_env import JinHillaScenarioGymEnv
from env.jin_hilla_live_vision_env import JinHillaVisionObservationWrapper
from core.vision.screen_source import SyntheticScreenSource


class JinHillaLiveVisionEnvTests(unittest.TestCase):
    def test_gymnasium_observation_wrapper_compliance(self):
        base_env = JinHillaScenarioGymEnv()
        wrapper_env = JinHillaVisionObservationWrapper(base_env, screen_source=SyntheticScreenSource())

        # Observation Space 규격 확인 (Box(7,))
        self.assertEqual(wrapper_env.observation_space.shape, (7,))
        self.assertEqual(wrapper_env.observation_space.dtype, np.float32)

        # Reset 시 관측값 검증
        obs, info = wrapper_env.reset()
        self.assertEqual(obs.shape, (7,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))

        # Step 시 관측값 검증
        next_obs, reward, terminated, truncated, info = wrapper_env.step(1)
        self.assertEqual(next_obs.shape, (7,))
        self.assertTrue(np.all(next_obs >= 0.0) and np.all(next_obs <= 1.0))


if __name__ == "__main__":
    unittest.main()
