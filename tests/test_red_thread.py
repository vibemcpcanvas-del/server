"""붉은 실 스케줄러 단위 테스트."""
from __future__ import annotations

import unittest
import random

from env.red_thread_scheduler import RedThreadScheduler
from env.jin_hilla_verus_env import VerusHillaTrainingEnv, Action


class SchedulerTests(unittest.TestCase):

    def setUp(self):
        self.sched = RedThreadScheduler(rng=random.Random(42))

    def test_reset_picks_valid_set(self):
        self.sched.reset()
        self.assertIn(self.sched.current_set, range(1, 7))
        self.assertLessEqual(self.sched.current_order, 7)

    def test_threads_spawn_sequentially(self):
        self.sched.reset()
        spawned_total = 0
        for _ in range(600):  # 10초
            spawned = self.sched.tick()
            spawned_total += len(spawned)
        # 세트당 7영역, 여러 세트가 진행되어야 함
        self.assertGreater(spawned_total, 0)

    def test_telegraph_precedes_active(self):
        """예고 → 활성 순서 확인."""
        self.sched.reset()
        saw_telegraph_only = False
        saw_active = False
        for _ in range(600):
            self.sched.tick()
            tel = self.sched.telegraph_lanes()
            act = self.sched.danger_lanes()
            if tel and not act:
                saw_telegraph_only = True
            if act:
                saw_active = True
        self.assertTrue(saw_telegraph_only)
        self.assertTrue(saw_active)

    def test_danger_lanes_valid_range(self):
        self.sched.reset()
        for _ in range(600):
            self.sched.tick()
            for lane in self.sched.danger_lanes():
                self.assertIn(lane, range(7))
            for lane in self.sched.telegraph_lanes():
                self.assertIn(lane, range(7))


class ThreatIntegrationTests(unittest.TestCase):
    """환경 통합: 붉은 실이 실제로 플레이어를 치는지."""

    def test_player_gets_hit_when_standing_in_thread(self):
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, seed=42)
        obs, _ = env.reset(seed=42)
        st = env.state
        st.player_lane = 3
        st.boss_lane = -1  # 보스 멀리 (뼈 파동 회피)
        st.bone_wave = None
        before_green = st.souls.green
        hit = False
        for _ in range(3000):  # 50초 — 반드시 실에 맞아야 함
            obs, r, term, trunc, info = env.step(int(Action.STAY))
            if st.souls.red > 0:
                hit = True
                break
            if term:
                break
        self.assertTrue(hit, "50초간 3레인 대기했는데 실에 한 번도 안 맞음 — 스케줄러 오류")

    def test_player_can_dodge_by_moving(self):
        """이동하는 플레이어는 덜 맞아야 함 (회피 학습 가능성 검증)."""
        env_stationary = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, seed=42)
        env_stationary.reset(seed=42)
        env_stationary.state.boss_lane = -1
        env_stationary.state.bone_wave = None

        env_moving = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, seed=42)
        env_moving.reset(seed=42)
        env_moving.state.boss_lane = -1
        env_moving.state.bone_wave = None

        # 정지: 맞은 횟수
        hits_static = 0
        for _ in range(1800):  # 30초
            env_stationary.step(int(Action.STAY))
            hits_static += (env_stationary.state.souls.red > 0) - (getattr(env_stationary, "_prev_red", 0) > 0)
            env_stationary._prev_red = env_stationary.state.souls.red

        # 이동: 매 틱 랜덤 이동 (회피 시도)
        hits_moving = 0
        prev_red = 0
        for _ in range(1800):
            env_moving.step(int(random.choice([0, 2])))  # 좌우 랜덤
            cur_red = env_moving.state.souls.red
            if cur_red > prev_red:
                hits_moving += 1
            prev_red = cur_red

        # 이동하는 쪽이 덜 맞거나 비슷해야 함 (확률적이므로 극단 검증만)
        self.assertLessEqual(hits_moving, hits_static + 3)

    def test_candles_accumulate(self):
        """실에 맞으면 촛불이 점등되고, 전부 켜지면 제단이 생성."""
        env = VerusHillaTrainingEnv(scythe_ticks=(2, 1, 1), max_steps=50000, seed=42)
        env.reset(seed=42)
        st = env.state
        st.boss_lane = -1
        st.bone_wave = None
        saw_altar = False
        for _ in range(6000):  # 100초 — 촛불 3개 → 제단
            env.step(int(Action.STAY))
            if st.altar_present:
                saw_altar = True
                break
        self.assertTrue(saw_altar, "100초간 제단이 생성되지 않음")


if __name__ == "__main__":
    import random
    random.seed(42)
    unittest.main()
