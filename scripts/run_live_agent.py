from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env.jin_hilla_gym_env import JinHillaScenarioGymEnv
from env.jin_hilla_live_vision_env import JinHillaVisionObservationWrapper
from core.priority_arbiter import PriorityArbiter, Action
from core.vision_schema import BossBattleState


def run_live_agent(
    model_path: str = "artifacts_m8_sb3_ppo/ppo_m8_policy.zip",
    stats_path: str = "artifacts_m8_sb3_ppo/vec_normalize.pkl",
    use_mock: bool = False,
    duration_sec: int = 10,
) -> None:
    print("[M9 Agent] 로컬 실시간 제어 에이전트 초기화 중...")

    # 1. 화면 캡처 소스 구성
    # v3 (2026-09-08): mock 기본 폐지 — 실기기 BetterCam(Desktop Duplication)이 기본.
    # Moonlight/Sunshine 원격 세션도 같은 화면을 캡처하므로 별도 분기 불필요.
    # SyntheticScreenSource는 tests/ 전용 픽스처로 격하 (라이브 실행 금지).
    if use_mock:
        raise SystemExit(
            "[M9 Agent] --mock은 라이브 실행에서 제거됨 (실입력 계층 전환 완료). "
            "합성 화면은 tests/ 픽스처만 사용. 실기기 캡처로 실행하세요."
        )
    print("[M9 Agent] Windows Desktop Duplication 고속 화면 캡처(BetterCam) 구동")
    from core.vision.screen_source_bettercam import BetterCamScreenSource
    screen_source = BetterCamScreenSource()

    # 2. Gymnasium 표준 Wrapper 환경 빌드
    def make_live_env():
        base_env = JinHillaScenarioGymEnv()
        return JinHillaVisionObservationWrapper(base_env, screen_source=screen_source)

    eval_env = make_vec_env(make_live_env, n_envs=1)
    if Path(stats_path).exists():
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    # 3. M8 PPO 신경망 정책 로드
    model = None
    if Path(model_path).exists():
        model = PPO.load(model_path, env=eval_env)
        print(f"[M9 Agent] 학습된 PPO 회피 모델 로드 성공: {model_path}")
    else:
        print(f"[M9 Agent] 주의: 모델 파일({model_path})이 없어 규칙 기반 정책으로 동작합니다.")

    # 4. 3계층 우선순위 중재기 초기화
    arbiter = PriorityArbiter(danger_red_threshold=2)

    print(f"[M9 Agent] 실시간 비전 제어 루프 가동 시작 (총 {duration_sec}초간 실행)...")
    obs = eval_env.reset()
    start_time = time.time()
    step_count = 0

    try:
        while time.time() - start_time < duration_sec:
            # L1: PPO 신경망 회피 액션 예측
            if model is not None:
                l1_action, _ = model.predict(obs, deterministic=True)
                l1_act_val = int(l1_action[0])
            else:
                l1_act_val = int(Action.STAY)

            # L3/L2: 우선순위 중재기 개입 (상태 역변환)
            current_state = BossBattleState(
                player_lane=float(obs[0][0]),
                hazard_lane=float(obs[0][1]),
                altar_lane=float(obs[0][2]),
                altar_active=bool(obs[0][3] > 0.5),
                green_skulls=int(round(obs[0][4] * 5)),
                red_skulls=int(round(obs[0][5] * 5)),
                time_ratio=float(obs[0][6]),
            )
            final_action = arbiter.arbitrate(current_state, l1_dodge_action=l1_act_val)

            # 환경 스텝 전진
            obs, _, done, _ = eval_env.step(np.array([final_action]))
            step_count += 1

            if step_count % 30 == 0:
                print(f"[루프 진행중] Step: {step_count} | Player: {current_state.player_lane:.2f} | RedSkulls: {current_state.red_skulls} | Executed Action: {Action(final_action).name}")

            if done[0]:
                obs = eval_env.reset()

            time.sleep(0.016)  # 약 60 FPS 제어 주기

    finally:
        if hasattr(screen_source, "close"):
            screen_source.close()
        print(f"[M9 Agent] 실행 종료: 총 {step_count}스텝 실시간 제어 완수.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts_m8_sb3_ppo/ppo_m8_policy.zip")
    parser.add_argument("--stats", default="artifacts_m8_sb3_ppo/vec_normalize.pkl")
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--duration", type=int, default=5)
    args = parser.parse_args()

    run_live_agent(
        model_path=args.model,
        stats_path=args.stats,
        use_mock=args.mock,
        duration_sec=args.duration,
    )
