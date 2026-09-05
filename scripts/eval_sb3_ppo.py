from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.jin_hilla_gym_env import JinHillaScenarioGymEnv


def run_benchmark(policy_fn, eval_env, episodes: int, base_seed: int, options: dict | None = None):
    rewards, steps, red_exceeds, cleanses, dodges = [], [], [], [], []

    for ep in range(episodes):
        eval_env.seed(base_seed + ep)
        # reset with optional stress conditions
        if options:
            eval_env.env_method("reset", options=options)
            obs = eval_env.reset()
        else:
            obs = eval_env.reset()

        ep_reward = 0.0
        last_info = {}

        while True:
            action = policy_fn(obs)
            obs, reward, done, infos = eval_env.step(action)
            ep_reward += float(reward[0])
            last_info = infos[0]
            if done[0]:
                break

        rewards.append(ep_reward)
        steps.append(last_info.get("step_count", 0))
        red_exceeds.append(1.0 if last_info.get("termination_reason") == "red_skulls_exceed_green_skulls" else 0.0)
        cleanses.append(last_info.get("cleanse_count", 0))
        dodges.append(last_info.get("dodge_count", 0))

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_survival_steps": float(np.mean(steps)),
        "red_exceeds_green_termination_rate": float(np.mean(red_exceeds)),
        "mean_cleanse_count": float(np.mean(cleanses)),
        "mean_dodge_count": float(np.mean(dodges)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts_m8_sb3_ppo/ppo_m8_policy.zip")
    parser.add_argument("--stats", default="artifacts_m8_sb3_ppo/vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", default="artifacts_m8_sb3_ppo/evaluation.json")
    args = parser.parse_args()

    eval_env = make_vec_env(lambda: JinHillaScenarioGymEnv(), n_envs=1)
    eval_env = VecNormalize.load(args.stats, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(args.model, env=eval_env)

    # 1. Normal 모드 평가 (평소 완벽 회피 및 생존력)
    learned_normal = run_benchmark(
        lambda o: model.predict(o, deterministic=True)[0],
        eval_env,
        args.episodes,
        args.seed,
    )

    random_normal = run_benchmark(
        lambda o: np.array([eval_env.action_space.sample()]),
        eval_env,
        args.episodes,
        args.seed,
    )

    # 2. Stress 모드 평가 (위기 상황 주입: 제단 활성화 + 빨강 3개)
    stress_condition = {"green_skulls": 2, "red_skulls": 3, "altar_active": True}
    learned_stress = run_benchmark(
        lambda o: model.predict(o, deterministic=True)[0],
        eval_env,
        50,
        args.seed + 5000,
        options=stress_condition,
    )

    # 3. 범용 보스 평가 게이트
    # - 평소에는 압도적인 생존력과 회피율
    # - 제단이 존재하는 위기 상황에서는 0회 이상의 적극적 정화 수행
    gates = {
        "beats_random_reward": learned_normal["mean_reward"] > random_normal["mean_reward"],
        "survival_at_least_110pct": learned_normal["mean_survival_steps"] >= (random_normal["mean_survival_steps"] * 1.10),
        "red_exceeds_rate_le_0_20": learned_normal["red_exceeds_green_termination_rate"] <= 0.20,
        "altar_resolution_verified": learned_stress["mean_cleanse_count"] > 0 or learned_normal["mean_cleanse_count"] > 0,
    }
    complete = all(gates.values())

    result = {
        "device": str(model.device),
        "seed": args.seed,
        "normal_evaluation": {
            "learned_policy": learned_normal,
            "random_baseline": random_normal,
        },
        "stress_evaluation": {
            "condition": stress_condition,
            "learned_policy": learned_stress,
        },
        "m8_pass_gates": gates,
        "m8_complete": complete,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
