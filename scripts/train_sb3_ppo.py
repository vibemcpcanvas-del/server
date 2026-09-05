from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.jin_hilla_gym_env import JinHillaScenarioGymEnv


def make_env_fn(seed: int):
    def _init():
        return JinHillaScenarioGymEnv(seed=seed)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts_m8_sb3_ppo")
    args = parser.parse_args()

    env_fns = [make_env_fn(args.seed + i) for i in range(args.n_envs)]
    vec_env = make_vec_env(env_fns[0], n_envs=args.n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "ppo_m8_policy")
    vec_env.save(output / "vec_normalize.pkl")
    print(f"PPO training finished. Model and stats saved to {output}")


if __name__ == "__main__":
    main()
