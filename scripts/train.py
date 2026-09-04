from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from env.jin_hilla_scenario_env import JinHillaScenarioEnv


class Policy(nn.Module):
    """128-128 MLP 정책망."""
    def __init__(self, observations: int, actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_episode(env: JinHillaScenarioEnv, policy: Policy, device: torch.device, train: bool):
    observation = env.reset()
    log_probs, rewards, entropies = [], [], []
    info: dict = {}
    while True:
        x = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(x)
        distribution = Categorical(logits=logits)
        action = distribution.sample() if train else torch.argmax(logits, dim=-1)
        observation, reward, terminated, truncated, info = env.step(int(action.item()))
        if train:
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
        rewards.append(reward)
        if terminated or truncated:
            return log_probs, rewards, entropies, info, terminated


def discount_rewards(rewards: list[float], gamma: float = 0.99) -> list[float]:
    discounted, running = [], 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        discounted.append(running)
    return list(reversed(discounted))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts_m8_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = JinHillaScenarioEnv()
    policy = Policy(
        env.observation_size if hasattr(env, "observation_size") else 7,
        env.action_size if hasattr(env, "action_size") else 4,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    episode_returns: list[float] = []
    baseline = 0.0
    baseline_momentum = 0.9
    entropy_coef = 0.02
    batch_size = max(1, args.batch_size)

    episode = 0
    while episode < args.episodes:
        batch_log_probs: list[torch.Tensor] = []
        batch_advantages: list[torch.Tensor] = []
        batch_entropies: list[torch.Tensor] = []

        current_batch = min(batch_size, args.episodes - episode)
        for _ in range(current_batch):
            log_probs, rewards, entropies, info, _ = run_episode(env, policy, device, True)

            # 에피소드-level 보상 형태화 (생존/정화 다목적 최적화 축소판)
            if rewards:
                termination_reason = info.get("termination_reason")
                cleanse_count = info.get("cleanse_count", 0)
                if termination_reason == "red_skulls_exceed_green_skulls":
                    rewards[-1] -= 10.0
                if cleanse_count > 0:
                    rewards[-1] += 5.0
                else:
                    rewards[-1] -= 2.0

            discounted = discount_rewards(rewards)
            returns_tensor = torch.tensor(discounted, dtype=torch.float32, device=device)

            # 이동평균 baseline (에피소드 간 지속적으로 갱신)
            if discounted:
                baseline = baseline_momentum * baseline + (1 - baseline_momentum) * discounted[0]
            advantages = returns_tensor - baseline

            batch_log_probs.append(torch.stack(log_probs))
            batch_advantages.append(advantages)
            batch_entropies.append(torch.stack(entropies))

            episode_returns.append(sum(rewards))
            episode += 1

            if episode % 250 == 0 or episode == args.episodes:
                window = episode_returns[-250:]
                print(f"episode={episode} mean_return={sum(window) / len(window):.3f}")

        # 배지 전역 기준으로 advantage 정규화 (에피소드 단위 정규화는 넬야 무의말함)
        all_advantages = torch.cat(batch_advantages)
        all_log_probs = torch.cat(batch_log_probs)
        all_entropies = torch.cat(batch_entropies)

        if all_advantages.numel() > 1:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        policy_loss = -(all_log_probs * all_advantages).sum() / current_batch
        entropy_bonus = all_entropies.sum() / current_batch
        loss = policy_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "m8_policy.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "observation_size": env.observation_size if hasattr(env, "observation_size") else 7,
            "action_size": env.action_size if hasattr(env, "action_size") else 4,
        },
        checkpoint,
    )

    metrics = {
        "git_commit": git_commit(),
        "environment": "jin_hilla_scenario_env",
        "environment_version": "M8",
        "seed": args.seed,
        "episodes": args.episodes,
        "batch_size": batch_size,
        "mean_training_return_last_250": sum(episode_returns[-250:]) / min(250, len(episode_returns)),
        "device": str(device),
        "checkpoint": str(checkpoint),
    }
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
