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


def run_episode(env: JinHillaScenarioEnv, policy: Policy, device: torch.device, seed: int, train: bool):
    observation = env.reset()
    log_probs, rewards, entropies = [], [], []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts_m8_v2")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = JinHillaScenarioEnv()
    policy = Policy(env.observation_size if hasattr(env, "observation_size") else 7, env.action_size if hasattr(env, "action_size") else 4).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    returns = []
    baseline = 0.0
    baseline_momentum = 0.9
    entropy_coef = 0.02

    for episode in range(args.episodes):
        log_probs, rewards, entropies, _, _ = run_episode(env, policy, device, args.seed + episode, True)

        discounted, running = [], 0.0
        for reward in reversed(rewards):
            running = reward + 0.99 * running
            discounted.append(running)
        discounted = list(reversed(discounted))

        returns_tensor = torch.tensor(discounted, dtype=torch.float32, device=device)

        # 이동평균 baseline
        if discounted:
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * discounted[0]
        advantages = returns_tensor - baseline

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(torch.stack(log_probs) * advantages).sum()
        entropy_bonus = torch.stack(entropies).sum()
        loss = policy_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returns.append(sum(rewards))

        if (episode + 1) % 250 == 0:
            print(f"episode={episode + 1} mean_return={sum(returns[-250:]) / 250:.3f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "m8_policy.pt"
    torch.save({
        "state_dict": policy.state_dict(),
        "observation_size": env.observation_size if hasattr(env, "observation_size") else 7,
        "action_size": env.action_size if hasattr(env, "action_size") else 4,
    }, checkpoint)

    metrics = {
        "git_commit": git_commit(),
        "environment": "jin_hilla_scenario_env",
        "environment_version": "M8",
        "seed": args.seed,
        "episodes": args.episodes,
        "mean_training_return_last_250": sum(returns[-250:]) / min(250, len(returns)),
        "device": str(device),
        "checkpoint": str(checkpoint),
    }
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
