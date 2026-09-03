from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from env.jin_hilla_scenario_env import JinHillaScenarioEnv
from scripts.train import Policy


def evaluate(policy, episodes: int, seed: int, random_policy: bool, device: torch.device, temperature: float = 1.0):
    env = JinHillaScenarioEnv()
    rewards, survival, failures, cleanses, dodges = [], [], 0, [], []
    for index in range(episodes):
        observation = env.reset()
        total = 0.0
        steps = 0
        cleansed = 0
        dodged = 0
        while True:
            if random_policy:
                action = random.randrange(4)
            else:
                x = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(x) / temperature
                if temperature == 0.0:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                rewards.append(total)
                survival.append(steps)
                cleanses.append(info.get("cleanse_count", 0))
                dodges.append(info.get("dodge_count", 0))
                failures += int(terminated and info.get("red_skulls", 0) > info.get("green_skulls", 0))
                break
    return {
        "episodes": episodes,
        "mean_reward": sum(rewards) / episodes,
        "mean_survival_steps": sum(survival) / episodes,
        "red_exceeds_green_termination_rate": failures / episodes,
        "mean_cleanse_count": sum(cleanses) / episodes,
        "mean_dodge_count": sum(dodges) / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts_m8_v2/m8_policy.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts_m8_v2/evaluation.json")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = Policy(saved["observation_size"], saved["action_size"]).to(device)
    policy.load_state_dict(saved["state_dict"])
    policy.eval()

    with torch.no_grad():
        learned = evaluate(policy, args.episodes, args.seed, False, device, args.temperature)
        baseline = evaluate(None, args.episodes, args.seed, True, device, 1.0)

    # M8 통과 게이트
    m8_pass_gates = {
        "beats_random_reward": learned["mean_reward"] > baseline["mean_reward"],
        "survival_at_least_110pct": learned["mean_survival_steps"] >= baseline["mean_survival_steps"] * 1.10,
        "red_exceeds_rate_le_0_20": learned["red_exceeds_green_termination_rate"] <= 0.20,
        "cleanse_more_than_random": learned["mean_cleanse_count"] > baseline["mean_cleanse_count"],
    }
    m8_complete = all(m8_pass_gates.values())

    result = {
        "device": str(device),
        "seed": args.seed,
        "temperature": args.temperature,
        "learned_policy": learned,
        "random_baseline": baseline,
        "m8_pass_gates": m8_pass_gates,
        "m8_complete": m8_complete,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
