from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from env.jin_hilla_scenario_env import JinHillaScenarioEnv, ScenarioConfig
from scripts.train import Policy


def evaluate(policy, episodes: int, seed: int, random_policy: bool, device: torch.device) -> dict:
    env = JinHillaScenarioEnv(ScenarioConfig())
    rewards, survival, failures, cleanses = [], [], 0, []
    for index in range(episodes):
        observation, _ = env.reset(seed=seed + index)
        total = 0.0
        while True:
            if random_policy:
                action = random.randrange(env.action_size)
            else:
                x = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy(x), dim=-1).item())
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                rewards.append(total); survival.append(info["scenario_steps"]); cleanses.append(info["cleanse_count"])
                failures += int(terminated and info.get("red_skulls", 0) > info.get("green_skulls", 0))
                break
    return {"episodes": episodes, "mean_reward": sum(rewards) / episodes, "mean_survival_steps": sum(survival) / episodes, "red_exceeds_green_termination_rate": failures / episodes, "mean_cleanse_count": sum(cleanses) / episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/m8_policy.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = JinHillaScenarioEnv(ScenarioConfig())
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = Policy(saved["observation_size"], saved["action_size"]).to(device)
    policy.load_state_dict(saved["state_dict"]); policy.eval()
    with torch.no_grad():
        learned = evaluate(policy, args.episodes, args.seed, False, device)
    baseline = evaluate(None, args.episodes, args.seed, True, device)
    result = {"device": str(device), "seed": args.seed, "learned_policy": learned, "random_baseline": baseline, "beats_random_reward": learned["mean_reward"] > baseline["mean_reward"]}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
