# Local and Colab GPU workflow

## Responsibility split

GitHub Actions validates the repository on CPU. It runs the M8 unit-test suite on every push to `main` and every pull request. It does not run GPU training, access a game client, or publish a gameplay model.

Run GPU-required training and evaluation on a local machine or in Google Colab. Keep source code, environment definitions, test cases, configuration, and training-result metadata in GitHub.

## Before every GPU session

```bash
git clone https://github.com/vibemcpcanvas-del/server.git
cd server
python -m unittest discover -s tests -v
```

Only train from a commit whose M8 validation passes. Record its commit ID before starting:

```bash
git rev-parse HEAD
```

## Colab setup

1. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Clone the repository and change into it.
3. Run the M8 test command above before starting a GPU session.
4. Keep notebooks or training scripts responsible for installing their own ML dependencies, such as PyTorch, because the M8 environment itself uses only the Python standard library.
5. Never add GitHub credentials, game credentials, or API tokens directly to notebooks committed to the repository.

You can check GPU availability in a Colab cell:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## Training result contract

Each training run should write a result JSON or CSV outside the tracked source tree, or upload it as a run artifact. Include at least:

```json
{
  "git_commit": "<git rev-parse HEAD>",
  "environment": "jin_hilla_training_env",
  "environment_version": "M8",
  "seed": 0,
  "total_steps": 0,
  "mean_episode_reward": 0.0,
  "mean_survival_steps": 0.0,
  "red_exceeds_green_termination_rate": 0.0,
  "altar_cleanse_count": 0,
  "policy": "<algorithm and settings>",
  "device": "<GPU or CPU name>"
}
```

Do not treat a generated checkpoint as ready for real-screen use solely because training completed. Compare it against a random baseline with the same scenario seeds, preserve the evaluation result, and validate all screen events independently before connecting it to a game-screen adapter.

## Suggested iteration

1. Make a source change and push it.
2. Wait for the `M8 validation` GitHub Actions check to pass.
3. Pull that exact commit in local Python or Colab.
4. Train or evaluate with GPU as required.
5. Save the checkpoint and metrics with the originating commit ID.
6. Promote only checkpoints that beat the recorded baseline and whose code still passes CI.
