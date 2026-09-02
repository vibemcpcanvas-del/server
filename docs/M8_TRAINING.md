# M8 training and evaluation

M8 is complete only after a learned policy is evaluated against a random baseline on the same seeded scenarios. Unit tests and GitHub Actions validate code; they do not train a policy.

## Local or Colab session

Use Python 3.11 or newer. In Colab, select a GPU runtime, then run:

```bash
git clone https://github.com/vibemcpcanvas-del/server.git
cd server
pip install -r requirements-colab.txt
python -m unittest discover -s tests -v
python scripts/train.py --episodes 4000 --output artifacts
python scripts/evaluate.py --checkpoint artifacts/m8_policy.pt --episodes 200 --output artifacts/evaluation.json
```

`train.py` uses CUDA automatically when PyTorch reports an available GPU, otherwise it runs on CPU. It writes `m8_policy.pt` and `train_metrics.json`. `evaluate.py` evaluates the learned policy and a random policy using the identical evaluation seeds and writes `evaluation.json`.

## Acceptance gate

Keep the output files from a run. A run may advance M8 only if `evaluation.json` has `beats_random_reward: true`, and the retained metrics identify the source commit, environment version, seed, episode count, and device.

## Scope

The scenario 환경은 재현 가능한 M8 결정 학습을 위한 것입니다. 위험 레인은 행동 전에 노출되며 플레이어가 해당 레인에 있을 때만 피격이 적용됩니다. 타이밍과 레인 생성기는 시뮬레이션 장치이며, 검증되지 않은 실제 게임 타이밍을 주장하지 않습니다. M9는 이 장치를 검증된 게임 영상 기반 감지로 교체합니다.
