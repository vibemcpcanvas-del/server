# M8 훈련 및 평가 가이드

## 상태 요약
- **상태**: **M8 범용 맵 회피 코어(Universal Map Evader) 검증 완료 (Pass)**
- **기반 아키텍처**: Stable-Baselines3 PPO + Gymnasium 래퍼 (`JinHillaScenarioGymEnv`)
- **주요 성과**:
  - 평균 생존: 349.4스텝 / 360스텝 만점 (랜덤 베이스라인 39.6스텝 대비 8.8배)
  - 평균 보상: 366.3점 (랜덤 베이스라인 22.7점 대비 16.1배)
  - 빨강 해골 초과 패배율: 3.0% (랜덤 베이스라인 100% 패배 대비 97% 감소)
  - 맵 위험 레인 회피 성공률: 98.5%+

## 핵심 산출물
- 모델 가중치: `artifacts_m8_sb3_ppo/ppo_m8_policy.zip` (Universal Map Evader v1)
- 정규화 통계: `artifacts_m8_sb3_ppo/vec_normalize.pkl`
- 평가 리포트: `artifacts_m8_sb3_ppo/evaluation.json`

## Colab 재현 절차
```bash
%cd /content/server
!git pull --ff-only origin main
%env PYTHONPATH=.
!pip install "stable-baselines3[extra]" gymnasium

# 1. 훈련
!python scripts/train_sb3_ppo.py --timesteps 300000 --n-envs 4 --output artifacts_m8_sb3_ppo

# 2. 평가 (Normal + Stress 모드)
!python scripts/eval_sb3_ppo.py \
  --model artifacts_m8_sb3_ppo/ppo_m8_policy.zip \
  --stats artifacts_m8_sb3_ppo/vec_normalize.pkl \
  --episodes 200 \
  --output artifacts_m8_sb3_ppo/evaluation.json
```

## 다음 단계
- M9: 실시간 화면 인식(CV) 및 3계층 우선순위 중재기(Priority Arbiter) 구축
- 상세 로드맵은 `ROADMAP.md` 참조.
