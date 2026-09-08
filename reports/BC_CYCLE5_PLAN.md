# BC 사이클 5 — 리서치 반영 개선 (재귀개선 루프 1회전)

## 리서치 (2026-09-09, 사이클 4 이후)
표준 ML 문헌에서 우리 문제와 정확히 일치하는 4개 방향 확인:

1. **BCO (BC from Observation)** — 상태 궤적만 있고 행동 라벨이 없을 때
   역동역학 모델로 pseudo-action을 추론한 후 BC. 우리의 템플릿 추적 dx 복원이
   바로 이 계열이었음을 확인 (방향이 표준과 일치했음을 검증).
2. **Action Chunking** (arXiv 2507.09061) — 연속 제어 BC의 compounding error가
   지수적으로 커지는 것을 막으려면 open-loop 액션 청크 예측이 필수.
   우리의 1스텝 예측은 청크 없음 → **사이클 5 핵심 개선: 3스텝 청크 예측**.
3. **Knowledge Informed Models** (IJCAI 2025) — 도메인 지식으로 정책 구조를
   고정, 데이터로 파라미터만 fitting. 우리의 "위협 레인 위치를 관측에 추가"
   접근과 일치. (관측 설계에 LLM 활용 아이디어)
4. **SAIL (Self-Supervised Adversarial Imitation Learning)** — trivial
   no-action local minimum(c2에서 우리가 빠진 함정)의 표준 해법은
   expert/learner 궤적 discriminator. 다음 단계 후보.

## 사이클 5 설계 (표준 방법 적용)
- **청크 예측**: 1스텝 행동 대신 미래 3스텝 청크 (LEFT/STAY/RIGHT 시퀀스)
  → compounding error 완화. 라벨: 연속 3프레임의 majority 행동.
- **클래스 불균형 처치**: STAY 62.9% 지배 문제 → balanced sampling +
  이동 클래스 가중 (c1~c4에서 class_weight=balanced를 썼지만 MLP엔 미적용이었음)
- **평가 기준 변경**: macro-F1을 주지표로 (accuracy는 majority에 유리)
  + 이동 클래스 recall을 1차 지표로 명시

## 측정 목표
- c4 baseline: 0.650 (majority 0.628)
- 이동 recall: c3 수준(LEFT 0.18, RIGHT 0.11) 이상 유지 + 청크 정확도 측정
- 목표: accuracy 0.70+, 이동 recall 0.25+
