# YOLO Cycle 1 Report — 자동 라벨링 → 학습 → 검증 첫 실사이클

일시: 2026-09-08 | 실행: Windows CPU (AMD Ryzen 5 5600X 12스레드, torch 2.14.0+cpu, ultralytics 8.4.143)
소스: `reports/spectator_v3_run2_aieye.mp4` (1366x768, 30fps, 13,930프레임, 진힐라 보스전)
스크립트: `scripts/yolo_pipeline/` (run_all.py로 전 단계 체인 실행)

## 파이프라인 요약

| 단계 | 스크립트 | 결과 |
|---|---|---|
| 1. 프레임 추출 | `extract_frames.py` | jsonl `is_bossfight` 구간(4735~13629)에서 15fps 샘플 → **4,280장** (`artifacts_yolo_cycle1/dataset/images/all/`) |
| 2. 자동 라벨링 | `auto_label.py` | `core/vision/real_parser_v2.py` HSV 파서 재사용(블롭 bbox 확장), YOLO txt 4,280개 — player 3,940 / boss 4,024 / 둘 다 3,700 / 배경 15장, 처리속도 38fps |
| 3. 라벨 검수 | `review_labels.py` | 랜덤 10장 박스 오버레이 → `reports/yolo_cycle1_review/` (육안 검수 아래) |
| 4. 학습 | `train_yolo.py` | YOLOv8n, 600 train / 150 val (전체 풀 4,280장 중 시간 균등 서브샘플), imgsz 640, batch 8, **10 epochs, CPU 약 25분** (1,499초) |
| 5. 검증 추론 | `infer_check.py` | val 샘플 12장 추론 → `artifacts_yolo_cycle1/infer/` + ultralytics val() 재측정 |

## 최종 검증 지표 (vs 파서 자동 라벨 GT — 사람 검수 GT 아님)

- **mAP50 = 0.569, mAP50-95 = 0.324, Precision = 0.478, Recall = 0.615** (best.pt, val 150장/275 인스턴스)
- 에포크별 mAP50: 0.114 → 0.271 → 0.292 → 0.405 → 0.497 → 0.493 → 0.507 → 0.501 → 0.559 → **0.569** (수렴 중, 10에포크로 미포화)
- CPU 추론 속도: 22.6ms/장 (약 44fps — 실시간 파서 대체 가능 속도)

## 라벨 품질 검수 (10장 육안)

- **player 라벨 8/10 정확.** 오탐 2: 바닥의 하얀 해골 픽업 아이템을 player로 라벨(파서의 흰색 블롭 한계).
- **boss 라벨 오탐 다수:** 화씨의 모래시계 탑 붉은 액체(약 466프레임, boss 라벨의 ~12%), 탑 장식 붉은 오브, 붉은 실 공격 이펙트(박스 9.1%가 12,000px² 초과 — 실+보스 병합형)에 박스가 붙음. 실제 보스가 화면에 있어도 파서가 배경 붉은 요소를 최대 블롭으로 선택하면 미라벨.
- 결론: **GT 오염이 그대로 모델 오류로 전이됨** — 추론에서도 모델이 붉은 실/장식을 boss로 검출하는 것을 육안 확인. 그래도 boss는 12/12 프레임에서 검출(conf 평균 0.39), player는 12장 중 3장(라벨 박스가 30x30px로 작고 가변적).

## 사이클 1 결론 및 다음 개선점

1. **엔드투엔드 골격 검증 완료**: MP4 → 자동라벨 → 학습 → mAP 측정 → 추론 배포까지 전부 재현 가능 스크립트로 동작.
2. 최우선 개선: **라벨 정화** — (a) 탑 장식 존(near (470,330)/(990,280)) 고정 오탐 필터, (b) 박스 면적 상한(실 병합 제거), (c) 붉은 실 프레임에서 boss 라벨 억제, (d) 소수(100~200장) 사람 검수로 GT 교정 후 재학습.
3. player 검출 강화: 라벨 박스 최소 크기/종횡비 제약, 또는 파서 ROI를 캐릭터 크기에 맞게 조정.
4. 학습량: 같은 설정으로 전체 4,280장 사용 시 ~13분/에포크 → WSL2 ROCm GPU(venv-dl)로 이전하면 전체 데이터 50에포크도 현실적.
5. val은 학습 프레임과 ±2프레임 인접(15fps 샘플)이라 시간 중복이 있음 — 다음 사이클에서는 구간 단위 분리 권장.

## 산출물

- 스크립트: `scripts/yolo_pipeline/{common,extract_frames,auto_label,review_labels,train_yolo,infer_check,run_all}.py`
- 데이터셋: `artifacts_yolo_cycle1/dataset/` (이미지 4,280장 + YOLO txt 라벨 + meta.json)
- 가중치: `artifacts_yolo_cycle1/train/yolov8n_cycle1/weights/best.pt` (6.2MB, 3.0M params)
- 검증 추론: `artifacts_yolo_cycle1/infer/` (12장 + val_metrics.json)
- 검수 샘플: `reports/yolo_cycle1_review/` (10장)
- 학습 로그/지표: `artifacts_yolo_cycle1/train/yolov8n_cycle1/` (results.csv, 혼동행렬, PR 곡선)
