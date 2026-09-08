# 노트북 배포 패키지 — 진힐라 봇 (2026-09-08)

ip24 원격 PC방 + 노트북(himoon) 환경에서 봇을 실행하기 위한 설치·실행 가이드.

## 하드웨어 요구
- RAM 8GB+ / CPU 4코어+ / GPU 불필요 (추론만 — CPU로 충분) / 저장 2GB
- Windows 10/11

## 설치 (한 번)
```bat
git clone https://github.com/vibemcpcanvas-del/server.git bot
cd bot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install bettercam pydirectinput ultralytics
```
모델 파일 2개 복사 (이 호스트에서 가져가기):
- `artifacts_verus_curriculum\verus_curriculum_v7_balanced.zip` → 같은 경로 유지
- `artifacts_yolo_cycle1\train\yolov8n_cycle2_gpu\weights\best.pt` (선택 — 검출 강화용)

## ip24 접속
1. `C:\hiip\HiipClient.exe` 실행 → 로그인 → 게임 세션 시작
2. himoon(스트리밍 클라이언트) 창이 뜨면 게임 화면 스트리밍 시작
3. 게임을 원격 머신에서 실행해둔 상태 (보스전 입장 전 대기 추천)

## 실행
```bat
:: 창 핸들 자동 스캔은 아직 — himoon 창 핸들 확인:
.venv\Scripts\python -c "import win32gui; wins=[]; cb=lambda h,_: wins.append((h, win32gui.GetClassName(h), win32gui.GetWindowText(h))) or win32gui.EnumWindows(cb, None); [print(w) for w in wins if 'moon' in w[1].lower() or 'hiip' in w[1].lower() or 'moon' in w[2].lower()]"

:: 관전 모드 (키 안 나감 — 판단 기록만)
.venv\Scripts\python core\spectator_mode.py --hwnd <himoon 핸들> --seconds 300 --fps 30 --out reports\laptop_log.json

:: 실전 모드 (키 발화 — 킬스위치: 바탕화면에 KILL 파일 생성 시 즉시 정지)
.venv\Scripts\python core\spectator_mode.py --hwnd <himoon 핸들> --seconds 600 --fps 30 --arm --out reports\laptop_r6.json
```

## 검증 순서 (첫 세션)
1. 관전 모드 60초 — 파서가 게임 화면을 인식하는지 (is_bossfight / skull 카운트)
   - 인식 안 되면: himoon 창 해상도가 1366x768이 아닐 수 있음 — 파서는 자동 리사이즈(v2.7)하니
     로그의 player_found/boss_found만 확인
2. 실입력 5초 테스트: himoon 창 포커스 → 좌측 이동 1회 → 캐릭터 실제 이동 확인 (눈으로)
3. 관전 모드로 보스전 기록 → 로그 분석 (이 호스트에서 처리)
4. --arm 실전

## 킬스위치
- 바탕화면에 `KILL` 파일 생성 = 즉시 정지 (모든 입력 해제)
- F12도 작동 (단, 노트북 포그라운드 상태 필요)

## 알려진 제약
- himoon 창에 포커스가 있어야 키가 들어감 (다른 창 클릭하면 봇 입력 간섭 — NO_FOCUS 게이트가 자동 정지)
- ip24 세션 종료 = 게임 소멸 (봇은 waiting 상태로 전환)
- 게임 화면이 스트리밍 지연(네트워크)을 포함 — 반응성은 네트워크 품질 의존
