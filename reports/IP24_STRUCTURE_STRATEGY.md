# ip24 원격 PC방 구조 분석 + 봇 통합 전략 (2026-09-08)

## 구조 파악
- ip24.co.kr = 원격 PC방(지피방) 서비스. 런처: `C:\hiip\HiipClient.exe`
- 스트리밍 클라이언트: `C:\hiip\remote\himoon.exe` — **Moonlight-Qt 포크 확정**
  (Qt5 + SDL2 + avcodec + gamecontrollerdb.txt = moonlight-qt와 동일 스택)
- 설정: `C:\hiip\remote\config.cfg` (moonlight 표준 conf 형식)

## 핵심 구조 전환
```
[기존 가정] 이 호스트에서 게임 실행 → 봇이 이 호스트에서 게임 제어
[ip24 실제] ip24 원격 PC방 머신에서 게임 실행
            ↓ 화면 스트림 (himoon으로 수신)
            이 호스트 = 뷰어/입력 송신자
```

**이것이 봇에 유리한 이유**:
1. 키 주입 대상이 himoon(게임가드 없는 일반 앱) — SendInput이 통할 가능성 높음
2. himoon이 받은 키는 Moonlight 프로토콜로 원격 머신에 전송 → 원격에서 실제 키 입력으로 발화
3. 화면 캡처는 himoon 창 (BetterCam 그대로)
4. 게임가드는 원격 머신의 게임에 적용되지만, 그곳에서 키를 누르는 건 "진짜 키보드"로 보임
   (Moonlight 입력 = 원격 머신의 Sunshine이 SendInput으로 발화 — 게임가드가 이걸 막는지는
   원격 환경에서만 확인 가능)

## 검증 절차 (himoon 접속 후)
1. himoon 창 캡처 → 파서로 게임 화면 인식 (리졸루션 재캘리브레이션 필요할 수 있음)
2. SendInput → himoon 창 포커스 → 원격 게임에서 캐릭터 이동 확인 (템플릿 매칭 측정)
3. 성공 시: spectator v3.2의 hwnd를 himoon으로 교체하고 R6 실전
4. 실패 시: himoon이 SDL에서 직접 키보드를 읽는지 확인 (SDL이 Windows 메시지가 아닌
   raw input을 쓰면 SendInput은 통하고, PostMessage는 안 통함 — 실측 필요)

## 준비 완료
- `scripts/r6_real_input_demo.py` — hwnd 파라미터로 himoon 지원하도록 수정 필요
- 관전 v3.2 — `--hwnd` 인자 이미 있음
- YOLO 사이클2 — 해상도만 맞으면 그대로 사용
