# viGEm 가상 게임패드 경로 검증 (2026-09-08)

## 검증 내용
1. **ViGEmBus 드라이버**: Running 확인 (Nefarius Virtual Gamepad Emulation Service)
2. **vgamepad 설치**: sdist 빌드가 ViGEmBus msiexec 프롬프트(블로킹)에서 멈추는 함정 →
   `VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true` + `--no-build-isolation`으로 해결 (0.1.3 설치 성공)
3. **가상 X360 패드 생성**: 성공
4. **XInput 라운드트립**: 패드 생성 전 NOT_CONNECTED(1167) → 생성 후 CONNECTED,
   스틱 -32768 값이 OS XInput API로 정확히 반영됨 (패드는 OS 레벨에서 완전히 살아있음)
5. **게임 반응 테스트**: 스틱 좌측 풀 홀드 2.5초 → 캐릭터 이동 0px (템플릿 매칭)

## 결론
- 가상 게임패드 채널은 OS 레벨에서 **완전 작동** (XInput 라운드트립 검증됨)
- 그러나 **메이플스토리 클라이언트는 게임패드 입력을 지원하지 않음** — 스틱 홀드에 무반응
- → viGEm 경로는 "입력은 OS에 도달하지만 게임이 그 입력을 읽지 않는" 상태로 종착

## 종합 (R6 입력 경로 전체)
| 경로 | OS 도달 | 게임 반영 | 판정 |
|---|---|---|---|
| SendInput (VK/스캔코드) | ✗ (GetAsyncKeyState 0x0) | ✗ | 게임가드 필터 |
| PostMessage | ✗ (UIPI 액세스 거부) | ✗ | 게임가드 권한 |
| viGEm 가상 게임패드 | ✅ (XInput 검증) | ✗ (게임 미지원) | 게임 한계 |

## 남은 옵션
1. **Sunshine 게임패드-키 매핑**: Moonlight 원격 세션 중 Sunshine이 게임패드를 키로
   변환해주는 기능(Sunshine 설정의 Keyboard/Controller 공존 모드) — 원격 세션 필수라
   로컬 검증 불가. 사용자가 Moonlight 접속 중일 때만 테스트 가능
2. **수동 하이브리드 운용**: 봇=관전 판단+힌트 표시, 사용자=키 실행 (현재 안정 운용점)
3. 게임가드 우회 시도는 운영정책(매크로) 리스크로 비권장 — 프로젝트 PM 결정 필요
