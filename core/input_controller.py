# -*- coding: utf-8 -*-
"""R4 — 키 입력 출력 계층 (pydirectinput-rgx 기반).

리서치 근거 (2026-09-07):
- pydirectinput-rgx 2.1.3 (2025-08, 유지보수 포크) — DirectX 게임 표준
- 킬스위치: GetAsyncKeyState & 0x8000 폴링 스레드 — 게임이 포커스를 가져도
  작동하는 Windows 표준 (WuWa Inventory Kamera의 StopSignal 패턴)
- 안전장치: 킬스위치(F12) + 포커스 체크 + keyUp 보장(스티키 키 방지)
  + 동일 방향 입력 최소 간격(진동 방지) + dry_run 모드

M9_PIPELINE R4 규격:
- Action 4종 매핑: LEFT/RIGHT → 방향키, STAY → 무입력, HARVEST → 스페이스
- 10행동 v3: (좌/유지/우)×(서브레인 -1/0/+1) + HARVEST(9)
"""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

import win32gui

# pydirectinput-rgx (설치: pip install pydirectinput-rgx)
try:
    import pydirectinput
    pydirectinput.FAILSAFE = True          # 마우스 좌상단 코너 = 비상 정지
    pydirectinput.PAUSE = 0.02             # 입력 간 최소 간격
    HAS_PDI = True
except ImportError:
    HAS_PDI = False

VK_F12 = 0x7B
_user32 = ctypes.windll.user32

# v3 10행동 → 키 매핑
# 0..2: 좌 이동(서브레인 -1/0/+1), 3..5: 유지, 6..8: 우 이동, 9: HARVEST
ACTION_KEYS = {
    "left": "left",     # 메이플 기본: ←→ 이동 (설정에 맞춰 변경 가능)
    "right": "right",
    "up": "up",         # 서브레인 미세 이동용 (사다리/점프 대체 — 게임 설정 확인 필요)
    "down": "down",
    "harvest": "space",  # 수집키 (기본 스페이스 — 게임 설정 확인 필요)
}

# 서브레인 접근 방식: 메이플은 레인 내 연속 x축이라 서브레인은 이동 "지속 시간"으로 근사
# sub -1 = 짧게(0.10s), 0 = 중간(0.18s), +1 = 길게(0.26s)
SUB_DURATION = {0: 0.10, 1: 0.18, 2: 0.26}


class KillSwitch:
    """GetAsyncKeyState 폴링 킬스위치 — 포커스 무관 (검증된 StopSignal 패턴)."""

    def __init__(self, vk: int = VK_F12, poll: float = 0.08):
        self._vk = vk
        self._poll = poll
        self._event = threading.Event()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="KillSwitch-F12")

    def start(self):
        self._thread.start()

    def is_set(self) -> bool:
        return self._event.is_set()

    def stop(self):
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._poll * 3)

    def _loop(self):
        while not self._event.is_set() and not self._shutdown.is_set():
            try:
                if _user32.GetAsyncKeyState(self._vk) & 0x8000:
                    self._event.set()
                    break
            except Exception:
                pass
            time.sleep(self._poll)


class InputController:
    """v7 행동 → 실제 키 입력. 모든 발화 전 안전 게이트를 통과해야 한다."""

    def __init__(self, hwnd: int, dry_run: bool = True,
                 same_key_cooldown: float = 0.30):
        self.hwnd = hwnd
        self.dry_run = dry_run
        self.same_key_cooldown = same_key_cooldown
        self._last_key_at: dict[str, float] = {}
        self._pressed: set[str] = set()
        self.kill = KillSwitch()
        self.kill.start()
        self.enabled = False   # arm() 호출 전까지 어떤 입력도 안 나감

    def arm(self, on: bool = True):
        """입력 활성화/비활성화. 비활성화 시 눌린 키 전부 해제."""
        self.enabled = on
        if not on:
            self.release_all()

    def game_has_focus(self) -> bool:
        try:
            return win32gui.GetForegroundWindow() == self.hwnd
        except Exception:
            return False

    def safe_to_act(self) -> tuple[bool, str]:
        if self.kill.is_set():
            return False, "KILL_SWITCH"
        if not self.enabled:
            return False, "NOT_ARMED"
        if not self.game_has_focus():
            return False, "NO_FOCUS"
        if not HAS_PDI:
            return False, "NO_PDI"
        return True, "ok"

    def _tap(self, key: str, duration: float):
        """keyDown→sleep→keyUp. dry_run이면 로그만. 예외 시에도 keyUp 보장."""
        now = time.time()
        last = self._last_key_at.get(key, 0.0)
        if now - last < self.same_key_cooldown:
            return False
        self._last_key_at[key] = now
        if self.dry_run:
            print(f"[dry] tap {key} {duration:.2f}s")
            return True
        try:
            import pydirectinput
            pydirectinput.keyDown(key)
            self._pressed.add(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            self._pressed.discard(key)
        except Exception:
            self._force_release(key)
            raise
        return True

    def _force_release(self, key: str):
        try:
            import pydirectinput
            pydirectinput.keyUp(key)
        except Exception:
            pass
        self._pressed.discard(key)

    def release_all(self):
        """스티키 키 방지 — 알려진 이동키 전부 keyUp."""
        for key in set(ACTION_KEYS.values()) | self._pressed:
            self._force_release(key)

    def act(self, action: int) -> tuple[bool, str]:
        """v3 10행동 실행. 반환: (실행됨, 사유).

        STAY는 키 입력이 발생하지 않는 행동 — 안전 게이트와 무관하게 항상 허용.
        (킬스위치·포커스 체크는 "입력이 나가는" 행동에만 적용)
        """
        act = int(action)
        if act == 9:
            ok, why = self.safe_to_act()
            if not ok:
                return False, why
            self._tap(ACTION_KEYS["harvest"], 0.05)
            return True, "harvest"
        d_lane, d_sub = divmod(act, 3)
        d_lane -= 1
        if d_lane == 0:
            return True, "stay"          # 유지 — 입력 없음
        ok, why = self.safe_to_act()
        if not ok:
            return False, why
        key = ACTION_KEYS["left"] if d_lane < 0 else ACTION_KEYS["right"]
        dur = SUB_DURATION[d_sub]
        self._tap(key, dur)
        return True, f"move_{key}_{dur:.2f}s"

    def status(self) -> str:
        ok, why = self.safe_to_act()
        return f"{'ENABLED' if self.enabled else 'disabled'} | {why} | pressed={self._pressed}"


if __name__ == "__main__":
    # 자가 테스트: dry_run 기본 — 실제 키는 나가지 않는다
    hwnd = 5899344
    ctl = InputController(hwnd, dry_run=True)
    print("kill switch: F12 | status:", ctl.status())
    for a in [0, 4, 8, 9, 1, 5]:
        print(f"action {a} ->", ctl.act(a))
    time.sleep(0.2)
    print("kill switch set? (F12 눌러보면 True):", ctl.kill.is_set())
    ctl.arm(False)
    print("released. done.")
