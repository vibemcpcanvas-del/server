# -*- coding: utf-8 -*-
"""R6 실입력 증명 영상 — 봇이 키를 발화하고 캐릭터가 실제로 이동하는 인과를
프레임마다 합성해 기록한다.

절차 (완전 자동, 사용자 개입 0):
  1. 게임 창 포그라운드 복구 + 채팅창 해제(ESC)
  2. 캡처 시작 → 좌/우 이동을 번갈아 5회 발화 (act 0/8)
  3. 매 발화 전후 프레임 캡처 → 화면 diff로 이동 감지
  4. 프레임마다 R6 REAL INPUT 패널 합성 (발화 키/시각/diff/이동 판정)
  5. ffmpeg rawvideo 파이프로 MP4 기록

안전: --arm 대상 키는 좌우 이동뿐. KILL 파일 상시 대기.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import cv2
import numpy as np
import bettercam
import win32gui
import win32con
import win32process
import pydirectinput

FF = (r"C:\Users\ROCmAdmin\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
OUT = r"C:\Users\ROCmAdmin\Desktop\test\server\reports\r6_real_input_proof.mp4"
W, H, FPS = 1366, 768, 30

def find_game() -> int:
    target = None
    def cb(h, _):
        nonlocal target
        if win32gui.IsWindowVisible(h) and win32gui.GetClassName(h) == "MapleStoryClass":
            target = h
    win32gui.EnumWindows(cb, None)
    return target

def foreground(hwnd: int):
    user32 = ctypes.windll.user32
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.5)
    user32.keybd_event(0x12, 0, 0, 0)
    time.sleep(0.05)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print("SetForegroundWindow:", e)
    time.sleep(0.05)
    user32.keybd_event(0x12, 0, 2, 0)
    time.sleep(0.5)

def draw_panel(frame: np.ndarray, entry: dict):
    cv2.rectangle(frame, (8, 8), (480, 130), (15, 15, 15), -1)
    cv2.rectangle(frame, (8, 8), (480, 130), (0, 200, 0), 2)
    cv2.putText(frame, "R6 REAL INPUT PROOF (live)", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
    color = (80, 200, 80) if entry.get("moved") else (150, 150, 150)
    lines = [
        f"step {entry.get('step', '-')}  key {entry.get('key', '-')}  fired {entry.get('fired', '-')}",
        f"screen diff {entry.get('diff', 0):,}  (idle ~4.6M)",
        f"MOVED: {'YES' if entry.get('moved') else 'no'}",
    ]
    for li, line in enumerate(lines):
        cv2.putText(frame, line, (18, 62 + li * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color if li == 2 else (230, 230, 230), 2)
    return frame

def main():
    hwnd = find_game()
    assert hwnd, "MapleStoryClass 창 없음"
    print("hwnd:", hwnd)
    foreground(hwnd)
    time.sleep(1.0)

    # 채팅창 해제 (1차 실측에서 발견된 함정)
    pydirectinput.press("esc")
    time.sleep(0.5)

    cam = bettercam.create(output_color="BGR")
    l, t, r_, b = win32gui.GetClientRect(hwnd)
    pt = win32gui.ClientToScreen(hwnd, (l, t))
    region = (pt[0], pt[1], pt[0] + (r_ - l), pt[1] + (b - t))
    print("region:", region)

    def grab():
        for _ in range(12):
            f = cam.grab(region=region)
            if f is not None:
                return f.copy()
            time.sleep(0.15)
        return None

    f0 = grab()
    diff_idle = int(np.abs(f0.astype(int) - grab().astype(int)).sum())
    print("idle diff:", diff_idle)

    writer = subprocess.Popen(
        [FF, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast",
         "-crf", "26", "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    from core.input_controller import InputController
    ctl = InputController(hwnd, dry_run=False)
    ctl.kill.start()
    ctl.arm(True)
    ok, why = ctl.safe_to_act()
    print("safe_to_act:", ok, why)
    if not ok:
        print("게이트 미충족 — 영상 없이 종료")
        writer.kill()
        return

    steps = [("LEFT", 0), ("RIGHT", 8), ("LEFT", 0), ("RIGHT", 8), ("LEFT", 0)]
    log = []
    for step, (key, act) in enumerate(steps):
        f_pre = grab()
        fired, why = ctl.act(act)
        time.sleep(1.2)   # 이동 + 화면 안정화
        f_post = grab()
        diff = int(np.abs(f_pre.astype(int) - f_post.astype(int)).sum())
        moved = bool(diff > diff_idle * 1.8)
        entry = {"step": step, "key": key, "act": act, "fired": fired,
                 "gate": why, "diff": diff, "moved": moved}
        log.append(entry)
        print(entry)
        # 발화 순간 프레임을 패널 합성해 1.2초 기록 (발화→이동 인과 시퀀스)
        for phase, (fr, moved_flag) in enumerate([(f_pre, False), (f_post, moved)]):
            e2 = dict(entry, moved=moved_flag if phase == 1 else False)
            e2["key"] = key if phase == 0 else f"{key} (after)"
            for _ in range(FPS // 2):   # 0.5초 x2
                frame = fr.copy()
                draw_panel(frame, e2)
                writer.stdin.write(frame.tobytes())
        # 트랜지션 0.3초
        e3 = dict(entry)
        for _ in range(FPS // 3):
            frame = f_post.copy()
            draw_panel(frame, e3)
            writer.stdin.write(frame.tobytes())
        if ctl.kill.is_set():
            print("KILL SWITCH")
            break

    ctl.arm(False)
    cam.release()
    writer.stdin.close()
    writer.wait(timeout=60)
    import os
    json.dump(log, open(r"C:\Users\ROCmAdmin\Desktop\test\server"
                        r"\reports\r6_real_input_log.json", "w"), indent=1)
    print(f"saved: {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB), log json")
    moved_n = sum(1 for e in log if e["moved"])
    print(f"발화 {len(log)}회 중 이동 확인 {moved_n}회")


if __name__ == "__main__":
    main()
