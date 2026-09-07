# -*- coding: utf-8 -*-
"""실화면 60초 녹화 (mss 15fps -> MP4) + R2 파서 오버레이 합성 버전 동시 생성.

표준 방식: 화면 영역 캡처(mss) -> ffmpeg rawvideo 파이프 -> H.264.
오버레이: 프레임마다 R2 파서 결과(보스전 여부/해골 3클래스/보스·플레이어 위치)
를 좌상단 패널로 합성 — "AI의 눈"이 무엇을 보는지 시각화.
"""
import ctypes
import subprocess
import threading
import time

import cv2
import numpy as np
import win32gui
import win32con
import mss

sys_path = r"C:\Users\ROCmAdmin\Desktop\test\server"
import sys
sys.path.insert(0, sys_path)
from core.vision.real_parser_v2 import parse_frame

FF = (r"C:\Users\ROCmAdmin\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
HWND = 5899344
REGION = {"left": 0, "top": 0, "width": 1024, "height": 768}
FPS = 15
DURATION = 60
W, H = 1024, 768


def ensure_window_visible():
    win32gui.ShowWindow(HWND, win32con.SW_RESTORE)
    time.sleep(0.3)
    game_tid = ctypes.windll.user32.GetWindowThreadProcessId(HWND, None)
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, True)
    ctypes.windll.user32.PostMessageW(HWND, 0x0112, 0xF120, 0)  # SC_RESTORE
    time.sleep(1.0)
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, False)
    print("window rect:", win32gui.GetWindowRect(HWND))


def record(out_plain: str, out_overlay: str):
    proc_p = subprocess.Popen(
        [FF, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast",
         "-crf", "26", "-pix_fmt", "yuv420p", out_plain],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc_o = subprocess.Popen(
        [FF, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast",
         "-crf", "26", "-pix_fmt", "yuv420p", out_overlay],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n = FPS * DURATION
    t0 = time.time()
    with mss.mss() as sct:
        for i in range(n):
            shot = sct.grab(REGION)
            frame = np.array(shot)[:, :, :3].copy()  # BGRA->BGR
            proc_p.stdin.write(frame.tobytes())
            # 오버레이: 4프레임마다 파싱 (파서 비용 절감) + 마지막 결과 표시
            panel = None
            if i % 4 == 0:
                try:
                    res = parse_frame(frame)
                except Exception:
                    res = {}
                panel = res
            elif i % 4 == 1 and "res" in dir():
                pass
            # 패널 그리기 (마지막 파싱 결과 재사용)
            if i % 4 == 0:
                global LAST
                LAST = res
            r = LAST if "LAST" in globals() else {}
            cv2.rectangle(frame, (8, 130), (320, 250), (20, 18, 14), -1)
            cv2.putText(frame, "AI EYE (R2 parser)", (16, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 2)
            if r.get("is_bossfight"):
                lines = [
                    f"bossfight  G/R/D {r.get('green_skulls','-')}/{r.get('red_skulls','-')}/{r.get('skulls_destroyed','-')}",
                    f"ss_timer {r.get('ss_timer_visible')}  warn {r.get('warning_visible')}",
                    f"player {'Y' if r.get('player_found') else 'N'}  boss {'Y' if r.get('boss_found') else 'N'}",
                    f"boss_dir {r.get('boss_dir_lanes', '-')} lanes",
                ]
            else:
                lines = ["not a bossfight", "(village / other UI)"]
            for li, line in enumerate(lines):
                cv2.putText(frame, line, (16, 176 + li * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            proc_o.stdin.write(frame.tobytes())
            # 프레임 페이싱
            target = t0 + (i + 1) / FPS
            dt = target - time.time()
            if dt > 0:
                time.sleep(dt)
    for p in (proc_p, proc_o):
        p.stdin.close()
        p.wait()
    import os
    for f in (out_plain, out_overlay):
        print("saved:", f, os.path.getsize(f), "bytes")


if __name__ == "__main__":
    ensure_window_visible()
    out1 = r"C:\Users\ROCmAdmin\Desktop\test\server\reports\live_maple.mp4"
    out2 = r"C:\Users\ROCmAdmin\Desktop\test\server\reports\live_maple_overlay.mp4"
    record(out1, out2)
