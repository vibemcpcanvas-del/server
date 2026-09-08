# -*- coding: utf-8 -*-
"""봇 움직임 오버레이 영상 생성.

spectator 녹화본(원본 프레임 + AI EYE 패널)을 프레임 단위로 재파싱하여
실측 플레이어/보스 좌표를 얻고, 로그의 v7 행동과 합쳐 다음을 그린다:
- 플레이어(=봇) 마커: 행동별 색 (파랑=좌 이동, 회백=유지, 주황=우 이동, 자홍=HARVEST)
- 이동 궤적(트레일): 최근 1.5초 잔상
- 보스 마커(빨강) + 근접 시 위협선
- HARVEST 폭발 링
- 우상단 BOT HUD: 좌표/행동/보스거리/해골

입력: reports/spectator_v3_run2_aieye.mp4 + reports/spectator_v3_run2.jsonl
출력: reports/aieye_botpath_full.mp4 (1366x768, 30fps, 원본과 동일 프레임 수)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque

import cv2
import numpy as np

BASE = r"C:\Users\ROCmAdmin\Desktop\test\server"
sys.path.insert(0, BASE)
from core.vision.real_parser_v2 import parse_frame  # noqa: E402

FF = (r"C:\Users\ROCmAdmin\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
SRC = os.path.join(BASE, "reports", "spectator_v3_run2_aieye.mp4")
LOG = os.path.join(BASE, "reports", "spectator_v3_run2.jsonl")
DST = os.path.join(BASE, "reports", "aieye_botpath_full.mp4")

W, H, FPS = 1366, 768, 30
TRAIL = 45  # 1.5초

ACT_COLOR = {  # BGR
    "L": (255, 160, 0),      # 파랑 계열 — 좌 레인 이동
    "S": (200, 210, 200),    # 회백 — 유지
    "R": (0, 150, 255),      # 주황 — 우 레인 이동
    "H": (255, 0, 255),      # 자홍 — HARVEST
}
ACT_LABEL = {"L": "<-", "S": "o", "R": "->", "H": "HARVEST"}


def act_family(a: int | None) -> str:
    if a is None:
        return "S"
    if a == 9:
        return "H"
    if a in (0, 1, 2):
        return "L"
    if a in (6, 7, 8):
        return "R"
    return "S"


def main():
    # 로그 로드
    entries = []
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    print(f"log entries: {len(entries)}")

    proc_in = subprocess.Popen(
        [FF, "-nostdin", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_out = subprocess.Popen(
        [FF, "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "30",
         "-pix_fmt", "yuv420p", DST],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = W * H * 3
    trail: deque[tuple[float, float]] = deque(maxlen=TRAIL)
    written = 0
    i = 0
    while True:
        buf = proc_in.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
        entry = entries[i] if i < len(entries) else {}
        act = entry.get("action")
        fam = act_family(act)

        # 재파싱 — 실측 좌표
        res = {}
        try:
            res = parse_frame(frame)
        except Exception:
            pass
        px = res.get("player_x") if res.get("player_found") else None
        py = res.get("player_y") if res.get("player_found") else None
        bx = res.get("boss_x") if res.get("boss_found") else None
        by = res.get("boss_y") if res.get("boss_found") else None

        if px is not None and py is not None:
            trail.append((float(px), float(py)))

        # 근접 위협선
        if px is not None and bx is not None:
            d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            if d < 350:
                cv2.line(frame, (int(px), int(py)), (int(bx), int(by)),
                         (0, 0, 220), 1, cv2.LINE_AA)

        # 트레일 (오래될수록 흐릿)
        tl = len(trail)
        for ti, (tx, ty) in enumerate(trail):
            alpha = (ti + 1) / tl
            c = (int(60 * alpha), int(230 * alpha), int(60 * alpha))
            cv2.circle(frame, (int(tx), int(ty)), 2, c, -1)

        # 보스 마커
        if bx is not None:
            pulse = 16 + (3 if (written // 5) % 2 else 0)
            cv2.circle(frame, (int(bx), int(by)), pulse, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "BOSS", (int(bx) - 24, int(by) - pulse - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 2)

        # 봇(플레이어) 마커 — 행동 색 + 방향 화살표
        if px is not None:
            col = ACT_COLOR[fam]
            cv2.circle(frame, (int(px), int(py)), 13, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(px), int(py)), 13, (255, 255, 255), 2, cv2.LINE_AA)
            lab = ACT_LABEL[fam]
            cv2.putText(frame, lab, (int(px) - 14, int(py) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if fam == "H":
                # HARVEST 폭발 링
                ring = (written % 12) * 4 + 6
                cv2.circle(frame, (int(px), int(py)), ring, ACT_COLOR["H"], 2, cv2.LINE_AA)
                cv2.putText(frame, "HARVEST!", (int(px) - 45, int(py) + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # BOT HUD (우상단)
        obs = entry.get("obs") or {}
        hud = [
            f"BOT  frame {i}  t={i // FPS}s",
            f"act {act if act is not None else '-'} ({fam})",
            f"dist {obs.get('boss_distance_px', '-')}px  dir {obs.get('boss_dir_lanes', '-')}",
            f"skull G{obs.get('green_skulls', '-')} R{obs.get('red_skulls', '-')} D{obs.get('skulls_destroyed', '-')}",
        ]
        x0, y0 = W - 320, 8
        cv2.rectangle(frame, (x0 - 8, y0), (W - 4, y0 + 84), (20, 18, 14), -1)
        cv2.rectangle(frame, (x0 - 8, y0), (W - 4, y0 + 84), (90, 90, 90), 1)
        for li, line in enumerate(hud):
            cv2.putText(frame, line, (x0, y0 + 20 + li * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 120), 1)

        try:
            proc_out.stdin.write(frame.tobytes())
            written += 1
        except Exception:
            break
        i += 1
        if i % 1500 == 0:
            print(f"  {i} frames...")

    proc_out.stdin.close()
    proc_out.wait(timeout=60)
    proc_in.kill()
    print(f"done: {written} frames -> {DST} ({os.path.getsize(DST)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
