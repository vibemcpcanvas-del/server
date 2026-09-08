# -*- coding: utf-8 -*-
"""행동 복제(BC) 데이터셋 추출 — 표준 ML: 관측 → 전문가 행동 라벨.

원천: reports/spectator_v3_run2_aieye.mp4 (285초 실전, 30fps)
- 관측: parse_frame 특징 (해골/보스방향/거리/붉은실) — 기존 파서 재사용, 신규 창작 없음
- 행동 라벨: 캐릭터 위치 변위(템플릿 추적) → LEFT/STAY/RIGHT
  (전문가=사용자의 키 입력을 직접 기록하지 못했으므로 위치 변위로 복원 — 표준 방식)
- 템플릿 추적: 이전 프레임 캐릭터 위치 크롭 → 다음 프레임 매칭 (conf<0.6 시 skip)

출력: reports/bc_dataset.jsonl  (1줄 = 1샘플: t, obs, action)
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import cv2
import numpy as np

VID = r"reports\spectator_v3_run2_aieye.mp4"
OUT = r"reports\bc_dataset.jsonl"

LOOKAHEAD = 5      # 5프레임(166ms) 변위로 행동 라벨
TH_STAY = 6.0      # |dx| < 6px → STAY (카메라 스크롤 노이즈 임계)
CONF_MIN = 0.55


def main():
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 2  # 15fps 샘플
    print(f"video {total} frames @ {fps:.0f}fps, sampling every {step}")

    from core.vision.real_parser_v2 import parse_frame

    # 1) 전 프레임 캐릭터 위치 추적 (템플릿 체이닝)
    positions = np.full(total, np.nan)
    confs = np.zeros(total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_tpl = None
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                i += 1
                continue
            if prev_tpl is None:
                # 초기: 흰 블롭으로 시드
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
                mask[:500, :] = 0
                mask[650:, :] = 0
                xs = np.where(mask > 0)[0]
                ys = np.where(mask > 0)[0]
                if len(xs) > 50:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    prev_tpl = frame[cy-60:cy+60, cx-60:cx+60]
                    positions[i] = cx
                    confs[i] = 1.0
            else:
                res = cv2.matchTemplate(frame, prev_tpl, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv >= CONF_MIN:
                    cx, cy = ml[0] + 60, ml[1] + 60
                    positions[i] = cx
                    confs[i] = mv
                    prev_tpl = frame[cy-60:cy+60, cx-60:cx+60]
                else:
                    prev_tpl = None  # 재시드 필요
        i += 1
        if i % 3000 == 0:
            print(f"  track {i}/{total}", flush=True)
    cap.release()
    valid = np.count_nonzero(~np.isnan(positions))
    print(f"tracked {valid}/{total//step} samples ({valid/(total//step)*100:.0f}%)")

    # 2) 관측 특징 + 행동 라벨 생성
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_out = 0
    with open(OUT, "w", encoding="utf-8") as out:
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    i += 1
                    continue
                if i + LOOKAHEAD * step >= total:
                    break
                x_now = positions[i]
                x_fut = positions[i + LOOKAHEAD * step]
                if np.isnan(x_now) or np.isnan(x_fut):
                    i += 1
                    continue
                dx = float(x_fut - x_now)
                if abs(dx) < TH_STAY:
                    action = "STAY"
                elif dx < 0:
                    action = "LEFT"
                else:
                    action = "RIGHT"

                res = parse_frame(frame)
                if not res.get("is_bossfight"):
                    i += 1
                    continue
                obs = {
                    "g": res.get("green_skulls", 0),
                    "r": res.get("red_skulls", 0),
                    "d": res.get("skulls_destroyed", 0),
                    "dir": round(float(res.get("boss_dir_lanes", 0.0)), 3),
                    "dist": round(float(res.get("boss_distance_px", 0.0)), 1),
                    "thread": bool(res.get("red_thread_present", False)),
                    "warn": bool(res.get("warning_visible", False)),
                }
                out.write(json.dumps({
                    "i": i, "t": round(i / fps, 2), "obs": obs,
                    "action": action, "dx": round(dx, 1),
                    "x": float(x_now),
                }) + "\n")
                n_out += 1
            i += 1
    cap.release()
    print(f"dataset: {n_out} samples -> {OUT}")


if __name__ == "__main__":
    main()
