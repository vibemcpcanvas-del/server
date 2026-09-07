# -*- coding: utf-8 -*-
"""R5 — 관전 모드 통합 루프 (R2 파서 + v7 정책 + R4 입력 컨트롤러).

안전 설계:
- 기본 관전 모드: 화면 → 파서 → v7 추론 → **결정만 로그** (키 입력 안 나감)
- --arm 플래그로 입력 활성화 시에도: KILL 파일 + 포커스 게이트 상시 작동
- 파싱 15fps, 정책 추론은 파싱 성공 시에만 (게임 밖 화면에서는 대기)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import cv2
import numpy as np
import win32gui
import win32con
import mss

from core.vision.real_parser_v2 import parse_frame
from core.input_controller import InputController


def restore_window(hwnd: int):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.3)
    game_tid = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, None)
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, True)
    ctypes.windll.user32.PostMessageW(hwnd, 0x0112, 0xF120, 0)
    time.sleep(1.0)
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, False)


def obs_from_parse(res: dict) -> dict | None:
    """파서 결과 → v7 관측에 필요한 핵심 값. 부족하면 None."""
    if not res.get("is_bossfight"):
        return None
    if not (res.get("player_found") and res.get("boss_found")):
        return None
    return {
        "green_skulls": res.get("green_skulls", 0),
        "red_skulls": res.get("red_skulls", 0),
        "skulls_destroyed": res.get("skulls_destroyed", 0),
        "boss_dir_lanes": res.get("boss_dir_lanes", 0.0),
        "ss_warning": res.get("warning_visible", False),
        "boss_distance_px": res.get("boss_distance_px", 0.0),
    }


def heuristic_action(obs: dict) -> int:
    """v7 관측 브리지가 완성되기 전 임시 정책 — 안전 회피 휴리스틱.

    v7은 60차원 프레임 스택 입력이라 실화면→60차원 완전 브리지는 다음 단계.
    현재는 '위험도 기반 회피'로 관전 루프 전체를 검증한다.
    """
    danger = obs["red_skulls"] + (1 if obs["ss_warning"] else 0)
    bd = obs["boss_dir_lanes"]
    if abs(bd) < 0.6:
        # 보스와 같은 레인 — 뼈 파동 위험 → 대각 이동
        return 8 if bd >= 0 else 0     # 보스 반대편 + 긴 지속
    if danger >= 3:
        return 4                        # 위험 높으면 유지
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwnd", type=lambda x: int(x, 0), default=5899344)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--arm", action="store_true",
                    help="입력 활성화 (기본: 관전-only, 결정 로그만)")
    ap.add_argument("--out", default=r"reports\spectator_log.json")
    args = ap.parse_args()

    restore_window(args.hwnd)
    ctl = InputController(args.hwnd, dry_run=not args.arm)
    ctl.kill.start()
    ctl.arm(args.arm)

    region = {"left": 0, "top": 0, "width": 1024, "height": 768}
    log = []
    n = args.seconds * args.fps
    print(f"=== 관전 모드 {args.seconds}s | arm={args.arm} | "
          f"kill file: {ctl.kill.kill_file} ===")
    with mss.mss() as sct:
        for i in range(n):
            t0 = time.time()
            frame = np.array(sct.grab(region))[:, :, :3]
            res = parse_frame(frame)
            obs = obs_from_parse(res)
            entry = {"t": round(time.time() - t0, 3), "is_bossfight": res.get("is_bossfight", False)}
            if obs is not None:
                action = heuristic_action(obs)
                fired, why = ctl.act(action)
                entry.update({"obs": obs, "action": action,
                              "fired": fired, "gate": why})
            else:
                entry["state"] = "waiting_for_bossfight"
            log.append(entry)
            if i % (args.fps * 5) == 0:
                k = "bossfight" if obs else "waiting"
                print(f"[{i // args.fps:3d}s] {k} "
                      f"{('act=' + str(entry.get('action')) + ' gate=' + entry.get('gate', '-')) if obs else ''} "
                      f"kill={ctl.kill.is_set()}")
            # 킬스위치 걸리면 즉시 종료
            if ctl.kill.is_set():
                print("!!! KILL SWITCH — 즉시 종료:", ctl.kill.reason())
                break
            dt = 1.0 / args.fps - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    ctl.arm(False)
    def _jsonable(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=_jsonable)
    bf = sum(1 for e in log if e.get("is_bossfight"))
    acts = sum(1 for e in log if "action" in e)
    print(f"완료: {len(log)}프레임 | 보스전 판정 {bf} | 행동 결정 {acts} | 로그 {args.out}")


if __name__ == "__main__":
    main()
