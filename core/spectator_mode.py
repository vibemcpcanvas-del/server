# -*- coding: utf-8 -*-
"""R5 — 통합 관전 모드 v3 (2026-09-08 재정비).

v3 변경 (최신 실시간 비전 에이전트 표준 대조 리서치 반영):
- 캡처: mss(16.7ms) → bettercam/Desktop Duplication API(0.24ms, 70배) — 실측
- 아키텍처: 순차 단일 스레드 → **캡처 스레드(producer) + 처리 루프(consumer) 분리**
  (Vulkan ML tutorial / PaliGemma 실시간 패턴: 캡처는 하드웨어 속도로 항상 돌고,
   처리는 최신 프레임을 소비 — 추론이 느려져도 캡처가 막히지 않음)
- 로깅: 종료 시 일괄 JSON(전투 중 사망 시 전체 유실 — 1차 구동 실증) →
  **JSONL 프레임 단위 증분 + 5초 스냅샷** 이중 저장
- 보스전 종료 자동 감지 종료 (2초 연속 비보스전) — 타이밍 맞추기 불필요
- 이중 구동 가드 (O_EXCL 파일락) — Hermes 실행 계층의 쌍 프로세스 대응

불변 (v2에서 유지):
- 기본 관전(키 미발화), --arm 시에도 KILL 파일 + 포커스 게이트 상시
- v7 정책 60차원 관측 브리지, 실패 시 휴리스틱 폴백
- v2.7 파서(이중 레이아웃 + Practice 판별)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import cv2
import numpy as np
import win32gui
import win32con
import bettercam

FF = (r"C:\Users\ROCmAdmin\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")

from core.vision.real_parser_v2 import parse_frame
from core.input_controller import InputController
from core.observation_bridge import ObservationBridge

MODEL_PATH = r"artifacts_verus_curriculum\verus_curriculum_v7_balanced.zip"


def load_policy():
    """v7 정책 로드. 실패 시 None (휴리스틱 폴백)."""
    try:
        from stable_baselines3 import PPO
        return PPO.load(MODEL_PATH, device="cpu")
    except Exception as e:
        print("v7 로드 실패 — 휴리스틱 폴백:", e)
        return None


def restore_window(hwnd: int):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.3)
    game_tid = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, None)
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, True)
    ctypes.windll.user32.PostMessageW(hwnd, 0x0112, 0xF120, 0)
    time.sleep(1.0)
    ctypes.windll.user32.AttachThreadInput(cur_tid, game_tid, False)


class ScreenCapture:
    """캡처 producer 스레드 — Desktop Duplication API.

    최신 프레임을 슬롯에 항상 갱신(overwrite)한다. 처리 루프는 언제든
    최신 프레임만 가져가므로 처리가 밀려도 지연이 누적되지 않는다
    (drop-oldest / latest-wins 패턴).
    """

    def __init__(self, hwnd: int, grab_hz: float = 60.0):
        l, t, r, b = win32gui.GetClientRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (l, t))
        self.region = (pt[0], pt[1], pt[0] + (r - l), pt[1] + (b - t))
        self.width, self.height = r - l, b - t
        self._cam = bettercam.create(output_color="BGR")
        self._interval = 1.0 / grab_hz
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.grab_count = 0
        self.error_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        next_t = time.perf_counter()
        while self._running:
            try:
                f = self._cam.grab(region=self.region)
                if f is not None:
                    with self._lock:
                        self._latest = f.copy()
                    self.grab_count += 1
            except Exception:
                self.error_count += 1
                time.sleep(0.05)
            next_t += self._interval
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

    def get(self) -> np.ndarray | None:
        """최신 프레임 반환 (변화 없으면 마지막 프레임 재사용)."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._cam.release()
        except Exception:
            pass


class JsonlLogger:
    """프레임 단위 증분 로거 — 프로세스 사망 시에도 직전 프레임까지 보존.

    JSONL(1줄=1프레임 append) + 주기적 스냅샷 JSON(구분석 호환) 이중 기록.
    """

    def __init__(self, out_base: str, snapshot_every_s: float = 5.0):
        self.jsonl_path = out_base.rsplit(".", 1)[0] + ".jsonl"
        self.snapshot_path = out_base
        self._every = snapshot_every_s
        self._entries: list[dict] = []
        self._last_snap = 0.0
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")
        self._jsonable = self._make_jsonable()

    @staticmethod
    def _make_jsonable():
        def _j(o):
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(type(o))
        return _j

    def append(self, entry: dict):
        self._entries.append(entry)
        try:
            self._fh.write(json.dumps(entry, ensure_ascii=False, default=self._jsonable) + "\n")
            # 30프레임마다 플러시 — 즉시성과 I/O 비용의 절충
            if len(self._entries) % 30 == 0:
                self._fh.flush()
        except Exception as e:
            print("jsonl write error:", e)
        now = time.time()
        if now - self._last_snap >= self._every:
            self.snapshot()
            self._last_snap = now

    def snapshot(self):
        """구 형식(JSON 배열) 스냅샷 — 분석 스크립트 호환용."""
        try:
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=1, default=self._jsonable)
        except Exception as e:
            print("snapshot write error:", e)

    def close(self):
        self.snapshot()
        try:
            self._fh.close()
        except Exception:
            pass


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
    """위험도 기반 안전 회피 휴리스틱 (v7 폴백)."""
    danger = obs["red_skulls"] + (1 if obs["ss_warning"] else 0)
    bd = obs["boss_dir_lanes"]
    if abs(bd) < 0.6:
        return 8 if bd >= 0 else 0
    return 4


def play_sound(kind: str = "enter_boss"):
    import winsound
    try:
        if kind == "enter_boss":
            winsound.Beep(880, 200)
            time.sleep(0.1)
            winsound.Beep(1320, 350)
        elif kind == "done":
            winsound.Beep(1320, 150)
            winsound.Beep(880, 300)
    except Exception as e:
        print("sound error:", e)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    k = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if k:
        ctypes.windll.kernel32.CloseHandle(k)
        return True
    return False


def acquire_lock(lock_path: str) -> bool:
    """이중 구동 가드 — O_EXCL 배타적 생성 (경쟁 안전)."""
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as lf:
                lf.write(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                old = int(open(lock_path, encoding="utf-8").read().strip() or "0")
            except (ValueError, OSError):
                old = 0
            if old == os.getpid() or not _pid_alive(old):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            print(f"이미 구동 중 (PID {old}) — 이중 기록 방지: 30s 대기 후 재확인")
            time.sleep(30)
            if _pid_alive(old):
                print("선점 구동 생존 — 조용히 종료 (기록 없음)")
                return False
            try:
                os.remove(lock_path)
            except OSError:
                pass
    print("락 획득 실패 — 조용히 종료")
    return False


def release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except OSError:
        pass


class VideoRecorder:
    """처리 루프 프레임을 그대로 MP4로 기록 + AI 판단 오버레이 합성.

    AI EYE 패널 = 파서가 본 것(해골/보스전) + v7 결정 행동 —
    "봇이 실화면을 보고 스스로 판단했다"는 증거가 프레임마다 새겨진다.
    ffmpeg rawvideo 파이프 (record_live.py 방식 유지).
    """

    def __init__(self, out_path: str, w: int, h: int, fps: int):
        self.proc = subprocess.Popen(
            [FF, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
             "-crf", "26", "-pix_fmt", "yuv420p", out_path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.w, self.h, self.fps = w, h, fps
        self.frames = 0
        self.last_entry: dict = {}

    def write(self, frame: np.ndarray, entry: dict):
        f = frame.copy()
        self.last_entry = entry
        # AI EYE 패널 (좌상단)
        cv2.rectangle(f, (8, 130), (340, 260), (20, 18, 14), -1)
        cv2.putText(f, "AI EYE (R2 parser + v7 policy)", (16, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 2)
        if entry.get("is_bossfight"):
            obs = entry.get("obs", {})
            lines = [
                f"bossfight  G/R/D {obs.get('green_skulls','-')}/{obs.get('red_skulls','-')}/{obs.get('skulls_destroyed','-')}",
                f"boss_dir {obs.get('boss_dir_lanes','-')} lanes  dist {int(obs.get('boss_distance_px', 0))}px",
                f"v7 action = {entry.get('action', '-')}  ({entry.get('policy', '-')})",
                f"gate {entry.get('gate', '-')}  fired {entry.get('fired', '-')}",
            ]
        else:
            lines = ["waiting for bossfight", "(village / lobby / other UI)"]
        for li, line in enumerate(lines):
            cv2.putText(f, line, (16, 176 + li * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        try:
            self.proc.stdin.write(f.tobytes())
            self.frames += 1
        except Exception:
            pass

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwnd", type=lambda x: int(x, 0), default=5899344)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30, help="처리 루프 주파수 (캡처는 60Hz 별도 스레드)")
    ap.add_argument("--arm", action="store_true",
                    help="입력 활성화 (기본: 관전-only, 결정 로그만)")
    ap.add_argument("--out", default=r"reports\spectator_log.json")
    ap.add_argument("--autostop-off", action="store_true",
                    help="보스전 종료 자동 감지 끔 (끝까지 녹화)")
    ap.add_argument("--enter-sound", action="store_true",
                    help="입장 시퀀스 시작 전 사운드 재생")
    ap.add_argument("--record", action="store_true",
                    help="AI EYE 오버레이 MP4 동시 기록 (납품 영상)")
    args = ap.parse_args()

    if not acquire_lock(args.out + ".lock"):
        sys.exit(0)

    if args.enter_sound:
        play_sound("enter_boss")
        print("🔔 사운드 재생됨 — 5초 후 입장 시퀀스 시작")
        time.sleep(5)

    restore_window(args.hwnd)
    ctl = InputController(args.hwnd, dry_run=not args.arm)
    ctl.kill.start()
    ctl.arm(args.arm)

    cap = ScreenCapture(args.hwnd, grab_hz=max(60, args.fps * 2))
    print("capture region:", cap.region, f"({cap.width}x{cap.height})")
    cap.start()
    model = load_policy()
    bridge = ObservationBridge()
    policy_name = "v7" if model is not None else "heuristic"

    logger = JsonlLogger(args.out)
    n = args.seconds * args.fps
    # 보스전 종료 자동 감지: 2초 연속 비보스전 + 보스전 경험 후
    autostop = not args.autostop_off
    seen_boss = False
    nonboss_streak = 0
    AUTOSTOP_FRAMES = args.fps * 2

    print(f"=== 관전 모드 v3 {args.seconds}s @ {args.fps}fps | arm={args.arm} | "
          f"policy={policy_name} | kill file: {ctl.kill.kill_file} ===")

    lat = {"parse": 0.0, "bridge": 0.0, "infer": 0.0, "loop": 0.0, "n": 0}
    end_reason = "time_limit"
    rec = VideoRecorder(args.out.replace(".json", "_aieye.mp4"),
                        cap.width, cap.height, args.fps) if args.record else None

    try:
        for i in range(n):
            t_loop = time.perf_counter()
            frame = cap.get()
            entry: dict = {"i": i, "t": round(time.time(), 3)}
            if frame is None:
                entry["state"] = "capture_warmup"
            else:
                t0 = time.perf_counter()
                res = parse_frame(frame)
                lat["parse"] += time.perf_counter() - t0
                entry["is_bossfight"] = bool(res.get("is_bossfight", False))

                # 보스전 종료 자동 감지
                if autostop:
                    if entry["is_bossfight"]:
                        seen_boss = True
                        nonboss_streak = 0
                    elif seen_boss:
                        nonboss_streak += 1
                        if nonboss_streak >= AUTOSTOP_FRAMES:
                            end_reason = "bossfight_ended"
                            logger.append(entry)
                            if rec is not None:
                                rec.write(frame, entry)
                            break

                obs = obs_from_parse(res)
                if obs is not None:
                    t0 = time.perf_counter()
                    vec = bridge.push(res)
                    lat["bridge"] += time.perf_counter() - t0
                    entry["obs"] = obs
                    if model is not None and vec is not None:
                        t0 = time.perf_counter()
                        action, _ = model.predict(vec, deterministic=True)
                        lat["infer"] += time.perf_counter() - t0
                        action = int(action)
                        entry["policy"] = "v7"
                    else:
                        action = heuristic_action(obs)
                        entry["policy"] = "heuristic"
                    fired, why = ctl.act(action)
                    entry.update({"action": action, "fired": fired, "gate": why})
                else:
                    entry["state"] = "waiting_for_bossfight"
                if rec is not None:
                    rec.write(frame, entry)
            lat["loop"] += time.perf_counter() - t_loop
            lat["n"] += 1
            logger.append(entry)

            if i % (args.fps * 5) == 0:
                k = "bossfight" if entry.get("is_bossfight") else \
                    ("warmup" if frame is None else "waiting")
                info = (f"act={entry.get('action')} policy={entry.get('policy', '-')}"
                        f" gate={entry.get('gate', '-')}") if "obs" in entry else ""
                print(f"[{i // args.fps:3d}s] {k} {info} kill={ctl.kill.is_set()}")

            if ctl.kill.is_set():
                print("!!! KILL SWITCH — 즉시 종료:", ctl.kill.reason())
                end_reason = "kill_switch"
                break

            dt = 1.0 / args.fps - (time.perf_counter() - t_loop)
            if dt > 0:
                time.sleep(dt)
    finally:
        ctl.arm(False)
        cap.stop()
        logger.close()
        if rec is not None:
            rec.close()
        release_lock(args.out + ".lock")

    nn = max(1, lat["n"])
    bf = sum(1 for e in logger._entries if e.get("is_bossfight"))
    acts = sum(1 for e in logger._entries if "action" in e)
    print(f"\n=== 종료 ({end_reason}) ===")
    print(f"프레임 {len(logger._entries)} | 보스전 판정 {bf} | 행동 결정 {acts}")
    print(f"평균 지연: parse {lat['parse']/nn*1000:.2f}ms + bridge {lat['bridge']/nn*1000:.2f}ms"
          f" + infer {lat['infer']/nn*1000:.2f}ms = 계산 {lat['loop']/nn*1000:.2f}ms/프레임")
    print(f"캡처: {cap.grab_count}회 성공, 오류 {cap.error_count}회")
    print(f"로그: {logger.jsonl_path} + {logger.snapshot_path}")
    if rec is not None:
        print(f"영상: {args.out.replace('.json', '_aieye.mp4')} ({rec.frames}프레임)")


if __name__ == "__main__":
    main()
