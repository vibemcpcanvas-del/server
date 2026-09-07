# -*- coding: utf-8 -*-
"""v7 정책 시뮬 플레이 영상 렌더러.

검증된 구현 방식 (커뮤니티 표준):
- Gymnasium 공식 문서의 render 관례를 따르되, 인코딩은 matplotlib 프레임 ->
  ffmpeg 파이프(이미지 시퀀스 표준 파이프라인). OpenCV VideoWriter(MJPG)보다
  H.264 호환성이 좋고 중간 파일이 없음.
- 7레인 맵 105200310 실측 지오메트리(x -945..1035)를 화면 좌표로 스케일.
- 스프라이트 대신 검증된 형태(해골/제단/실) 아이콘 렌더 — 게임 그래픽 저작물
  재사용 최소화(파생물 리스크 관리).
- HUD: 보상 누적, 초록/빨간/파괴 해골, Soul Split 카운트다운, 페이즈.
"""
import os
import sys
import subprocess

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")
os.chdir(r"C:\Users\ROCmAdmin\Desktop\test\server")

import numpy as np
import cv2
from stable_baselines3 import PPO
from env.boss_env_base import BossProfile
from env.verus_hilla_v3 import VerusHillaEnvV3

W, H, FPS = 1280, 720, 30
SCALE = 0.42  # 게임 60tps -> 영상 30fps (2틱당 1프레임, 사람이 볼 속도)

PROFILE_PATH = r"env\profiles\verus_hilla.json"
OUT_DIR = r"reports"
profile = BossProfile.load(PROFILE_PATH)

# 맵 실측: x -945..1035 (1980px), 7레인, y=135 단일 평면
LANE_W = 1980 / 7.0
def lane_x(lane: float) -> int:
    """레인(소수 허용) -> 화면 x."""
    return int((lane * LANE_W) / 1980 * (W - 80) + 40)

GROUND_Y = int(H * 0.72)

COLORS = {
    "bg": (18, 14, 26),
    "lane": (60, 50, 80),
    "sub": (45, 38, 60),
    "boss": (60, 60, 220),      # BGR 진홍
    "boss_aura": (200, 160, 60),
    "player": (240, 240, 240),
    "green": (90, 220, 120),
    "pink": (180, 90, 220),
    "olive": (60, 90, 90),
    "altar": (200, 160, 60),
    "candle": (60, 160, 250),
    "thread": (70, 70, 230),
    "telegraph": (90, 200, 200),
    "bone": (200, 200, 90),
    "text": (230, 230, 230),
}


def draw_hud(img, st, total_r, info):
    bar_h = 54
    cv2.rectangle(img, (0, 0), (W, bar_h), (30, 22, 16), -1)
    cv2.putText(img, f"reward {total_r:7.2f}", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["text"], 2)
    # 해골 상태
    g = st.green_skulls
    r = st.red_skulls
    d = 5 - g - r
    x = 320
    for _ in range(g):
        cv2.circle(img, (x, 27), 14, COLORS["green"], -1); x += 34
    for _ in range(r):
        cv2.circle(img, (x, 27), 14, COLORS["pink"], -1); x += 34
    for _ in range(d):
        cv2.circle(img, (x, 27), 14, COLORS["olive"], -1); x += 34
    ss = st.soul_split_seconds_remaining
    if ss < 900000:
        col = (80, 80, 255) if ss < 10 else COLORS["text"]
        cv2.putText(img, f"SoulSplit {ss:5.1f}s", (560, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(img, f"P{st.phase}  HP {st.boss_hp_pct:4.1f}%",
                (W - 260, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["text"], 2)


def render_frame(st, info, total_r) -> np.ndarray:
    img = np.full((H, W, 3), COLORS["bg"], dtype=np.uint8)
    # 7레인 바닥
    for lane in range(7):
        x0, x1 = lane_x(lane), lane_x(lane + 1)
        cv2.rectangle(img, (x0 + 3, GROUND_Y - 8), (x1 - 3, GROUND_Y + 26),
                      COLORS["lane"], -1)
        cv2.line(img, (x1, GROUND_Y - 30), (x1, GROUND_Y + 30), COLORS["sub"], 1)
    # 붉은 실: 예고(연두) -> 활성(빨강 세로 기둥)
    for lane in st.thread_telegraph_lanes:
        x0, x1 = lane_x(lane), lane_x(lane + 1)
        cv2.rectangle(img, (x0 + 6, 90), (x1 - 6, GROUND_Y + 20),
                      COLORS["telegraph"], 2)
    for lane in st.thread_danger_lanes:
        x0, x1 = lane_x(lane), lane_x(lane + 1)
        cx = (x0 + x1) // 2
        cv2.rectangle(img, (cx - 14, 90), (cx + 14, GROUND_Y + 20),
                      COLORS["thread"], -1)
    # 뼈 파동 위험 레인
    if st.bone_wave_active:
        for lane in st.bone_wave_danger_lanes:
            x0, x1 = lane_x(lane), lane_x(lane + 1)
            cv2.rectangle(img, (x0 + 8, 120), (x1 - 8, GROUND_Y + 18),
                          COLORS["bone"], 2)
    # 제단
    if st.altar_present and st.altar_lane is not None:
        x0, x1 = lane_x(st.altar_lane), lane_x(st.altar_lane + 1)
        cx = (x0 + x1) // 2
        cv2.drawMarker(img, (cx, GROUND_Y - 40), COLORS["altar"],
                       cv2.MARKER_TILTED_CROSS, 42, 5)
        cv2.putText(img, "ALTAR", (cx - 34, GROUND_Y - 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["altar"], 2)
    # 촛불
    for i in range(st.candles_lit):
        cv2.circle(img, (60 + i * 22, H - 46), 8, COLORS["candle"], -1)
    # 보스 + 플레이어
    bx = (lane_x(st.boss_lane) + lane_x(st.boss_lane + 1)) // 2
    px = (lane_x(st.player_lane) + lane_x(st.player_lane + 1)) // 2
    px += int((st.player_sublane - 1) * LANE_W / 3 / 1980 * (W - 80))
    cv2.circle(img, (bx, GROUND_Y - 46), 26, COLORS["boss_aura"], 3)
    cv2.circle(img, (bx, GROUND_Y - 46), 20, COLORS["boss"], -1)
    cv2.circle(img, (px, GROUND_Y - 20), 16, COLORS["player"], -1)
    # 조작 표시
    act = info.get("last_action")
    if act is not None:
        cv2.putText(img, f"action {act}", (px - 40, GROUND_Y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["player"], 2)
    draw_hud(img, st, total_r, info)
    return img


def record(model_path: str, out_mp4: str, episodes: int = 3, seed0: int = 9000):
    model = PPO.load(model_path, device="cpu")
    ff = r"C:\Users\ROCmAdmin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
    proc = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", out_mp4],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    profile2 = BossProfile.load(PROFILE_PATH)
    for ep in range(episodes):
        env = VerusHillaEnvV3(profile2, seed=seed0 + ep, max_steps=900)
        obs, info = env.reset(seed=seed0 + ep)
        total_r, steps, done = 0.0, 0, False
        last_action = None
        while not done and steps < 900:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            done = term or trunc
            info = dict(info)
            info["last_action"] = action
            frame = render_frame(env.state, info, total_r)
            # 2틱 유지 (60tps -> 30fps)
            proc.stdin.write(frame.tobytes())
            proc.stdin.write(frame.tobytes())
        reason = info.get("termination_reason") or "time_limit"
        print(f"ep{ep}: steps={steps} reward={total_r:.1f} reason={reason}")
        # 에피소드 사이 1초 블랙
        blk = np.zeros((H, W, 3), dtype=np.uint8)
        for _ in range(FPS):
            proc.stdin.write(blk.tobytes())
    proc.stdin.close()
    proc.wait()
    print("saved:", out_mp4, os.path.getsize(out_mp4), "bytes")


if __name__ == "__main__":
    out = os.path.join(OUT_DIR, "v7_sim_play.mp4")
    record(r"artifacts_verus_curriculum\verus_curriculum_v7_balanced.zip",
           out, episodes=3)
