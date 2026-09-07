# -*- coding: utf-8 -*-
"""R2 — 실화면 비전 파서 v2 (하이브리드: HSV 색추적 1차 + ROI 정밀 2차).

v2 PM 캘리브레이션 반영 (C_260903_013537_2 비전 실측):
- 해골 UI 고정 위치: x=600..765, y=90..115, 5슬롯
  * 초록 해골 = 노랑빛 녹색 (H 35..60)
  * 빨간 해골 = 핑크/마젠타 (H 140..180)
- SS 타이머: x=655..710, y=130..148 (흰색 숫자 존재성)
- WARNING: 타이머 주변 빨간 텍스트 픽셀 존재성
- 보스 HP바: y=2..12, x=270..1090 (핑크 바) → 보스전 판별
- 플레이어: 화면 하단부(y 430..660) 백색/크림 클러스터 (스킬바 y>680 제외)
- 보스: 플레이 영역 내 "빨간 머리(H0..10,고채도)" + 큰 블롭
- 붉은 실: 플레이 영역의 세로 빨간 기둥 (면적 기반)
- 서브레인: 280px/3 ≈ 93.3px

출력: 파싱 결과 dict + 소요시간(ms). 카메라 앵커 미해결(플레이어 상대 좌표 제공).
"""
from __future__ import annotations

import time
import cv2
import numpy as np

# ── 고정 UI 관심영역 (1366x768 실측) ──
ROI_SKULLS = (600, 88, 766, 118)      # x0,y0,x1,y1
ROI_TIMER = (640, 126, 726, 152)
ROI_BOSS_HP = (270, 2, 1090, 14)
PLAY_AREA = (0, 120, 1366, 660)       # 플레이 영역 (UI 제외)
LANE_W = 280.0
SUBLANE_W = LANE_W / 3.0


def _hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def detect_is_bossfight(img: np.ndarray) -> bool:
    """보스 HP바(핑크) 존재 → 보스전 화면."""
    x0, y0, x1, y1 = ROI_BOSS_HP
    roi = img[y0:y1, x0:x1]
    h = _hsv(roi)
    # 핑크/마젠타 계열 바
    pink = cv2.inRange(h, (140, 60, 120), (175, 255, 255))
    red = cv2.inRange(h, (0, 60, 120), (10, 255, 255))
    frac = (pink.sum() + red.sum()) / 255.0 / ((x1 - x0) * (y1 - y0))
    return frac > 0.25


def detect_skulls(img: np.ndarray) -> dict:
    """해골 5슬롯 3클래스 — GT 프레임 실측 통계 기반 (r2_skull_stats.py).

    실측 (6 GT 프레임 × 30슬롯, 2026-09-07):
      g(생존 초록):  V median 136 (66..159), H 31..169 — H는 불안정, V가 구분 축
      d(파괴 올리브): V median 72  (66..137), H/S는 g와 겹침
      r(빼앗김 핑크): H=169 고정, S 170..191, V 146..163
    규칙: H∈[160,180) & S≥150 → r / 슬롯 V-median ≥100 → g / 아니면 d
    V-median은 조명·오버랩에 강함(픽셀 카운트 대비).
    """
    x0, y0, x1, y1 = ROI_SKULLS
    roi = img[y0:y1, x0:x1]
    w = roi.shape[1] / 5.0
    green = red = destroyed = 0
    for i in range(5):
        s = roi[:, int(i * w):int((i + 1) * w)]
        hs = _hsv(s)
        # 핑크 판정: 고채도 마젠타 픽셀 수
        pink = cv2.inRange(hs, (160, 150, 120), (180, 255, 255))
        pink_px = int(pink.sum()) // 255
        # 슬롯 명도 중간값 (해골 픽셀 — 검은 배경 제외)
        v = hs[:, :, 2]
        vmask = v > 40
        v_med = float(np.median(v[vmask])) if vmask.sum() > 20 else float(v.mean())
        if pink_px >= 100:
            red += 1
        elif v_med >= 96.0:
            green += 1
        else:
            destroyed += 1
    return {"green_skulls": green, "red_skulls": red,
            "skulls_destroyed": destroyed, "skulls_seen": green + red}


def detect_soul_split(img: np.ndarray) -> dict:
    """SS 타이머 존재 + WARNING 텍스트."""
    x0, y0, x1, y1 = ROI_TIMER
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 흰색 숫자 픽셀
    digits = int((g > 200).sum())
    # WARNING: 타이머 아래 넓은 영역의 빨간 텍스트
    wy0, wy1 = y1, min(768, y1 + 26)
    wroi = img[wy0:wy1, 560:800]
    hw = _hsv(wroi)
    warn = cv2.inRange(hw, (0, 150, 150), (8, 255, 255))
    warn2 = cv2.inRange(hw, (172, 150, 150), (180, 255, 255))
    warn_px = (int(warn.sum()) + int(warn2.sum())) // 255
    return {
        "ss_timer_visible": digits > 60,
        "ss_timer_pixels": digits,
        "warning_visible": warn_px > 120,
        "warning_pixels": warn_px,
    }


def _largest_blob(mask: np.ndarray, min_area: int = 300):
    """최대 연결요소 중심 반환. 없으면 None."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best, best_a = None, min_area
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > best_a:
            best_a, best = a, i
    if best is None:
        return None
    cx, cy = cent[best]
    return float(cx), float(cy), int(best_a)


def detect_player(img: np.ndarray) -> dict:
    """플레이어: 하단부 흰색/크림 클러스터."""
    x0, y0, x1, y1 = PLAY_AREA[0], 430, PLAY_AREA[2], 660
    roi = img[y0:y1, x0:x1]
    h = _hsv(roi)
    white = cv2.inRange(h, (0, 0, 190), (180, 55, 255))
    blob = _largest_blob(white, min_area=250)
    if blob is None:
        return {"player_found": False}
    cx, cy, area = blob
    return {"player_found": True, "player_x": cx + x0, "player_y": cy + y0,
            "player_area": area}


def detect_boss(img: np.ndarray) -> dict:
    """보스: 빨간 머리(고채도 빨강, 플레이영역 상중단) + 주변 청록 아우라."""
    x0, y0, x1, y1 = PLAY_AREA[0], 240, PLAY_AREA[2], 620
    roi = img[y0:y1, x0:x1]
    h = _hsv(roi)
    hair = cv2.inRange(h, (0, 130, 110), (8, 255, 255))
    hair2 = cv2.inRange(h, (172, 130, 110), (180, 255, 255))
    mask = cv2.bitwise_or(hair, hair2)
    blob = _largest_blob(mask, min_area=180)
    if blob is None:
        return {"boss_found": False}
    cx, cy, area = blob
    return {"boss_found": True, "boss_x": cx + x0, "boss_y": cy + y0,
            "boss_area": area}


def detect_red_threads(img: np.ndarray) -> dict:
    """붉은 실: 플레이 영역의 세로 빨간 기둥 면적."""
    x0, y0, x1, y1 = PLAY_AREA
    roi = img[y0:y1, x0:x1]
    h = _hsv(roi)
    r = cv2.inRange(h, (0, 140, 100), (6, 255, 255))
    r2 = cv2.inRange(h, (174, 140, 100), (180, 255, 255))
    mask = cv2.bitwise_or(r, r2)
    px = int(mask.sum()) // 255
    return {"red_thread_present": px > 800, "red_thread_pixels": px}


def parse_frame(img: np.ndarray) -> dict:
    """1프레임 전체 파싱 — 실시간 루프의 눈."""
    out: dict = {}
    out["is_bossfight"] = detect_is_bossfight(img)
    if not out["is_bossfight"]:
        return out
    out.update(detect_skulls(img))
    out.update(detect_soul_split(img))
    out.update(detect_player(img))
    out.update(detect_boss(img))
    out.update(detect_red_threads(img))
    if out.get("player_found") and out.get("boss_found"):
        dx = out["boss_x"] - out["player_x"]
        lanes = dx / LANE_W
        out["boss_dir_lanes"] = round(lanes, 2)          # + = 보스가 오른쪽
        out["boss_distance_px"] = round(abs(dx), 1)
    return out


def parse_path(path: str) -> tuple[dict, float]:
    img = cv2.imread(path)
    if img is None:
        return {"error": "imread_failed", "is_bossfight": False}, 0.0
    if img.shape[:2] != (768, 1366):
        img = cv2.resize(img, (1366, 768))
    t0 = time.perf_counter()
    res = parse_frame(img)
    return res, (time.perf_counter() - t0) * 1000.0
