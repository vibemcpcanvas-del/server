# -*- coding: utf-8 -*-
"""R2 — 실화면 비전 파서 v2.1 (해상도 정규화 버전).

v2 캘리브레이션 (C시리즈 1366x768 실측) + v2.1 신규:
- **입력 프레임을 기준 해상도(1366x768)로 리사이즈 후 파싱** — 창 크기/위치
  무관하게 동작 (보스 입장 실측: 창이 1382x807로 이동해도 해골 UI 인식).
  리사이즈 비용: 1366x768 -> 1366x768은 no-op, 다른 크기는 cv2.resize 1회 (~1ms).
- 색 임계값·3클래스 해골 분류는 GT 실측값 그대로 유지.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

BASE_W, BASE_H = 1366, 768

# ── 고정 UI 관심영역 (1366x768 클라이언트 캡처 기준) ──
# 2026-09-07 실보스전 실측: 해골 캡슐이 남은시간 우측 x795..965, y30..75에 위치
# (구 레이아웃 x600..765는 이전 패치 — 폐기)
ROI_SKULLS = (795, 30, 965, 75)       # x0,y0,x1,y1
ROI_TIMER = (640, 126, 726, 152)
ROI_BOSS_HP = (270, 2, 1090, 14)
PLAY_AREA = (0, 120, 1366, 660)       # 플레이 영역 (UI 제외)
LANE_W = 280.0
SUBLANE_W = LANE_W / 3.0


def _hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def detect_is_bossfight(img: np.ndarray) -> bool:
    """보스전 판별 — HP바 또는 해골 캡슐 존재.

    v2.6: Practice 모드는 보스 HP바가 없음 → 해골 캡슐(3클래스 해골 2개 이상
    관측)으로도 보스전 판정. HP바가 있으면 즉시 True.
    """
    # 1차: 보스 HP바 (핑크 풀폭)
    x0, y0, x1, y1 = ROI_BOSS_HP
    roi = img[y0:y1, x0:x1]
    h = _hsv(roi)
    pink = cv2.inRange(h, (140, 60, 120), (175, 255, 255))
    red = cv2.inRange(h, (0, 60, 120), (10, 255, 255))
    frac = (pink.sum() + red.sum()) / 255.0 / ((x1 - x0) * (y1 - y0))
    if frac > 0.25:
        return True
    # 2차: 해골 캡슐 (신규 ROI에서 실제 관측 해골 2개 이상 — 파괴만 있으면 빈 프레임)
    sk = _detect_skulls_fixed(img)
    return sk["skulls_seen"] >= 2


def detect_skulls(img: np.ndarray) -> dict:
    """해골 5슬롯 3클래스 — 삼중 경로 (v2.7).

    실측된 두 레이아웃을 모두 지원:
    ① 신규 ROI(795..965, y30..75) — 2026-09-07 실보스전 레이아웃
    ② 구 ROI(600..765, y88..118) — 이전 패치 레이아웃 (GT 95/95)
    관측 픽셀이 많은 쪽을 신뢰 (레이아웃 자동 판별).
    3클래스 판정: 핑크 H150..180&S>=150 / 초록 H25..62&V>=100&V_med>=96 / 파괴=나머지
    """
    new = _detect_skulls_fixed(img)                       # 신규 ROI
    old_roi_img = _shift_legacy(img)                      # 구 ROI 영역 크롭+스케일
    old = _detect_skulls_fixed_on(old_roi_img) if old_roi_img is not None else None

    new_seen = new["skulls_seen"]
    old_seen = old["skulls_seen"] if old else 0
    if old_seen > new_seen:
        return old
    return new


def _shift_legacy(img: np.ndarray) -> np.ndarray | None:
    """구 레이아웃 ROI(600..765, y88..118) 크롭 — 그대로 반환(좌표 동일 해상도 가정)."""
    ch, cw = img.shape[:2]
    if cw < 766 or ch < 119:
        return None
    return img[88:118, 600:766]


def _detect_skulls_fixed_on(roi: np.ndarray) -> dict:
    """주어진 ROI 크롭에서 5슬롯 3클래스 판정 (규칙 공유)."""
    w = roi.shape[1] / 5.0
    green = red = destroyed = 0
    for i in range(5):
        s = roi[:, int(i * w):int((i + 1) * w)]
        hs = _hsv(s)
        pink_px = int(cv2.inRange(hs, (150, 150, 100), (180, 255, 255)).sum()) // 255
        v = hs[:, :, 2]
        vmask = v > 60
        v_med = float(np.median(v[vmask])) if vmask.sum() > 20 else 0.0
        h_med = float(np.median(hs[:, :, 0][vmask])) if vmask.sum() > 20 else 0.0
        if pink_px >= 100:
            red += 1
        elif 25 <= h_med <= 62 and v_med >= 96:
            green += 1
        else:
            destroyed += 1
    return {"green_skulls": green, "red_skulls": red,
            "skulls_destroyed": destroyed, "skulls_seen": green + red,
            "skull_slots": 5, "layout": "legacy"}


def _detect_skulls_dynamic(img: np.ndarray) -> dict | None:
    """동적 경로: 타이머 앵커 + 블롭 피크 (레이아웃 변형 대응). 실패 시 None.

    v2.6 판정 규칙 (라이브 실측):
      r(핑크): H150..180 & S>=150 & V>=100
      g(초록): H25..62 & S>=140 & V>=100
      d(파괴): 위 어느 것도 아님 (무채색·어두움)
    """
    ch, cw = img.shape[:2]
    y1 = int(ch * 0.12)
    strip = img[0:y1, :]
    hs = _hsv(strip)
    g_mask = cv2.inRange(hs, (35, 90, 120), (60, 255, 255))
    p_mask = cv2.inRange(hs, (150, 140, 120), (180, 255, 255))
    skull_mask = cv2.bitwise_or(g_mask, p_mask)

    cols = skull_mask.sum(axis=0)  # x별 픽셀 수
    total_active = int((cols > 60).sum())
    if total_active < 40:
        # 동적 탐지 실패 → 구 고정 ROI 폴백
        return _detect_skulls_fixed(img)

    # 앵커: 남은시간 텍스트(큰 흰 숫자). 해골 UI는 그 인접(±0.35폭)에만 존재.
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(gray, 200, 255)
    wcols = white.sum(axis=0)
    wwin = int(cw * 0.15)
    if wwin < 30:
        return _detect_skulls_fixed(img)
    wcsum = np.concatenate([[0], np.cumsum(wcols)])
    wa, wbest = 0, -1.0
    for a in range(0, cw - wwin):
        d = float(wcsum[a + wwin] - wcsum[a])
        if d > wbest:
            wbest, wa = d, a
    timer_cx = wa + wwin // 2

    # 해골 블롭 피크 5개 직접 탐지: 스무딩한 히스토그램에서 국소 최대치.
    # 탐색 범위 타이머 중심 ±22%폭 (실측: 해골 5개는 타이머 옆 200px에 밀집,
    # 첫 해골이 타이머보다 왼쪽인 케이스도 커버)
    kernel = np.ones(int(cw * 0.01) | 1, dtype=float)
    smooth = np.convolve(cols, kernel / kernel.sum(), mode="same")
    lo = max(0, timer_cx - int(cw * 0.22))
    hi = min(cw, timer_cx + int(cw * 0.22))
    if hi - lo < 60:
        return _detect_skulls_fixed(img)
    min_h = max(200.0, float(smooth[lo:hi].max()) * 0.25)
    peaks = []
    i = lo
    while i < hi:
        if smooth[i] >= min_h and smooth[i] == smooth[max(lo, i - 20):i + 20].max():
            if not peaks or i - peaks[-1] > cw * 0.03:
                peaks.append(i)
        i += 1
    if len(peaks) < 5:
        # 피크 부족 → 앵커 근처 균등 5분할 폴백
        win = int(cw * 0.16)
        a0 = max(0, min(cw - win, timer_cx - win // 2 - int(cw * 0.02)))
        peaks = [a0 + int((i + 0.5) * win / 5) for i in range(5)]
    peaks = peaks[:5]
    # 피크 중심 기준 반폭 슬롯 (피크 간격 = 해골 간격)
    if len(peaks) >= 2:
        spacing = int(np.median(np.diff(peaks[:5]))) if len(peaks) >= 5 else int(cw * 0.045)
    else:
        spacing = int(cw * 0.045)
    half = max(14, spacing // 2)
    # 판정 밴드: 각 피크의 y 중심(해골 블롭 y위치) ±22px — UI 프레임 테두리 제외
    # y밴드는 g_mask(초록) 기준 — 핑크 마스크는 남은시간 텍스트 오염 가능 (74s 실측)
    band_ys = []
    peak_ys = []
    for cx in peaks[:5]:
        colband = g_mask[:, max(0, cx - 15):cx + 15]
        ys = np.where(colband.sum(axis=1) > 30)[0]
        by0 = int(ys.min()) if len(ys) else 20
        by1 = int(ys.max()) if len(ys) else 70
        band_ys.append((by0, by1))
        peak_ys.append((by0 + by1) / 2.0)
    peaks_y_std = float(np.std(peak_ys)) if len(peak_ys) >= 3 else 99.0

    green = red = destroyed = 0
    centers = peaks[:5]
    for (cx, (by0, by1)) in zip(centers, band_ys):
        by0 = max(0, by0 - 2)
        by1 = min(strip.shape[0], by1 + 3)
        slot = strip[by0:by1, max(0, cx - half):cx + half]
        shs = _hsv(slot)
        # r(핑크): 고채도 마젠타 (라이브 실측 S 191~198)
        pink_px = int(cv2.inRange(shs, (150, 150, 100), (180, 255, 255)).sum()) // 255
        v = shs[:, :, 2]
        vmask = v > 60
        v_med = float(np.median(v[vmask])) if vmask.sum() > 20 else 0.0
        h_med = float(np.median(hs_slot_h(shs, vmask))) if vmask.sum() > 20 else 0.0
        s_med = float(np.median(hs_slot_s(shs, vmask))) if vmask.sum() > 20 else 0.0
        if pink_px >= 100:
            red += 1
        elif 25 <= h_med <= 95 and v_med >= 100:
            green += 1
        else:
            destroyed += 1
    return {"green_skulls": green, "red_skulls": red,
            "skulls_destroyed": destroyed, "skulls_seen": green + red,
            "skull_slots": len(centers), "skull_peaks": [int(p) for p in centers],
            "peaks_y_std": peaks_y_std,
            "timer_cx": timer_cx, "layout": "dynamic"}


def hs_slot_h(hsv_slot: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return hsv_slot[:, :, 0][mask]


def hs_slot_s(hsv_slot: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return hsv_slot[:, :, 1][mask]


def _detect_skulls_fixed(img: np.ndarray) -> dict:
    """신규 레이아웃 ROI(795..965, y30..75) — 라이브 실측 기반."""
    x0, y0, x1, y1 = ROI_SKULLS
    roi = img[y0:y1, x0:x1]
    return _detect_skulls_fixed_on(roi)


def detect_soul_split(img: np.ndarray) -> dict:
    """SS 타이머 존재 + WARNING 텍스트."""
    x0, y0, x1, y1 = ROI_TIMER
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    digits = int((g > 200).sum())
    wy0, wy1 = y1, min(BASE_H, y1 + 26)
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
    """1프레임 전체 파싱 — 실시간 루프의 눈.

    v2.1: 입력이 기준 해상도가 아니면 리사이즈해서 파싱 (창 크기 무관).
    """
    if img.shape[1] != BASE_W or img.shape[0] != BASE_H:
        img = cv2.resize(img, (BASE_W, BASE_H), interpolation=cv2.INTER_AREA)
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
    t0 = time.perf_counter()
    res = parse_frame(img)
    return res, (time.perf_counter() - t0) * 1000.0
