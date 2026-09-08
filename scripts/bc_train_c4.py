# -*- coding: utf-8 -*-
"""BC 사이클 4 — 실 검출 정밀화: 세로 기둥 컨투어 분리 (보스 본체와 구분).

붉은 컨투어 중 세로 종횡비 ≥ 2.0인 것만 "실"로 인식 → 레인 분포 정확도 상승 가설.
"""
from __future__ import annotations

import json
import numpy as np
import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

VID = r"reports\spectator_v3_run2_aieye.mp4"
DATA = r"reports\bc_dataset.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
STACK = 4
STEP = 2


def obs_vec(o: dict) -> list:
    return [
        o["g"] / 5.0, o["r"] / 5.0, o["d"] / 5.0,
        np.clip(o["dir"] / 6.0, -1, 1),
        np.clip(o["dist"] / 1300.0, 0, 1),
        1.0 if o["thread"] else 0.0,
        1.0 if o["warn"] else 0.0,
    ]


def thread_lanes(frame: np.ndarray) -> list:
    """세로 기둥 컨투어만 붉은 실로 인식해 레인 분포 반환 (+피크 위치)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask = m1 | m2
    mask[:130, :] = 0
    mask[640:, :] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w = frame.shape[1]
    lanes = [0] * 7
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 30 or cv2.contourArea(c) < 300:
            continue
        if ch / max(1, cw) < 2.0:   # 세로 기둥만
            continue
        lane = min(6, (x + cw // 2) * 7 // w)
        lanes[lane] += cv2.contourArea(c)
    peak = lanes.index(max(lanes)) / 7.0 if max(lanes) > 0 else 0.5
    return [min(1.0, v / 3000.0) for v in lanes] + [peak]


def main():
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    cap = cv2.VideoCapture(VID)
    lane_feats = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            ok, frame = cap.retrieve()
            if ok:
                lane_feats[i] = thread_lanes(frame)
        i += 1
    cap.release()
    print(f"thread-contour features: {len(lane_feats)} frames")

    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    rows4, labels = [], []
    for idx in range(STACK, len(samples)):
        s = samples[idx]
        if s["i"] not in lane_feats:
            continue
        feats = []
        for k in range(STACK, 0, -1):
            sp = samples[idx - k]
            sprev = samples[idx - k - 1] if idx - k - 1 >= 0 else sp
            dx = sp["x"] - sprev["x"]
            feats.append(np.clip(dx / 20.0, -1, 1))
            feats.append((sp["x"] - x_min) / max(1, x_max - x_min))
        feats += lane_feats[s["i"]] + obs_vec(s["obs"])
        rows4.append(feats)
        labels.append(ACT[s["action"]])
    X4 = np.array(rows4, dtype=np.float32)
    y = np.array(labels)
    print(f"features: c4={X4.shape[1]}, samples={len(y)}")

    vals, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / len(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c4 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=42, early_stopping=True)
    s4 = cross_val_score(c4, X4, y, cv=cv, scoring="accuracy")
    print(f"\n=== CV accuracy ===")
    print(f"cycle2 (baseline+stack): 0.642")
    print(f"cycle3 (+raw red lanes): 0.664")
    print(f"cycle4 (+thread contours): {s4.mean():.3f} ± {s4.std():.3f}")
    print(f"majority baseline:        {baseline:.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X4, y, test_size=0.2, stratify=y, random_state=42)
    c4.fit(Xtr, ytr)
    ypr = c4.predict(Xte)
    print("\nclassification report (cycle4 holdout):")
    print(classification_report(yte, ypr, target_names=["LEFT", "STAY", "RIGHT"], digits=3))

    import pickle
    with open(r"artifacts_bc\bc_policy_c4.pkl", "wb") as f:
        pickle.dump({"model": c4, "x_min": x_min, "x_max": x_max}, f)
    print("saved: artifacts_bc/bc_policy_c4.pkl")


if __name__ == "__main__":
    main()
