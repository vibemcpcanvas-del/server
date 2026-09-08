# -*- coding: utf-8 -*-
"""BC 사이클 3 — 위협 공간 위치 관측 추가 (붉은실 레인 분포).

관측 확장: 붉은 픽셀의 7레인 분포(정규화) — 파서 확장 대신 데이터 추출 단계에서
직접 계산 (표준 ML: 특징 엔지니어링은 데이터 파이프라인에서).
사이클 2 (0.642, 이동 recall 0) 대비 측정.
"""
from __future__ import annotations

import json
import numpy as np
import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

VID = r"reports\spectator_v3_run2_aieye.mp4"
DATA = r"reports\bc_dataset.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
STACK = 4
STEP = 2  # 15fps


def obs_vec(o: dict) -> list:
    return [
        o["g"] / 5.0, o["r"] / 5.0, o["d"] / 5.0,
        np.clip(o["dir"] / 6.0, -1, 1),
        np.clip(o["dist"] / 1300.0, 0, 1),
        1.0 if o["thread"] else 0.0,
        1.0 if o["warn"] else 0.0,
    ]


def red_lanes(frame: np.ndarray) -> list:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask = m1 | m2
    mask[:130, :] = 0
    mask[640:, :] = 0
    w = mask.shape[1]
    return [int(mask[:, i*w//7:(i+1)*w//7].sum() / 255) for i in range(7)]


def main():
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    print(f"samples: {len(samples)}")

    # 1) 붉은 레인 분포 추출 (15fps 샘플 위치에 대응)
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
                lane_feats[i] = red_lanes(frame)
        i += 1
    cap.release()
    print(f"lane features extracted: {len(lane_feats)} frames")

    # 2) 특징 구성: c2 특징(위치+속도 스택) + 붉은 레인 분포(현재+정규화)
    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    rows2, rows3, labels = [], [], []
    for idx in range(STACK, len(samples)):
        s = samples[idx]
        if s["i"] not in lane_feats:
            continue
        feats2, feats3 = [], []
        for k in range(STACK, 0, -1):
            sp = samples[idx - k]
            sprev = samples[idx - k - 1] if idx - k - 1 >= 0 else sp
            dx = sp["x"] - sprev["x"]
            feats2.append(np.clip(dx / 20.0, -1, 1))
            feats2.append((sp["x"] - x_min) / max(1, x_max - x_min))
        lf = lane_feats[s["i"]]
        max_lane = max(lf) or 1
        lanes_norm = [min(1.0, v / 3000.0) for v in lf]     # 위협 강도 레인별
        peak = lanes_norm.index(max(lanes_norm)) / 7.0       # 위협 피크 위치
        feats3 = feats2 + lanes_norm + [peak]
        rows2.append(feats2 + obs_vec(s["obs"]))
        rows3.append(feats3 + obs_vec(s["obs"]))
        labels.append(ACT[s["action"]])
    X2 = np.array(rows2, dtype=np.float32)
    X3 = np.array(rows3, dtype=np.float32)
    y = np.array(labels)
    print(f"features: c2={X2.shape[1]}, c3={X3.shape[1]}, samples={len(y)}")

    vals, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / len(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    c2 = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1500, random_state=42, early_stopping=True)
    s2 = cross_val_score(c2, X2, y, cv=cv, scoring="accuracy")
    c3 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=42, early_stopping=True)
    s3 = cross_val_score(c3, X3, y, cv=cv, scoring="accuracy")
    print(f"\n=== CV accuracy ===")
    print(f"cycle1 (obs only):      0.344")
    print(f"cycle2 (+pos/vel stack): {s2.mean():.3f} ± {s2.std():.3f}")
    print(f"cycle3 (+threat lanes):  {s3.mean():.3f} ± {s3.std():.3f}")
    print(f"majority baseline:       {baseline:.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X3, y, test_size=0.2, stratify=y, random_state=42)
    c3.fit(Xtr, ytr)
    ypr = c3.predict(Xte)
    print("\nclassification report (cycle3 holdout):")
    print(classification_report(yte, ypr, target_names=["LEFT", "STAY", "RIGHT"], digits=3))
    print(f"train acc: {c3.score(Xtr, ytr):.3f}, test acc: {c3.score(Xte, yte):.3f}")

    import pickle
    with open(r"artifacts_bc\bc_policy_c3.pkl", "wb") as f:
        pickle.dump({"model": c3, "x_min": x_min, "x_max": x_max}, f)
    print("saved: artifacts_bc/bc_policy_c3.pkl")


if __name__ == "__main__":
    main()
