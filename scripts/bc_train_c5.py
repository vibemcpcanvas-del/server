# -*- coding: utf-8 -*-
"""BC 사이클 5 — 리서치 반영: 액션 청크 예측 + 클래스 불균형 처치 + macro-F1 주지표.

리서치 근거 (reports/BC_CYCLE5_PLAN.md):
- Action Chunking (arXiv 2507.09061): 1스텝 예측의 compounding error → 청크 예측
- SAIL: trivial STAY local minimum → 클래스 가중 + macro-F1 주지표
- c4(0.650) 대비 측정. 이동 recall 유지+향상이 성공 판정.
"""
from __future__ import annotations

import json
import numpy as np
import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score

VID = r"reports\spectator_v3_run2_aieye.mp4"
DATA = r"reports\bc_dataset.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
STACK = 4
STEP = 2
CHUNK = 3          # 액션 청크 길이 (3스텝 = 200ms)
MOVE_W = 3.0       # 이동 클래스 가중


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

    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    rows, labels, chunk_labels = [], [], []
    for idx in range(STACK, len(samples) - CHUNK * 3):
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
        lf = lane_feats[s["i"]]
        lanes_norm = [min(1.0, v / 3000.0) for v in lf]
        peak = lanes_norm.index(max(lanes_norm)) / 7.0
        feats += lanes_norm + [peak] + obs_vec(s["obs"])
        rows.append(feats)
        # 현재 행동 라벨
        labels.append(ACT[s["action"]])
        # 청크 라벨: 미래 9프레임(3스텝)의 행동 시퀀스를 tuple로
        future = samples[idx + 1: idx + 1 + CHUNK * 3]
        chunk = tuple(ACT[f["action"]] for f in future[::3])
        chunk_labels.append(chunk)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    chunks = np.array(chunk_labels)
    print(f"dataset: {X.shape[0]} samples, features {X.shape[1]}")

    # 클래스 가중
    vals, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / len(y)
    # 균형 샘플링: 이동 클래스 오버샘플
    idx_move = np.where(y != ACT["STAY"])[0]
    idx_stay = np.where(y == ACT["STAY"])[0]
    n_stay_keep = len(idx_move)  # 이동 수만큼만 STAY 유지 → 균형
    rng = np.random.RandomState(42)
    idx_stay_sampled = rng.choice(idx_stay, size=min(n_stay_keep, len(idx_stay)), replace=False)
    idx_bal = np.concatenate([idx_move, idx_stay_sampled])
    rng.shuffle(idx_bal)
    Xb, yb, cb = X[idx_bal], y[idx_bal], chunks[idx_bal]
    print(f"balanced: {len(Xb)} samples (STAY {np.sum(yb==1)}, LEFT {np.sum(yb==0)}, RIGHT {np.sum(yb==2)})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 사이클 5: MLP + 클래스 가중 (balanced data라 불필요할 수 있으나 유지)
    c5 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=42, early_stopping=True)
    s5 = cross_val_score(c5, Xb, yb, cv=cv, scoring="f1_macro")
    a5 = cross_val_score(c5, Xb, yb, cv=cv, scoring="accuracy")
    print(f"\n=== 사이클 5 (균형 샘플링 + 청크 데이터) ===")
    print(f"macro-F1: {s5.mean():.3f} ± {s5.std():.3f}")
    print(f"accuracy: {a5.mean():.3f} ± {a5.std():.3f}")

    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=42)
    c5.fit(Xtr, ytr)
    ypr = c5.predict(Xte)
    print("\nclassification report (cycle5 holdout, balanced):")
    print(classification_report(yte, ypr, target_names=["LEFT", "STAY", "RIGHT"], digits=3))

    # 청크 예측 평가 (미래 3스텝 majority 행동)
    chunk_maj = np.array([np.bincount(c).argmax() for c in cb])
    c5c = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=42, early_stopping=True)
    s5c = cross_val_score(c5c, Xb, chunk_maj, cv=cv, scoring="accuracy")
    print(f"\n청크(majority-of-3) 예측 accuracy: {s5c.mean():.3f}")

    import pickle
    with open(r"artifacts_bc\bc_policy_c5.pkl", "wb") as f:
        pickle.dump({"model": c5, "x_min": x_min, "x_max": x_max}, f)
    print("saved: artifacts_bc/bc_policy_c5.pkl")


if __name__ == "__main__":
    main()
