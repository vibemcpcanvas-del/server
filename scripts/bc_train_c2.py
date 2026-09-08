# -*- coding: utf-8 -*-
"""BC 사이클 2 — 관측 확장 (player_x 정규화 + 프레임 스택 dx 히스토리) + MLP.

사이클 1 (0.351) 대비 측정. 기준선: majority 0.629.
가설: 위치+이동 히스토리가 들어오면 전문가 행동 예측이 유의미하게 올라간다.
"""
from __future__ import annotations

import json
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

DATA = r"reports\bc_dataset.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
STACK = 4


def obs_vec(o: dict) -> list:
    return [
        o["g"] / 5.0, o["r"] / 5.0, o["d"] / 5.0,
        np.clip(o["dir"] / 6.0, -1, 1),
        np.clip(o["dist"] / 1300.0, 0, 1),
        1.0 if o["thread"] else 0.0,
        1.0 if o["warn"] else 0.0,
    ]


def main():
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    # 프레임 스택: 최근 STACK개의 (x 정규화, dx) 히스토리 + 현재 obs
    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    rows, labels = [], []
    for idx in range(STACK, len(samples)):
        feats = []
        for k in range(STACK, 0, -1):
            s = samples[idx - k]
            s_prev = samples[idx - k - 1] if idx - k - 1 >= 0 else s
            dx = s["x"] - s_prev["x"]
            feats.append(np.clip(dx / 20.0, -1, 1))               # 이동 속도
            feats.append((s["x"] - x_min) / max(1, x_max - x_min))  # 화면 내 위치
        feats.extend(obs_vec(samples[idx]["obs"]))
        rows.append(feats)
        labels.append(ACT[samples[idx]["action"]])
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    print(f"dataset: {X.shape[0]} samples x {X.shape[1]} features (obs 7 + stack {STACK}x2)")

    # majority baseline
    vals, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / len(y)
    print(f"majority baseline: {baseline:.3f}")

    # 사이클 1 대조 (동일 obs만) — 공정 비교
    X1 = np.array([obs_vec(s["obs"]) for s in samples[STACK:]], dtype=np.float32)
    y1 = y
    from sklearn.linear_model import LogisticRegression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s1 = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                         X1, y1, cv=cv, scoring="accuracy")

    # 사이클 2: MLP + 확장 관측
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1500, random_state=42,
                        early_stopping=True)
    s2 = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n=== CV accuracy ===")
    print(f"cycle1 (7 obs, Logistic): {s1.mean():.3f} ± {s1.std():.3f}")
    print(f"cycle2 ({X.shape[1]} feats, MLP):    {s2.mean():.3f} ± {s2.std():.3f}")
    print(f"majority baseline:      {baseline:.3f}")

    # holdout 상세
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    print("\nclassification report (holdout):")
    print(classification_report(yte, ypr, target_names=["LEFT", "STAY", "RIGHT"], digits=3))
    print(f"train acc: {clf.score(Xtr, ytr):.3f}, test acc: {clf.score(Xte, yte):.3f}")

    import pickle, os
    os.makedirs(r"artifacts_bc", exist_ok=True)
    with open(r"artifacts_bc\bc_policy_c2.pkl", "wb") as f:
        pickle.dump({"model": clf, "x_min": x_min, "x_max": x_max}, f)
    print("saved: artifacts_bc/bc_policy_c2.pkl")


if __name__ == "__main__":
    main()
