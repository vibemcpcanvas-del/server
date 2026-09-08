# -*- coding: utf-8 -*-
"""행동 복제(BC) 훈련 — 표준 ML 최소 파이프라인.

데이터: reports/bc_dataset.jsonl (4,162 샘플 — 전문가(사용자)의 실전 이동)
모델: sklearn LogisticRegression (표준 분류기 — 훈련 1초, 해석 가능, 하이퍼파라미터 최소)
평가: 5-fold 교차검증 + 클래스별 정밀도/재현율 + v7 대조

이 스크립트의 목적:
- 하드코딩 시뮬이 아니라 **실전 데이터에서 학습**했음을 측정으로 증명
- v7 (시뮬 훈련 PPO)과 전문가 데이터 정합성 비교 — "v7은 전문가와 얼마나 다른가"
"""
from __future__ import annotations

import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

DATA = r"reports\bc_dataset.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
ACT_NAME = {v: k for k, v in ACT.items()}


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
    X = np.array([obs_vec(s["obs"]) for s in samples], dtype=np.float32)
    y = np.array([ACT[s["action"]] for s in samples])

    # 1) 표준 교차검증
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"=== BC 모델 5-fold CV ===")
    print(f"accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # 2) holdout 상세
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    print("\nclassification report (holdout):")
    print(classification_report(yte, ypr, target_names=["LEFT", "STAY", "RIGHT"], digits=3))
    print("confusion matrix (rows=true):")
    print(confusion_matrix(yte, ypr))

    # 계수 해석 — 무엇이 행동을 결정하는가
    print("\n=== 계수 (logistic, LEFT/STAY/RIGHT 각각) ===")
    feats = ["green", "red", "destroyed", "boss_dir", "boss_dist", "thread", "warn"]
    for ci, cname in enumerate(["LEFT", "STAY", "RIGHT"]):
        terms = ", ".join(f"{f}={clf.coef_[ci][fi]:+.2f}" for fi, f in enumerate(feats))
        print(f"  {cname}: {terms}")

    # 3) v7 대조 — v7이라면 이 데이터에서 어떤 행동을 골랐을까
    # v7의 관전 기록: act 6(우)=79.4%, act 2(좌)=17.5% → RIGHT/LEFT 편향
    # 전문가 데이터: STAY가 62.9% — v7은 전문가와 확연히 다름
    v7_map = {6: "RIGHT", 2: "LEFT", 0: "LEFT", 8: "RIGHT", 4: "STAY", 9: None}
    import collections
    v7log = [json.loads(l) for l in open(r"reports\spectator_v3_run2.jsonl", encoding="utf-8")]
    v7acts = collections.Counter(v7_map.get(e.get("action")) for e in v7log if e.get("obs"))
    tot_v7 = sum(v for k, v in v7acts.items() if k)
    print("\n=== v7 vs 전문가 분포 대조 ===")
    for name in ("LEFT", "STAY", "RIGHT"):
        exp_n = sum(1 for s in samples if s["action"] == name)
        v7_n = v7acts.get(name, 0)
        print(f"  {name}: 전문가 {exp_n/len(samples)*100:.1f}% vs v7 {v7_n/tot_v7*100:.1f}%")

    # 4) 모델 저장
    import pickle
    with open(r"artifacts_bc\bc_policy.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("\nsaved: artifacts_bc/bc_policy.pkl")
    # 학습 곡선 요약 (train acc)
    print(f"train accuracy: {clf.score(Xtr, ytr):.3f}, test accuracy: {clf.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
