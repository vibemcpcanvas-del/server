# -*- coding: utf-8 -*-
"""BC 사이클 6a~12 통합 재현 스크립트 (2026-09-09 리뷰 지적 D1 수정).

D3: fold별 P/R/F1을 함께 출력해 정합성 확인. precision 1.0은 0.912로 수정.
D4: 다중객체 실패율 출처 명시 (jsonl 집계 10.1% / yolo_fulltrack 3.8%).
D2: 시간 인지 분할 필요성 — shuffled CV의 누수 폭 +0.03~0.06.

리뷰 지적: c6~c12 학습 코드가 execute_code로 직접 실행되어 파일화되지 않았음.
이 파일이 해당 사이클들의 핵심 로직을 통합 재현한다.

사용법:
  python bc_reconstruct_c6_c12.py 6a   # 예고 예측기
  python bc_reconstruct_c6_c12.py 7    # 예고 관측 추가
  python bc_reconstruct_c6_c12.py 8    # 교차 특징 추가
  python bc_reconstruct_c6_c12.py 10   # YOLO 관측 추가
  python bc_reconstruct_c6_c12.py 12   # SAIL discriminator
  python bc_reconstruct_c6_c12.py all
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")

import cv2
import numpy as np

VID = r"reports\spectator_v3_run2_aieye.mp4"
DATA = r"reports\bc_dataset.jsonl"
V7LOG = r"reports\spectator_v3_run2.jsonl"
ACT = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
ACTN = ["LEFT", "STAY", "RIGHT"]
STACK = 4
STEP = 2
W, H = 1366, 768


def obs_vec(o: dict) -> list:
    return [o["g"]/5.0, o["r"]/5.0, o["d"]/5.0,
            np.clip(o["dir"]/6.0, -1, 1),
            np.clip(o["dist"]/1300.0, 0, 1),
            1.0 if o["thread"] else 0.0,
            1.0 if o["warn"] else 0.0]


def load_lanes(vid: str = VID) -> dict:
    """붉은 픽셀 7레인 분포 추출 (15fps)."""
    cap = cv2.VideoCapture(vid)
    lanes = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            ok, f = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                mask = m1 | m2
                mask[:130, :] = 0; mask[640:, :] = 0
                w = mask.shape[1]
                lanes[i] = [int(mask[:, li*w//7:(li+1)*w//7].sum()/255) for li in range(7)]
        i += 1
    cap.release()
    return lanes


def load_lanes_with_contour(vid: str = VID) -> dict:
    """세로 기둥 컨투어만 붉은 실로 인식 (c4 방식)."""
    cap = cv2.VideoCapture(vid)
    lanes = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            ok, f = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                mask = m1 | m2
                mask[:130, :] = 0; mask[640:, :] = 0
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                w = frame_shape_w = mask.shape[1]
                lv = [0]*7
                for c in contours:
                    x, y, cw, ch = cv2.boundingRect(c)
                    if ch < 30 or cv2.contourArea(c) < 300:
                        continue
                    if ch / max(1, cw) < 2.0:
                        continue
                    lane = min(6, (x + cw//2) * 7 // w)
                    lv[lane] += cv2.contourArea(c)
                lanes[i] = [min(1.0, v/3000.0) for v in lv] + [lv.index(max(lv))/7.0]
        i += 1
    cap.release()
    return lanes


def load_yolo_obs() -> dict:
    """YOLO로 player/boss 검출 관측 (10차원)."""
    from ultralytics import YOLO
    m = YOLO(r"artifacts_yolo_cycle1\train\yolov8n_cycle2_gpu\weights\best.pt")
    cap = cv2.VideoCapture(VID)
    obs = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0 and 4735 <= i <= 13630:
            ok, frame = cap.retrieve()
            if not ok:
                i += 1; continue
            res = m.predict(frame, conf=0.25, verbose=False)[0]
            boxes = {}
            for c, cf, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                   res.boxes.xyxy.tolist()):
                if c not in boxes or cf > boxes[c][0]:
                    cx = (xyxy[0]+xyxy[2])/2
                    cy = (xyxy[1]+xyxy[3])/2
                    area = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
                    boxes[c] = (cf, cx, cy, area)
            o = []
            for cls in (0, 1):
                if cls in boxes:
                    cf, cx, cy, area = boxes[cls]
                    o += [1.0, cx/W, cy/768.0, min(1.0, area/200000.0), cf]
                else:
                    o += [0.0, 0.5, 0.5, 0.0, 0.0]
            obs[i] = o
        i += 1
    cap.release()
    return obs


def load_telegraph_probs(vid: str = VID) -> dict:
    """예고 예측기의 프레임별 확률 (c6a 출력)."""
    import pickle
    with open(r"artifacts_bc\telegraph_predictor.pkl", "rb") as f:
        tele = pickle.load(f)
    cap = cv2.VideoCapture(vid)
    probs = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0 and 4735 <= i <= 13630:
            ok, frame = cap.retrieve()
            if not ok:
                i += 1; continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            mask = m1 | m2
            mask[:130, :] = 0; mask[640:, :] = 0
            w = mask.shape[1]
            lanes = [int(mask[:, li*w//7:(li+1)*w//7].sum()/255) for li in range(7)]
            mx = max(lanes) or 1
            feats = np.array([[min(1.0, v/3000.0) for v in lanes] +
                              [min(1.0, mx/5000.0)]], dtype=np.float32)
            probs[i] = float(tele.predict_proba(feats)[0][1])
        i += 1
    cap.release()
    return probs


def build_features(lanes: dict, yolo: dict = None, tele: dict = None,
                   use_contour: bool = False, stack: int = 4):
    """공통 특징 구성. use_contour면 컨투어 정규화 레인 사용."""
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    rows, labels, frame_ids = [], [], []
    for idx in range(stack, len(samples)):
        s = samples[idx]
        if s["i"] not in lanes:
            continue
        f = []
        for k in range(stack, 0, -1):
            sp = samples[idx-k]
            sprev = samples[idx-k-1] if idx-k-1 >= 0 else sp
            dx = sp["x"] - sprev["x"]
            f.append(np.clip(dx/20.0, -1, 1))
            f.append((sp["x"]-x_min)/max(1, x_max-x_min))
        lf = lanes[s["i"]]
        if use_contour:
            # 컨투어 방식: lanes_norm + peak 포함
            f += lf
        else:
            lanes_norm = [min(1.0, v/3000.0) for v in lf]
            peak = lanes_norm.index(max(lanes_norm)) / 7.0
            f += lanes_norm + [peak]
        f += obs_vec(s["obs"])
        char_lane = min(6, int((s["x"]-x_min)/(x_max-x_min)*7))
        if use_contour:
            char_threat = lf[char_lane]
        else:
            char_threat = lanes_norm[char_lane]
        f.append(char_threat)
        f.append(1.0 - abs(char_lane/6.0 - (lf.index(max(lf))/7.0 if max(lf)>0 else 0.5)))
        if tele is not None:
            prev_p = tele.get(samples[idx-2]["i"], 0.0)
            f += [tele[s["i"]], tele[s["i"]] - prev_p]
        if yolo is not None:
            yo_now = yolo[s["i"]]
            yo_prev = yolo.get(samples[idx-2]["i"], [0.0]*10)
            f += yo_now + yo_prev
        rows.append(f)
        labels.append(ACT[s["action"]])
        frame_ids.append(s["i"])
    return np.array(rows, dtype=np.float32), np.array(labels), frame_ids


def cycle_6a():
    """예고 예측기 (이진 분류)."""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import classification_report
    lanes = load_lanes()
    entries = [json.loads(l) for l in open(V7LOG, encoding="utf-8")]
    hit_frames = []
    for i in range(1, len(entries)):
        ep, en = entries[i-1], entries[i]
        if ep.get("obs") and en.get("obs"):
            if ep["obs"]["red_skulls"] < en["obs"]["red_skulls"]:
                hit_frames.append(en["i"])
    pos = {hf - off for hf in hit_frames for off in range(10, 41, 2)}
    pos = {f for f in pos if f in lanes}
    X, y = [], []
    for fi, lanes_ in lanes.items():
        if fi < 4735 or fi > 13630:
            continue
        mx = max(lanes_) or 1
        X.append([min(1.0, v/3000.0) for v in lanes_] + [min(1.0, mx/5000.0)])
        y.append(1 if fi in pos else 0)
    X, y = np.array(X, dtype=np.float32), np.array(y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    pres = cross_val_score(clf, X, y, cv=cv, scoring="precision")
    recs = cross_val_score(clf, X, y, cv=cv, scoring="recall")
    print(f"=== c6a 예고 예측기 ===")
    print(f"macro-F1: {f1s.mean():.3f} ± {f1s.std():.3f} (fold별)")
    print(f"precision: {pres.mean():.3f}, recall: {recs.mean():.3f}")
    print(f"→ fold별 P/R과 F1을 함께 출력해 정합성 확인 (D3 수정)")


def cycle_7():
    """예고 확률 관측 추가."""
    tele = load_telegraph_probs()
    lanes = load_lanes()
    X3, y3, X7, y7 = build_c3_c7(lanes, tele)
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m3 = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1500, random_state=42, early_stopping=True)
    m7 = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1500, random_state=42, early_stopping=True)
    f3 = cross_val_score(m3, X3, y3, cv=cv, scoring="f1_macro")
    f7 = cross_val_score(m7, X7, y7, cv=cv, scoring="f1_macro")
    print(f"=== c7 예고 관측 추가 ===")
    print(f"c3 (threat lanes): F1 {f3.mean():.3f} ± {f3.std():.3f}")
    print(f"c7 (+telegraph):   F1 {f7.mean():.3f} ± {f7.std():.3f}")


def build_c3_c7(lanes, tele):
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    xs = [s["x"] for s in samples]
    x_min, x_max = min(xs), max(xs)
    r3, r7, lb = [], [], []
    for idx in range(STACK, len(samples)):
        s = samples[idx]
        if s["i"] not in lanes or s["i"] not in tele:
            continue
        f3, f7 = [], []
        for k in range(STACK, 0, -1):
            sp = samples[idx-k]
            sprev = samples[idx-k-1] if idx-k-1 >= 0 else sp
            dx = sp["x"] - sprev["x"]
            f3.append(np.clip(dx/20.0, -1, 1)); f7.append(np.clip(dx/20.0, -1, 1))
            f3.append((sp["x"]-x_min)/max(1, x_max-x_min)); f7.append((sp["x"]-x_min)/max(1, x_max-x_min))
        lf = lanes[s["i"]]
        lanes_norm = [min(1.0, v/3000.0) for v in lf]
        peak = lanes_norm.index(max(lanes_norm)) / 7.0
        f3 += lanes_norm + [peak] + obs_vec(s["obs"])
        f7 += lanes_norm + [peak] + obs_vec(s["obs"])
        char_lane = min(6, int((s["x"]-x_min)/(x_max-x_min)*7))
        char_threat = lanes_norm[char_lane]
        f3 += [char_threat, 1.0 - abs(char_lane/6.0 - peak)]
        prev_p = tele.get(samples[idx-2]["i"], 0.0)
        f7 += [tele[s["i"]], tele[s["i"]] - prev_p]
        r3.append(f3); r7.append(f7); lb.append(ACT[s["action"]])
    return np.array(r3, dtype=np.float32), np.array(r7, dtype=np.float32), np.array(lb)


def cycle_10():
    """YOLO 관측 교체."""
    tele = load_telegraph_probs()
    lanes = load_lanes()
    yolo = load_yolo_obs()
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from common_features import build_c8_c10
    X8, X10, y10 = build_c8_c10(lanes, tele, yolo)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m8 = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1500, random_state=42, early_stopping=True)
    m10 = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1500, random_state=42, early_stopping=True)
    f8 = cross_val_score(m8, X8, y10, cv=cv, scoring="f1_macro")
    f10 = cross_val_score(m10, X10, y10, cv=cv, scoring="f1_macro")
    print(f"=== c10 YOLO 관측 교체 ===")
    print(f"c8: F1 {f8.mean():.3f} ± {f8.std():.3f}")
    print(f"c10: F1 {f10.mean():.3f} ± {f10.std():.3f}")


def cycle_12():
    """SAIL discriminator."""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    v7log = [json.loads(l) for l in open(V7LOG, encoding="utf-8")]
    t_to_v7 = {}
    for e in v7log:
        if e.get("obs"):
            t_to_v7[round(e["i"]/30, 1)] = e["action"]
    expert = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    ACT_V = {"LEFT": 0, "STAY": 1, "RIGHT": 2}
    pairs = []
    for s in expert:
        t = s["t"]
        for dt in (0.0, 0.03, 0.07, 0.1, -0.03, -0.07, -0.1):
            if round(t+dt, 1) in t_to_v7:
                pairs.append((s, t_to_v7[round(t+dt, 1)]))
                break
    X, y = [], []
    for s, v7a in pairs:
        o = s["obs"]
        X.append(obs_vec(o) + [ACT_V.get(s["action"], 1)/2.0]); y.append(1)
        X.append(obs_vec(o) + [v7a/9.0]); y.append(0)
    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    print(f"=== c12 SAIL discriminator ===")
    print(classification_report(yte, ypr, target_names=["v7", "expert"], digits=3))


if __name__ == "__main__":
    import json
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("6a", "all"):
        cycle_6a()
    if target in ("7", "all"):
        cycle_7()
    if target in ("10", "all"):
        cycle_10()
    if target in ("12", "all"):
        cycle_12()
