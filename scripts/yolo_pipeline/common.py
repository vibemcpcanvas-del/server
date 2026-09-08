# -*- coding: utf-8 -*-
"""YOLO auto-label -> train -> validate pipeline — shared paths/constants (cycle 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

VIDEO = REPO / "reports" / "spectator_v3_run2_aieye.mp4"
JSONL = REPO / "reports" / "spectator_v3_run2.jsonl"

OUT = REPO / "artifacts_yolo_cycle1"
DATASET = OUT / "dataset"
IMG_ALL = DATASET / "images" / "all"
LBL_ALL = DATASET / "labels" / "all"
META = DATASET / "meta.json"
DATA_YAML = DATASET / "data.yaml"
TRAIN_TXT = DATASET / "train.txt"
VAL_TXT = DATASET / "val.txt"

WEIGHTS_DIR = OUT / "weights"
TRAIN_DIR = OUT / "train"
INFER_DIR = OUT / "infer"
REVIEW_DIR = REPO / "reports" / "yolo_cycle1_review"

CLASSES = {0: "player", 1: "boss"}
FRAMES_EVERY = 2  # 30fps video -> 15fps sample

# Cycle-1 CPU budget: evenly subsample the labeled pool for training/val.
# Set to None to train on the full 4280-frame pool (~13 min/epoch on this CPU).
TRAIN_SUBSAMPLE = 600
VAL_SUBSAMPLE = 150


def even_sample(items: list, n: int) -> list:
    """Evenly spaced subsample preserving temporal coverage."""
    if n is None or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(k * step)] for k in range(n)]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def bossfight_indices() -> set:
    """Frame indices flagged is_bossfight in the spectator jsonl log."""
    idx: set = set()
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("is_bossfight"):
                idx.add(int(d["i"]))
    return idx


def labeled_pairs() -> list:
    """(image_path, label_path) pairs for every label file on disk."""
    out = []
    for lbl in sorted(LBL_ALL.glob("*.txt")):
        img = IMG_ALL / (lbl.stem + ".jpg")
        if img.exists():
            out.append((img, lbl))
    return out
