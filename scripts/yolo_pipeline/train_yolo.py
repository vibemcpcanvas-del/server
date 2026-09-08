# -*- coding: utf-8 -*-
"""Step 4 — split train/val, write data.yaml, train YOLOv8n on Windows CPU.

90/10 split by frame id (2 frames apart in time -> val is effectively temporal,
no leakage in the usual sense but adjacent-frame similarity is acknowledged in
the cycle report). Also emits a class-balance summary for the report.
"""
from __future__ import annotations

import json
import shutil

import yaml

from common import (CLASSES, DATA_YAML, DATASET, IMG_ALL, LBL_ALL, META, OUT,
                    TRAIN_DIR, TRAIN_SUBSAMPLE, TRAIN_TXT, VAL_SUBSAMPLE, VAL_TXT,
                    even_sample, labeled_pairs)


def main() -> None:
    pairs = labeled_pairs()
    assert len(pairs) >= 100, f"too few labeled frames to train: {len(pairs)}"

    train_all, val_all = [], []
    for img, lbl in pairs:
        fi = int(img.stem)
        (train_all if fi % 10 else val_all).append((img, lbl))
    # CPU-budget subsample (evenly spaced over time) — full pool stays on disk
    train = even_sample(train_all, TRAIN_SUBSAMPLE)
    val = even_sample(val_all, VAL_SUBSAMPLE)
    for name, split in (("train", train), ("val", val)):
        txt = TRAIN_TXT if name == "train" else VAL_TXT
        txt.write_text("\n".join(str(p[0]) for p in split) + "\n", encoding="utf-8")
    yaml.safe_dump({"path": str(DATASET), "train": "train.txt", "val": "val.txt",
                    "names": CLASSES}, DATA_YAML.open("w", encoding="utf-8"),
                   allow_unicode=True, default_flow_style=False)
    print(f"split: train={len(train)}/{len(train_all)} val={len(val)}/{len(val_all)} "
          f"(CPU-budget subsample)")
    print(f"data.yaml -> {DATA_YAML}")

    # class balance from meta.json
    meta = json.loads(META.read_text(encoding="utf-8"))
    tp = sum(1 for v in meta.values() if "player" in v)
    tb = sum(1 for v in meta.values() if "boss" in v)
    print(f"class presence: player in {tp}/{len(meta)} frames, boss in {tb}/{len(meta)}")
    (OUT / "dataset_stats.json").write_text(json.dumps({
        "labeled_frames": len(pairs),
        "train_pool": len(train_all), "val_pool": len(val_all),
        "train": len(train), "val": len(val),
        "player_frames": tp, "boss_frames": tb}, indent=2), encoding="utf-8")

    # ── train YOLOv8n — Windows CPU, small cycle-1 budget ──
    from ultralytics import YOLO
    shutil.copy2(DATASET / "data.yaml", OUT / "data.yaml")  # readability copy
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=2,
        patience=20,
        project=str(TRAIN_DIR),
        name="yolov8n_cycle1",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
