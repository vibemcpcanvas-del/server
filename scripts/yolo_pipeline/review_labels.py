# -*- coding: utf-8 -*-
"""Step 3 — label quality review: save 10 random frames with boxes drawn.

Draws the parser-generated YOLO boxes on the images into
reports/yolo_cycle1_review/ (green = player, red = boss) and prints per-sample
box pixel sizes for a quick sanity check.
"""
from __future__ import annotations

import random

import cv2

from common import IMG_ALL, LBL_ALL, REVIEW_DIR, labeled_pairs


def draw(path, lines, color, text):
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    for ln in lines:
        c, cx, cy, bw, bh = (float(v) for v in ln.split())
        x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        col = (0, 255, 0) if c == 0 else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, "player" if c == 0 else "boss", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(str(REVIEW_DIR / f"{path.stem}{text}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return [(ln.split()[0], round(float(ln.split()[3]) * W), round(float(ln.split()[4]) * H))
            for ln in lines]


def main() -> None:
    pairs = labeled_pairs()
    assert len(pairs) >= 50, f"too few labeled frames: {len(pairs)}"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    picks = rng.sample(pairs, 10)
    for k, (img, lbl) in enumerate(picks, 1):
        lines = [l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        sizes = draw(img, lines, (0, 0, 0), "_review")
        print(f"{k:2d}. {img.stem}: " + " ".join(f"cls{c} {w}x{h}px" for c, w, h in sizes))
    print(f"saved 10 review images -> {REVIEW_DIR}")


if __name__ == "__main__":
    main()
