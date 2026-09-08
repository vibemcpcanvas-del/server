# -*- coding: utf-8 -*-
"""Step 2 — auto-label extracted frames with the existing HSV parser (YOLO txt).

Runs core/vision/real_parser_v2.py parse_frame() on every extracted frame and
re-uses the parser's own blob masks to emit axis-aligned bounding boxes in YOLO
format: "<cls> <cx> <cy> <w> <h>" (normalized). Also writes meta.json with the
full parse result per frame for later stats/review.

cls 0 = player (white blob, bottom play area), cls 1 = boss (red hair blob).
"""
from __future__ import annotations

import json
import time

import cv2

from common import CLASSES, IMG_ALL, LBL_ALL, META
from core.vision import real_parser_v2 as rp


def _blob(img, kind: str):
    """Largest-blob bbox using the parser's exact thresholds (same mask, box instead of center)."""
    if kind == "player":
        x0, y0, x1, y1, min_area = rp.PLAY_AREA[0], 430, rp.PLAY_AREA[2], 660, 250
        mask = cv2.inRange(rp._hsv(img[y0:y1, x0:x1]), (0, 0, 190), (180, 55, 255))
    else:  # boss: red hair + magenta
        x0, y0, x1, y1, min_area = rp.PLAY_AREA[0], 240, rp.PLAY_AREA[2], 620, 180
        h = rp._hsv(img[y0:y1, x0:x1])
        mask = cv2.bitwise_or(
            cv2.inRange(h, (0, 130, 110), (8, 255, 255)),
            cv2.inRange(h, (172, 130, 110), (180, 255, 255)),
        )
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best, best_a = None, min_area
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > best_a:
            best_a, best = a, i
    if best is None:
        return None
    l, t, w, hh, area = (int(stats[best, k]) for k in range(5))
    if w < 8 or hh < 8:  # degenerate sliver guard
        return None
    return {"bx": l + x0, "by": t + y0, "w": w, "h": hh, "area": area,
            "x": float(l + x0 + w / 2), "y": float(t + y0 + hh / 2)}


def main() -> None:
    imgs = sorted(IMG_ALL.glob("*.jpg"))
    print(f"images to label: {len(imgs)}")
    LBL_ALL.mkdir(parents=True, exist_ok=True)

    W, H = rp.BASE_W, rp.BASE_H
    meta: dict = {}
    counts = {"player": 0, "boss": 0, "both": 0, "bf_background": 0, "parser_not_bf": 0}
    t0 = time.perf_counter()

    for k, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip unreadable: {img_path.name}")
            continue
        fi = int(img_path.stem)
        res = rp.parse_frame(img)
        entry: dict = {"is_bossfight": bool(res.get("is_bossfight"))}
        lines: list = []
        n_found = 0
        for cls_id, kind in CLASSES.items():
            bb = _blob(img, kind)
            if bb is not None and res.get(f"{kind}_found"):
                cx = (bb["bx"] + bb["w"] / 2) / W
                cy = (bb["by"] + bb["h"] / 2) / H
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bb['w'] / W:.6f} {bb['h'] / H:.6f}")
                entry[kind] = bb
                counts[kind] += 1
                n_found += 1
        if not res.get("is_bossfight"):
            counts["parser_not_bf"] += 1
        elif n_found == 0:
            counts["bf_background"] += 1
        elif n_found == 2:
            counts["both"] += 1

        (LBL_ALL / (img_path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                        encoding="utf-8")
        meta[fi] = entry
        if k % 500 == 0:
            rate = k / (time.perf_counter() - t0)
            print(f"  {k}/{len(imgs)} ({rate:.0f} fps)", flush=True)

    META.write_text(json.dumps(meta, indent=0), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"labeled {len(imgs)} frames in {dt:.0f}s ({len(imgs) / dt:.0f} fps)")
    print(f"boxes: player={counts['player']} boss={counts['boss']} both={counts['both']}")
    print(f"backgrounds: bossfight_no_object={counts['bf_background']} "
          f"parser_disagrees_with_jsonl={counts['parser_not_bf']}")
    print(f"meta -> {META}")


if __name__ == "__main__":
    main()
