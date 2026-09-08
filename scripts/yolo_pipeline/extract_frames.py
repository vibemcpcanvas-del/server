# -*- coding: utf-8 -*-
"""Step 1 — extract bossfight frames from spectator_v3_run2 video at 15fps sample.

Reads the is_bossfight frame indices from reports/spectator_v3_run2.jsonl and
writes every FRAMES_EVERY-th bossfight frame to artifacts_yolo_cycle1/dataset/images/all
as <frame_index>.jpg (idempotent: skips files already on disk).
"""
from __future__ import annotations

import cv2

from common import FRAMES_EVERY, IMG_ALL, VIDEO, bossfight_indices


def main() -> None:
    boss = bossfight_indices()
    lo, hi = min(boss), max(boss)
    expected = len([i for i in range(lo, hi + 1, FRAMES_EVERY) if i in boss])
    print(f"bossfight frames in jsonl: {len(boss)} (range {lo}..{hi})")
    print(f"expected extraction (every {FRAMES_EVERY}f): {expected}")

    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened(), f"cannot open {VIDEO}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    IMG_ALL.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    i = lo
    while True:
        ok, frame = cap.read()
        if not ok or i > hi:
            break
        if i % FRAMES_EVERY == 0 and i in boss:
            path = IMG_ALL / f"{i:06d}.jpg"
            if path.exists():
                skipped += 1
            else:
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1
        if i % 1000 == 0:
            print(f"  frame {i} .. saved/skipped {saved}/{skipped}", flush=True)
        i += 1
    cap.release()
    print(f"extracted: {saved} frames ({skipped} already on disk) -> {IMG_ALL}")
    assert saved >= expected * 0.99, f"extraction incomplete: {saved}/{expected}"


if __name__ == "__main__":
    main()
