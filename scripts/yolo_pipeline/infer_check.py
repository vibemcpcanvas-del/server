# -*- coding: utf-8 -*-
"""Step 5 — inference sanity check: run the trained model on sample val frames.

Runs the best checkpoint on 12 val-set frames, draws detections, saves to
artifacts_yolo_cycle1/infer/, prints per-frame detections with confidence, and
reports per-class confidence summary. Also runs ultralytics val() for a proper
mAP against the parser-generated ground truth.
"""
from __future__ import annotations

import cv2

from common import DATASET, INFER_DIR, TRAIN_DIR, VAL_TXT


def main() -> None:
    from ultralytics import YOLO

    weights = TRAIN_DIR / "yolov8n_cycle1" / "weights" / "best.pt"
    assert weights.exists(), f"trained weights missing: {weights}"
    model = YOLO(str(weights))

    val_imgs = [l for l in VAL_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert val_imgs, "no val images listed"
    rng_every = max(1, len(val_imgs) // 12)
    picks = val_imgs[::rng_every][:12]

    INFER_DIR.mkdir(parents=True, exist_ok=True)
    confs = {"player": [], "boss": []}
    for k, p in enumerate(picks, 1):
        res = model.predict(p, conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
        img = cv2.imread(p)
        det = []
        for box, cls_id, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist()):
            name = model.names[int(cls_id)]
            confs[name].append(cf)
            det.append((name, round(cf, 2)))
            x0, y0, x1, y1 = (int(v) for v in box)
            col = (0, 255, 0) if name == "player" else (0, 0, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, f"{name} {cf:.2f}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imwrite(str(INFER_DIR / f"infer_{k:02d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{k:2d}. {p.split(chr(92))[-1] if chr(92) in p else p.split('/')[-1]}: "
              + (", ".join(f"{n} {c}" for n, c in det) or "NO DETECTION"))

    print("\nconfidence summary (conf>=0.25):")
    for name, arr in confs.items():
        if arr:
            print(f"  {name}: n={len(arr)} mean={sum(arr) / len(arr):.3f} "
                  f"min={min(arr):.3f} max={max(arr):.3f}")
        else:
            print(f"  {name}: NO DETECTIONS")

    print("\nultralytics val() against parser labels (mAP50 etc.):")
    metrics = model.val(data=str(DATASET / "data.yaml"), split="val",
                        device="cpu", imgsz=640, verbose=False)
    r = metrics.results_dict
    print(f"  mAP50={r['metrics/mAP50(B)']:.4f} mAP50-95={r['metrics/mAP50-95(B)']:.4f} "
          f"precision={r['metrics/precision(B)']:.4f} recall={r['metrics/recall(B)']:.4f}")
    (INFER_DIR / "val_metrics.json").write_text(
        __import__("json").dumps({k: float(v) for k, v in r.items()}, indent=2),
        encoding="utf-8")
    print(f"saved inference samples -> {INFER_DIR}")


if __name__ == "__main__":
    main()
