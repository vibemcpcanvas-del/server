# -*- coding: utf-8 -*-
"""Full YOLO cycle-1 pipeline: extract -> auto-label -> review -> train -> infer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = ["extract_frames.py", "auto_label.py", "review_labels.py",
         "train_yolo.py", "infer_check.py"]

if __name__ == "__main__":
    only = sys.argv[1:] or STEPS
    here = Path(__file__).parent
    for s in STEPS:
        if s not in only:
            continue
        print(f"\n===== {s} =====", flush=True)
        rc = subprocess.call([sys.executable, str(here / s)], cwd=str(here))
        if rc != 0:
            print(f"STEP FAILED: {s} (exit {rc})")
            sys.exit(rc)
    print("\nALL STEPS DONE")
