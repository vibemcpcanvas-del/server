# -*- coding: utf-8 -*-
"""WSL2 ROCm GPU로 YOLO 사이클 학습 — cycle 2+ 표준 경로.

사이클 1에서 CPU(25분/10에포크)로 골격 검증 완료. 사이클 2부터는
WSL2 ROCm(venv-dl, RX 6600 XT)로 학습해 전체 풀(4,280장) 사용 가능.

검증된 실행 조합 (2026-09-08 스모크):
  HSA_OVERRIDE_GFX_VERSION=10.3.0     — gfx1032 강제
  HSA_ENABLE_DXG_DETECTION=1          — WSL dxg 감지
  HSA_ENABLE_SDMA=0                   — ★필수: SDMA 큐 assert(librocdxg wddm
                                        queue.cpp:1104) 회피. H2D/D2H 복사 경로.
  LD_PRELOAD=...librocprofiler-register.so — rocprofiler 충돌 회피
  cocotrain 스모크: 1에포크 OK, GPU 추론 1.9ms (CPU 26.6ms의 14배)

Windows 경로는 /mnt/c/... 로 WSL에 그대로 보임 — 데이터셋 복사 불필요.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_WSL = "/mnt/c/Users/ROCmAdmin/Desktop/test/server"
BASE_WIN = r"C:\Users\ROCmAdmin\Desktop\test\server"
PY = "/home/rocmuser/venv-dl/bin/python"
DISTRO = "Ubuntu-24.04-ROCmLab"
OUT = Path(BASE_WIN) / "artifacts_yolo_cycle1" / "train"

ENVS = [
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_DXG_DETECTION=1",
    "HSA_ENABLE_SDMA=0",
    "LD_PRELOAD=$HOME/venv-dl/lib/python3.12/site-packages/torch/lib/"
    "librocprofiler-register.so",
]

TRAIN_SCRIPT = r'''
import sys
sys.path.insert(0, r"/mnt/c/Users/ROCmAdmin/Desktop/test/server/scripts/yolo_pipeline")
import yaml, json, shutil
from pathlib import Path
from common import (CLASSES, DATA_YAML, DATASET, META, OUT, TRAIN_DIR,
                    TRAIN_TXT, VAL_TXT, labeled_pairs)

# 전체 풀 사용 (CPU 예산 제한 없음)
pairs = labeled_pairs()
train = [(i, l) for i, l in pairs if int(i.stem) % 10]
val = [(i, l) for i, l in pairs if int(i.stem) % 10 == 0]
TRAIN_TXT.write_text("\n".join(str(p[0]) for p in train) + "\n", encoding="utf-8")
VAL_TXT.write_text("\n".join(str(p[0]) for p in val) + "\n", encoding="utf-8")
# data.yaml의 path는 Windows 경로(\\Desktop\\...) — WSL에서 /mnt/c/... 로 치환
ds_rel = str(DATASET).replace("\\", "/")
cfg = {"path": ds_rel.replace("C:/Users/ROCmAdmin/Desktop/test/server",
                              "/mnt/c/Users/ROCmAdmin/Desktop/test/server"),
       "train": "train.txt", "val": "val.txt", "names": CLASSES}
yaml.safe_dump(cfg, DATA_YAML.open("w", encoding="utf-8"),
               allow_unicode=True, default_flow_style=False)
print(f"full-pool split: train={len(train)} val={len(val)}")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data=str(DATA_YAML), epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 20,
    imgsz=640, batch=16, device=0, workers=4, patience=20,
    project=str(TRAIN_DIR), name="yolov8n_cycle2_gpu", exist_ok=True, verbose=False,
)
print("CYCLE2 GPU TRAIN DONE")
'''


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    script_path = Path(BASE_WIN) / "scripts" / "yolo_pipeline" / "wsl_train_script.py"
    script_path.write_text(TRAIN_SCRIPT, encoding="utf-8")
    script_wsl = BASE_WSL + "/scripts/yolo_pipeline/wsl_train_script.py"
    cmd = ("export " + " ".join(ENVS) +
           f"; cd {BASE_WSL} && {PY} {script_wsl} {epochs}")
    r = subprocess.run(
        ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=7200)
    print(r.stdout[-3000:])
    if r.returncode:
        print("ERR:", (r.stderr or "")[-1500:], file=sys.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
