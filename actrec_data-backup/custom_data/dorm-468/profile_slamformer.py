"""Profile SLAM-Former startup stages: import, model load, inference."""
import os
import sys
import time
import shutil
import tempfile
import glob
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SLAM_ROOT = "/home/ghr/fs/Junyi/SLAM-Former"
CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
TARGET_SIZE = 518
RETENTION_RATIO = 0.5
KF_TH = 0.1
N_IMAGES = 32

os.chdir(SLAM_ROOT)
sys.path.insert(0, os.path.join(SLAM_ROOT, "src", "croco"))

# ── Stage 1: Import ──
t0 = time.time()
import torch
torch.backends.cuda.enable_flash_sdp(False)
t_torch = time.time()

sys.path.insert(0, SLAM_ROOT)
from slam.demo import SLAM
t_import = time.time()

# ── Stage 2: Model load ──
t2 = time.time()
output_dir = tempfile.mkdtemp(prefix="slamformer_out_")
slam = SLAM(
    outdir=output_dir,
    ckpt_path=CKPT_PATH,
    target_size=TARGET_SIZE,
    retention_ratio=RETENTION_RATIO,
    kf_th=KF_TH,
    save_gmem=False,
)
t_model = time.time()

# ── Stage 3: Prepare images ──
import cv2
import re

def image_sort_key(path):
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")), key=image_sort_key)
selected = all_images[:N_IMAGES]

frame_ids = []
for p in selected:
    match = re.search(r'\d+(?:\.\d+)?', os.path.basename(p))
    frame_ids.append(float(match.group()) if match else float(len(frame_ids)))

slam.K = None
t_prep = time.time()

# ── Stage 4: Inference ──
t4 = time.time()
for fid, img_path in zip(frame_ids, selected):
    img = cv2.imread(img_path)
    slam.step(fid, img)
result = slam.terminate()
t_infer = time.time()

shutil.rmtree(output_dir, ignore_errors=True)

# ── Summary ──
total = t_infer - t0
print("\n" + "=" * 50)
print(f"  SLAM-Former profiling ({N_IMAGES} images)")
print("=" * 50)
print(f"  torch import:    {t_torch - t0:7.2f}s")
print(f"  other imports:   {t_import - t_torch:7.2f}s")
print(f"  model load:      {t_model - t2:7.2f}s")
print(f"  image prep:      {t_prep - t_model:7.2f}s")
print(f"  inference:       {t_infer - t4:7.2f}s")
print(f"  ─────────────────────────")
print(f"  total:           {total:7.2f}s")
print(f"  overhead:        {total - (t_infer - t4):7.2f}s")
print("=" * 50)
