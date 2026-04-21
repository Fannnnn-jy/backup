"""Benchmark Pi3X with varying number of input images."""
import os, sys, glob, json, time, shutil, tempfile
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PI3_ROOT = "/home/ghr/fs/Junyi/Pi3"
CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/Pi3X.safetensors"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
COUNTS = [1, 2, 4, 8, 16, 32, 63]
RESULTS_FILE = os.path.join(RESULTS_DIR, "pi3x.json")
PLY_PATH = os.path.join(RESULTS_DIR, "pi3x_reconstruction.ply")

os.chdir(PI3_ROOT)
sys.path.insert(0, PI3_ROOT)

from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X
from safetensors.torch import load_file

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(all_images) >= 63, f"Expected >=63 images, found {len(all_images)}"

skip_benchmark = os.path.exists(RESULTS_FILE)
skip_ply = os.path.exists(PLY_PATH)

if skip_benchmark and skip_ply:
    print(f"Benchmark JSON and PLY both exist, nothing to do.")
    sys.exit(0)

if skip_benchmark:
    print(f"Benchmark JSON already exists: {RESULTS_FILE}")
    print("Skipping benchmark, only running full reconstruction...")
    run_counts = [COUNTS[-1]]
else:
    run_counts = COUNTS

# Load model once
print("Loading Pi3X model...")
model = Pi3X(use_multimodal=False).eval()
weight = load_file(CKPT_PATH)
model.load_state_dict(weight, strict=False)
model = model.to("cuda")

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Warmup
print("Warmup...")
tmpdir = tempfile.mkdtemp()
for f in all_images[:2]:
    shutil.copy(f, tmpdir)
warmup_imgs, _ = load_multimodal_data(tmpdir, conditions=dict(intrinsics=None, poses=None, depths=None), interval=1, device="cuda")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    _ = model(imgs=warmup_imgs)
del warmup_imgs
shutil.rmtree(tmpdir)
torch.cuda.empty_cache()

results = []
for n in run_counts:
    print(f"\n{'='*50}")
    print(f"  Pi3X: {n} images")
    print(f"{'='*50}")

    # Create temp dir with subset of images
    tmpdir = tempfile.mkdtemp()
    for f in all_images[:n]:
        shutil.copy(f, tmpdir)

    imgs, _ = load_multimodal_data(tmpdir, conditions=dict(intrinsics=None, poses=None, depths=None), interval=1, device="cuda")
    shutil.rmtree(tmpdir)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model(imgs=imgs)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"  Time: {elapsed:.2f}s | Peak GPU Mem: {peak_mem:.2f} GB")
    results.append({
        "num_images": n,
        "time_sec": round(elapsed, 3),
        "peak_mem_gb": round(peak_mem, 3),
    })

    # Save point cloud for the full set
    if n == COUNTS[-1] and not skip_ply:
        print("  Saving Pi3X reconstruction PLY...")
        masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], PLY_PATH)
        print(f"  Saved to {PLY_PATH}")

    del imgs, res
    torch.cuda.empty_cache()

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "Pi3X", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
