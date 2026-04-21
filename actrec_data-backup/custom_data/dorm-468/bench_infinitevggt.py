"""Benchmark InfiniteVGGT with varying number of input images."""
import os, sys, glob, json, time
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IVGGT_ROOT = "/home/ghr/fs/Junyi/InfiniteVGGT"
CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/models-InfiniteVGGT/checkpoints.pth"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
COUNTS = [1, 2, 4, 8, 16, 32, 63]
RESULTS_FILE = os.path.join(RESULTS_DIR, "infinitevggt.json")
PLY_PATH = os.path.join(RESULTS_DIR, "infinitevggt_reconstruction.ply")

os.chdir(IVGGT_ROOT)
sys.path.insert(0, os.path.join(IVGGT_ROOT, "src"))

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
import open3d as o3d

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(all_images) >= 63, f"Expected >=63 images, found {len(all_images)}"

# Load model once
print("Loading InfiniteVGGT model...")
model = StreamVGGT(total_budget=1200000)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt, strict=True)
model = model.to("cuda")
model.eval()
del ckpt

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Warmup
print("Warmup...")
warmup_imgs = load_and_preprocess_images(all_images[:2]).to("cuda")
warmup_frames = [{"img": warmup_imgs[i].unsqueeze(0)} for i in range(2)]
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    _ = model.inference(warmup_frames, cache_results=False)
del warmup_imgs, warmup_frames
torch.cuda.empty_cache()

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

results = []
for n in run_counts:
    print(f"\n{'='*50}")
    print(f"  InfiniteVGGT: {n} images")
    print(f"{'='*50}")

    subset = all_images[:n]
    images = load_and_preprocess_images(subset).to("cuda")
    frames = [{"img": images[i].unsqueeze(0)} for i in range(n)]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        output = model.inference(frames, cache_results=True)

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
        print("  Saving InfiniteVGGT reconstruction PLY...")
        all_pts3d = [res['pts3d_in_other_view'].squeeze(0) for res in output.ress]
        all_conf = [res['conf'].squeeze(0) for res in output.ress]

        pts_list, col_list = [], []
        for i, (pts, conf) in enumerate(zip(all_pts3d, all_conf)):
            pts_np = pts.cpu().numpy().reshape(-1, 3)
            conf_np = conf.cpu().numpy().reshape(-1)
            # Get colors from input images
            col_np = images[i].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
            valid = np.isfinite(pts_np).all(axis=1) & (conf_np > 1.5)
            pts_list.append(pts_np[valid])
            col_list.append(col_np[valid])

        merged_pts = np.concatenate(pts_list)
        merged_colors = np.clip(np.concatenate(col_list), 0, 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pts)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        ply_path = PLY_PATH
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  Saved to {ply_path} ({len(merged_pts)} points)")

    del images, frames, output
    torch.cuda.empty_cache()

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "InfiniteVGGT", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
