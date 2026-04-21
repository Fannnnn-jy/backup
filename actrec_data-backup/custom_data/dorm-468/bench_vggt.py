"""Benchmark VGGT with varying number of input images."""
import os, sys, glob, json, time
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/ghr/fs/Junyi/proj")

IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
COUNTS = [1, 2, 4, 8, 16, 32, 63]
RESULTS_FILE = os.path.join(RESULTS_DIR, "vggt.json")
PLY_PATH = os.path.join(RESULTS_DIR, "vggt_reconstruction.ply")

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(all_images) >= 63, f"Expected >=63 images, found {len(all_images)}"

skip_benchmark = os.path.exists(RESULTS_FILE)
skip_ply = os.path.exists(PLY_PATH)

if skip_benchmark and skip_ply:
    print(f"Benchmark JSON and PLY both exist, nothing to do.")
    sys.exit(0)

# Load model once
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import open3d as o3d

print("Loading VGGT model...")
model = VGGT.from_pretrained(
    "facebook/VGGT-1B",
    cache_dir="/home/ghr/fs/Junyi/data/proj/model_weights"
).to("cuda")
model.eval()

# Warmup
print("Warmup...")
warmup_imgs = load_and_preprocess_images(all_images[:2]).to("cuda")
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _ = model(warmup_imgs)
del warmup_imgs
torch.cuda.empty_cache()

skip_benchmark = os.path.exists(RESULTS_FILE)
if skip_benchmark:
    print(f"Benchmark JSON already exists: {RESULTS_FILE}")
    print("Skipping benchmark, only running full reconstruction...")
    run_counts = [COUNTS[-1]]
else:
    run_counts = COUNTS

results = []
for n in run_counts:
    print(f"\n{'='*50}")
    print(f"  VGGT: {n} images")
    print(f"{'='*50}")

    subset = all_images[:n]
    processed = load_and_preprocess_images(subset).to("cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(processed)

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
        print("  Saving VGGT reconstruction PLY...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], processed.shape[-2:]
        )
        depth = predictions["depth"].squeeze(0).cpu().numpy()
        intrinsics = intrinsic.squeeze(0).cpu().numpy()
        extrinsics = extrinsic.squeeze(0).cpu().numpy()
        input_imgs = predictions["images"].squeeze(0).cpu().numpy()
        point_maps = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        points = point_maps.reshape(-1, 3)
        colors = input_imgs.transpose(0, 2, 3, 1).reshape(-1, 3)
        # Filter invalid points
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        o3d.io.write_point_cloud(PLY_PATH, pcd)
        print(f"  Saved to {PLY_PATH} ({len(points)} points)")

    del processed, predictions
    torch.cuda.empty_cache()

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "VGGT", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
