"""Benchmark CUT3R with varying number of input images."""
import os, sys, glob, json, time
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CUT3R_ROOT = "/home/ghr/fs/Junyi/CUT3R"
MODEL_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/cut3r/cut3r_512_dpt_4_64.pth"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
COUNTS = [1, 2, 4, 8, 16, 32, 63]
RESULTS_FILE = os.path.join(RESULTS_DIR, "cut3r.json")
PLY_PATH = os.path.join(RESULTS_DIR, "cut3r_reconstruction.ply")

os.chdir(CUT3R_ROOT)
sys.path.insert(0, CUT3R_ROOT)

from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r(MODEL_PATH)

from src.dust3r.model import ARCroco3DStereo
from src.dust3r.inference import inference
from src.dust3r.utils.image import load_images

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(all_images) >= 63, f"Expected >=63 images, found {len(all_images)}"

# Load model once
print("Loading CUT3R model...")
model = ARCroco3DStereo.from_pretrained(MODEL_PATH).to("cuda")
model.eval()

def prepare_views(img_paths, size=512):
    images = load_images(img_paths, size=size)
    views = []
    for i, im in enumerate(images):
        view = {
            "img": im["img"],
            "ray_map": torch.full(
                (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views

# Warmup
print("Warmup...")
warmup_views = prepare_views(all_images[:2])
with torch.inference_mode():
    _ = inference(warmup_views, model, "cuda")
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
    print(f"  CUT3R: {n} images")
    print(f"{'='*50}")

    subset = all_images[:n]
    views = prepare_views(subset)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode():
        outputs, state_args = inference(views, model, "cuda")

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
        print("  Saving CUT3R reconstruction PLY...")
        from src.dust3r.utils.camera import pose_encoding_to_camera
        from src.dust3r.post_process import estimate_focal_knowing_depth
        from src.dust3r.utils.geometry import geotrf

        pts3ds_self = [out["pts3d_in_self_view"].cpu() for out in outputs["pred"]]
        conf_self = [out["conf_self"].cpu() for out in outputs["pred"]]
        pr_poses = [pose_encoding_to_camera(out["camera_pose"].clone()).cpu() for out in outputs["pred"]]

        all_pts = []
        all_colors = []
        for i, (pself, pose) in enumerate(zip(pts3ds_self, pr_poses)):
            pts_world = geotrf(pose, pself)  # transform to world
            color = 0.5 * (outputs["views"][i]["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            pts_np = pts_world.reshape(-1, 3).numpy()
            col_np = color.reshape(-1, 3).numpy()
            conf_np = conf_self[i].reshape(-1).numpy()
            valid = np.isfinite(pts_np).all(axis=1) & (conf_np > 1.5)
            all_pts.append(pts_np[valid])
            all_colors.append(col_np[valid])

        merged_pts = np.concatenate(all_pts)
        merged_colors = np.clip(np.concatenate(all_colors), 0, 1)

        ply_path = PLY_PATH
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(merged_pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with open(ply_path, "w") as pf:
            pf.write(header)
            for p, c in zip(merged_pts, (merged_colors * 255).astype(np.uint8)):
                pf.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        print(f"  Saved to {ply_path} ({len(merged_pts)} points)")

    del outputs, state_args, views
    torch.cuda.empty_cache()

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "CUT3R", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
