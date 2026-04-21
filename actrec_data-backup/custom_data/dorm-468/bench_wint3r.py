"""Benchmark WinT3R with varying number of input images."""
import os, sys, glob, json, time, shutil, tempfile
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

WINT3R_ROOT = "/home/ghr/fs/Junyi/WinT3R"
CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/WinT3R.bin"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
# COUNTS = [1, 2, 4, 8, 16, 32, 63]
COUNTS = [63]
RESULTS_FILE = os.path.join(RESULTS_DIR, "wint3r.json")
PLY_PATH = os.path.join(RESULTS_DIR, "wint3r_reconstruction.ply")

os.chdir(WINT3R_ROOT)
sys.path.insert(0, WINT3R_ROOT)

from dust3r.utils.image import load_images_for_eval as load_images
from dust3r.utils.image import depth_edge
from dust3r.utils.misc import move_to_device
from dust3r.utils.vis_utils import write_ply
from layers.pose_enc import pose_encoding_to_extri

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
print("Loading WinT3R model...")
from dust3r.wint3r import WinT3R
model = WinT3R(
    state_size=1024,
    state_pe="2d",
    pos_embed="RoPE100",
    patch_embed_cls="ManyAR_PatchEmbed",
    img_size=[512, 512],
    head_type="conv",
    enc_embed_dim=1024,
    enc_depth=24,
    enc_num_heads=16,
    dec_embed_dim=768,
    dec_depth=12,
    dec_num_heads=12,
    landscape_only=False,
).cuda()

weights = torch.load(CKPT_PATH, weights_only=False)
model.load_state_dict(weights, strict=False)
model.eval()
del weights

def recover_image(normalized_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(normalized_tensor * std + mean, 0, 1)

# Warmup
print("Warmup...")
tmpdir = tempfile.mkdtemp()
for f in all_images[:2]:
    shutil.copy(f, tmpdir)
warmup_data = load_images(tmpdir, size=512, verbose=False, crop=True, interval=1)
warmup_batch = move_to_device(warmup_data, "cuda")
with torch.no_grad():
    _ = model(warmup_batch, ret_first_pred=False, mode="online")
del warmup_batch, warmup_data
shutil.rmtree(tmpdir)
torch.cuda.empty_cache()

results = []
for n in run_counts:
    print(f"\n{'='*50}")
    print(f"  WinT3R: {n} images")
    print(f"{'='*50}")

    # Create temp dir with subset of images
    tmpdir = tempfile.mkdtemp()
    for f in all_images[:n]:
        shutil.copy(f, tmpdir)

    dataset = load_images(tmpdir, size=512, verbose=False, crop=True, interval=1)
    shutil.rmtree(tmpdir)
    batch = move_to_device(dataset, "cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        pred = model(batch, ret_first_pred=False, mode="online")

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
        print("  Saving WinT3R reconstruction PLY...")
        N_pred = pred['pts_local'].shape[1]
        imgs = torch.stack([recover_image(view['img'].detach().cpu()) for view in batch], dim=1)
        colors = imgs[:, :N_pred].permute(0, 1, 3, 4, 2)

        extrinsics = pose_encoding_to_extri(pred["camera_pos_enc"][-1])
        R_cam_to_world = extrinsics[:, :, :3, :3]
        t_cam_to_world = extrinsics[:, :, :3, 3]
        world_pts = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, pred['pts_local']) + t_cam_to_world[:, :, None, None]
        pred_depth = pred['pts_local'][..., 2:]

        masks_depth = pred_depth[..., 0] < 500.0
        masks_edge = ~depth_edge(pred_depth.squeeze(-1), rtol=0.05)
        masks = masks_depth * masks_edge

        write_ply(
            world_pts[masks].detach().cpu().numpy().reshape(-1, 3),
            colors[masks.detach().cpu()].reshape(-1, 3).detach().cpu().numpy(),
            PLY_PATH
        )
        print(f"  Saved to {PLY_PATH}")

    del batch, dataset, pred
    torch.cuda.empty_cache()

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "WinT3R", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
