#!/usr/bin/env bash
set -euo pipefail

# One-shot VGGT reconstruction + evaluation.
#
# Loops over RENDERED_DIRS × ORDERS, running VGGT once per combination.
# Outputs result JSON (with frame order) + colored PLY to results_traj_sampling/.
#
# Usage:
#   bash proj/src/traj_sampling/run_vggt.sh
#
# Configuration: edit RENDERED_DIRS and ORDERS below, or override via env vars.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
MODEL_ID="${MODEL_ID:-facebook/VGGT-1B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/home/ghr/fs/Junyi/data/proj/model_weights}"
RESULTS_DIR="${SCRIPT_DIR}/results_traj_sampling"
DEVICE="${DEVICE:-cuda}"
MAX_POINTS="${MAX_POINTS:-20000}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
COVERAGE_DILATION="${COVERAGE_DILATION:-1}"
OVERWRITE="${OVERWRITE:-true}"

# ── Hardcoded rendered folders (edit here) ───────────────────────────────
RENDERED_DIRS=(
#   "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_run_000"
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp2_run_000"
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp5_run_000"
)

# ── Frame orderings to run (edit here) ───────────────────────────────────
# normal:  original order
# reverse: reversed
# shuffle: random permutation (seed=42)
# repeat:  shuffled, each frame twice (seed=42)
ORDERS=(normal reverse shuffle repeat)

# ─────────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_DIR}"

n_total=0
n_run=0
n_skip=0

for RENDERED_DIR in "${RENDERED_DIRS[@]}"; do
  META_JSON="${RENDERED_DIR}/meta.json"

  if [[ ! -f "${META_JSON}" ]]; then
    echo "[warn] meta.json not found, skipping: ${META_JSON}" >&2
    continue
  fi

  # Read metadata
  if command -v jq &>/dev/null; then
    scene_id="$(jq -r '.scene_id' "${META_JSON}")"
    method="$(jq -r '.method'     "${META_JSON}")"
    run_id="$(jq -r '.run_id'     "${META_JSON}")"
  else
    scene_id="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['scene_id'])" "${META_JSON}")"
    method="$(  "${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['method'])"   "${META_JSON}")"
    run_id="$(  "${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_id'])"   "${META_JSON}")"
  fi

  safe_scene="$(echo "${scene_id}" | tr '/' '__')"

  for ORDER in "${ORDERS[@]}"; do
    result_tag="${method}_${safe_scene}_${run_id}_${ORDER}"
    result_json="${RESULTS_DIR}/${result_tag}_vggt.json"
    n_total=$((n_total + 1))

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${result_json}" ]]; then
      echo "[skip] ${result_tag}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== VGGT | method=${method} scene=${scene_id} run=${run_id} order=${ORDER} ===="

    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" -u - "${META_JSON}" "${DATA_DIR}" "${MODEL_ID}" "${MODEL_CACHE_DIR}" \
        "${RESULTS_DIR}" "${result_tag}" \
        "${DEVICE}" "${MAX_POINTS}" "${DEPTH_MAX}" "${COVERAGE_DILATION}" "${ORDER}" \
        <<'PYEOF'
import sys, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(".")))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.pointcloud_eval import chamfer_distance, _camera_centers_from_extrinsics, _umeyama_alignment
from src.envs.voxel_carving import depth_to_world_points

# ── args from shell ──────────────────────────────────────────────────────
meta_json_path    = sys.argv[1]
data_dir          = sys.argv[2]
model_id          = sys.argv[3]
model_cache_dir   = sys.argv[4]
results_dir       = sys.argv[5]
result_tag        = sys.argv[6]
device_str        = sys.argv[7]
max_points        = int(sys.argv[8])
depth_max         = float(sys.argv[9])
coverage_dilation = int(sys.argv[10])
order_mode        = sys.argv[11]   # normal | reverse | shuffle | repeat

results_path = Path(results_dir)
results_path.mkdir(parents=True, exist_ok=True)

# ── load meta ────────────────────────────────────────────────────────────
with open(meta_json_path) as f:
    meta = json.load(f)

rendered_dir = Path(meta_json_path).parent
scene_id = meta["scene_id"]
intrinsics = np.array(meta["intrinsics"], dtype=np.float32)
frames = meta["frames"]

# ── reorder frames ──────────────────────────────────────────────────────
N_unique = len(frames)
indices = list(range(N_unique))
if order_mode == "reverse":
    indices = indices[::-1]
elif order_mode == "shuffle":
    import random; random.seed(42)
    random.shuffle(indices)
elif order_mode == "repeat":
    import random; random.seed(42)
    random.shuffle(indices)
    indices = indices + indices  # each frame twice
elif order_mode != "normal":
    raise ValueError(f"Unknown order mode: {order_mode}")

frames_ordered = [frames[i] for i in indices]
frame_order = indices  # save for JSON output

rgb_paths    = [str(rendered_dir / fr["rgb"])   for fr in frames_ordered]
gt_depths    = [np.load(str(rendered_dir / fr["depth"])) for fr in frames_ordered]
gt_positions = [np.array(fr["position"],        dtype=np.float32) for fr in frames_ordered]
gt_quats     = [np.array(fr["quaternion_wxyz"], dtype=np.float32) for fr in frames_ordered]
N = len(frames_ordered)
print(f"[vggt] {N} frames (unique={N_unique}, order={order_mode}) from {rendered_dir}")

# ── GT point cloud ───────────────────────────────────────────────────────
fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

def _build_gt_pointcloud(depths, positions, quats, d_max):
    all_pts = []
    for d, pos, q in zip(depths, positions, quats):
        pts, valid = depth_to_world_points(
            d, pos, q,
            fx=fx, fy=fy, cx=cx, cy=cy,
            forward_axis="x+",
            depth_is_z=True,
            depth_max=d_max,
            depth_max_is_no_hit=True,
            pose_is_camera_to_world=True,
        )
        if pts.shape[0] > 0:
            all_pts.append(pts)
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(all_pts, axis=0).astype(np.float32)

gt_pts = _build_gt_pointcloud(gt_depths, gt_positions, gt_quats, depth_max)
print(f"[vggt] GT point cloud: {gt_pts.shape[0]} points")

# ── voxel grid for coverage ──────────────────────────────────────────────
h5_path = Path(data_dir) / f"{scene_id}.glb.voxels.h5"
use_voxel_grid = h5_path.exists()

if use_voxel_grid:
    import h5py
    with h5py.File(str(h5_path), "r") as hf:
        gt_grid    = hf["grid"][:].astype(bool)
        vox_origin = hf["origin"][:].astype(np.float32)
        vox_pitch  = float(hf["pitch"][()])
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(gt_grid)
    observable_mask = gt_grid & ~interior
    n_surface = int(observable_mask.sum())
    print(f"[vggt] GT voxels: {int(gt_grid.sum())} total, {n_surface} surface")

# ── load VGGT ────────────────────────────────────────────────────────────
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print(f"[vggt] Loading VGGT on {device} …")
model = VGGT.from_pretrained(model_id, cache_dir=model_cache_dir or None, local_files_only=True).to(device)
model.eval()

# ── run VGGT on ALL frames ───────────────────────────────────────────────
print(f"[vggt] Running inference on {N} frames …")
images = load_and_preprocess_images(rgb_paths).to(device)
with torch.inference_mode():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        preds = model(images)

extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
depth_np = preds["depth"].squeeze(0).detach().float().cpu().numpy()
if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
    depth_np = depth_np[..., 0]

extr_np = extrinsics.squeeze(0).detach().float().cpu().numpy()
intr_np = pred_intrinsics.squeeze(0).detach().float().cpu().numpy()

point_maps = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)
pts_all = point_maps.reshape(-1, 3)
valid = np.isfinite(depth_np.reshape(-1)) & (depth_np.reshape(-1) > 0.0)

img_np = preds["images"].squeeze(0).detach().float().cpu().numpy()  # (T, 3, H, W)
colors_all = img_np.transpose(0, 2, 3, 1).reshape(-1, 3)

pred_pts    = pts_all[valid].astype(np.float32)
pred_colors = np.clip(colors_all[valid], 0.0, 1.0).astype(np.float32)

# ── alignment (Umeyama) ─────────────────────────────────────────────────
pred_centres = _camera_centers_from_extrinsics(extr_np)
gt_centres = np.stack(gt_positions, axis=0)

R_align, t_align, scale = _umeyama_alignment(pred_centres, gt_centres)
pred_aligned = (scale * (pred_pts @ R_align.T) + t_align).astype(np.float32)
print(f"[vggt] Alignment scale={scale:.6f}, pred points={pred_aligned.shape[0]}")

# ── Metric 1: voxel coverage (pure GT-based) ───────────────────────────
if use_voxel_grid:
    shape = np.array(gt_grid.shape)
    covered_mask = np.zeros_like(gt_grid, dtype=bool)

    # only mark GT depth reprojection points as covered
    ijk_gt = np.floor((gt_pts - vox_origin) / vox_pitch).astype(int)
    inside_gt = np.all((ijk_gt >= 0) & (ijk_gt < shape), axis=1)
    idx_gt = ijk_gt[inside_gt]
    covered_mask[idx_gt[:, 0], idx_gt[:, 1], idx_gt[:, 2]] = True

    if coverage_dilation > 0:
        from scipy.ndimage import binary_dilation
        struct = np.zeros((3,3,3), dtype=bool)
        struct[1,1,1] = True
        struct[1,1,0] = struct[1,1,2] = True
        struct[1,0,1] = struct[1,2,1] = True
        struct[0,1,1] = struct[2,1,1] = True
        covered_for_metric = covered_mask.copy()
        for _ in range(coverage_dilation):
            covered_for_metric = binary_dilation(covered_for_metric, structure=struct)
    else:
        covered_for_metric = covered_mask
    vc = float((covered_for_metric & observable_mask).sum()) / n_surface if n_surface else 0.0
else:
    voxel_size = 0.05
    gt_v = np.unique(np.floor(gt_pts / voxel_size).astype(int), axis=0)
    gt_s = set(map(tuple, gt_v))
    vc = len(gt_s) / len(gt_s) if gt_s else 0.0  # trivially 1.0 when gt_pts exist

# ── Metric 2: Chamfer distance ──────────────────────────────────────────
cd = chamfer_distance(pred_aligned, gt_pts, max_points=max_points)
cd_val = float(cd) if cd is not None else float("nan")

print(f"\n[vggt] === Results ===")
print(f"  voxel_coverage   = {vc:.6f}")
print(f"  chamfer_distance = {cd_val:.6f}")

# ── save colored PLY (binary) ─────────────────────────────────────────────
ply_path = results_path / f"{result_tag}_vggt.ply"
cols_u8 = (pred_colors * 255).astype(np.uint8)
n_pts = pred_aligned.shape[0]
header = (
    "ply\nformat binary_little_endian 1.0\n"
    f"element vertex {n_pts}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "end_header\n"
)
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
verts = np.empty(n_pts, dtype=dt)
verts["x"] = pred_aligned[:, 0]
verts["y"] = pred_aligned[:, 1]
verts["z"] = pred_aligned[:, 2]
verts["r"] = cols_u8[:, 0]
verts["g"] = cols_u8[:, 1]
verts["b"] = cols_u8[:, 2]
with ply_path.open("wb") as f:
    f.write(header.encode("ascii"))
    f.write(verts.tobytes())
print(f"[vggt] Colored PLY → {ply_path}")

# ── save result JSON ─────────────────────────────────────────────────────
result_json = results_path / f"{result_tag}_vggt.json"
result_out = {
    "scene_id":         scene_id,
    "method":           meta.get("method", "unknown"),
    "run_id":           meta.get("run_id", "run_000"),
    "model":            "VGGT",
    "order":            order_mode,
    "num_views":        N,
    "num_unique_views": N_unique,
    "frame_order":      frame_order,
    "voxel_coverage":   round(vc, 6),
    "chamfer_distance": round(cd_val, 6),
    "scale":            round(scale, 6),
    "pred_points":      int(pred_aligned.shape[0]),
    "gt_points":        int(gt_pts.shape[0]),
}
with open(result_json, "w") as f:
    json.dump(result_out, f, indent=2)
print(f"[vggt] Result JSON → {result_json}")
PYEOF
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_vggt done: ${n_run} run, ${n_skip} skipped, ${n_total} total ===="
