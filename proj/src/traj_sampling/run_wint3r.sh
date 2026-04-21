#!/usr/bin/env bash
set -euo pipefail

# One-shot WinT3R reconstruction + evaluation.
#
# Loops over RENDERED_DIRS × ORDERS, running WinT3R once per combination.
# Outputs result JSON (with frame order) + colored PLY to results_traj_sampling/.
#
# Usage:
#   bash proj/src/traj_sampling/run_wint3r.sh
#
# Configuration: edit RENDERED_DIRS and ORDERS below, or override via env vars.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
WINT3R_REPO="${WINT3R_REPO:-/home/ghr/fs/Junyi/WinT3R}"
MODEL_PATH="${MODEL_PATH:-/home/ghr/fs/Junyi/data/proj/model_weights/WinT3R.bin}"
WINT3R_SIZE="${WINT3R_SIZE:-512}"
WINT3R_INFERENCE_MODE="${WINT3R_INFERENCE_MODE:-online}"
RESULTS_DIR="${SCRIPT_DIR}/results_traj_sampling"
DEVICE="${DEVICE:-cuda}"
MAX_POINTS="${MAX_POINTS:-20000}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
COVERAGE_DILATION="${COVERAGE_DILATION:-1}"
OVERWRITE="${OVERWRITE:-true}"

# ── Hardcoded rendered folders (edit here) ───────────────────────────────
RENDERED_DIRS=(
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_run_000"
#   "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp2_run_000"
#   "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp5_run_000"
)

# ── Frame orderings to run (edit here) ───────────────────────────────────
# normal:  original order
# reverse: reversed
# shuffle: random permutation (seed=42)
# repeat:  shuffled, each frame twice (seed=42)
ORDERS=(normal reverse shuffle repeat)

# ─────────────────────────────────────────────────────────────────────────

if [[ ! -d "${WINT3R_REPO}" ]]; then
  echo "[error] WinT3R repo not found: ${WINT3R_REPO}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] WinT3R weights not found: ${MODEL_PATH}" >&2
  exit 1
fi

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
    result_json="${RESULTS_DIR}/${result_tag}_wint3r.json"
    n_total=$((n_total + 1))

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${result_json}" ]]; then
      echo "[skip] ${result_tag}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== WinT3R | method=${method} scene=${scene_id} run=${run_id} order=${ORDER} ===="

    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" -u - "${META_JSON}" "${DATA_DIR}" "${WINT3R_REPO}" "${MODEL_PATH}" \
        "${WINT3R_SIZE}" "${WINT3R_INFERENCE_MODE}" \
        "${RESULTS_DIR}" "${result_tag}" \
        "${DEVICE}" "${MAX_POINTS}" "${DEPTH_MAX}" "${COVERAGE_DILATION}" "${ORDER}" \
        <<'PYEOF'
import sys, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(".")))

from src.utils.pointcloud_eval import chamfer_distance, _camera_centers_from_extrinsics, _umeyama_alignment
from src.envs.voxel_carving import depth_to_world_points

# ── args from shell ──────────────────────────────────────────────────────
meta_json_path       = sys.argv[1]
data_dir             = sys.argv[2]
wint3r_repo          = sys.argv[3]
model_path           = sys.argv[4]
wint3r_size          = int(sys.argv[5])
wint3r_inference_mode = sys.argv[6]
results_dir          = sys.argv[7]
result_tag           = sys.argv[8]
device_str           = sys.argv[9]
max_points           = int(sys.argv[10])
depth_max            = float(sys.argv[11])
coverage_dilation    = int(sys.argv[12])
order_mode           = sys.argv[13]   # normal | reverse | shuffle | repeat

results_path = Path(results_dir)
results_path.mkdir(parents=True, exist_ok=True)

# ── setup WinT3R imports ────────────────────────────────────────────────
wint3r_repo_path = Path(wint3r_repo).expanduser().resolve()
if str(wint3r_repo_path) not in sys.path:
    sys.path.insert(0, str(wint3r_repo_path))

from dust3r.utils.image import depth_edge, load_images_for_eval
from dust3r.utils.misc import move_to_device
from dust3r.wint3r import WinT3R
from layers.pose_enc import pose_encoding_to_extri

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
print(f"[wint3r] {N} frames (unique={N_unique}, order={order_mode}) from {rendered_dir}")

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
print(f"[wint3r] GT point cloud: {gt_pts.shape[0]} points")

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
    print(f"[wint3r] GT voxels: {int(gt_grid.sum())} total, {n_surface} surface")

# ── load WinT3R ──────────────────────────────────────────────────────────
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print(f"[wint3r] Loading WinT3R on {device} …")
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
).to(device)
weights = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(weights, strict=False)
model.eval()

# ── helper: recover original RGB from WinT3R normalised tensor ──────────
def _recover_wint3r_image(normalized_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=normalized_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=normalized_tensor.dtype).view(3, 1, 1)
    return torch.clamp(normalized_tensor * std + mean, 0.0, 1.0)

# ── run WinT3R on ALL frames ────────────────────────────────────────────
print(f"[wint3r] Running inference on {N} frames …")
dataset = load_images_for_eval(rgb_paths, size=wint3r_size, verbose=False, crop=True)
batch = move_to_device(dataset, device)
imgs = torch.stack(
    [_recover_wint3r_image(view["img"].detach().cpu()) for view in batch],
    dim=1,
)

with torch.no_grad():
    outputs = model(batch, ret_first_pred=False, mode=wint3r_inference_mode)

colors = imgs.permute(0, 1, 3, 4, 2)
extrinsics_cam_to_world = pose_encoding_to_extri(outputs["camera_pos_enc"][-1]).detach()
R_cam_to_world = extrinsics_cam_to_world[:, :, :3, :3]
t_cam_to_world = extrinsics_cam_to_world[:, :, :3, 3]
pts_local = outputs["pts_local"].detach()
pts_world = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, pts_local) + t_cam_to_world[:, :, None, None]
pred_depth = pts_local[..., 2]

# validity masks
mask_depth_max = 500.0
edge_rtol = 0.05
masks_depth = pred_depth < mask_depth_max
masks_edge = ~depth_edge(pred_depth, rtol=edge_rtol)
valid = masks_depth & masks_edge

batch_size, num_views = pred_depth.shape[:2]

# extrinsics: cam-to-world → world-to-cam (3×4)
cam2world_h = torch.eye(4, dtype=extrinsics_cam_to_world.dtype, device=extrinsics_cam_to_world.device)
cam2world_h = cam2world_h.view(1, 1, 4, 4).repeat(batch_size, num_views, 1, 1)
cam2world_h[:, :, :3, :4] = extrinsics_cam_to_world
extr_np = torch.linalg.inv(cam2world_h)[:, :, :3, :4].detach().cpu().numpy().astype(np.float32)[0]

pts_flat = pts_world[valid].detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
valid_cpu = valid.detach().cpu()
colors_flat = torch.clamp(colors[valid_cpu], 0.0, 1.0).detach().cpu().numpy().reshape(-1, 3).astype(np.float32)

pred_pts = pts_flat
pred_colors = colors_flat
print(f"[wint3r] Predicted points: {pred_pts.shape[0]}")

# ── alignment (Umeyama) ─────────────────────────────────────────────────
pred_centres = _camera_centers_from_extrinsics(extr_np)
gt_centres = np.stack(gt_positions, axis=0)

R_align, t_align, scale = _umeyama_alignment(pred_centres, gt_centres)
pred_aligned = (scale * (pred_pts @ R_align.T) + t_align).astype(np.float32)
print(f"[wint3r] Alignment scale={scale:.6f}, pred points={pred_aligned.shape[0]}")

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

print(f"\n[wint3r] === Results ===")
print(f"  voxel_coverage   = {vc:.6f}")
print(f"  chamfer_distance = {cd_val:.6f}")

# ── save colored PLY (binary) ─────────────────────────────────────────────
ply_path = results_path / f"{result_tag}_wint3r.ply"
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
print(f"[wint3r] Colored PLY → {ply_path}")

# ── save result JSON ─────────────────────────────────────────────────────
result_json = results_path / f"{result_tag}_wint3r.json"
result_out = {
    "scene_id":         scene_id,
    "method":           meta.get("method", "unknown"),
    "run_id":           meta.get("run_id", "run_000"),
    "model":            "WinT3R",
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
print(f"[wint3r] Result JSON → {result_json}")
PYEOF
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_wint3r done: ${n_run} run, ${n_skip} skipped, ${n_total} total ===="
