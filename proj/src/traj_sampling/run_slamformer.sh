#!/usr/bin/env bash
set -euo pipefail

# One-shot SLAM-Former reconstruction + evaluation.
#
# Loops over RENDERED_DIRS × ORDERS, running SLAM-Former once per combination.
# Outputs result JSON (with frame order) + colored PLY to results_traj_sampling/.
#
# Usage:
#   bash proj/src/traj_sampling/run_slamformer.sh
#
# Configuration: edit RENDERED_DIRS and ORDERS below, or override via env vars.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
SLAM_ROOT="${SLAM_ROOT:-/home/ghr/fs/Junyi/SLAM-Former}"
CKPT_PATH="${CKPT_PATH:-/home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model}"
TARGET_SIZE="${TARGET_SIZE:-518}"
RESULTS_DIR="${SCRIPT_DIR}/results_traj_sampling"
DEVICE="${DEVICE:-cuda}"
MAX_POINTS="${MAX_POINTS:-20000}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
COVERAGE_DILATION="${COVERAGE_DILATION:-1}"
OVERWRITE="${OVERWRITE:-true}"

# Ensure croco is importable
export PYTHONPATH="${SLAM_ROOT}/src/croco:${PYTHONPATH:-}"

# ── Hardcoded rendered folders (edit here) ───────────────────────────────
RENDERED_DIRS=(
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_run_000"
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp2_run_000"
  "${SCRIPT_DIR}/my_trajectories/rendered/replicacad__apt_0/manual_interp5_run_000"
)

# ── Frame orderings to run (edit here) ───────────────────────────────────
# normal:  original order
# reverse: reversed
# shuffle: random permutation (seed=42)
# repeat:  shuffled, each frame twice (seed=42)
ORDERS=(normal reverse shuffle repeat)
# ORDERS=(normal)

# ─────────────────────────────────────────────────────────────────────────

if [[ ! -d "${SLAM_ROOT}" ]]; then
  echo "[error] SLAM-Former repo not found: ${SLAM_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] SLAM-Former weights not found: ${CKPT_PATH}" >&2
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
    result_json="${RESULTS_DIR}/${result_tag}_slamformer.json"
    n_total=$((n_total + 1))

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${result_json}" ]]; then
      echo "[skip] ${result_tag}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== SLAM-Former | method=${method} scene=${scene_id} run=${run_id} order=${ORDER} ===="

    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" -u - "${META_JSON}" "${DATA_DIR}" "${SLAM_ROOT}" "${CKPT_PATH}" \
        "${TARGET_SIZE}" "${RESULTS_DIR}" "${result_tag}" \
        "${DEVICE}" "${MAX_POINTS}" "${DEPTH_MAX}" "${COVERAGE_DILATION}" "${ORDER}" \
        <<'PYEOF'
import sys, json, os, tempfile
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(".")))

from src.utils.pointcloud_eval import chamfer_distance, _umeyama_alignment
from src.envs.voxel_carving import depth_to_world_points

# ── args from shell ──────────────────────────────────────────────────────
meta_json_path    = sys.argv[1]
data_dir          = sys.argv[2]
slam_root         = sys.argv[3]
ckpt_path         = sys.argv[4]
target_size       = int(sys.argv[5])
results_dir       = sys.argv[6]
result_tag        = sys.argv[7]
device_str        = sys.argv[8]
max_points        = int(sys.argv[9])
depth_max         = float(sys.argv[10])
coverage_dilation = int(sys.argv[11])
order_mode        = sys.argv[12]   # normal | reverse | shuffle | repeat

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
print(f"[slamformer] {N} frames (unique={N_unique}, order={order_mode}) from {rendered_dir}")

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
print(f"[slamformer] GT point cloud: {gt_pts.shape[0]} points")

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
    print(f"[slamformer] GT voxels: {int(gt_grid.sum())} total, {n_surface} surface")

# ── setup SLAM-Former ───────────────────────────────────────────────────
slam_root_path = Path(slam_root).expanduser().resolve()
if str(slam_root_path) not in sys.path:
    sys.path.insert(0, str(slam_root_path))
croco = slam_root_path / "src" / "croco"
if croco.exists() and str(croco) not in sys.path:
    sys.path.insert(0, str(croco))

torch.backends.cuda.enable_flash_sdp(False)
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

print(f"[slamformer] Loading SLAM-Former on {device} …")
prev_cwd = os.getcwd()
os.chdir(str(slam_root_path))
from slam.demo import SLAM
slam = SLAM(
    outdir=tempfile.mkdtemp(prefix="slamformer_oneshot_"),
    ckpt_path=ckpt_path,
    target_size=target_size,
    retention_ratio=0.5,
    kf_th=0.1,
    save_gmem=False,
)

# ── run SLAM-Former on ALL frames ───────────────────────────────────────
import cv2
print(f"[slamformer] Running inference on {N} frames …")
for i in range(N):
    img = cv2.imread(rgb_paths[i])
    if img is None:
        print(f"[slamformer] WARNING: failed to read image: {rgb_paths[i]}")
        continue
    slam.step(float(i), img)

os.chdir(prev_cwd)

# Ensure final backend runs so map_opt covers ALL keyframes
# (without this, map_opt only has keyframes up to the last multiple of backend_every,
#  causing len(kfids) != S and falling back to ICP)
slam.terminate()

# Extract point cloud + camera poses
if slam.map_opt is not None:
    map_to_use = slam.map_opt
elif slam.map is not None:
    map_to_use = slam.map
else:
    raise RuntimeError("SLAM-Former produced no map")

result = slam.model.extract(slam.maybe_to_cuda(map_to_use))
pts_raw = result['points'].cpu().numpy()      # (1, S, H, W, 3)
conf    = result['conf'].cpu().numpy()         # (1, S, H, W)
camera_poses = result['camera_poses'].cpu().numpy()[0]  # (S, 4, 4)

S = pts_raw.shape[1]
colors_all = torch.stack(slam.frames[:S]).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()[:, ::-1]  # BGR→RGB
colors_all = np.clip(colors_all * 255, 0, 255).astype(np.uint8)

pts_flat  = pts_raw.reshape(-1, 3)
conf_flat = conf.reshape(-1)
conf_threshold = np.percentile(conf_flat, 15)
mask = conf_flat >= conf_threshold

pred_pts    = pts_flat[mask].astype(np.float32)
pred_colors = np.ascontiguousarray(colors_all[mask])

print(f"[slamformer] Predicted points: {pred_pts.shape[0]} (conf >= {conf_threshold:.4f})")

# ── alignment (Umeyama) ─────────────────────────────────────────────────
# Use slam.kfids to map keyframe index → input frame index, so that
# pred_centres[i] is paired with gt_positions[kfids[i]] (not gt_positions[i]).
gt_centres_all = np.stack(gt_positions, axis=0)
pred_centres = camera_poses[:, :3, 3]  # (S, 3)

kfids = getattr(slam, "kfids", None)
# map_opt may cover only the first S keyframes (if terminate() didn't fully sync);
# truncate kfids to S so the mapping stays consistent with the extracted map.
if kfids is not None and len(kfids) >= S:
    if len(kfids) > S:
        print(f"[slamformer] NOTE: len(kfids)={len(kfids)} > S={S}, truncating to first {S} entries")
    kfids = kfids[:S]
    # Map each keyframe back to its input frame index
    valid_kf = [i for i, fid in enumerate(kfids) if fid < len(gt_positions)]
    if len(valid_kf) >= 3:
        matched_pred = pred_centres[valid_kf]
        matched_gt   = gt_centres_all[[kfids[i] for i in valid_kf]]
        print(f"[slamformer] Alignment: using kfids mapping, {len(valid_kf)} matched keyframes out of {S}")
        R_align, t_align, scale = _umeyama_alignment(matched_pred, matched_gt)
        pred_aligned = (scale * (pred_pts @ R_align.T) + t_align).astype(np.float32)
    else:
        print(f"[slamformer] WARNING: only {len(valid_kf)} valid kfids, falling back to ICP")
        kfids = None  # trigger ICP fallback below

if kfids is None or (kfids is not None and len(kfids) != S):
    # Fallback: ICP alignment on point clouds (robust when frame mapping is unavailable)
    from scipy.spatial import cKDTree
    print(f"[slamformer] Alignment: ICP fallback (S={S}, N={len(gt_positions)})")

    icp_n = min(5000, pred_pts.shape[0], gt_pts.shape[0])
    src = pred_pts[np.random.choice(pred_pts.shape[0], icp_n, replace=False)].copy()
    tgt = gt_pts[np.random.choice(gt_pts.shape[0], icp_n, replace=False)].copy()

    # Initial: centroid alignment
    R_align = np.eye(3, dtype=np.float32)
    t_align = tgt.mean(0) - src.mean(0)
    scale = 1.0

    for _icp_iter in range(50):
        transformed = scale * (src @ R_align.T) + t_align
        tree = cKDTree(tgt)
        dists, idx = tree.query(transformed)
        keep = dists < np.percentile(dists, 75)
        if keep.sum() < 10:
            break
        R_align, t_align, scale = _umeyama_alignment(src[keep], tgt[idx[keep]])

    pred_aligned = (scale * (pred_pts @ R_align.T) + t_align).astype(np.float32)

print(f"[slamformer] Alignment scale={scale:.6f}, aligned points={pred_aligned.shape[0]}")

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

print(f"\n[slamformer] === Results ===")
print(f"  voxel_coverage   = {vc:.6f}")
print(f"  chamfer_distance = {cd_val:.6f}")

# ── save colored PLY (binary) ─────────────────────────────────────────────
ply_path = results_path / f"{result_tag}_slamformer.ply"
cols_u8 = pred_colors if pred_colors.dtype == np.uint8 else (np.clip(pred_colors, 0, 1) * 255).astype(np.uint8)
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
print(f"[slamformer] Colored PLY → {ply_path}")

# ── save result JSON ─────────────────────────────────────────────────────
result_json = results_path / f"{result_tag}_slamformer.json"
result_out = {
    "scene_id":         scene_id,
    "method":           meta.get("method", "unknown"),
    "run_id":           meta.get("run_id", "run_000"),
    "model":            "SLAM-Former",
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
print(f"[slamformer] Result JSON → {result_json}")

# cleanup
del slam
torch.cuda.empty_cache()
PYEOF
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_slamformer done: ${n_run} run, ${n_skip} skipped, ${n_total} total ===="
