from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _frame_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"frame_(\d+)_rgb\.png$", path.name)
    if m is None:
        return (10**9, path.name)
    return (int(m.group(1)), path.name)


def _find_rgb_frames(raw_frames_dir: Path) -> List[Path]:
    files = sorted(raw_frames_dir.glob("frame_*_rgb.png"), key=_frame_sort_key)
    return [p for p in files if p.is_file()]


def _write_xyzrgb_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors).reshape(-1, 3)
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0.0, 1.0) * 255.0
        cols = cols.astype(np.uint8)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"Points/colors count mismatch: {pts.shape[0]} vs {cols.shape[0]}")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT reconstruction from raw RGB frames and unproject depth into a point cloud."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("/home/ghr/fs/Junyi/proj/debug/vggt_first_episode/episode_0000_env_00/raw_frames"),
        help="Directory containing frame_XXXX_rgb.png files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ghr/fs/Junyi/proj/debug/vggt_first_episode/episode_0000_env_00/reconstruct_from_raw_rgb"),
        help="Directory to save VGGT outputs.",
    )
    parser.add_argument("--model-id", type=str, default="facebook/VGGT-1B", help="HuggingFace model id.")
    parser.add_argument("--cache-dir", type=str, default="", help="Model cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Load model only from local cache.",
    )
    parser.add_argument(
        "--allow-download",
        dest="local_files_only",
        action="store_false",
        help="Allow downloading model weights when not found locally.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. auto/cuda/cpu.")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max-frames", type=int, default=0, help="Use first N frames. 0 means all.")
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Preprocessing mode in vggt.utils.load_fn.load_and_preprocess_images.",
    )
    parser.add_argument("--depth-min", type=float, default=1e-8, help="Depth threshold for valid points.")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save VGGT intermediate feature map for downstream networks.",
    )
    parser.add_argument(
        "--intermediate-layer-idx",
        type=int,
        default=-1,
        help="Aggregated-token layer index for intermediate feature extraction (default: -1, last layer).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_frames_dir = args.raw_frames_dir.expanduser().resolve()
    if not raw_frames_dir.exists():
        raise FileNotFoundError(f"raw_frames_dir not found: {raw_frames_dir}")

    rgb_paths = _find_rgb_frames(raw_frames_dir)
    if not rgb_paths:
        raise RuntimeError(f"No RGB frames found in {raw_frames_dir} (expected frame_*_rgb.png)")

    if args.max_frames > 0:
        rgb_paths = rgb_paths[: int(args.max_frames)]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] raw_frames_dir={raw_frames_dir}")
    print(f"[INFO] num_frames={len(rgb_paths)}")
    print(f"[INFO] output_dir={output_dir}")

    device = _resolve_device(args.device)
    print(f"[INFO] device={device}")

    images = load_and_preprocess_images([str(p) for p in rgb_paths], mode=args.preprocess_mode).to(device)
    # Model expects [S, 3, H, W] or [B, S, 3, H, W]. We use [S, 3, H, W].
    print(f"[INFO] images.shape={tuple(images.shape)}")

    model = VGGT.from_pretrained(
        args.model_id,
        local_files_only=bool(args.local_files_only),
        cache_dir=args.cache_dir or None,
    ).to(device)
    model.eval()

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            predictions = model(
                images,
                return_intermediates=bool(args.save_intermediate),
                intermediate_layer_idx=int(args.intermediate_layer_idx),
            )

    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    depth = predictions["depth"]
    if depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    extr_np = extrinsics.squeeze(0).detach().cpu().numpy().astype(np.float32)
    intr_np = intrinsics.squeeze(0).detach().cpu().numpy().astype(np.float32)
    img_np = predictions["images"].squeeze(0).detach().cpu().numpy().astype(np.float32)  # (S, 3, H, W)

    point_maps = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)
    points = point_maps.reshape(-1, 3)
    colors = img_np.transpose(0, 2, 3, 1).reshape(-1, 3)
    valid = np.isfinite(depth_np.reshape(-1)) & (depth_np.reshape(-1) > float(args.depth_min))
    points = points[valid]
    colors = colors[valid]

    np.save(str(output_dir / "rgb_paths.npy"), np.array([str(p) for p in rgb_paths], dtype=object))
    np.save(str(output_dir / "pred_depth.npy"), depth_np)
    np.save(str(output_dir / "pred_extrinsics.npy"), extr_np)
    np.save(str(output_dir / "pred_intrinsics.npy"), intr_np)
    np.save(str(output_dir / "pred_point_map.npy"), point_maps.astype(np.float32))
    np.save(str(output_dir / "pred_points.npy"), points.astype(np.float32))
    np.save(str(output_dir / "pred_colors.npy"), colors.astype(np.float32))

    if bool(args.save_intermediate):
        feat = predictions["intermediate_feature_map"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        patch_tokens = predictions["intermediate_patch_tokens"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.save(str(output_dir / "pred_intermediate_feature_map.npy"), feat)
        np.save(str(output_dir / "pred_intermediate_patch_tokens.npy"), patch_tokens)
        with (output_dir / "pred_intermediate_meta.txt").open("w", encoding="utf-8") as f:
            f.write(f"intermediate_layer_idx={int(predictions['intermediate_layer_idx'])}\n")
            f.write(f"patch_start_idx={int(predictions['patch_start_idx'])}\n")
            f.write(f"feature_map_shape={tuple(feat.shape)}\n")
            f.write(f"patch_tokens_shape={tuple(patch_tokens.shape)}\n")

    _write_xyzrgb_ply(points, colors, output_dir / "pred_points_rgb.ply")

    print(f"[INFO] saved points={points.shape[0]}")
    print(f"[INFO] saved: {output_dir / 'pred_points_rgb.ply'}")
    print("[INFO] done")


if __name__ == "__main__":
    main()
