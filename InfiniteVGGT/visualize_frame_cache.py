import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.append("src/")

from viser_utils import PointCloudViewer, convert_scene_output_to_glb


def _to_numpy_image(img_tensor: torch.Tensor) -> np.ndarray:
    if img_tensor.ndim != 4 or img_tensor.shape[0] != 1:
        raise ValueError(f"Expected image tensor shape (1, 3, H, W), got {tuple(img_tensor.shape)}")
    return img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()


def _to_numpy_mat(tensor: torch.Tensor, expected_ndim: int) -> np.ndarray:
    if tensor.ndim != expected_ndim + 1 or tensor.shape[0] != 1:
        raise ValueError(f"Expected tensor with leading batch dim 1, got {tuple(tensor.shape)}")
    return tensor[0].detach().cpu().numpy()


def load_frame_cache(cache_dir: str, frame_stride: int = 1, max_frames: int | None = None):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {cache_dir}")

    files = files[::frame_stride]
    if max_frames is not None:
        files = files[:max_frames]

    pts_list = []
    color_list = []
    conf_list = []
    cams2world = []
    focals = []
    principal_points = []

    for path in files:
        payload = torch.load(path, map_location="cpu")
        pred = payload["pred"]
        view = payload["view"]

        pts = _to_numpy_mat(pred["pts3d_in_other_view"], 3)
        conf = _to_numpy_mat(pred["conf"], 2)
        img = _to_numpy_image(view["img"])
        c2w = _to_numpy_mat(view["camera_pose"], 2)

        intrinsic_tensor = view.get("intrinsic", pred.get("intrinsic"))
        if intrinsic_tensor is None:
            raise KeyError(f"Missing intrinsic matrix in {path}")
        intrinsic = _to_numpy_mat(intrinsic_tensor, 2)

        pts_list.append(pts.astype(np.float32, copy=False))
        color_list.append(img.astype(np.float32, copy=False))
        conf_list.append(conf.astype(np.float32, copy=False))
        cams2world.append(c2w.astype(np.float32, copy=False))
        focals.append(float(intrinsic[0, 0]))
        principal_points.append(intrinsic[:2, 2].astype(np.float32, copy=False))

    cams2world = np.stack(cams2world, axis=0)
    return {
        "files": files,
        "pts": pts_list,
        "colors": color_list,
        "confs": conf_list,
        "R": cams2world[:, :3, :3],
        "t": cams2world[:, :3, 3],
        "cams2world": cams2world,
        "focal": np.asarray(focals, dtype=np.float32),
        "pp": np.stack(principal_points, axis=0),
    }


def export_glb(vis_data, out_path: str, vis_threshold: float, show_camera: bool, as_pointcloud: bool):
    outdir = os.path.dirname(out_path) or "."
    os.makedirs(outdir, exist_ok=True)
    save_name = os.path.splitext(os.path.basename(out_path))[0]
    masks = [conf > vis_threshold for conf in vis_data["confs"]]
    saved_path = convert_scene_output_to_glb(
        outdir=outdir,
        imgs=vis_data["colors"],
        pts3d=vis_data["pts"],
        mask=masks,
        focals=vis_data["focal"],
        cams2world=vis_data["cams2world"],
        show_cam=show_camera,
        as_pointcloud=as_pointcloud,
        save_name=save_name,
    )
    print(f"Exported GLB to {saved_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize StreamVGGT per-frame cache files written by --frame_cache_dir."
    )
    parser.add_argument("--cache_dir", required=True, help="Directory containing 000000.pt, 000001.pt, ...")
    parser.add_argument("--port", type=int, default=9999, help="Viser server port")
    parser.add_argument("--device", default="cpu", help="Viewer device, cpu is sufficient")
    parser.add_argument("--vis_threshold", type=float, default=1.5, help="Confidence threshold for display")
    parser.add_argument("--downsample_factor", type=int, default=50, help="Initial point downsample factor")
    parser.add_argument("--frame_stride", type=int, default=1, help="Load every Nth frame")
    parser.add_argument("--max_frames", type=int, default=None, help="Load at most this many frames")
    parser.add_argument("--no_viewer", action="store_true", help="Do not launch the interactive viewer")
    parser.add_argument("--glb_out", default=None, help="Optional path to export a .glb scene")
    parser.add_argument(
        "--glb_as_pointcloud",
        action="store_true",
        help="Export GLB as a point cloud instead of a fused mesh",
    )
    parser.add_argument("--hide_camera", action="store_true", help="Do not show camera frustums")
    args = parser.parse_args()

    vis_data = load_frame_cache(
        cache_dir=args.cache_dir,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    total_points = int(sum(np.prod(conf.shape) for conf in vis_data["confs"]))
    print(
        f"Loaded {len(vis_data['files'])} frames from {args.cache_dir} "
        f"({total_points} raw points before confidence filtering/downsampling)."
    )

    if args.glb_out:
        export_glb(
            vis_data=vis_data,
            out_path=args.glb_out,
            vis_threshold=args.vis_threshold,
            show_camera=not args.hide_camera,
            as_pointcloud=args.glb_as_pointcloud,
        )

    if args.no_viewer:
        return

    print(f"Launching Viser on port {args.port}")
    viewer = PointCloudViewer(
        model=None,
        state_args=None,
        pc_list=vis_data["pts"],
        color_list=vis_data["colors"],
        conf_list=vis_data["confs"],
        cam_dict={
            "R": vis_data["R"],
            "t": vis_data["t"],
            "focal": vis_data["focal"],
            "pp": vis_data["pp"],
        },
        gt_poses=None,
        device=args.device,
        port=args.port,
        show_camera=not args.hide_camera,
        vis_threshold=args.vis_threshold,
        downsample_factor=args.downsample_factor,
    )
    viewer.run()


if __name__ == "__main__":
    main()
