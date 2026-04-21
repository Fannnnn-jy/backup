"""GLEAM pose generation for GLBs that are already normalized to:
  - Y-up input
  - X/Z in [-1, 1]
  - Y in [0, 1]

This variant fixes the main incompatibility in the original pipeline:
the GLB scene graph transforms are baked before concatenation, so
normalization applied upstream during export is preserved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parents[1].name == "baselines":
    PROJ_ROOT = THIS_FILE.parents[2]
    GLEAM_ROOT = THIS_FILE.parents[3] / "GLEAM"
else:
    PROJ_ROOT = THIS_FILE.parents[1] / "proj"
    GLEAM_ROOT = THIS_FILE.parent

sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(GLEAM_ROOT))

from baselines.gleam_poses import generate as base_generate


def _bake_scene_to_single_mesh(glb_path: str):
    import trimesh

    loaded = trimesh.load(glb_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()

    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported GLB asset type: {type(loaded)}")

    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        node_transform, geom_name = loaded.graph[node_name]
        geom = loaded.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        if node_transform is not None:
            mesh.apply_transform(node_transform)
        meshes.append(mesh)

    if not meshes:
        raise ValueError(f"No trimesh geometries found in {glb_path}")
    return trimesh.util.concatenate(meshes)


def preprocess_normalized_single_glb(
    glb_path: str,
    work_dir: str,
    dataset_name: str = "glb_single",
    drone_height: float = 0.5,
    grid_reso: int = 128,
    overwrite: bool = False,
) -> str:
    """Preprocess one already-normalized GLB into GLEAM artifacts.

    Differences from the original pipeline:
      1. Bake scene graph transforms before concatenation.
      2. Preserve normalized scale/frame after Y-up -> Z-up.
      3. Skip re-centering and large-scene rescaling heuristics.
    """
    base_generate._ensure_gleam_preprocess_imports()
    base_generate._normalize_legacy_data_gleam_layout(work_dir)

    glb_dir = str(Path(glb_path).parent)
    glb_file = Path(glb_path).name

    orig_cwd = os.getcwd()
    os.chdir(work_dir)

    import legged_gym as _lgy

    orig_root = _lgy.OPEN_ROBOT_ROOT_DIR
    orig_up_root = base_generate.unified_preprocess.OPEN_ROBOT_ROOT_DIR
    _lgy.OPEN_ROBOT_ROOT_DIR = work_dir
    base_generate.unified_preprocess.OPEN_ROBOT_ROOT_DIR = work_dir

    try:
        print(f"\n[preprocess-normalized] GLB -> {work_dir}")
        glb_files = [glb_file]
        base = "data_gleam"
        ply_dir = os.path.join(base, f"objects/{dataset_name}_ply_center")
        obj_dir = os.path.join(base, f"objects/{dataset_name}_obj")
        urdf_dir = os.path.join(base, f"urdf/{dataset_name}")
        for directory in (ply_dir, obj_dir, urdf_dir):
            os.makedirs(directory, exist_ok=True)

        for scene_idx, glb_name in enumerate(glb_files):
            glb_scene_path = os.path.join(glb_dir, glb_name)
            ply_path = os.path.join(ply_dir, f"scene_{scene_idx}.ply")
            obj_path = os.path.join(obj_dir, f"scene_{scene_idx}.obj")
            urdf_path = os.path.join(urdf_dir, f"scene_{scene_idx}.urdf")
            bounds_path = os.path.join(base, f"objects/{dataset_name}_scene_{scene_idx}_bounds.json")
            if not overwrite and base_generate.unified_preprocess._outputs_exist(ply_path, obj_path, urdf_path):
                base_generate.unified_preprocess._log_skip_existing("Step 0 normalized", ply_path, obj_path, urdf_path)
                continue

            mesh = _bake_scene_to_single_mesh(glb_scene_path)
            vertices_yup = np.asarray(mesh.vertices, dtype=np.float64)
            bounds_yup = {
                "min": vertices_yup.min(axis=0).tolist(),
                "max": vertices_yup.max(axis=0).tolist(),
            }

            # Preserve normalized frame while converting only the up-axis convention.
            vertices_zup = (base_generate.Y_UP_TO_Z_UP @ vertices_yup.T).T
            mesh.vertices = vertices_zup
            bounds_zup = {
                "min": vertices_zup.min(axis=0).tolist(),
                "max": vertices_zup.max(axis=0).tolist(),
            }

            mesh.export(ply_path)
            mesh.export(obj_path)
            obj_rel = os.path.relpath(obj_path, urdf_dir)
            base_generate.unified_preprocess._generate_scene_urdf(scene_idx, obj_rel, urdf_path)

            with open(bounds_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "glb_path": glb_scene_path,
                        "assumption": "input_glb_is_already_normalized_y_up",
                        "bounds_y_up_baked": bounds_yup,
                        "bounds_z_up_preserved": bounds_zup,
                    },
                    handle,
                    indent=2,
                )

            print(f"  vertices={len(vertices_zup):,}")
            print(f"  Y-up baked bounds: min={np.round(bounds_yup['min'], 4)} max={np.round(bounds_yup['max'], 4)}")
            print(f"  Z-up kept  bounds: min={np.round(bounds_zup['min'], 4)} max={np.round(bounds_zup['max'], 4)}")
            print(f"  PLY  -> {ply_path}")
            print(f"  OBJ  -> {obj_path}")
            print(f"  URDF -> {urdf_path}")

        base_generate.voxelize_scenes(dataset_name, grid_reso, overwrite=overwrite)
        base_generate.merge_voxel_data(dataset_name, grid_reso, overwrite=overwrite)
        base_generate.generate_2d_maps(dataset_name, grid_reso, height=drone_height, overwrite=overwrite)
        base_generate.convert_to_h5(dataset_name, grid_reso, overwrite=overwrite)
        base_generate._normalize_legacy_data_gleam_layout(work_dir)
    finally:
        os.chdir(orig_cwd)
        _lgy.OPEN_ROBOT_ROOT_DIR = orig_root
        base_generate.unified_preprocess.OPEN_ROBOT_ROOT_DIR = orig_up_root

    data_dir = os.path.join(work_dir, "data_gleam")
    print(f"[preprocess-normalized] Done. Data dir: {data_dir}")
    return data_dir


def preprocess_single_glb_with_mode(
    glb_path: str,
    work_dir: str,
    dataset_name: str = "glb_single",
    drone_height: float = 0.5,
    grid_reso: int = 128,
    overwrite: bool = False,
    input_glb_mode: str = "normalized",
) -> str:
    mode = input_glb_mode.strip().lower()
    if mode == "normalized":
        return preprocess_normalized_single_glb(
            glb_path=glb_path,
            work_dir=work_dir,
            dataset_name=dataset_name,
            drone_height=drone_height,
            grid_reso=grid_reso,
            overwrite=overwrite,
        )
    if mode == "raw":
        return base_generate.preprocess_single_glb(
            glb_path=glb_path,
            work_dir=work_dir,
            dataset_name=dataset_name,
            drone_height=drone_height,
            grid_reso=grid_reso,
            overwrite=overwrite,
        )
    raise ValueError(f"Unsupported input_glb_mode: {input_glb_mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GLEAM exploration poses for either normalized or raw Y-up GLB scenes."
    )
    parser.add_argument("--glb", required=True, help="Path to the input Y-up .glb scene file.")
    parser.add_argument("--scene_id", required=True, help="Scene identifier, e.g. replicacad/apt_0.")
    parser.add_argument("--n_views", type=int, default=30, help="Number of views to collect per episode.")
    parser.add_argument("--drone_height", type=float, default=0.5, help="Drone Z-height in normalized Z-up world.")
    parser.add_argument("--run_id", default="run_000")
    parser.add_argument("--method", default="gleam")
    parser.add_argument("--output", default="baselines/poses", help="Root directory for pose JSON output.")
    parser.add_argument(
        "--work_dir",
        default=None,
        help="Directory for preprocessing artifacts. Defaults to Junyi/data/proj/baseline_data/tmp.",
    )
    parser.add_argument(
        "--input_glb_mode",
        choices=("normalized", "raw"),
        default="normalized",
        help="Use 'normalized' for GLBs already mapped to X/Z[-1,1], Y[0,1]; use 'raw' for the original pipeline.",
    )
    parser.add_argument("--keep_work_dir", action="store_true", help="Keep the preprocessing work directory.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild preprocessing artifacts even if they already exist.")
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--camera_near", type=float, default=0.01)
    parser.add_argument("--camera_far", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=1, help="Print one GLEAM progress line every N steps. Use 0 to disable.")
    parser.add_argument("--init_x", type=float, default=None, help="Fixed starting X in normalized Z-up coords.")
    parser.add_argument("--init_y", type=float, default=None, help="Fixed starting Y in normalized Z-up coords.")
    parser.add_argument(
        "--init_yaw_deg",
        type=float,
        default=None,
        help="Fixed starting yaw in degrees (Z-up, CCW from +X). If omitted, uses init-map sampling.",
    )
    args = parser.parse_args()

    if not base_generate.GLEAM_POLICY_CKPT.is_file():
        raise FileNotFoundError(f"GLEAM checkpoint not found: {base_generate.GLEAM_POLICY_CKPT}")

    work_dir = str(Path(args.work_dir or base_generate.DEFAULT_GLEAM_WORK_DIR))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    scene_safe = args.scene_id.replace("/", "__").replace(" ", "_")
    dataset_name = f"glb_{scene_safe}"

    print(
        f"[mode] input_glb_mode={args.input_glb_mode} "
        f"(dataset_name remains {dataset_name}; reuse requires --overwrite if outputs already exist)"
    )

    data_dir = preprocess_single_glb_with_mode(
        glb_path=args.glb,
        work_dir=work_dir,
        dataset_name=dataset_name,
        drone_height=args.drone_height,
        grid_reso=128,
        overwrite=args.overwrite,
        input_glb_mode=args.input_glb_mode,
    )

    forced_init_xy = (args.init_x, args.init_y) if args.init_x is not None and args.init_y is not None else None
    forced_init_yaw = float(np.deg2rad(args.init_yaw_deg)) if args.init_yaw_deg is not None else None

    poses_zup = base_generate.run_gleam_episode(
        glb_data_dir=data_dir,
        dataset_name=dataset_name,
        drone_height=args.drone_height,
        n_views=args.n_views,
        ckpt_path=str(base_generate.GLEAM_POLICY_CKPT),
        device=args.device,
        buffer_size=args.buffer_size,
        log_every=args.log_every,
        forced_init_xy=forced_init_xy,
        forced_init_yaw=forced_init_yaw,
    )

    pose_json = base_generate.poses_zup_to_json(
        poses_zup=poses_zup,
        scene_id=args.scene_id,
        method=args.method,
        run_id=args.run_id,
        fov_deg=args.fov_deg,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_near=args.camera_near,
        camera_far=args.camera_far,
    )

    out_dir = Path(args.output) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_safe}_{args.run_id}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(pose_json, handle, indent=2)
    print(f"\n[done] Pose JSON saved to {out_path}")
    print(f"       Views: {len(pose_json['views'])}")

if __name__ == "__main__":
    main()
