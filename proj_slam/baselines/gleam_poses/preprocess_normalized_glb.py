"""Preprocess-only entry for GLBs that are already normalized to:
  - Y-up input
  - X/Z in [-1, 1]
  - Y in [0, 1]

This script only produces the GLEAM preprocessing artifacts and does not
launch the policy, simulation rollout, or pose JSON generation.
"""
from __future__ import annotations
print(1)
import argparse
from pathlib import Path
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

from baselines.gleam_poses.generate_normalized_glb import preprocess_single_glb_with_mode
from baselines.gleam_poses import generate as base_generate


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a normalized or raw Y-up GLB into GLEAM artifacts only."
    )
    parser.add_argument("--glb", required=True, help="Path to the input Y-up .glb scene file.")
    parser.add_argument("--scene_id", required=True, help="Scene identifier, e.g. replicacad/apt_0.")
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
    parser.add_argument("--drone_height", type=float, default=0.5, help="Height used when generating 2D occupancy/init maps.")
    parser.add_argument("--grid_reso", type=int, default=128, help="Voxel grid resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild preprocessing artifacts even if they already exist.")
    args = parser.parse_args()

    work_dir = str(Path(args.work_dir or base_generate.DEFAULT_GLEAM_WORK_DIR))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    scene_safe = args.scene_id.replace("/", "__").replace(" ", "_")
    dataset_name = f"glb_{scene_safe}"

    print(
        f"[mode] input_glb_mode={args.input_glb_mode} "
        f"(dataset_name remains {dataset_name}; switching modes in one work_dir usually needs --overwrite)"
    )

    data_dir = preprocess_single_glb_with_mode(
        glb_path=args.glb,
        work_dir=work_dir,
        dataset_name=dataset_name,
        drone_height=args.drone_height,
        grid_reso=args.grid_reso,
        overwrite=args.overwrite,
        input_glb_mode=args.input_glb_mode,
    )

    gt_dir = Path(data_dir) / "gt" / f"gt_{dataset_name}"
    urdf_dir = Path(data_dir) / "urdf" / dataset_name
    obj_dir = Path(data_dir) / "objects" / f"{dataset_name}_obj"
    ply_dir = Path(data_dir) / "objects" / f"{dataset_name}_ply_center"

    print("\n[done] Preprocessing complete")
    print(f"       data_dir: {data_dir}")
    print(f"       gt_dir:   {gt_dir}")
    print(f"       urdf_dir: {urdf_dir}")
    print(f"       obj_dir:  {obj_dir}")
    print(f"       ply_dir:  {ply_dir}")


if __name__ == "__main__":
    main()
