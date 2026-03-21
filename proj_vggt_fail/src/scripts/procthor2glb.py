"""Convert ProcTHOR scene_instance configs to textured GLB offline.

This script reads ProcTHOR scene_instance.json files from ai2thor-hab,
loads the referenced stage/object assets, applies per-instance transforms,
and exports a combined GLB. It can optionally use the higher quality
ai2thorhab-uncompressed object assets.

Example:
    python -m actrec.scripts.procthor2glb \
        --dataset-root /datasets/.../ai2thor-hab \
        --object-asset-dir /datasets/.../ai2thorhab-uncompressed/assets/objects \
        --scene-id ProcTHOR-Test-606 \
        --output-dir ./procthor_glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import trimesh


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _qmult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_from_axis_deg(axis: Iterable[float], degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return trimesh.transformations.quaternion_about_axis(np.deg2rad(degrees), axis)


def _proc_thor_stage_transform() -> np.ndarray:
    rot_x = _quat_from_axis_deg([1, 0, 0], 90)
    rot_y = _quat_from_axis_deg([0, -1, 0], 90)
    quat = _qmult(rot_x, rot_y)
    return trimesh.transformations.quaternion_matrix(quat)


def _proc_thor_object_transform(
    translation: Optional[Iterable[float]],
    rotation_wxyz: Optional[Iterable[float]],
    scale: Optional[Iterable[float]],
) -> np.ndarray:
    # Match ManiSkill ProcTHOR builder: rotate + remap translation axes.
    remapped_translation = None
    if translation is not None:
        t = np.asarray(translation, dtype=np.float32)
        remapped_translation = np.array([t[0], -t[2], t[1]], dtype=np.float32)

    rot_x = _quat_from_axis_deg([1, 0, 0], 90)
    quat = rot_x
    if rotation_wxyz is not None:
        quat = _qmult(rot_x, np.asarray(rotation_wxyz, dtype=np.float32))

    mat = trimesh.transformations.quaternion_matrix(quat)
    if remapped_translation is not None:
        mat[:3, 3] = remapped_translation

    if scale is not None:
        sc = np.asarray(scale, dtype=np.float32)
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = sc[0]
        scale_mat[1, 1] = sc[1]
        scale_mat[2, 2] = sc[2]
        mat = mat @ scale_mat
    return mat


def _z_up_to_y_up_transform() -> np.ndarray:
    # Convert from z-up to y-up (glTF convention).
    return trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])


def _y_up_to_z_up_transform() -> np.ndarray:
    # Convert from y-up (glTF convention) to z-up.
    return trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])


def _load_meshes(file_path: Path, transform: Optional[np.ndarray]) -> List[trimesh.Trimesh]:
    try:
        asset = trimesh.load(file_path, force="scene", process=False)
    except Exception as exc:  # pragma: no cover - best effort load
        print(f"[warn] failed to load {file_path}: {exc}")
        return []

    meshes: List[trimesh.Trimesh] = []

    if isinstance(asset, trimesh.Trimesh):
        mesh = asset.copy()
        if transform is not None:
            mesh.apply_transform(transform)
        meshes.append(mesh)
        return meshes

    if isinstance(asset, trimesh.Scene):
        for node_name in asset.graph.nodes_geometry:
            node_transform, geom_name = asset.graph[node_name]
            geom = asset.geometry[geom_name]
            mesh = geom.copy()
            if node_transform is not None:
                mesh.apply_transform(node_transform)
            if transform is not None:
                mesh.apply_transform(transform)
            meshes.append(mesh)
    return meshes


def _add_meshes(scene: trimesh.Scene, meshes: List[trimesh.Trimesh], name_prefix: str) -> None:
    for idx, mesh in enumerate(meshes):
        geom_name = f"{name_prefix}_geom_{idx}"
        node_name = f"{name_prefix}_node_{idx}"
        scene.add_geometry(mesh, geom_name=geom_name, node_name=node_name)


def _resolve_render_asset(config_path: Path, render_asset: str) -> Path:
    asset_path = (config_path.parent / render_asset).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Render asset not found: {asset_path}")
    return asset_path


def _resolve_object_config(objects_cfg_dir: Path, template_name: str) -> Path:
    if not template_name.startswith("objects/"):
        raise FileNotFoundError(f"Unexpected object template name: {template_name}")
    rel = template_name[len("objects/") :]
    cfg = objects_cfg_dir / f"{rel}.object_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Object config not found: {cfg}")
    return cfg


def _resolve_stage_config(stages_cfg_dir: Path, template_name: str) -> Path:
    if not template_name.startswith("stages/"):
        raise FileNotFoundError(f"Unexpected stage template name: {template_name}")
    rel = template_name[len("stages/") :]
    cfg = stages_cfg_dir / f"{rel}.stage_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Stage config not found: {cfg}")
    return cfg


def _resolve_object_asset(obj_cfg_path: Path, render_asset: str, object_asset_dir: Optional[Path]) -> Path:
    if object_asset_dir is None:
        return _resolve_render_asset(obj_cfg_path, render_asset)
    asset_name = Path(render_asset).name
    asset_path = (object_asset_dir / asset_name).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Object asset not found: {asset_path}")
    return asset_path


def _scene_instance_paths(scenes_root: Path, scene_id: Optional[str]) -> List[Path]:
    if scene_id:
        if scene_id.endswith(".scene_instance.json"):
            candidate = scenes_root / scene_id
            if candidate.exists():
                return [candidate]
        matches = list(scenes_root.rglob(f"{scene_id}.scene_instance.json"))
        if matches:
            return matches
    return sorted(scenes_root.rglob("*.scene_instance.json"))


def convert_scene(
    scene_path: Path,
    objects_cfg_dir: Path,
    stages_cfg_dir: Path,
    object_asset_dir: Optional[Path],
    output_path: Path,
    export_y_up: bool,
    normalize_axes: bool,
) -> None:
    data = _load_json(scene_path)
    scene = trimesh.Scene()

    stage_instance = data.get("stage_instance", {})
    scene_transform = _z_up_to_y_up_transform() if export_y_up else np.eye(4, dtype=np.float32)
    stage_template = stage_instance.get("template_name")
    if stage_template:
        stage_cfg_path = _resolve_stage_config(stages_cfg_dir, stage_template)
        stage_cfg = _load_json(stage_cfg_path)
        stage_asset = _resolve_render_asset(stage_cfg_path, stage_cfg["render_asset"])
        stage_transform = _proc_thor_stage_transform()
        stage_meshes = _load_meshes(stage_asset, transform=scene_transform @ stage_transform)
        _add_meshes(scene, stage_meshes, name_prefix="stage")

    object_instances = data.get("object_instances", [])
    for idx, obj in enumerate(object_instances):
        template_name = obj.get("template_name")
        if not template_name:
            continue
        try:
            obj_cfg_path = _resolve_object_config(objects_cfg_dir, template_name)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue
        obj_cfg = _load_json(obj_cfg_path)
        try:
            obj_asset = _resolve_object_asset(obj_cfg_path, obj_cfg["render_asset"], object_asset_dir)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue

        translation = obj.get("translation")
        rotation = obj.get("rotation")
        scale = obj.get("non_uniform_scale")
        transform = _proc_thor_object_transform(translation, rotation, scale)
        transform = scene_transform @ transform

        obj_meshes = _load_meshes(obj_asset, transform)
        _add_meshes(scene, obj_meshes, name_prefix=f"obj_{idx}")

    # if normalize_axes and scene.geometry:
    #     bounds = scene.bounds
    #     if bounds is not None and np.isfinite(bounds).all():
    #         if export_y_up:
    #             rot = _y_up_to_z_up_transform()
    #             tmp_scene = scene.copy()
    #             tmp_scene.apply_transform(rot)
    #             rot_bounds = tmp_scene.bounds
    #             if rot_bounds is not None and np.isfinite(rot_bounds).all():
    #                 center_xy = 0.5 * (rot_bounds[0, :2] + rot_bounds[1, :2])
    #                 shift_view = np.array([-center_xy[0], -center_xy[1], -rot_bounds[0, 2]], dtype=np.float32)
    #                 rot_inv = np.linalg.inv(rot)
    #                 shift = rot_inv[:3, :3] @ shift_view
    #                 transform = np.eye(4, dtype=np.float32)
    #                 transform[:3, 3] = shift
    #                 scene.apply_transform(transform)
    #                 print(f"[normalize] shift_glb={np.round(shift, 4)} view_shift={np.round(shift_view, 4)}")
    #             else:
    #                 print("[normalize] warning: rotated bounds unavailable; skip shift.")
    #         else:
    #             center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    #             shift = np.array([-center_xy[0], -center_xy[1], -bounds[0, 2]], dtype=np.float32)
    #             transform = np.eye(4, dtype=np.float32)
    #             transform[:3, 3] = shift
    #             scene.apply_transform(transform)
    #             print(f"[normalize] shift={np.round(shift, 4)}")
    #     else:
    #         print("[normalize] warning: scene bounds unavailable; skip shift.")

    if normalize_axes and scene.geometry:
        bounds = scene.bounds
        if bounds is not None and np.isfinite(bounds).all():
            # 1. 获取当前边界
            min_xyz = bounds[0]
            max_xyz = bounds[1]
            extents = max_xyz - min_xyz # [width, height, depth]
            
            # 2. 计算各轴的中心点
            center_x = 0.5 * (min_xyz[0] + max_xyz[0])
            center_y = min_xyz[1] # Y轴要从0开始，所以平移量是最小值
            center_z = 0.5 * (min_xyz[2] + max_xyz[2])

            # 3. 构造平移矩阵 (T_trans)
            # 将 XZ 移到原点中心，将 Y 移到 0 点
            shift = np.array([-center_x, -center_y, -center_z], dtype=np.float32)
            T_trans = np.eye(4, dtype=np.float32)
            T_trans[:3, 3] = shift

            # 4. 构造缩放矩阵 (T_scale)
            # X 目标跨度 2.0 (范围 -1到1)
            # Y 目标跨度 1.0 (范围 0到1)
            # Z 目标跨度 2.0 (范围 -1到1)
            scale_x = 2.0 / (extents[0] + 1e-8)
            scale_y = 1.0 / (extents[1] + 1e-8)
            scale_z = 2.0 / (extents[2] + 1e-8)
            
            T_scale = np.eye(4, dtype=np.float32)
            T_scale[0, 0] = scale_x
            T_scale[1, 1] = scale_y
            T_scale[2, 2] = scale_z

            # 5. 合并并应用变换
            combined_transform = T_scale @ T_trans
            scene.apply_transform(combined_transform)

            print(f"[normalize] Extents: {np.round(extents, 4)}")
            print(f"[normalize] Target: Y[0,1], XZ[-1,1]")
            print(f"[normalize] Applied Scale: [{scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f}]")
        else:
            print("[normalize] warning: scene bounds unavailable; skip normalization.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ProcTHOR scenes to textured GLB offline.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to ai2thor-hab root (contains assets/ and configs/).",
    )
    parser.add_argument(
        "--scenes-dir",
        default="configs/scenes/ProcTHOR",
        help="Relative path to ProcTHOR scene_instance.json directory.",
    )
    parser.add_argument(
        "--object-asset-dir",
        default="",
        help="Optional object asset dir (e.g. ai2thorhab-uncompressed/assets/objects).",
    )
    parser.add_argument("--scene-id", default="", help="Convert a single scene id.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit number of scenes.")
    parser.add_argument("--output-dir", default="procthor_glb", help="Output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GLB files.")
    parser.add_argument("--export-y-up", action="store_true", help="Export GLB in y-up coordinates.")
    parser.add_argument(
        "--normalize-axes",
        action="store_true",
        default=True,
        help="Normalize scene axes so z_min=0 and x/y are centered around 0.",
    )
    parser.add_argument(
        "--no-normalize-axes",
        dest="normalize_axes",
        action="store_false",
        help="Disable axis normalization.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    scenes_root = dataset_root / args.scenes_dir
    if not scenes_root.exists():
        raise FileNotFoundError(f"Scenes dir not found: {scenes_root}")

    objects_cfg_dir = dataset_root / "configs/objects"
    stages_cfg_dir = dataset_root / "configs/stages"
    if not objects_cfg_dir.exists():
        raise FileNotFoundError(f"Object config dir not found: {objects_cfg_dir}")
    if not stages_cfg_dir.exists():
        raise FileNotFoundError(f"Stage config dir not found: {stages_cfg_dir}")

    object_asset_dir = Path(args.object_asset_dir).resolve() if args.object_asset_dir else None
    if object_asset_dir is not None and not object_asset_dir.exists():
        raise FileNotFoundError(f"Object asset dir not found: {object_asset_dir}")

    scene_paths = _scene_instance_paths(scenes_root, args.scene_id or None)
    if args.max_scenes > 0:
        scene_paths = scene_paths[: args.max_scenes]
    if not scene_paths:
        raise FileNotFoundError("No scene_instance.json files found.")

    output_dir = Path(args.output_dir)

    for scene_path in scene_paths:
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        output_path = output_dir / f"{scene_id}.glb"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists")
            continue
        print(f"[convert] {scene_id} -> {output_path}")
        convert_scene(
            scene_path,
            objects_cfg_dir,
            stages_cfg_dir,
            object_asset_dir,
            output_path,
            export_y_up=args.export_y_up,
            normalize_axes=args.normalize_axes,
        )


if __name__ == "__main__":
    main()
