"""
Convert HSSD scene_instance configs into textured GLB meshes.

This script loads a scene_instance.json, resolves the stage and object
render assets, applies per-instance transforms, and exports a single GLB
with textures preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import trimesh
from yourdfpy import URDF


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _build_object_index(objects_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for cfg_path in objects_root.rglob("*.object_config.json"):
        name = cfg_path.name.replace(".object_config.json", "")
        index[name] = cfg_path
    return index


def _build_stage_index(stages_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for cfg_path in stages_root.glob("*.stage_config.json"):
        name = cfg_path.name.replace(".stage_config.json", "")
        index[f"stages/{name}"] = cfg_path
    return index


def _build_urdf_index(urdf_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for cfg_path in urdf_root.rglob("*.ao_config.json"):
        name = cfg_path.name.replace(".ao_config.json", "")
        index[name] = cfg_path
    return index


def _compose_transform(
    translation: Optional[Iterable[float]] = None,
    rotation_wxyz: Optional[Iterable[float]] = None,
    scale: Optional[Iterable[float]] = None,
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    if translation is not None:
        mat[:3, 3] = np.asarray(translation, dtype=np.float32)
    if rotation_wxyz is not None:
        quat = np.asarray(rotation_wxyz, dtype=np.float32)
        if quat.shape[0] == 4:
            rot = trimesh.transformations.quaternion_matrix(quat)
            mat = mat @ rot
    if scale is not None:
        sc = np.asarray(scale, dtype=np.float32)
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = sc[0]
        scale_mat[1, 1] = sc[1]
        scale_mat[2, 2] = sc[2]
        mat = mat @ scale_mat
    return mat


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


def _resolve_urdf_asset(config_path: Path, urdf_asset: str) -> Path:
    asset_path = (config_path.parent / urdf_asset).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"URDF asset not found: {asset_path}")
    return asset_path


def _parse_initial_joint_pose(robot: URDF, pose_data: dict | None) -> Dict[str, float]:
    if not pose_data:
        return {}
    joint_names = robot.actuated_joint_names
    joint_cfg: Dict[str, float] = {}
    for key, value in pose_data.items():
        if isinstance(key, str) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(joint_names):
                joint_cfg[joint_names[idx]] = float(value)
        elif isinstance(key, str) and key in robot.joint_map:
            joint_cfg[key] = float(value)
    return joint_cfg


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _is_door_name(name: str) -> bool:
    tokens = _tokenize(name)
    return "door" in tokens or "doors" in tokens


def _is_door_object(template_name: str, semantics: Dict[str, str], names: Dict[str, str]) -> bool:
    category = semantics.get(template_name, "")
    obj_name = names.get(template_name, "")
    return _is_door_name(category) or _is_door_name(obj_name)


def _load_semantics_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"[warn] semantics CSV not found: {path}")
        return {}
    semantics: Dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return semantics
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            cat = ""
            if len(row) > 3 and row[3].strip():
                cat = row[3].strip()
            elif len(row) > 4 and row[4].strip():
                cat = row[4].strip()
            semantics[key] = cat
    return semantics


def _load_object_names(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"[warn] objects.json not found: {path}")
        return {}
    with path.open("r") as f:
        data = json.load(f)
    names: Dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                name = value.get("name", "")
                if name:
                    names[key] = name
    return names


def _load_fpmodels_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"[warn] fpmodels CSV not found: {path}")
        return {}
    fpmodels: Dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return fpmodels
        idx_main = header.index("main_category") if "main_category" in header else None
        idx_wn = header.index("wnsynsetkey") if "wnsynsetkey" in header else None
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            parts = []
            if idx_main is not None and idx_main < len(row):
                parts.append(row[idx_main])
            if idx_wn is not None and idx_wn < len(row):
                parts.append(row[idx_wn])
            fpmodels[key] = " ".join(p for p in parts if p)
    return fpmodels


def _collect_door_names(robot: URDF) -> set[str]:
    names: set[str] = set()
    for link_name in robot.link_map.keys():
        if _is_door_name(link_name):
            names.add(link_name)
    for joint_name, joint in robot.joint_map.items():
        joint_parent = getattr(joint, "parent", None)
        joint_child = getattr(joint, "child", None)
        if (
            _is_door_name(joint_name)
            or (joint_child and _is_door_name(joint_child))
            or (joint_parent and _is_door_name(joint_parent))
        ):
            names.add(joint_name)
            if joint_parent:
                names.add(joint_parent)
            if joint_child:
                names.add(joint_child)
    return names


def _urdf_scene_meshes(
    robot_scene: trimesh.Scene,
    transform: Optional[np.ndarray],
    skip_names: Optional[set[str]] = None,
    skip_substrings: Optional[Iterable[str]] = None,
) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []
    for node_name in robot_scene.graph.nodes_geometry:
        node_transform, geom_name = robot_scene.graph[node_name]
        if skip_names and (node_name in skip_names or geom_name in skip_names):
            continue
        if skip_substrings:
            name_blob = f"{node_name} {geom_name}".lower()
            if any(sub in name_blob for sub in skip_substrings):
                continue
        geom = robot_scene.geometry[geom_name]
        mesh = geom.copy()
        if node_transform is not None:
            mesh.apply_transform(node_transform)
        if transform is not None:
            mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HSSD scenes to textured GLB.")
    parser.add_argument(
        "--dataset-root",
        default="/datasets/v2p/current/fs/Junyi/data/scene_datasets/hssd",
        help="HSSD dataset root.",
    )
    parser.add_argument(
        "--scenes-dir",
        default="scenes",
        help="Subdirectory containing scene_instance.json files (e.g. scenes, scenes-uncluttered).",
    )
    parser.add_argument(
        "--output-dir",
        default="/datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/HSSD_glb",
        help="Output directory for combined GLB files.",
    )
    parser.add_argument(
        "--scene-id",
        default="",
        help="Optional scene id (stem) to convert, e.g. 102343992.",
    )
    parser.add_argument("--max-scenes", type=int, default=-1, help="Max number of scenes to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--include-articulated",
        action="store_true",
        help="Include articulated objects from articulated_object_instances.",
    )
    parser.add_argument(
        "--remove-doors",
        dest="remove_doors",
        action="store_true",
        help="Remove door objects/meshes (name or semantic category contains 'door').",
    )
    parser.add_argument(
        "--keep-doors",
        dest="remove_doors",
        action="store_false",
        help="Keep door objects/meshes.",
    )
    parser.add_argument(
        "--semantics-csv",
        default="",
        help="Override path to hssd_obj_semantics_condensed.csv.",
    )
    parser.add_argument(
        "--objects-json",
        default="",
        help="Override path to metadata/objects.json.",
    )
    parser.add_argument(
        "--fpmodels-csv",
        default="",
        help="Override path to metadata/fpmodels-with-decomposed.csv.",
    )
    parser.add_argument("--log-removed", action="store_true", help="Log removed door objects.")
    parser.set_defaults(remove_doors=True)
    return parser.parse_args()


def _scene_instance_paths(scene_dir: Path, scene_id: str) -> List[Path]:
    if scene_id:
        path = scene_dir / f"{scene_id}.scene_instance.json"
        return [path] if path.exists() else []
    return sorted(scene_dir.glob("*.scene_instance.json"))


def convert_scene(
    scene_path: Path,
    stage_index: Dict[str, Path],
    object_index: Dict[str, Path],
    urdf_index: Dict[str, Path],
    output_path: Path,
    include_articulated: bool,
    remove_doors: bool,
    semantics: Dict[str, str],
    object_names: Dict[str, str],
    fpmodels: Dict[str, str],
    log_removed: bool,
) -> None:
    data = _load_json(scene_path)

    scene = trimesh.Scene()

    stage_instance = data.get("stage_instance", {})
    stage_template = stage_instance.get("template_name")
    if stage_template:
        stage_cfg_path = stage_index.get(stage_template)
        if stage_cfg_path is None:
            raise FileNotFoundError(f"Stage config not found for template: {stage_template}")
        stage_cfg = _load_json(stage_cfg_path)
        stage_asset = _resolve_render_asset(stage_cfg_path, stage_cfg["render_asset"])
        stage_meshes = _load_meshes(stage_asset, transform=None)
        _add_meshes(scene, stage_meshes, name_prefix="stage")

    object_instances = data.get("object_instances", [])
    for idx, obj in enumerate(object_instances):
        template_name = obj.get("template_name")
        if not template_name:
            continue
        if remove_doors and _is_door_object(template_name, semantics, object_names):
            if log_removed:
                print(f"[remove] obj {template_name} name='{object_names.get(template_name, '')}'")
            continue
        if remove_doors and _is_door_name(fpmodels.get(template_name, "")):
            if log_removed:
                print(f"[remove] obj {template_name} fpmodels='{fpmodels.get(template_name, '')}'")
            continue
        obj_cfg_path = object_index.get(template_name)
        if obj_cfg_path is None:
            print(f"[warn] missing object config for {template_name}")
            continue
        obj_cfg = _load_json(obj_cfg_path)
        obj_asset = _resolve_render_asset(obj_cfg_path, obj_cfg["render_asset"])

        translation = obj.get("translation")
        rotation = obj.get("rotation")
        scale = obj.get("non_uniform_scale")
        transform = _compose_transform(translation, rotation, scale)

        obj_meshes = _load_meshes(obj_asset, transform)
        _add_meshes(scene, obj_meshes, name_prefix=f"obj_{idx}")

    if include_articulated:
        articulated_instances = data.get("articulated_object_instances", [])
        for idx, inst in enumerate(articulated_instances):
            template_name = inst.get("template_name")
            if not template_name:
                continue
            if remove_doors and _is_door_object(template_name, semantics, object_names):
                if log_removed:
                    print(f"[remove] articulated {template_name} name='{object_names.get(template_name, '')}'")
                continue
            if remove_doors and _is_door_name(fpmodels.get(template_name, "")):
                if log_removed:
                    print(f"[remove] articulated {template_name} fpmodels='{fpmodels.get(template_name, '')}'")
                continue
            ao_cfg_path = urdf_index.get(template_name)
            if ao_cfg_path is None:
                print(f"[warn] missing ao_config for {template_name}")
                continue
            ao_cfg = _load_json(ao_cfg_path)
            urdf_path = _resolve_urdf_asset(ao_cfg_path, ao_cfg["urdf_filepath"])
            try:
                robot = URDF.load(str(urdf_path))
            except Exception as exc:  # pragma: no cover
                print(f"[warn] failed to load URDF {urdf_path}: {exc}")
                continue

            joint_cfg = _parse_initial_joint_pose(robot, inst.get("initial_joint_pose"))
            if joint_cfg:
                robot.update_cfg(joint_cfg)

            translation = inst.get("translation")
            rotation = inst.get("rotation")
            scale = inst.get("non_uniform_scale")
            transform = _compose_transform(translation, rotation, scale)
            skip_names = _collect_door_names(robot) if remove_doors else None
            skip_substrings = ["door"] if remove_doors else None
            meshes = _urdf_scene_meshes(robot.scene, transform, skip_names=skip_names, skip_substrings=skip_substrings)
            _add_meshes(scene, meshes, name_prefix=f"art_{idx}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    scenes_dir = dataset_root / args.scenes_dir
    output_dir = Path(args.output_dir)

    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes dir not found: {scenes_dir}")

    object_index = _build_object_index(dataset_root / "objects")
    stage_index = _build_stage_index(dataset_root / "stages")
    urdf_index = _build_urdf_index(dataset_root / "urdf")
    semantics_path = (
        Path(args.semantics_csv) if args.semantics_csv else dataset_root / "metadata/hssd_obj_semantics_condensed.csv"
    )
    objects_path = Path(args.objects_json) if args.objects_json else dataset_root / "metadata/objects.json"
    fpmodels_path = (
        Path(args.fpmodels_csv) if args.fpmodels_csv else dataset_root / "metadata/fpmodels-with-decomposed.csv"
    )
    semantics = _load_semantics_map(semantics_path)
    object_names = _load_object_names(objects_path)
    fpmodels = _load_fpmodels_map(fpmodels_path)

    scene_paths = _scene_instance_paths(scenes_dir, args.scene_id)
    if args.max_scenes > 0:
        scene_paths = scene_paths[: args.max_scenes]

    if not scene_paths:
        raise FileNotFoundError("No scene_instance.json files found.")

    for scene_path in scene_paths:
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        output_path = output_dir / f"{scene_id}.glb"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists")
            continue
        print(f"[convert] {scene_id} -> {output_path}")
        convert_scene(
            scene_path,
            stage_index,
            object_index,
            urdf_index,
            output_path,
            include_articulated=args.include_articulated,
            remove_doors=args.remove_doors,
            semantics=semantics,
            object_names=object_names,
            fpmodels=fpmodels,
            log_removed=args.log_removed,
        )


if __name__ == "__main__":
    main()
