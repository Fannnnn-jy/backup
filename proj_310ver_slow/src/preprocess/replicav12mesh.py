"""
Convert Replica v1 scenes into GLB/PLY/OBJ meshes.

Replica v1 scenes are organized as:

    <dataset_root>/<scene_id>/mesh.ply

Unlike ReplicaCAD, there is no scene_instance.json to resolve. Each scene is a
single mesh with vertex colors, so this script focuses on:

- loading the per-scene mesh
- optionally rotating from z-up to glTF y-up
- optionally normalizing to Y:[0,1], X/Z:[-1,1]
- optionally adding a white ceiling
- exporting a single scene or batch-converting the whole dataset
"""

from __future__ import annotations

import fnmatch
import os.path as osp
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import transforms3d
import trimesh
import tyro


DEFAULT_DATASET_ROOT = Path("/home/ghr/fs/Junyi/data/proj/replica_v1")
DEFAULT_MESH_NAME = "mesh.ply"
DEFAULT_CEILING_Y_OFFSET = -0.01


def load_mesh_as_open3d(file_path: str, apply_transform: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh | None:
    """Load a mesh file and convert it to Open3D while preserving vertex colors when possible."""
    try:
        mesh = trimesh.load(file_path, force="mesh", process=False)
    except Exception as exc:
        print(f"[error] failed to load mesh {file_path}: {exc}")
        return None

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    if apply_transform is not None:
        o3d_mesh.transform(apply_transform)

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.asarray(mesh.visual.vertex_colors)
        if colors.ndim == 2 and colors.shape[0] == len(mesh.vertices):
            if colors.shape[1] >= 3:
                rgb = colors[:, :3].astype(np.float64)
                if np.max(rgb) > 1.0:
                    rgb = rgb / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def load_mesh_as_trimesh(file_path: str, apply_transform: Optional[np.ndarray] = None) -> List[trimesh.Trimesh]:
    """Load a mesh file as one or more trimesh meshes."""
    try:
        asset = trimesh.load(file_path, force="scene", process=False)
    except Exception as exc:
        print(f"[error] failed to load mesh {file_path}: {exc}")
        return []

    meshes: List[trimesh.Trimesh] = []

    if isinstance(asset, trimesh.Trimesh):
        mesh = asset.copy()
        if apply_transform is not None:
            mesh.apply_transform(apply_transform)
        meshes.append(mesh)
        return meshes

    if isinstance(asset, trimesh.Scene):
        for node_name in asset.graph.nodes_geometry:
            node_transform, geom_name = asset.graph[node_name]
            geom = asset.geometry[geom_name]
            mesh = geom.copy()
            if node_transform is not None:
                mesh.apply_transform(node_transform)
            if apply_transform is not None:
                mesh.apply_transform(apply_transform)
            meshes.append(mesh)

    return meshes


def add_meshes_to_trimesh_scene(scene: trimesh.Scene, meshes: List[trimesh.Trimesh], name_prefix: str) -> None:
    """Add meshes to a trimesh.Scene with stable names."""
    for idx, mesh in enumerate(meshes):
        geom_name = f"{name_prefix}_geom_{idx}"
        node_name = f"{name_prefix}_node_{idx}"
        scene.add_geometry(mesh, geom_name=geom_name, node_name=node_name)


def get_glb_y_up_transform() -> np.ndarray:
    """Rotate from z-up to glTF y-up convention."""
    rot = transforms3d.euler.euler2mat(-np.pi / 2, 0, 0, "sxyz")
    transform = np.eye(4)
    transform[:3, :3] = rot
    return transform


def compute_axis_normalize_transform(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    """
    Build an affine transform that maps:
      - y: [min_y, max_y] -> [0, 1]
      - x/z: centered and scaled to [-1, 1]
    """
    eps = 1e-8
    extents = bounds_max - bounds_min

    center_x = 0.5 * (bounds_min[0] + bounds_max[0])
    center_z = 0.5 * (bounds_min[2] + bounds_max[2])
    min_y = bounds_min[1]

    translate = np.eye(4, dtype=np.float64)
    translate[:3, 3] = np.array([-center_x, -min_y, -center_z], dtype=np.float64)

    scale = np.eye(4, dtype=np.float64)
    scale[0, 0] = 2.0 / max(float(extents[0]), eps)
    scale[1, 1] = 1.0 / max(float(extents[1]), eps)
    scale[2, 2] = 2.0 / max(float(extents[2]), eps)

    return scale @ translate


def _get_open3d_bounds(geometries: List[o3d.geometry.TriangleMesh]) -> Optional[np.ndarray]:
    mins = []
    maxs = []
    for geom in geometries:
        if geom is None or len(geom.vertices) == 0:
            continue
        verts = np.asarray(geom.vertices)
        mins.append(verts.min(axis=0))
        maxs.append(verts.max(axis=0))

    if not mins:
        return None

    min_xyz = np.min(np.stack(mins, axis=0), axis=0)
    max_xyz = np.max(np.stack(maxs, axis=0), axis=0)
    return np.stack([min_xyz, max_xyz], axis=0)


def normalize_open3d_geometries_to_target_axes(geometries: List[o3d.geometry.TriangleMesh]) -> None:
    bounds = _get_open3d_bounds(geometries)
    if bounds is None or not np.isfinite(bounds).all():
        print("[normalize] warning: open3d bounds unavailable; skip normalization.")
        return

    min_xyz, max_xyz = bounds
    transform = compute_axis_normalize_transform(min_xyz, max_xyz)
    for geom in geometries:
        if geom is not None and len(geom.vertices) > 0:
            geom.transform(transform)

    extents = max_xyz - min_xyz
    print(f"[normalize] Extents: {np.round(extents, 4)}")
    print("[normalize] Target: Y[0,1], XZ[-1,1]")


def normalize_trimesh_scene_to_target_axes(scene: trimesh.Scene) -> None:
    if scene is None or not scene.geometry:
        print("[normalize] warning: empty trimesh scene; skip normalization.")
        return

    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all():
        print("[normalize] warning: scene bounds unavailable; skip normalization.")
        return

    min_xyz = bounds[0]
    max_xyz = bounds[1]
    transform = compute_axis_normalize_transform(min_xyz, max_xyz)
    scene.apply_transform(transform)

    extents = max_xyz - min_xyz
    print(f"[normalize] Extents: {np.round(extents, 4)}")
    print("[normalize] Target: Y[0,1], XZ[-1,1]")


def add_white_ceiling_open3d(
    geometries: List[o3d.geometry.TriangleMesh],
    thickness: float = 0.02,
    pad_ratio: float = 0.02,
    y_offset: float = DEFAULT_CEILING_Y_OFFSET,
) -> None:
    """Add a thin white ceiling above the current scene bounds."""
    bounds = _get_open3d_bounds(geometries)
    if bounds is None or not np.isfinite(bounds).all():
        print("[ceiling] warning: open3d bounds unavailable; skip ceiling.")
        return

    min_xyz, max_xyz = bounds
    extents = np.maximum(max_xyz - min_xyz, 1e-6)
    pad_x = extents[0] * pad_ratio
    pad_z = extents[2] * pad_ratio

    width = extents[0] + 2.0 * pad_x
    depth = extents[2] + 2.0 * pad_z
    ceiling = o3d.geometry.TriangleMesh.create_box(width=width, height=thickness, depth=depth)
    ceiling.translate([min_xyz[0] - pad_x, max_xyz[1] + y_offset, min_xyz[2] - pad_z])
    ceiling.compute_vertex_normals()
    ceiling_colors = np.ones((len(ceiling.vertices), 3), dtype=np.float64)
    ceiling.vertex_colors = o3d.utility.Vector3dVector(ceiling_colors)
    geometries.append(ceiling)
    print("[ceiling] Added white ceiling (open3d).")


def add_white_ceiling_trimesh(
    scene: trimesh.Scene,
    thickness: float = 0.02,
    pad_ratio: float = 0.02,
    y_offset: float = DEFAULT_CEILING_Y_OFFSET,
) -> None:
    """Add a thin white ceiling above the current scene bounds."""
    if scene is None or not scene.geometry:
        print("[ceiling] warning: empty trimesh scene; skip ceiling.")
        return

    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all():
        print("[ceiling] warning: trimesh bounds unavailable; skip ceiling.")
        return

    min_xyz = bounds[0]
    max_xyz = bounds[1]
    extents = np.maximum(max_xyz - min_xyz, 1e-6)
    pad_x = extents[0] * pad_ratio
    pad_z = extents[2] * pad_ratio

    width = extents[0] + 2.0 * pad_x
    depth = extents[2] + 2.0 * pad_z
    center = np.array(
        [
            0.5 * (min_xyz[0] + max_xyz[0]),
            max_xyz[1] + y_offset + 0.5 * thickness,
            0.5 * (min_xyz[2] + max_xyz[2]),
        ],
        dtype=np.float64,
    )

    ceiling = trimesh.creation.box(extents=[width, thickness, depth])
    ceiling.apply_translation(center)
    white_material = trimesh.visual.material.PBRMaterial(
        name="ceiling_white",
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        emissiveFactor=np.array([0.2, 0.2, 0.2], dtype=np.float32),
    )
    ceiling.visual = trimesh.visual.TextureVisuals(material=white_material)
    scene.add_geometry(ceiling, geom_name="ceiling_white_geom", node_name="ceiling_white_node")
    print("[ceiling] Added white ceiling (trimesh).")


def resolve_scene_dir(dataset_root: Path, scene_id: Optional[str], scene_dir: Optional[str], mesh_name: str) -> Path:
    if scene_dir:
        resolved = Path(scene_dir).expanduser().resolve()
    elif scene_id:
        resolved = (dataset_root / scene_id).resolve()
    else:
        raise ValueError("Either scene_id or scene_dir must be provided for single-scene conversion.")

    mesh_path = resolved / mesh_name
    if not mesh_path.exists():
        raise FileNotFoundError(f"Replica v1 mesh not found: {mesh_path}")
    return resolved


def discover_scene_dirs(dataset_root: Path, mesh_name: str, scene_pattern: str) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    scene_dirs = []
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        if not fnmatch.fnmatch(path.name, scene_pattern):
            continue
        if (path / mesh_name).exists():
            scene_dirs.append(path)

    if not scene_dirs:
        raise FileNotFoundError(
            f"No Replica v1 scenes found in {dataset_root} with mesh '{mesh_name}' and pattern '{scene_pattern}'."
        )
    return scene_dirs


def parse_replica_v1_scene(
    scene_dir: str,
    mesh_name: str = DEFAULT_MESH_NAME,
    export_y_up: bool = True,
    normalize_axes: bool = True,
    add_white_ceiling: bool = True,
) -> List[o3d.geometry.TriangleMesh]:
    """Load a Replica v1 scene as Open3D geometries."""
    scene_path = Path(scene_dir).expanduser().resolve()
    mesh_path = scene_path / mesh_name
    if not mesh_path.exists():
        raise FileNotFoundError(f"Replica v1 mesh not found: {mesh_path}")

    print(f"Loading Replica v1 scene from: {scene_path}")
    transform = get_glb_y_up_transform() if export_y_up else np.eye(4)
    geom = load_mesh_as_open3d(str(mesh_path), transform)
    geometries = [geom] if geom is not None else []

    if normalize_axes:
        normalize_open3d_geometries_to_target_axes(geometries)
    if add_white_ceiling:
        add_white_ceiling_open3d(geometries)

    return geometries


def parse_replica_v1_scene_trimesh(
    scene_dir: str,
    mesh_name: str = DEFAULT_MESH_NAME,
    export_y_up: bool = True,
    normalize_axes: bool = True,
    add_white_ceiling: bool = False,
) -> trimesh.Scene:
    """Load a Replica v1 scene as a trimesh.Scene."""
    scene_path = Path(scene_dir).expanduser().resolve()
    mesh_path = scene_path / mesh_name
    if not mesh_path.exists():
        raise FileNotFoundError(f"Replica v1 mesh not found: {mesh_path}")

    print(f"Loading Replica v1 scene from: {scene_path}")
    transform = get_glb_y_up_transform() if export_y_up else np.eye(4)
    meshes = load_mesh_as_trimesh(str(mesh_path), transform)
    scene = trimesh.Scene()
    add_meshes_to_trimesh_scene(scene, meshes, scene_path.name)

    if normalize_axes:
        normalize_trimesh_scene_to_target_axes(scene)
    if add_white_ceiling:
        add_white_ceiling_trimesh(scene)

    return scene


def create_coordinate_frame(size: float = 0.5) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def visualize_geometry(geometries: List[o3d.geometry.TriangleMesh], show_frame: bool = True) -> None:
    if not geometries:
        print("No geometries found in scene.")
        return

    vis_geometries = geometries.copy()
    if show_frame:
        vis_geometries.append(create_coordinate_frame(size=1.0))

    o3d.visualization.draw_geometries(
        vis_geometries,
        window_name="Replica v1 Scene",
        width=1200,
        height=800,
        left=50,
        top=50,
    )


def save_geometry(path: str, geometry) -> None:
    """Save Open3D geometry to .glb/.ply/.obj."""
    file_ext = osp.splitext(path)[1].lower()
    supported_formats = [".glb", ".ply", ".obj"]
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}")

    geometries = geometry if isinstance(geometry, list) else [geometry]
    geometries = [g for g in geometries if g is not None]
    if not geometries:
        raise ValueError("No valid geometries to save.")

    if len(geometries) == 1:
        combined_mesh = geometries[0]
    else:
        print(f"Combining {len(geometries)} meshes...")
        combined_mesh = o3d.geometry.TriangleMesh()
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for mesh in geometries:
            if len(mesh.vertices) == 0:
                continue
            vertices = np.asarray(mesh.vertices)
            all_vertices.append(vertices)

            faces = np.asarray(mesh.triangles)
            if len(faces) > 0:
                all_faces.append(faces + vertex_offset)

            if mesh.has_vertex_colors():
                all_colors.append(np.asarray(mesh.vertex_colors))

            vertex_offset += len(vertices)

        combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
        if all_faces:
            combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_faces))
        if all_colors:
            combined_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        combined_mesh.compute_vertex_normals()

    print(f"Saving mesh to: {path}")
    if file_ext in (".ply", ".obj"):
        success = o3d.io.write_triangle_mesh(path, combined_mesh)
        if not success:
            raise RuntimeError(f"Failed to save mesh to {path}")
        return

    vertices = np.asarray(combined_mesh.vertices)
    faces = np.asarray(combined_mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if combined_mesh.has_vertex_colors():
        colors = np.asarray(combined_mesh.vertex_colors)
        colors_rgba = np.hstack([np.clip(colors, 0.0, 1.0) * 255.0, np.full((len(colors), 1), 255.0)])
        trimesh_mesh.visual.vertex_colors = colors_rgba.astype(np.uint8)
    trimesh_mesh.export(path)


def save_trimesh_scene_as_glb(path: str, scene: trimesh.Scene) -> None:
    if scene is None or len(scene.geometry) == 0:
        raise ValueError("No geometries to save to GLB.")
    print(f"Saving scene to: {path}")
    scene.export(path)


def _default_output_dir(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}_glb"


def convert_single_scene(
    scene_dir: Path,
    *,
    mesh_name: str,
    save_path: Optional[Path],
    output_dir: Optional[Path],
    output_ext: str,
    show_frame: bool,
    vis: bool,
    export_y_up: bool,
    normalize_axes: bool,
    add_white_ceiling: bool,
    overwrite: bool,
) -> Path | None:
    if save_path is None:
        out_dir = output_dir if output_dir is not None else _default_output_dir(scene_dir.parent)
        save_path = out_dir / f"{scene_dir.name}{output_ext}"

    save_path = save_path.expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not overwrite:
        print(f"[skip] output exists: {save_path}")
        return save_path

    save_ext = save_path.suffix.lower()
    geometries = None
    if vis or save_ext != ".glb":
        geometries = parse_replica_v1_scene(
            str(scene_dir),
            mesh_name=mesh_name,
            export_y_up=export_y_up,
            normalize_axes=normalize_axes,
            add_white_ceiling=add_white_ceiling,
        )

    if save_ext == ".glb":
        scene = parse_replica_v1_scene_trimesh(
            str(scene_dir),
            mesh_name=mesh_name,
            export_y_up=export_y_up,
            normalize_axes=normalize_axes,
            add_white_ceiling=add_white_ceiling,
        )
        save_trimesh_scene_as_glb(str(save_path), scene)
    elif geometries is not None:
        save_geometry(str(save_path), geometries)

    if vis and geometries is not None:
        visualize_geometry(geometries, show_frame=show_frame)

    return save_path


def main(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    scene_id: str | None = None,
    scene_dir: str | None = None,
    scene_pattern: str = "*",
    mesh_name: str = DEFAULT_MESH_NAME,
    show_frame: bool = True,
    vis: bool = False,
    save_path: str | None = None,
    output_dir: str | None = None,
    output_ext: str = ".glb",
    export_y_up: bool = True,
    normalize_axes: bool = True,
    add_white_ceiling: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Convert Replica v1 scenes to meshes.

    Usage patterns:
    - Single scene to a custom path:
        --scene-id apartment_0 --save-path /tmp/apartment_0.glb
    - Single scene to an output directory:
        --scene-id apartment_0 --output-dir /tmp/replica_v1_glb
    - Batch convert all scenes:
        --output-dir /tmp/replica_v1_glb
    """
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve() if output_dir else None
    save_path_obj = Path(save_path).expanduser().resolve() if save_path else None

    if save_path_obj is not None and output_dir_path is not None:
        raise ValueError("Use either save_path or output_dir, not both.")

    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"

    single_scene_mode = scene_id is not None or scene_dir is not None
    if single_scene_mode:
        scene_path = resolve_scene_dir(dataset_root_path, scene_id, scene_dir, mesh_name)
        convert_single_scene(
            scene_path,
            mesh_name=mesh_name,
            save_path=save_path_obj,
            output_dir=output_dir_path,
            output_ext=output_ext,
            show_frame=show_frame,
            vis=vis,
            export_y_up=export_y_up,
            normalize_axes=normalize_axes,
            add_white_ceiling=add_white_ceiling,
            overwrite=overwrite,
        )
        return

    scene_dirs = discover_scene_dirs(dataset_root_path, mesh_name, scene_pattern)
    batch_output_dir = output_dir_path if output_dir_path is not None else _default_output_dir(dataset_root_path)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(scene_dirs)} Replica v1 scenes under: {dataset_root_path}")
    print(f"Output directory: {batch_output_dir}")
    for scene_path in scene_dirs:
        print(f"\n=== Converting {scene_path.name} ===")
        convert_single_scene(
            scene_path,
            mesh_name=mesh_name,
            save_path=None,
            output_dir=batch_output_dir,
            output_ext=output_ext,
            show_frame=show_frame,
            vis=vis,
            export_y_up=export_y_up,
            normalize_axes=normalize_axes,
            add_white_ceiling=add_white_ceiling,
            overwrite=overwrite,
        )

    print("\nReplica v1 conversion complete.")


if __name__ == "__main__":
    tyro.cli(main)
