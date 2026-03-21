"""
Script to parse ReplicaCAD scene_instance.json config files and visualize with Open3D
Based on urdf2open3d.py and ReplicaCAD scene_builder.py
"""

import numpy as np
import json
import os.path as osp
from typing import List, Optional

import open3d as o3d
import transforms3d
import trimesh
import tyro
from yourdfpy import URDF


def load_mesh_as_open3d(file_path: str, apply_transform: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh file and convert to Open3D TriangleMesh

    Args:
        file_path: Path to mesh file (.glb, .obj, etc.)
        apply_transform: Optional 4x4 transformation matrix to apply

    Returns:
        Open3D TriangleMesh
    """
    try:
        # Load with trimesh first for better format support
        mesh = trimesh.load(file_path, force="mesh")

        # Convert to Open3D
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Apply transformation if provided
        if apply_transform is not None:
            o3d_mesh.transform(apply_transform)

        # Compute normals for better visualization
        o3d_mesh.compute_vertex_normals()

        # Add color based on material if available
        if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
            colors = mesh.visual.face_colors
            if len(colors) > 0:
                # Convert face colors to vertex colors (simplified)
                vertex_colors = np.tile(colors[0][:3] / 255.0, (len(o3d_mesh.vertices), 1))
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        else:
            # Default random color
            color = np.random.rand(3)
            vertex_colors = np.tile(color, (len(o3d_mesh.vertices), 1))
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return o3d_mesh

    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def load_mesh_as_trimesh(file_path: str, apply_transform: Optional[np.ndarray] = None) -> List[trimesh.Trimesh]:
    """
    Load a mesh file and return a list of trimesh meshes preserving textures/materials.

    Args:
        file_path: Path to mesh file (.glb, .obj, etc.)
        apply_transform: Optional 4x4 transformation matrix to apply

    Returns:
        List of trimesh.Trimesh meshes (may be multiple for a scene)
    """
    try:
        asset = trimesh.load(file_path, force="scene", process=False)
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
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


def add_meshes_to_trimesh_scene(scene: trimesh.Scene, meshes: List[trimesh.Trimesh], name_prefix: str):
    """Add a list of meshes to a trimesh.Scene with unique names."""
    for idx, mesh in enumerate(meshes):
        geom_name = f"{name_prefix}_geom_{idx}"
        node_name = f"{name_prefix}_node_{idx}"
        scene.add_geometry(mesh, geom_name=geom_name, node_name=node_name)


def urdf_to_open3d_geometries(
    urdf_model: URDF, transform: Optional[np.ndarray] = None
) -> List[o3d.geometry.TriangleMesh]:
    """
    Convert URDF model to Open3D geometries

    Args:
        urdf_model: URDF model from yourdfpy
        transform: Optional 4x4 transformation matrix to apply to all geometries

    Returns:
        List of Open3D geometries
    """
    geometries = []

    # Get scene with current configuration
    scene = urdf_model.scene

    # Extract geometries from the scene
    for node_name in scene.graph.nodes_geometry:
        # Get the transform for this node
        node_transform, geometry_name = scene.graph[node_name]

        # Get the actual geometry
        geometry = scene.geometry[geometry_name]

        # Convert trimesh geometry to Open3D
        if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
            # Create Open3D triangle mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(geometry.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(geometry.faces)

            # Apply node transform
            mesh.transform(node_transform)

            # Apply additional transform if provided
            if transform is not None:
                mesh.transform(transform)

            # Compute normals for better visualization
            mesh.compute_vertex_normals()

            # Add colors
            if hasattr(geometry.visual, "face_colors"):
                colors = geometry.visual.face_colors
                if colors is not None and len(colors) > 0:
                    # Convert face colors to vertex colors (simplified)
                    vertex_colors = np.tile(colors[0][:3] / 255.0, (len(mesh.vertices), 1))
                    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            else:
                # Default color scheme - different colors for different parts
                color = np.random.rand(3)
                vertex_colors = np.tile(color, (len(mesh.vertices), 1))
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            geometries.append(mesh)

    return geometries


def get_replica_cad_transform():
    """
    Get the transformation matrix for ReplicaCAD coordinate system conversion
    ReplicaCAD uses a different xyz convention, rotated by 90 degrees around X axis
    """
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    transform_matrix = transforms3d.quaternions.quat2mat(q)
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = transform_matrix
    return transform_4x4


def get_glb_y_up_transform():
    """
    Get the transformation matrix to convert from z-up to y-up (glTF convention).
    """
    R = transforms3d.euler.euler2mat(-np.pi / 2, 0, 0, "sxyz")
    T = np.eye(4)
    T[:3, :3] = R
    return T


def parse_replica_cad_scene(scene_config_path: str, asset_dir: str) -> List[o3d.geometry.TriangleMesh]:
    """
    Parse ReplicaCAD scene configuration and return Open3D geometries

    Args:
        scene_config_path: Path to scene_instance.json file
        asset_dir: Path to ReplicaCAD dataset assets

    Returns:
        List of Open3D geometries representing the complete scene
    """
    print(f"Loading scene configuration from: {scene_config_path}")

    with open(scene_config_path) as f:
        config = json.load(f)

    geometries = []

    # Get ReplicaCAD coordinate transformation
    replica_transform = get_replica_cad_transform()

    # Load background/stage
    print("Loading background/stage...")
    background_template_name = osp.basename(config["stage_instance"]["template_name"])
    bg_path = osp.join(asset_dir, f"stages/{background_template_name}.glb")

    if osp.exists(bg_path):
        bg_mesh = load_mesh_as_open3d(bg_path, replica_transform)
        if bg_mesh is not None:
            geometries.append(bg_mesh)
            print(f"  Loaded background: {background_template_name}")
    else:
        print(f"  Warning: Background file not found: {bg_path}")

    # Load object instances
    print("Loading object instances...")
    for obj_num, obj_meta in enumerate(config["object_instances"]):
        template_name = osp.basename(obj_meta["template_name"])

        # Load object configuration
        obj_config_path = osp.join(asset_dir, f"configs/objects/{template_name}.object_config.json")
        if not osp.exists(obj_config_path):
            print(f"  Warning: Object config not found: {obj_config_path}")
            continue

        with open(obj_config_path) as f:
            obj_config = json.load(f)

        # Get visual file path
        visual_file = osp.join(osp.dirname(obj_config_path), obj_config["render_asset"])
        if not osp.exists(visual_file):
            print(f"  Warning: Visual file not found: {visual_file}")
            continue

        # Create transformation matrix for this object
        pos = np.array(obj_meta["translation"])
        rot = np.array(obj_meta["rotation"])  # quaternion [w, x, y, z]

        # Create pose transformation
        obj_transform = np.eye(4)
        obj_transform[:3, :3] = transforms3d.quaternions.quat2mat(rot)
        obj_transform[:3, 3] = pos

        # Apply scale if specified (check both object meta and object config)
        scale = 1.0
        if "uniform_scale" in obj_meta:
            scale = obj_meta["uniform_scale"]
            print(f"    Applied instance scale: {scale}")
        elif "scale" in obj_config:
            scale = obj_config["scale"]
            print(f"    Applied config scale: {scale}")
        elif "uniform_scale" in obj_config:
            scale = obj_config["uniform_scale"]
            print(f"    Applied config uniform_scale: {scale}")

        if scale != 1.0:
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] *= scale
            obj_transform = obj_transform @ scale_matrix

        # Combine with ReplicaCAD coordinate transformation
        final_transform = replica_transform @ obj_transform

        # Load and transform mesh
        obj_mesh = load_mesh_as_open3d(visual_file, final_transform)
        if obj_mesh is not None:
            geometries.append(obj_mesh)
            print(f"  Loaded object {obj_num}: {template_name} ({obj_meta['motion_type']})")

    # Load articulated objects (URDFs)
    print("Loading articulated objects...")
    for i, articulated_meta in enumerate(config["articulated_object_instances"]):
        template_name = articulated_meta["template_name"]

        # Skip doors for now (often problematic)
        # if "door" in template_name:
        #     print(f"  Skipping door: {template_name}")
        #     continue

        urdf_path = osp.join(asset_dir, f"urdf/{template_name}/{template_name}.urdf")
        if not osp.exists(urdf_path):
            print(f"  Warning: URDF not found: {urdf_path}")
            continue

        try:
            # Load URDF
            robot = URDF.load(urdf_path)

            # Create transformation matrix for this articulation
            pos = np.array(articulated_meta["translation"])
            rot = np.array(articulated_meta["rotation"])  # quaternion [w, x, y, z]

            # Create pose transformation
            articulation_transform = np.eye(4)
            articulation_transform[:3, :3] = transforms3d.quaternions.quat2mat(rot)
            articulation_transform[:3, 3] = pos

            # Apply scale if specified
            if "uniform_scale" in articulated_meta:
                scale = articulated_meta["uniform_scale"]
                scale_matrix = np.eye(4)
                scale_matrix[:3, :3] *= scale
                articulation_transform = articulation_transform @ scale_matrix
                print(f"    Applied scale: {scale}")

            # Combine with ReplicaCAD coordinate transformation
            final_transform = replica_transform @ articulation_transform

            # Convert URDF to Open3D geometries
            urdf_geometries = urdf_to_open3d_geometries(robot, final_transform)
            geometries.extend(urdf_geometries)

            print(f"  Loaded articulation {i}: {template_name} ({len(urdf_geometries)} parts)")

        except Exception as e:
            print(f"  Error loading URDF {template_name}: {e}")

    return geometries


def parse_replica_cad_scene_trimesh(
    scene_config_path: str,
    asset_dir: str,
    export_y_up: bool = True,
    include_articulated: bool = True,
    skip_doors: bool = True,
) -> trimesh.Scene:
    """
    Parse ReplicaCAD scene configuration and return a trimesh.Scene with textures/materials preserved.

    Args:
        scene_config_path: Path to scene_instance.json file
        asset_dir: Path to ReplicaCAD dataset assets
        export_y_up: If True, rotate to glTF's y-up convention
        include_articulated: Whether to include articulated objects (URDFs)
        skip_doors: Whether to skip articulated objects with "door" in template_name
    """
    print(f"Loading scene configuration from: {scene_config_path}")

    with open(scene_config_path) as f:
        config = json.load(f)

    scene = trimesh.Scene()

    replica_transform = get_replica_cad_transform()
    export_transform = get_glb_y_up_transform() if export_y_up else np.eye(4)
    global_transform = export_transform @ replica_transform

    # Load background/stage
    print("Loading background/stage (trimesh)...")
    background_template_name = osp.basename(config["stage_instance"]["template_name"])
    bg_path = osp.join(asset_dir, f"stages/{background_template_name}.glb")
    if osp.exists(bg_path):
        bg_meshes = load_mesh_as_trimesh(bg_path, global_transform)
        if bg_meshes:
            add_meshes_to_trimesh_scene(scene, bg_meshes, f"bg_{background_template_name}")
            print(f"  Loaded background: {background_template_name}")
    else:
        print(f"  Warning: Background file not found: {bg_path}")

    # Load object instances
    print("Loading object instances (trimesh)...")
    for obj_num, obj_meta in enumerate(config["object_instances"]):
        template_name = osp.basename(obj_meta["template_name"])
        obj_config_path = osp.join(asset_dir, f"configs/objects/{template_name}.object_config.json")
        if not osp.exists(obj_config_path):
            print(f"  Warning: Object config not found: {obj_config_path}")
            continue

        with open(obj_config_path) as f:
            obj_config = json.load(f)

        visual_file = osp.join(osp.dirname(obj_config_path), obj_config["render_asset"])
        if not osp.exists(visual_file):
            print(f"  Warning: Visual file not found: {visual_file}")
            continue

        pos = np.array(obj_meta["translation"])
        rot = np.array(obj_meta["rotation"])  # quaternion [w, x, y, z]

        obj_transform = np.eye(4)
        obj_transform[:3, :3] = transforms3d.quaternions.quat2mat(rot)
        obj_transform[:3, 3] = pos

        scale = 1.0
        if "uniform_scale" in obj_meta:
            scale = obj_meta["uniform_scale"]
            print(f"    Applied instance scale: {scale}")
        elif "scale" in obj_config:
            scale = obj_config["scale"]
            print(f"    Applied config scale: {scale}")
        elif "uniform_scale" in obj_config:
            scale = obj_config["uniform_scale"]
            print(f"    Applied config uniform_scale: {scale}")

        if scale != 1.0:
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] *= scale
            obj_transform = obj_transform @ scale_matrix

        final_transform = global_transform @ obj_transform
        obj_meshes = load_mesh_as_trimesh(visual_file, final_transform)
        if obj_meshes:
            add_meshes_to_trimesh_scene(scene, obj_meshes, f"obj_{obj_num}_{template_name}")
            print(f"  Loaded object {obj_num}: {template_name} ({obj_meta['motion_type']})")

    # Load articulated objects (URDFs)
    if include_articulated:
        print("Loading articulated objects (trimesh)...")
        for i, articulated_meta in enumerate(config["articulated_object_instances"]):
            template_name = articulated_meta["template_name"]
            if skip_doors and "door" in template_name:
                continue

            urdf_path = osp.join(asset_dir, f"urdf/{template_name}/{template_name}.urdf")
            if not osp.exists(urdf_path):
                print(f"  Warning: URDF not found: {urdf_path}")
                continue

            try:
                robot = URDF.load(urdf_path)

                pos = np.array(articulated_meta["translation"])
                rot = np.array(articulated_meta["rotation"])

                articulation_transform = np.eye(4)
                articulation_transform[:3, :3] = transforms3d.quaternions.quat2mat(rot)
                articulation_transform[:3, 3] = pos

                if "uniform_scale" in articulated_meta:
                    scale = articulated_meta["uniform_scale"]
                    scale_matrix = np.eye(4)
                    scale_matrix[:3, :3] *= scale
                    articulation_transform = articulation_transform @ scale_matrix
                    print(f"    Applied scale: {scale}")

                final_transform = global_transform @ articulation_transform
                urdf_meshes: List[trimesh.Trimesh] = []

                if hasattr(robot, "scene") and robot.scene is not None:
                    urdf_scene = robot.scene
                    if isinstance(urdf_scene, trimesh.Scene):
                        for node_name in urdf_scene.graph.nodes_geometry:
                            node_transform, geom_name = urdf_scene.graph[node_name]
                            geom = urdf_scene.geometry[geom_name]
                            mesh = geom.copy()
                            if node_transform is not None:
                                mesh.apply_transform(node_transform)
                            mesh.apply_transform(final_transform)
                            urdf_meshes.append(mesh)
                    elif isinstance(urdf_scene, trimesh.Trimesh):
                        mesh = urdf_scene.copy()
                        mesh.apply_transform(final_transform)
                        urdf_meshes.append(mesh)

                if urdf_meshes:
                    add_meshes_to_trimesh_scene(scene, urdf_meshes, f"art_{i}_{template_name}")
                    print(f"  Loaded articulation {i}: {template_name} ({len(urdf_meshes)} parts)")
            except Exception as e:
                print(f"  Error loading URDF {template_name}: {e}")

    return scene


def create_coordinate_frame(size=0.5):
    """Create a coordinate frame for reference"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def visualize_geometry(geometries: list[o3d.geometry.TriangleMesh], show_frame: bool = True):
    """
    Main function to visualize ReplicaCAD scene

    Args:
        scene_config_path: Path to scene_instance.json file
        asset_dir: Path to ReplicaCAD dataset assets
        show_frame: Whether to show coordinate frame
    """

    if not geometries:
        print("No geometries found in scene!")
        return

    print(f"\nFound {len(geometries)} geometries total")

    # Prepare visualization
    vis_geometries = geometries.copy()

    # Add coordinate frame if requested
    if show_frame:
        coord_frame = create_coordinate_frame(size=1.0)
        vis_geometries.append(coord_frame)

    # Visualize
    print("Starting visualization...")
    o3d.visualization.draw_geometries(
        vis_geometries,
        window_name="ReplicaCAD Scene",
        width=1200,
        height=800,
        left=50,
        top=50,
    )


def save_geometry(path: str, geometry):
    """
    Save Open3D geometry to file (supports .glb, .ply, .obj formats)

    Args:
        path: Output file path with extension (.glb, .ply, .obj)
        geometry: Either a single o3d.geometry.TriangleMesh or list of meshes
    """
    # Get file extension to determine format
    file_ext = osp.splitext(path)[1].lower()
    supported_formats = [".glb", ".ply", ".obj"]

    if file_ext not in supported_formats:
        print(f"Unsupported file format: {file_ext}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return

    # Handle both single mesh and list of meshes
    if isinstance(geometry, list):
        geometries = geometry
    else:
        geometries = [geometry]

    # Filter out None geometries
    geometries = [g for g in geometries if g is not None]

    if not geometries:
        print("No valid geometries to save")
        return

    # If multiple geometries, combine them into a single mesh
    if len(geometries) == 1:
        combined_mesh = geometries[0]
    else:
        print(f"Combining {len(geometries)} meshes...")
        combined_mesh = o3d.geometry.TriangleMesh()

        # Combine all vertices and faces
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for mesh in geometries:
            if len(mesh.vertices) == 0:
                continue

            # Add vertices
            vertices = np.asarray(mesh.vertices)
            all_vertices.append(vertices)

            # Add faces with vertex offset
            faces = np.asarray(mesh.triangles)
            if len(faces) > 0:
                faces_offset = faces + vertex_offset
                all_faces.append(faces_offset)

            # Add colors if available
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                all_colors.append(colors)
            else:
                # Generate random color for this mesh
                color = np.random.rand(3)
                colors = np.tile(color, (len(vertices), 1))
                all_colors.append(colors)

            vertex_offset += len(vertices)

        if all_vertices:
            # Combine all data
            combined_vertices = np.vstack(all_vertices)
            combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)

            if all_faces:
                combined_faces = np.vstack(all_faces)
                combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)

            if all_colors:
                combined_colors = np.vstack(all_colors)
                combined_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)

            # Compute normals
            combined_mesh.compute_vertex_normals()

    print(f"Saving combined mesh to: {path}")

    # Save based on file format
    if file_ext in [".ply", ".obj"]:
        # Use Open3D for PLY or OBJ
        success = o3d.io.write_triangle_mesh(path, combined_mesh)
        if success:
            print(f"Successfully saved {len(geometries)} geometries to PLY: {path}")
        else:
            print(f"Failed to save PLY file: {path}")

    elif file_ext == ".glb":
        # Use trimesh for GLB (Open3D doesn't support GLB export)
        vertices = np.asarray(combined_mesh.vertices)
        faces = np.asarray(combined_mesh.triangles)

        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # NOTE: Apply inverse transform to make y+ up axis, which is required by glTF standard
        # Rotate 90 degrees around x-axis to convert from y-up to z-up
        R = transforms3d.euler.euler2mat(-np.pi / 2, 0, 0, "sxyz")
        T = np.eye(4)
        T[:3, :3] = R
        trimesh_mesh.apply_transform(T)

        # Add vertex colors if available
        if combined_mesh.has_vertex_colors():
            colors = np.asarray(combined_mesh.vertex_colors)
            # Convert to 0-255 range and add alpha channel
            colors_rgba = np.hstack([colors * 255, np.full((len(colors), 1), 255)])
            trimesh_mesh.visual.vertex_colors = colors_rgba.astype(np.uint8)

        trimesh_mesh.export(path)
        print(f"Successfully saved {len(geometries)} geometries to GLB: {path}")


# Keep the old function name for backward compatibility
def save_geometry_as_glb(path: str, geometry):
    """Deprecated: Use save_geometry() instead"""
    save_geometry(path, geometry)


def save_trimesh_scene_as_glb(path: str, scene: trimesh.Scene):
    """Save a trimesh.Scene to GLB, preserving textures/materials when possible."""
    if scene is None or len(scene.geometry) == 0:
        print("No geometries to save to GLB")
        return
    print(f"Saving textured scene to: {path}")
    scene.export(path)
    print(f"Successfully saved scene to GLB: {path}")


def main(
    scene_cfg: str,
    asset_dir: str = None,
    show_frame: bool = True,
    vis: bool = False,
    save_path: str = None,
    export_textures: bool = True,
    export_y_up: bool = True,
):
    # Check if scene config file exists
    if not osp.exists(scene_cfg):
        print(f"Error: Scene config file not found: {scene_cfg}")
        return

    # Determine asset directory
    if asset_dir:
        asset_dir = asset_dir
    else:
        # Try to infer from ManiSkill default location
        maniskill_data_dir = osp.expanduser("~/.maniskill/data")
        asset_dir = osp.join(maniskill_data_dir, "scene_datasets/replica_cad_dataset")

        if not osp.exists(asset_dir):
            print(f"Error: Asset directory not found: {asset_dir}")
            print("Please specify asset directory with --asset-dir or ensure ManiSkill data is installed")
            return

    print(f"Using asset directory: {asset_dir}")

    geometries = None
    save_ext = osp.splitext(save_path)[1].lower() if save_path else ""

    if vis or (save_path and not (export_textures and save_ext == ".glb")):
        # Open3D path for visualization or non-GLB exports
        geometries = parse_replica_cad_scene(scene_cfg, asset_dir)

    if save_path:
        if export_textures and save_ext == ".glb":
            textured_scene = parse_replica_cad_scene_trimesh(
                scene_cfg,
                asset_dir,
                export_y_up=export_y_up,
            )
            save_trimesh_scene_as_glb(save_path, textured_scene)
        else:
            save_geometry(save_path, geometries)

    if vis:
        visualize_geometry(geometries, show_frame)


if __name__ == "__main__":
    tyro.cli(main)
