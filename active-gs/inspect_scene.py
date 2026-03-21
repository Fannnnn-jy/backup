"""
Inspect a GLB/mesh scene: print bounding box in the planner (Z-up) world frame
and suggest a valid init_pose for the planner config.

Usage:
    python inspect_scene.py scene=glb/custom
"""

import hydra
import numpy as np
import trimesh
import os


def _resolve_path(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _axis_alignment_matrix(from_axis, to_axis):
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    src = axes[from_axis.lower()]
    dst = axes[to_axis.lower()]
    if np.allclose(src, dst):
        R = np.eye(3)
    elif np.allclose(src, -dst):
        orth = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(orth, src)), 1.0):
            orth = np.array([0.0, 1.0, 0.0])
        v = np.cross(src, orth)
        v = v / np.linalg.norm(v)
        R = -np.eye(3) + 2.0 * np.outer(v, v)
    else:
        v = np.cross(src, dst)
        s = np.linalg.norm(v)
        c = float(np.dot(src, dst))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    T = np.eye(4)
    T[:3, :3] = R
    return T


@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(cfg):
    scene_cfg = cfg.scene
    scene_id = _resolve_path(scene_cfg.scene_id)
    mesh_path = _resolve_path(scene_cfg.get("mesh_path", None)) or scene_id
    scene_up_axis = str(scene_cfg.get("scene_up_axis", "z"))
    planner_up_axis = str(scene_cfg.get("planner_up_axis", scene_up_axis))

    print(f"\nLoading mesh: {mesh_path}")
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry.get(geometry_name, None)
            if isinstance(geometry, trimesh.Trimesh):
                geometry = geometry.copy()
                geometry.apply_transform(transform)
                geometries.append(geometry)
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    # Apply scene → planner world transform
    scene_to_world = _axis_alignment_matrix(scene_up_axis, planner_up_axis)
    mesh.apply_transform(scene_to_world)

    bbox = np.array(mesh.bounding_box.bounds)
    extents = bbox[1] - bbox[0]
    center = (bbox[0] + bbox[1]) / 2.0

    print(f"\n{'='*60}")
    print(f"  scene_up_axis   : {scene_up_axis}")
    print(f"  planner_up_axis : {planner_up_axis}  (Z is vertical in planner)")
    print(f"  bbox min [X,Y,Z]: {bbox[0].tolist()}")
    print(f"  bbox max [X,Y,Z]: {bbox[1].tolist()}")
    print(f"  extents [X,Y,Z] : {extents.tolist()}")
    print(f"  center  [X,Y,Z] : {center.tolist()}")

    # Suggest a starting position: scene center XY, lower-third Z height
    px = float(center[0])
    py = float(center[1])
    pz = float(bbox[0][2] + extents[2] * 0.3)  # 30% height from floor

    # Camera looking toward +X axis, camera "down" toward -Z (Z-up world, OpenCV convention)
    # c2w columns: [cam_X_in_world | cam_Y_in_world | cam_Z_in_world | position]
    # Looking along +X: cam_Z=[1,0,0], cam_Y=[0,0,-1] (down), cam_X=[0,-1,0] (right)
    # init_pose as 4x4 row-major for yaml:
    # [[0,-1,0,px],[-1,0,0,py],[0,0,-1,pz],[0,0,0,1]]  (looking -Y, right = -X, down = -Z)
    # Simpler: camera looking toward -Y axis (into scene), right = +X, down = -Z
    #   cam_X=[1,0,0], cam_Y=[0,0,-1], cam_Z=[0,-1,0]
    init_pose = [
        [1, 0,  0, px],
        [0, 0, -1, py],
        [0, -1, 0, pz],
        [0, 0,  0, 1],
    ]
    print(f"\n  Suggested init_pose (OpenCV c2w, Z-up world):")
    print(f"    Camera at ({px:.3f}, {py:.3f}, {pz:.3f}), looking along -Y")
    print(f"    Paste into config/planner/confidence.yaml:")
    print(f"    init_pose: {init_pose}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
