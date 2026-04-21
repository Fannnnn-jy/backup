#!/usr/bin/env python3
"""
Convert an iGibson BEHAVIOR HDF5 demo into a textured .glb scene (Y-up).

Usage:
    python igibson_hdf5_to_glb.py \
        --hdf5 /path/to/bottling_fruit_example.hdf5 \
        --dataset /path/to/ig_dataset \
        --key /path/to/igibson.key \
        --frame 0 \
        --output scene.glb
"""

import argparse, io, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import trimesh
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# AES decryption (matches iGibson's C++ tinyobjloader decrypt)
# ---------------------------------------------------------------------------
def load_key(key_path: str):
    with open(key_path) as f:
        lines = f.read().strip().split("\n")
    return bytes.fromhex(lines[0]), bytes.fromhex(lines[1])


def decrypt_bytes(data: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(data), AES.block_size)


def decrypt_file(path: str, key: bytes, iv: bytes) -> bytes:
    with open(path, "rb") as f:
        return decrypt_bytes(f.read(), key, iv)


# ---------------------------------------------------------------------------
# Mesh loading helpers
# ---------------------------------------------------------------------------
def load_obj_str(obj_text: str, mtl_override: dict | None = None) -> trimesh.Trimesh:
    """Load an OBJ from a decoded string, returning a single Trimesh."""
    f = trimesh.util.wrap_as_stream(obj_text)
    mesh = trimesh.load(f, file_type="obj", process=True, force="mesh")
    return mesh


def load_encrypted_obj(path: str, key: bytes, iv: bytes) -> trimesh.Trimesh:
    obj_bytes = decrypt_file(path, key, iv)
    obj_text = obj_bytes.decode("utf-8", errors="replace")
    # Strip mtllib line to avoid file-not-found errors
    lines = []
    for line in obj_text.split("\n"):
        if line.strip().startswith("mtllib"):
            continue
        lines.append(line)
    obj_clean = "\n".join(lines)
    f = trimesh.util.wrap_as_stream(obj_clean)
    mesh = trimesh.load(f, file_type="obj", process=True, force="mesh")
    return mesh


def load_plain_obj(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, file_type="obj", process=True, force="mesh")


def load_texture_image(material_dir: str, key: bytes, iv: bytes) -> Image.Image | None:
    """Load DIFFUSE texture (encrypted or plain) from a material directory."""
    enc_path = os.path.join(material_dir, "DIFFUSE.encrypted.png")
    plain_path = os.path.join(material_dir, "DIFFUSE.png")
    combined_path = os.path.join(material_dir, "COMBINED.png")

    if os.path.exists(enc_path):
        png_bytes = decrypt_file(enc_path, key, iv)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    elif os.path.exists(plain_path):
        return Image.open(plain_path).convert("RGB")
    elif os.path.exists(combined_path):
        return Image.open(combined_path).convert("RGB")
    return None


def apply_texture_to_mesh(mesh: trimesh.Trimesh, image: Image.Image) -> trimesh.Trimesh:
    """Apply a texture image to a mesh using its existing UV coordinates."""
    if mesh.visual and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv = mesh.visual.uv
    else:
        uv = np.zeros((len(mesh.vertices), 2))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


# ---------------------------------------------------------------------------
# Coordinate transform: iGibson is Z-up → we want Y-up
# ---------------------------------------------------------------------------
# Rotation matrix: Z-up to Y-up  (rotate -90° around X)
_Z_UP_TO_Y_UP = np.array([
    [1,  0,  0, 0],
    [0,  0, -1, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1],
], dtype=float)


def euler_rpy_to_matrix(rpy, xyz):
    """Build a 4x4 transform from roll-pitch-yaw (XYZ extrinsic) + translation."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    R = Rotation.from_euler("xyz", [r, p, y]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return T


def quat_pos_to_matrix(orientation, position):
    """Build a 4x4 transform from quaternion (xyzw) + position."""
    R = Rotation.from_quat(orientation).as_matrix()  # scipy uses xyzw
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


# ---------------------------------------------------------------------------
# Object mesh resolver
# ---------------------------------------------------------------------------
def find_object_visual_meshes(dataset_root, category, model):
    """Find visual mesh files for an object given its category and model id."""
    obj_dir = os.path.join(dataset_root, "objects", category, model)
    if not os.path.isdir(obj_dir):
        return [], None

    visual_dir = os.path.join(obj_dir, "shape", "visual")
    material_dir = os.path.join(obj_dir, "material")

    meshes = []
    if os.path.isdir(visual_dir):
        for fn in sorted(os.listdir(visual_dir)):
            full = os.path.join(visual_dir, fn)
            if fn.endswith(".encrypted.obj"):
                meshes.append(("encrypted", full))
            elif fn.endswith(".obj"):
                meshes.append(("plain", full))

    mat_dir = material_dir if os.path.isdir(material_dir) else None
    return meshes, mat_dir


def find_scene_visual_mesh(dataset_root, scene_id, part_name):
    """Find scene-level visual mesh (walls, floors, ceilings)."""
    visual_dir = os.path.join(dataset_root, "scenes", scene_id, "shape", "visual")
    material_base = os.path.join(dataset_root, "scenes", scene_id, "material")

    obj_path = os.path.join(visual_dir, f"{part_name}_vm.obj")
    if not os.path.exists(obj_path):
        return None, None

    mat_dir = os.path.join(material_base, part_name)
    if not os.path.isdir(mat_dir):
        mat_dir = None
    return obj_path, mat_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert iGibson HDF5 demo to GLB")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 demo file")
    parser.add_argument("--dataset", required=True, help="Path to ig_dataset root")
    parser.add_argument("--key", required=True, help="Path to igibson.key")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to export (default: 0)")
    parser.add_argument("--output", default="scene.glb", help="Output GLB path")
    parser.add_argument("--no-scene", action="store_true", help="Skip scene structure (walls/floors/ceilings)")
    parser.add_argument("--no-furniture", action="store_true", help="Skip non-task furniture")
    args = parser.parse_args()

    key, iv = load_key(args.key)
    scene_parts = []  # list of trimesh.Trimesh to merge

    # --- Read HDF5 metadata ---
    with h5py.File(args.hdf5, "r") as f:
        metadata = dict(f.attrs)
        scene_id = metadata["/metadata/scene_id"]
        urdf_name = metadata["/metadata/urdf_file"]
        obj_body_map_str = metadata["/metadata/obj_body_id_to_name"]
        n_frames = f["frame_data"].shape[0]

        if args.frame >= n_frames:
            print(f"Error: frame {args.frame} out of range (max {n_frames - 1})")
            sys.exit(1)

        # Parse body_id -> semantic name mapping
        body_id_to_name = {}
        for line in obj_body_map_str.strip().split("\n"):
            if ":" in line:
                bid, bname = line.split(":", 1)
                body_id_to_name[bid.strip()] = bname.strip()

        # Read physics data for the requested frame
        physics = {}
        for bid in f["physics_data"].keys():
            pos = f["physics_data"][bid]["position"][args.frame]
            ori = f["physics_data"][bid]["orientation"][args.frame]
            js = f["physics_data"][bid]["joint_state"][args.frame] if f["physics_data"][bid]["joint_state"].shape[1] > 0 else np.array([])
            physics[bid] = {"position": pos, "orientation": ori, "joint_state": js}

    print(f"Scene: {scene_id}, URDF: {urdf_name}, Frame: {args.frame}/{n_frames}")
    print(f"Objects tracked: {list(body_id_to_name.values())}")

    # --- Parse URDF ---
    urdf_path = os.path.join(args.dataset, "scenes", scene_id, "urdf", urdf_name + ".urdf")
    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        sys.exit(1)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build link info
    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name", "")
        if name == "world":
            continue
        cat = link_el.get("category", "")
        model = link_el.get("model", "")
        scope = link_el.get("object_scope", "")
        xyz_str = link_el.get("xyz", "0 0 0")
        rpy_str = link_el.get("rpy", "0 0 0")
        xyz = [float(x) for x in xyz_str.split()]
        rpy = [float(x) for x in rpy_str.split()]
        links[name] = {
            "category": cat,
            "model": model,
            "scope": scope,
            "xyz": xyz,
            "rpy": rpy,
        }

    # Map scope names to body IDs for dynamic pose override
    scope_to_bid = {}
    for bid, bname in body_id_to_name.items():
        scope_to_bid[bname] = bid

    # --- Load scene structure ---
    if not args.no_scene:
        for part in ["wall", "floor_0", "floor_1", "ceiling"]:
            obj_path, mat_dir = find_scene_visual_mesh(args.dataset, scene_id, part)
            if obj_path is None:
                continue
            print(f"  Loading scene part: {part}")
            try:
                mesh = load_plain_obj(obj_path)
                if mat_dir:
                    img = load_texture_image(mat_dir, key, iv)
                    if img:
                        mesh = apply_texture_to_mesh(mesh, img)
                scene_parts.append(mesh)
            except Exception as e:
                print(f"    Warning: failed to load {part}: {e}")

    # --- Load objects ---
    # Category name mapping (URDF category -> filesystem directory)
    CATEGORY_DIR_MAP = {
        "fridge": "fridge",
        "bottom_cabinet": "bottom_cabinet",
        "bottom_cabinet_no_top": "bottom_cabinet_no_top",
        "top_cabinet": "top_cabinet",
        "countertop": "countertop",
        "strawberry": "strawberry",
        "peach": "peach",
        "jar": "jar",
        "carving_knife": "carving_knife",
        "pot_plant": "pot_plant",
        "breakfast_table": "breakfast_table",
        "straight_chair": "straight_chair",
        "floor_lamp": "floor_lamp",
        "sofa": "sofa",
        "coffee_table": "coffee_table",
        "bed": "bed",
        "chest_of_drawers": "chest_of_drawers",
        "desk": "desk",
        "swivel_chair": "swivel_chair",
        "standing_tv": "standing_tv",
        "washer": "washer",
        "dryer": "dryer",
        "toilet": "toilet",
        "sink": "sink",
        "bathtub": "bathtub",
        "shower": "shower",
        "mirror": "mirror",
        "oven": "oven",
        "stove": "stove",
        "dishwasher": "dishwasher",
        "microwave": "microwave",
    }

    # Task-relevant object scopes (from HDF5)
    task_scopes = set(body_id_to_name.values())

    # Skip categories that are just grouping wrappers
    SKIP_CATEGORIES = {"multiplexer", "grouper", "agent"}

    loaded_count = 0
    skipped_count = 0

    for link_name, info in links.items():
        cat = info["category"]
        model = info["model"]
        scope = info["scope"]

        if cat in SKIP_CATEGORIES or not model:
            continue

        # Scene structure parts are handled separately
        if cat in ("walls", "floors", "ceilings"):
            continue

        # Skip non-task furniture if requested
        is_task_obj = scope in task_scopes
        if args.no_furniture and not is_task_obj:
            skipped_count += 1
            continue

        # Resolve category directory
        cat_dir = CATEGORY_DIR_MAP.get(cat, cat)
        mesh_infos, mat_dir = find_object_visual_meshes(args.dataset, cat_dir, model)

        if not mesh_infos:
            print(f"  Skip {link_name} ({cat}/{model}): no mesh found")
            skipped_count += 1
            continue

        # Determine pose
        if is_task_obj and scope in scope_to_bid:
            bid = scope_to_bid[scope]
            if bid in physics:
                pos = physics[bid]["position"]
                ori = physics[bid]["orientation"]
                T_obj = quat_pos_to_matrix(ori, pos)
            else:
                T_obj = euler_rpy_to_matrix(info["rpy"], info["xyz"])
        else:
            T_obj = euler_rpy_to_matrix(info["rpy"], info["xyz"])

        # Load and transform meshes
        obj_meshes = []
        for mtype, mpath in mesh_infos:
            try:
                if mtype == "encrypted":
                    m = load_encrypted_obj(mpath, key, iv)
                else:
                    m = load_plain_obj(mpath)
                obj_meshes.append(m)
            except Exception as e:
                print(f"    Warning: failed to load {mpath}: {e}")

        if not obj_meshes:
            skipped_count += 1
            continue

        # Apply texture
        texture_img = None
        if mat_dir:
            texture_img = load_texture_image(mat_dir, key, iv)

        for m in obj_meshes:
            if texture_img:
                m = apply_texture_to_mesh(m, texture_img)
            m.apply_transform(T_obj)
            scene_parts.append(m)
            loaded_count += 1

        print(f"  Loaded {link_name} ({cat}/{model}) - {len(obj_meshes)} mesh(es)")

    print(f"\nLoaded {loaded_count} object meshes, skipped {skipped_count}")
    print(f"Total scene parts: {len(scene_parts)}")

    if not scene_parts:
        print("Error: no meshes loaded")
        sys.exit(1)

    # --- Merge and transform to Y-up ---
    print("Merging scene...")
    scene = trimesh.Scene()
    for i, part in enumerate(scene_parts):
        scene.add_geometry(part, node_name=f"part_{i}", geom_name=f"geom_{i}")

    # Apply Z-up to Y-up transform
    scene.apply_transform(_Z_UP_TO_Y_UP)

    # --- Export ---
    print(f"Exporting to {args.output} ...")
    scene.export(args.output, file_type="glb")
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done! Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
