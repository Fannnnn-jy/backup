#!/usr/bin/env python3
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import trimesh
from trimesh.exchange import gltf
from trimesh.visual.material import PBRMaterial, SimpleMaterial
from trimesh.visual.texture import TextureVisuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Matterport textured mesh package to a textured GLB."
    )
    parser.add_argument(
        "input_path",
        help="Path to a matterport_mesh.zip file or an extracted matterport_mesh directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output GLB path. Defaults to <input>.glb.",
    )
    return parser.parse_args()


def is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def extract_if_needed(input_path: Path):
    if is_zip_file(input_path):
        temp_dir = tempfile.TemporaryDirectory(prefix="matterport_mesh_")
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(temp_dir.name)
        return Path(temp_dir.name), temp_dir
    return input_path, None


def find_obj(root: Path) -> Path:
    obj_files = sorted(root.rglob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No OBJ file found under: {root}")
    if len(obj_files) > 1:
        names = ", ".join(str(path) for path in obj_files[:5])
        raise RuntimeError(
            f"Expected exactly one OBJ file under {root}, found {len(obj_files)}: {names}"
        )
    return obj_files[0]


def default_output_path(input_path: Path, obj_path: Path) -> Path:
    if is_zip_file(input_path):
        return input_path.with_suffix(".glb")
    if input_path.is_dir():
        return input_path / f"{obj_path.stem}.glb"
    return obj_path.with_suffix(".glb")


def load_scene(obj_path: Path) -> trimesh.Scene:
    loaded = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    scene = trimesh.Scene()
    scene.add_geometry(loaded)
    return scene


def normalize_scene_materials(scene: trimesh.Scene) -> None:
    """Rewrite textured OBJ materials so exported glTF keeps textures at full brightness."""
    for geometry in scene.geometry.values():
        visual = getattr(geometry, "visual", None)
        if not isinstance(visual, TextureVisuals):
            continue

        material = getattr(visual, "material", None)
        image = getattr(material, "image", None)
        if image is None:
            continue

        material_name = getattr(material, "name", None)
        geometry.visual = TextureVisuals(
            uv=visual.uv,
            image=image,
            material=PBRMaterial(
                name=material_name,
                baseColorTexture=image,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True,
            ),
        )


def mark_textured_materials_unlit(tree: dict) -> None:
    """Mark textured materials as unlit for viewers that honor KHR_materials_unlit."""
    materials = tree.get("materials")
    if not isinstance(materials, list):
        return

    extensions_used = set(tree.get("extensionsUsed", []))
    for material in materials:
        if not isinstance(material, dict):
            continue
        pbr = material.get("pbrMetallicRoughness")
        if not isinstance(pbr, dict) or "baseColorTexture" not in pbr:
            continue

        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0
        material["extensions"] = {"KHR_materials_unlit": {}}
        material["doubleSided"] = True
        extensions_used.add("KHR_materials_unlit")

    if extensions_used:
        tree["extensionsUsed"] = sorted(extensions_used)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input does not exist: {input_path}", file=sys.stderr)
        return 1

    temp_dir = None
    try:
        extracted_root, temp_dir = extract_if_needed(input_path)
        obj_path = find_obj(extracted_root)
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else default_output_path(input_path, obj_path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = load_scene(obj_path)
        normalize_scene_materials(scene)
        output_path.write_bytes(
            gltf.export_glb(scene, tree_postprocessor=mark_textured_materials_unlit)
        )

        print(f"OBJ: {obj_path}")
        print(f"GLB: {output_path}")
        return 0
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
