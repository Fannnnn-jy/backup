"""Count rooms per ProcTHOR scene using stage GLB metadata.

Room ids are inferred from wall node names in stage GLB files:
WallInside#wall|<room_id>|...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import trimesh


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


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


def _resolve_stage_config(stages_cfg_dir: Path, template_name: str) -> Path:
    if not template_name.startswith("stages/"):
        raise FileNotFoundError(f"Unexpected stage template name: {template_name}")
    rel = template_name[len("stages/") :]
    cfg = stages_cfg_dir / f"{rel}.stage_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Stage config not found: {cfg}")
    return cfg


def _resolve_render_asset(config_path: Path, render_asset: str) -> Path:
    asset_path = (config_path.parent / render_asset).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Render asset not found: {asset_path}")
    return asset_path


def _count_rooms_from_stage(glb_path: Path) -> int:
    scene = trimesh.load(glb_path, force="scene", process=False)
    room_tokens = set()
    for name in scene.graph.nodes_geometry:
        if not isinstance(name, str):
            continue
        if name.startswith("WallInside#wall|"):
            parts = name.split("|")
            if len(parts) > 1:
                token = parts[1]
                if token != "exterior":
                    room_tokens.add(token)
    return len(room_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count rooms per ProcTHOR scene.")
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
    parser.add_argument("--scene-id", default="", help="Single scene id.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit number of scenes.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path mapping scene_id -> room_count.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    scenes_root = dataset_root / args.scenes_dir
    stages_cfg_dir = dataset_root / "configs/stages"
    if not scenes_root.exists():
        raise FileNotFoundError(f"Scenes dir not found: {scenes_root}")
    if not stages_cfg_dir.exists():
        raise FileNotFoundError(f"Stage config dir not found: {stages_cfg_dir}")

    scene_paths = _scene_instance_paths(scenes_root, args.scene_id or None)
    if args.max_scenes > 0:
        scene_paths = scene_paths[: args.max_scenes]
    if not scene_paths:
        raise FileNotFoundError("No scene_instance.json files found.")

    cache: Dict[Path, int] = {}
    results: Dict[str, int] = {}

    for scene_path in scene_paths:
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        data = _load_json(scene_path)
        stage_template = data.get("stage_instance", {}).get("template_name")
        if not stage_template:
            print(f"[warn] {scene_id}: missing stage_instance.template_name")
            continue
        stage_cfg_path = _resolve_stage_config(stages_cfg_dir, stage_template)
        stage_cfg = _load_json(stage_cfg_path)
        stage_asset = _resolve_render_asset(stage_cfg_path, stage_cfg["render_asset"])
        if stage_asset not in cache:
            cache[stage_asset] = _count_rooms_from_stage(stage_asset)
        room_count = cache[stage_asset]
        results[scene_id] = room_count
        print(f"{scene_id}\t{room_count}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
