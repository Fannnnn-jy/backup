from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CNN/RNN weights from a trained checkpoint.")
    parser.add_argument("--model", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument(
        "--mode",
        choices=("cnn", "rnn", "both"),
        default="both",
        help="Which weights to visualize.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/weight_viz",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=64,
        help="Max number of CNN filters to tile.",
    )
    return parser.parse_args()


def _load_state_dict(model_path: Path):
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_val = float(np.min(array))
    max_val = float(np.max(array))
    if math.isclose(min_val, max_val):
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def _tile_filters(filters: np.ndarray, max_filters: int) -> np.ndarray:
    num_filters = min(filters.shape[0], max_filters)
    filters = filters[:num_filters]
    height, width = filters.shape[1], filters.shape[2]
    grid = int(math.ceil(math.sqrt(num_filters)))
    tiled = np.zeros((grid * height, grid * width), dtype=filters.dtype)
    for idx in range(num_filters):
        row = idx // grid
        col = idx % grid
        tiled[row * height : (row + 1) * height, col * width : (col + 1) * width] = filters[idx]
    return tiled


def _find_weight_keys(state_dict: dict, prefix: str, dims: int) -> list[str]:
    keys = []
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        if not key.endswith("weight"):
            continue
        if getattr(tensor, "ndim", None) != dims:
            continue
        keys.append(key)
    keys.sort()
    return keys


def _pick_first_conv_key(keys: Iterable[str]) -> str | None:
    keys = list(keys)
    if not keys:
        return None
    for key in keys:
        if ".0.weight" in key:
            return key
    return keys[0]


def _visualize_cnn(state_dict: dict, out_dir: Path, max_filters: int) -> list[Path]:
    saved = []
    for prefix in ("actor_cnns.", "critic_cnns."):
        keys = _find_weight_keys(state_dict, prefix, dims=4)
        key = _pick_first_conv_key(keys)
        if key is None:
            continue
        weight = state_dict[key].detach().cpu().numpy()
        filters = weight.mean(axis=1)
        tiled = _tile_filters(filters, max_filters)
        img = _normalize_to_uint8(tiled)
        colored = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        out_path = out_dir / f"{prefix.strip('.')}__{key.replace('.', '_')}.png"
        cv2.imwrite(str(out_path), colored)
        saved.append(out_path)
    return saved


def _visualize_rnn(state_dict: dict, out_dir: Path) -> list[Path]:
    saved = []
    for prefix in ("memory_a.rnn.", "memory_c.rnn."):
        key = None
        for candidate in (f"{prefix}weight_ih_l0", f"{prefix}weight_ih_l0_reverse"):
            if candidate in state_dict:
                key = candidate
                break
        if key is None:
            for candidate in state_dict:
                if candidate.startswith(prefix) and "weight_ih" in candidate:
                    key = candidate
                    break
        if key is None:
            continue
        weight = state_dict[key].detach().cpu().numpy()
        img = _normalize_to_uint8(weight)
        colored = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        out_path = out_dir / f"{prefix.strip('.')}__{key.replace('.', '_')}.png"
        cv2.imwrite(str(out_path), colored)
        saved.append(out_path)
    return saved


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = _load_state_dict(model_path)

    saved_paths: list[Path] = []
    if args.mode in ("cnn", "both"):
        saved_paths.extend(_visualize_cnn(state_dict, out_dir, args.max_filters))
    if args.mode in ("rnn", "both"):
        saved_paths.extend(_visualize_rnn(state_dict, out_dir))

    if saved_paths:
        print("Saved weight visualizations:")
        for path in saved_paths:
            print(f"- {path}")
    else:
        print("No matching weights found for the requested mode.")


if __name__ == "__main__":
    main()
