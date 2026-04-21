"""Complete pipeline: Y-up GLB → GLEAM preprocessing → GLEAM planning → baseline pose JSON.

Usage:
    python baselines/gleam_poses/generate.py \\
        --glb  /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/procthor/ProcTHOR-Test-21.glb \\
        --scene_id procthor/ProcTHOR-Test-21 \\
        --n_views 30 \\
        --drone_height 0.5 \\
        --run_id run_000 \\
        --output baselines/poses

Coordinate conventions
-----------------------
Input GLB  : Y-up  (GLTF standard)  X∈[-1,1], Y∈[0,1], Z∈[-1,1]
GLEAM world: Z-up  (Isaac Gym)       X∈[-1,1], Y∈[-1,1], Z∈[0,1]
Output JSON: Y-up  (SAPIEN/baseline) same as input GLB

The Y-up→Z-up rotation applied at preprocess time is undone when writing the
baseline JSON so that render_from_poses.py (SAPIEN) gets native GLB coordinates.
"""
from __future__ import annotations


import os
import sys

# ========== 第一步：设置编译相关环境变量 ==========
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'
os.environ['CUDAHOSTCXX'] = 'g++-11'

# ========== 第二步：设置PYTHONPATH ==========
# 1. 获取原有PYTHONPATH（避免覆盖）
original_pythonpath = os.environ.get('PYTHONPATH', '')
# 2. 添加Isaac Gym的python路径
isaacgym_python_path = '/home/ghr/fs/Junyi/GLEAM/isaacgym/python'
# 3. 拼接新的PYTHONPATH（优先用Isaac Gym路径）
new_pythonpath = f"{isaacgym_python_path}:{original_pythonpath}" if original_pythonpath else isaacgym_python_path
os.environ['PYTHONPATH'] = new_pythonpath

# ========== 验证是否设置成功（可选） ==========
print(f"CC: {os.environ['CC']}")
print(f"CXX: {os.environ['CXX']}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")


import argparse
import ctypes
import json
import shutil
import sysconfig
import tempfile
import time
from pathlib import Path

import numpy as np

# ── GLEAM imports ────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
GLEAM_ROOT = Path(__file__).resolve().parents[3] / "GLEAM"
DEFAULT_GLEAM_WORK_DIR = PROJ_ROOT.parent / "data" / "proj" / "baseline_data" / "tmp"
GLEAM_POLICY_CKPT = Path(
    "/home/ghr/fs/Junyi/data/proj/model_weights/GLEAM/rl_model_40000000_steps.zip"
)
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(GLEAM_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / "torch_extensions"))

unified_preprocess = None
preprocess_glb_meshes = None
voxelize_scenes = None
merge_voxel_data = None
generate_2d_maps = None
convert_to_h5 = None


def _normalize_legacy_data_gleam_layout(work_dir: str) -> None:
    """Flatten legacy data_gleam/customized_data/* into data_gleam/*."""
    data_root = Path(work_dir) / "data_gleam"
    legacy_root = data_root / "customized_data"
    if not legacy_root.is_dir():
        return

    migrated = False
    for name in ("gt", "objects", "urdf"):
        src = legacy_root / name
        dst = data_root / name
        if not src.exists():
            continue
        if dst.exists():
            print(f"[preprocess] Legacy path kept in place because destination exists: {src} -> {dst}")
            continue
        shutil.move(str(src), str(dst))
        migrated = True
        print(f"[preprocess] Migrated legacy path: {src} -> {dst}")

    try:
        legacy_root.rmdir()
    except OSError:
        pass
    else:
        if migrated:
            print(f"[preprocess] Removed empty legacy directory: {legacy_root}")


def _preload_libpython_for_isaacgym() -> Path | None:
    """Preload libpython and runtime deps so Isaac Gym can load from Conda/source."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lib_names = [
        sysconfig.get_config_var("LDLIBRARY"),
        f"libpython{version}.so.1.0",
        f"libpython{version}.so",
    ]

    prefixes = [
        Path(p)
        for p in (
            sysconfig.get_config_var("LIBDIR"),
            os.environ.get("CONDA_PREFIX"),
            sys.prefix,
            sys.base_prefix,
        )
        if p
    ]

    candidates: list[Path] = []
    for prefix in prefixes:
        prefix_path = Path(prefix)
        if prefix_path.name == "lib":
            lib_dir = prefix_path
        else:
            lib_dir = prefix_path / "lib"
        for lib_name in lib_names:
            if lib_name:
                candidates.append(lib_dir / lib_name)

    seen: set[Path] = set()
    loaded_libpython: Path | None = None
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        loaded_libpython = candidate
        break

    if os.name != "nt":
        try:
            ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

        bindings_dir = GLEAM_ROOT / "isaacgym" / "python" / "isaacgym" / "_bindings" / "linux-x86_64"
        for lib_name in (
            "libboost_system.so.1.68.0",
            "libboost_thread.so.1.68.0",
            "libarch.so",
            "libtf.so",
            "libmem_filesys.so",
            "libPhysXGpu_64.so",
        ):
            lib_path = bindings_dir / lib_name
            if lib_path.is_file():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    return loaded_libpython


def _ensure_pkg_resources_packaging() -> None:
    """Compatibility shim for older torch codepaths with newer setuptools."""
    try:
        import packaging
        import pkg_resources
    except ImportError:
        return

    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging


def _ensure_gleam_preprocess_imports() -> None:
    """Import Isaac Gym before any GLEAM modules that import torch."""
    global unified_preprocess
    global preprocess_glb_meshes
    global voxelize_scenes
    global merge_voxel_data
    global generate_2d_maps
    global convert_to_h5

    if unified_preprocess is not None:
        return

    _ensure_pkg_resources_packaging()
    _preload_libpython_for_isaacgym()
    import isaacgym  # noqa: F401
    import data_gleam.unified_preprocess as _unified_preprocess
    from data_gleam.unified_preprocess import (
        preprocess_glb_meshes as _preprocess_glb_meshes,
        voxelize_scenes as _voxelize_scenes,
        merge_voxel_data as _merge_voxel_data,
        generate_2d_maps as _generate_2d_maps,
        convert_to_h5 as _convert_to_h5,
    )

    unified_preprocess = _unified_preprocess
    preprocess_glb_meshes = _preprocess_glb_meshes
    voxelize_scenes = _voxelize_scenes
    merge_voxel_data = _merge_voxel_data
    generate_2d_maps = _generate_2d_maps
    convert_to_h5 = _convert_to_h5

# ── Coordinate constants ──────────────────────────────────────────────────────
# Y-up (GLTF) → Z-up (Isaac Gym):  new=[x, -z, y]
Y_UP_TO_Z_UP = np.array([[1, 0,  0],
                          [0, 0, -1],
                          [0, 1,  0]], dtype=np.float64)

# Z-up → Y-up  (inverse = transpose of the above)
Z_UP_TO_Y_UP = Y_UP_TO_Z_UP.T   # [[1,0,0],[0,0,1],[0,-1,0]]


def _mesh_bounds(vertices: np.ndarray) -> dict[str, list[float]]:
    verts = np.asarray(vertices, dtype=np.float64)
    return {
        "min": verts.min(axis=0).tolist(),
        "max": verts.max(axis=0).tolist(),
    }


def _raw_preprocess_transform_path(dataset_name: str, scene_idx: int) -> str:
    return os.path.join(
        "data_gleam",
        "objects",
        f"{dataset_name}_scene_{scene_idx}_raw_to_gleam_preprocess_transform.json",
    )


def _write_raw_scene_transform_metadata(
    *,
    transform_path: str,
    glb_path: str,
    translation_z_up_before_scale: np.ndarray,
    uniform_scale_after_translation: float,
    bounds_raw_y_up: dict[str, list[float]],
    bounds_z_up_after_rotation: dict[str, list[float]],
    bounds_z_up_after_centering: dict[str, list[float]],
    bounds_z_up_after_preprocess: dict[str, list[float]],
    scene_height_after_centering_before_scale: float,
) -> None:
    translation = np.asarray(translation_z_up_before_scale, dtype=np.float64).reshape(3)
    scale = float(uniform_scale_after_translation)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = scale * Y_UP_TO_Z_UP
    affine[:3, 3] = scale * translation
    affine_inv = np.linalg.inv(affine)

    payload = {
        "glb_path": glb_path,
        "transform_semantics": (
            "Maps scene-space points from the raw GLB Y-up frame into the GLEAM "
            "preprocess scene frame used for mesh export, voxelization, and planning. "
            "This is not a camera extrinsic transform, not a per-view pose transform, "
            "and not a dataset-level normalization to a fixed [-1,1]/[0,1] box."
        ),
        "source_frame": "raw_glb_scene_y_up",
        "target_frame": "gleam_preprocess_scene_z_up",
        "applies_to": "scene/world points such as mesh vertices and camera centers",
        "does_not_apply_to": [
            "camera extrinsic matrices directly",
            "voxel indices",
            "image-plane coordinates",
        ],
        "operations_applied_in_order": [
            {
                "type": "axis_rotation",
                "from_up_axis": "Y",
                "to_up_axis": "Z",
                "matrix_3x3_row_major": Y_UP_TO_Z_UP.tolist(),
            },
            {
                "type": "translation_in_rotated_z_up_frame",
                "meaning": "move XY center to origin and move floor to Z=0",
                "translation_xyz": translation.tolist(),
            },
            {
                "type": "uniform_scale_after_translation",
                "scale": scale,
                "applied_only_when_scene_height_gt_100m": True,
            },
        ],
        "raw_y_up_to_gleam_preprocess_affine_4x4_row_major": affine.tolist(),
        "gleam_preprocess_to_raw_y_up_affine_4x4_row_major": affine_inv.tolist(),
        "translation_in_rotated_z_up_before_scale": translation.tolist(),
        "uniform_scale_after_translation": scale,
        "scene_height_after_centering_before_scale": float(scene_height_after_centering_before_scale),
        "bounds_raw_y_up_before_preprocess": bounds_raw_y_up,
        "bounds_z_up_after_rotation_before_centering": bounds_z_up_after_rotation,
        "bounds_z_up_after_centering_before_scale": bounds_z_up_after_centering,
        "bounds_z_up_after_full_preprocess": bounds_z_up_after_preprocess,
    }

    with open(transform_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _preprocess_raw_single_glb_step0(
    *,
    glb_path: str,
    dataset_name: str,
    overwrite: bool = False,
) -> tuple[str, str, str, str]:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for GLB loading: pip install trimesh") from exc

    glb_path = str(Path(glb_path))
    glb_name = Path(glb_path).name
    scene_idx = 0
    base = "data_gleam"
    ply_dir = os.path.join(base, f"objects/{dataset_name}_ply_center")
    obj_dir = os.path.join(base, f"objects/{dataset_name}_obj")
    urdf_dir = os.path.join(base, f"urdf/{dataset_name}")
    transform_path = _raw_preprocess_transform_path(dataset_name, scene_idx)
    for directory in (ply_dir, obj_dir, urdf_dir):
        os.makedirs(directory, exist_ok=True)

    ply_path = os.path.join(ply_dir, f"scene_{scene_idx}.ply")
    obj_path = os.path.join(obj_dir, f"scene_{scene_idx}.obj")
    urdf_path = os.path.join(urdf_dir, f"scene_{scene_idx}.urdf")

    if not overwrite and unified_preprocess._outputs_exist(ply_path, obj_path, urdf_path, transform_path):
        unified_preprocess._log_skip_existing("Step 0 raw", ply_path, obj_path, urdf_path, transform_path)
        return ply_path, obj_path, urdf_path, transform_path

    print(f"[0/1] {glb_name}")
    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Scene):
        meshes = [geom.copy() for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No trimesh geometries found in {glb_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene.copy()

    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh after loading {glb_path}")

    vertices_y_up = np.asarray(mesh.vertices, dtype=np.float64)
    bounds_raw_y_up = _mesh_bounds(vertices_y_up)

    vertices_z_up = (Y_UP_TO_Z_UP @ vertices_y_up.T).T
    mesh.vertices = vertices_z_up
    bounds_z_up_after_rotation = _mesh_bounds(vertices_z_up)

    v = np.asarray(mesh.vertices, dtype=np.float64)
    x_c = (v[:, 0].max() + v[:, 0].min()) / 2.0
    y_c = (v[:, 1].max() + v[:, 1].min()) / 2.0
    z_min = v[:, 2].min()
    translation_z_up = np.array([-x_c, -y_c, -z_min], dtype=np.float64)
    mesh.apply_translation(translation_z_up)

    vertices_centered = np.asarray(mesh.vertices, dtype=np.float64)
    bounds_z_up_after_centering = _mesh_bounds(vertices_centered)
    z_range = float(vertices_centered[:, 2].max() - vertices_centered[:, 2].min())
    scale = 1.0
    if z_range > 100.0:
        scale = 100.0 / z_range
        mesh.apply_scale(scale)
        print(f"  Scaled by {scale:.4f} (height was {z_range:.1f} m)")

    vertices_final = np.asarray(mesh.vertices, dtype=np.float64)
    bounds_z_up_after_preprocess = _mesh_bounds(vertices_final)

    mesh.export(ply_path)
    mesh.export(obj_path)
    obj_rel = os.path.relpath(obj_path, urdf_dir)
    unified_preprocess._generate_scene_urdf(scene_idx, obj_rel, urdf_path)
    _write_raw_scene_transform_metadata(
        transform_path=transform_path,
        glb_path=glb_path,
        translation_z_up_before_scale=translation_z_up,
        uniform_scale_after_translation=scale,
        bounds_raw_y_up=bounds_raw_y_up,
        bounds_z_up_after_rotation=bounds_z_up_after_rotation,
        bounds_z_up_after_centering=bounds_z_up_after_centering,
        bounds_z_up_after_preprocess=bounds_z_up_after_preprocess,
        scene_height_after_centering_before_scale=z_range,
    )

    print(f"  vertices={len(vertices_final):,}  Z-range=[{vertices_final[:,2].min():.2f}, {vertices_final[:,2].max():.2f}] m")
    print(f"  PLY  → {ply_path}")
    print(f"  OBJ  → {obj_path}")
    print(f"  URDF → {urdf_path}")
    print(f"  XFM  → {transform_path}")
    return ply_path, obj_path, urdf_path, transform_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – preprocess a single GLB into a GLEAM-compatible data directory
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_single_glb(
    glb_path: str,
    work_dir: str,
    dataset_name: str = "glb_single",
    drone_height: float = 0.5,
    grid_reso: int = 128,
    overwrite: bool = False,
) -> str:
    """Preprocess one GLB file and return the data directory.

    Output structure inside work_dir:
        {work_dir}/
          data_gleam/
            objects/{dataset_name}_ply_center/   ← intermediate PLY
            objects/{dataset_name}_obj/           ← OBJ for URDF
            urdf/{dataset_name}/                  ← scene_0.urdf
            gt/gt_{dataset_name}/                 ← all GT tensors
    """
    _ensure_gleam_preprocess_imports()
    _normalize_legacy_data_gleam_layout(work_dir)

    # unified_preprocess uses relative paths based on CWD, so cd into work_dir
    orig_cwd = os.getcwd()
    os.chdir(work_dir)

    # Temporarily override OPEN_ROBOT_ROOT_DIR used inside unified_preprocess
    import legged_gym as _lgy
    orig_root = _lgy.OPEN_ROBOT_ROOT_DIR
    orig_up_root = unified_preprocess.OPEN_ROBOT_ROOT_DIR
    _lgy.OPEN_ROBOT_ROOT_DIR = work_dir
    unified_preprocess.OPEN_ROBOT_ROOT_DIR = work_dir

    try:
        print(f"\n[preprocess] GLB → {work_dir}")
        # Step 0: load GLB, rotate Y→Z, centre, optionally scale large scenes,
        # export PLY+OBJ+URDF, and save the raw-scene -> GLEAM-preprocess transform.
        _preprocess_raw_single_glb_step0(
            glb_path=glb_path,
            dataset_name=dataset_name,
            overwrite=overwrite,
        )

        # Steps 2–5: voxelise, merge, 2-D map, HDF5
        voxelize_scenes(dataset_name, grid_reso, overwrite=overwrite)
        merge_voxel_data(dataset_name, grid_reso, overwrite=overwrite)
        generate_2d_maps(dataset_name, grid_reso, height=drone_height, overwrite=overwrite)
        convert_to_h5(dataset_name, grid_reso, overwrite=overwrite)
        _normalize_legacy_data_gleam_layout(work_dir)
    finally:
        os.chdir(orig_cwd)
        _lgy.OPEN_ROBOT_ROOT_DIR = orig_root
        unified_preprocess.OPEN_ROBOT_ROOT_DIR = orig_up_root

    data_dir = os.path.join(work_dir, "data_gleam")
    print(f"[preprocess] Done. Data dir: {data_dir}")
    return data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – run GLEAM for one episode and collect world poses
# ─────────────────────────────────────────────────────────────────────────────

def _gleam_env_and_policy(data_dir, dataset_name, drone_height, device, ckpt_path, buffer_size,
                          forced_init_xy=None, forced_init_yaw=None):
    """Build the single-scene GLEAM env and load the fixed policy checkpoint."""
    _ensure_pkg_resources_packaging()
    from isaacgym import gymapi, gymutil
    import torch
    from gleam.env.config_gleam_eval import Config_GLEAM_Eval
    from gleam.wrapper.env_wrapper_gleam import EnvWrapperGLEAM
    from gleam.network.encoder import Encoder_GLEAM
    from stable_baselines3.common.policies import ActorCriticPolicy_Discrete_Eval
    from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
    from baselines.gleam_poses.env_glb import Env_GLEAM_SingleGLB
    from legged_gym.utils.helpers import class_to_dict

    # ── config ────────────────────────────────────────────────────────────────
    import copy
    cfg = copy.deepcopy(Config_GLEAM_Eval)
    cfg.env.num_envs        = 1
    cfg.visual_input.stack  = buffer_size

    # ── sim params ────────────────────────────────────────────────────────────
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim_params)
    if device.startswith("cuda"):
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True

    # ── env ───────────────────────────────────────────────────────────────────
    print("[policy] creating Env_GLEAM_SingleGLB ...")
    env_raw = Env_GLEAM_SingleGLB(
        cfg          = cfg,
        sim_params   = sim_params,
        physics_engine = gymapi.SIM_PHYSX,
        sim_device   = device,
        headless     = True,
        glb_data_dir = data_dir,
        dataset_name = dataset_name,
        drone_height = drone_height,
        forced_init_xy  = forced_init_xy,
        forced_init_yaw = forced_init_yaw,
    )
    print("[policy] Env_GLEAM_SingleGLB created")
    env = EnvWrapperGLEAM(env_raw)
    print("[policy] EnvWrapperGLEAM created")

    # ── policy ────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=[],
        features_extractor_class=Encoder_GLEAM,
        features_extractor_kwargs=dict(
            encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
            net_param={"transformer_params": [[1, 256], [1, 256]],
                       "append_hidden_shapes": [256, 256]},
            state_input_shape=(buffer_size * 6,),
            visual_input_shape=(1, 128, 128),
        ),
    )
    model = PPO_Grid_Obs(
        policy=ActorCriticPolicy_Discrete_Eval,
        policy_kwargs=policy_kwargs,
        env=env,
        device=device,
    )
    model.set_parameters(ckpt_path)
    print(f"[policy] Loaded weights from {ckpt_path}")

    return env_raw, env, model


def run_gleam_episode(
    glb_data_dir: str,
    dataset_name: str,
    drone_height: float,
    n_views: int,
    ckpt_path: "Optional[str]",
    device: str = "cuda:0",
    buffer_size: int = 30,
    log_every: int = 1,
    forced_init_xy: "Optional[tuple]" = None,
    forced_init_yaw: "Optional[float]" = None,
) -> list[dict]:
    """Run GLEAM for one episode and return poses in Z-up world coordinates.

    Each element: {"position_zup": np.ndarray(3,), "yaw": float}

    Args:
        forced_init_xy:  optional (x, y) starting position in Z-up world coords.
        forced_init_yaw: optional starting yaw in radians (Z-up frame).
    """
    env_raw, env, model = _gleam_env_and_policy(
        glb_data_dir, dataset_name, drone_height, device, ckpt_path, buffer_size,
        forced_init_xy=forced_init_xy, forced_init_yaw=forced_init_yaw,
    )
    import torch

    def _tensor_scalar(value, default=None):
        if value is None:
            return default
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return default

    def _gleam_status_line(step_idx: int, reward, done, step_elapsed_s: float, total_elapsed_s: float) -> str:
        pose = env_raw.poses[0].detach().cpu().numpy()
        action_xy = None
        if hasattr(env_raw, "actions") and env_raw.actions is not None:
            action_xy = env_raw.actions[0, :2].detach().cpu().tolist()
        pose_idx = None
        if hasattr(env_raw, "poses_idx") and env_raw.poses_idx is not None:
            pose_idx = env_raw.poses_idx[0].detach().cpu().tolist()
        coverage = None
        if hasattr(env_raw, "reward_layout_ratio_buf") and len(env_raw.reward_layout_ratio_buf) > 0:
            coverage = _tensor_scalar(env_raw.reward_layout_ratio_buf[-1][0], default=0.0)
        collision = _tensor_scalar(getattr(env_raw, "collision_flag", None)[0], default=False) if hasattr(env_raw, "collision_flag") else False
        collision_rigid = _tensor_scalar(getattr(env_raw, "collision_rigid", None)[0], default=False) if hasattr(env_raw, "collision_rigid") else False
        collision_vis = _tensor_scalar(getattr(env_raw, "collision_vis", None)[0], default=False) if hasattr(env_raw, "collision_vis") else False
        cur_len = _tensor_scalar(getattr(env_raw, "cur_episode_length", None)[0], default=step_idx + 1) if hasattr(env_raw, "cur_episode_length") else step_idx + 1
        reward_value = _tensor_scalar(reward[0] if torch.is_tensor(reward) else reward, default=0.0)
        done_value = bool(_tensor_scalar(done[0] if torch.is_tensor(done) else done, default=False))
        return (
            f"[gleam] step={step_idx + 1}/{n_views} "
            f"ep_len={int(cur_len)} "
            f"reward={reward_value:.4f} "
            f"coverage={float(coverage or 0.0):.4f} "
            f"pose=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) "
            f"yaw={pose[5]:.3f} "
            f"pose_idx={pose_idx} "
            f"action_xy={action_xy} "
            f"collision={bool(collision)} "
            f"(rigid={bool(collision_rigid)}, vis={bool(collision_vis)}) "
            f"done={done_value} "
            f"step_time={step_elapsed_s:.2f}s "
            f"elapsed={total_elapsed_s:.2f}s"
        )

    episode_start = time.perf_counter()
    print(f"[gleam] Resetting env for dataset={dataset_name}...")
    reset_start = time.perf_counter()
    obs = env.reset()
    reset_elapsed = time.perf_counter() - reset_start
    init_pose = env_raw.poses[0].detach().cpu().numpy()
    print(
        f"[gleam] Reset complete in {reset_elapsed:.2f}s "
        f"init_pose=({init_pose[0]:.3f}, {init_pose[1]:.3f}, {init_pose[2]:.3f}) "
        f"yaw={init_pose[5]:.3f}"
    )
    poses_zup: list[dict] = []

    for step_idx in range(n_views):
        # ── decide action ─────────────────────────────────────────────────
        step_start = time.perf_counter()
        if ckpt_path:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
        else:
            # Random movement in XY, hold Z/roll/pitch/yaw
            action = torch.full((1, 6), 64, dtype=torch.int64, device=device)
            action[0, 0] = torch.randint(10, 118, (1,)).item()
            action[0, 1] = torch.randint(10, 118, (1,)).item()

        obs, reward, done, _ = env.step(action)
        step_elapsed = time.perf_counter() - step_start

        # ── record pose AFTER the step ────────────────────────────────────
        pose = env_raw.poses[0].cpu().numpy()   # [x, y, z, roll, pitch, yaw]
        poses_zup.append({
            "position_zup": pose[:3].copy(),
            "yaw":          float(pose[5]),
        })

        if log_every > 0 and ((step_idx + 1) % log_every == 0 or bool(done[0])):
            print(
                _gleam_status_line(
                    step_idx=step_idx,
                    reward=reward,
                    done=done,
                    step_elapsed_s=step_elapsed,
                    total_elapsed_s=time.perf_counter() - episode_start,
                )
            )

        if done[0]:
            break

    print(
        f"[gleam] Collected {len(poses_zup)} poses in "
        f"{time.perf_counter() - episode_start:.2f}s."
    )
    return poses_zup


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – convert Z-up GLEAM poses → Y-up baseline JSON
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_z(yaw: float) -> np.ndarray:
    """3×3 rotation matrix for rotation by yaw around Z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def _mat_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    from scipy.spatial.transform import Rotation as Rot
    q_xyzw = Rot.from_matrix(R).as_quat()   # scipy uses xyzw
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def poses_zup_to_json(
    poses_zup: list[dict],
    scene_id: str,
    method: str,
    run_id: str,
    fov_deg: float,
    image_width: int,
    image_height: int,
    camera_near: float,
    camera_far: float,
) -> dict:
    """Convert Z-up GLEAM poses → Y-up baseline pose JSON dict.

    Coordinate conversion for each pose:
        p_yup  = Z_UP_TO_Y_UP @ p_zup         (position)
        R_c2w_zup = rotation_around_Z(yaw)
        R_c2w_yup = Z_UP_TO_Y_UP @ R_c2w_zup  (camera-to-world rotation)
    """
    M = Z_UP_TO_Y_UP   # [[1,0,0],[0,0,1],[0,-1,0]]

    views = []
    for i, p in enumerate(poses_zup):
        pos_zup = p["position_zup"]
        yaw     = p["yaw"]

        # Position
        pos_yup = (M @ pos_zup).tolist()

        # Rotation: camera-to-world in Y-up
        R_c2w_yup = M @ _rotation_z(yaw)
        quat_wxyz = _mat_to_quat_wxyz(R_c2w_yup)

        views.append({
            "view_idx":        i,
            "position":        [round(v, 6) for v in pos_yup],
            "quaternion_wxyz": [round(v, 6) for v in quat_wxyz],
        })

    return {
        "scene_id":              scene_id,
        "method":                method,
        "run_id":                run_id,
        "max_views":             len(views),
        "fov_deg":               fov_deg,
        "image_width":           image_width,
        "image_height":          image_height,
        "camera_near":           camera_near,
        "camera_far":            camera_far,
        "quaternion_convention": "wxyz",
        "coordinate_frame":      "camera_to_world",
        "views":                 views,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate GLEAM exploration poses for a Y-up GLB scene.")
    parser.add_argument("--glb",      required=True,
                        help="Path to the Y-up .glb scene file.")
    parser.add_argument("--scene_id", required=True,
                        help="Scene identifier, e.g. procthor/ProcTHOR-Test-21.")
    parser.add_argument("--n_views",  type=int, default=30,
                        help="Number of views to collect per episode.")
    parser.add_argument("--drone_height", type=float, default=0.5,
                        help="Drone Z-height in Z-up world (default 0.5).")
    parser.add_argument("--run_id",   default="run_000")
    parser.add_argument("--method",   default="gleam")
    parser.add_argument("--output",
                        default="baselines/poses",
                        help="Root directory for pose JSON output.")
    parser.add_argument("--work_dir", default=None,
                        help="Directory for preprocessing artifacts. "
                             "Defaults to Junyi/data/proj/baseline_data/tmp.")
    parser.add_argument("--keep_work_dir", action="store_true",
                        help="Keep the preprocessing work directory.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild preprocessing artifacts even if they already exist.")
    parser.add_argument("--fov_deg",        type=float, default=90.0)
    parser.add_argument("--image_width",    type=int,   default=512)
    parser.add_argument("--image_height",   type=int,   default=512)
    parser.add_argument("--camera_near",    type=float, default=0.01)
    parser.add_argument("--camera_far",     type=float, default=10.0)
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--buffer_size",    type=int,   default=30)
    parser.add_argument("--log_every",      type=int,   default=1,
                        help="Print one GLEAM progress line every N steps. Use 0 to disable.")
    parser.add_argument("--init_x",     type=float, default=None,
                        help="Fixed starting X in Z-up world coords (skip init-map sampling).")
    parser.add_argument("--init_y",     type=float, default=None,
                        help="Fixed starting Y in Z-up world coords (skip init-map sampling).")
    parser.add_argument("--init_yaw_deg", type=float, default=None,
                        help="Fixed starting yaw in degrees (Z-up, CCW from +X). "
                             "If omitted, uses the environment's default reset yaw.")
    args = parser.parse_args()

    if not GLEAM_POLICY_CKPT.is_file():
        raise FileNotFoundError(
            f"GLEAM checkpoint not found: {GLEAM_POLICY_CKPT}"
        )

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    cleanup = False
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = str(DEFAULT_GLEAM_WORK_DIR)
    work_dir = str(Path(work_dir))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    scene_safe = args.scene_id.replace("/", "__").replace(" ", "_")
    dataset_name = f"glb_{scene_safe}"

    data_dir = preprocess_single_glb(
        glb_path     = args.glb,
        work_dir     = work_dir,
        dataset_name = dataset_name,
        drone_height = args.drone_height,
        overwrite    = args.overwrite,
    )

    # ── 2. GLEAM episode ──────────────────────────────────────────────────────
    forced_init_xy = (args.init_x, args.init_y) if args.init_x is not None and args.init_y is not None else None
    forced_init_yaw = float(np.deg2rad(args.init_yaw_deg)) if args.init_yaw_deg is not None else None

    poses_zup = run_gleam_episode(
        glb_data_dir = data_dir,
        dataset_name = dataset_name,
        drone_height = args.drone_height,
        n_views      = args.n_views,
        ckpt_path    = str(GLEAM_POLICY_CKPT),
        device       = args.device,
        buffer_size  = args.buffer_size,
        log_every    = args.log_every,
        forced_init_xy  = forced_init_xy,
        forced_init_yaw = forced_init_yaw,
    )

    # ── 3. Convert + save ─────────────────────────────────────────────────────
    pose_json = poses_zup_to_json(
        poses_zup    = poses_zup,
        scene_id     = args.scene_id,
        method       = args.method,
        run_id       = args.run_id,
        fov_deg      = args.fov_deg,
        image_width  = args.image_width,
        image_height = args.image_height,
        camera_near  = args.camera_near,
        camera_far   = args.camera_far,
    )

    out_dir = Path(args.output) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_safe}_{args.run_id}.json"
    with open(out_path, "w") as f:
        json.dump(pose_json, f, indent=2)
    print(f"\n[done] Pose JSON saved to {out_path}")
    print(f"       Views: {len(pose_json['views'])}")

    # ── cleanup ───────────────────────────────────────────────────────────────
    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
