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

import argparse
import ctypes
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

# ── GLEAM imports ────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
GLEAM_ROOT = Path(__file__).resolve().parents[3] / "GLEAM"
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – preprocess a single GLB into a GLEAM-compatible data directory
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_single_glb(
    glb_path: str,
    work_dir: str,
    dataset_name: str = "glb_single",
    drone_height: float = 0.5,
    grid_reso: int = 128,
) -> str:
    """Preprocess one GLB file and return the data directory.

    Output structure inside work_dir:
        {work_dir}/
          customized_data/
            objects/{dataset_name}_ply_center/   ← intermediate PLY
            objects/{dataset_name}_obj/           ← OBJ for URDF
            urdf/{dataset_name}/                  ← scene_0.urdf
            gt/gt_{dataset_name}/                 ← all GT tensors
    """
    _ensure_gleam_preprocess_imports()

    glb_dir  = str(Path(glb_path).parent)
    glb_file = Path(glb_path).name

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
        # Step 0: load GLB, rotate Y→Z, centre, export PLY+OBJ+URDF
        preprocess_glb_meshes(dataset_name, glb_dir, glb_file=glb_file)

        # Height string for file naming (e.g. 0.5 → "0d5", 1.5 → "1d5")
        h_str = (f"{int(drone_height)}"
                 if drone_height == int(drone_height)
                 else f"{drone_height:.1f}".replace(".", "d"))

        # Steps 2–5: voxelise, merge, 2-D map, HDF5
        voxelize_scenes(dataset_name, grid_reso)
        merge_voxel_data(dataset_name, grid_reso)
        generate_2d_maps(dataset_name, grid_reso, height=drone_height)
        convert_to_h5(dataset_name, grid_reso)
    finally:
        os.chdir(orig_cwd)
        _lgy.OPEN_ROBOT_ROOT_DIR = orig_root
        unified_preprocess.OPEN_ROBOT_ROOT_DIR = orig_up_root

    data_dir = os.path.join(work_dir, "data_gleam", "customized_data")
    print(f"[preprocess] Done. Data dir: {data_dir}")
    return data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – run GLEAM for one episode and collect world poses
# ─────────────────────────────────────────────────────────────────────────────

def _gleam_env_and_policy(data_dir, dataset_name, drone_height, device, ckpt_path, buffer_size):
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

    # ── config ────────────────────────────────────────────────────────────────
    import copy
    cfg = copy.deepcopy(Config_GLEAM_Eval)
    cfg.env.num_envs        = 1
    cfg.visual_input.stack  = buffer_size

    # ── sim params ────────────────────────────────────────────────────────────
    sim_params = gymutil.parse_sim_params({
        "physics_engine": "physx",
        "sim_device":     device,
        "headless":       True,
    })

    # ── env ───────────────────────────────────────────────────────────────────
    env_raw = Env_GLEAM_SingleGLB(
        cfg          = cfg,
        sim_params   = sim_params,
        physics_engine = gymapi.SIM_PHYSX,
        sim_device   = device,
        headless     = True,
        glb_data_dir = data_dir,
        dataset_name = dataset_name,
        drone_height = drone_height,
    )
    env = EnvWrapperGLEAM(env_raw)

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
    ckpt_path: str | None,
    device: str = "cuda:0",
    buffer_size: int = 30,
) -> list[dict]:
    """Run GLEAM for one episode and return poses in Z-up world coordinates.

    Each element: {"position_zup": np.ndarray(3,), "yaw": float}
    """
    env_raw, env, model = _gleam_env_and_policy(
        glb_data_dir, dataset_name, drone_height, device, ckpt_path, buffer_size
    )
    import torch

    obs = env.reset()
    poses_zup: list[dict] = []

    for step_idx in range(n_views):
        # ── decide action ─────────────────────────────────────────────────
        if ckpt_path:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
        else:
            # Random movement in XY, hold Z/roll/pitch/yaw
            action = torch.full((1, 6), 64, dtype=torch.int64, device=device)
            action[0, 0] = torch.randint(10, 118, (1,)).item()
            action[0, 1] = torch.randint(10, 118, (1,)).item()

        obs, _, done, _ = env.step(action)

        # ── record pose AFTER the step ────────────────────────────────────
        pose = env_raw.poses[0].cpu().numpy()   # [x, y, z, roll, pitch, yaw]
        poses_zup.append({
            "position_zup": pose[:3].copy(),
            "yaw":          float(pose[5]),
        })

        if done[0]:
            break

    print(f"[gleam] Collected {len(poses_zup)} poses.")
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
                        help="Temp directory for preprocessing. "
                             "Defaults to a system temp dir (deleted after use).")
    parser.add_argument("--keep_work_dir", action="store_true",
                        help="Keep the preprocessing work directory.")
    parser.add_argument("--fov_deg",        type=float, default=90.0)
    parser.add_argument("--image_width",    type=int,   default=512)
    parser.add_argument("--image_height",   type=int,   default=512)
    parser.add_argument("--camera_near",    type=float, default=0.01)
    parser.add_argument("--camera_far",     type=float, default=10.0)
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--buffer_size",    type=int,   default=30)
    args = parser.parse_args()

    if not GLEAM_POLICY_CKPT.is_file():
        raise FileNotFoundError(
            f"GLEAM checkpoint not found: {GLEAM_POLICY_CKPT}"
        )

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    cleanup = False
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="gleam_glb_")
        cleanup = not args.keep_work_dir

    scene_safe = args.scene_id.replace("/", "__").replace(" ", "_")
    dataset_name = f"glb_{scene_safe}"

    data_dir = preprocess_single_glb(
        glb_path     = args.glb,
        work_dir     = work_dir,
        dataset_name = dataset_name,
        drone_height = args.drone_height,
    )

    # ── 2. GLEAM episode ──────────────────────────────────────────────────────
    poses_zup = run_gleam_episode(
        glb_data_dir = data_dir,
        dataset_name = dataset_name,
        drone_height = args.drone_height,
        n_views      = args.n_views,
        ckpt_path    = str(GLEAM_POLICY_CKPT),
        device       = args.device,
        buffer_size  = args.buffer_size,
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
