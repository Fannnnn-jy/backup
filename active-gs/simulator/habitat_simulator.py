import habitat_sim
import numpy as np
import quaternion
import torch
import os
import logging
import trimesh
from .utils import *

# turn off non-critical log from simulator
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


MESH_EXTENSIONS = {
    ".glb",
    ".gltf",
    ".obj",
    ".ply",
    ".stl",
    ".off",
    ".dae",
    ".3ds",
    ".fbx",
}


def _resolve_path(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _axis_alignment_matrix(from_axis, to_axis):
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    from_axis = str(from_axis).lower()
    to_axis = str(to_axis).lower()
    if from_axis not in axes or to_axis not in axes:
        raise ValueError(f"invalid axis alignment: from={from_axis}, to={to_axis}")

    src = axes[from_axis]
    dst = axes[to_axis]
    if np.allclose(src, dst):
        R = np.eye(3, dtype=np.float32)
    elif np.allclose(src, -dst):
        # 180-degree rotation around any axis orthogonal to src.
        orth = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if np.allclose(np.abs(np.dot(orth, src)), 1.0):
            orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(src, orth)
        v = v / np.linalg.norm(v)
        R = -np.eye(3, dtype=np.float32) + 2.0 * np.outer(v, v)
    else:
        v = np.cross(src, dst)
        s = np.linalg.norm(v)
        c = float(np.dot(src, dst))
        vx = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=np.float32,
        )
        R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T


def _load_scene_mesh(mesh_path, world_from_scene=None):
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        # Merge all geometries in world coordinates so bbox/metrics match Habitat.
        geometries = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry.get(geometry_name, None)
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            geometry = geometry.copy()
            geometry.apply_transform(transform)
            geometries.append(geometry)
        if not geometries:
            raise ValueError(f"no mesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
        print(f"loaded scene mesh from {mesh_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    else:
        mesh = loaded
        print(f"loaded mesh from {mesh_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"failed to load a valid mesh from {mesh_path}")

    if world_from_scene is not None:
        mesh.apply_transform(world_from_scene)

    return mesh



class HabitatSimulator:
    def __init__(self, simulator_cfg, scene_cfg):
        print("\n ----------configure habitat simulator----------")

        # get simulator backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.gpu_device_id = 0
        backend_cfg.load_semantic_mesh = bool(
            getattr(simulator_cfg, "load_semantic_mesh", False)
        )
        scene_id = _resolve_path(scene_cfg.scene_id)
        if not os.path.exists(scene_id):
            raise FileNotFoundError(f"scene file not found: {scene_id}")
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = simulator_cfg.physics.enable
        self.scene_name = scene_cfg.scene_name
        self.has_missing_surface = bool(scene_cfg.get("has_missing_surface", False))

        mesh_path = _resolve_path(scene_cfg.get("mesh_path", None))
        if mesh_path is None:
            scene_ext = os.path.splitext(scene_id)[1].lower()
            if scene_ext in MESH_EXTENSIONS:
                mesh_path = scene_id
            else:
                raise ValueError(
                    "mesh_path is required when scene_id is not a mesh asset."
                )
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"mesh file not found: {mesh_path}")

        scene_up_axis = str(scene_cfg.get("scene_up_axis", "z"))
        planner_up_axis = str(scene_cfg.get("planner_up_axis", scene_up_axis))
        self.scene_up_axis = scene_up_axis
        self.planner_up_axis = planner_up_axis
        self.scene_to_world = _axis_alignment_matrix(scene_up_axis, planner_up_axis)
        self.world_to_scene = np.linalg.inv(self.scene_to_world).astype(np.float32)
        # scene → Habitat (Y-up) world frame: e.g. Z-up scene → Y-up Habitat
        self.scene_to_habitat = _axis_alignment_matrix(scene_up_axis, "y")
        self.habitat_to_scene = np.linalg.inv(self.scene_to_habitat).astype(np.float32)
        self.world_to_z_up = _axis_alignment_matrix(planner_up_axis, "z") # id

        self.mesh = _load_scene_mesh(mesh_path, world_from_scene=self.scene_to_world)
        self.bbox = np.array(self.mesh.bounding_box.bounds)

        # get sensor config
        sensor_specs = []
        self.resolution = np.array(simulator_cfg.sensor.resolution)
        H, W = self.resolution
        self.fov = np.array(simulator_cfg.sensor.fov)
        vfov, hfov = self.fov
        self.intrinsic = compute_camera_intrinsic(
            H, W, vfov, hfov, normalize=simulator_cfg.sensor.normalize
        )
        self.depth_noise_co = simulator_cfg.sensor.depth_noise_co
        self.depth_range = simulator_cfg.sensor.depth_range
        self.sensor_position = np.array(simulator_cfg.sensor.position, dtype=np.float32)
        self.debug_log_pose = bool(getattr(simulator_cfg, "debug_log_pose", False))
        self.debug_log_interval = max(
            int(getattr(simulator_cfg, "debug_log_interval", 1)), 1
        )
        self._sim_step = 0

        self.init_pos = np.array([0, -1, 0.5])
        self.init_quat = quaternion.from_rotation_matrix(np.eye(3)) # [1, 0, 0, 0]  # [w, x, y, z]

        # all sensors have the same intrinsic
        for sensor_type in simulator_cfg.sensor.sensor_type:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_type
            sensor_spec.sensor_type = SENSOR_TYPE[sensor_type]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_spec.resolution = [H, W]
            sensor_spec.vfov = vfov
            sensor_spec.hfov = hfov
            sensor_spec.position = simulator_cfg.sensor.position
            sensor_specs.append(sensor_spec)

        # get agent config
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        print(
            "\n scene:",
            self.scene_name,
            "\n scene_up_axis:",
            scene_up_axis,
            "\n planner_up_axis:",
            planner_up_axis,
            "\n bounding_box:",
            self.bbox.tolist(),
            "\n resolution:",
            self.resolution.tolist(),
            "\n fov:",
            self.fov.tolist(),
            "\n depth_range:",
            self.depth_range,
            "\n depth_noise_co:",
            self.depth_noise_co,
        )

        # spawn simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        if backend_cfg.enable_physics:
            self.sim.set_gravity(simulator_cfg.physics.gravity)

        print("\n ----------load habitat simulator----------")
        self.data = {}

    def planner_to_habitat_pose(self, c2w):
        c2w = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
        c2w_scene_cv = self.world_to_scene @ c2w
        c2w_scene_gl = opencv_to_opengl_camera(c2w_scene_cv)
        return self.scene_to_habitat @ c2w_scene_gl

    def habitat_to_planner_pose(self, c2w_habitat):
        c2w_habitat = np.asarray(c2w_habitat, dtype=np.float32).reshape(4, 4)
        c2w_scene_gl = self.habitat_to_scene @ c2w_habitat
        c2w_scene_cv = opengl_to_opencv_camera(c2w_scene_gl)
        return self.scene_to_world @ c2w_scene_cv
    
    def yup_world_to_zup_world_c2w(self, c2w_yup):
        c2w_yup = np.asarray(c2w_yup, dtype=np.float32).reshape(4, 4)

        T_z_from_y = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        return T_z_from_y @ c2w_yup

    def planner_to_json_pose(self, c2w):
        # Export poses through the same Habitat transform path used at render time,
        # then convert them into a canonical OpenCV camera_to_world matrix in Z-up.
        # print("c2w", c2w)
        # c2w_habitat = self.planner_to_habitat_pose(c2w)
        # print("c2w_habitat", c2w_habitat)
        # c2w_world = self.habitat_to_planner_pose(c2w_habitat)
        c2w_world = self.yup_world_to_zup_world_c2w(c2w)
        # print("c2w_world", c2w_world)
        # return self.world_to_z_up @ c2w_world
        
        return c2w_world

    def simulate(self, c2w, valid_mask_only=False, require_gt=False):
        self._sim_step += 1
        # Transform: world (planner frame) → scene → Habitat (Y-up, OpenGL camera)
        c2w_habitat = self.planner_to_habitat_pose(c2w.numpy())
        orientation = quaternion.from_rotation_matrix(c2w_habitat[:3, :3])
        position = np.array(c2w_habitat[:3, 3], dtype=np.float64)

        agent_state = habitat_sim.agent.AgentState(position, orientation)
        self.sim.get_agent(0).set_state(agent_state)

        # if self.debug_log_pose and self._sim_step % self.debug_log_interval == 0:
        #     sensor_tf = np.eye(4, dtype=np.float32)
        #     sensor_tf[:3, 3] = self.sensor_position
        #     camera_habitat = c2w_habitat @ sensor_tf
        #     camera_cv = opengl_to_opencv_camera(camera_habitat)
        #     agent_cv = c2w.numpy()
        #     print(
        #         "\n [sim_pose]",
        #         f"step={self._sim_step}",
        #         f"agent_cv_xyz={np.round(agent_cv[:3, 3], 4).tolist()}",
        #         f"agent_scene_cv_xyz={np.round(c2w_scene_cv[:3, 3], 4).tolist()}",
        #         f"sensor_local_xyz={np.round(self.sensor_position, 4).tolist()}",
        #         f"camera_cv_xyz={np.round(camera_cv[:3, 3], 4).tolist()}",
        #         f"delta_cv_xyz={np.round(camera_cv[:3, 3] - agent_cv[:3, 3], 4).tolist()}",
        #     )
        #     print(
        #         " [sim_pose]",
        #         f"agent_habitat_xyz={np.round(c2w_habitat[:3, 3], 4).tolist()}",
        #         f"camera_habitat_xyz={np.round(camera_habitat[:3, 3], 4).tolist()}",
        #     )

        # get observations
        obs = self.sim.get_sensor_observations()
        color = obs.get("color", None)
        depth = obs.get("depth", None)
        valid_mask = None
        # import cv2
        # bgr_img = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
        # cv2.imwrite("debug.png", bgr_img) 
        # assert False, "stop here for debugging"

        if (
            self.debug_log_pose
            and self._sim_step % self.debug_log_interval == 0
            and depth is not None
        ):
            raw_valid = depth > 0
            print(
                " [sim_depth]",
                f"raw_valid_ratio={raw_valid.mean():.4f}",
                f"raw_min={float(depth.min()):.4f}",
                f"raw_max={float(depth.max()):.4f}",
            )

        # use for planning purpose to exclude missing surfaces
        if valid_mask_only and depth is not None:
            valid_mask = depth > 0
            return valid_mask

        # use for mapping and test
        else:
            if color is not None:
                rgb = color[:, :, :3] / 255.0
                rgb = torch.from_numpy(rgb.astype(np.float32))
                rgb = rgb.permute(2, 0, 1)  # (C, H, W)

            if depth is not None:
                valid_mask = depth > 0  # missing surface return depth=0

                if not require_gt:
                    # for mapping
                    range_mask = (depth > self.depth_range[0]) & (
                        depth < self.depth_range[1]
                    )
                    depth_noise_std = depth * self.depth_noise_co
                    depth_noise = np.random.normal(scale=depth_noise_std)
                    depth += depth_noise
                    depth[~range_mask] = -1.0  # depth = -1 for out of range

                depth[~valid_mask] = -2.0  # depth = -2 for missing surfaces
                depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(
                    0
                )  # (1, H, W)

            data_frame = {
                "extrinsic": c2w,
                "intrinsic": self.intrinsic,
                "rgb": rgb,
                "depth": depth,
                "depth_range": torch.tensor(self.depth_range),
            }

            return data_frame
