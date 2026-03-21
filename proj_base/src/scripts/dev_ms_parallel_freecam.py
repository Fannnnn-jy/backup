import math
import os

os.environ["MS_ASSET_DIR"] = "/datasets/v2p/current/fs/Junyi"

import gymnasium as gym
import numpy as np
import sapien
import torch
import tyro
from torch import Tensor
from torchvision.utils import make_grid

from actrec.utils.plot import plot_rgb_depth
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils


def pose_from_lookat(eye, target, up=(0, 0, 1), device=None) -> Tensor:
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

    if not isinstance(eye, torch.Tensor):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
        assert eye.ndim == 1, eye.ndim
        assert len(eye) == 3, len(eye)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32, device=device)
        assert target.ndim == 1, target.ndim
        assert len(target) == 3, len(target)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, dtype=torch.float32, device=device)
        assert up.ndim == 1, up.ndim
        assert len(up) == 3, len(up)

    def normalize_tensor(x, eps=1e-6):
        x = x.view(-1, 3)
        norm = torch.linalg.norm(x, dim=-1)
        zero_vectors = norm < eps
        x[zero_vectors] = torch.zeros(3, device=x.device).float()
        x[~zero_vectors] /= norm[~zero_vectors].view(-1, 1)
        return x

    forward = normalize_tensor(target - eye)

    up = normalize_tensor(up)
    left = torch.cross(up, forward, dim=-1)
    left = normalize_tensor(left)
    up = torch.cross(forward, left, dim=-1)
    rotation = torch.stack([forward, left, up], dim=-1)
    return torch.cat([eye, matrix_to_quaternion(rotation)], dim=-1)


def main(
    env_id: str = "ReplicaCAD_SceneManipulation-v1",
    save_path: str = "ReplicaCAD_SceneManipulation-v1_demo.png",
):
    T = 10
    shift = torch.tensor([0, -0.5, +0.5])
    cam_poses = []
    for i in range(T):
        cam_from = torch.tensor([math.sin(i / T * 2 * math.pi), math.cos(i / T * 2 * math.pi), 1]) + shift
        cam_lookat = torch.tensor([0, 0, 0]) + shift
        cam_pose = pose_from_lookat(cam_from.unsqueeze(0), cam_lookat.unsqueeze(0)).to("cuda")
        cam_poses.append(cam_pose)

    np.random.seed(0)
    torch.manual_seed(0)
    env_kwargs = dict(
        obs_mode="sensor_data",
        reward_mode=None,
        control_mode=None,
        num_envs=4,
        render_backend="gpu",
        sim_backend="physx_cuda",
        new_sensor_configs={
            "free_cam": dict(
                uid="free_cam",
                pose=sapien.Pose(),
                width=256,
                height=256,
                near=0.001,
                far=100,
                # fov=1.5708,
                fov=1.0,
            )
        },
    )
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    camera = env.unwrapped._sensors["free_cam"].camera

    pose = cam_poses[5]
    sapien_pose = sapien_utils.Pose.create_from_pq(p=pose[:, :3], q=pose[:, 3:])
    camera.mount.set_pose(sapien_pose.sp)
    env.scene.step()
    env.scene.update_render()
    camera.take_picture()
    camera._render_cameras[0].take_picture()
    raw_rgb, raw_points_seg = camera.get_picture(["Color", "PositionSegmentation"])

    rgb = raw_rgb[..., :3] / 255.0  # (N, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2)  # (N, 3, H, W)
    depth = raw_points_seg[..., 2]  # (N, H, W)
    depth = depth.unsqueeze(1)  # (N, 1, H, W)
    depth = depth / 1000.0  # mm -> m
    depth = -depth  # opengl -> opencv
    rgb_grid = make_grid(rgb, nrow=2, padding=0)
    depth_grid = make_grid(depth, nrow=2, padding=0)[0]

    fig = plot_rgb_depth(rgb_grid.permute(1, 2, 0).cpu().numpy(), depth_grid.cpu().numpy())
    fig.savefig(save_path)


if __name__ == "__main__":
    tyro.cli(main)
