import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.pose_enc import quat_to_mat, mat_to_quat

def depth_to_world_coords_points(
    depth_map,
    extrinsic,
    intrinsic,
    eps=1e-8,
):
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (B, H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (B, 3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (B, 3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (B, H, W, 3) and valid depth mask (B, H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    # cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic)    ##camera to world

    R_cam_to_world = extrinsic[:, :3, :3]     #  B, 3, 3
    t_cam_to_world = extrinsic[:, :3, 3]      #  B, 3

    # Apply the rotation and translation to the camera coordinates
    # world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    world_coords_points = torch.einsum("bij,bhwj->bhwi", R_cam_to_world, cam_coords_points) + t_cam_to_world[:, None, None]

    return world_coords_points

def depth_to_cam_coords_points(depth_map, intrinsic):
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    B, H, W = depth_map.shape
    assert intrinsic.shape[-2:] == (3, 3), "Intrinsic matrix must be 3x3"
    # assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[:, 0, 0], intrinsic[:, 1, 1]   # B
    cu, cv = intrinsic[:, 0, 2], intrinsic[:, 1, 2]   # B

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))   # H, W  
    u = torch.from_numpy(u).to(depth_map)[None,: , :].repeat([B, 1, 1])   # B, H, W  
    v = torch.from_numpy(v).to(depth_map)[None,: , :].repeat([B, 1, 1])   # B, H, W  

    # Unproject to camera coordinates
    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)    # B H W 3

    return cam_coords

def compute_relative_poses(poses):
    """
    计算每个相机相对于第一个相机的相对位姿（XYZW顺序）
    :param poses: 输入位姿 [B, N, 7]，前3维为平移向量(x,y,z)，后4维为旋转四元数(x,y,z,w)
    :return: 相对位姿 [B, N, 7]，第一个相机与自身的相对位姿为 [0,0,0, 0,0,0,1]
    """
    B, N, _ = poses.shape
    
    # 分离平移和四元数 [B, N, 3] 和 [B, N, 4] (XYZW顺序)
    t = poses[..., :3]
    q = poses[..., 3:]  # [x, y, z, w]
    
    # 获取参考相机（第一个相机）的位姿 [B, 1, 3] 和 [B, 1, 4]
    t_ref = t[:, :1, :]
    q_ref = q[:, :1, :]  # XYZW顺序

    # 计算相对平移: t_rel = q_ref^{-1} * (t_i - t_ref) * q_ref
    delta_t = t - t_ref  # [B, N, 3]
    
    # 计算相对旋转: q_rel = q_ref^{-1} ⊗ q_i (XYZW顺序)
    # 四元数逆在XYZW顺序下为 [-x, -y, -z, w]
    q_ref_inv = torch.cat([-q_ref[..., :3], q_ref[..., 3:]], dim=-1)  # [B, 1, 4]
    q_rel = quaternion_multiply_xyzw(q_ref_inv, q)  # [B, N, 4] (XYZW顺序)

    # 将相对平移转换到参考相机坐标系
    R_ref_inv = quat_to_mat(q_ref_inv)  # [B, 1, 3, 3]
    t_rel = torch.einsum('bijk,bnk->bnj', R_ref_inv, delta_t)  # [B, N, 3]

    # 合并结果 [B, N, 7]
    relative_poses = torch.cat([t_rel, q_rel], dim=-1)
    
    return relative_poses

def quaternion_inverse_xyzw(q):
    """计算XYZW顺序四元数的逆 [x, y, z, w] -> [-x, -y, -z, w]"""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def compute_pairwise_relative_poses(poses):
    """
    计算所有相机两两之间的相对位姿
    :param poses: 输入位姿 [B, N, 7] (前3维平移, 后4维XYZW四元数)
    :return: 相对位姿 [B, N, N, 7] (第[i,j]位表示从相机i到相机j的变换)
    """
    B, N, _ = poses.shape
    device = poses.device
    
    # 分离平移和四元数
    t = poses[..., :3]  # [B, N, 3]
    q = poses[..., 3:]  # [B, N, 4] (XYZW)
    
    # 计算每个相机的旋转矩阵和逆旋转矩阵
    R = quat_to_mat(q)  # [B, N, 3, 3]
    R_inv = R.transpose(-1, -2)  # [B, N, 3, 3] (旋转矩阵的逆=转置)
    
    # 计算相对平移: t_ij = R_i^T (t_j - t_i)
    t_diff = t.unsqueeze(2) - t.unsqueeze(1)  # [B, N, N, 3] (t_j - t_i)
    t_rel = torch.einsum('bnik,bnjk->bnji', R_inv, t_diff)  # [B, N, N, 3]
    
    # 计算相对旋转: q_ij = q_i^{-1} ⊗ q_j
    q_inv = quaternion_inverse_xyzw(q)  # [B, N, 4]
    q_rel = quaternion_multiply_xyzw(
        q_inv.unsqueeze(2),  # [B, N, 1, 4]
        q.unsqueeze(1)       # [B, 1, N, 4]
    )  # [B, N, N, 4]
    
    # 合并结果 [B, N, N, 7]
    relative_poses = torch.cat([t_rel, q_rel], dim=-1)
    
    # 对角线(i=j)设为单位位姿 [0,0,0, 0,0,0,1]
    identity = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device, dtype=poses.dtype)
    relative_poses[:, torch.arange(N), torch.arange(N)] = identity
    
    return relative_poses

def quaternion_multiply_xyzw(q1, q2):
    """
    四元数乘法 (XYZW顺序) q1 ⊗ q2 [B, N, 4]
    """
    # 提取四元数分量 (XYZW顺序)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # 四元数乘法公式[1](@ref)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([x, y, z, w], dim=-1)  # 保持XYZW顺序
