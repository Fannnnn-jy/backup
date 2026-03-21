import torch
from torch import Tensor


def pose_from_lookat(eye, target, up=(0, 0, 1), device=None) -> Tensor:
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

    if not isinstance(eye, Tensor):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
        assert eye.ndim == 1, eye.ndim
        assert len(eye) == 3, len(eye)
    if not isinstance(target, Tensor):
        target = torch.tensor(target, dtype=torch.float32, device=device)
        assert target.ndim == 1, target.ndim
        assert len(target) == 3, len(target)
    if not isinstance(up, Tensor):
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
