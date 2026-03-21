import torch
from PIL import Image
from dust3r.utils.image import depth_edge
import math
import os
from dust3r.utils.misc import move_to_device
import argparse

import numpy as np
from dust3r.utils.vis_utils import write_ply

from dust3r.utils.image import load_images_for_eval as load_images
from layers.pose_enc import pose_encoding_to_extri
from typing import Union

def save_image_grid(images: np.ndarray, grid_shape: tuple, save_path: str):
    """
    images: numpy array of shape (N, H, W, 3)
    grid_shape: (rows, cols)
    """
    H, W = images.shape[1], images.shape[2]
    grid = np.zeros((grid_shape[0]*H, grid_shape[1]*W, 3), dtype=np.uint8)
    
    for i in range(min(len(images), grid_shape[0]*grid_shape[1])):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        grid[row*H:(row+1)*H, col*W:(col+1)*W] = images[i]
    
    Image.fromarray(grid).save(save_path)

def save_image_grid_auto(images: Union[np.ndarray, torch.Tensor], save_path: str):
    num_images = images.shape[0]
    """
    images: np.ndarray of shape (N, H, W, 3) in [0, 255] or torch.Tensor of shape (N, 3, H, W) in range [0, 1]
    """
    if isinstance(images, torch.Tensor):
        assert images.ndim == 4 and (images.shape[1] == 3 or images.shape[-1] == 3), f"images must be a 4D torch tensor with shape (N, 3, H, W) or (N, H, W, 3)"
        if images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        images = (images.detach().cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(images, np.ndarray):
        assert images.ndim == 4 and images.shape[3] == 3, f"images must be a 4D numpy array with shape (N, H, W, 3)"
    else:
        raise ValueError(f"images must be a numpy array or a torch tensor, but got {type(images)}")

    cols = math.floor(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    save_image_grid(images, (rows, cols), save_path)

def postprocess(points, colors):
    clean_mask1 = ~depth_edge(points[::2, 0, ..., 2], rtol=0.03)
    clean_mask2 = ~depth_edge(points[1::2, 0, ..., 2], rtol=0.03)

    clean_points = torch.cat([points[::2, 0, clean_mask1[0]], points[::2, 1, clean_mask2[0]]], dim=1).reshape(-1, 3)
    clean_colors = torch.cat([colors[::2, 0, clean_mask1[0]], colors[::2, 1, clean_mask2[0]]], dim=1).reshape(-1, 3)

    return clean_points.detach().cpu().numpy(), clean_colors.cpu().numpy()

def recover_image(normalized_tensor):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_tensor = normalized_tensor * std + mean
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    return unnormalized_tensor
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference with WinT3R ")
    
    parser.add_argument("--data_path", type=str, default='examples/001',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="Path to save the output .ply file.")
    parser.add_argument("--save_name", type=str, default='recon.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--inference_mode", type=str, default='online',
                        help="WinT3R inference mode. online or offline")
    parser.add_argument("--interval", type=int, default=10,
                        help="Interval to sample video. Default: 10 for video")
    parser.add_argument("--ckpt", type=str, default='/home/ghr/fs/Junyi/data/proj/model_weights/WinT3R.bin',
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    args = parser.parse_args()

    file = "examples/001"
    dataset = load_images(args.data_path, size=512, verbose=True, crop=True, interval=args.interval)

    from dust3r.wint3r import WinT3R
    model = WinT3R(
            state_size=1024,
            state_pe="2d",
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=[512, 512],
            head_type="conv",
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            landscape_only=False,
        ).cuda()

    weights = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(weights, strict=False)

    batch = move_to_device(dataset, args.device)
    imgs = torch.stack([recover_image(view['img'].detach().cpu()) for view in batch], dim=1)

    model.eval()
    with torch.no_grad():
        pred = model(batch, ret_first_pred=False, mode=args.inference_mode)

    colors = imgs.permute(0, 1, 3, 4, 2)
    B, N, H, W, _ = colors.shape 

    extrinsics = pose_encoding_to_extri(pred["camera_pos_enc"][-1])
    R_cam_to_world = extrinsics[:, :, :3, :3]     #  B, S, 3, 3
    t_cam_to_world = extrinsics[:, :, :3, 3]      #  B, S, 3
    world_coords_points = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, pred['pts_local']) + t_cam_to_world[:, :, None, None]   #B, S, H, W, 3
    pred_pts = world_coords_points
    pred_conf = pred['conf']
    pred_depth = pred['pts_local'][..., 2:]

    pred = {"conf": pred_conf, "points": pred_pts, "depth":pred_depth}

    masks_depth = pred['depth'][..., 0]  < 500.0
    masks_edge = ~depth_edge(pred_depth.squeeze(-1), rtol=0.05)
    masks = masks_depth*masks_edge

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(args.save_dir, exist_ok=True)
    write_ply(pred['points'][masks].detach().cpu().numpy().reshape(-1, 3), colors[masks.detach().cpu()].reshape(-1, 3).detach().cpu().numpy(), save_path)
    save_image_grid_auto(colors[0].permute(0,3,1,2), save_path.replace('ply', 'png'))
    print('Saved in', save_path)




