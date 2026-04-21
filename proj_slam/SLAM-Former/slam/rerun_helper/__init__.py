import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

from .geometry_utils import NormalGenerator


import rerun as rr
from .visualization_utils import reverse_imagenet_normalize, colormap_image


from typing import Dict, Any

# depth prediction normals computer
#PRED_FORMAT_SIZE = [480,640]#[192, 256]
PRED_FORMAT_SIZE = [680,1200]#[192, 256]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_normals = NormalGenerator(PRED_FORMAT_SIZE[0], PRED_FORMAT_SIZE[1]).to(device)


def to_device(input_dict, key_ignores=[], device="cuda"):
    """ " Moves tensors in the input dict to the gpu and ignores tensors/elements
    as with keys in key_ignores.
    """
    for k, v in input_dict.items():
        if k not in key_ignores:
            input_dict[k] = v.to(device).float()
    return input_dict


def log_source_data(src_entity_path: str, src_data: Dict[str, Any]) -> None:
    src_images_k3hw = reverse_imagenet_normalize(
        torch.tensor(src_data["image_b3hw"][0].to(device))
    )
    num_src_cameras = src_data["world_T_cam_b44"][0].shape[0]
    for src_idx in range(num_src_cameras):
        src_cam_path = f"{src_entity_path}/{src_idx}"
        world_T_cam_44 = src_data["world_T_cam_b44"][0][src_idx].squeeze().cpu().numpy()
        K_44 = src_data["K_s0_b44"][0][src_idx].squeeze().cpu().numpy()
        log_camera(src_cam_path, world_T_cam_44, K_44)
        log_image(src_cam_path, src_images_k3hw[src_idx], denormalize=False)



def log_camera(
        entity_path: str, world_T_cam_44: torch.Tensor, K_44: torch.Tensor, kfd=False, update=False,
) -> None:
    assert world_T_cam_44.shape == (4, 4)
    assert K_44.shape == (4, 4)
    # Convert and log camera parameters
    Rot, trans = world_T_cam_44[:3, :3], world_T_cam_44[:3, 3]
    K_33 = K_44[:3, :3]

    K_33[:2] /= 4

    rr.log(entity_path, rr.Transform3D(translation=trans, mat3x3=Rot))#, axis_length=0))
    if not update: # frontend
        if not kfd:
            rr.log(
                entity_path+'/frustum',
                rr.Pinhole(
                    #image_from_camera=K_33,
                    #width=PRED_FORMAT_SIZE[1]/4,
                    #height=PRED_FORMAT_SIZE[0]/4,
                    fov_y=0.7853982,
                    aspect_ratio=1.7777778,
                    #camera_xyz=rr.ViewCoordinates.RUB,
                    camera_xyz=None,
                    image_plane_distance=0.1,
                    color=[0, 255, 0],
                    line_width=0.003,
                ),
            )
        else:
            rr.log(
                entity_path+'/frustum',
                rr.Pinhole(
                    image_from_camera=K_33,
                    width=PRED_FORMAT_SIZE[1]/4,
                    height=PRED_FORMAT_SIZE[0]/4,
                ),
            )
    else:# backend
        pass



def log_window(
    entity_path: str, world_T_cam_44: torch.Tensor, K_44: torch.Tensor
) -> None:
    assert world_T_cam_44.shape == (4, 4)
    assert K_44.shape == (4, 4)
    # Convert and log camera parameters
    Rot, trans = world_T_cam_44[:3, :3], world_T_cam_44[:3, 3]
    rr.log(entity_path, rr.Transform3D(translation=trans, mat3x3=Rot))#, axis_length=0))



def log_image(
    entity_path: str, color_frame_b3hw: torch.Tensor, denormalize=True
) -> None:
    # Image logging
    color_frame_3hw = color_frame_b3hw.squeeze(0)
    if denormalize:
        main_color_3hw = reverse_imagenet_normalize(color_frame_3hw)
    else:
        main_color_3hw = color_frame_3hw
    pil_image = Image.fromarray(
        np.uint8(main_color_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_image = pil_image.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log(f"{entity_path}/image/rgb", rr.Image(pil_image))


def log_rerun(
    entity_path: str,
    cur_data: Dict[str, Any],
    src_data: Dict[str, Any],
    outputs: Dict[str, Any],
    scene_trimesh_mesh: trimesh.Trimesh,
    should_log_source_cams: bool = True,
) -> None:
    """
    Logs camera intri/extri, depth, rgb, and mesh to rerun.
    """
    curr_entity_path = f"{entity_path}/current_cam"
    src_entity_path = f"{entity_path}/source_cam"
    if should_log_source_cams:
        log_source_data(src_entity_path, src_data)

    world_T_cam_44 = cur_data["world_T_cam_b44"].squeeze().cpu().numpy()
    K_44 = cur_data["K_s0_b44"].squeeze().cpu().numpy()
    log_camera(curr_entity_path, world_T_cam_44, K_44)

    # Depth logging
    depth_pred = outputs["depth_pred_s0_b1hw"]
    our_depth_3hw = depth_pred.squeeze(0)
    our_depth_hw3 = our_depth_3hw.permute(1, 2, 0)
    rr.log(
        f"{curr_entity_path}/image/depth",
        rr.DepthImage(our_depth_hw3.numpy(force=True)),
    )

    # Normal logging
    invK_s0_b44 = cur_data["invK_s0_b44"].to(device)
    normals_b3hw = compute_normals(depth_pred, invK_s0_b44)
    our_normals_3hw = 0.5 * (1 + normals_b3hw).squeeze(0)
    pil_normal = Image.fromarray(
        np.uint8(our_normals_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    rr.log(f"{curr_entity_path}/image/normal", rr.Image(pil_normal))

    # Image logging
    color_frame_b3hw = (
        cur_data["high_res_color_b3hw"]
        if "high_res_color_b3hw" in cur_data
        else cur_data["image_b3hw"]
    )
    color_frame_3hw = color_frame_b3hw.squeeze(0)
    main_color_3hw = reverse_imagenet_normalize(color_frame_3hw)
    pil_image = Image.fromarray(
        np.uint8(main_color_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_image = pil_image.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log(f"{curr_entity_path}/image/rgb", rr.Image(pil_image))

    # lowest cost guess from the cost volume
    lowest_cost_bhw = outputs["lowest_cost_bhw"]
    lowest_cost_3hw = colormap_image(
        lowest_cost_bhw,
        vmin=0,
        vmax=5,
    )
    pil_cost = Image.fromarray(
        np.uint8(lowest_cost_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_cost = pil_cost.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log("lowest_cost_volume", rr.Image(pil_cost))

    # Fused mesh logging
    rr.log(
        f"{entity_path}/mesh",
        rr.Mesh3D(
            vertex_positions=scene_trimesh_mesh.vertices,
            triangle_indices=scene_trimesh_mesh.faces,
            vertex_colors=scene_trimesh_mesh.visual.vertex_colors,
        ),
    )


