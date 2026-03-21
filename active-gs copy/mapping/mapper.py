import torch
import time

from utils.operations import *
from utils.common import Camera, Mapper2Gui, FakeQueue, TextColors
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from depth_anything_3.api import DepthAnything3
# depth_model = DepthAnything3.from_pretrained("depth-anything/da3mono-large", cache_dir='/home/ghr/fs/Junyi/model_weights')
# depth_model = depth_model.to(device=device)

def depth_prediction(rgb_np, intrinsic=None):
    if rgb_np.dtype != np.uint8:
        rgb_uint8 = (rgb_np * 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb_np
    img_numpy_list = [rgb_uint8]
    prediction = depth_model.inference(img_numpy_list, intrinsics=intrinsic)
    depth_504 = prediction.depth[0]
    depth_512 = cv2.resize(depth_504, (512, 512), interpolation=cv2.INTER_LINEAR)

    return depth_512 # numpy array

def depth2pred(dataframe):
    rgb = dataframe["rgb"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1) 
    intrinsic = dataframe["intrinsic"].unsqueeze(0).cpu().numpy() 
    depth_pred = depth_prediction(rgb, intrinsic=intrinsic)
    dataframe['depth'] = torch.from_numpy(depth_pred).unsqueeze(0).to(torch.float32)  # [1, H, W]
    return dataframe

def visualize_and_analyze(dataframe, i):
    output_dir = "depth_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 数据提取与预处理 ---
    # RGB: [1, 3, H, W] -> [H, W, 3]
    rgb = dataframe["rgb"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1) 
    intrinsic = dataframe["intrinsic"].unsqueeze(0).cpu().numpy() 

    # 原始深度与平滑深度
    depth_raw = dataframe["depth"].squeeze(0).cpu().numpy()
    depth_smooth = get_smooth_depth(depth_raw)
    depth_pred = depth_prediction(rgb, intrinsic=intrinsic)
    
    print(depth_raw.shape, depth_smooth.shape, depth_pred.shape)
    print(depth_raw.dtype, depth_smooth.dtype, depth_pred.dtype)
    print(depth_raw.min(), depth_raw.max(), depth_smooth.min(), depth_smooth.max(), depth_pred.min(), depth_pred.max())
    
    # --- 2. 尺度对齐 (Scale Alignment) ---
    # 目的：将预测的相对深度缩放到与 Raw Depth 一致的量级（如米）
    valid_mask = depth_raw > 0
    if valid_mask.any():
        # 使用中位数对齐，这种方法对异常值鲁棒性最强
        scale = np.median(depth_raw[valid_mask]) / np.median(depth_pred)
        depth_pred_aligned = depth_pred * scale
        print(f"-> Calculated Scale Factor: {scale:.4f}")
    else:
        depth_pred_aligned = depth_pred
        print("-> Warning: No valid depth for scaling.")

    # --- 3. 确定统一的颜色映射范围 (Vmin, Vmax) ---
    # 使用 Raw Depth 的分位数来定义色阶，防止离群点撑大范围
    if valid_mask.any():
        v_min = np.percentile(depth_raw[valid_mask], 2)   # 2% 分位数
        v_max = np.percentile(depth_raw[valid_mask], 98)  # 98% 分位数
    else:
        v_min, v_max = depth_raw.min(), depth_raw.max()

    # --- 4. 打印统计信息 ---
    print(f"\n===== Frame {i} Statistics (Aligned) =====")
    depth_maps_to_compare = [
        ("Raw Depth", depth_raw), 
        ("Smooth Depth", depth_smooth),
        ("Predicted (Aligned)", depth_pred_aligned)
    ]
    for name, d_map in depth_maps_to_compare:
        vals = d_map[valid_mask] if (valid_mask.any() and name != "Predicted (Aligned)") else d_map.flatten()
        print(f"[{name}] Mean: {vals.mean():.3f}m | Max: {vals.max():.3f}m | Min: {vals.min():.3f}m")

    # --- 5. 图像保存：单图保存 ---
    for name, d_map in [("raw", depth_raw), ("smooth", depth_smooth), ("pred", depth_pred_aligned)]:
        plt.figure(figsize=(10, 8))
        # 核心：使用统一的 vmin 和 vmax
        im = plt.imshow(d_map, cmap='magma', vmin=v_min, vmax=v_max)
        plt.title(f"{name.upper()} DEPTH (Unified Scale: {v_min:.1f}-{v_max:.1f}m)")
        plt.axis('off')
        plt.colorbar(im, shrink=0.8)
        
        save_path = os.path.join(output_dir, f"frame_{i:04d}_depth_{name}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # --- 6. 图像保存：并排展示图 ---
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    display_list = [
        ("RGB", rgb, None),
        ("Raw Depth (GT)", depth_raw, 'magma'),
        ("Smooth Depth", depth_smooth, 'magma'),
        ("Predicted (Aligned)", depth_pred_aligned, 'magma')
    ]
    
    for ax, (title, data, cmap) in zip(axes, display_list):
        if cmap:
            # 核心：并排图中也必须强制统一颜色量程
            im = ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max)
            plt.colorbar(im, ax=ax, shrink=0.7, label='Meters' if title != "RGB" else "")
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    comp_path = os.path.join(output_dir, f"frame_{i:04d}_comparison_all.png")
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close()

    print(f"All images saved. Unified Range: [{v_min:.2f}, {v_max:.2f}]")

    # 替换 assert False 为 sys.exit 避免 SIGSEGV 崩溃
    import sys
    print(f"\nStopping at Frame {i} for inspection. Check: {output_dir}")
    sys.exit(0)

class IncrementalMapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # map instance
        self.gaussian_map = None
        self.voxel_map = None

        # gui related
        self.use_gui = False
        self.q_mapper2gui = FakeQueue()
        self.q_gui2mapper = FakeQueue()
        self.pause = False
        self.init = False

    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map

    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
        self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device)

    def get_new_dataframe(self, i):

        # return way points to the nbv
        path = self.planner.plan(self.current_map, self.simulator, self.recorder)

        # for visualization only
        if self.use_gui:
            for pose in path:
                dataframe = self.simulator.simulate(pose)
                camera_frame = Camera.init_from_mapper(None, dataframe)
                self.q_mapper2gui.put(
                    Mapper2Gui(
                        current_frame=camera_frame,
                    )
                )
                time.sleep(0.05)

        # dataframe at nbv as keyframe
        dataframe = self.simulator.simulate(path[-1])
        # dataframe = self.simulator.simulate(path[0])

        # dataframe = depth2pred(dataframe) ### 注视掉是原来的深度读取逻辑
        
        # visualize_and_analyze(self.simulator.simulate(path[0]), 0) ###
        
        camera_frame = Camera.init_from_mapper(i, dataframe)
        self.q_mapper2gui.put(
            Mapper2Gui(
                current_frame=camera_frame,
            )
        )
        return dataframe

    def run(self):
        torch.cuda.empty_cache()
        self.init_map()
        frame_id = 0

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )
        while self.recorder is None or self.recorder.is_alive:
            # pause information from gui
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            dataframe = self.get_new_dataframe(frame_id) #
            dataframe = {k: v.to(self.device) for k, v in dataframe.items()}

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")
            t_mapper_start = time.time()

            # update gaussian map
            self.gaussian_map.update(dataframe)

            # update voxel map
            self.voxel_map.update(dataframe)

            t_mapper = time.time() - t_mapper_start
            frame_id += 1

            # send map to gui for visualization
            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                )
            )

            # update recorder or/and save map
            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper)
                self.recorder.log()
                self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                if self.recorder.require_record:
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")
                    self.recorder.save_path()
            time.sleep(0.1)

        if self.recorder is not None:
            # Flush latest keyframe poses even if mission ends before next record interval.
            self.recorder.save_path()

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )
