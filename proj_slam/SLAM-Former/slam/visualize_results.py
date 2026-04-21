#!/usr/bin/env python3
"""
visualize_results.py
Standalone script to visualize pre-saved SLAM results using Viser.

Reads:
- final_pc/*.npz - Per-frame point clouds and confidence masks
- final_traj.txt - Camera trajectory in TUM format
- final.ply - Combined point cloud (optional, for reference)

Usage:
    python visualize_results.py --result_dir /path/to/results --port 8080
"""

import os
import sys
import argparse
import glob
import numpy as np
import viser
import viser.transforms as tf
import time
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
from natsort import natsorted

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def load_trajectory(traj_path):
    """Load trajectory from TUM format file.
    
    TUM format: timestamp tx ty tz qx qy qz qw
    
    Returns:
        poses: (N, 4, 4) array of camera poses (c2w)
        timestamps: list of timestamps
    """
    data = np.loadtxt(traj_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    poses = []
    timestamps = []
    
    for row in data:
        ts = row[0]
        tx, ty, tz = row[1:4]
        qx, qy, qz, qw = row[4:8]
        
        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = [tx, ty, tz]
        
        poses.append(pose)
        timestamps.append(ts)
    
    return np.array(poses), timestamps


def load_pointclouds(pc_dir):
    """Load per-frame point clouds from npz files.
    
    Each npz file contains:
        - pointcloud: (H, W, 3) point positions
        - mask: (H, W) confidence mask
        - color: (H, W, 3) RGB colors (optional)
    
    Returns:
        pts_list: list of point clouds
        mask_list: list of masks
        colors_list: list of colors (or None entries if not available)
        frame_ids: list of frame IDs
    """
    npz_files = natsorted(glob.glob(os.path.join(pc_dir, "*.npz")))
    
    pts_list = []
    mask_list = []
    colors_list = []
    frame_ids = []
    
    for npz_path in npz_files:
        data = np.load(npz_path)
        pts = data['pointcloud']  # (H, W, 3)
        mask = data['mask']       # (H, W)
        
        # Load colors if available
        if 'color' in data:
            colors = data['color']  # (H, W, 3)
        else:
            colors = None
        
        pts_list.append(pts)
        mask_list.append(mask)
        colors_list.append(colors)
        
        # Extract frame ID from filename
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        try:
            frame_ids.append(float(fname))
        except ValueError:
            frame_ids.append(len(frame_ids))
    
    return pts_list, mask_list, colors_list, frame_ids


def load_ply(ply_path):
    """Load combined point cloud from PLY file."""
    if not O3D_AVAILABLE:
        print("[yellow]Open3D not available, skipping PLY loading.[/yellow]")
        return None, None
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    return points, colors


class ResultViewer:
    """Interactive 3D viewer for pre-saved SLAM results."""
    
    def __init__(
        self,
        pts_list,
        mask_list,
        poses,
        colors_list=None,
        frame_ids=None,
        combined_pts=None,
        combined_colors=None,
        port=8080,
        downsample_factor=10,
    ):
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        
        self.pts_list = pts_list
        self.mask_list = mask_list
        self.poses = poses
        self.colors_list = colors_list
        self.frame_ids = frame_ids or list(range(len(pts_list)))
        self.combined_pts = combined_pts
        self.combined_colors = combined_colors
        
        self.num_frames = len(pts_list)
        self.pc_handles = []
        self.cam_handles = []
        self.frame_nodes = []
        self.fourd = False
        
        # Generate camera gradient colors
        self._generate_camera_colors()
        
        # Setup GUI
        self._setup_gui(downsample_factor)
        
        self.server.on_client_connect(self._connect_client)
    
    def _generate_camera_colors(self):
        """Generate gradient colors for cameras."""
        if self.num_frames > 1:
            normalized_indices = np.arange(self.num_frames) / (self.num_frames - 1)
        else:
            normalized_indices = np.array([0.0])
        cmap = cm.viridis
        self.camera_colors = cmap(normalized_indices)
    
    def _setup_gui(self, downsample_factor):
        """Setup interactive GUI elements."""
        
        # Reset up direction
        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set camera 'up' to current camera's 'up'.",
        )
        
        @gui_reset_up.on_click
        def _(event):
            client = event.client
            if client:
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
        
        # 3D/4D toggle
        button3 = self.server.gui.add_button("4D (Current Frame Only)")
        button4 = self.server.gui.add_button("3D (All Frames)")
        
        @button3.on_click
        def _(_):
            self.fourd = True
        
        @button4.on_click
        def _(_):
            self.fourd = False
        
        # Point size slider
        self.psize_slider = self.server.add_gui_slider(
            "Point Size", min=0.0001, max=0.05, step=0.0001, initial_value=0.002
        )
        
        @self.psize_slider.on_update
        def _(_):
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value
        
        # Camera size slider
        self.camsize_slider = self.server.add_gui_slider(
            "Camera Size", min=0.01, max=0.5, step=0.01, initial_value=0.02
        )
        
        @self.camsize_slider.on_update
        def _(_):
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
        
        # Downsample slider
        self.downsample_slider = self.server.add_gui_slider(
            "Downsample Factor", min=1, max=100, step=1, initial_value=downsample_factor
        )
        
        @self.downsample_slider.on_update
        def _(_):
            self._regenerate_point_clouds()
        
        # Show camera checkbox
        self.show_camera_checkbox = self.server.add_gui_checkbox("Show Cameras", initial_value=True)
        
        @self.show_camera_checkbox.on_update
        def _(_):
            for handle in self.cam_handles:
                handle.visible = self.show_camera_checkbox.value
        
        # Combined point cloud toggle
        if self.combined_pts is not None:
            self.show_combined_checkbox = self.server.add_gui_checkbox("Show Combined PLY", initial_value=False)
            
            @self.show_combined_checkbox.on_update
            def _(_):
                if hasattr(self, 'combined_pc_handle'):
                    self.combined_pc_handle.visible = self.show_combined_checkbox.value
    
    def _regenerate_point_clouds(self):
        """Regenerate all point clouds with current settings."""
        for handle in self.pc_handles:
            try:
                handle.remove()
            except:
                pass
        self.pc_handles.clear()
        
        for i in range(self.num_frames):
            self._add_point_cloud(i)
    
    def _add_point_cloud(self, idx):
        """Add a point cloud to the scene."""
        pts = self.pts_list[idx]
        mask = self.mask_list[idx]
        
        # Apply mask and reshape
        pts_flat = pts.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        pts_masked = pts_flat[mask_flat]
        
        # Generate colors (use per-frame colors if available, otherwise gradient)
        if self.colors_list is not None and idx < len(self.colors_list) and self.colors_list[idx] is not None:
            colors = self.colors_list[idx].reshape(-1, 3)[mask_flat]
        else:
            # Use frame-based gradient color
            color = self.camera_colors[idx][:3]
            colors = np.tile(color, (len(pts_masked), 1))
        
        # Downsample
        ds = int(self.downsample_slider.value)
        if ds > 1 and len(pts_masked) > 0:
            indices = np.arange(0, len(pts_masked), ds)
            pts_masked = pts_masked[indices]
            colors = colors[indices]
        
        if len(pts_masked) == 0:
            return
        
        handle = self.server.add_point_cloud(
            name=f"/frames/{idx}/pts",
            points=pts_masked,
            colors=colors,
            point_size=self.psize_slider.value,
        )
        self.pc_handles.append(handle)
    
    def _add_camera(self, idx):
        """Add a camera frustum to the scene."""
        if idx >= len(self.poses):
            return
        
        pose = self.poses[idx]
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        q = tf.SO3.from_matrix(R).wxyz
        
        # Default camera params
        fov = 1.0
        aspect = 1.33
        
        camera_color = self.camera_colors[idx]
        camera_color_rgb = tuple((camera_color[:3] * 255).astype(int))
        
        handle = self.server.add_camera_frustum(
            name=f"/frames/{idx}/camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=self.camsize_slider.value,
            color=camera_color_rgb,
        )
        self.cam_handles.append(handle)
    
    def _connect_client(self, client):
        """Handle client connection."""
        client.gui.add_text("Frames", str(self.num_frames))
        client.gui.add_text("Status", "Ready")
    
    def animate(self):
        """Setup animation and playback controls."""
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Frame", min=0, max=max(0, self.num_frames - 1), step=1, initial_value=0
            )
            gui_next = self.server.add_gui_button("Next")
            gui_prev = self.server.add_gui_button("Prev")
            gui_playing = self.server.add_gui_checkbox("Playing", False)
            gui_fps = self.server.add_gui_slider("Playback FPS", min=1, max=30, step=1, initial_value=5)
        
        @gui_next.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames
        
        @gui_prev.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames
        
        @gui_playing.on_update
        def _(_):
            gui_timestep.disabled = gui_playing.value
            gui_next.disabled = gui_playing.value
            gui_prev.disabled = gui_playing.value
        
        # Create frame nodes
        self.server.add_frame("/frames", show_axes=False)
        self.frame_nodes = []
        
        for i in range(self.num_frames):
            self.frame_nodes.append(
                self.server.add_frame(f"/frames/{i}", show_axes=False)
            )
            self._add_point_cloud(i)
            if self.show_camera_checkbox.value:
                self._add_camera(i)
        
        # Add combined point cloud if available
        if self.combined_pts is not None and len(self.combined_pts) > 0:
            colors = self.combined_colors if self.combined_colors is not None else np.ones_like(self.combined_pts) * 0.5
            self.combined_pc_handle = self.server.add_point_cloud(
                name="/combined_ply",
                points=self.combined_pts[::100],  # heavily downsampled
                colors=colors[::100] if colors is not None else None,
                point_size=0.001,
            )
            if hasattr(self, 'show_combined_checkbox'):
                self.combined_pc_handle.visible = self.show_combined_checkbox.value
        
        # Main loop
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % self.num_frames
            
            for i, frame_node in enumerate(self.frame_nodes):
                frame_node.visible = i <= gui_timestep.value if not self.fourd else i == gui_timestep.value
            
            time.sleep(1.0 / gui_fps.value)
    
    def run(self):
        """Run the viewer."""
        print(f"Viser server running at http://localhost:{self.server.get_port()}")
        print("Press Ctrl+C to exit.")
        self.animate()


def main():
    parser = argparse.ArgumentParser(description="Visualize pre-saved SLAM results")
    parser.add_argument("--result_dir", type=str, default='path/to/output_dir/of/demo.py', help="Directory containing final_pc/, final_traj.txt, final.ply")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--downsample", type=int, default=10, help="Point cloud downsample factor")
    args = parser.parse_args()
    
    result_dir = args.result_dir
    
    # Load trajectory
    traj_path = os.path.join(result_dir, "final_traj.txt")
    if not os.path.exists(traj_path):
        print(f"[red]Trajectory file not found: {traj_path}[/red]")
        sys.exit(1)
    
    poses, timestamps = load_trajectory(traj_path)
    print(f"Loaded {len(poses)} poses from {traj_path}")
    
    # Load per-frame point clouds
    pc_dir = os.path.join(result_dir, "final_pc")
    if not os.path.isdir(pc_dir):
        print(f"[red]Point cloud directory not found: {pc_dir}[/red]")
        sys.exit(1)
    
    pts_list, mask_list, colors_list, frame_ids = load_pointclouds(pc_dir)
    print(f"Loaded {len(pts_list)} point clouds from {pc_dir}")
    
    # Check if colors are available
    has_colors = any(c is not None for c in colors_list)
    if has_colors:
        print(f"Colors loaded from npz files")
    else:
        print(f"No colors found in npz files, using frame-based gradient colors")
    
    # Load combined PLY (optional)
    ply_path = os.path.join(result_dir, "final.ply")
    combined_pts, combined_colors = None, None
    if os.path.exists(ply_path):
        combined_pts, combined_colors = load_ply(ply_path)
        if combined_pts is not None:
            print(f"Loaded combined PLY with {len(combined_pts)} points")
    
    # Create and run viewer
    viewer = ResultViewer(
        pts_list=pts_list,
        mask_list=mask_list,
        poses=poses,
        colors_list=colors_list if has_colors else None,
        frame_ids=frame_ids,
        combined_pts=combined_pts,
        combined_colors=combined_colors,
        port=args.port,
        downsample_factor=args.downsample,
    )
    viewer.run()


if __name__ == "__main__":
    main()
