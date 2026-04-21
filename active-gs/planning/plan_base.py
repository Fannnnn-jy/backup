import numpy as np
import torch
from einops import repeat
import time
import pdb
from tqdm import tqdm
import cv2

from .utils import (
    PathPlanner,
    cal_flight_time,
    inplace_rotation,
    rotation_from_z_batch,
    select_points_within_cone,
    wp2path,
)
from utils.common import FakeQueue, Planner2Gui


class PlanBase:
    def __init__(self, cfg, device, planner_up_axis="z"):
        self.device = device
        self.pitch_angle = cfg.pitch_angle
        self.robot_size = cfg.robot_size
        self.radius = cfg.radius
        self.flight_speed = 1.0
        self.pose = torch.tensor(cfg.init_pose).type(torch.float32)
        self.init = False

        _up_axes = {"x": [1.,0.,0.], "y": [0.,1.,0.], "z": [0.,0.,1.]}
        world_up = torch.tensor(_up_axes.get(planner_up_axis, [0.,0.,1.]), dtype=torch.float32)
        self.world_down = -world_up  # camera "down" reference = -world_up

        self.path_planner = PathPlanner()
        self.path_length_factor = cfg.path_length_factor

        self.use_confidence = cfg.use_confidence
        self.sample_num = cfg.sample_num
        self.max_roi_sample_num = cfg.max_roi_sample_num  # cfg.roi_sample_num

        # gui related
        self.q_planner2gui = FakeQueue()
        self.q_gui2planner = FakeQueue()

    def plan(self, map, simulator, recorder):
        gaussian_map, voxel_map = map
        t_planning = 0
        if self.init:
            ##### graph update
            t_sampling_start = time.time()
            robot_space = self.get_robot_space(voxel_map)
            voxel_map.update_graph(robot_space)

            ##### view candidate sampling
            if self.max_roi_sample_num > 0:
                voxel_map.update_utility(gaussian_map, self.use_confidence)
                roi_candidates = self.generate_roi_candidates(
                    voxel_map, self.max_roi_sample_num
                )
            else:
                roi_candidates = torch.tensor([])

            if self.sample_num - len(roi_candidates) > 0:
                random_candidates = self.generate_random_candidates(
                    voxel_map, self.sample_num - len(roi_candidates)
                )
            else:
                random_candidates = torch.tensor([])

            total_candidates = torch.cat((roi_candidates, random_candidates), dim=0)
            view_xyz = total_candidates[:, :3, 3]
            view_direction = total_candidates[:, :3, 2]
            t_planning += time.time() - t_sampling_start
            self.q_planner2gui.put(Planner2Gui(view_xyz, view_direction))
            print(
                f"\n generate {len(roi_candidates)} roi samples, {len(random_candidates)} random samples"
            )

            ##### utility calculation
            utility_list, t_utility = self.cal_utility(
                gaussian_map, voxel_map, total_candidates, simulator
            )
            t_planning += t_utility

            ##### path planning
            t_path_start = time.time()
            wp_list, wp_length_list = self.path_planner.search_goal(
                self.pose[:3, 3].numpy(),
                total_candidates[:, :3, 3].numpy(),
                voxel_map,
            )
            t_planning += time.time() - t_path_start

            ##### nbv selection
            reachable_mask = torch.isfinite(
                torch.tensor(wp_length_list, dtype=torch.float32)
            )
            if torch.any(reachable_mask):
                reachable_indices = torch.nonzero(reachable_mask, as_tuple=False).view(-1)
                reachable_utilities = utility_list[reachable_indices]
                reachable_lengths = [
                    wp_length_list[idx] for idx in reachable_indices.cpu().tolist()
                ]
                score_list = self.cal_view_scores(
                    reachable_utilities, reachable_lengths
                )
                best_reachable_idx = torch.argmax(score_list).item()
                nbv_id = reachable_indices[best_reachable_idx].item()
                if nbv_id < len(roi_candidates):
                    print("select roi!!!!!!!!!!!!!!!!!!")
                nbv = total_candidates[nbv_id]
                wp_indices = wp_list[nbv_id]
                if len(wp_indices) == 0:
                    raise RuntimeError(
                        self._format_planning_diagnostics(
                            voxel_map=voxel_map,
                            roi_candidates=roi_candidates,
                            random_candidates=random_candidates,
                            total_candidates=total_candidates,
                            utility_list=utility_list,
                            wp_length_list=wp_length_list,
                            reason=(
                                f"selected reachable candidate {nbv_id} returned an empty waypoint list"
                            ),
                            selected_candidate_id=nbv_id,
                        )
                    )
                else:
                    waypoints = voxel_map.index_2_xyz(wp_indices).cpu()
            else:
                raise RuntimeError(
                    self._format_planning_diagnostics(
                        voxel_map=voxel_map,
                        roi_candidates=roi_candidates,
                        random_candidates=random_candidates,
                        total_candidates=total_candidates,
                        utility_list=utility_list,
                        wp_length_list=wp_length_list,
                        reason="no reachable view candidates found",
                    )
                )

        else:
            # move to closest voxel center as initial position
            nbv = torch.eye(4)
            nbv[:3, :3] = self.pose[:3, :3]
            nbv_index = voxel_map.xyz_2_index(self.pose[:3, 3])
            nbv_xyz = voxel_map.index_2_xyz([nbv_index])[0].cpu()
            nbv[:3, 3] = nbv_xyz
            waypoints = torch.stack([self.pose[:3, 3], nbv_xyz])
            self.init = True

        camera_path, path_length = wp2path(
            self.pose[:3, :3],
            nbv[:3, :3],
            waypoints,
            world_down=self.world_down,
        )
        if len(camera_path) == 0:
            raise RuntimeError("planner produced an empty path (no motion candidate).")
        self.pose = nbv

        if recorder is not None:
            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
            recorder.update_time("planning", t_planning)
            recorder.update_time("flight", t_flight)
            recorder.update_path(camera_path, path_length)

        return camera_path

    def generate_random_candidates(self, voxel_map, num):
        """
        generate random view candidates around current pose
        """

        if num <= 0:
            return torch.empty((0, 4, 4), dtype=torch.float32)

        voxel_centers = voxel_map.voxel_centers.cpu().numpy()
        free_mask = voxel_map.free_mask_w_margin.cpu().numpy()

        range_from_start = np.linalg.norm(
            voxel_centers - self.pose[:3, 3].numpy(), axis=1
        )
        within_range = range_from_start <= self.radius
        valid_mask = free_mask & within_range
        if not np.any(valid_mask):
            # Fallback: no free voxel in local radius, search globally in free space.
            valid_mask = free_mask
        valid_centers = voxel_centers[valid_mask]

        if len(valid_centers) == 0:
            # Last fallback: stay at current pose to keep planner alive.
            return self.pose.unsqueeze(0).repeat(num, 1, 1).clone().type(torch.float32)

        random_indices = np.random.choice(
            len(valid_centers), size=num, replace=(len(valid_centers) < num)
        )
        view_positions = valid_centers[random_indices]
        candidates = inplace_rotation(
            view_positions, pitch_angle=self.pitch_angle, num=num
        )
        return candidates

    def generate_roi_candidates(self, voxel_map, num):
        """
        generate targeted view candidates arount ROI
        """

        roi_candiates = torch.tensor([])
        sample_per_roi = 5
        free_mask = voxel_map.free_mask_w_margin
        free_points = voxel_map.voxel_centers[free_mask]

        roi_mask = voxel_map.roi_mask
        roi_centers = voxel_map.voxel_centers[roi_mask]
        roi_normals = voxel_map.voxel_normal[roi_mask]
        roi_distance = torch.linalg.norm(
            roi_centers - self.pose[:3, 3].unsqueeze(0).to(self.device), dim=1
        )
        _, closest_roi_index = torch.sort(roi_distance)
        for roi_index in closest_roi_index:
            roi_center = roi_centers[roi_index]
            roi_normal = roi_normals[roi_index]
            candiate_positions, candidate_views = select_points_within_cone(
                roi_center,
                roi_normal,
                d_close=0.3,
                d_far=2.0,
                cosine_sim=0.5,
                free_points=free_points,
                voxel_map=voxel_map,
                pitch_angle=self.pitch_angle,
            )
            num_candidates = len(candiate_positions)

            # assign voxel center as final xyz
            if num_candidates > 0:
                if num_candidates > sample_per_roi:
                    selected_index = np.random.choice(
                        range(num_candidates),
                        size=sample_per_roi,
                        replace=False,
                    )
                    candiate_positions = candiate_positions[selected_index]
                    candidate_views = candidate_views[selected_index]

                Ts = torch.tensor(
                    repeat(np.eye(4), "h w -> n h w", n=len(candiate_positions))
                )
                Ts[:, :3, 3] = candiate_positions
                Ts[:, :3, :3] = rotation_from_z_batch(candidate_views, world_down=self.world_down)

                roi_candiates = torch.cat((roi_candiates, Ts), dim=0)

            if len(roi_candiates) >= num:
                return roi_candiates.type(torch.float32).cpu()

        return roi_candiates.type(torch.float32).cpu()

    def get_robot_space(self, voxel_map):
        range_from_start = torch.linalg.norm(
            voxel_map.voxel_centers - self.pose[:3, 3].unsqueeze(0).to(self.device),
            dim=1,
        )
        robot_space = range_from_start < self.robot_size
        return robot_space

    def cal_view_scores(self, view_utilities, path_lengths):
        """
        calculate the score of each viewpoint based on its utility and travel cost
        """

        path_lengths = torch.tensor(path_lengths, dtype=torch.float32)
        valid_candidate_mask = ~torch.isinf(path_lengths)
        if not torch.any(valid_candidate_mask):
            return torch.full_like(path_lengths, float("-inf"))

        valid_path_length_sum = torch.sum(path_lengths[valid_candidate_mask])
        if valid_path_length_sum > 0:
            path_lengths = path_lengths / valid_path_length_sum
        else:
            path_lengths = torch.zeros_like(path_lengths)
        path_lengths[~valid_candidate_mask] = 10000000

        utility_sum = torch.sum(view_utilities)
        if utility_sum > 0:
            view_utilities = view_utilities / utility_sum
        else:
            view_utilities = torch.zeros_like(view_utilities)
        view_utilities[torch.isnan(view_utilities)] = 0
        if torch.all(view_utilities == 0):
            view_scores = torch.rand_like(view_utilities)
        else:
            view_scores = view_utilities - self.path_length_factor * path_lengths
        return view_scores

    def _connected_component(self, graph, start_index):
        start_index = tuple(start_index)
        if start_index not in graph:
            return set(), False

        visited = {start_index}
        stack = [start_index]
        while stack:
            current = stack.pop()
            for neighbor, _ in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited, True

    def _format_planning_diagnostics(
        self,
        voxel_map,
        roi_candidates,
        random_candidates,
        total_candidates,
        utility_list,
        wp_length_list,
        reason,
        selected_candidate_id=None,
        max_candidates_to_print=10,
    ):
        graph = voxel_map.graph.dense_graph
        start_pose = self.pose[:3, 3].detach().cpu()
        start_index = tuple(voxel_map.xyz_2_index(start_pose))
        component, start_in_graph = self._connected_component(graph, start_index)
        start_index_tensor = torch.tensor(start_index, device=voxel_map.device).view(1, 3)
        start_linear_index = int(voxel_map.to_linear_indices(start_index_tensor)[0].item())
        free_mask = voxel_map.free_mask
        free_margin_mask = voxel_map.free_mask_w_margin
        occ_mask = voxel_map.occ_mask
        unknown_mask = voxel_map.unknown_mask
        robot_space = self.get_robot_space(voxel_map)

        candidate_positions = total_candidates[:, :3, 3].detach().to(voxel_map.device)
        candidate_xyz = candidate_positions.cpu()
        candidate_indices = [tuple(voxel_map.xyz_2_index(xyz)) for xyz in candidate_xyz]
        candidate_free_mask = voxel_map.in_free_space(candidate_positions).cpu().numpy()

        path_lengths = np.array(wp_length_list, dtype=np.float32)
        reachable_mask = np.isfinite(path_lengths)
        in_graph_mask = np.array([idx in graph for idx in candidate_indices], dtype=bool)
        same_component_mask = np.array(
            [idx in component for idx in candidate_indices], dtype=bool
        )
        roi_mask = np.arange(len(total_candidates)) < len(roi_candidates)

        utility_cpu = utility_list.detach().cpu().numpy()
        utility_nonzero = int(np.count_nonzero(np.abs(utility_cpu) > 1e-8))
        utility_sum = float(np.sum(utility_cpu))

        lines = [
            f"Planning failure: {reason}",
            (
                "start_pose_xyz="
                f"{np.round(start_pose.numpy(), 4).tolist()} "
                f"start_index={start_index} start_in_graph={start_in_graph}"
            ),
            (
                "graph_nodes="
                f"{len(graph)} component_size={len(component)} "
                f"free_margin_voxels={int(free_margin_mask.sum().item())} "
                f"free_voxels={int(free_mask.sum().item())} "
                f"occ_voxels={int(occ_mask.sum().item())} "
                f"unknown_voxels={int(unknown_mask.sum().item())} "
                f"robot_space_voxels={int(robot_space.sum().item())}"
            ),
            (
                "start_voxel_state: "
                f"free_margin={bool(free_margin_mask[start_linear_index].item())} "
                f"free={bool(free_mask[start_linear_index].item())} "
                f"occ={bool(occ_mask[start_linear_index].item())} "
                f"unknown={bool(unknown_mask[start_linear_index].item())}"
            ),
            (
                "candidates: "
                f"roi={len(roi_candidates)} random={len(random_candidates)} total={len(total_candidates)} "
                f"free={int(candidate_free_mask.sum())} in_graph={int(in_graph_mask.sum())} "
                f"same_component={int(same_component_mask.sum())} reachable={int(reachable_mask.sum())}"
            ),
            (
                "utility: "
                f"sum={utility_sum:.6f} nonzero={utility_nonzero}/{len(utility_cpu)} "
                f"min={float(np.min(utility_cpu)):.6f} max={float(np.max(utility_cpu)):.6f}"
            ),
        ]

        if selected_candidate_id is not None:
            sel = int(selected_candidate_id)
            lines.append(
                "selected_candidate: "
                f"id={sel} kind={'roi' if roi_mask[sel] else 'random'} "
                f"xyz={np.round(candidate_xyz[sel].numpy(), 4).tolist()} "
                f"index={candidate_indices[sel]} free={bool(candidate_free_mask[sel])} "
                f"in_graph={bool(in_graph_mask[sel])} "
                f"same_component={bool(same_component_mask[sel])} "
                f"path_length={wp_length_list[sel]} utility={float(utility_cpu[sel]):.6f}"
            )

        ranked_ids = np.argsort(-utility_cpu)[: min(max_candidates_to_print, len(utility_cpu))]
        lines.append("top_candidates_by_utility:")
        for rank, cid in enumerate(ranked_ids.tolist(), start=1):
            lines.append(
                f"  {rank}. id={cid} kind={'roi' if roi_mask[cid] else 'random'} "
                f"xyz={np.round(candidate_xyz[cid].numpy(), 4).tolist()} "
                f"index={candidate_indices[cid]} free={bool(candidate_free_mask[cid])} "
                f"in_graph={bool(in_graph_mask[cid])} "
                f"same_component={bool(same_component_mask[cid])} "
                f"path_length={wp_length_list[cid]} utility={float(utility_cpu[cid]):.6f}"
            )

        return "\n".join(lines)

    def cal_utility(self):
        raise NotImplementedError
