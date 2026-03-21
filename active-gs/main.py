import hydra
import torch
import warnings
import torch.multiprocessing as mp
import os
import yaml
from omegaconf import OmegaConf

from visualization import gui
from utils.common import MissionRecorder
from simulator import get_simulator
from mapping import get_mapper
from planning import get_planner

warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_mp_context():
    return mp.get_context("spawn")


def _normalize_scene_name(scene_name, scene_root=None):
    scene_name = str(scene_name)
    scene_root = (
        os.path.abspath(os.path.expanduser(str(scene_root)))
        if scene_root is not None
        else None
    )

    if os.path.isabs(scene_name):
        resolved_scene_id = os.path.abspath(os.path.expanduser(scene_name))
        if scene_root is not None:
            try:
                logical_scene_name = os.path.relpath(resolved_scene_id, scene_root)
            except ValueError:
                logical_scene_name = os.path.basename(resolved_scene_id)
        else:
            logical_scene_name = os.path.basename(resolved_scene_id)
    else:
        logical_scene_name = scene_name
        resolved_scene_id = (
            os.path.join(scene_root, scene_name) if scene_root is not None else scene_name
        )

    logical_scene_name = logical_scene_name.replace("\\", "/")
    resolved_scene_id = os.path.abspath(os.path.expanduser(resolved_scene_id))
    scene_export_id = os.path.splitext(logical_scene_name)[0]
    scene_tag = scene_export_id.replace("/", "__")
    return logical_scene_name, resolved_scene_id, scene_export_id, scene_tag


def _prepare_scene_cfg(cfg):
    input_scene_name = cfg.scene_name if cfg.scene_name is not None else cfg.scene.scene_name
    logical_scene_name, resolved_scene_id, scene_export_id, scene_tag = (
        _normalize_scene_name(input_scene_name, cfg.scene.get("scene_root", None))
    )
    cfg.scene.scene_name = logical_scene_name
    cfg.scene.scene_id = resolved_scene_id
    return scene_export_id, scene_tag


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg):
    scene_export_id, scene_tag = _prepare_scene_cfg(cfg)

    # set up mode config
    if cfg.debug:
        mission_recorder = None

    else:
        experiment_path = os.path.join(
            cfg.experiment.output_dir,
            str(cfg.experiment.exp_id),
            scene_export_id,
            cfg.planner.planner_name,
            str(cfg.experiment.run_id),
        )
        os.makedirs(experiment_path, exist_ok=True)

        # save experiment configuration
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(f"{experiment_path}/exp_config.yaml", "w") as file:
            yaml.dump(cfg_dict, file)

        mission_recorder = MissionRecorder(experiment_path, cfg.experiment)

    # load components
    mapping_agent = get_mapper(cfg, device)
    simulator = get_simulator(cfg)
    planner = get_planner(cfg, device)

    # set up gui messages
    if cfg.use_gui:
        mp_ctx = _get_mp_context()
        init_event = mp_ctx.Event()
        q_mapper2gui = mp_ctx.Queue()
        q_gui2mapper = mp_ctx.Queue()
        q_planner2gui = mp_ctx.Queue()
        q_gui2planner = mp_ctx.Queue()

        mapping_agent.use_gui = True
        mapping_agent.q_mapper2gui = q_mapper2gui
        mapping_agent.q_gui2mapper = q_gui2mapper

        planner.q_planner2gui = q_planner2gui
        planner.q_gui2planner = q_planner2gui

        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
            "planner_receive": q_planner2gui,
            "planner_send": q_gui2planner,
        }
        gui_process = mp_ctx.Process(
            target=gui.run,
            args=(init_event, cfg.gui, params_gui),
        )
        gui_process.start()
        init_event.wait()

    # load components to mapping module
    mapping_agent.load_recorder(mission_recorder)
    mapping_agent.load_simulator(simulator)
    mapping_agent.load_planner(planner)

    # start mission
    mapping_agent.run()

    # save keyframe poses to baseline directory
    if not cfg.debug and mission_recorder is not None:
        run_id_str = f"run_{int(cfg.experiment.run_id):03d}"
        pose_dir = os.path.join(
            cfg.experiment.pose_output_dir, cfg.experiment.pose_method_name
        )
        pose_path = os.path.join(pose_dir, f"{scene_tag}_{run_id_str}.json")
        metadata = {
            "scene_id": scene_export_id,
            "method": cfg.experiment.pose_method_name,
            "run_id": run_id_str,
            "max_views": int(cfg.experiment.max_frames),
            "fov_deg": float(simulator.fov[0]),
            "image_height": int(simulator.resolution[0]),
            "image_width": int(simulator.resolution[1]),
            "camera_near": float(simulator.depth_range[0]),
            "camera_far": float(simulator.depth_range[1]),
        }
        mission_recorder.save_baseline_poses(
            pose_path,
            metadata,
            simulator=simulator,
        )

    # wait for gui to be closed
    if cfg.use_gui:
        gui_process.join()


if __name__ == "__main__":
    main()
