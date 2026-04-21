import os
from pathlib import Path
from isaacgym import gymapi
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args
from gleam.wrapper.env_wrapper_gleam import EnvWrapperGLEAM
from gleam.network.encoder import Encoder_GLEAM
from gleam.env.env_gleam_custom import Env_GLEAM_Custom, discover_custom_scene_specs
from gleam.env.config_gleam_eval import Config_GLEAM_Eval
from gleam.env.config_gleam import DroneCfgPPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks_gleam import EvalCallback_GLEAM
from stable_baselines3.common.evaluation_gleam import evaluate_policy_grid_obs
from stable_baselines3.common.policies import ActorCriticPolicy_Discrete_Eval
from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
from stable_baselines3.utils import get_time_str
from wandb_utils import team_name, project_name
from wandb_utils.wandb_callback import WandbCallback

OPEN_ROBOT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DEFAULT_CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/GLEAM/rl_model_40000000_steps.zip"
DEFAULT_BASELINE_POSE_ROOT = Path(__file__).resolve().parents[3] / "proj" / "baselines" / "poses"
DEFAULT_BASELINE_POSE_METHOD = "gleam-gt-depth"


def _parse_scene_prefixes(raw_value):
    if raw_value is None or len(raw_value.strip()) == 0:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _largest_divisor_leq(total, upper):
    upper = max(1, min(total, upper))
    for value in range(upper, 0, -1):
        if total % value == 0:
            return value
    return 1


def main():
    additional_args = [
        {
            "name": "--eval_device",
            "type": str,
            "default": "cuda:0",
        },
        {
            "name": "--buffer_size",
            "type": int,
            "default": 30,
            "help": "length of buffer"
        },
        {
            "name": "--n_steps",
            "type": int,
            "default": 512,
            "help": "number of steps to collect in each env"
        },
        {
            "name": "--batch_size",
            "type": int,
            "default": 128,
            "help": "SGD batch size"
        },
        {
            "name": "--save_freq",
            "type": int,
            "default": 50000,
            "help": "save the model per <save_freq> iter"
        },
        {
            "name": "--total_iters",
            "type": int,
            "default": 1250,
            "help": "the number of training iters"
        },
        {
            "name": "--n_epochs",
            "type": int,
            "default": 5
        },
        {
            "name": "--use_target_kl",
            "type": bool,
            "default": True
        },
        {
            "name": "--target_kl",
            "type": float,
            "default": 0.05
        },
        {
            "name": "--vf_coeff",
            "type": float,
            "default": 0.8
        },
        {
            "name": "--ent_coeff",
            "type": float,
            "default": 0.01
        },
        {
            "name": "--lr",
            "type": float,
            "default": 0.0
        },
        {
            "name": "--unflatten_terrain",
            "type": bool,
            "default": False
        },
        {
            "name": "--first_view_camera",
            "type": bool,
            "default": False
        },
        {
            "name": "--custom_gt_root",
            "type": str,
            "default": "data_gleam/gt",
            "help": "Root directory containing custom per-scene GT files.",
        },
        {
            "name": "--custom_urdf_root",
            "type": str,
            "default": "data_gleam/urdf",
            "help": "Root directory containing custom URDF scene assets.",
        },
        {
            "name": "--custom_scene_prefixes",
            "type": str,
            "default": None,
            "help": "Comma-separated scene prefixes. If omitted, auto-discover from custom_gt_root.",
        },
        {
            "name": "--custom_grid_size",
            "type": int,
            "default": 128,
        },
        {
            "name": "--custom_motion_height",
            "type": float,
            "default": None,
        },
        {
            "name": "--drone_height",
            "type": float,
            "default": None,
            "help": "Alias of --custom_motion_height for the evaluation flight height.",
        },
        {
            "name": "--custom_num_eval_round",
            "type": int,
            "default": 3,
        },
        {
            "name": "--custom_save_path",
            "type": str,
            "default": None,
        },
        {
            "name": "--custom_sampled_init_root",
            "type": str,
            "default": "/home/ghr/fs/Junyi/proj/baselines/sampled_inits",
            "help": "Directory containing per-scene sampled init TSV files named like <scene>_xyz_yaw.tsv.",
        },
        {
            "name": "--custom_debug_obs",
            "type": bool,
            "default": False,
            "help": "Dump per-step custom evaluation observations for one env.",
        },
        {
            "name": "--custom_debug_obs_env_idx",
            "type": int,
            "default": 0,
            "help": "Vectorized env index to dump when custom_debug_obs is enabled.",
        },
        {
            "name": "--custom_debug_obs_max_steps",
            "type": int,
            "default": 20,
            "help": "Maximum number of steps to dump per round when custom_debug_obs is enabled.",
        },
        {
            "name": "--custom_debug_obs_save_tensors",
            "type": bool,
            "default": False,
            "help": "Also save raw observation tensors as .pt files.",
        },
        {
            "name": "--custom_debug_obs_depth_vis_max",
            "type": float,
            "default": 10.0,
            "help": "Fixed max depth in meters for debug depth PNG visualization.",
        },
        {
            "name": "--custom_baseline_pose_root",
            "type": str,
            "default": str(DEFAULT_BASELINE_POSE_ROOT),
            "help": "Root directory for exported baseline pose JSON files.",
        },
        {
            "name": "--custom_baseline_pose_method",
            "type": str,
            "default": DEFAULT_BASELINE_POSE_METHOD,
            "help": "Method name used for exported baseline pose JSON files.",
        },
    ]
    reward_args = [
        {
            "name": "--surface_coverage",
            "type": float,
            "default": 1.0,
            "help": "surface coverage ratio"
        },
        {
            "name": "--only_positive_rewards",
            "type": bool,
            "default": False,
            "help": "If true negative total rewards are clipped at zero (avoids early termination problems)"
        },
        {
            "name": "--max_contact_force",
            "type": float,
            "default": 100,
            "help": "Forces above this value are penalized"
        },
    ]

    args = get_args(additional_args + reward_args)
    args.task = "eval_gleam_custom"

    requested_motion_height = args.custom_motion_height
    if args.drone_height is not None:
        if requested_motion_height is not None and args.drone_height != requested_motion_height:
            raise ValueError(
                f"Received conflicting heights: --custom_motion_height={requested_motion_height} "
                f"and --drone_height={args.drone_height}"
            )
        requested_motion_height = args.drone_height

    scene_specs, inferred_motion_height = discover_custom_scene_specs(
        gt_root=args.custom_gt_root,
        urdf_root=args.custom_urdf_root,
        grid_size=args.custom_grid_size,
        motion_height=requested_motion_height,
        sampled_init_root=args.custom_sampled_init_root,
        scene_prefixes=_parse_scene_prefixes(args.custom_scene_prefixes),
    )
    resolved_motion_height = inferred_motion_height if requested_motion_height is None else requested_motion_height
    Env_GLEAM_Custom.configure(
        scene_specs=scene_specs,
        grid_size=args.custom_grid_size,
        motion_height=resolved_motion_height,
        num_eval_round=args.custom_num_eval_round,
        save_path=args.custom_save_path,
        debug_obs=args.custom_debug_obs,
        debug_obs_env_idx=args.custom_debug_obs_env_idx,
        debug_obs_max_steps=args.custom_debug_obs_max_steps,
        debug_obs_save_tensors=args.custom_debug_obs_save_tensors,
        debug_obs_depth_vis_max=args.custom_debug_obs_depth_vis_max,
        baseline_pose_output_root=args.custom_baseline_pose_root,
        baseline_pose_method=args.custom_baseline_pose_method,
    )
    task_registry.register(args.task, Env_GLEAM_Custom, Config_GLEAM_Eval, DroneCfgPPO)

    discovered_scene_count = len(scene_specs)
    requested_num_envs = args.num_envs if args.num_envs is not None else min(discovered_scene_count, 4)
    adjusted_num_envs = _largest_divisor_leq(discovered_scene_count, requested_num_envs)
    if adjusted_num_envs != requested_num_envs:
        print(
            f"[custom] Adjust num_envs from {requested_num_envs} to {adjusted_num_envs} "
            f"so it divides num_scene={discovered_scene_count}."
        )
    args.num_envs = adjusted_num_envs

    print(f"[custom] Discovered {discovered_scene_count} scene(s)")
    print(f"[custom] motion_height={resolved_motion_height}")
    if args.custom_debug_obs:
        debug_root = args.custom_save_path or "./gleam/output/eval_gleam_custom"
        print(
            f"[custom] debug_obs enabled: env_idx={args.custom_debug_obs_env_idx}, "
            f"max_steps={args.custom_debug_obs_max_steps}, "
            f"depth_vis_max={args.custom_debug_obs_depth_vis_max}m, "
            f"output={debug_root}/debug_obs"
        )
    for idx, scene_spec in enumerate(scene_specs):
        print(
            f"  [{idx}] prefix={scene_spec.prefix} "
            f"urdf={scene_spec.urdf_path}"
        )
    print(
        f"[custom] baseline pose export: root={args.custom_baseline_pose_root} "
        f"method={args.custom_baseline_pose_method}"
    )

    use_wandb = False

    ckpt_path = args.ckpt_path or DEFAULT_CKPT_PATH
    args.ckpt_path = ckpt_path
    print(ckpt_path)
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist!"

    exp_name = args.task
    seed = 0
    trial_name = f"{exp_name}_{get_time_str()}" \
        if args.exp_name is None or len(args.exp_name) == 0 \
        else f"{args.exp_name}_{get_time_str()}"
    log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs", trial_name)
    print("[LOGGING] We start logging training data into {}".format(log_dir))

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.visual_input.stack = args.buffer_size
    env_cfg.max_episode_length = 20

    env_eval, env_cfg_eval = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_cfg_dict = {key: value for key, value in env_cfg_eval.__dict__.items()}

    env = EnvWrapperGLEAM(env_eval)

    config = dict(
        algo=dict(
            policy=ActorCriticPolicy_Discrete_Eval,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=Encoder_GLEAM,
                features_extractor_kwargs=dict(
                    encoder_param={
                        "hidden_shapes": [256, 256],
                        "visual_dim": 256
                    },
                    net_param={
                        "transformer_params": [[1, 256], [1, 256]],
                        "append_hidden_shapes": [256, 256]
                    },
                    state_input_shape=(args.buffer_size * 6,),
                    visual_input_shape=(1, 128, 128),
                )
            ),
            env=env,
            learning_rate=args.lr,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=args.target_kl if args.use_target_kl else None,
            max_grad_norm=1,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=0.2,
            vf_coef=args.vf_coeff,
            clip_range_vf=0.2,
            ent_coef=args.ent_coeff,
            tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device=args.sim_device,
        ),
        gpu_simulation=True,
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    callbacks = [
        EvalCallback_GLEAM(
            eval_env=env_eval,
            n_eval_episodes=10000,
            log_path=log_dir,
            eval_freq=1000000000,
            deterministic=False,
            render=True,
            verbose=1,
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=exp_name,
                project_name=project_name,
                config={**config, **env_cfg_dict},
            )
        )
    callbacks = CallbackList(callbacks)

    model = PPO_Grid_Obs(**config["algo"])
    if ckpt_path:
        model.set_parameters(ckpt_path)

    evaluate_policy_grid_obs(
        model=model,
        env=env_eval,
        deterministic=True,
    )
    print(ckpt_path)
    print("Done.")


if __name__ == '__main__':
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()

    main()

    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Total wall-clock time: {:.3f}min".format((t_end - t_start) / 60))
