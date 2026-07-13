#!/usr/bin/env python3
"""Train RobotLab-style Go2W locomotion with the HIM runner."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for package_path in (PROJECT_ROOT / "rsl_rl", PROJECT_ROOT / "source" / "robotlab_go2w_him"):
    sys.path.insert(0, str(package_path))

parser = argparse.ArgumentParser(description="Train HIMLoco Go2W in Isaac Lab.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--num_steps_per_env", type=int, default=None)
parser.add_argument("--num_mini_batches", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument(
    "--profile_timing",
    action="store_true",
    default=False,
    help="Print a synchronized module timing sample once per training iteration.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import robotlab_go2w_him.tasks.locomotion.velocity.config.go2w  # noqa: F401
from robotlab_go2w_him.rsl_rl_compat import HimLocoRslRlVecEnvAdapter
from robotlab_go2w_him.tasks.locomotion.velocity.config.go2w.agents.him_ppo_cfg import get_go2w_him_train_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import HIMOnPolicyRunner


VIDEO_TERRAIN_NAME = "pyramid_stairs_inv"


def _find_terrain_column(terrain_generator, terrain_name: str) -> int:
    """Return a representative column generated for the requested sub-terrain."""
    if terrain_generator is None:
        raise ValueError("Video terrain selection requires a terrain generator.")

    sub_terrains = terrain_generator.sub_terrains
    if terrain_name not in sub_terrains:
        available = ", ".join(sub_terrains)
        raise ValueError(f"Unknown video terrain '{terrain_name}'. Available terrains: {available}")

    proportions = [sub_cfg.proportion for sub_cfg in sub_terrains.values()]
    total_proportion = sum(proportions)
    if total_proportion <= 0.0:
        raise ValueError("Terrain proportions must sum to a positive value.")

    target_index = tuple(sub_terrains).index(terrain_name)
    cumulative_proportions = []
    cumulative = 0.0
    for proportion in proportions:
        cumulative += proportion / total_proportion
        cumulative_proportions.append(cumulative)

    matching_columns = []
    for column in range(terrain_generator.num_cols):
        sample = column / terrain_generator.num_cols + 0.001
        generated_index = next(
            index for index, upper_bound in enumerate(cumulative_proportions) if sample < upper_bound
        )
        if generated_index == target_index:
            matching_columns.append(column)

    if not matching_columns:
        raise ValueError(
            f"Terrain '{terrain_name}' is not represented by any of the {terrain_generator.num_cols} columns."
        )
    return matching_columns[len(matching_columns) // 2]


def main():
    train_cfg = get_go2w_him_train_cfg()
    if args_cli.seed is not None:
        train_cfg["seed"] = args_cli.seed
    if args_cli.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        train_cfg["runner"]["num_steps_per_env"] = args_cli.num_steps_per_env
    if args_cli.num_mini_batches is not None:
        train_cfg["algorithm"]["num_mini_batches"] = args_cli.num_mini_batches
    if args_cli.run_name:
        train_cfg["runner"]["run_name"] = args_cli.run_name
    train_cfg["runner"]["resume"] = args_cli.resume
    train_cfg["runner"]["resume_path"] = args_cli.resume_path
    train_cfg["runner"]["profile_timing"] = args_cli.profile_timing

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = train_cfg["seed"]
    env_cfg.debug_reward_interval = train_cfg["runner"]["num_steps_per_env"]
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.video:
        num_envs = env_cfg.scene.num_envs
        terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        inverted_stair_col = _find_terrain_column(terrain_generator, VIDEO_TERRAIN_NAME)
        video_env_index = min(
            int((inverted_stair_col + 0.5) * num_envs / terrain_generator.num_cols), num_envs - 1
        )
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = video_env_index
        # Follow the robot from a high, downward-looking angle so the surrounding
        # inverted staircase remains the dominant feature in the recording.
        env_cfg.viewer.eye = (-4.5, -4.5, 5.5)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.4)

    torch.manual_seed(train_cfg["seed"])

    experiment_name = train_cfg["runner"]["experiment_name"]
    log_root_path = os.path.abspath(os.path.join("logs", "him_rsl_rl", experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if train_cfg["runner"]["run_name"]:
        log_dir += f"_{train_cfg['runner']['run_name']}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print(f"[INFO] Video folder: {video_kwargs['video_folder']}")
        print(f"[INFO] Video interval: {args_cli.video_interval} steps")
        print(f"[INFO] Video length: {args_cli.video_length} steps")
        print(
            f"[INFO] Video camera tracks env {env_cfg.viewer.env_index} "
            f"on inverted-pyramid down-stairs terrain column {inverted_stair_col}."
        )
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = HimLocoRslRlVecEnvAdapter(env, use_occupancy=train_cfg.get("use_occupancy", False))

    runner = HIMOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args_cli.device or env.device)
    if train_cfg["runner"]["resume"] and train_cfg["runner"]["resume_path"] is not None:
        runner.load(train_cfg["runner"]["resume_path"])
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"])
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
