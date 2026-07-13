#!/usr/bin/env python3
"""Play a trained RobotLab-style Go2W HIM policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for package_path in (PROJECT_ROOT / "rsl_rl", PROJECT_ROOT / "source" / "robotlab_go2w_him"):
    sys.path.insert(0, str(package_path))

parser = argparse.ArgumentParser(description="Play a trained HIMLoco Go2W checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model_*.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=None, help="Override the play environment count.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run at the environment control rate.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Control the velocity command by keyboard.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import robotlab_go2w_him.tasks.locomotion.velocity.config.go2w  # noqa: F401
from robotlab_go2w_him.rsl_rl_compat import HimLocoRslRlVecEnvAdapter
from robotlab_go2w_him.tasks.locomotion.velocity.config.go2w.agents.him_ppo_cfg import (
    get_go2w_him_train_cfg,
)
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import HIMOnPolicyRunner


TASK_NAME = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-Play-v0"


def main():
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_cfg = get_go2w_him_train_cfg()
    # The runner only needs one storage slot for inference.
    train_cfg["runner"]["num_steps_per_env"] = 1

    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    keyboard = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
                sim_device=env_cfg.sim.device,
            )
        )

        def keyboard_velocity_command(env):
            command = keyboard.advance().unsqueeze(0).expand(env.num_envs, -1)
            return command * command.new_tensor((2.0, 2.0, 0.25))

        env_cfg.observations.policy.velocity_commands = ObsTerm(func=keyboard_velocity_command)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-3.5, -4.0, 2.2)
        env_cfg.viewer.lookat = (0.6, 0.0, 0.45)

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env = HimLocoRslRlVecEnvAdapter(env, use_occupancy=train_cfg.get("use_occupancy", False))

    runner = HIMOnPolicyRunner(env, train_cfg, log_dir=None, device=args_cli.device or env.device)
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    occupancy_obs = env.get_occupancy_observations()
    step_dt = env.unwrapped.step_dt

    print(f"[INFO] Task: {TASK_NAME}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Environments: {env.num_envs}")
    if keyboard is not None:
        print(keyboard)

    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                if keyboard is not None:
                    command = keyboard.advance()
                    command_term = env.unwrapped.command_manager.get_term("base_velocity")
                    command_term.vel_command_b[:] = command
                actions = policy(obs, occupancy_obs)
                obs, _, _, _, _, _, _ = env.step(actions)
                occupancy_obs = env.get_occupancy_observations()

            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
