"""Gym registrations for RobotLab-style Go2W HIM tasks."""

import gymnasium as gym


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0",
    entry_point="robotlab_go2w_him.envs:HimLocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WRoughEnvCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-Play-v0",
    entry_point="robotlab_go2w_him.envs:HimLocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WHIMPlayEnvCfg",
    },
)

