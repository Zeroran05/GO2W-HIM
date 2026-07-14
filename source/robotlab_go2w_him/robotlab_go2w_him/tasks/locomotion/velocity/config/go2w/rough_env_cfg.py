# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robotlab_go2w_him.tasks.locomotion.velocity.mdp as mdp
from robotlab_go2w_him.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robotlab_go2w_him.assets import ROBOTLAB_GO2W_CFG  # isort: skip


# Single switch for the 3D occupancy scanner, observation group, and HIM encoder.
USE_OCCUPANCY = False


GO2W_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
]
GO2W_LEG_JOINT_NAMES = [name for name in GO2W_JOINT_NAMES if "foot" not in name]
GO2W_WHEEL_JOINT_NAMES = [name for name in GO2W_JOINT_NAMES if "foot" in name]
GO2W_FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class UnitreeGo2WHIMActionsCfg:
    """Action specifications for the MDP."""

    hybrid = mdp.Go2WHybridActionCfg(
        asset_name="robot",
        joint_names=GO2W_JOINT_NAMES,
        wheel_joint_names=GO2W_WHEEL_JOINT_NAMES,
        action_scale={".*_hip_joint": 0.125, ".*": 0.25},
        vel_scale=5.0,
        decimation=4,
        enable_delay=True,
        preserve_order=True,
    )


@configclass
class UnitreeGo2WHIMObservationsCfg:
    """Exact HIMLoco 57-D actor and 262-D asymmetric critic contracts."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        disturbance = ObsTerm(
            func=mdp.external_force_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        height_scan = ObsTerm(
            func=mdp.height_scan_him,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5, "scale": 5.0},
        )
        contact_forces = ObsTerm(
            func=mdp.contact_forces_normalized,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2W_FOOT_BODY_NAMES, preserve_order=True),
                "force_range": (0.0, 50.0),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OccCfg(ObsGroup):
        occupancy = ObsTerm(
            func=mdp.robot_centric_occupancy,
            params={
                "sensor_cfg": SceneEntityCfg("occupancy_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    occ: OccCfg = OccCfg()


@configclass
class UnitreeGo2WRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class UnitreeGo2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2WHIMActionsCfg = UnitreeGo2WHIMActionsCfg()
    observations: UnitreeGo2WHIMObservationsCfg = UnitreeGo2WHIMObservationsCfg()
    rewards: UnitreeGo2WRewardsCfg = UnitreeGo2WRewardsCfg()

    # Emit per-term reward statistics for the HIM runner's iteration log.
    debug_reward_terms = True
    debug_reward_interval = 48
    debug_reward_top_k = 5

    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = ROBOTLAB_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if not USE_OCCUPANCY:
            self.scene.occupancy_scanner = None
            self.observations.occ = None
        elif self.scene.occupancy_scanner is not None:
            self.scene.occupancy_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
            self.scene.occupancy_scanner.update_period = self.decimation * self.sim.dt

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hip_default_l2.weight = -0.5
        self.rewards.hip_default_l2.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.run_still_no_wheels_l1.weight = -0.05
        self.rewards.run_still_no_wheels_l1.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.05
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.params["force_ratio"] = 3.0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 0.6

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel.func = mdp.command_lin_vel_x_curriculum
        self.curriculum.command_levels_lin_vel.params = {
            "command_name": "base_velocity",
            "reward_name": "track_lin_vel_xy_exp",
            "max_curriculum": 1.5,
            "increment": 0.2,
            "threshold": 0.8,
            "tracking_reward_scale": 1.5,
        }
        self.curriculum.command_levels_ang_vel = None


@configclass
class UnitreeGo2WHIMPlayEnvCfg(UnitreeGo2WRoughEnvCfg):
    """Small deterministic-observation environment for policy evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.disable_zero_weight_rewards()
