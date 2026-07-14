# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=joint_pos_rel.device)
    wheel_ids = torch.as_tensor(wheel_asset_cfg.joint_ids, device=joint_pos_rel.device)
    joint_pos_rel[:, torch.isin(joint_ids, wheel_ids)] = 0.0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def height_scan_him(env, sensor_cfg: SceneEntityCfg, offset: float = 0.5, scale: float = 5.0) -> torch.Tensor:
    """HIMLoco height convention: clipped base-relative terrain height, then scaled."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    base_z = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(1)
    heights = torch.clamp(base_z - offset - sensor.data.ray_hits_w[..., 2], -1.0, 1.0)
    return heights * scale


def robot_centric_occupancy(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_range: tuple[float, float] = (-0.8, 1.0),
    resolution: float = 0.1,
    shape: tuple[int, int, int] = (18, 25, 16),
) -> torch.Tensor:
    """Build the HIMLoco gravity-aligned local occupancy grid from vertical terrain rays.

    The returned int8 tensor uses [z, x, y] layout. Unknown cells are -1,
    observed free cells are 0, and the terrain surface cell is 1.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    z_bins, x_bins, y_bins = shape
    expected_rays = x_bins * y_bins

    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    if ray_hits_z.shape[1] != expected_rays:
        raise ValueError(
            f"occupancy scanner expected {expected_rays} rays for shape {shape}, got {ray_hits_z.shape[1]}"
        )

    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    surface_z = (ray_hits_z - base_z).reshape(env.num_envs, x_bins, y_bins)
    valid = torch.isfinite(surface_z)

    z_centers = torch.arange(z_bins, device=env.device, dtype=surface_z.dtype) * resolution
    z_centers = (z_centers + z_range[0] + 0.5 * resolution).view(1, z_bins, 1, 1)
    surface = surface_z.unsqueeze(1)
    valid = valid.unsqueeze(1)

    occupancy = torch.full(
        (env.num_envs, z_bins, x_bins, y_bins), -1, dtype=torch.int8, device=env.device
    )
    free = valid & (z_centers > surface + 0.5 * resolution)
    occupancy[free] = 0

    in_grid = valid & (surface >= z_range[0]) & (surface < z_range[1])
    occupied = in_grid & (torch.abs(z_centers - surface) <= 0.5 * resolution)
    occupancy[occupied] = 1

    below_grid = valid & (surface < z_range[0])
    occupancy[below_grid.expand_as(occupancy)] = 0
    return occupancy


def contact_forces_normalized(
    env, sensor_cfg: SceneEntityCfg, force_range: tuple[float, float] = (0.0, 50.0)
) -> torch.Tensor:
    """Flatten four 3-D foot contact forces and map the configured range to [-1, 1]."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].reshape(env.num_envs, -1)
    lower, upper = force_range
    return 2.0 * (forces - lower) / (upper - lower) - 1.0


def external_force_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")) -> torch.Tensor:
    """Expose the simulator external-force buffer to the asymmetric critic."""
    asset: Articulation = env.scene[asset_cfg.name]
    force_buffer = getattr(asset, "_external_force_b", None)
    if force_buffer is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return force_buffer[:, asset_cfg.body_ids, :].reshape(env.num_envs, -1)[:, :3]
