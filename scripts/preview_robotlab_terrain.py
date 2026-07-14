#!/usr/bin/env python3
"""Preview the full RobotLab Go2W training terrain without spawning robots."""

from __future__ import annotations

import argparse
import copy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the full RobotLab Go2W training terrain.")
parser.add_argument("--rows", type=int, default=10, help="Number of terrain curriculum rows to preview.")
parser.add_argument("--cols", type=int, default=20, help="Number of terrain type columns to preview.")
parser.add_argument("--border-width", type=float, default=20.0, help="Outer terrain border width in meters.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


def _column_ranges(terrain_gen_cfg) -> list[tuple[str, int, int, float]]:
    """Return the inclusive column range assigned to every configured sub-terrain."""
    proportions = [cfg.proportion for cfg in terrain_gen_cfg.sub_terrains.values()]
    total = sum(proportions)
    cumulative = []
    value = 0.0
    for proportion in proportions:
        value += proportion / total
        cumulative.append(value)

    columns_by_name = {name: [] for name in terrain_gen_cfg.sub_terrains}
    names = tuple(terrain_gen_cfg.sub_terrains)
    for column in range(terrain_gen_cfg.num_cols):
        sample = column / terrain_gen_cfg.num_cols + 0.001
        terrain_index = next(index for index, upper_bound in enumerate(cumulative) if sample < upper_bound)
        columns_by_name[names[terrain_index]].append(column)

    return [
        (name, columns[0], columns[-1], terrain_gen_cfg.sub_terrains[name].proportion / total)
        for name, columns in columns_by_name.items()
        if columns
    ]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    terrain_gen_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrain_gen_cfg.num_rows = args_cli.rows
    terrain_gen_cfg.num_cols = args_cli.cols
    terrain_gen_cfg.border_width = args_cli.border_width
    terrain_gen_cfg.curriculum = True

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.34, 0.31), roughness=0.75),
        debug_vis=False,
    )
    terrain_cfg.class_type(terrain_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1200.0)
    light_cfg.func("/World/skyLight", light_cfg)

    terrain_width_x = terrain_gen_cfg.size[0] * terrain_gen_cfg.num_rows
    terrain_width_y = terrain_gen_cfg.size[1] * terrain_gen_cfg.num_cols
    terrain_center = [terrain_width_x * 0.5, terrain_width_y * 0.5, 0.0]
    sim.set_camera_view(
        eye=[
            terrain_center[0] + terrain_width_x * 0.45,
            terrain_center[1] + terrain_width_y * 0.35,
            max(terrain_width_x, terrain_width_y) * 0.75,
        ],
        target=terrain_center,
    )
    sim.reset()

    print("[INFO] RobotLab Go2W full training terrain preview.")
    print(f"[INFO] Grid: rows={terrain_gen_cfg.num_rows}, cols={terrain_gen_cfg.num_cols}")
    print("[INFO] Terrain type columns:")
    for name, first_column, last_column, proportion in _column_ranges(terrain_gen_cfg):
        print(f"[INFO]   columns {first_column:02d}-{last_column:02d}: {name} (proportion={proportion:.3f})")
    print("[INFO] Difficulty increases along rows. Move the Isaac Sim viewport camera freely to inspect.")

    while simulation_app.is_running():
        sim.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
