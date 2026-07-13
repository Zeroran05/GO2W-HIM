"""Custom action terms for Go2W wheel-legged control."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class Go2WHybridAction(ActionTerm):
    """Apply leg actions as position targets and wheel actions as velocity targets."""

    cfg: "Go2WHybridActionCfg"
    _asset: Articulation

    def __init__(self, cfg: Go2WHybridActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            self.cfg.wheel_joint_names, preserve_order=self.cfg.preserve_order
        )
        wheel_id_set = set(int(i) for i in self._wheel_ids)
        self._leg_ids = [int(i) for i in self._joint_ids if int(i) not in wheel_id_set]
        self._leg_action_ids = [
            action_id for action_id, joint_id in enumerate(self._joint_ids) if int(joint_id) not in wheel_id_set
        ]
        self._wheel_action_ids = [
            action_id for action_id, joint_id in enumerate(self._joint_ids) if int(joint_id) in wheel_id_set
        ]
        self._leg_action_scale = self._resolve_leg_action_scale()
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._last_actions = torch.zeros_like(self._raw_actions)
        self._delayed_actions = torch.zeros(self.num_envs, self.cfg.decimation, self.action_dim, device=self.device)
        self._delay_substep = 0

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _resolve_leg_action_scale(self) -> torch.Tensor:
        leg_names = [self._joint_names[action_id] for action_id in self._leg_action_ids]
        if isinstance(self.cfg.action_scale, dict):
            scales = []
            for joint_name in leg_names:
                matched_scale = None
                for pattern, scale in self.cfg.action_scale.items():
                    if re.fullmatch(pattern, joint_name):
                        matched_scale = float(scale)
                        break
                if matched_scale is None:
                    raise ValueError(f"No action_scale pattern matched joint '{joint_name}'.")
                scales.append(matched_scale)
            return torch.tensor(scales, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.full((1, len(leg_names)), float(self.cfg.action_scale), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._delay_substep = 0
        if self.cfg.enable_delay:
            delay_steps = torch.randint(0, self.cfg.decimation, (self.num_envs, 1), device=self.device)
            for substep in range(self.cfg.decimation):
                switched = (substep >= delay_steps).to(actions.dtype)
                self._delayed_actions[:, substep] = self._last_actions + (actions - self._last_actions) * switched
        else:
            self._delayed_actions[:] = actions.unsqueeze(1)

    def apply_actions(self):
        if self.cfg.enable_delay:
            substep = min(self._delay_substep, self.cfg.decimation - 1)
            self._processed_actions[:] = self._delayed_actions[:, substep]
        else:
            self._processed_actions[:] = self._raw_actions

        if self._leg_ids:
            default_pos = self._asset.data.default_joint_pos[:, self._leg_ids]
            leg_actions = self._processed_actions[:, self._leg_action_ids]
            leg_targets = default_pos + leg_actions * self._leg_action_scale
            self._asset.set_joint_position_target(leg_targets, joint_ids=self._leg_ids)
        if self._wheel_ids:
            wheel_actions = self._processed_actions[:, self._wheel_action_ids]
            wheel_targets = wheel_actions * self.cfg.vel_scale
            self._asset.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_ids)

        self._delay_substep += 1
        if self._delay_substep >= self.cfg.decimation:
            self._last_actions[:] = self._raw_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._last_actions[:] = 0.0
            self._delayed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._last_actions[env_ids] = 0.0
            self._delayed_actions[env_ids] = 0.0


@configclass
class Go2WHybridActionCfg(ActionTermCfg):
    """Configuration for the 16-D Go2W hybrid leg-position and wheel-velocity action."""

    class_type: type[ActionTerm] = Go2WHybridAction
    joint_names: list[str] = None
    wheel_joint_names: list[str] = None
    action_scale: float | dict[str, float] = 0.25
    vel_scale: float = 5.0
    decimation: int = 4
    enable_delay: bool = True
    preserve_order: bool = True
