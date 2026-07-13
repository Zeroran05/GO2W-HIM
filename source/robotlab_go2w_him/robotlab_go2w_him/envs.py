"""Environment wrappers for HIMLoco compatibility."""

from __future__ import annotations

import time

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import RayCaster


class HimLocoManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env for the HIMLoco compatibility task."""

    def step(self, action: torch.Tensor):
        profile_timing = getattr(self, "_profile_step_timing", False)
        timing = {}
        stamp = self._timing_stamp if profile_timing else None
        step_start = stamp() if profile_timing else 0.0

        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        if profile_timing:
            now = stamp()
            timing["action_process"] = (now - step_start) * 1000.0
            phase_start = now

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        if profile_timing:
            now = stamp()
            timing["physics_decimation"] = (now - phase_start) * 1000.0
            phase_start = now

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self._update_termination_debug_info()
        if profile_timing:
            now = stamp()
            timing["termination"] = (now - phase_start) * 1000.0
            phase_start = now

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._update_reward_debug_info_if_needed()
        if profile_timing:
            now = stamp()
            timing["reward"] = (now - phase_start) * 1000.0
            phase_start = now

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            terminal_obs = self.observation_manager.compute()
            self.extras["termination_ids"] = reset_env_ids
            if isinstance(terminal_obs, dict) and "critic" in terminal_obs:
                self.extras["termination_privileged_obs"] = terminal_obs["critic"][reset_env_ids].clone()
            elif isinstance(terminal_obs, dict) and "policy" in terminal_obs:
                self.extras["termination_privileged_obs"] = terminal_obs["policy"][reset_env_ids].clone()
            else:
                self.extras["termination_privileged_obs"] = terminal_obs[reset_env_ids].clone()
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            self.extras["termination_ids"] = torch.empty(0, dtype=torch.long, device=self.device)
            self.extras["termination_privileged_obs"] = torch.empty(0, 0, device=self.device)
        if profile_timing:
            now = stamp()
            timing["terminal_obs_reset"] = (now - phase_start) * 1000.0
            phase_start = now

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if hasattr(self, "disturbance_force_b"):
            self.disturbance_force_b.zero_()
        if profile_timing:
            now = stamp()
            timing["commands_events_observation"] = (now - phase_start) * 1000.0
            timing["env_internal_total"] = (now - step_start) * 1000.0
            self.extras["step_timing"] = timing
        else:
            self.extras.pop("step_timing", None)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _timing_stamp(self) -> float:
        """Synchronize only on an explicitly requested profiling step."""
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _update_termination_debug_info(self):
        term_dones = self.termination_manager._term_dones
        term_counts = term_dones.sum(dim=0)
        self.extras["termination_debug"] = {
            "num_envs": self.num_envs,
            # Keep counters on the GPU during rollout. The runner synchronizes
            # them once per iteration when it formats the log output.
            "term_names": tuple(self.termination_manager.active_terms),
            "counts": torch.cat((self.reset_buf.sum().view(1), term_counts)),
        }

    def _update_reward_debug_info_if_needed(self):
        self.extras.pop("reward_debug", None)
        if not getattr(self.cfg, "debug_reward_terms", False):
            return

        interval = max(int(getattr(self.cfg, "debug_reward_interval", 100)), 1)
        if self.common_step_counter % interval != 0:
            return

        step_reward = self.reward_manager._step_reward.detach()
        names = self.reward_manager._term_names
        mean_total = self.reward_buf.mean().item()
        min_total = self.reward_buf.min().item()
        max_total = self.reward_buf.max().item()

        rows = []
        terms_by_name = {}
        for idx, name in enumerate(names):
            values = step_reward[:, idx]
            mean_value = values.mean().item()
            min_value = values.min().item()
            max_value = values.max().item()
            max_abs = values.abs().max().item()
            terms_by_name[name] = {
                "mean": mean_value,
                "min": min_value,
                "max": max_value,
                "max_abs": max_abs,
            }
            rows.append(
                (
                    max_abs,
                    name,
                    mean_value,
                    min_value,
                    max_value,
                )
            )
        rows.sort(reverse=True)
        top_k = min(int(getattr(self.cfg, "debug_reward_top_k", 8)), len(rows))

        self.extras["reward_debug"] = {
            "step": self.common_step_counter,
            "total_mean": mean_total,
            "total_min": min_total,
            "total_max": max_total,
            "term_names": list(names),
            "terms": rows[:top_k],
            "terms_by_name": terms_by_name,
            "base_height": self._get_base_height_debug_if_available(),
        }

    def _get_base_height_debug_if_available(self):
        try:
            base_height_cfg = self.reward_manager.get_term_cfg("base_height")
        except ValueError:
            return None

        sensor_cfg = base_height_cfg.params.get("sensor_cfg")
        target_height = float(base_height_cfg.params.get("target_height", 0.0))
        robot = self.scene["robot"]
        if sensor_cfg is None:
            base_height = robot.data.root_pos_w[:, 2]
        else:
            sensor: RayCaster = self.scene.sensors[sensor_cfg.name]
            base_height = torch.mean(robot.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2], dim=1)

        error = base_height - target_height
        return {
            "mean": base_height.mean().item(),
            "min": base_height.min().item(),
            "max": base_height.max().item(),
            "err_mean": error.mean().item(),
            "err_abs_max": error.abs().max().item(),
        }
