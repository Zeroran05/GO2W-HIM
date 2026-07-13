"""Compatibility helpers for running the original HIMLoco RSL-RL code on Isaac Lab."""

from __future__ import annotations

import torch


class HimLocoRslRlVecEnvAdapter:
    """Expose an Isaac Lab Gymnasium env through the original `rsl_rl.env.VecEnv` interface."""

    def __init__(
        self,
        env,
        clip_actions: float = 100.0,
        clip_observations: float = 100.0,
        num_one_step_obs: int = 57,
        num_one_step_privileged_obs: int = 262,
        use_occupancy: bool = False,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.max_episode_length = self.unwrapped.max_episode_length
        self.episode_length_buf = self.unwrapped.episode_length_buf
        self.extras = self.unwrapped.extras
        self.clip_actions = clip_actions
        self.clip_observations = clip_observations
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_privileged_obs = num_one_step_privileged_obs
        self.use_occupancy = use_occupancy

        self.obs_buf = None
        self.privileged_obs_buf = None
        self.occupancy_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_obs = num_one_step_obs * 6
        self.num_privileged_obs = num_one_step_privileged_obs
        self.occupancy_obs_shape = (18, 25, 16) if self.use_occupancy else None

    def reset(self, env_ids=None):
        if env_ids is not None:
            raise NotImplementedError("Partial reset is managed by Isaac Lab during step().")
        obs, _ = self.env.reset()
        self._cache_obs(obs)
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated
        self._cache_obs(obs)
        self.rew_buf = rewards
        self.reset_buf = dones
        if torch.any(dones) and "log" in infos and "episode" not in infos:
            infos["episode"] = infos["log"]
        elif "episode" in infos:
            infos.pop("episode")
        self.extras = infos

        termination_ids = infos.get("termination_ids")
        if termination_ids is None:
            termination_ids = dones.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = infos.get("termination_privileged_obs")
        if termination_privileged_obs is None or termination_privileged_obs.numel() == 0:
            termination_privileged_obs = self.privileged_obs_buf[termination_ids]

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
        )

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        return self.privileged_obs_buf

    def get_occupancy_observations(self) -> torch.Tensor | None:
        return self.occupancy_obs_buf

    def set_timing_profile_enabled(self, enabled: bool) -> None:
        """Enable synchronized timing for the next environment step."""
        self.unwrapped._profile_step_timing = enabled

    def close(self):
        self.env.close()

    def _cache_obs(self, obs):
        if isinstance(obs, dict):
            policy_obs = obs["policy"]
            critic_obs = obs.get("critic", policy_obs)
            occupancy_obs = obs.get("occ", None) if self.use_occupancy else None
            if isinstance(occupancy_obs, dict):
                occupancy_obs = occupancy_obs.get("occupancy", None)
        else:
            policy_obs = obs
            critic_obs = obs
            occupancy_obs = None
        self.obs_buf = torch.clip(policy_obs, -self.clip_observations, self.clip_observations)
        self.privileged_obs_buf = torch.clip(critic_obs, -self.clip_observations, self.clip_observations)
        expected_obs_dim = self.num_one_step_obs * 6
        if self.obs_buf.shape[-1] != expected_obs_dim:
            raise ValueError(f"HIM actor observation must be {expected_obs_dim}-D, got {self.obs_buf.shape[-1]}.")
        if self.privileged_obs_buf.shape[-1] != self.num_one_step_privileged_obs:
            raise ValueError(
                f"HIM critic observation must be {self.num_one_step_privileged_obs}-D, "
                f"got {self.privileged_obs_buf.shape[-1]}."
            )
        self.occupancy_obs_buf = occupancy_obs.to(torch.int8) if occupancy_obs is not None else None
        self.num_obs = self.obs_buf.shape[-1]
        self.num_privileged_obs = self.privileged_obs_buf.shape[-1]
        if self.use_occupancy and self.occupancy_obs_buf is not None:
            self.occupancy_obs_shape = tuple(self.occupancy_obs_buf.shape[1:])
