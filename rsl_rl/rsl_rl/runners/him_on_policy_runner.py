# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # HIMPPO
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.profile_timing = self.cfg.get("profile_timing", False)

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            getattr(self.env, "occupancy_obs_shape", None),
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        occupancy_obs = self.env.get_occupancy_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        occupancy_obs = occupancy_obs.to(self.device) if occupancy_obs is not None else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            termination_debug_names = ()
            termination_debug_counts = None
            termination_debug_total_samples = 0
            collection_timing = None
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    profile_step = self.profile_timing and i == self.num_steps_per_env - 1
                    self.env.set_timing_profile_enabled(profile_step)
                    if profile_step:
                        policy_start = self._timing_stamp()
                    actions = self.alg.act(obs, critic_obs, occupancy_obs)
                    if profile_step:
                        policy_end = self._timing_stamp()
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)
                    if profile_step:
                        env_end = self._timing_stamp()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    occupancy_obs = self.env.get_occupancy_observations()
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    occupancy_obs = occupancy_obs.to(self.device) if occupancy_obs is not None else None
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        termination_debug = infos.get("termination_debug")
                        if termination_debug is not None:
                            termination_debug_total_samples += int(termination_debug["num_envs"])
                            termination_debug_names = termination_debug["term_names"]
                            if termination_debug_counts is None:
                                termination_debug_counts = torch.zeros_like(termination_debug["counts"])
                            termination_debug_counts += termination_debug["counts"]
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    if profile_step:
                        post_end = self._timing_stamp()
                        collection_timing = {
                            "policy_him": (policy_end - policy_start) * 1000.0,
                            "env_step_total": (env_end - policy_end) * 1000.0,
                            "runner_post": (post_end - env_end) * 1000.0,
                        }
                        collection_timing.update(infos.get("step_timing", {}))
                        collection_timing["sample_total"] = (
                            collection_timing["policy_him"]
                            + collection_timing["env_step_total"]
                            + collection_timing["runner_post"]
                        )
                        if "env_internal_total" in collection_timing:
                            collection_timing["env_adapter"] = max(
                                collection_timing["env_step_total"] - collection_timing["env_internal_total"], 0.0
                            )

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, occupancy_obs)
                
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                if key.startswith("Curriculum/"):
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        reward_debug = locs.get("infos", {}).get("reward_debug", None)
        reward_debug_string = ""
        if reward_debug is not None:
            self.writer.add_scalar("RewardDebug/total_mean", reward_debug["total_mean"], locs["it"])
            self.writer.add_scalar("RewardDebug/total_min", reward_debug["total_min"], locs["it"])
            self.writer.add_scalar("RewardDebug/total_max", reward_debug["total_max"], locs["it"])
            terms_by_name = reward_debug.get("terms_by_name", {})
            for name, stats in terms_by_name.items():
                self.writer.add_scalar(f"RewardDebugTerms/{name}", stats["mean"], locs["it"])

            reward_debug_string += f"""{'Reward/total:':>{pad}} {reward_debug['total_mean']:+.4f}\n"""
            for name in reward_debug.get("term_names", terms_by_name.keys()):
                stats = terms_by_name.get(name)
                if stats is not None:
                    reward_debug_string += f"""{('Reward/' + name):>{pad}} {stats['mean']:+.4f}\n"""
            base_height = reward_debug.get("base_height")
            if base_height is not None:
                self.writer.add_scalar("RewardDebug/base_height_mean", base_height["mean"], locs["it"])
                self.writer.add_scalar("RewardDebug/base_height_err_abs_max", base_height["err_abs_max"], locs["it"])
                reward_debug_string += f"""{'Base height raw/err:':>{pad}} {base_height['mean']:+.4f} / {base_height['err_mean']:+.4f}\n"""

        timing_string = ""
        collection_timing = locs.get("collection_timing")
        if collection_timing is not None:
            timing_order = (
                ("sample_total", "Timing/sample_total"),
                ("policy_him", "Timing/policy_him"),
                ("env_step_total", "Timing/env_step_total"),
                ("env_adapter", "Timing/env_adapter"),
                ("action_process", "Timing/action_process"),
                ("physics_decimation", "Timing/physics_decimation"),
                ("termination", "Timing/termination"),
                ("reward", "Timing/reward"),
                ("terminal_obs_reset", "Timing/terminal_obs_reset"),
                ("commands_events_observation", "Timing/commands_events_obs"),
                ("runner_post", "Timing/runner_post"),
            )
            timing_string += f"""{'Timing sample (one step):':>{pad}} synchronized\n"""
            for key, label in timing_order:
                if key in collection_timing:
                    value = collection_timing[key]
                    self.writer.add_scalar(f"Timing/{key}_ms", value, locs["it"])
                    timing_string += f"""{(label + ':'):>{pad}} {value:8.3f} ms\n"""

        termination_debug_string = ""
        termination_debug_total_samples = locs.get("termination_debug_total_samples", 0)
        termination_debug_counts = locs.get("termination_debug_counts")
        if termination_debug_total_samples > 0 and termination_debug_counts is not None:
            counts = termination_debug_counts.tolist()
            termination_debug_total_resets = counts[0]
            termination_debug_accum = dict(zip(locs.get("termination_debug_names", ()), counts[1:]))
            reset_ratio = termination_debug_total_resets / termination_debug_total_samples
            self.writer.add_scalar("Termination/reset_ratio", reset_ratio, locs["it"])
            termination_debug_string += f"""{'Termination/reset_rate:':>{pad}} {reset_ratio:.4f}\n"""
            for name in termination_debug_accum:
                count = termination_debug_accum.get(name, 0)
                per_step_env = count / termination_debug_total_samples
                per_reset = count / max(termination_debug_total_resets, 1)
                self.writer.add_scalar(f"TerminationTerms/{name}_per_step_env", per_step_env, locs["it"])
                self.writer.add_scalar(f"TerminationTerms/{name}_per_reset", per_reset, locs["it"])
                termination_debug_string += f"""{('Termination/' + name):>{pad}} {per_reset:.4f}\n"""

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        reward_summary_string = reward_debug_string
        if len(locs['rewbuffer']) > 0:
            reward_summary_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        summary_string = (f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        summary_string += termination_debug_string

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{reward_summary_string}"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{timing_string}""")

        log_string += ep_string
        log_string += summary_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def _timing_stamp(self):
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
