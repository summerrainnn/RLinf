# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Env worker extension for preference data collection.

Adds ``collect_preference_epoch`` which runs one evaluation epoch, tracks
per-environment cumulative rewards and success flags, and returns them for use
by the :class:`~rlinf.runners.preference_collection_runner.PreferenceCollectionRunner`.
"""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class PreferenceEnvWorker(EnvWorker):
    """EnvWorker with an extra ``collect_preference_epoch`` method.

    All other behaviour (training ``interact``, standard ``evaluate``) is
    inherited unchanged from :class:`~rlinf.workers.env.env_worker.EnvWorker`.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Preference collection
    # ------------------------------------------------------------------

    def collect_preference_epoch(
        self, input_channel: Channel, output_channel: Channel
    ) -> dict[str, Any]:
        """Run one eval epoch and return per-environment episode metrics.

        Always runs for the full ``n_eval_chunk_steps`` to collect complete
        trajectories (no early exit on task success).  Rewards and lengths
        are accumulated for the entire epoch.

        Args:
            input_channel: Channel carrying rollout→env action chunks.
            output_channel: Channel carrying env→rollout observations.

        Returns:
            Dict with keys:

            * ``"rewards"``  – float32 array [N_envs], cumulative episode reward.
            * ``"successes"``– bool array   [N_envs], True when episode succeeded.
            * ``"lengths"``  – int32 array  [N_envs], episode step count.
            * ``"video_paths"`` – list[str], per-env mp4 video paths
              (only when per_env_video is enabled, otherwise empty list).
        """
        n_envs_total = self.eval_num_envs_per_stage * self.stage_num
        cum_rewards = np.zeros(n_envs_total, dtype=np.float32)
        successes = np.zeros(n_envs_total, dtype=bool)
        lengths = np.zeros(n_envs_total, dtype=np.int32)
        # Track whether success has been recorded for each env to avoid
        # overwriting a True with a later False (transient success states).
        success_recorded = np.zeros(n_envs_total, dtype=bool)

        env_offset = 0
        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].is_start = True
            extracted_obs, infos = self.eval_env_list[stage_id].reset()
            env_output = EnvOutput(
                obs=extracted_obs,
                final_obs=infos.get("final_observation", None),
            )
            self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

        for eval_step in range(self.n_eval_chunk_steps):
            env_offset = 0
            for stage_id in range(self.stage_num):
                n_stage_envs = self.eval_num_envs_per_stage
                raw_chunk_actions = self.recv_chunk_actions(input_channel, mode="eval")

                chunk_actions = prepare_actions(
                    raw_chunk_actions=raw_chunk_actions,
                    env_type=self.cfg.env.eval.env_type,
                    model_type=self.cfg.actor.model.model_type,
                    num_action_chunks=self.cfg.actor.model.num_action_chunks,
                    action_dim=self.cfg.actor.model.action_dim,
                    policy=self.cfg.actor.model.get("policy_setup", None),
                    wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
                )

                obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
                    self.eval_env_list[stage_id].chunk_step(chunk_actions)
                )
                if isinstance(obs_list, (list, tuple)):
                    extracted_obs = obs_list[-1] if obs_list else None
                if isinstance(infos_list, (list, tuple)):
                    infos = infos_list[-1] if infos_list else None

                chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

                # Always accumulate rewards and lengths for the full trajectory
                if isinstance(chunk_rewards, torch.Tensor):
                    ep_rewards = chunk_rewards.sum(dim=-1).cpu().numpy().astype(np.float32)
                else:
                    ep_rewards = np.zeros(n_stage_envs, dtype=np.float32)

                for local_i in range(n_stage_envs):
                    global_i = env_offset + local_i
                    cum_rewards[global_i] += float(ep_rewards[local_i])
                    lengths[global_i] += self.cfg.actor.model.num_action_chunks

                # Capture success flags for newly-done envs (from termination/truncation)
                done_flags = chunk_dones[:, -1].cpu().numpy()
                if done_flags.any():
                    success_tensor = self._extract_success(infos, done_flags)
                    for local_i in range(n_stage_envs):
                        global_i = env_offset + local_i
                        if done_flags[local_i] and not success_recorded[global_i]:
                            success_recorded[global_i] = True
                            successes[global_i] = bool(success_tensor[local_i])

                # Also detect success directly from infos (works even with
                # ignore_terminations=True, where terminations are suppressed)
                self._mark_success_from_infos(
                    infos, success_recorded, successes, env_offset, n_stage_envs
                )

                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos.get("final_observation", None),
                )

                # Send obs for all steps except the last (same protocol as evaluate()).
                if eval_step < self.n_eval_chunk_steps - 1:
                    self.send_env_batch(
                        output_channel, env_output.to_dict(), mode="eval"
                    )

                env_offset += n_stage_envs

        self.finish_rollout(mode="eval")

        # Collect per-env video paths from all stages
        video_paths: list[str] = []
        for stage_id in range(self.stage_num):
            env_wrapper = self.eval_env_list[stage_id]
            if hasattr(env_wrapper, "get_per_env_video_paths"):
                video_paths.extend(env_wrapper.get_per_env_video_paths())

        return {
            "rewards": cum_rewards,
            "successes": successes,
            "lengths": lengths,
            "video_paths": video_paths,
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_success_from_infos(
        infos: dict,
        episode_done: np.ndarray,
        successes: np.ndarray,
        env_offset: int,
        n_stage_envs: int,
    ) -> None:
        """Mark envs as done when success is detected in raw infos.

        With ``ignore_terminations=True``, the termination signal is suppressed
        but ``infos["success"]`` (ManiSkill) or ``infos["episode"]["success_once"]``
        still reports whether the task has been completed.  This helper marks
        those envs as done so the loop can exit early.
        """
        success_flags = None
        # Prefer cumulative "success_once" from episode tracking (survives
        # transient success states, e.g. object placed then displaced)
        if "episode" in infos:
            ep = infos["episode"]
            s = ep.get("success_once", ep.get("success", None))
            if s is not None:
                success_flags = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        # Fallback: top-level "success" (raw ManiSkill info, instantaneous)
        if success_flags is None and "success" in infos:
            s = infos["success"]
            success_flags = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)

        if success_flags is None:
            return

        for local_i in range(min(n_stage_envs, len(success_flags))):
            global_i = env_offset + local_i
            if not episode_done[global_i] and bool(success_flags[local_i]):
                episode_done[global_i] = True
                successes[global_i] = True

    @staticmethod
    def _extract_success(infos: dict, done_flags: np.ndarray) -> np.ndarray:
        """Extract per-env success flags from ``infos`` at episode end.

        Checks both ``"success"`` (maniskill) and ``"s"`` (legacy) key names.
        Falls back to zeros (failure) if neither key is present.
        """
        n = len(done_flags)
        result = np.zeros(n, dtype=bool)
        try:
            # auto_reset=True: success in final_info
            if "final_info" in infos and "episode" in infos["final_info"]:
                ep = infos["final_info"]["episode"]
                s = ep.get("success", ep.get("s", None))
                if s is not None:
                    s_np = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
                    done_indices = np.where(done_flags)[0]
                    for idx_in_done, global_idx in enumerate(done_indices):
                        if idx_in_done < len(s_np):
                            result[global_idx] = bool(s_np[idx_in_done])
                    return result
            # auto_reset=False: success in infos["episode"] or top-level infos
            # ManiSkill stores "success_once" / "success_at_end" in episode dict
            if "episode" in infos:
                ep = infos["episode"]
                s = ep.get("success", ep.get("success_at_end", ep.get("success_once", ep.get("s", None))))
                if s is not None:
                    s_np = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
                    for i in range(min(n, len(s_np))):
                        result[i] = bool(s_np[i])
            # Fallback: check top-level "success" key (ManiSkill raw info)
            if not result.any() and "success" in infos:
                s = infos["success"]
                s_np = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
                for i in range(min(n, len(s_np))):
                    result[i] = bool(s_np[i])
        except Exception:
            pass
        return result
