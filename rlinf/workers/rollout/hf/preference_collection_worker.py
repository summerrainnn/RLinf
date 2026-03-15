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

"""Rollout worker extension for preference data collection.

During collection the worker runs eval-style rollouts.  Video recording is
handled by the :class:`~rlinf.envs.wrappers.record_video.RecordVideo` wrapper
in per-env mode, so this worker only needs to extract the task description.
"""

from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig

from rlinf.data.preference_data import EpisodeRecord
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _extract_task_description(obs: dict[str, Any], env_idx: int) -> str:
    """Extract task description string for *env_idx*."""
    td = obs.get("task_descriptions", None)
    if td is None:
        return ""
    if isinstance(td, (list, tuple)):
        return str(td[env_idx]) if env_idx < len(td) else ""
    return str(td)


class PreferenceCollectionWorker(MultiStepRolloutWorker):
    """Rollout worker for preference learning that delegates video recording
    to the RecordVideo wrapper's per-env mode.

    Only extracts task descriptions during rollout — no keyframe sampling.

    Communication protocol
    ~~~~~~~~~~~~~~~~~~~~~~
    Exactly mirrors the existing ``evaluate`` loop::

        env sends:  n_stages initial_obs  +  (n_steps - 1) * n_stages step_obs
        rollout:    n_steps * n_stages receives  (first = initial, last = near-final)

    Args:
        cfg: Full Hydra config.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    async def collect_episodes_epoch(
        self, input_channel: Channel, output_channel: Channel
    ) -> list[dict[str, Any]]:
        """Run one epoch of eval-style rollout and return per-env task descriptions.

        Returns a list with one dict per environment containing:

        * ``task_description``: str

        Args:
            input_channel: Channel carrying env → rollout env outputs.
            output_channel: Channel carrying rollout → env action chunks.

        Returns:
            List of dicts with task descriptions, one per environment.
        """
        if self.enable_offload:
            self.reload_model()

        total_envs = self.eval_batch_size * self.num_pipeline_stages
        n_steps = self.n_eval_chunk_steps

        task_desc_arr: list[str] = [""] * total_envs

        for step in range(n_steps):
            env_offset = 0
            for _stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel, mode="eval")

                obs = env_output["obs"]
                actions, _ = self.predict(obs, mode="eval")
                self.send_chunk_actions(output_channel, actions, mode="eval")

                n = self.eval_batch_size
                # Capture task description from the first observation only
                if step == 0:
                    for local_idx in range(n):
                        global_idx = env_offset + local_idx
                        task_desc_arr[global_idx] = _extract_task_description(obs, local_idx)

                env_offset += n

        if self.enable_offload:
            self.offload_model()

        return [
            {"task_description": task_desc_arr[i]}
            for i in range(total_envs)
        ]


def build_episode_records(
    task_descriptions: list[str],
    video_paths: list[str],
    rewards: np.ndarray,
    successes: np.ndarray,
    lengths: np.ndarray,
) -> list[EpisodeRecord]:
    """Combine task descriptions and video paths with env metrics into EpisodeRecords.

    Args:
        task_descriptions: Per-env task description strings.
        video_paths: Per-env mp4 video file paths.
        rewards: Per-env cumulative rewards [N_envs].
        successes: Per-env success flags [N_envs].
        lengths: Per-env episode step counts [N_envs].

    Returns:
        List of :class:`~rlinf.data.preference_data.EpisodeRecord`, one per env.
    """
    records: list[EpisodeRecord] = []
    for td, vp, rew, suc, ln in zip(task_descriptions, video_paths, rewards, successes, lengths):
        records.append(
            EpisodeRecord(
                task_description=td,
                cumulative_reward=float(rew),
                success=bool(suc),
                episode_length=int(ln),
                video_path=vp if vp else None,
            )
        )
    return records
