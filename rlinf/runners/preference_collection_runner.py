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

"""Runner for collecting preference data by rolling out a trained policy.

Runs multiple evaluation epochs, records per-episode observations and
environment metrics, creates preference pairs (success as chosen, failure
as rejected), and saves the dataset to disk.

Usage example::

    python collect_preference_data.py --config-name maniskill_collect_preference_openpi
"""

import os
import typing
from datetime import datetime

import numpy as np

from rlinf.data.preference_data import (
    EpisodeRecord,
    PreferencePair,
    create_preference_pairs,
    save_preference_pairs,
    summarize_preference_dataset,
)
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger
from rlinf.workers.rollout.hf.preference_collection_worker import build_episode_records

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.preference_env_worker import PreferenceEnvWorker
    from rlinf.workers.rollout.hf.preference_collection_worker import (
        PreferenceCollectionWorker,
    )


class PreferenceCollectionRunner:
    """Orchestrates preference data collection over multiple rollout epochs.

    In each epoch:

    1. Both ``env_worker.collect_preference_epoch()`` and
       ``rollout_worker.collect_episodes_epoch()`` are launched in parallel
       and communicate via channels (same pattern as ``EmbodiedEvalRunner``).
    2. Episode records are assembled from the combined env + rollout outputs.
    3. After all epochs complete, episodes are paired into preference pairs and
       saved to ``output_path``.

    Args:
        cfg: Full Hydra config.  Preference-specific keys are read from
            ``cfg.preference``:

            * ``num_collection_epochs`` – number of evaluation epochs to run.
            * ``output_path`` – where to save the ``.pkl`` preference dataset.
            * ``min_reward_diff`` – minimum reward gap to keep a pair (default 0).

        rollout: :class:`~rlinf.workers.rollout.hf.preference_collection_worker.PreferenceCollectionWorker`
            worker group.
        env: :class:`~rlinf.workers.env.preference_env_worker.PreferenceEnvWorker`
            worker group.
    """

    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "PreferenceCollectionWorker",
        env: "PreferenceEnvWorker",
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        self.logger = get_logger()

        pref_cfg = cfg.get("preference", None)
        if pref_cfg is not None:
            self.num_collection_epochs: int = int(pref_cfg.num_collection_epochs)
            self.output_path: str = str(pref_cfg.output_path)
            self.min_reward_diff: float = float(pref_cfg.get("min_reward_diff", 0.0))
        else:
            # Defaults used when this runner is embedded inside another runner
            # (e.g. RewardModelEvalRunner) and no dedicated preference config is present.
            self.num_collection_epochs = 1
            self.output_path = ""
            self.min_reward_diff = 0.0

        # Precompute rollout parameters.
        self._n_total_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )

    def init_workers(self) -> None:
        """Initialize rollout and env workers (no actor needed)."""
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        ckpt_path = self.cfg.runner.get("ckpt_path", None)
        if ckpt_path:
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            # Rollout worker load is done inside init_worker via cfg.runner.ckpt_path

    def _run_one_epoch(self) -> list[EpisodeRecord]:
        """Run one collection epoch and return a flat list of EpisodeRecords."""
        # Launch env and rollout workers in parallel (they communicate via channels)
        env_handle: Handle = self.env.collect_preference_epoch(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.collect_episodes_epoch(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )

        # Wait for both
        env_results_list = env_handle.wait()  # list of dicts (one per Ray actor)
        raw_obs_list = rollout_handle.wait()  # list of lists  (one per Ray actor)

        # Combine across multiple Ray actors
        all_records: list[EpisodeRecord] = []
        for env_result, raw_obs in zip(env_results_list, raw_obs_list):
            if env_result is None or raw_obs is None:
                continue

            # Extract task descriptions from rollout worker output
            task_descriptions = [ep["task_description"] for ep in raw_obs]

            # Get video paths from env worker (may be empty if per_env_video is off)
            video_paths = env_result.get("video_paths", [])
            # Pad with empty strings if video_paths is shorter than task_descriptions
            if len(video_paths) < len(task_descriptions):
                video_paths = list(video_paths) + [""] * (len(task_descriptions) - len(video_paths))

            records = build_episode_records(
                task_descriptions=task_descriptions,
                video_paths=video_paths,
                rewards=env_result["rewards"],
                successes=env_result["successes"],
                lengths=env_result["lengths"],
            )
            all_records.extend(records)
        return all_records

    def _build_output_path(self) -> str:
        """Add a timestamp directory to the output path.

        Given ``/some/dir/data.pkl``, produces
        ``/some/dir/{YYYYMMDD-HH:MM:SS}/data.pkl`` so that successive runs do
        not overwrite each other.
        """
        if not self.output_path:
            return self.output_path
        parent = os.path.dirname(os.path.abspath(self.output_path))
        basename = os.path.basename(self.output_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        return os.path.join(parent, timestamp, basename)

    def run(self) -> None:
        """Run all collection epochs, create preference pairs, and save dataset."""
        all_episodes: list[EpisodeRecord] = []

        self.logger.info(
            f"Starting preference data collection: {self.num_collection_epochs} epochs."
        )

        for epoch in range(self.num_collection_epochs):
            records = self._run_one_epoch()
            all_episodes.extend(records)
            n_success = sum(1 for r in records if r.success)
            n_fail = sum(1 for r in records if not r.success)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_collection_epochs}: "
                f"collected {len(records)} episodes "
                f"({n_success} success, {n_fail} failure). "
                f"Total so far: {len(all_episodes)}."
            )

        self.logger.info(
            f"Total episodes collected: {len(all_episodes)}.  Creating preference pairs …"
        )

        pairs: list[PreferencePair] = create_preference_pairs(
            all_episodes, min_reward_diff=self.min_reward_diff
        )

        summary = summarize_preference_dataset(pairs)
        self.logger.info(f"Preference dataset summary: {summary}")

        output_path = self._build_output_path()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_preference_pairs(pairs, output_path)
        self.logger.info(f"Saved {len(pairs)} preference pairs to {output_path}")
