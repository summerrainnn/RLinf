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

"""Reward model evaluation runner.

Validates a trained :class:`~rlinf.models.reward_model.VLMRewardModel` by:

1. Rolling out the policy in the simulator to collect episode pairs
   (identical to :class:`~rlinf.runners.preference_collection_runner`).
2. Scoring both episodes in every pair with the reward model.
3. Comparing RM-implied preference with ground-truth (env-reward-based)
   preference and computing **pair accuracy**.

Config keys (``cfg.reward_model_eval``):
    * ``checkpoint_dir`` – path to a saved reward model checkpoint.
    * ``model_path``     – original VLM path (only needed if backbone was not
                            saved inside checkpoint_dir).
    * ``num_eval_epochs``– number of rollout epochs to collect pairs.
    * ``min_reward_diff``– minimum reward gap to keep a pair.
    * ``batch_size``     – batch size for reward model inference.
    * ``device``         – "auto" | "cuda" | "cpu".
    * ``output_path``    – optional path to save per-pair results as JSON.
"""

from __future__ import annotations

import copy
import json
import os
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from rlinf.data.preference_data import (
    EpisodeRecord,
    PreferencePair,
    create_preference_pairs,
    get_keyframes,
    load_preference_pairs,
)
from rlinf.models.reward_model.vlm_reward_model import build_reward_model
from rlinf.runners.preference_collection_runner import PreferenceCollectionRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.preference_env_worker import PreferenceEnvWorker
    from rlinf.workers.rollout.hf.preference_collection_worker import (
        PreferenceCollectionWorker,
    )

logger = get_logger()


class RewardModelEvalRunner:
    """Evaluates a trained reward model by comparing its preference rankings
    against ground-truth environment rewards on freshly collected trajectory pairs.

    Args:
        cfg: Full Hydra config.
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

        eval_cfg = cfg.reward_model_eval

        self.checkpoint_dir: str = eval_cfg.checkpoint_dir
        self.model_path: str = eval_cfg.get("model_path", eval_cfg.checkpoint_dir)
        self.num_eval_epochs: int = int(eval_cfg.get("num_eval_epochs", 10))
        self.min_reward_diff: float = float(eval_cfg.get("min_reward_diff", 0.0))
        self.batch_size: int = int(eval_cfg.get("batch_size", 8))
        self.output_path: Optional[str] = eval_cfg.get("output_path", None)
        self.n_keyframes: int = int(eval_cfg.get("n_keyframes", 8))

        device_str = eval_cfg.get("device", "auto")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto"
            else torch.device(device_str)
        )

        # The collection sub-runner handles rollout logistics
        self._collection_runner = PreferenceCollectionRunner(
            cfg=cfg, rollout=rollout, env=env
        )

    def init_workers(self) -> None:
        """Initialize env and rollout workers, then load the reward model."""
        self._collection_runner.init_workers()

        logger.info(f"Loading reward model from {self.checkpoint_dir}")
        eval_cfg = self.cfg.reward_model_eval
        use_dummy = bool(eval_cfg.get("use_dummy", False))
        self.reward_model = build_reward_model(
            use_dummy=use_dummy,
            model_path=self.model_path,
            checkpoint_dir=self.checkpoint_dir,
            hidden_dim=int(eval_cfg.get("hidden_dim", 512)),
        )
        if not use_dummy:
            self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_episodes(self, episodes: list[EpisodeRecord]) -> list[float]:
        """Score a list of episodes with the reward model.

        Supports both video-based and legacy keyframe-based episodes.

        Args:
            episodes: Episode records to score.

        Returns:
            Float list of reward model scores (same order as input).
        """
        all_scores: list[float] = []
        for start in range(0, len(episodes), self.batch_size):
            batch = episodes[start : start + self.batch_size]
            keyframes_list = [
                get_keyframes(ep, self.n_keyframes) for ep in batch
            ]
            scores = self.reward_model(
                task_descriptions=[ep.task_description for ep in batch],
                keyframes_list=keyframes_list,
            )
            all_scores.extend(scores.cpu().float().tolist())
        return all_scores

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Collect episodes, score with RM, compute pair accuracy, and report.

        Returns:
            Dict with evaluation metrics.
        """
        # 1. Collect episodes
        logger.info(
            f"Collecting episodes for RM eval: {self.num_eval_epochs} epochs."
        )
        all_episodes: list[EpisodeRecord] = []
        for epoch in range(self.num_eval_epochs):
            records = self._collection_runner._run_one_epoch()
            all_episodes.extend(records)
            logger.info(
                f"Epoch {epoch + 1}/{self.num_eval_epochs}: "
                f"collected {len(records)} episodes (total {len(all_episodes)})."
            )

        # 2. Create ground-truth preference pairs (same pairing strategy)
        pairs: list[PreferencePair] = create_preference_pairs(
            all_episodes, min_reward_diff=self.min_reward_diff
        )
        logger.info(f"Created {len(pairs)} preference pairs for evaluation.")

        if len(pairs) == 0:
            logger.warning("No pairs to evaluate — consider reducing min_reward_diff.")
            return {"pair_accuracy": 0.0, "num_pairs": 0}

        # 3. Score both trajectories in every pair
        chosen_episodes = [p.chosen for p in pairs]
        rejected_episodes = [p.rejected for p in pairs]

        logger.info("Scoring chosen trajectories with the reward model …")
        chosen_scores = self._score_episodes(chosen_episodes)

        logger.info("Scoring rejected trajectories with the reward model …")
        rejected_scores = self._score_episodes(rejected_episodes)

        # 4. Compute agreement accuracy
        #    GT label:  chosen.cumulative_reward >= rejected.cumulative_reward (always True by construction)
        #    RM label:  score_chosen > score_rejected
        correct = 0
        results = []
        for pair, sc, sr in zip(pairs, chosen_scores, rejected_scores):
            rm_prefers_chosen = sc > sr
            correct += int(rm_prefers_chosen)
            results.append(
                {
                    "task": pair.chosen.task_description,
                    "chosen_reward": pair.chosen.cumulative_reward,
                    "rejected_reward": pair.rejected.cumulative_reward,
                    "reward_margin": pair.reward_margin,
                    "chosen_rm_score": sc,
                    "rejected_rm_score": sr,
                    "rm_correct": rm_prefers_chosen,
                    "chosen_success": pair.chosen.success,
                    "rejected_success": pair.rejected.success,
                    "pair_type": "success" if pair.chosen.success else "failure",
                }
            )

        pair_accuracy = correct / len(pairs)

        # Break down by pair type (within-success vs within-failure)
        success_pairs = [r for r in results if r["pair_type"] == "success"]
        failure_pairs = [r for r in results if r["pair_type"] == "failure"]
        success_acc = (
            sum(1 for r in success_pairs if r["rm_correct"]) / len(success_pairs)
            if success_pairs
            else float("nan")
        )
        failure_acc = (
            sum(1 for r in failure_pairs if r["rm_correct"]) / len(failure_pairs)
            if failure_pairs
            else float("nan")
        )

        metrics = {
            "pair_accuracy": pair_accuracy,
            "num_pairs": len(pairs),
            "num_correct": correct,
            "within_success_accuracy": success_acc,
            "within_failure_accuracy": failure_acc,
            "num_success_pairs": len(success_pairs),
            "num_failure_pairs": len(failure_pairs),
        }

        logger.info(
            f"Reward model evaluation results:\n"
            f"  Overall pair accuracy     : {pair_accuracy:.4f}  ({correct}/{len(pairs)})\n"
            f"  Within-success pair accuracy: {success_acc:.4f}  ({len(success_pairs)} pairs)\n"
            f"  Within-failure pair accuracy: {failure_acc:.4f}  ({len(failure_pairs)} pairs)"
        )

        # 5. Optionally save per-pair details
        if self.output_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump({"metrics": metrics, "pairs": results}, f, indent=2)
            logger.info(f"Saved per-pair results to {self.output_path}")

        return metrics
