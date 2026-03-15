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
from rlinf.data.trajectory_dataset import ScoringResult, TrajectoryDataset
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
        self.eval_data_path: Optional[str] = eval_cfg.get("eval_data_path", None) or None

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
        """Collect or load episodes, score with RM, compute pair accuracy.

        If ``eval_data_path`` is set (a ``.json`` TrajectoryDataset), loads
        data from file and skips rollouts.  Otherwise falls back to the
        original rollout-based evaluation.

        Returns:
            Dict with evaluation metrics.
        """
        if self.eval_data_path and self.eval_data_path.endswith(".json"):
            return self._run_from_file()
        return self._run_from_rollouts()

    # ------------------------------------------------------------------
    # File-based evaluation (TrajectoryDataset JSON)
    # ------------------------------------------------------------------

    def _run_from_file(self) -> dict:
        """Load a TrajectoryDataset, score all trajectories, compute accuracy."""
        logger.info(f"Loading evaluation data from {self.eval_data_path}")
        dataset = TrajectoryDataset.load(self.eval_data_path)
        logger.info(
            f"Loaded {len(dataset.trajectories)} trajectories, "
            f"{len(dataset.groups)} groups."
        )
        return self._evaluate_dataset(dataset)

    def _evaluate_dataset(self, dataset: TrajectoryDataset) -> dict:
        """Score all trajectories with RM and compute pairwise accuracy."""
        # 1. Score all trajectories
        logger.info("Scoring trajectories with the reward model …")
        rm_scores = self._score_trajectory_records(dataset.trajectories)

        # 2. Add RM scores as a ScoringResult
        from pathlib import Path

        rm_scoring = ScoringResult(
            scorer_name=f"rm_{Path(self.checkpoint_dir).name}",
            scorer_type="reward_model",
            model_name=self.model_path,
            scores={i: s for i, s in enumerate(rm_scores)},
        )
        dataset.scoring_results.append(rm_scoring)

        # 3. Save dataset with RM scores
        if self.output_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True
            )
            dataset.save(self.output_path)
            logger.info(f"Saved scored dataset to {self.output_path}")

        # 4. Get ground-truth scores
        gt_scoring = dataset.get_final_scores()
        if gt_scoring is None:
            # Fallback to env cumulative reward
            gt_scoring = ScoringResult(
                scorer_name="env_cumulative_reward",
                scorer_type="env_reward",
                scores={
                    i: t.cumulative_reward
                    for i, t in enumerate(dataset.trajectories)
                    if t.cumulative_reward is not None
                },
            )

        # 5. Compute pairwise accuracy
        metrics = self._compute_group_accuracy(
            dataset.groups, rm_scoring, gt_scoring
        )
        logger.info(
            f"Reward model evaluation results:\n"
            f"  Overall pair accuracy: {metrics['pair_accuracy']:.4f} "
            f"({metrics['num_correct']}/{metrics['num_pairs']})"
        )
        return metrics

    @torch.no_grad()
    def _score_trajectory_records(
        self, trajectories: list
    ) -> list[float]:
        """Score TrajectoryRecord objects with the RM."""
        all_scores: list[float] = []
        for start in range(0, len(trajectories), self.batch_size):
            batch = trajectories[start : start + self.batch_size]
            episodes = []
            for t in batch:
                episodes.append(
                    EpisodeRecord(
                        task_description=t.language_instruction,
                        cumulative_reward=t.cumulative_reward or 0.0,
                        success=t.success or False,
                        episode_length=t.episode_length or 0,
                        video_path=t.video_path,
                    )
                )
            keyframes_list = [
                get_keyframes(ep, self.n_keyframes) for ep in episodes
            ]
            scores = self.reward_model(
                task_descriptions=[ep.task_description for ep in episodes],
                keyframes_list=keyframes_list,
            )
            all_scores.extend(scores.cpu().float().tolist())
        return all_scores

    @staticmethod
    def _compute_group_accuracy(
        groups: list[list[int]],
        rm_scoring: ScoringResult,
        gt_scoring: ScoringResult,
    ) -> dict:
        """Compute pairwise accuracy across all groups."""
        correct = 0
        total = 0
        for group in groups:
            scored = [
                (idx, rm_scoring.scores.get(idx), gt_scoring.scores.get(idx))
                for idx in group
            ]
            scored = [
                (idx, rm, gt)
                for idx, rm, gt in scored
                if rm is not None and gt is not None
            ]
            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    if scored[i][2] == scored[j][2]:
                        continue  # equal ground truth — skip
                    rm_prefers_i = scored[i][1] > scored[j][1]
                    gt_prefers_i = scored[i][2] > scored[j][2]
                    if rm_prefers_i == gt_prefers_i:
                        correct += 1
                    total += 1
        return {
            "pair_accuracy": correct / total if total > 0 else 0.0,
            "num_pairs": total,
            "num_correct": correct,
        }

    # ------------------------------------------------------------------
    # Rollout-based evaluation (original behavior)
    # ------------------------------------------------------------------

    def _run_from_rollouts(self) -> dict:
        """Original rollout-based evaluation pipeline."""
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
