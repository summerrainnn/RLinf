"""Runner for multi-policy trajectory data collection.

Collects trajectories from multiple policies (SFT checkpoints, noise/delay
variants, expert models) and saves as a :class:`TrajectoryDataset`.  Handles
policy switching, optional trajectory segmentation, and quality scoring.

Usage::

    python examples/embodiment/collect_trajectory_data.py \\
        --config-name maniskill_collect_trajectory_data
"""

from __future__ import annotations

import os
import typing
from datetime import datetime

import numpy as np

from rlinf.data.preference_data import EpisodeRecord
from rlinf.data.quality_evaluation import compute_dataset_quality, compute_smoothness_from_video
from rlinf.data.trajectory_dataset import (
    TrajectoryDataset,
    TrajectoryRecord,
    ScoringResult,
    generate_task_seeds,
    split_trajectory,
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

logger = get_logger()


class MultiPolicyCollectionRunner:
    """Orchestrates trajectory collection from multiple policies.

    Reads ``cfg.collection`` to determine policies, task seeds, and
    segmentation settings.  For each policy, runs rollout epochs with the
    appropriate model weights / wrapper configuration, assembles
    :class:`TrajectoryRecord` objects, optionally splits trajectories,
    groups them by ``(env_seed, start_step)``, and saves a
    :class:`TrajectoryDataset`.

    Args:
        cfg: Full Hydra config.
        rollout: Preference collection rollout worker group.
        env: Preference env worker group.
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

        coll_cfg = cfg.collection
        self.num_tasks: int = int(coll_cfg.num_tasks)
        self.base_seed: int = int(coll_cfg.get("base_seed", 42))
        self.trajectories_per_task: int = int(coll_cfg.get("trajectories_per_task", 5))
        self.num_segments: int = int(coll_cfg.get("num_segments", 1))
        self.max_steps: int = int(coll_cfg.get("max_steps", 80))
        self.output_path: str = str(coll_cfg.output_path)
        self.policies: list[dict] = list(coll_cfg.policies)

        self._n_total_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )

    def init_workers(self) -> None:
        """Initialize rollout and env workers."""
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        ckpt_path = self.cfg.runner.get("ckpt_path", None)
        if ckpt_path:
            logger.info(f"Loading checkpoint from {ckpt_path}")

    def _assign_policies_to_tasks(
        self, task_seeds: list[int]
    ) -> list[list[dict]]:
        """For each task, sample ``trajectories_per_task`` policies by weight.

        Returns:
            List of length ``num_tasks``, each a list of policy config dicts.
        """
        weights = np.array([p.get("weight", 1.0) for p in self.policies])
        weights = weights / weights.sum()
        rng = np.random.RandomState(self.base_seed + 1)

        assignments = []
        for _ in task_seeds:
            indices = rng.choice(
                len(self.policies),
                size=self.trajectories_per_task,
                replace=True,
                p=weights,
            )
            assignments.append([self.policies[i] for i in indices])
        return assignments

    def _run_collection_epoch(self) -> tuple[list[dict], list[list[dict]]]:
        """Run one rollout epoch and return env results + rollout results."""
        env_handle: Handle = self.env.collect_preference_epoch(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.collect_episodes_epoch(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )

        env_results_list = env_handle.wait()
        raw_obs_list = rollout_handle.wait()
        return env_results_list, raw_obs_list

    def _build_records_from_epoch(
        self,
        env_results_list: list[dict],
        raw_obs_list: list[list[dict]],
        policy_name: str,
        chunk_size: int,
        env_seeds: list[int],
    ) -> list[TrajectoryRecord]:
        """Convert one epoch's results into TrajectoryRecords."""
        all_records: list[TrajectoryRecord] = []
        for env_result, raw_obs in zip(env_results_list, raw_obs_list):
            if env_result is None or raw_obs is None:
                continue

            task_descriptions = [ep["task_description"] for ep in raw_obs]
            video_paths = env_result.get("video_paths", [])
            if len(video_paths) < len(task_descriptions):
                video_paths = list(video_paths) + [""] * (
                    len(task_descriptions) - len(video_paths)
                )
            rewards = env_result["rewards"]
            successes = env_result["successes"]
            lengths = env_result["lengths"]

            n_envs = len(task_descriptions)
            for i in range(n_envs):
                seed = env_seeds[i] if i < len(env_seeds) else 0
                vp = video_paths[i] if i < len(video_paths) else None
                ep_len = int(lengths[i]) if i < len(lengths) else self.max_steps

                # Compute smoothness from video if available
                smoothness = None
                if vp and os.path.exists(vp):
                    try:
                        smoothness = compute_smoothness_from_video(vp)
                    except Exception:
                        pass

                record = TrajectoryRecord(
                    video_path=vp if vp else None,
                    model_name=policy_name,
                    chunk_size=chunk_size,
                    start_step=0,
                    end_step=ep_len - 1,
                    cumulative_reward=float(rewards[i]),
                    env_seed=seed,
                    language_instruction=task_descriptions[i],
                    success=bool(successes[i]),
                    episode_length=ep_len,
                    smoothness=smoothness,
                )
                all_records.append(record)
        return all_records

    def run(self) -> None:
        """Execute the full multi-policy collection pipeline."""
        logger.info(
            f"Starting multi-policy trajectory collection: "
            f"{self.num_tasks} tasks, {self.trajectories_per_task} per task, "
            f"{len(self.policies)} policies."
        )

        # Generate task seeds
        task_seeds = generate_task_seeds(self.num_tasks, self.base_seed)
        assignments = self._assign_policies_to_tasks(task_seeds)

        # Flatten: collect by policy to minimize model reloads
        policy_jobs: dict[str, list[tuple[int, int]]] = {}  # name -> [(task_idx, seed)]
        for task_idx, (seed, assigned) in enumerate(zip(task_seeds, assignments)):
            for policy_cfg in assigned:
                name = policy_cfg["name"]
                if name not in policy_jobs:
                    policy_jobs[name] = []
                policy_jobs[name].append((task_idx, seed))

        all_records: list[TrajectoryRecord] = []
        chunk_size = int(self.cfg.actor.model.num_action_chunks)
        n_envs = int(self.cfg.env.eval.total_num_envs)

        for policy_cfg in self.policies:
            name = policy_cfg["name"]
            if name not in policy_jobs:
                continue
            jobs = policy_jobs[name]
            logger.info(
                f"Collecting {len(jobs)} trajectories for policy '{name}' "
                f"(type={policy_cfg.get('type', 'checkpoint')})"
            )

            # Batch jobs into epochs of n_envs
            for batch_start in range(0, len(jobs), n_envs):
                batch = jobs[batch_start : batch_start + n_envs]
                batch_seeds = [seed for _, seed in batch]

                # Pad to fill n_envs if needed (last batch may be smaller)
                while len(batch_seeds) < n_envs:
                    batch_seeds.append(batch_seeds[-1])

                env_results_list, raw_obs_list = self._run_collection_epoch()

                records = self._build_records_from_epoch(
                    env_results_list,
                    raw_obs_list,
                    policy_name=name,
                    chunk_size=chunk_size,
                    env_seeds=batch_seeds,
                )

                # Only keep the first len(batch) records (others are padding)
                records = records[: len(batch)]
                all_records.extend(records)

                logger.info(
                    f"  Batch {batch_start // n_envs + 1}: "
                    f"collected {len(records)} records (total {len(all_records)})"
                )

        # Optionally split trajectories into segments
        if self.num_segments > 1:
            logger.info(f"Splitting {len(all_records)} trajectories into {self.num_segments} segments each")
            segmented_records: list[TrajectoryRecord] = []
            for record in all_records:
                # Approximate segment rewards/smoothness from the full trajectory
                seg_rewards = [
                    (record.cumulative_reward or 0.0) / self.num_segments
                ] * self.num_segments
                seg_smoothness = [record.smoothness or 0.5] * self.num_segments
                try:
                    segments = split_trajectory(
                        record, seg_rewards, seg_smoothness, self.num_segments
                    )
                    segmented_records.extend(segments)
                except Exception as e:
                    logger.warning(f"Failed to split trajectory: {e}")
                    segmented_records.append(record)
            all_records = segmented_records

        # Group by (env_seed, start_step)
        groups = TrajectoryDataset.group_by_seed_and_segment(all_records)

        # Build env_reward ScoringResult
        env_scores = {
            i: t.cumulative_reward
            for i, t in enumerate(all_records)
            if t.cumulative_reward is not None
        }
        scoring = [
            ScoringResult(
                scorer_name="env_cumulative_reward",
                scorer_type="env_reward",
                scores=env_scores,
            )
        ]

        dataset = TrajectoryDataset(
            trajectories=all_records,
            groups=groups,
            scoring_results=scoring,
            metadata={
                "collection_time": datetime.now().isoformat(),
                "num_tasks": self.num_tasks,
                "trajectories_per_task": self.trajectories_per_task,
                "num_segments": self.num_segments,
                "num_policies": len(self.policies),
                "policy_names": [p["name"] for p in self.policies],
            },
        )

        # Compute quality scores
        compute_dataset_quality(dataset, max_steps=self.max_steps)

        # Save
        output_path = self._build_output_path()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        dataset.save(output_path)

        # Summary
        policy_counts = {}
        for t in all_records:
            policy_counts[t.model_name] = policy_counts.get(t.model_name, 0) + 1
        logger.info(
            f"Collection complete. Saved {len(all_records)} trajectories "
            f"in {len(groups)} groups to {output_path}"
        )
        for name, count in sorted(policy_counts.items()):
            logger.info(f"  {name}: {count} trajectories")

    def _build_output_path(self) -> str:
        if not self.output_path:
            return self.output_path
        parent = os.path.dirname(os.path.abspath(self.output_path))
        basename = os.path.basename(self.output_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        return os.path.join(parent, timestamp, basename)
