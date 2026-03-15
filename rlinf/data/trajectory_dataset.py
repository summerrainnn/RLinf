"""Trajectory dataset format for preference learning and reward model training.

Supports multi-policy collection, multi-perspective scoring, ranking annotations,
and backward compatibility with the legacy PreferencePair format.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class TrajectoryRecord:
    """Metadata for a single trajectory segment."""

    video_path: Optional[str]
    model_name: str
    chunk_size: int
    start_step: int
    end_step: int
    cumulative_reward: Optional[float]
    env_seed: int
    language_instruction: str
    success: Optional[bool] = None
    episode_length: Optional[int] = None
    smoothness: Optional[float] = None
    quality_metrics: Optional[dict] = None
    quality_score: Optional[float] = None


@dataclass
class ScoringResult:
    """A single scoring pass over (a subset of) trajectories in a dataset."""

    scorer_name: str
    scorer_type: str  # "human" | "ai" | "reward_model" | "env_reward"
    scores: dict[int, Optional[float]]
    prompt: Optional[str] = None
    model_name: Optional[str] = None
    perspective_name: Optional[str] = None
    rationales: Optional[dict[int, str]] = None


@dataclass
class TrajectoryDataset:
    """A complete trajectory dataset with grouping and scoring information."""

    trajectories: list[TrajectoryRecord]
    groups: list[list[int]]
    scoring_results: list[ScoringResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ---- Serialization ----

    def save(self, path: str) -> None:
        """Save dataset to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trajectories": [asdict(t) for t in self.trajectories],
            "groups": self.groups,
            "scoring_results": [
                {
                    **asdict(sr),
                    "scores": {str(k): v for k, v in sr.scores.items()},
                    "rationales": (
                        {str(k): v for k, v in sr.rationales.items()}
                        if sr.rationales
                        else None
                    ),
                }
                for sr in self.scoring_results
            ],
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TrajectoryDataset":
        """Load dataset from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        trajectories = [TrajectoryRecord(**t) for t in data["trajectories"]]
        scoring_results = []
        for sr_raw in data.get("scoring_results", []):
            sr_raw["scores"] = {int(k): v for k, v in sr_raw["scores"].items()}
            if sr_raw.get("rationales"):
                sr_raw["rationales"] = {
                    int(k): v for k, v in sr_raw["rationales"].items()
                }
            scoring_results.append(ScoringResult(**sr_raw))
        return cls(
            trajectories=trajectories,
            groups=data["groups"],
            scoring_results=scoring_results,
            metadata=data.get("metadata", {}),
        )

    # ---- Grouping utilities ----

    @staticmethod
    def group_by_seed_and_segment(
        trajectories: list[TrajectoryRecord],
    ) -> list[list[int]]:
        """Group trajectories by (env_seed, start_step)."""
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for idx, t in enumerate(trajectories):
            buckets[(t.env_seed, t.start_step)].append(idx)
        return list(buckets.values())

    # ---- Scoring utilities ----

    def get_final_scores(self) -> Optional[ScoringResult]:
        """Return the last overall scoring result (perspective_name is None)."""
        for sr in reversed(self.scoring_results):
            if sr.perspective_name is None:
                return sr
        return None

    def compute_average_perspective_scores(self) -> ScoringResult:
        """Average all per-perspective ScoringResults into an overall score."""
        perspective_results = [
            sr for sr in self.scoring_results if sr.perspective_name is not None
        ]
        if not perspective_results:
            raise ValueError("No perspective scoring results found")
        all_indices: set[int] = set()
        for sr in perspective_results:
            all_indices.update(sr.scores.keys())
        avg_scores: dict[int, Optional[float]] = {}
        for idx in all_indices:
            vals = [
                sr.scores[idx]
                for sr in perspective_results
                if idx in sr.scores and sr.scores[idx] is not None
            ]
            avg_scores[idx] = sum(vals) / len(vals) if vals else None
        return ScoringResult(
            scorer_name="average_perspectives",
            scorer_type=perspective_results[0].scorer_type,
            scores=avg_scores,
            prompt=None,
            model_name=None,
            perspective_name=None,
        )

    # ---- Ranking to scores conversion ----

    @staticmethod
    def ranking_to_scores(
        group_indices: list[int], ranking: list[int]
    ) -> dict[int, float]:
        """Convert a within-group ranking to numeric scores.

        Args:
            group_indices: trajectory indices in the group (unused but kept for API clarity).
            ranking: ordered list of trajectory indices, ranking[0] = best.

        Returns:
            {trajectory_index: score} where best gets len(ranking) points, worst gets 1.
        """
        n = len(ranking)
        return {ranking[i]: float(n - i) for i in range(n)}

    # ---- Merge datasets ----

    @staticmethod
    def merge(datasets: list["TrajectoryDataset"]) -> "TrajectoryDataset":
        """Merge multiple TrajectoryDatasets, re-indexing trajectories and groups."""
        merged_trajectories: list[TrajectoryRecord] = []
        merged_groups: list[list[int]] = []
        merged_scoring: list[ScoringResult] = []
        offset = 0
        for ds in datasets:
            merged_trajectories.extend(ds.trajectories)
            for group in ds.groups:
                merged_groups.append([idx + offset for idx in group])
            for sr in ds.scoring_results:
                new_scores = {k + offset: v for k, v in sr.scores.items()}
                new_rationales = None
                if sr.rationales:
                    new_rationales = {k + offset: v for k, v in sr.rationales.items()}
                merged_scoring.append(
                    ScoringResult(
                        scorer_name=sr.scorer_name,
                        scorer_type=sr.scorer_type,
                        scores=new_scores,
                        prompt=sr.prompt,
                        model_name=sr.model_name,
                        perspective_name=sr.perspective_name,
                        rationales=new_rationales,
                    )
                )
            offset += len(ds.trajectories)
        return TrajectoryDataset(
            trajectories=merged_trajectories,
            groups=merged_groups,
            scoring_results=merged_scoring,
            metadata={"merged_from": [ds.metadata for ds in datasets]},
        )

    # ---- Legacy format conversion ----

    @classmethod
    def from_preference_pairs(
        cls, pairs: list
    ) -> "TrajectoryDataset":
        """Convert from legacy list[PreferencePair] format."""
        trajectories: list[TrajectoryRecord] = []
        groups: list[list[int]] = []
        for pair in pairs:
            idx_c = len(trajectories)
            trajectories.append(
                TrajectoryRecord(
                    video_path=getattr(pair.chosen, "video_path", None),
                    model_name="unknown",
                    chunk_size=1,
                    start_step=0,
                    end_step=(
                        pair.chosen.episode_length - 1
                        if pair.chosen.episode_length
                        else -1
                    ),
                    cumulative_reward=pair.chosen.cumulative_reward,
                    env_seed=0,
                    language_instruction=pair.chosen.task_description,
                    success=pair.chosen.success,
                    episode_length=pair.chosen.episode_length,
                )
            )
            idx_r = len(trajectories)
            trajectories.append(
                TrajectoryRecord(
                    video_path=getattr(pair.rejected, "video_path", None),
                    model_name="unknown",
                    chunk_size=1,
                    start_step=0,
                    end_step=(
                        pair.rejected.episode_length - 1
                        if pair.rejected.episode_length
                        else -1
                    ),
                    cumulative_reward=pair.rejected.cumulative_reward,
                    env_seed=0,
                    language_instruction=pair.rejected.task_description,
                    success=pair.rejected.success,
                    episode_length=pair.rejected.episode_length,
                )
            )
            groups.append([idx_c, idx_r])
        env_scores: dict[int, Optional[float]] = {
            i: t.cumulative_reward
            for i, t in enumerate(trajectories)
            if t.cumulative_reward is not None
        }
        scoring = [
            ScoringResult(
                scorer_name="env_cumulative_reward",
                scorer_type="env_reward",
                scores=env_scores,
            )
        ]
        return cls(
            trajectories=trajectories, groups=groups, scoring_results=scoring
        )


# ---- Trajectory splitting utilities ----


def split_video_file(video_path: str, num_segments: int, fps: int = 5) -> list[str]:
    """Split an mp4 video into equal segments.

    Args:
        video_path: path to source mp4 file.
        num_segments: number of segments to create.
        fps: output frame rate.

    Returns:
        List of output segment file paths.
    """
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        frames = list(reader)
    finally:
        reader.close()

    seg_size = len(frames) // num_segments
    paths: list[str] = []
    base = video_path.rsplit(".mp4", 1)[0]
    for i in range(num_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < num_segments - 1 else len(frames)
        seg_path = f"{base}_seg{i}.mp4"
        writer = imageio.get_writer(seg_path, fps=fps)
        for frame in frames[start:end]:
            writer.append_data(frame)
        writer.close()
        paths.append(seg_path)
    return paths


def split_trajectory(
    record: TrajectoryRecord,
    segment_rewards: list[float],
    segment_smoothness: list[float],
    num_segments: int,
    fps: int = 5,
) -> list[TrajectoryRecord]:
    """Split a full trajectory into segments.

    Args:
        record: the original TrajectoryRecord.
        segment_rewards: per-segment cumulative reward, length = num_segments.
        segment_smoothness: per-segment smoothness, length = num_segments.
        num_segments: how many segments.
        fps: video fps for split files.

    Returns:
        List of TrajectoryRecord, one per segment.
    """
    total_steps = record.end_step - record.start_step + 1
    seg_length = total_steps // num_segments

    seg_video_paths: list[Optional[str]] = [None] * num_segments
    if record.video_path:
        seg_video_paths = split_video_file(record.video_path, num_segments, fps)

    segments: list[TrajectoryRecord] = []
    for i in range(num_segments):
        seg_start = record.start_step + i * seg_length
        if i < num_segments - 1:
            seg_end = record.start_step + (i + 1) * seg_length - 1
        else:
            seg_end = record.end_step
        segments.append(
            TrajectoryRecord(
                video_path=seg_video_paths[i],
                model_name=record.model_name,
                chunk_size=record.chunk_size,
                start_step=seg_start,
                end_step=seg_end,
                cumulative_reward=segment_rewards[i],
                env_seed=record.env_seed,
                language_instruction=record.language_instruction,
                success=record.success if i == num_segments - 1 else None,
                episode_length=record.episode_length,
                smoothness=segment_smoothness[i],
            )
        )
    return segments


def generate_task_seeds(num_tasks: int, base_seed: int = 42) -> list[int]:
    """Generate deterministic, non-repeating env seeds."""
    import numpy as np

    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31, size=num_tasks).tolist()
