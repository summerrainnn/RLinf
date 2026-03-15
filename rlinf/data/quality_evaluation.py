"""Trajectory quality evaluation utilities.

Provides functions to compute smoothness, composite quality scores, and
batch quality evaluation for TrajectoryDataset instances.
"""

from typing import Optional

import numpy as np


def compute_smoothness(actions: np.ndarray) -> float:
    """Compute action-sequence smoothness.

    Args:
        actions: shape ``[T, action_dim]``, normalized action sequence.

    Returns:
        Smoothness in (0, 1]. Higher is smoother.
        Formula: ``1 / (1 + mean(||a_{t+1} - a_t||_2))``.
    """
    if len(actions) < 2:
        return 1.0
    diffs = np.diff(actions, axis=0)  # [T-1, action_dim]
    l2_norms = np.linalg.norm(diffs, axis=1)  # [T-1]
    return 1.0 / (1.0 + float(np.mean(l2_norms)))


def compute_smoothness_from_video(video_path: str) -> float:
    """Compute visual smoothness from video frames (fallback when no actions).

    Formula: ``1 / (1 + mean(|frame_{t+1} - frame_t|) / 255)``.
    """
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        frames = [f.astype(np.float32) for f in reader]
    finally:
        reader.close()
    if len(frames) < 2:
        return 1.0
    diffs = []
    for i in range(len(frames) - 1):
        diff = float(np.mean(np.abs(frames[i + 1] - frames[i])))
        diffs.append(diff)
    return 1.0 / (1.0 + float(np.mean(diffs)) / 255.0)


def compute_trajectory_quality(
    success: Optional[bool],
    cumulative_reward: Optional[float],
    episode_length: Optional[int],
    smoothness: Optional[float],
    max_steps: int = 80,
    weights: Optional[dict] = None,
) -> tuple[float, dict]:
    """Compute a single trajectory's quality score.

    Default weight design:
    - success:    range {0, 1}, weight 5.0 (most important)
    - reward:     range ~[0, 2], weight 1.0
    - efficiency: 1 - steps/max_steps, range [0, 1], weight 1.0
    - smoothness: range (0, 1], weight 0.5

    Total score range: roughly [0, 8.5].

    Args:
        success, cumulative_reward, episode_length, smoothness: from TrajectoryRecord.
        max_steps: maximum episode steps (for efficiency normalization).
        weights: override default weights.

    Returns:
        ``(quality_score, quality_metrics)`` tuple.
    """
    if weights is None:
        weights = {
            "success": 5.0,
            "reward": 1.0,
            "efficiency": 1.0,
            "smoothness": 0.5,
        }

    metrics: dict[str, Optional[float]] = {}
    score = 0.0

    metrics["success"] = float(success) if success is not None else None
    if success is not None:
        score += weights["success"] * float(success)

    metrics["reward"] = cumulative_reward
    if cumulative_reward is not None:
        score += weights["reward"] * cumulative_reward

    if episode_length is not None and max_steps > 0:
        efficiency = 1.0 - episode_length / max_steps
        metrics["efficiency"] = max(0.0, efficiency)
    else:
        metrics["efficiency"] = None
    if metrics["efficiency"] is not None:
        score += weights["efficiency"] * metrics["efficiency"]

    metrics["smoothness"] = smoothness
    if smoothness is not None:
        score += weights["smoothness"] * smoothness

    return score, metrics


def compute_dataset_quality(dataset, **kwargs) -> None:
    """Compute quality scores for all trajectories in a dataset (in-place).

    Writes ``quality_score`` and ``quality_metrics`` on each
    ``TrajectoryRecord``.
    """
    for t in dataset.trajectories:
        score, metrics = compute_trajectory_quality(
            success=t.success,
            cumulative_reward=t.cumulative_reward,
            episode_length=t.episode_length,
            smoothness=t.smoothness,
            **kwargs,
        )
        t.quality_score = score
        t.quality_metrics = metrics
