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

"""Data structures and utilities for robot preference learning."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio
import numpy as np


@dataclass
class EpisodeRecord:
    """Record of a single robot episode for preference learning.

    Supports two storage modes:

    1. **Video mode** (preferred): ``video_path`` points to an mp4 file
       containing the full episode recording.  Frames are loaded on demand
       via :func:`load_keyframes_from_video`.
    2. **Legacy keyframe mode**: ``keyframes`` stores a list of uint8 numpy
       arrays sampled at evenly-spaced intervals.

    Use :func:`get_keyframes` as the unified access point — it picks the
    right source automatically.

    Attributes:
        task_description: Natural-language task instruction for this episode.
        cumulative_reward: Undiscounted sum of per-step rewards.
        success: Whether the episode ended in a successful outcome.
        episode_length: Number of primitive environment steps executed.
        video_path: Path to the mp4 video file (preferred storage).
        keyframes: Legacy list of N [H, W, C] uint8 numpy arrays.
        wrist_keyframes: Optional list of N wrist-camera frames.
    """

    task_description: str
    cumulative_reward: float
    success: bool
    episode_length: int
    video_path: Optional[str] = None
    keyframes: Optional[list] = None  # list of N [H, W, C] uint8 numpy arrays
    wrist_keyframes: Optional[list] = None  # list of N [H, W, C] uint8, or None


@dataclass
class PreferencePair:
    """A pair of episodes with an implicit preference label.

    ``chosen`` has a higher cumulative reward than ``rejected``.
    Both episodes belong to the same group (both successful or both failed).
    """

    chosen: EpisodeRecord  # Higher-reward trajectory (preferred)
    rejected: EpisodeRecord  # Lower-reward trajectory (dis-preferred)

    @property
    def reward_margin(self) -> float:
        """Difference in cumulative reward between chosen and rejected."""
        return self.chosen.cumulative_reward - self.rejected.cumulative_reward


def save_preference_pairs(pairs: list[PreferencePair], path: str) -> None:
    """Save a list of preference pairs to a pickle file.

    Args:
        pairs: List of ``PreferencePair`` objects.
        path: Destination file path (parent directories are created automatically).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_preference_pairs(path: str) -> list[PreferencePair]:
    """Load preference pairs from a pickle file.

    Args:
        path: Source file path.

    Returns:
        List of ``PreferencePair`` objects.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Video ↔ keyframe utilities
# ---------------------------------------------------------------------------


def load_keyframes_from_video(
    video_path: str, n_frames: int = 8
) -> list[np.ndarray]:
    """Load N evenly-spaced frames from an mp4 video file.

    Args:
        video_path: Path to the mp4 video file.
        n_frames: Number of frames to sample (default 8).

    Returns:
        List of N uint8 numpy arrays of shape [H, W, C].

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be read.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = imageio.get_reader(video_path)
    try:
        try:
            total_frames = reader.count_frames()
        except Exception:
            # Fallback: read all frames to count
            all_frames = list(reader)
            total_frames = len(all_frames)
            if total_frames == 0:
                return []
            indices = np.linspace(0, total_frames - 1, n_frames).round().astype(int)
            return [all_frames[i].copy() for i in indices]

        if total_frames == 0:
            return []

        indices = np.linspace(0, total_frames - 1, n_frames).round().astype(int)
        frames = []
        for idx in indices:
            frame = reader.get_data(int(idx))
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frames.append(frame)
        return frames
    finally:
        reader.close()


def get_keyframes(episode: EpisodeRecord, n_frames: int = 8) -> list[np.ndarray]:
    """Unified access to episode keyframes — supports both video and legacy formats.

    Args:
        episode: An ``EpisodeRecord`` instance.
        n_frames: Number of frames to sample when loading from video.

    Returns:
        List of uint8 numpy arrays of shape [H, W, C].
    """
    if episode.video_path:
        return load_keyframes_from_video(episode.video_path, n_frames)
    if episode.keyframes:
        return episode.keyframes
    return []


def create_preference_pairs(
    episodes: list[EpisodeRecord],
    min_reward_diff: float = 0.0,
) -> list[PreferencePair]:
    """Create preference pairs from a flat list of episodes.

    Pairing strategy (within-group):
      1. Split episodes into ``successes`` and ``failures`` groups.
      2. Within each group, randomly shuffle and pair episodes two-by-two.
      3. For each pair the episode with higher cumulative reward becomes
         ``chosen`` and the other becomes ``rejected``.
      4. Discard pairs whose reward difference is below ``min_reward_diff``.

    Args:
        episodes: Flat list of episodes from one or more rollout epochs.
        min_reward_diff: Minimum absolute reward difference to keep a pair.
            Pairs with smaller margins are discarded.

    Returns:
        List of ``PreferencePair`` objects (from both groups combined).
    """
    successes = [ep for ep in episodes if ep.success]
    failures = [ep for ep in episodes if not ep.success]

    rng = np.random.default_rng()

    def _pair_within_group(group: list[EpisodeRecord]) -> list[PreferencePair]:
        rng.shuffle(group)
        pairs: list[PreferencePair] = []
        # Take consecutive pairs (0,1), (2,3), ...
        for i in range(0, len(group) - 1, 2):
            a, b = group[i], group[i + 1]
            if a.cumulative_reward >= b.cumulative_reward:
                chosen, rejected = a, b
            else:
                chosen, rejected = b, a
            diff = chosen.cumulative_reward - rejected.cumulative_reward
            if diff >= min_reward_diff:
                pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
        return pairs

    pairs = _pair_within_group(successes) + _pair_within_group(failures)
    rng.shuffle(pairs)
    return pairs


def summarize_preference_dataset(pairs: list[PreferencePair]) -> dict:
    """Return a summary dict describing the dataset.

    Args:
        pairs: List of ``PreferencePair`` objects.

    Returns:
        Dict with counts and average reward margins.
    """
    if not pairs:
        return {"total": 0}

    margins = [p.reward_margin for p in pairs]

    return {
        "total": len(pairs),
        "avg_reward_margin": float(np.mean(margins)),
        "min_reward_margin": float(np.min(margins)),
        "max_reward_margin": float(np.max(margins)),
    }
