#!/usr/bin/env python3
"""Extract state features from trajectory videos using a sliding window.

Reads one or more TrajectoryDataset JSON files, extracts CLIP features from
sliding-window video clips, and saves the result as an ``.npz`` file that
can be consumed by ``visualize_umap.py``.

Output ``.npz`` format::

    features     : (N, feature_dim) float32 — one row per state
    group_labels : (N,) str                 — group label for each state
    metadata     : dict                     — extraction parameters

Usage::

    # Multiple files, grouped by file
    python toolkits/extract_features.py \\
        --data file1.json file2.json \\
        --labels "Policy A" "Policy B" \\
        --output features.npz

    # Single file, grouped by policy name
    python toolkits/extract_features.py \\
        --data file.json \\
        --group-by policy \\
        --output features.npz
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Video path resolution (container ↔ host)
# ---------------------------------------------------------------------------


def _resolve_video_path(video_path: Optional[str]) -> Optional[str]:
    """Resolve video path, handling container vs host path mapping."""
    if not video_path:
        return None
    p = Path(video_path)
    if p.exists():
        return str(p)
    if video_path.startswith("/workspace/RLinf/"):
        host_path = Path.home() / video_path[len("/workspace/RLinf/"):]
        if host_path.exists():
            return str(host_path)
    return None


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------


def load_all_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file.

    Args:
        video_path: Path to the mp4 video file.

    Returns:
        (T, H, W, 3) uint8 numpy array.
    """
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        frames = []
        for frame in reader:
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def extract_clips(
    all_frames: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """Extract sliding-window clips from a frame array.

    Args:
        all_frames: (T, H, W, 3) uint8 array.
        window_size: Number of consecutive frames per clip.
        stride: Step size between clip start positions.

    Returns:
        List of (window_size, H, W, 3) uint8 arrays.
    """
    total = all_frames.shape[0]
    if total < window_size:
        # If the video is shorter than one window, return the whole thing
        return [all_frames]

    clips = []
    start = 0
    while start + window_size <= total:
        clips.append(all_frames[start : start + window_size])
        start += stride
    return clips


# ---------------------------------------------------------------------------
# CLIP feature extractor
# ---------------------------------------------------------------------------


class CLIPFeatureExtractor:
    """Extract image features using CLIP ViT-B/32."""

    FEATURE_DIM = 512

    def __init__(self, device: str = "cuda"):
        import os

        from transformers import CLIPModel, CLIPProcessor

        # Use hf-mirror.com for network accessibility
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        model_name = "openai/clip-vit-base-patch32"

        # Cache models under the project tree so they persist on the host
        # when running inside a container with -v .:/workspace/RLinf
        _project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = str(_project_root / "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.processor = CLIPProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = CLIPModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def extract_clip_feature(self, frames: np.ndarray) -> np.ndarray:
        """Extract a single feature vector from a clip of frames.

        Strategy: extract CLIP image features for each frame independently,
        then average-pool across frames.

        Args:
            frames: (window_size, H, W, 3) uint8 numpy array.

        Returns:
            (feature_dim,) float32 numpy array.
        """
        pil_images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)  # (N, 512)
        # L2-normalize then average
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        mean_feature = outputs.mean(dim=0)  # (512,)
        mean_feature = mean_feature / mean_feature.norm()
        return mean_feature.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_clip_features_batch(
        self, clips: list[np.ndarray], batch_size: int = 512
    ) -> list[np.ndarray]:
        """Extract feature vectors for multiple clips with batched inference.

        Flattens all frames across clips, runs CLIP in large batches, then
        re-groups and average-pools per clip.

        Args:
            clips: List of (window_size, H, W, 3) uint8 numpy arrays.
            batch_size: Max images per forward pass (tune for GPU memory).

        Returns:
            List of (feature_dim,) float32 numpy arrays, one per clip.
        """
        if not clips:
            return []

        import torch.nn.functional as F

        # Record per-clip boundaries (cumulative frame counts)
        boundaries: list[int] = []
        total = 0
        for clip in clips:
            total += clip.shape[0]
            boundaries.append(total)

        # Concatenate all frames across all clips
        all_frames = np.concatenate(clips, axis=0)  # (total, H, W, 3)

        # CLIP normalization constants
        _mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        _std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

        # Preprocess and infer in chunks (skip PIL for speed)
        all_features = torch.empty(total, 512, dtype=torch.float32)
        num_batches = (total + batch_size - 1) // batch_size
        pbar = tqdm(range(0, total, batch_size), total=num_batches,
                    desc=f"CLIP inference ({total} frames)", leave=False)
        for start in pbar:
            end = min(start + batch_size, total)
            chunk = torch.from_numpy(all_frames[start:end].copy())
            chunk = chunk.permute(0, 3, 1, 2).float().div_(255.0)
            if chunk.shape[2] != 224 or chunk.shape[3] != 224:
                chunk = F.interpolate(
                    chunk, size=(224, 224), mode="bicubic", align_corners=False,
                )
            chunk = (chunk - _mean) / _std
            chunk = chunk.to(self.device)
            feats = self.model.get_image_features(pixel_values=chunk)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features[start:end] = feats.cpu()
            del chunk

        # Re-group by clip and average-pool
        results = []
        prev = 0
        for b in boundaries:
            clip_feats = all_features[prev:b]  # (window_size, 512)
            mean_feat = clip_feats.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()
            results.append(mean_feat.numpy().astype(np.float32))
            prev = b

        return results


class VideoMAEFeatureExtractor:
    """Extract spatio-temporal video features using VideoMAE.

    Unlike CLIP which processes frames independently, VideoMAE uses 3D tubelet
    patches with full spatio-temporal self-attention, capturing motion and
    temporal dynamics across frames.

    Model: MCG-NJU/videomae-base (ViT-B, 768-dim, 16-frame input).
    """

    FEATURE_DIM = 768
    NUM_FRAMES = 16

    def __init__(self, device: str = "cuda"):
        import os

        from transformers import VideoMAEModel

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        model_name = "MCG-NJU/videomae-base"

        _project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = str(_project_root / "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.model = VideoMAEModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        self.model.eval()
        self.device = device

    @staticmethod
    def _sample_indices(total_frames: int, num_samples: int) -> np.ndarray:
        """Uniformly sample frame indices, padding with last frame if needed."""
        if total_frames >= num_samples:
            return np.linspace(0, total_frames - 1, num_samples, dtype=int)
        return np.array(
            list(range(total_frames))
            + [total_frames - 1] * (num_samples - total_frames),
            dtype=int,
        )

    @torch.no_grad()
    def extract_clip_features_batch(
        self, clips: list[np.ndarray], batch_size: int = 512
    ) -> list[np.ndarray]:
        """Extract temporal features for multiple video clips.

        Each clip is processed as a whole video with 3D spatio-temporal
        attention, so features capture motion and temporal dynamics.

        Args:
            clips: List of (window_size, H, W, 3) uint8 numpy arrays.
            batch_size: Max clips per forward pass (clamped to 8 internally
                because VideoMAE is ~16x heavier per sample than CLIP).

        Returns:
            List of (768,) float32 numpy arrays, one per clip.
        """
        if not clips:
            return []

        import torch.nn.functional as F

        effective_bs = min(batch_size, 8)

        # ImageNet normalization (used by VideoMAE)
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

        results = []
        num_batches = (len(clips) + effective_bs - 1) // effective_bs
        pbar = tqdm(range(0, len(clips), effective_bs), total=num_batches,
                    desc=f"VideoMAE inference ({len(clips)} clips)", leave=False)
        for start in pbar:
            batch_clips = clips[start : start + effective_bs]
            B = len(batch_clips)

            # Sample NUM_FRAMES per clip and stack: (B, 16, H, W, 3)
            sampled = []
            for clip in batch_clips:
                indices = self._sample_indices(clip.shape[0], self.NUM_FRAMES)
                sampled.append(clip[indices])
            batch = np.stack(sampled)

            # Manual preprocessing (same as VideoMAEImageProcessor, faster)
            pixel_values = torch.from_numpy(batch.copy()).float().div_(255.0)
            # (B, 16, H, W, 3) -> (B, 16, 3, H, W)
            pixel_values = pixel_values.permute(0, 1, 4, 2, 3)

            if pixel_values.shape[3] != 224 or pixel_values.shape[4] != 224:
                T_f = pixel_values.shape[1]
                pixel_values = pixel_values.reshape(
                    B * T_f, 3, pixel_values.shape[3], pixel_values.shape[4]
                )
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(224, 224),
                    mode="bicubic",
                    align_corners=False,
                )
                pixel_values = pixel_values.reshape(B, T_f, 3, 224, 224)

            pixel_values = (pixel_values - _mean) / _std
            pixel_values = pixel_values.to(self.device)

            outputs = self.model(pixel_values=pixel_values)
            # last_hidden_state: (B, num_patches, 768) -> mean pool -> (B, 768)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)

            for i in range(B):
                results.append(features[i].cpu().numpy().astype(np.float32))
            del pixel_values

        return results


# ---------------------------------------------------------------------------
# Extractor factory
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    "clip": CLIPFeatureExtractor,
    "videomae": VideoMAEFeatureExtractor,
}


def create_extractor(name: str, device: str = "cuda"):
    """Create a feature extractor by name ('clip' or 'videomae')."""
    if name not in _EXTRACTORS:
        raise ValueError(
            f"Unknown extractor '{name}'. Choose from: {list(_EXTRACTORS)}"
        )
    return _EXTRACTORS[name](device=device)


def get_feature_dim(name: str) -> int:
    """Return the output feature dimension for an extractor type."""
    return _EXTRACTORS[name].FEATURE_DIM


# ---------------------------------------------------------------------------
# Feature extraction pipeline
# ---------------------------------------------------------------------------


def _cache_key(
    video_path: str, window_size: int, stride: int, extractor: str = "clip",
) -> str:
    """Compute a cache key for a (video, params, extractor) tuple."""
    h = hashlib.sha256(video_path.encode()).hexdigest()[:16]
    # Backward-compatible: CLIP keys keep old format so existing caches work
    if extractor == "clip":
        return f"{h}_w{window_size}_s{stride}"
    return f"{h}_w{window_size}_s{stride}_{extractor}"


def extract_features_for_trajectories(
    trajectories: list,
    extractor,
    window_size: int,
    stride: int,
    cache_dir: Optional[str] = None,
    batch_size: int = 512,
    extractor_type: str = "clip",
) -> np.ndarray:
    """Extract sliding-window features for a list of trajectories.

    Args:
        trajectories: List of TrajectoryRecord objects (need .video_path).
        extractor: Feature extractor instance (CLIPFeatureExtractor or VideoMAEFeatureExtractor).
        window_size: Sliding window frame count.
        stride: Sliding window step.
        cache_dir: Optional directory for per-video feature caching.
        batch_size: Max items per forward pass.
        extractor_type: Extractor name for cache key disambiguation.

    Returns:
        (total_states, feature_dim) float32 numpy array.
    """
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    all_features = []
    skipped = 0

    for traj in tqdm(trajectories, desc="Extracting features", leave=False):
        video_path = _resolve_video_path(traj.video_path)
        if video_path is None:
            skipped += 1
            continue

        # Check cache
        if cache_dir:
            key = _cache_key(video_path, window_size, stride, extractor_type)
            cache_path = Path(cache_dir) / f"{key}.npy"
            if cache_path.exists():
                feats = np.load(str(cache_path))
                all_features.append(feats)
                continue

        # Load frames and extract clips
        frames = load_all_frames(video_path)
        if frames.shape[0] == 0:
            skipped += 1
            continue

        clips = extract_clips(frames, window_size, stride)

        # Batched feature extraction
        feats_list = extractor.extract_clip_features_batch(clips, batch_size)

        if not feats_list:
            skipped += 1
            continue

        feats = np.stack(feats_list, axis=0)  # (n_clips, feature_dim)

        # Save cache
        if cache_dir:
            np.save(str(cache_path), feats)

        all_features.append(feats)

    if skipped > 0:
        print(f"Warning: skipped {skipped} trajectories (video not found or empty)")

    if not all_features:
        return np.empty((0, extractor.FEATURE_DIM), dtype=np.float32)
    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from trajectory videos"
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="One or more TrajectoryDataset JSON files",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Display labels for each data file (default: filenames)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["file", "policy"],
        default="file",
        help="Grouping strategy: 'file' (per data file) or 'policy' (per model_name)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.npz",
        help="Output .npz file path",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Sliding window frame count (default: 10)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Sliding window step (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (default: auto)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching per-video features",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Max trajectories per group (for down-sampling large datasets)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs for parallel feature extraction (default: 8)",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["clip", "videomae"],
        default="clip",
        help="Feature extractor: 'clip' (per-frame, 512-d) or 'videomae' (temporal, 768-d)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Max images per CLIP forward pass (default: 512)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from rlinf.data.trajectory_dataset import TrajectoryDataset

    # Load datasets
    datasets = []
    for p in args.data:
        print(f"Loading {p} ...")
        ds = TrajectoryDataset.load(p)
        datasets.append(ds)
        print(f"  {len(ds.trajectories)} trajectories")

    # Build labels
    if args.labels:
        if len(args.labels) != len(args.data):
            print(
                f"Error: {len(args.labels)} labels provided "
                f"for {len(args.data)} data files"
            )
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(p).stem for p in args.data]

    # Group trajectories
    groups: dict[str, list] = {}
    if args.group_by == "file":
        for label, ds in zip(labels, datasets):
            groups[label] = list(ds.trajectories)
    elif args.group_by == "policy":
        all_by_policy: dict[str, list] = defaultdict(list)
        for ds in datasets:
            for t in ds.trajectories:
                all_by_policy[t.model_name].append(t)
        groups = dict(all_by_policy)

    # Down-sample if requested
    if args.max_trajectories:
        rng = np.random.default_rng(42)
        for label in groups:
            trajs = groups[label]
            if len(trajs) > args.max_trajectories:
                indices = rng.choice(
                    len(trajs), args.max_trajectories, replace=False
                )
                groups[label] = [trajs[i] for i in sorted(indices)]

    # Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_gpus = args.num_gpus if device.startswith("cuda") else 1

    # Extract features per group
    all_features = []
    all_labels = []

    if num_gpus <= 1:
        # Single-GPU path
        print(f"Using device: {device}")
        print(f"Loading {args.extractor} model ...")
        extractor = create_extractor(args.extractor, device=device)
        for label, trajs in groups.items():
            print(f"Extracting features for '{label}' ({len(trajs)} trajectories) ...")
            feats = extract_features_for_trajectories(
                trajs, extractor, args.window_size, args.window_stride,
                args.cache_dir, batch_size=args.batch_size,
                extractor_type=args.extractor,
            )
            if feats.shape[0] == 0:
                print(f"  Warning: no features extracted for '{label}'")
                continue
            print(f"  {feats.shape[0]} states extracted (dim={feats.shape[1]})")
            all_features.append(feats)
            all_labels.extend([label] * feats.shape[0])
    else:
        # Multi-GPU path: shard all trajectories across GPUs by round-robin
        print(f"Using {num_gpus} GPUs for parallel extraction")

        flat_trajs = []
        flat_labels = []
        for label, trajs in groups.items():
            for t in trajs:
                flat_trajs.append(t)
                flat_labels.append(label)

        shards = [[] for _ in range(num_gpus)]
        shard_labels = [[] for _ in range(num_gpus)]
        for i, (t, lbl) in enumerate(zip(flat_trajs, flat_labels)):
            rank = i % num_gpus
            shards[rank].append(t)
            shard_labels[rank].append(lbl)

        manager = mp.Manager()
        result_dict = manager.dict()

        def _gpu_worker(rank, shard, shard_lbls, window_size, stride,
                        cache_dir, result_dict, batch_size, extractor_type):
            """Worker: extract features on one GPU, return (features, labels)."""
            dev = f"cuda:{rank}"
            ext = create_extractor(extractor_type, device=dev)
            feat_dim = ext.FEATURE_DIM
            feat_chunks = []
            out_labels = []
            for traj, lbl in tqdm(
                list(zip(shard, shard_lbls)),
                desc=f"GPU {rank}",
                position=rank,
                leave=True,
            ):
                video_path = _resolve_video_path(traj.video_path)
                if video_path is None:
                    continue
                if cache_dir:
                    key = _cache_key(video_path, window_size, stride, extractor_type)
                    cp = Path(cache_dir) / f"{key}.npy"
                    if cp.exists():
                        f = np.load(str(cp))
                        feat_chunks.append(f)
                        out_labels.extend([lbl] * f.shape[0])
                        continue
                frames = load_all_frames(video_path)
                if frames.shape[0] == 0:
                    continue
                clips = extract_clips(frames, window_size, stride)
                fl = ext.extract_clip_features_batch(clips, batch_size=batch_size)
                if not fl:
                    continue
                f = np.stack(fl, axis=0)
                if cache_dir:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(str(cp), f)
                feat_chunks.append(f)
                out_labels.extend([lbl] * f.shape[0])
            if feat_chunks:
                result_dict[rank] = (np.concatenate(feat_chunks, axis=0), out_labels)
            else:
                result_dict[rank] = (np.empty((0, feat_dim), dtype=np.float32), [])

        feat_dim = get_feature_dim(args.extractor)

        processes = []
        for rank in range(num_gpus):
            if not shards[rank]:
                result_dict[rank] = (np.empty((0, feat_dim), dtype=np.float32), [])
                continue
            p = mp.Process(
                target=_gpu_worker,
                args=(rank, shards[rank], shard_labels[rank],
                      args.window_size, args.window_stride,
                      args.cache_dir, result_dict, args.batch_size,
                      args.extractor),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for rank in range(num_gpus):
            feats, lbls = result_dict[rank]
            if feats.shape[0] > 0:
                all_features.append(feats)
                all_labels.extend(lbls)
        print(f"All GPUs finished.")

    if not all_features:
        print("Error: no features extracted from any group")
        sys.exit(1)

    features = np.concatenate(all_features, axis=0)
    group_labels = np.array(all_labels)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        features=features,
        group_labels=group_labels,
        metadata=np.array(
            {
                "extractor": args.extractor,
                "window_size": args.window_size,
                "window_stride": args.window_stride,
                "group_by": args.group_by,
                "data_files": args.data,
            }
        ),
    )
    print(f"\nSaved {features.shape[0]} features to {args.output}")
    for label in dict.fromkeys(all_labels):
        count = sum(1 for l in all_labels if l == label)
        print(f"  {label}: {count} states")


if __name__ == "__main__":
    main()
