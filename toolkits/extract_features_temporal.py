#!/usr/bin/env python3
"""Extract temporally-structured CLIP features from trajectory videos.

Unlike ``extract_features.py`` which flattens all sliding-window clips into a
single pool, this script preserves per-trajectory, per-timestep structure.  It
uses the **same** sliding-window approach (window_size + stride) as the original
script, but keeps track of which clips belong to which trajectory and at which
temporal position — producing a 3-D feature tensor suitable for the
parallel-planes visualization in ``visualize_temporal_umap.py``.

Example: for a 50-frame video with window_size=10, stride=5, clips are extracted
at frames 0-9, 5-14, 10-19, … , 40-49 (9 clips).  Each clip becomes one
time-plane in the final visualization.

Output ``.npz`` format::

    features     : (num_trajectories, max_clips, feature_dim) float32
    group_labels : (num_trajectories,) str
    valid_mask   : (num_trajectories, max_clips) bool
    metadata     : dict — extraction parameters

Usage::

    python toolkits/extract_features_temporal.py \\
        --data file1.json file2.json \\
        --labels "Policy A" "Policy B" \\
        --window-size 10 --window-stride 5 \\
        --output features_temporal.npz
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Single-trajectory extraction (sliding window, with cache)
# ---------------------------------------------------------------------------


def extract_one_trajectory_sliding(
    traj,
    extractor,
    window_size: int,
    stride: int,
    batch_size: int = 512,
    cache_dir: str | None = None,
    extractor_type: str = "clip",
    truncate_at_success: bool = False,
) -> np.ndarray | None:
    """Extract sliding-window features for one trajectory.

    Args:
        truncate_at_success: If True and the trajectory has a
            ``first_success_step``, only use video frames up to that step.

    Returns:
        (num_clips, feature_dim) float32 array, or *None* if the video is
        missing / empty.  The array is compatible with the cache format used
        by ``extract_features.py``.
    """
    from toolkits.extract_features import (
        _cache_key,
        _resolve_video_path,
        extract_clips,
        load_all_frames,
    )

    video_path = _resolve_video_path(traj.video_path)
    if video_path is None:
        return None

    # Build a cache key suffix when truncation is active so truncated and
    # full extractions are cached separately.
    trunc_step = None
    if truncate_at_success and getattr(traj, "first_success_step", None) is not None:
        trunc_step = traj.first_success_step

    cache_suffix = f"_trunc{trunc_step}" if trunc_step is not None else ""

    # Check cache (same key scheme as extract_features.py → reusable)
    if cache_dir:
        key = _cache_key(video_path, window_size, stride, extractor_type) + cache_suffix
        cache_path = Path(cache_dir) / f"{key}.npy"
        if cache_path.exists():
            return np.load(str(cache_path))

    frames = load_all_frames(video_path)
    if frames.shape[0] == 0:
        return None

    # Truncate frames at first success step if requested
    if trunc_step is not None and trunc_step < frames.shape[0]:
        # Keep frames up to and including the success step
        frames = frames[: trunc_step + 1]

    clips = extract_clips(frames, window_size, stride)
    if not clips:
        return None

    feat_list = extractor.extract_clip_features_batch(clips, batch_size)
    if not feat_list:
        return None

    feats = np.stack(feat_list, axis=0)  # (num_clips, dim)

    # Save cache
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), feats)

    return feats


# ---------------------------------------------------------------------------
# Multi-GPU worker (module-level for spawn safety)
# ---------------------------------------------------------------------------


def _gpu_worker_temporal(
    rank, shard, shard_lbls, window_size, stride, cache_dir, result_dict,
    batch_size, extractor_type, truncate_at_success=False,
):
    """Extract sliding-window temporal features on a single GPU.

    Processes trajectories one at a time to keep memory usage bounded.
    """
    from toolkits.extract_features import (
        _cache_key,
        _resolve_video_path,
        create_extractor,
        extract_clips,
        load_all_frames,
    )

    dev = f"cuda:{rank}"

    # --- Phase 1: Separate cached vs uncached trajectories ---
    cached_results: dict[int, tuple[np.ndarray, str]] = {}
    to_process: list[tuple[int, str, str, int | None]] = []

    for i, (traj, lbl) in enumerate(zip(shard, shard_lbls)):
        video_path = _resolve_video_path(traj.video_path)
        if video_path is None:
            continue

        trunc_step = None
        if truncate_at_success and getattr(traj, "first_success_step", None) is not None:
            trunc_step = traj.first_success_step
        cache_suffix = f"_trunc{trunc_step}" if trunc_step is not None else ""

        if cache_dir:
            key = _cache_key(video_path, window_size, stride, extractor_type) + cache_suffix
            cache_path = Path(cache_dir) / f"{key}.npy"
            if cache_path.exists():
                cached_results[i] = (np.load(str(cache_path)), lbl)
                continue
        to_process.append((i, video_path, lbl, trunc_step))

    print(f"GPU {rank}: {len(cached_results)} cached, {len(to_process)} to extract")

    if to_process:
        # --- Phase 2: Load & infer in chunks to bound memory ---
        # With 6 GPU workers × ~834 trajs × ~17 clips × 2.4MB/clip ≈ 34GB
        # per worker, loading all at once can OOM.  Process in chunks of
        # CHUNK_TRAJS trajectories (~8GB clip data per chunk).
        import os
        from concurrent.futures import ThreadPoolExecutor

        CHUNK_TRAJS = 200  # ~200 trajs × 17 clips × 2.4MB ≈ 8GB

        ext = create_extractor(extractor_type, device=dev)
        total_clips = 0
        total_trajs = 0

        for chunk_start in range(0, len(to_process), CHUNK_TRAJS):
            chunk = to_process[chunk_start : chunk_start + CHUNK_TRAJS]

            # Parallel video loading within this chunk
            def _load_and_clip(item):
                idx, vp, lbl, trunc_step = item
                frames = load_all_frames(vp)
                if frames.shape[0] == 0:
                    return (idx, None, lbl, vp, trunc_step)
                if trunc_step is not None and trunc_step < frames.shape[0]:
                    frames = frames[: trunc_step + 1]
                clips = extract_clips(frames, window_size, stride)
                if not clips:
                    return (idx, None, lbl, vp, trunc_step)
                return (idx, clips, lbl, vp, trunc_step)

            num_workers = min(os.cpu_count() or 4, len(chunk), 8)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                loaded = list(tqdm(
                    pool.map(_load_and_clip, chunk),
                    total=len(chunk),
                    desc=f"GPU {rank} chunk {chunk_start // CHUNK_TRAJS + 1}",
                    position=rank,
                    leave=True,
                ))

            # Aggregate clips from this chunk into one mega-batch
            mega_clips: list[np.ndarray] = []
            clip_traj_map: list[int] = []
            valid_loaded: list[tuple[int, list, str, str, int | None]] = []

            for idx, clips, lbl, vp, trunc_step in loaded:
                if clips is None:
                    continue
                li = len(valid_loaded)
                valid_loaded.append((idx, clips, lbl, vp))
                for clip in clips:
                    mega_clips.append(clip)
                    clip_traj_map.append(li)

            del loaded

            if mega_clips:
                all_feats = ext.extract_clip_features_batch(
                    mega_clips, batch_size,
                )
                del mega_clips

                # Map features back to trajectories
                from collections import defaultdict
                feat_by_traj: dict[int, list[np.ndarray]] = defaultdict(list)
                for feat, li in zip(all_feats, clip_traj_map):
                    feat_by_traj[li].append(feat)
                del all_feats, clip_traj_map

                for li, (idx, clips, lbl, vp, trunc_step) in enumerate(valid_loaded):
                    feats = np.stack(feat_by_traj[li])
                    total_clips += feats.shape[0]
                    total_trajs += 1
                    if cache_dir:
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        cache_suffix = f"_trunc{trunc_step}" if trunc_step is not None else ""
                        key = _cache_key(vp, window_size, stride, extractor_type) + cache_suffix
                        cp = Path(cache_dir) / f"{key}.npy"
                        np.save(str(cp), feats)
                    cached_results[idx] = (feats, lbl)

                del feat_by_traj, valid_loaded

        print(
            f"GPU {rank}: {extractor_type} on {total_clips} clips "
            f"from {total_trajs} trajectories"
        )

    # --- Assemble ordered results ---
    results: list[np.ndarray] = []
    labels: list[str] = []
    for i in sorted(cached_results.keys()):
        feats, lbl = cached_results[i]
        results.append(feats)
        labels.append(lbl)

    result_dict[rank] = (results, labels)


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_temporal_features(
    data_paths: list[str],
    labels: list[str],
    group_by: str = "file",
    window_size: int = 10,
    stride: int = 5,
    device: str | None = None,
    max_trajectories: int | None = None,
    num_gpus: int = 8,
    batch_size: int = 512,
    cache_dir: str | None = None,
    extractor_type: str = "clip",
    truncate_at_success: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract temporal features from JSON trajectory datasets.

    Returns:
        features:     (N, max_clips, D) float32  — NaN-padded
        group_labels: (N,) str array
        valid_mask:   (N, max_clips) bool
    """
    from rlinf.data.trajectory_dataset import TrajectoryDataset

    # Load datasets
    datasets = []
    for p in data_paths:
        print(f"Loading {p} ...")
        ds = TrajectoryDataset.load(p)
        datasets.append(ds)
        print(f"  {len(ds.trajectories)} trajectories")

    # Group trajectories
    groups: dict[str, list] = {}
    if group_by == "file":
        for label, ds in zip(labels, datasets):
            groups[label] = list(ds.trajectories)
    elif group_by == "policy":
        all_by_policy: dict[str, list] = defaultdict(list)
        for ds in datasets:
            for t in ds.trajectories:
                all_by_policy[t.model_name].append(t)
        groups = dict(all_by_policy)

    # Down-sample
    if max_trajectories:
        rng = np.random.default_rng(42)
        for label in groups:
            trajs = groups[label]
            if len(trajs) > max_trajectories:
                idx = rng.choice(len(trajs), max_trajectories, replace=False)
                groups[label] = [trajs[i] for i in sorted(idx)]

    # Flatten for processing
    flat_trajs: list = []
    flat_labels: list[str] = []
    for label, trajs in groups.items():
        for t in trajs:
            flat_trajs.append(t)
            flat_labels.append(label)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_multi_gpu = num_gpus > 1 and device.startswith("cuda")

    # Per-trajectory feature arrays (variable length)
    traj_features: list[np.ndarray] = []
    traj_labels: list[str] = []

    if not use_multi_gpu:
        from toolkits.extract_features import create_extractor

        print(f"Using device: {device}")
        print(f"Loading {extractor_type} model ...")
        ext = create_extractor(extractor_type, device=device)

        for traj, lbl in tqdm(
            list(zip(flat_trajs, flat_labels)), desc="Extracting"
        ):
            feats = extract_one_trajectory_sliding(
                traj, ext, window_size, stride, batch_size, cache_dir,
                extractor_type, truncate_at_success,
            )
            if feats is not None:
                traj_features.append(feats)
                traj_labels.append(lbl)
    else:
        ctx = mp.get_context("spawn")
        print(f"Using {num_gpus} GPUs for parallel extraction")

        shards: list[list] = [[] for _ in range(num_gpus)]
        shard_labels: list[list[str]] = [[] for _ in range(num_gpus)]
        for i, (t, lbl) in enumerate(zip(flat_trajs, flat_labels)):
            rank = i % num_gpus
            shards[rank].append(t)
            shard_labels[rank].append(lbl)

        manager = ctx.Manager()
        result_dict = manager.dict()

        processes: list[tuple[int, mp.Process]] = []
        for rank in range(num_gpus):
            if not shards[rank]:
                result_dict[rank] = ([], [])
                continue
            p = ctx.Process(
                target=_gpu_worker_temporal,
                args=(
                    rank, shards[rank], shard_labels[rank],
                    window_size, stride, cache_dir, result_dict, batch_size,
                    extractor_type, truncate_at_success,
                ),
            )
            p.start()
            processes.append((rank, p))

        for rank, p in processes:
            p.join()

        # Collect results; for failed workers, fall back to single-GPU
        failed_ranks: list[int] = []
        for rank, p in processes:
            if p.exitcode != 0 or rank not in result_dict:
                print(
                    f"WARNING: GPU {rank} worker failed "
                    f"(exit code {p.exitcode}), will retry on CPU/GPU 0"
                )
                failed_ranks.append(rank)
            else:
                feat_list, lbls = result_dict[rank]
                traj_features.extend(feat_list)
                traj_labels.extend(lbls)

        # Retry failed shards sequentially on first available device
        if failed_ranks:
            from toolkits.extract_features import create_extractor

            retry_dev = device or "cuda:0"
            print(f"Retrying {len(failed_ranks)} failed shards on {retry_dev} ...")
            ext = create_extractor(extractor_type, device=retry_dev)

            for rank in failed_ranks:
                for traj, lbl in tqdm(
                    list(zip(shards[rank], shard_labels[rank])),
                    desc=f"Retry shard {rank}",
                ):
                    feats = extract_one_trajectory_sliding(
                        traj, ext, window_size, stride, batch_size,
                        cache_dir, extractor_type, truncate_at_success,
                    )
                    if feats is not None:
                        traj_features.append(feats)
                        traj_labels.append(lbl)

        # Also collect results from empty shards
        for rank in range(num_gpus):
            if rank in result_dict and rank not in [r for r, _ in processes]:
                feat_list, lbls = result_dict[rank]
                traj_features.extend(feat_list)
                traj_labels.extend(lbls)

        print("All GPUs finished.")

    if not traj_features:
        print("Error: no features extracted")
        sys.exit(1)

    # Pad variable-length sequences to (N, max_clips, D)
    max_clips = max(f.shape[0] for f in traj_features)
    dim = traj_features[0].shape[1]
    N = len(traj_features)

    features = np.full((N, max_clips, dim), np.nan, dtype=np.float32)
    for i, f in enumerate(traj_features):
        features[i, : f.shape[0]] = f

    group_labels = np.array(traj_labels)
    valid_mask = ~np.isnan(features[:, :, 0])

    clip_counts = [f.shape[0] for f in traj_features]
    print(
        f"Clip counts: min={min(clip_counts)}, max={max(clip_counts)}, "
        f"median={int(np.median(clip_counts))}"
    )

    return features, group_labels, valid_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract temporally-structured CLIP features"
    )
    parser.add_argument(
        "--data", type=str, nargs="+", required=True,
        help="TrajectoryDataset JSON files",
    )
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument(
        "--group-by", type=str, choices=["file", "policy"], default="file",
    )
    parser.add_argument("--output", type=str, default="features_temporal.npz")
    parser.add_argument(
        "--window-size", type=int, default=10,
        help="Sliding window frame count (default: 10)",
    )
    parser.add_argument(
        "--window-stride", type=int, default=5,
        help="Sliding window step (default: 5)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--extractor", type=str, choices=["clip", "videomae"], default="clip",
        help="Feature extractor: 'clip' (per-frame, 512-d) or 'videomae' (temporal, 768-d)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for caching per-video features (shared with extract_features.py)",
    )
    parser.add_argument(
        "--truncate-at-success", action="store_true", default=False,
        help="Truncate each trajectory at its first success step before extracting features",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build labels
    if args.labels:
        if len(args.labels) != len(args.data):
            print(
                f"Error: {len(args.labels)} labels for {len(args.data)} files"
            )
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(p).stem for p in args.data]

    features, group_labels, valid_mask = extract_temporal_features(
        data_paths=args.data,
        labels=labels,
        group_by=args.group_by,
        window_size=args.window_size,
        stride=args.window_stride,
        device=args.device,
        max_trajectories=args.max_trajectories,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        extractor_type=args.extractor,
        truncate_at_success=args.truncate_at_success,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        features=features,
        group_labels=group_labels,
        valid_mask=valid_mask,
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
    N, T, D = features.shape
    print(f"\nSaved {N} trajectories x {T} max timesteps to {args.output}")
    for label in dict.fromkeys(group_labels):
        count = np.sum(group_labels == label)
        print(f"  {label}: {count} trajectories")


if __name__ == "__main__":
    main()
