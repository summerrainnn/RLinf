#!/usr/bin/env python3
"""Visualize state distributions using UMAP dimensionality reduction.

Reads ``.npz`` feature files produced by ``extract_features.py`` (or other
compatible extractors), reduces them to 2-D with UMAP, and renders a
scatter + KDE plot matching the project's reference style.

For convenience, ``--data`` can be used instead of ``--features`` to directly
pass TrajectoryDataset JSON files — features will be extracted on the fly.

Output ``.npz`` format expected::

    features     : (N, feature_dim) float32
    group_labels : (N,) str

Usage::

    # From pre-extracted features
    python toolkits/visualize_umap.py \\
        --features features.npz \\
        --output state_distribution.png

    # Convenience: directly from JSON (auto-extracts features)
    python toolkits/visualize_umap.py \\
        --data file1.json file2.json \\
        --labels "Policy A" "Policy B" \\
        --output state_distribution.png
"""

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_features(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load and merge features from one or more ``.npz`` files.

    Returns:
        (features, group_labels) — concatenated across all files.
    """
    all_features = []
    all_labels = []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        all_features.append(data["features"])
        all_labels.append(data["group_labels"])
    features = np.concatenate(all_features, axis=0)
    group_labels = np.concatenate(all_labels, axis=0)
    return features, group_labels


# ---------------------------------------------------------------------------
# UMAP reduction
# ---------------------------------------------------------------------------


def reduce_with_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Reduce features to 2-D with UMAP.

    Args:
        features: (N, D) float32 array.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.

    Returns:
        (N, 2) float32 array.
    """
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(features)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Reference-image colour palette (red, blue, green, orange, purple, teal)
_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]


def plot_state_distribution(
    embeddings_by_group: dict[str, np.ndarray],
    output_path: str,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> None:
    """Render scatter + KDE plot matching the reference style.

    Style: white background, filled KDE contours with gradient transparency,
    small semi-transparent scatter dots, top-centered legend, no axes.

    Args:
        embeddings_by_group: ``{group_label: (N, 2)}`` dict.
        output_path: Destination image file path.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Output resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, (label, emb) in enumerate(embeddings_by_group.items()):
        color = _COLORS[i % len(_COLORS)]

        # KDE filled contours (gradient via overlapping levels)
        try:
            sns.kdeplot(
                x=emb[:, 0],
                y=emb[:, 1],
                fill=True,
                alpha=0.35,
                levels=10,
                color=color,
                ax=ax,
                bw_adjust=0.8,
            )
        except Exception:
            # KDE can fail with too few points — fall back to scatter only
            pass

        # Scatter dots
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            s=10,
            alpha=0.6,
            color=color,
            label=label,
            edgecolors="none",
            zorder=5,
        )

    # Legend at the top, matching reference layout
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(embeddings_by_group), 4),
        fontsize=12,
        frameon=False,
        markerscale=3,
    )

    # Clean axes — no ticks, no spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Convenience: direct JSON → features extraction
# ---------------------------------------------------------------------------


def _gpu_worker_fn(rank, shard, shard_lbls, ws, st, cd, rd, bs):
    """Module-level worker function for multi-GPU feature extraction (spawn-safe)."""
    from toolkits.extract_features import (
        CLIPFeatureExtractor,
        _cache_key,
        _resolve_video_path,
        extract_clips,
        load_all_frames,
    )
    from tqdm import tqdm

    dev = f"cuda:{rank}"
    ext = CLIPFeatureExtractor(device=dev)
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
        if cd:
            key = _cache_key(video_path, ws, st)
            cp = Path(cd) / f"{key}.npy"
            if cp.exists():
                f = np.load(str(cp))
                feat_chunks.append(f)
                out_labels.extend([lbl] * f.shape[0])
                continue
        frames = load_all_frames(video_path)
        if frames.shape[0] == 0:
            continue
        clips = extract_clips(frames, ws, st)
        fl = ext.extract_clip_features_batch(clips, batch_size=bs)
        if not fl:
            continue
        f = np.stack(fl, axis=0)
        if cd:
            Path(cd).mkdir(parents=True, exist_ok=True)
            np.save(str(cp), f)
        feat_chunks.append(f)
        out_labels.extend([lbl] * f.shape[0])
    if feat_chunks:
        rd[rank] = (np.concatenate(feat_chunks, axis=0), out_labels)
    else:
        rd[rank] = (np.empty((0, 512), dtype=np.float32), [])


def _extract_features_from_json(
    data_paths: list[str],
    labels: list[str],
    group_by: str,
    window_size: int,
    window_stride: int,
    device: str,
    max_trajectories: int | None,
    cache_dir: str | None,
    num_gpus: int = 8,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run feature extraction in-process, with optional multi-GPU support."""
    from toolkits.extract_features import (
        CLIPFeatureExtractor,
        _cache_key,
        _resolve_video_path,
        extract_clips,
        extract_features_for_trajectories,
        load_all_frames,
    )

    from rlinf.data.trajectory_dataset import TrajectoryDataset

    import torch

    # Load datasets
    datasets = []
    for p in data_paths:
        print(f"Loading {p} ...")
        datasets.append(TrajectoryDataset.load(p))

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
                indices = rng.choice(len(trajs), max_trajectories, replace=False)
                groups[label] = [trajs[i] for i in sorted(indices)]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_multi_gpu = num_gpus > 1 and device.startswith("cuda")

    all_features = []
    all_labels = []

    if not use_multi_gpu:
        print(f"Using device: {device}")
        print("Loading CLIP model ...")
        extractor = CLIPFeatureExtractor(device=device)
        for label, trajs in groups.items():
            print(f"Extracting features for '{label}' ({len(trajs)} trajectories) ...")
            feats = extract_features_for_trajectories(
                trajs, extractor, window_size, window_stride, cache_dir,
                batch_size=batch_size,
            )
            if feats.shape[0] == 0:
                print(f"  Warning: no features extracted for '{label}'")
                continue
            print(f"  {feats.shape[0]} states (dim={feats.shape[1]})")
            all_features.append(feats)
            all_labels.extend([label] * feats.shape[0])
    else:
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
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

        manager = ctx.Manager()
        result_dict = manager.dict()

        processes = []
        for rank in range(num_gpus):
            if not shards[rank]:
                result_dict[rank] = (np.empty((0, 512), dtype=np.float32), [])
                continue
            p = ctx.Process(
                target=_gpu_worker_fn,
                args=(rank, shards[rank], shard_labels[rank],
                      window_size, window_stride, cache_dir, result_dict,
                      batch_size),
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
        print("All GPUs finished.")

    if not all_features:
        print("Error: no features extracted")
        sys.exit(1)

    return np.concatenate(all_features, axis=0), np.array(all_labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP state-distribution visualization"
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Pre-extracted .npz feature file(s)",
    )
    input_group.add_argument(
        "--data",
        type=str,
        nargs="+",
        help="TrajectoryDataset JSON file(s) — features extracted on the fly",
    )

    # Labeling / grouping (for --data mode)
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Display labels for each data file (--data mode only)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["file", "policy"],
        default="file",
        help="Grouping strategy (--data mode only)",
    )

    # Feature extraction params (for --data mode)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs for parallel feature extraction")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Max images per CLIP forward pass (default: 512)")

    # UMAP params
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="UMAP n_neighbors"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="state_distribution.png",
        help="Output image path",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,8",
        help="Figure size as 'width,height' (default: '10,8')",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------
    # 1. Obtain features + group labels
    # -------------------------------------------------------------------
    if args.features:
        print(f"Loading features from {args.features} ...")
        features, group_labels = load_features(args.features)
    else:
        # --data mode: extract on the fly
        labels = args.labels
        if labels is None:
            raw_labels = [Path(p).stem for p in args.data]
            if len(set(raw_labels)) < len(raw_labels):
                # File names collide — prepend parent directory name
                labels = [
                    f"{Path(p).parent.name}/{Path(p).stem}" for p in args.data
                ]
            else:
                labels = raw_labels
        elif len(labels) != len(args.data):
            print(
                f"Error: {len(labels)} labels for {len(args.data)} data files"
            )
            sys.exit(1)

        features, group_labels = _extract_features_from_json(
            data_paths=args.data,
            labels=labels,
            group_by=args.group_by,
            window_size=args.window_size,
            window_stride=args.window_stride,
            device=args.device,
            max_trajectories=args.max_trajectories,
            cache_dir=args.cache_dir,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
        )

    print(f"Total states: {features.shape[0]}, feature dim: {features.shape[1]}")
    unique_labels = list(dict.fromkeys(group_labels))
    for label in unique_labels:
        count = np.sum(group_labels == label)
        print(f"  {label}: {count} states")

    # -------------------------------------------------------------------
    # 2. UMAP dimensionality reduction
    # -------------------------------------------------------------------
    print(
        f"Running UMAP (n_neighbors={args.n_neighbors}, "
        f"min_dist={args.min_dist}) ..."
    )
    embeddings = reduce_with_umap(
        features, n_neighbors=args.n_neighbors, min_dist=args.min_dist
    )

    # -------------------------------------------------------------------
    # 3. Split by group and plot
    # -------------------------------------------------------------------
    embeddings_by_group: dict[str, np.ndarray] = {}
    for label in unique_labels:
        mask = group_labels == label
        embeddings_by_group[label] = embeddings[mask]

    figsize = tuple(float(x) for x in args.figsize.split(","))
    plot_state_distribution(
        embeddings_by_group,
        output_path=args.output,
        figsize=figsize,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
