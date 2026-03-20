#!/usr/bin/env python3
"""Quality distribution visualizer for TrajectoryDataset files.

Displays histograms, box plots, and summary statistics for trajectory quality
scores across one or more datasets.

Usage::

    python toolkits/visualize_quality.py --data ds1.json ds2.json --labels "Round 1" "Round 2"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rlinf.data.quality_evaluation import compute_dataset_quality
from rlinf.data.trajectory_dataset import TrajectoryDataset


def _build_stats(scores: list[float]) -> dict:
    arr = np.array(scores)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Quality distribution visualizer")
    parser.add_argument(
        "--data", nargs="+", required=True, help="TrajectoryDataset JSON paths"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Display labels for each dataset"
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--bins", type=int, default=20, help="Number of histogram bins"
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots as PNG files and exit (no Gradio server)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quality_distribution.png",
        help="Output path prefix for saved plots (used with --save-only)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = args.labels or [Path(p).stem for p in args.data]
    if len(labels) != len(args.data):
        labels = [Path(p).stem for p in args.data]

    # Load and score datasets
    datasets: list[TrajectoryDataset] = []
    all_scores: list[list[float]] = []
    all_stats: list[dict] = []
    for path in args.data:
        ds = TrajectoryDataset.load(path)
        compute_dataset_quality(ds)
        datasets.append(ds)
        scores = [
            t.quality_score for t in ds.trajectories if t.quality_score is not None
        ]
        all_scores.append(scores)
        all_stats.append(_build_stats(scores) if scores else {})

    # Build plots
    def make_histogram():
        fig, ax = plt.subplots(figsize=(10, 5))
        for scores, label in zip(all_scores, labels):
            if scores:
                ax.hist(scores, bins=args.bins, alpha=0.6, label=label)
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.set_title("Quality Score Distribution")
        ax.legend()
        plt.tight_layout()
        return fig

    def make_boxplot():
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [s for s in all_scores if s]
        bp_labels = [l for l, s in zip(labels, all_scores) if s]
        if data:
            ax.boxplot(data, labels=bp_labels)
        ax.set_ylabel("Quality Score")
        ax.set_title("Quality Score Box Plot")
        plt.tight_layout()
        return fig

    # Build stats table
    table_data = []
    for label, stats in zip(labels, all_stats):
        if stats:
            success_count = sum(
                1
                for t in datasets[labels.index(label)].trajectories
                if t.success is True
            )
            total = stats["count"]
            success_rate = success_count / total if total > 0 else 0
            table_data.append([
                label,
                stats["count"],
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{success_rate:.1%}",
            ])

    # --- Save-only mode: write PNGs and exit ---
    if args.save_only:
        output_base = Path(args.output)
        output_dir = output_base.parent
        stem = output_base.stem
        suffix = output_base.suffix or ".png"
        output_dir.mkdir(parents=True, exist_ok=True)

        hist_path = output_dir / f"{stem}_histogram{suffix}"
        box_path = output_dir / f"{stem}_boxplot{suffix}"

        fig_hist = make_histogram()
        fig_hist.savefig(str(hist_path), dpi=200, bbox_inches="tight")
        plt.close(fig_hist)
        print(f"Saved histogram to {hist_path}")

        fig_box = make_boxplot()
        fig_box.savefig(str(box_path), dpi=200, bbox_inches="tight")
        plt.close(fig_box)
        print(f"Saved box plot to {box_path}")

        # Print stats table
        print("\nSummary Statistics:")
        headers = ["Dataset", "Count", "Mean", "Std", "Median", "Min", "Max", "Success Rate"]
        print(" | ".join(headers))
        print("-" * 80)
        for row in table_data:
            print(" | ".join(str(x) for x in row))

        print("\nDone (save-only mode).")
        return

    # --- Gradio mode ---
    import gradio as gr

    with gr.Blocks(title="Quality Distribution Viewer") as app:
        gr.Markdown("## Trajectory Quality Distribution")

        with gr.Row():
            gr.Plot(value=make_histogram, label="Histogram")
            gr.Plot(value=make_boxplot, label="Box Plot")

        gr.Dataframe(
            value=table_data,
            headers=[
                "Dataset", "Count", "Mean", "Std", "Median", "Min", "Max",
                "Success Rate",
            ],
            interactive=False,
            label="Summary Statistics",
        )

    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
