#!/usr/bin/env python3
"""Web-based visualizer for preference / trajectory data.

Supports both legacy ``.pkl`` (PreferencePair) and new ``.json``
(TrajectoryDataset) formats.

Usage::

    # Legacy pkl format
    python toolkits/visualize_preference_data.py --data path/to/preference.pkl

    # New JSON format
    python toolkits/visualize_preference_data.py --data path/to/trajectories.json

    # With a YAML config
    python toolkits/visualize_preference_data.py --config toolkits/config/visualize_preference.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_video_path(video_path):
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


def load_data(path: str):
    """Load data and return a TrajectoryDataset (auto-detect format)."""
    from rlinf.data.trajectory_dataset import TrajectoryDataset

    if path.endswith(".json"):
        return TrajectoryDataset.load(path)

    # Legacy pkl format
    from rlinf.data.preference_data import load_preference_pairs

    pairs = load_preference_pairs(path)
    return TrajectoryDataset.from_preference_pairs(pairs)


# ---------------------------------------------------------------------------
# Win rate computation
# ---------------------------------------------------------------------------


def compute_winrate(dataset, scorer_name: str) -> dict:
    """Compute win rate from a dataset using a specific scorer."""
    scoring = None
    for sr in dataset.scoring_results:
        if sr.scorer_name == scorer_name:
            scoring = sr
            break
    if scoring is None:
        return {"wins": {}, "total": 0}

    models = set(t.model_name for t in dataset.trajectories)
    wins = {m: 0 for m in models}
    ties = 0
    total = 0
    for group in dataset.groups:
        if len(group) != 2:
            continue
        t0, t1 = dataset.trajectories[group[0]], dataset.trajectories[group[1]]
        s0 = scoring.scores.get(group[0])
        s1 = scoring.scores.get(group[1])
        if s0 is None or s1 is None:
            continue
        total += 1
        if s0 > s1:
            wins[t0.model_name] = wins.get(t0.model_name, 0) + 1
        elif s1 > s0:
            wins[t1.model_name] = wins.get(t1.model_name, 0) + 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "total": total}


# ---------------------------------------------------------------------------
# Legacy pair helpers
# ---------------------------------------------------------------------------


def _pair_category(t_chosen, t_rejected) -> str:
    if t_chosen.success and not t_rejected.success:
        return "SF"
    if t_chosen.success and t_rejected.success:
        return "SS"
    if not t_chosen.success and not t_rejected.success:
        return "FF"
    return "MX"


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------


def build_app(dataset, data_path: str):
    import gradio as gr
    from rlinf.data.trajectory_dataset import TrajectoryDataset

    n_traj = len(dataset.trajectories)
    n_groups = len(dataset.groups)
    n_scorings = len(dataset.scoring_results)

    # Gather policy names
    policy_names = sorted(set(t.model_name for t in dataset.trajectories))
    policy_counts = {}
    for t in dataset.trajectories:
        policy_counts[t.model_name] = policy_counts.get(t.model_name, 0) + 1

    summary = (
        f"**{Path(data_path).name}** — "
        f"{n_traj} trajectories, {n_groups} groups, "
        f"{n_scorings} scoring results, "
        f"{len(policy_names)} policies"
    )

    # ---- Overview tab data ----
    policy_table = [
        [name, policy_counts[name]] for name in policy_names
    ]

    # Group table
    group_table = []
    for g_idx, group in enumerate(dataset.groups):
        trajs = [dataset.trajectories[i] for i in group]
        models = ", ".join(t.model_name for t in trajs)
        avg_reward = np.mean([
            t.cumulative_reward for t in trajs if t.cumulative_reward is not None
        ]) if any(t.cumulative_reward is not None for t in trajs) else float("nan")
        group_table.append([
            g_idx, len(group), models,
            f"{avg_reward:.4f}" if not np.isnan(avg_reward) else "N/A",
        ])

    # ---- Callbacks ----

    def filter_groups(policy_filter: str):
        if policy_filter == "All":
            return group_table
        return [r for r in group_table if policy_filter in r[2]]

    def show_group(group_idx):
        if group_idx is None:
            return "Select a group.", []
        group_idx = int(group_idx)
        if group_idx < 0 or group_idx >= n_groups:
            return "Invalid group index.", []
        group = dataset.groups[group_idx]
        trajs = [dataset.trajectories[i] for i in group]

        lines = [f"### Group #{group_idx} ({len(group)} trajectories)\n"]
        lines.append(f"**Task:** {trajs[0].language_instruction}\n")
        lines.append("| # | Model | Reward | Length | Success | Seed |")
        lines.append("|---|-------|--------|--------|---------|------|")
        for i, t in enumerate(trajs):
            lines.append(
                f"| {i + 1} | {t.model_name} | "
                f"{t.cumulative_reward if t.cumulative_reward is not None else 'N/A'} | "
                f"{t.episode_length or 'N/A'} | {t.success} | {t.env_seed} |"
            )

        # Show scores if available
        for sr in dataset.scoring_results:
            scored_in_group = {
                i: sr.scores.get(idx)
                for i, idx in enumerate(group)
                if idx in sr.scores
            }
            if scored_in_group:
                persp = f" ({sr.perspective_name})" if sr.perspective_name else ""
                lines.append(
                    f"\n**Scores from {sr.scorer_name}{persp}:** "
                    + ", ".join(
                        f"T{i + 1}={v}" for i, v in sorted(scored_in_group.items())
                    )
                )

        info_md = "\n".join(lines)

        videos = []
        for t in trajs:
            vp = _resolve_video_path(t.video_path)
            if vp:
                videos.append(vp)

        return info_md, videos

    def compute_stats():
        """Generate statistics text."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lines = ["## Statistics\n"]

        # Policy distribution
        lines.append("### Policy Distribution\n")
        for name in policy_names:
            pct = policy_counts[name] / n_traj * 100
            lines.append(f"- **{name}**: {policy_counts[name]} ({pct:.1f}%)")

        # Success rate by policy
        lines.append("\n### Success Rate by Policy\n")
        for name in policy_names:
            p_trajs = [t for t in dataset.trajectories if t.model_name == name]
            successes = sum(1 for t in p_trajs if t.success is True)
            total = len(p_trajs)
            rate = successes / total * 100 if total > 0 else 0
            lines.append(f"- **{name}**: {successes}/{total} ({rate:.1f}%)")

        # Scoring results summary
        if dataset.scoring_results:
            lines.append("\n### Scoring Results\n")
            for sr in dataset.scoring_results:
                valid_scores = [
                    v for v in sr.scores.values() if v is not None
                ]
                persp = f" [{sr.perspective_name}]" if sr.perspective_name else ""
                if valid_scores:
                    lines.append(
                        f"- **{sr.scorer_name}**{persp} ({sr.scorer_type}): "
                        f"{len(valid_scores)} scored, "
                        f"mean={np.mean(valid_scores):.3f}, "
                        f"std={np.std(valid_scores):.3f}"
                    )

        # Win rate (if 2-model groups)
        model_set = set(t.model_name for t in dataset.trajectories)
        if len(model_set) == 2 and dataset.scoring_results:
            lines.append("\n### Win Rate\n")
            for sr in dataset.scoring_results:
                if sr.perspective_name is not None:
                    continue
                wr = compute_winrate(dataset, sr.scorer_name)
                if wr["total"] > 0:
                    lines.append(f"**{sr.scorer_name}** ({wr['total']} matchups):")
                    for model, w in wr["wins"].items():
                        rate = w / wr["total"] * 100
                        lines.append(f"  - {model}: {w} wins ({rate:.1f}%)")
                    lines.append(f"  - Ties: {wr.get('ties', 0)}")

        # Score distribution plot
        fig = None
        overall_sr = dataset.get_final_scores()
        if overall_sr:
            valid = [v for v in overall_sr.scores.values() if v is not None]
            if valid:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(valid, bins=20, alpha=0.7, edgecolor="black")
                ax.set_xlabel("Score")
                ax.set_ylabel("Count")
                ax.set_title(f"Score Distribution ({overall_sr.scorer_name})")
                plt.tight_layout()

        return "\n".join(lines), fig

    def on_table_select(evt: gr.SelectData, current_filter: str):
        filtered = filter_groups(current_filter)
        row = evt.index[0]
        if row < len(filtered):
            return int(filtered[row][0])
        return 0

    # ---- Layout ----

    with gr.Blocks(title="Trajectory Data Viewer") as app:
        gr.Markdown(f"## Trajectory Data Viewer\n{summary}")

        with gr.Tabs():
            # Tab 1: Group Browser
            with gr.TabItem("Groups"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=450):
                        filter_dropdown = gr.Dropdown(
                            choices=["All"] + policy_names,
                            value="All",
                            label="Filter by policy",
                        )
                        grp_table = gr.Dataframe(
                            value=group_table,
                            headers=["#", "Size", "Models", "Avg Reward"],
                            datatype=["number", "number", "str", "str"],
                            interactive=False,
                            max_height=600,
                        )
                        group_idx_input = gr.Number(
                            value=0, label="Group index", precision=0,
                        )

                    with gr.Column(scale=2):
                        info_display = gr.Markdown("Select a group.")
                        video_gallery = gr.Gallery(
                            label="Trajectory Videos", columns=4, height=400,
                        )

                filter_dropdown.change(
                    fn=filter_groups, inputs=[filter_dropdown], outputs=[grp_table]
                )
                grp_table.select(
                    fn=on_table_select,
                    inputs=[filter_dropdown],
                    outputs=[group_idx_input],
                )
                group_idx_input.change(
                    fn=show_group,
                    inputs=[group_idx_input],
                    outputs=[info_display, video_gallery],
                )

            # Tab 2: Statistics
            with gr.TabItem("Statistics"):
                stats_btn = gr.Button("Compute Statistics")
                stats_md = gr.Markdown()
                stats_plot = gr.Plot(label="Score Distribution")
                stats_btn.click(
                    fn=compute_stats, outputs=[stats_md, stats_plot]
                )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Web-based preference / trajectory data visualizer"
    )
    parser.add_argument(
        "--data", type=str, help="Path to .pkl or .json data file"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860)",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save summary plots as PNG files and exit (no Gradio server)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preference_data.png",
        help="Output path prefix for saved plots (used with --save-only)",
    )
    args = parser.parse_args()

    data_path = args.data
    if args.config and not data_path:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_path = cfg.get("data_path", cfg.get("data", None))

    if not data_path:
        parser.error("Must provide --data or --config with data_path")

    print(f"Loading data from: {data_path}")
    dataset = load_data(data_path)
    print(
        f"Loaded {len(dataset.trajectories)} trajectories, "
        f"{len(dataset.groups)} groups."
    )

    # --- Save-only mode ---
    if args.save_only:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_base = Path(args.output)
        output_dir = output_base.parent
        stem = output_base.stem
        suffix = output_base.suffix or ".png"
        output_dir.mkdir(parents=True, exist_ok=True)

        n_traj = len(dataset.trajectories)

        # Gather policy names
        policy_names = sorted(set(t.model_name for t in dataset.trajectories))
        policy_counts = {}
        for t in dataset.trajectories:
            policy_counts[t.model_name] = policy_counts.get(t.model_name, 0) + 1

        # 1. Policy distribution bar chart
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(policy_names, [policy_counts[n] for n in policy_names], alpha=0.7)
        ax1.set_xlabel("Policy")
        ax1.set_ylabel("Count")
        ax1.set_title("Policy Distribution")
        plt.tight_layout()
        p1 = output_dir / f"{stem}_policy_dist{suffix}"
        fig1.savefig(str(p1), dpi=200, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved policy distribution to {p1}")

        # 2. Reward distribution histogram
        rewards = [
            t.cumulative_reward
            for t in dataset.trajectories
            if t.cumulative_reward is not None
        ]
        if rewards:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.hist(rewards, bins=20, alpha=0.7, edgecolor="black")
            ax2.set_xlabel("Cumulative Reward")
            ax2.set_ylabel("Count")
            ax2.set_title("Reward Distribution")
            plt.tight_layout()
            p2 = output_dir / f"{stem}_reward_dist{suffix}"
            fig2.savefig(str(p2), dpi=200, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved reward distribution to {p2}")

        # 3. Success rate by policy
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        success_rates = []
        for name in policy_names:
            p_trajs = [t for t in dataset.trajectories if t.model_name == name]
            successes = sum(1 for t in p_trajs if t.success is True)
            total = len(p_trajs)
            rate = successes / total * 100 if total > 0 else 0
            success_rates.append(rate)
        ax3.bar(policy_names, success_rates, alpha=0.7, color="green")
        ax3.set_xlabel("Policy")
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_title("Success Rate by Policy")
        ax3.set_ylim(0, 105)
        plt.tight_layout()
        p3 = output_dir / f"{stem}_success_rate{suffix}"
        fig3.savefig(str(p3), dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved success rate to {p3}")

        # 4. Score distribution (if scoring results exist)
        overall_sr = dataset.get_final_scores()
        if overall_sr:
            valid = [v for v in overall_sr.scores.values() if v is not None]
            if valid:
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                ax4.hist(valid, bins=20, alpha=0.7, edgecolor="black")
                ax4.set_xlabel("Score")
                ax4.set_ylabel("Count")
                ax4.set_title(f"Score Distribution ({overall_sr.scorer_name})")
                plt.tight_layout()
                p4 = output_dir / f"{stem}_score_dist{suffix}"
                fig4.savefig(str(p4), dpi=200, bbox_inches="tight")
                plt.close(fig4)
                print(f"Saved score distribution to {p4}")

        # Print text summary
        print(f"\n--- Summary ---")
        print(f"File: {Path(data_path).name}")
        print(f"Trajectories: {n_traj}")
        print(f"Groups: {len(dataset.groups)}")
        print(f"Scoring results: {len(dataset.scoring_results)}")
        print(f"Policies: {len(policy_names)}")
        for name in policy_names:
            p_trajs = [t for t in dataset.trajectories if t.model_name == name]
            successes = sum(1 for t in p_trajs if t.success is True)
            total = len(p_trajs)
            rate = successes / total * 100 if total > 0 else 0
            print(f"  {name}: {total} trajectories, {rate:.1f}% success")

        print("\nDone (save-only mode).")
        return

    # --- Gradio mode ---
    import gradio as gr  # noqa: F811

    app = build_app(dataset, data_path)
    print(f"\nStarting server on port {args.port}.")
    print(
        f"Access via SSH port forwarding:  "
        f"ssh -L {args.port}:localhost:{args.port} user@server"
    )
    print(f"Then open:  http://localhost:{args.port}\n")
    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
