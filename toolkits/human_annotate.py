#!/usr/bin/env python3
"""Human annotation interface for TrajectoryDataset.

Launches a Gradio app where annotators can rank trajectories within each group
by selecting rankings via dropdown menus.

Usage::

    python toolkits/human_annotate.py \\
        --dataset data.json \\
        --annotator Alice \\
        --port 7862
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rlinf.data.trajectory_dataset import ScoringResult, TrajectoryDataset


def _resolve_video_path(video_path: Optional[str]) -> Optional[str]:
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


def build_annotation_app(dataset: TrajectoryDataset, annotator_name: str, save_path: str):
    import gradio as gr

    total_groups = len(dataset.groups)
    annotations: dict[int, list[int]] = {}  # group_idx -> ranking (list of traj indices, best first)
    equal_flags: dict[int, bool] = {}  # group_idx -> equally preferable flag

    def render_group(group_idx: int):
        if group_idx < 0 or group_idx >= total_groups:
            return "Invalid group index.", [], []
        group = dataset.groups[group_idx]
        trajs = [dataset.trajectories[i] for i in group]
        n = len(trajs)

        info_parts = [f"### Group {group_idx + 1}/{total_groups} ({n} trajectories)\n"]
        info_parts.append(f"**Task:** {trajs[0].language_instruction}\n")
        info_parts.append("| # | Model | Reward | Length | Success |")
        info_parts.append("|---|-------|--------|--------|---------|")
        for i, t in enumerate(trajs):
            info_parts.append(
                f"| {i + 1} | {t.model_name} | "
                f"{t.cumulative_reward if t.cumulative_reward is not None else 'N/A'} | "
                f"{t.episode_length or 'N/A'} | {t.success} |"
            )
        info_md = "\n".join(info_parts)

        videos = []
        for t in trajs:
            vp = _resolve_video_path(t.video_path)
            videos.append(vp)

        return info_md, videos, n

    def save_annotations():
        all_scores: dict[int, Optional[float]] = {}
        for g_idx, ranking in annotations.items():
            group = dataset.groups[g_idx]
            if equal_flags.get(g_idx, False) and len(group) == 2:
                for idx in group:
                    all_scores[idx] = 1.0
            else:
                n = len(ranking)
                for rank_pos, traj_idx in enumerate(ranking):
                    all_scores[traj_idx] = float(n - rank_pos)

        sr = ScoringResult(
            scorer_name=f"human_{annotator_name}",
            scorer_type="human",
            scores=all_scores,
        )
        dataset.scoring_results.append(sr)
        dataset.save(save_path)
        return f"Saved {len(annotations)}/{total_groups} annotated groups to {save_path}"

    with gr.Blocks(title="Human Trajectory Annotation") as app:
        gr.Markdown(f"## Human Annotation — {annotator_name}")
        gr.Markdown(f"Dataset: {total_groups} groups, "
                    f"{len(dataset.trajectories)} trajectories")

        with gr.Row():
            group_idx_input = gr.Number(
                value=0, label="Group index (0-based)", precision=0
            )
            progress_text = gr.Textbox(
                value=f"Annotated: 0/{total_groups}", label="Progress",
                interactive=False,
            )

        info_display = gr.Markdown("Select a group to start.")
        video_gallery = gr.Gallery(label="Trajectory Videos", columns=4, height=400)

        with gr.Row():
            ranking_inputs = []
            for i in range(5):  # support up to 5 trajectories per group
                dd = gr.Dropdown(
                    choices=[], label=f"Trajectory {i + 1} rank",
                    visible=False,
                )
                ranking_inputs.append(dd)

        equal_checkbox = gr.Checkbox(
            label="Equally preferable (only for 2-trajectory groups)",
            visible=False,
        )

        with gr.Row():
            submit_btn = gr.Button("Submit ranking for this group")
            save_btn = gr.Button("Save all annotations", variant="primary")
            next_btn = gr.Button("Next group")

        save_status = gr.Textbox(label="Status", interactive=False)

        def load_group(g_idx):
            g_idx = int(g_idx)
            info_md, videos, n = render_group(g_idx)
            updates = [info_md]

            video_items = [(v, f"Traj {i + 1}") for i, v in enumerate(videos) if v]
            updates.append(video_items)

            choices = [str(r) for r in range(1, n + 1)]
            for i in range(5):
                if i < n:
                    updates.append(gr.update(
                        choices=choices,
                        value=str(i + 1),
                        visible=True,
                        label=f"Trajectory {i + 1} rank (1=best)",
                    ))
                else:
                    updates.append(gr.update(visible=False, choices=[]))
            updates.append(gr.update(visible=(n == 2)))
            return updates

        def submit_ranking(g_idx, eq, *ranks):
            g_idx = int(g_idx)
            if g_idx < 0 or g_idx >= total_groups:
                return "Invalid group index"
            group = dataset.groups[g_idx]
            n = len(group)

            equal_flags[g_idx] = eq and n == 2
            if not equal_flags[g_idx]:
                rank_values = [int(ranks[i]) for i in range(n)]
                sorted_indices = sorted(range(n), key=lambda i: rank_values[i])
                annotations[g_idx] = [group[i] for i in sorted_indices]
            else:
                annotations[g_idx] = list(group)

            return f"Annotated: {len(annotations)}/{total_groups}"

        def go_next(g_idx):
            return min(int(g_idx) + 1, total_groups - 1)

        load_outputs = (
            [info_display, video_gallery] + ranking_inputs + [equal_checkbox]
        )
        group_idx_input.change(
            fn=load_group, inputs=[group_idx_input], outputs=load_outputs
        )
        submit_btn.click(
            fn=submit_ranking,
            inputs=[group_idx_input, equal_checkbox] + ranking_inputs,
            outputs=[progress_text],
        )
        next_btn.click(fn=go_next, inputs=[group_idx_input], outputs=[group_idx_input])
        save_btn.click(fn=save_annotations, outputs=[save_status])

    return app


def main():
    parser = argparse.ArgumentParser(description="Human trajectory annotation")
    parser.add_argument("--dataset", required=True, help="TrajectoryDataset JSON path")
    parser.add_argument("--annotator", default="anonymous", help="Annotator name")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    dataset = TrajectoryDataset.load(args.dataset)
    save_path = args.output or args.dataset
    print(f"Loaded {len(dataset.trajectories)} trajectories, "
          f"{len(dataset.groups)} groups")

    app = build_annotation_app(dataset, args.annotator, save_path)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
