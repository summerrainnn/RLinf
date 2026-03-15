#!/usr/bin/env python3
"""Web-based visualizer for preference data (.pkl files).

Launches a Gradio app that you can access from a local browser via SSH port
forwarding.

Usage::

    # On the remote server:
    python toolkits/visualize_preference_data.py --data path/to/preference.pkl

    # Then on your local machine, forward the port:
    #   ssh -L 7860:localhost:7860 user@server
    # Open http://localhost:7860 in your browser.

    # Or with a YAML config:
    python toolkits/visualize_preference_data.py --config toolkits/config/visualize_preference.yaml
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(path: str):
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from rlinf.data.preference_data import load_preference_pairs

    return load_preference_pairs(path)


def _resolve_video_path(video_path):
    """Resolve video path, handling container vs host path mapping.

    Video paths stored in preference data may use container paths
    (``/workspace/RLinf/...``) while the visualizer may run on the host
    (``~/...``).  This function tries the original path first, then
    falls back to the mapped host path.
    """
    if not video_path:
        return None
    p = Path(video_path)
    if p.exists():
        return str(p)
    # Container → host mapping: /workspace/RLinf/ → ~/
    if video_path.startswith("/workspace/RLinf/"):
        host_path = Path.home() / video_path[len("/workspace/RLinf/"):]
        if host_path.exists():
            return str(host_path)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _category(pair) -> str:
    if pair.chosen.success and not pair.rejected.success:
        return "Success vs Failure"
    if pair.chosen.success and pair.rejected.success:
        return "Both Success"
    if not pair.chosen.success and not pair.rejected.success:
        return "Both Failure"
    return "Mixed"


def _cat_short(pair) -> str:
    if pair.chosen.success and not pair.rejected.success:
        return "SF"
    if pair.chosen.success and pair.rejected.success:
        return "SS"
    if not pair.chosen.success and not pair.rejected.success:
        return "FF"
    return "MX"


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------


def build_app(pairs, data_path: str):
    n_pairs = len(pairs)

    # Pre-compute list table data
    table_rows = []
    for i, p in enumerate(pairs):
        table_rows.append([
            i,
            _cat_short(p),
            f"{p.reward_margin:+.4f}",
            f"{p.chosen.cumulative_reward:.4f}",
            f"{p.rejected.cumulative_reward:.4f}",
            f"{p.chosen.episode_length}",
            f"{p.rejected.episode_length}",
        ])

    # Summary text
    if n_pairs > 0:
        sf = sum(1 for p in pairs if p.chosen.success and not p.rejected.success)
        ss = sum(1 for p in pairs if p.chosen.success and p.rejected.success)
        ff = sum(1 for p in pairs if not p.chosen.success and not p.rejected.success)
        mx = n_pairs - sf - ss - ff
        margins = [p.reward_margin for p in pairs]
        summary = (
            f"**{Path(data_path).name}** — {n_pairs} pairs  |  "
            f"SF: {sf}  SS: {ss}  FF: {ff}  Mixed: {mx}  |  "
            f"Margin: min={min(margins):.4f}  avg={np.mean(margins):.4f}  "
            f"max={max(margins):.4f}"
        )
    else:
        summary = "No pairs loaded."

    # ---- Callbacks ----

    def filter_table(category: str):
        if category == "All":
            return table_rows
        return [r for r in table_rows if r[1] == category]

    def show_pair(pair_idx):
        if pair_idx is None:
            return "Select a pair.", None, None
        pair_idx = int(pair_idx)
        if pair_idx < 0 or pair_idx >= n_pairs:
            return "Invalid index.", None, None

        pair = pairs[pair_idx]
        c = pair.chosen
        r = pair.rejected

        # Resolve video paths (handle container vs host)
        c_video = _resolve_video_path(c.video_path)
        r_video = _resolve_video_path(r.video_path)

        c_source = "video" if c_video else ("missing" if c.video_path else "no video")
        r_source = "video" if r_video else ("missing" if r.video_path else "no video")

        info_md = (
            f"### Pair #{pair_idx}\n\n"
            f"**Task:** {c.task_description}\n\n"
            f"**Category:** {_category(pair)}  |  "
            f"**Reward margin:** {pair.reward_margin:.4f}\n\n"
            f"| | Reward | Length | Success | Source |\n"
            f"|---|---|---|---|---|\n"
            f"| **Chosen** | {c.cumulative_reward:.4f} | {c.episode_length} "
            f"| {c.success} | {c_source} |\n"
            f"| **Rejected** | {r.cumulative_reward:.4f} | {r.episode_length} "
            f"| {r.success} | {r_source} |\n"
        )

        return info_md, c_video, r_video

    def on_table_select(evt: gr.SelectData, current_filter: str):
        row = evt.index[0]
        filtered = filter_table(current_filter)
        if row < len(filtered):
            return int(filtered[row][0])
        return 0

    # ---- Layout ----

    with gr.Blocks(title="Preference Data Viewer") as app:
        gr.Markdown(f"## Preference Data Viewer\n{summary}")

        with gr.Row():
            # Left column: list
            with gr.Column(scale=1, min_width=420):
                filter_state = gr.Radio(
                    ["All", "SF", "SS", "FF", "MX"],
                    value="All",
                    label="Filter by category",
                )
                pair_table = gr.Dataframe(
                    value=table_rows,
                    headers=[
                        "#", "Cat", "Margin", "Chosen R",
                        "Rejected R", "C Len", "R Len",
                    ],
                    datatype=[
                        "number", "str", "str", "str", "str", "str", "str",
                    ],
                    interactive=False,
                    max_height=600,
                )
                pair_idx_input = gr.Number(
                    value=0,
                    label="Pair index (type or click table row)",
                    precision=0,
                )

            # Right column: detail
            with gr.Column(scale=2):
                info_display = gr.Markdown(
                    "Select a pair from the table or enter an index."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Chosen Episode")
                        chosen_video = gr.Video(
                            label="Chosen",
                            height=360,
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("#### Rejected Episode")
                        rejected_video = gr.Video(
                            label="Rejected",
                            height=360,
                            interactive=False,
                        )

        # Wire events
        all_outputs = [info_display, chosen_video, rejected_video]

        filter_state.change(
            fn=filter_table, inputs=[filter_state], outputs=[pair_table]
        )

        pair_table.select(
            fn=on_table_select,
            inputs=[filter_state],
            outputs=[pair_idx_input],
        )

        pair_idx_input.change(
            fn=show_pair,
            inputs=[pair_idx_input],
            outputs=all_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Web-based preference data visualizer"
    )
    parser.add_argument("--data", type=str, help="Path to preference .pkl file")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860)",
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

    print(f"Loading preference data from: {data_path}")
    pairs = load_pairs(data_path)
    print(f"Loaded {len(pairs)} preference pairs.")

    app = build_app(pairs, data_path)
    print(f"\nStarting server on port {args.port}.")
    print(
        f"Access via SSH port forwarding:  "
        f"ssh -L {args.port}:localhost:{args.port} user@server"
    )
    print(f"Then open:  http://localhost:{args.port}\n")
    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
