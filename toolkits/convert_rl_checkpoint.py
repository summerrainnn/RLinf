#!/usr/bin/env python3
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
"""Convert an RLinf RL checkpoint to OpenPi HuggingFace-compatible format.

An RL checkpoint (e.g. global_step_150/) contains:
    actor/model_state_dict/full_weights.pt   — trained weights
    actor/dcp_checkpoint/                    — FSDP distributed checkpoint (not needed)

An HF-compatible checkpoint (e.g. RLinf-Pi0-ManiSkill-25Main-SFT/) contains:
    model.safetensors                        — weights in safetensors format
    model_state_dict/full_weights.pt         — OR weights in PyTorch format
    physical-intelligence/maniskill/norm_stats.json  — normalization stats
    assets/                                  — additional assets
    metadata.pt, metadata.txt               — training metadata

This script bridges the gap by creating a new directory that:
1. Links/copies full_weights.pt from the RL checkpoint
2. Links/copies norm_stats, assets, and metadata from the reference SFT model

The output directory can be used directly as ``model_path`` in OpenPi configs.

Usage:
    python toolkits/convert_rl_checkpoint.py \\
        --rl-checkpoint logs/.../checkpoints/global_step_150 \\
        --sft-model /path/to/RLinf-Pi0-ManiSkill-25Main-SFT \\
        --output /path/to/output_model

    # Use --copy to hard-copy files instead of symlinks
    python toolkits/convert_rl_checkpoint.py \\
        --rl-checkpoint logs/.../checkpoints/global_step_150 \\
        --sft-model /path/to/RLinf-Pi0-ManiSkill-25Main-SFT \\
        --output /path/to/output_model \\
        --copy
"""

import argparse
import os
import shutil
import sys


def find_full_weights(rl_checkpoint_dir: str) -> str:
    """Find full_weights.pt in the RL checkpoint directory."""
    candidates = [
        os.path.join(rl_checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"),
        os.path.join(rl_checkpoint_dir, "model_state_dict", "full_weights.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find full_weights.pt in {rl_checkpoint_dir}. "
        f"Searched: {candidates}"
    )


def link_or_copy(src: str, dst: str, use_copy: bool) -> None:
    """Create a symlink or copy a file/directory."""
    if use_copy:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        src_abs = os.path.abspath(src)
        os.symlink(src_abs, dst)


def convert(
    rl_checkpoint_dir: str,
    sft_model_dir: str,
    output_dir: str,
    use_copy: bool = False,
) -> None:
    if os.path.exists(output_dir):
        print(f"Error: output directory already exists: {output_dir}")
        sys.exit(1)

    # Validate inputs
    weights_path = find_full_weights(rl_checkpoint_dir)
    if not os.path.isdir(sft_model_dir):
        print(f"Error: SFT model directory not found: {sft_model_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating HF-compatible checkpoint at: {output_dir}")

    # 1. Link/copy full_weights.pt
    dst_weights_dir = os.path.join(output_dir, "model_state_dict")
    os.makedirs(dst_weights_dir, exist_ok=True)
    link_or_copy(weights_path, os.path.join(dst_weights_dir, "full_weights.pt"), use_copy)
    print(f"  weights: {weights_path}")

    # 2. Link/copy metadata and norm_stats from SFT model
    items_to_copy = [
        "metadata.pt",
        "metadata.txt",
        "physical-intelligence",
        "assets",
    ]
    for item in items_to_copy:
        src = os.path.join(sft_model_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            link_or_copy(src, dst, use_copy)
            print(f"  {item}: linked from SFT model")
        else:
            print(f"  {item}: not found in SFT model (skipped)")

    mode = "copied" if use_copy else "symlinked"
    print(f"\nDone! Files {mode}. Output directory: {output_dir}")
    print(f"Use as model_path in config: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLinf RL checkpoint to HF-compatible format"
    )
    parser.add_argument(
        "--rl-checkpoint",
        required=True,
        help="Path to RL checkpoint directory (e.g. .../global_step_150)",
    )
    parser.add_argument(
        "--sft-model",
        required=True,
        help="Path to reference SFT model directory (e.g. RLinf-Pi0-ManiSkill-25Main-SFT)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output HF-compatible directory",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Hard-copy files instead of creating symlinks (uses more disk space)",
    )
    args = parser.parse_args()

    convert(
        rl_checkpoint_dir=args.rl_checkpoint,
        sft_model_dir=args.sft_model,
        output_dir=args.output,
        use_copy=args.copy,
    )


if __name__ == "__main__":
    main()
