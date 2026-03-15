#!/usr/bin/env python3
"""Rebalance policy sampling weights based on quality score distribution.

After an initial data collection round, this script analyzes the quality score
distribution of each policy and computes new weights that promote uniform
coverage across the quality spectrum.

Usage::

    python toolkits/rebalance_weights.py \\
        --dataset round1.json \\
        --config examples/embodiment/config/maniskill_collect_trajectory_data.yaml \\
        --output examples/embodiment/config/maniskill_collect_trajectory_data_round2.yaml \\
        --num_bins 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rlinf.data.quality_evaluation import compute_dataset_quality
from rlinf.data.trajectory_dataset import TrajectoryDataset


def compute_new_weights(
    dataset_path: str,
    original_config_path: str,
    output_config_path: str,
    num_bins: int = 10,
    quality_weights: dict | None = None,
) -> dict[str, float]:
    """Compute rebalanced policy weights for uniform quality coverage.

    Algorithm:
    1. Load dataset, compute quality scores.
    2. Bin the score range into ``num_bins`` equal bins.
    3. For each policy p, compute its bin distribution (fraction of its
       trajectories in each bin).
    4. Solve constrained optimization: minimize ||A @ w - target||^2
       subject to w >= 0, sum(w) = 1, where target is uniform.
    5. Write new config with updated weights.

    Returns:
        Dictionary mapping policy name to new weight.
    """
    from scipy.optimize import minimize as scipy_minimize

    dataset = TrajectoryDataset.load(dataset_path)
    if quality_weights:
        compute_dataset_quality(dataset, weights=quality_weights)
    else:
        compute_dataset_quality(dataset)

    scores = [
        t.quality_score
        for t in dataset.trajectories
        if t.quality_score is not None
    ]
    if not scores:
        raise ValueError("No quality scores computed — check data")

    policy_names = sorted(set(t.model_name for t in dataset.trajectories))
    policy_to_idx: dict[str, list[int]] = {name: [] for name in policy_names}
    for i, t in enumerate(dataset.trajectories):
        if t.quality_score is not None:
            policy_to_idx[t.model_name].append(i)

    P = len(policy_names)
    score_min = min(scores) - 1e-6
    score_max = max(scores) + 1e-6
    bin_edges = np.linspace(score_min, score_max, num_bins + 1)

    # Build count_matrix: [num_bins, P]
    A = np.zeros((num_bins, P))
    for p_idx, name in enumerate(policy_names):
        traj_indices = policy_to_idx[name]
        if not traj_indices:
            continue
        p_scores = [scores[i] for i in traj_indices]
        hist, _ = np.histogram(p_scores, bins=bin_edges)
        A[:, p_idx] = hist / len(traj_indices)

    target = np.ones(num_bins) / num_bins

    def objective(w):
        return float(np.sum((A @ w - target) ** 2))

    w0 = np.ones(P) / P
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, None)] * P
    result = scipy_minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    w = result.x

    new_weights = {name: round(float(w[i]), 6) for i, name in enumerate(policy_names)}
    print("Optimized policy weights:")
    for name, wt in sorted(new_weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {wt:.6f}")

    _update_config_weights(original_config_path, output_config_path, new_weights)
    print(f"\nWritten updated config to: {output_config_path}")
    return new_weights


def _update_config_weights(
    input_path: str, output_path: str, new_weights: dict[str, float]
) -> None:
    """Copy a YAML config and update policy weights."""
    import yaml

    with open(input_path) as f:
        cfg = yaml.safe_load(f)

    policies = cfg.get("collection", {}).get("policies", [])
    for p in policies:
        name = p.get("name", "")
        if name in new_weights:
            p["weight"] = new_weights[name]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Rebalance policy sampling weights")
    parser.add_argument("--dataset", required=True, help="TrajectoryDataset JSON path")
    parser.add_argument("--config", required=True, help="Original YAML config path")
    parser.add_argument("--output", required=True, help="Output YAML config path")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of quality bins")
    args = parser.parse_args()

    compute_new_weights(args.dataset, args.config, args.output, args.num_bins)


if __name__ == "__main__":
    main()
