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

"""Entry point for multi-policy trajectory data collection.

Usage::

    bash examples/embodiment/collect_trajectory_data.sh maniskill_collect_trajectory_data
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import get_robot_control_mode
from rlinf.envs import SupportedEnvType
from rlinf.runners.multi_policy_collection_runner import MultiPolicyCollectionRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.preference_env_worker import PreferenceEnvWorker
from rlinf.workers.rollout.hf.preference_collection_worker import (
    PreferenceCollectionWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_collect_trajectory_data",
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    if (
        SupportedEnvType(cfg.env.train.env_type) == SupportedEnvType.MANISKILL
        or SupportedEnvType(cfg.env.eval.env_type) == SupportedEnvType.MANISKILL
    ):
        control_mode = get_robot_control_mode(cfg.actor.model.policy_setup)
        cfg.env.train.init_params.control_mode = control_mode
        cfg.env.eval.init_params.control_mode = control_mode
    OmegaConf.set_struct(cfg, True)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = PreferenceCollectionWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
    )

    env_placement = component_placement.get_strategy("env")
    env_group = PreferenceEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    runner = MultiPolicyCollectionRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
