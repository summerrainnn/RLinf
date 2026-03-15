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

"""Entry point for reward model training.

Supports both single-GPU and multi-GPU (DDP via ``torchrun``) training.
DDP initialization is handled internally by ``RewardModelTrainer``.

Usage::

    # Single GPU
    python examples/embodiment/train_reward_model.py

    # Multi-GPU (DDP)
    bash examples/embodiment/train_reward_model.sh maniskill_train_reward_model
"""

import json
import os

import hydra
from omegaconf.omegaconf import OmegaConf

from rlinf.runners.reward_model_trainer import RewardModelTrainer


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_train_reward_model",
)
def main(cfg) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    trainer = RewardModelTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
