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

from rlinf.runners.embodied_runner import EmbodiedRunner


class EmbodiedRMPPORunner(EmbodiedRunner):
    """PPO runner that injects reward model scores before advantage computation.

    Supports multi-checkpoint RM reward injection at intermediate time steps
    via the ``num_rm_checkpoints`` config key.
    """

    def __init__(self, cfg, actor, rollout, env, critic=None, reward=None, run_timer=None):
        super().__init__(cfg, actor, rollout, env, critic, reward, run_timer)
        self.rm_cfg = cfg.reward
        self._rm_metrics = {}

    def _post_rollout(self):
        """Override: apply RM reward blending before GAE."""
        with self.timer("rm_scoring"):
            rm_results = self.actor.apply_rm_rewards(
                use_dummy=self.rm_cfg.get("use_dummy", True),
                env_reward_weight=self.rm_cfg.get("env_reward_weight", 1.0),
                env_terminal_reward_weight=self.rm_cfg.get("env_terminal_reward_weight", 0.0),
                rm_reward_weight=self.rm_cfg.get("rm_reward_weight", 1.0),
                num_rm_checkpoints=int(self.rm_cfg.get("num_rm_checkpoints", 1)),
            ).wait()
        # rm_results is a list of dicts (one per actor worker), take the first
        self._rm_metrics = {f"rm/{k}": v for k, v in rm_results[0].items()}

    def _get_extra_metrics(self):
        """Return RM metrics collected in _post_rollout."""
        metrics = self._rm_metrics
        self._rm_metrics = {}
        return metrics
