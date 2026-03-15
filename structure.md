RLinf Codebase Architecture - Comprehensive Exploration Report

---
1. OVERALL DIRECTORY STRUCTURE

Top-level directories:
- /home/tianyi/align-VLA/RLinf/rlinf/ - Main package
    - config.py - Configuration system (1293 lines)
    - scheduler/ - Ray distributed primitives
    - workers/ - Ray remote actor workers
    - runners/ - Training loop orchestrators
    - models/ - Vision-language and embodied models
    - envs/ - Environment implementations
    - data/ - Data structures and I/O utilities
    - algorithms/ - Advantage functions and losses
    - utils/ - Logging, placement, checkpointing
    - agents/ - Agent loops (SearchR1, rStar2, multi-turn)
    - hybrid_engines/ - SGLang and vLLM rollout server integration
- /home/tianyi/align-VLA/RLinf/examples/embodiment/ - Training scripts and configs
- /home/tianyi/align-VLA/RLinf/tests/ - Unit and E2E tests
- /home/tianyi/align-VLA/RLinf/toolkits/ - Additional tools and evaluation scripts

---
2. CONFIGURATION SYSTEM (rlinf/config.py)

Key Components:

SupportedModel Enum (lines 44-72):
- Reasoning models: QWEN2_5, QWEN2_5_VL, QWEN3, QWEN3_MOE
- Embodied models: OPENVLA, OPENVLA_OFT, OPENPI, MLP_POLICY, GR00T, DEXBOTIC_PI, CNN_POLICY, FLOW_POLICY, CMA_POLICY
- SFT models: QWEN2_5_VL_SFT, QWEN3_VL_SFT, QWEN3_VL_MOE_SFT

Validation Functions:
- validate_embodied_cfg() - Validates embodied task configs, checks actor_critic loss requires value_head, validates environment parallelization settings
- validate_fsdp_cfg() - FSDP training backend validation
- validate_megatron_cfg() - Megatron training backend validation
- validate_reasoning_cfg() - Validates reasoning tasks with importance sampling
- validate_rollout_cfg() - Validates rollout backends (sglang, vllm)

Key Config Validation:
- Environment total_num_envs must be divisible by world size and pipeline stages
- Global batch size must align with micro batch size and world size
- Actor critic loss requires add_value_head: True
- Rollout models must specify model_path

---
3. RUNNERS DIRECTORY (rlinf/runners/)

Files:
- embodied_runner.py - Synchronous embodied PPO training
- embodied_eval_runner.py - Evaluation-only runner
- async_embodied_runner.py - Asynchronous training (SAC)
- async_ppo_embodied_runner.py - Async PPO training
- reasoning_runner.py - LLM reasoning task training
- agent_runner.py - Agent-based training (SearchR1, rStar2)
- coding_online_rl_runner.py - Online coding RL
- sft_runner.py - Supervised fine-tuning runner

EmbodiedRunner (238-292 lines):
class EmbodiedRunner:
    def __init__(cfg, actor, rollout, env, critic=None, reward=None)
    def init_workers()  # Initialize all worker groups
    def update_rollout_weights()  # Sync model from actor to rollout
    def evaluate()  # Run evaluation loop
    def run()  # Main training loop
    def _save_checkpoint()  # Save model checkpoints
    def set_max_steps()  # Calculate max training steps

Training Loop Flow (embodied_runner.py:157-263):
1. For each step:
    - Sync weights to rollout worker (every weight_sync_interval steps)
    - Generate rollouts (env → rollout → actor channels)
    - Compute advantages and returns
    - Actor training
    - Evaluation (every val_check_interval steps)
    - Save checkpoint (every save_interval steps)
    - Log metrics

EmbodiedEvalRunner (81 lines):
- Simplified runner for evaluation-only mode
- Calls rollout.evaluate() and env.evaluate() sequentially
- Returns evaluation metrics

AsyncEmbodiedRunner (async SAC training):
- Continuous async training loop
- Decouples rollout, env, and actor updates
- Uses metric channels for async metric collection
- Supports replay buffer for SAC algorithms

---
4. WORKERS DIRECTORY (rlinf/workers/)

Structure:
workers/
    actor/         # Model training workers
    env/           # Environment simulation workers
    rollout/       # Policy rollout workers (HF/SGLang/vLLM)
    reward/        # Reward computation workers
    inference/     # LLM inference workers
    agent/         # Agent execution workers
    sft/           # SFT training workers

EnvWorker (env_worker.py - 600 lines)

Key Methods:
def init_worker()  # Initialize env instances and mapping
def _setup_env_and_wrappers()  # Create environments with wrappers
def env_interact_step(chunk_actions, stage_id)  # Single environment step
def env_evaluate_step(raw_actions, stage_id)  # Evaluation step
def recv_chunk_actions(input_channel, mode)  # Receive actions from rollout
def send_env_batch(output_channel, env_batch, mode)  # Send obs/rewards
def bootstrap_step()  # Initialize episode observations
def interact()  # Main training rollout loop
def evaluate()  # Main evaluation loop

Environment Output (chunk_step):
Returns: (obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list)
- chunk_rewards: [B, num_action_chunks]
- chunk_terminations: [B, num_action_chunks]
- chunk_truncations: [B, num_action_chunks]
- Infos contain: episode (success, return, length), final_info, final_observation, intervene_action, intervene_flag

Environment Support:
- ManiSkill, LIBERO, RoboTwin, IsaacLab, MetaWorld
- BEHAVIOR (OmniGibson), CALVIN, RoboCasa
- RealWorld, FrankaSim, Habitat
- World models: OpenSora, WAN

---
5. DATA STRUCTURES (rlinf/data/embodied_io_struct.py - 661 lines)

EnvOutput (lines 30-241)

@dataclass
class EnvOutput:
    obs: dict[str, Any]  # Current observation
    final_obs: Optional[dict[str, Any]]  # Observation at episode end
    dones: torch.Tensor  # [B, num_chunks]
    terminations: torch.Tensor  # [B, num_chunks]
    truncations: torch.Tensor  # [B, num_chunks]
    rewards: torch.Tensor  # [B, num_chunks]
    intervene_actions: Optional[torch.Tensor]  # Human intervention actions
    intervene_flags: Optional[torch.Tensor]  # Human intervention flags

    def to_dict()  # Convert to standard dict format
    @staticmethod
    def merge_env_outputs()  # Merge multiple env outputs

Observation dict keys:
- main_images: [N_ENV, H, W, C]
- wrist_images: [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
- extra_view_images: [N_ENV, N_IMG, H, W, C]
- states: Proprioceptive state
- task_descriptions: Task language descriptions

ChunkStepResult (lines 243-276)

@dataclass
class ChunkStepResult:
    actions: torch.Tensor  # [B, action_dim]
    prev_logprobs: torch.Tensor  # [B, action_dim]
    prev_values: torch.Tensor  # [B, 1]
    dones/terminations/truncations: torch.Tensor
    rewards: torch.Tensor  # [B, 1]
    forward_inputs: dict[str, torch.Tensor]  # Model inputs for training
    versions: torch.Tensor  # Model version tracking

Trajectory (lines 278-394)

@dataclass
class Trajectory:
    max_episode_length: int
    model_weights_id: str  # UUID + update count
    actions: torch.Tensor  # [T, B, action_dim]
    intervene_flags: torch.Tensor  # [T, B]
    rewards: torch.Tensor  # [T, B]
    terminations/truncations/dones: torch.Tensor
    prev_logprobs/prev_values: torch.Tensor
    versions: torch.Tensor
    forward_inputs: dict[str, Any]  # [T, B, ...]
    curr_obs/next_obs: dict[str, Any]  # Observation trajectories

EmbodiedRolloutResult (lines 396-595)

- Accumulates chunk steps and transitions
- Converts to trajectory tensors
- Supports intervention action update
- Can split trajectories for parallel processing

---
6. ALGORITHMS DIRECTORY (rlinf/algorithms/)

Files:
- advantages.py - Advantage estimation (GAE, GRPO, ReinPP)
- losses.py - Policy losses (PPO actor-critic, etc.)
- registry.py - Algorithm registration system
- utils.py - Helper functions (normalization, KL penalty)
- rewards/ - Reward computation implementations

Registered Advantages (advantages.py)

@register_advantage("gae")  # Generalized Advantage Estimation
@register_advantage("grpo")  # Group Relative Policy Optimization
@register_advantage("grpo_dynamic")  # GRPO for multi-turn scenarios
@register_advantage("reinpp")  # Reinforce++ with optional baseline

GAE Implementation:
- Backward iteration through trajectory
- TD residual: delta = r + γ*V(s') - V(s)
- GAE: A = delta + γ*λ*A'
- Optional advantage normalization

Registered Policy Losses (losses.py)

@register_policy_loss("actor_critic")  # PPO actor + critic
@register_policy_loss("decoupled_actor_critic")  # PPO with proximal anchor
@register_policy_loss("actor")  # Pure GRPO actor loss

PPO Actor Loss:
- Importance sampling ratio: r = exp(log_π - log_π_old)
- Clipped ratio: r_clip = clamp(r, 1-ε, 1+ε)
- Loss: max(-A*r, -A*r_clip)
- Optional dual clip and behavioral filtering

PPO Critic Loss:
- Value prediction clipping
- Huber loss for robustness
- Explained variance computation

Registry Pattern (registry.py)

def register_advantage(name: str)  # Decorator for advantage functions
def get_adv_and_returns(name: str)  # Retrieve by name
def calculate_adv_and_returns(**kwargs)  # Unified entry point
def policy_loss(**kwargs)  # Unified loss entry point

---
7. MODELS DIRECTORY (rlinf/models/embodiment/)

Model Types:
- openpi/ - OpenPi VLA model (new)
- openvla/ - OpenVLA base model
- openvla_oft/ - OpenVLA with OFT adapters
- gr00t/ - Google robotics model
- mlp_policy/, cnn_policy/, flow_policy/ - Simple policies
- dexbotic_pi/ - Dexterous manipulation model
- modules/ - Shared model components

Base Interface (base_policy.py):
class BasePolicy:
    def forward()  # Generate actions
    def forward_value_head()  # Compute value estimates

---
8. ENVIRONMENTS DIRECTORY (rlinf/envs/)

Environment Types (from init.py):
- ManiSkill, LIBERO, RoboTwin, IsaacLab
- MetaWorld, BEHAVIOR (OmniGibson), CALVIN, RoboCasa
- RealWorld, FrankaSim, Habitat
- World models: OpenSora, WAN

Environment Interface:
def __init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
def reset() -> (obs, info)
def chunk_step(actions) -> (obs_list, rewards, terminations, truncations, infos)

Rewards & Terminations in Info:
- episode: Dict with r (return), l (length), s (success)
- final_info: Final episode metrics
- final_observation: Terminal observation
- intervene_action/intervene_flag: Human intervention data

---
9. CONFIGURATION FILES (examples/embodiment/config/)

Example: maniskill_ppo_openpi.yaml

Structure:
defaults:
    - env/maniskill_put_on_plate_in_scene_25_main@env.train
    - model/pi0@actor.model
    - training_backend/fsdp@actor.fsdp_config

cluster:
    num_nodes: 1
    component_placement: actor,env,rollout: all  # All on same node

runner:
    task_type: embodied
    max_epochs: 1000
    val_check_interval: 10  # Eval every 10 steps
    save_interval: 50       # Save every 50 steps

algorithm:
    adv_type: gae           # GAE advantages
    loss_type: actor_critic # PPO with critic
    normalize_advantages: True
    rollout_epoch: 1        # Number of rollout phases per training step
    eval_rollout_epoch: 1

    reward_type: chunk_level    # Reward at chunk level (not token)
    logprob_type: token_level   # Logprob at token level
    entropy_type: chunk_level

    gamma: 0.99             # Discount factor
    gae_lambda: 0.95        # GAE parameter
    entropy_bonus: 0.005    # Entropy regularization
    clip_ratio_high/low: 0.2
    value_clip: 0.2

env:
    train:
    total_num_envs: 320     # 320 parallel environments
    max_steps_per_rollout_epoch: 80  # Episode length
    auto_reset: True
    eval:
    total_num_envs: 320
    max_steps_per_rollout_epoch: 80
    save_video: True

actor:
    training_backend: fsdp
    micro_batch_size: 32    # Per-GPU batch
    global_batch_size: 5120 # Total batch
    model:
    model_name: openpi
    num_action_chunks: 5   # 5-action predictions
    action_dim: 7          # 7D action space
    add_value_head: True   # For PPO critic

---
10. TRAINING SCRIPT (examples/embodiment/train_embodied_agent.py)

Flow:
@hydra.main(config_path="config", config_name="maniskill_ppo_openvlaoft")
def main(cfg):
    cfg = validate_cfg(cfg)  # Validate all configs

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create worker groups (Ray actors)
    actor_group = ActorWorkerCls.create_group(cfg).launch(
        cluster, name="ActorGroup", placement_strategy=actor_placement
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name="RolloutGroup", placement_strategy=rollout_placement
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name="EnvGroup", placement_strategy=env_placement
    )

    # Create runner and run training
    runner = EmbodiedRunner(cfg, actor_group, rollout_group, env_group)
    runner.init_workers()
    runner.run()

---
11. EXECUTION SCRIPTS

run_embodiment.sh

# Sets up environment variables (MUJOCO_GL, OMNIGIBSON paths, etc.)
# Expects: CONFIG_NAME=maniskill_ppo_openpi (or custom)
# Logs to: logs/{timestamp}-{CONFIG_NAME}/
# Runs: python train_embodied_agent.py --config-path config/ --config-name {CONFIG_NAME}

eval_embodiment.sh

# Uses osmesa for headless rendering (vs egl for training)
# Runs: python eval_embodied_agent.py (inference-only)

---
12. KEY ARCHITECTURAL INSIGHTS

Data Flow Architecture

Env Worker                Rollout Worker              Actor Worker
─────────                ──────────────              ────────────
1. Reset envs      →     receive initial obs

2. Step envs       →     policy forward  →          (wait for trajectory)
    send obs_dict         send actions

3. Get rewards     →     record trajectory          compute advantages
    send rewards
                        send trajectory  →         train policy

Component Placement Strategies

- HybridComponentPlacement: Multi-node setup with flexible worker distribution
- ModelParallelComponentPlacement: For multi-node reasoning tasks

Reward Flow

1. Env returns: chunk_rewards [B, num_chunks], infos with episode metrics
2. EnvWorker extracts: episode returns/lengths/success flags
3. Rollout receives and stores rewards in trajectory
4. Actor uses for advantage computation

Action Execution

1. Model generates action chunks (e.g., 5 actions of 7D)
2. Rollout sends chunks to EnvWorker
3. EnvWorker calls prepare_actions() for environment-specific formatting
4. Env applies actions via chunk_step()

Termination Semantics

- terminations: Task completed (natural episode end)
- truncations: Time limit reached
- dones: terminations OR truncations
- Used for GAE bootstrap decisions and episode metrics

---
13. KEY FILES SUMMARY

┌──────────────────────────────────────────────────────┬───────┬─────────────────────────────┐
│                         Path                         │ Lines │           Purpose           │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ config.py                                            │ 1293  │ Config validation and enums │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ runners/embodied_runner.py                           │ 292   │ Sync PPO training loop      │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ runners/async_embodied_runner.py                     │ 203   │ Async SAC training          │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ workers/env/env_worker.py                            │ 600   │ Environment simulation      │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ data/embodied_io_struct.py                           │ 661   │ Data structures             │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ algorithms/advantages.py                             │ 286   │ GAE, GRPO, ReinPP           │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ algorithms/losses.py                                 │ 447   │ PPO losses                  │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ algorithms/registry.py                               │ 119   │ Algorithm registration      │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ examples/embodiment/train_embodied_agent.py          │ 80    │ Main training script        │
├──────────────────────────────────────────────────────┼───────┼─────────────────────────────┤
│ examples/embodiment/config/maniskill_ppo_openpi.yaml │ 174   │ Example config              │
└──────────────────────────────────────────────────────┴───────┴─────────────────────────────┘

---
14. DEPENDENCY TRACKING

- Config: Uses OmegaConf for YAML-based Hydra configuration
- Distributed: Ray for distributed actor management
- Models: HuggingFace transformers, Prismatic (VLM), Torch
- Environments: Env-specific packages (maniskill2, libero, etc.)
- Training: FSDP for distributed training, Megatron for reasoning models
- Logging: TensorBoard, Weights & Biases, SwanLab support

## Project: /home/tianyi/align-VLA/RLinf

Docker container path: /workspace/RLinf/align-VLA/RLinf (do NOT run code on host)

## Architecture Overview
- Ray distributed: Cluster → WorkerGroup (Ray remote actors) → Channel communication
- Config: Hydra YAML, validate_cfg() must pass before running
- Workers communicate env↔rollout via Channels; actor receives trajectories from rollout
- EnvWorker.evaluate() protocol: n_stages reset obs → (n_steps-1)*n_stages step obs (last step NOT sent)
- Rollout receives exactly n_eval_chunk_steps obs per epoch (first=reset, last=penultimate obs)