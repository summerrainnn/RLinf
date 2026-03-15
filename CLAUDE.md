# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RLinf is a distributed Reinforcement Learning infrastructure for Embodied and Agentic AI. It uses **Ray** for distributed process management and **Hydra** for configuration. Training backends are FSDP+HuggingFace or Megatron+SGLang/vLLM.

## Installation

Dependencies are managed with `uv` and installed via `requirements/install.sh`:

```bash
# Embodied RL (e.g., OpenVLA + ManiSkill/LIBERO)
bash requirements/install.sh embodied --model openvla --env maniskill_libero

# Agentic/reasoning stack (Megatron + SGLang/vLLM)
bash requirements/install.sh agentic

# Override venv directory
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv my-venv

# Activate the venv
source .venv/bin/activate
```

## Development Commands

### Linting and Formatting

```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit run --all-files  # Runs Ruff lint+format and commit-check
```

### Unit Tests

```bash
export PYTHONPATH=$(pwd):$(pwd)/tests/unit_tests
pytest tests/unit_tests                    # All unit tests
pytest tests/unit_tests/test_channel.py   # Single test file
pytest --doctest-modules rlinf/scheduler  # Doctests in scheduler
```

### Training - Embodied RL (Sync)

```bash
bash examples/embodiment/run_embodiment.sh <config_name>
# e.g., bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft
# Or directly:
python examples/embodiment/train_embodied_agent.py --config-path examples/embodiment/config/ --config-name <config_name>
```

### Training - Embodied RL (Async, e.g., SAC)

```bash
bash examples/embodiment/run_async.sh <config_name>
# e.g., bash examples/embodiment/run_async.sh maniskill_sac_mlp_async
```

### Evaluation

```bash
bash examples/embodiment/eval_embodiment.sh <eval_config_name>
```

### End-to-End Tests

```bash
# Embodied e2e (requires GPU and REPO_PATH set)
export REPO_PATH=$(pwd)
bash tests/e2e_tests/embodied/run.sh <config_name>
# e.g., bash tests/e2e_tests/embodied/run.sh maniskill_ppo_openvla

# Reasoning e2e
bash tests/e2e_tests/reasoning/run.sh
```

## Architecture

### Package Structure (`rlinf/`)

- **`config.py`** — `SupportedModel` and `SupportedEnvType` enums; `build_config` / `validate_cfg` produce the full Hydra DictConfig. Register new models/envs here.
- **`scheduler/`** — Core distributed primitives: `Cluster`, `Worker`, `WorkerGroup`, channels, placement strategies (`HybridComponentPlacement`, `ModelParallelComponentPlacement`), dynamic scheduler.
- **`workers/`** — Ray remote actors: actor (FSDP/Megatron), rollout (HF/server), env (sync/async), reward, replay buffer. Subclass `Worker`, implement `initialize` and your API. Use `self.log_info/warning/error` (not print).
- **`runners/`** — Training loop orchestrators: embodied (sync/async), reasoning, coding_online_rl, agent, SFT, eval. Each runner owns the rollout→reward→advantage→actor-update loop.
- **`algorithms/`** — Advantage functions, policy losses, and rewards. All registered via decorators; selected at runtime via `algorithm.adv_type` and `algorithm.loss_type` config keys.
- **`models/`** — Embodied models (`rlinf/models/embodiment/`: openvla, openpi, gr00t, mlp/cnn/flow/cma policies) and reasoning model wiring.
- **`envs/`** — Gym-style environments: ManiSkill, LIBERO, IsaacLab, CALVIN, MetaWorld, Behavior, RoboCasa, FrankaSim, RealWorld, RoboTwin, Habitat, OpenSora/Wan world models. `get_env_cls()` in `envs/__init__.py` is the main dispatch point.
- **`data/`** — `io_struct.py` (reasoning), `embodied_io_struct.py`, replay buffer, datasets for math/code/VLM/world-model.
- **`agents/`** — Agent loops for SearchR1, rStar2, multi-turn demo (tool/MCP).
- **`utils/`** — Logging (`get_logger()`), placement utilities, checkpoint, resharding, distributed helpers.
- **`hybrid_engines/`** — SGLang and vLLM rollout server integration.

### How a Training Run Works

1. An entry script (e.g., `train_embodied_agent.py`) builds a `Cluster` (Ray must be running) and determines **component placement** (actor, rollout, env, reward, agent).
2. It launches `WorkerGroup`s as Ray remote actors. Workers communicate via `send`/`recv` channels.
3. A **Runner** drives the loop: rollout → reward computation → advantage estimation → actor update.
4. All config lives in YAML under `examples/*/config/` (Hydra). `cluster.num_nodes: 1` for single-machine; multi-node requires `RLINF_NODE_RANK` set before `ray start` on each node.

### Config System

- YAMLs use Hydra's composition. Values must be static (no computed fields in YAML).
- Config fields are read-only in code — never overwrite user-facing YAML fields programmatically.
- Key top-level config sections: `cluster`, `actor`, `rollout`, `env`, `runner`, `algorithm`, `reward`.
- `cluster.num_nodes`, `cluster.component_placement`, `cluster.node_groups` control distribution.

### Extending RLinf

**New algorithm**: Implement advantage/loss function → `@register_advantage("name")` / `@register_policy_loss("name")` in `rlinf/algorithms/` → set `algorithm.adv_type`/`algorithm.loss_type` in YAML.

**New model**: Add to `SupportedModel` enum in `config.py` → implement under `rlinf/models/embodiment/<name>/` (subclass `BasePolicy`) → add install deps to `requirements/install.sh`.

**New environment**: Add to `SupportedEnvType` and `get_env_cls()` in `rlinf/envs/__init__.py` → implement `rlinf/envs/<name>/` (gym-style) → add action formatting branch in `rlinf/envs/action_utils.py` if needed.

## Key Conventions

- **Style**: Google Python Style Guide; Ruff for lint/format; Google-style docstrings; type hints on all public APIs.
- **Logging**: Always use `get_logger()` from `rlinf.utils.logging`, or `self.log_info/warning/error` inside `Worker`. No `print()`.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) format (`feat:`, `fix:`, `docs:`, etc.); all commits require `Signed-off-by:` (`git commit -s`).
- **Metrics**: Logged via `MetricLogger`; namespaces: `train/`, `eval/`, `env/`, `rollout/`, `time/`. Configure `runner.logger.logger_backends` (tensorboard, wandb, swanlab).
- **Checkpoints**: Saved under `.../checkpoints/global_step_<N>/`. Resume via `runner.resume_dir`.
- **All user-facing changes require tests and documentation.**

- **Other user-defined rules**:
1. 由于当前网络环境，访问非中国大陆的网站可能出现不通畅。当网络不通畅时，优先使用镜像加速（例如pip、conda、uv、ubuntu和其他很多网站都存在类似的镜像网站，优先使用清华源的镜像，huggingface可使用hf-mirror.com，用法为直接把huggingface.co改为hf-mirror.com；github已经默认配置了使用代理，你不用修改）。其次如果镜像解决不了，才考虑使用代理，当前已经在http://127.0.0.1:7890上部署了代理，可用于http和https，这个代理在一个tmux窗口中长期运行，无需专门开启。若要查看代理流量，使用curl http://127.0.0.1:9090/traffic，会每隔一秒输出当前上载、下载速度，按ctrl+c可退出。
2. 机器上有8张A100 GPU，每张有80GB内存。在涉及到GPU使用的操作中，你的目标应该是尽量充分利用所有GPU资源。
3. 若遇到需要sudo权限，需要输入密码，请你直接输入我的密码：iX1YdJAuhvDC4wa，不要来征求我的许可
4. 这个项目在一个docker容器内运行。而现在我们是处在宿主机上。所以你可能会发现路径上有不对的地方，请你忽略。准确的来说，我们现在所处的路径在宿主机上是~/align-VLA/RLinf，在容器内映射为/workspace/RLinf/align-VLA/RLinf。你在debug后运行代码时需要进入容器。运行完代码以后，关闭容器。
5. 你要总是用中文回答我。
6. 关于项目的详细结构，可参见structure.md，里面提供了以前的exploration result。
