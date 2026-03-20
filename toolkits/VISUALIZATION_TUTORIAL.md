# 数据可视化工具使用教程

本项目 `toolkits/` 目录下提供了 4 个数据可视化脚本，覆盖偏好数据浏览、质量分布分析、特征提取和 UMAP 状态分布可视化。

---

## 目录

1. [依赖总览](#1-依赖总览)
2. [visualize_preference_data.py — 偏好/轨迹数据浏览器](#2-visualize_preference_datapy)
3. [visualize_quality.py — 质量分布可视化](#3-visualize_qualitypy)
4. [extract_features.py — CLIP 特征提取](#4-extract_featurespy)
5. [visualize_umap.py — UMAP 状态分布可视化](#5-visualize_umappy)
6. [extract_features_temporal.py — 时序结构特征提取](#6-extract_features_temporalpy)
7. [visualize_temporal_umap.py — 时序轨迹演化可视化（平行平面图）](#7-visualize_temporal_umappy)

---

## 1. 依赖总览

### 公共依赖（所有脚本都需要）

| 包 | 用途 |
|---|---|
| `numpy` | 数组计算 |
| `matplotlib` | 绑图绘图 |

### 各脚本额外依赖

| 脚本 | 额外依赖 | 用途 |
|---|---|---|
| `visualize_preference_data.py` | `gradio` | Web UI 浏览界面 |
| | `pyyaml` | 读取 YAML 配置（`--config` 模式时） |
| `visualize_quality.py` | `gradio` | Web UI 浏览界面 |
| `extract_features.py` | `torch` | CLIP 模型推理 |
| | `transformers` | 加载 CLIP 模型和 Processor |
| | `Pillow` | 图像处理 |
| | `imageio` | 视频帧读取 |
| | `tqdm` | 进度条 |
| `visualize_umap.py` | `umap-learn` | UMAP 降维 |
| | `seaborn` | KDE 等高线绘制 |
| | （`--data` 模式时）同 `extract_features.py` 的全部依赖 | 自动调用特征提取 |
| `extract_features_temporal.py` | 同 `extract_features.py` 的全部依赖 | 滑动窗口特征提取（保留时序结构） |
| `visualize_temporal_umap.py` | `umap-learn` | UMAP 降维 |
| | `scipy` | 三次样条曲线平滑 |
| | （`--data` 模式时）同 `extract_features_temporal.py` 的全部依赖 | 自动调用特征提取 |

### 一键安装

```bash
pip install numpy matplotlib gradio pyyaml torch transformers Pillow imageio tqdm umap-learn seaborn scipy
```

> **注意**：`torch` 建议根据 CUDA 版本安装对应版本，参考 https://pytorch.org/get-started/locally/。如果只用 `visualize_preference_data.py` 和 `visualize_quality.py`，不需要安装 `torch`、`transformers` 等重量级依赖。

### 项目内部依赖

所有脚本都依赖项目内部模块，需要从项目根目录运行，或者设置 `PYTHONPATH`：

```bash
cd /path/to/RLinf
export PYTHONPATH=$(pwd):$PYTHONPATH
```

内部模块依赖：
- `rlinf.data.trajectory_dataset` — `TrajectoryDataset`, `TrajectoryRecord`, `ScoringResult`
- `rlinf.data.preference_data` — `load_preference_pairs`（旧格式 pkl 兼容）
- `rlinf.data.quality_evaluation` — `compute_dataset_quality`（质量评分计算）

---

## 2. `visualize_preference_data.py`

**功能**：基于 Gradio 的 Web 交互界面，浏览偏好/轨迹数据集。支持旧格式（`.pkl`）和新格式（`.json`）。

### 功能特性

- **Groups 标签页**：按组浏览轨迹，支持按 policy 筛选，显示视频回放、reward、分数等详情
- **Statistics 标签页**：policy 分布、成功率、评分统计、win rate、分数分布直方图
- **Save-only 模式**：不启动 Web 服务，直接导出统计图表为 PNG

### 使用方法

#### Gradio 交互模式

```bash
# 新格式 JSON
python toolkits/visualize_preference_data.py --data path/to/trajectories.json

# 旧格式 pkl
python toolkits/visualize_preference_data.py --data path/to/preference.pkl

# 使用 YAML 配置
python toolkits/visualize_preference_data.py --config toolkits/config/visualize_preference.yaml

# 指定端口
python toolkits/visualize_preference_data.py --data data.json --port 8080
```

启动后通过 SSH 端口转发访问：
```bash
ssh -L 7860:localhost:7860 user@server
# 浏览器打开 http://localhost:7860
```

#### Save-only 模式（无 GUI，导出图片）

```bash
python toolkits/visualize_preference_data.py --data data.json --save-only --output output_vis/pref
```

输出文件：
- `pref_policy_dist.png` — Policy 分布柱状图
- `pref_reward_dist.png` — Reward 分布直方图
- `pref_success_rate.png` — 各 policy 成功率
- `pref_score_dist.png` — 评分分布直方图（如果存在评分数据）

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | （必选之一） | `.pkl` 或 `.json` 数据文件路径 |
| `--config` | （必选之一） | YAML 配置文件路径（内含 `data_path`） |
| `--port` | `7860` | Gradio 服务端口 |
| `--save-only` | `False` | 仅导出 PNG，不启动 Web 服务 |
| `--output` | `preference_data.png` | Save-only 模式的输出路径前缀 |

---

## 3. `visualize_quality.py`

**功能**：对一个或多个 TrajectoryDataset 文件进行质量评分（smoothness + 综合质量分），并以直方图、箱线图和统计表的形式展示分布。

### 使用方法

#### Gradio 交互模式

```bash
# 单个数据集
python toolkits/visualize_quality.py --data dataset.json

# 多数据集对比
python toolkits/visualize_quality.py \
    --data ds1.json ds2.json \
    --labels "Round 1" "Round 2"

# 指定端口和直方图 bins 数量
python toolkits/visualize_quality.py --data ds.json --port 7861 --bins 30
```

#### Save-only 模式

```bash
python toolkits/visualize_quality.py \
    --data ds1.json ds2.json \
    --labels "Round 1" "Round 2" \
    --save-only --output output_vis/quality
```

输出文件：
- `quality_histogram.png` — 质量分数直方图
- `quality_boxplot.png` — 质量分数箱线图

终端还会打印统计表（Count, Mean, Std, Median, Min, Max, Success Rate）。

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | （必选） | 一个或多个 `.json` 文件路径 |
| `--labels` | 文件名 | 各数据集的显示标签 |
| `--port` | `7861` | Gradio 服务端口 |
| `--bins` | `20` | 直方图 bin 数量 |
| `--save-only` | `False` | 仅导出 PNG |
| `--output` | `quality_distribution.png` | 输出路径前缀 |

---

## 4. `extract_features.py`

**功能**：从轨迹视频中使用 CLIP ViT-B/32 模型提取滑动窗口特征，输出 `.npz` 中间文件，供 `visualize_umap.py` 使用。

### 工作原理

1. 读取 TrajectoryDataset JSON，获取每条轨迹的视频路径
2. 用 `imageio` 加载视频的所有帧
3. 按滑动窗口切分帧序列（window_size=10, stride=5）
4. 对每个窗口，用 CLIP ViT-B/32 提取图像特征并平均池化
5. 输出 `(N, 512)` 的特征矩阵

### 使用方法

```bash
# 多文件，按文件分组
python toolkits/extract_features.py \
    --data file1.json file2.json \
    --labels "Policy A" "Policy B" \
    --output features.npz

# 单文件，按 policy 名自动分组
python toolkits/extract_features.py \
    --data file.json \
    --group-by policy \
    --output features.npz

# 使用缓存加速重复运行
python toolkits/extract_features.py \
    --data file.json \
    --cache-dir .feature_cache/ \
    --output features.npz

# 限制每组最大轨迹数（大数据集采样）
python toolkits/extract_features.py \
    --data file.json \
    --max-trajectories 50 \
    --output features.npz
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | （必选） | 一个或多个 `.json` 文件路径 |
| `--labels` | 文件名 | 各数据文件的显示标签 |
| `--group-by` | `file` | 分组策略：`file`（按文件）或 `policy`（按模型名） |
| `--output` | `features.npz` | 输出 `.npz` 文件路径 |
| `--window-size` | `10` | 滑动窗口帧数 |
| `--window-stride` | `5` | 滑动窗口步长 |
| `--device` | 自动检测 | 推理设备（`cuda` 或 `cpu`） |
| `--cache-dir` | 无 | 特征缓存目录（按视频哈希缓存） |
| `--max-trajectories` | 无 | 每组最大轨迹数 |

### 输出格式 (.npz)

```python
data = np.load("features.npz", allow_pickle=True)
data["features"]      # (N, 512) float32 — 特征向量
data["group_labels"]  # (N,) str — 每个状态的组标签
data["metadata"]      # dict — 提取参数记录
```

---

## 5. `visualize_umap.py`

**功能**：将高维状态特征通过 UMAP 降至二维，绘制 scatter + KDE 分布图，用于可视化不同 policy/数据集的状态空间分布差异。

### 两种输入模式

1. **`--features` 模式**：直接传入 `extract_features.py` 产出的 `.npz` 文件
2. **`--data` 模式**：直接传入 JSON 文件，内部自动调用特征提取（需要 GPU + CLIP 依赖）

### 使用方法

```bash
# 模式 1：从预提取的特征文件
python toolkits/visualize_umap.py \
    --features features.npz \
    --output state_distribution.png

# 模式 2：直接从 JSON（自动提取特征）
python toolkits/visualize_umap.py \
    --data file1.json file2.json \
    --labels "Policy A" "Policy B" \
    --output state_distribution.png

# 按 policy 分组（同一个 JSON 内多 policy）
python toolkits/visualize_umap.py \
    --data combined.json \
    --group-by policy \
    --output state_distribution.png

# 自定义 UMAP 参数和图片尺寸
python toolkits/visualize_umap.py \
    --features features.npz \
    --n-neighbors 30 --min-dist 0.2 \
    --figsize 12,10 --dpi 300 \
    --output high_res_plot.png
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--features` | （二选一） | 预提取的 `.npz` 特征文件 |
| `--data` | （二选一） | TrajectoryDataset JSON 文件 |
| `--labels` | 文件名 | 各文件的显示标签（`--data` 模式） |
| `--group-by` | `file` | 分组策略（`--data` 模式） |
| `--window-size` | `10` | 滑动窗口帧数（`--data` 模式） |
| `--window-stride` | `5` | 滑动窗口步长（`--data` 模式） |
| `--device` | 自动 | 推理设备（`--data` 模式） |
| `--cache-dir` | 无 | 特征缓存目录（`--data` 模式） |
| `--max-trajectories` | 无 | 每组最大轨迹数（`--data` 模式） |
| `--n-neighbors` | `15` | UMAP n_neighbors 参数 |
| `--min-dist` | `0.1` | UMAP min_dist 参数 |
| `--output` | `state_distribution.png` | 输出图片路径 |
| `--figsize` | `10,8` | 图片尺寸（宽,高，英寸） |
| `--dpi` | `200` | 输出分辨率 |

---

## 6. `extract_features_temporal.py`

**功能**：与 `extract_features.py` 使用相同的 CLIP 滑动窗口提取策略，但**保留每条轨迹的时序结构**——输出三维张量 `(N, T, 512)` 而非扁平的 `(N_total, 512)`。

### 与 `extract_features.py` 的区别

| | `extract_features.py` | `extract_features_temporal.py` |
|---|---|---|
| **输出形状** | `(所有clip总数, 512)` | `(轨迹数, max_clips, 512)` |
| **时序信息** | 丢弃（所有 clip 平等混合） | 保留（第 i 个 clip ↔ 第 i 个时间步） |
| **用途** | 2D 散点 / KDE 状态分布 | 3D 平行平面时序演化图 |
| **缓存** | 支持，key = `(video_hash, window_size, stride)` | 支持，**与前者缓存完全兼容、可互相复用** |

### 工作原理

1. 读取 TrajectoryDataset JSON，获取每条轨迹的视频路径
2. 用 `imageio` 加载全部帧
3. 按滑动窗口切分：frames 0\~9, 5\~14, 10\~19, … (window_size=10, stride=5)
4. 对每个窗口，CLIP ViT-B/32 提取图像特征并平均池化 → 1 个 512 维向量
5. 不同轨迹的 clip 数可能不同（视频长度不一），短的用 NaN 填充，`valid_mask` 标记有效位置
6. 输出 `(N, max_clips, 512)` + `valid_mask (N, max_clips)`

### 使用方法

```bash
# 基本用法：两个数据集对比
python toolkits/extract_features_temporal.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --output features_temporal.npz

# 使用缓存 + 自定义窗口参数
python toolkits/extract_features_temporal.py \
    --data data.json \
    --group-by policy \
    --window-size 10 --window-stride 5 \
    --cache-dir .feature_cache/ \
    --output features_temporal.npz

# 限制轨迹数量（大数据集采样）+ 指定 batch size
python toolkits/extract_features_temporal.py \
    --data data.json \
    --max-trajectories 30 \
    --batch-size 256 \
    --output features_temporal.npz

# 单 GPU 运行
python toolkits/extract_features_temporal.py \
    --data data.json \
    --num-gpus 1 --device cuda:0 \
    --output features_temporal.npz
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | （必选） | 一个或多个 `.json` 文件路径 |
| `--labels` | 文件名 | 各数据文件的显示标签 |
| `--group-by` | `file` | 分组策略：`file`（按文件）或 `policy`（按模型名） |
| `--output` | `features_temporal.npz` | 输出 `.npz` 文件路径 |
| `--window-size` | `10` | 滑动窗口帧数 |
| `--window-stride` | `5` | 滑动窗口步长 |
| `--device` | 自动检测 | 推理设备（`cuda` 或 `cpu`） |
| `--num-gpus` | `8` | 多 GPU 并行提取的 GPU 数量 |
| `--batch-size` | `512` | 每次 CLIP 前向传播的最大图片数 |
| `--cache-dir` | 无 | 特征缓存目录（与 `extract_features.py` 共享） |
| `--max-trajectories` | 无 | 每组最大轨迹数 |

### 输出格式 (.npz)

```python
data = np.load("features_temporal.npz", allow_pickle=True)
data["features"]      # (N, max_clips, 512) float32 — NaN 填充无效位置
data["group_labels"]  # (N,) str — 每条轨迹的组标签
data["valid_mask"]    # (N, max_clips) bool — True 表示有效特征
data["metadata"]      # dict — 提取参数记录
```

---

## 7. `visualize_temporal_umap.py`

**功能**：将 `extract_features_temporal.py` 产出的时序结构特征通过 **全局 UMAP** 降至二维，然后在 3D 空间中绘制平行竖直平面图——每个平面对应一个滑动窗口时间步，每条轨迹是一条穿越各平面的曲线。

### 可视化效果

```
        t₀       t₁       t₂       t₃       t₄
        │        │        │        │        │
   ┌────┤   ┌────┤   ┌────┤   ┌────┤   ┌────┤
   │ ●──┼───┼─●──┼───┼─●──┼───┼─●──┼───┼─●  │  ← 一条红色轨迹
   │    │   │    │   │    │   │    │   │    │
   │ ●──┼───┼─●──┼───┼─●──┼───┼─●──┼───┼─●  │  ← 一条蓝色轨迹
   └────┤   └────┤   └────┤   └────┤   └────┤
        │        │        │        │        │
     平面0    平面1    平面2    平面3    平面4
```

- **X 轴**：时间步（滑动窗口位置，对应平面深度）
- **Y / Z 轴**：UMAP 二维投影（平面内的坐标）
- 同组轨迹用相同颜色（红、蓝、绿等）
- 有 4+ 个有效时间点的轨迹使用三次样条平滑插值

### 两种输入模式

1. **`--features` 模式**：传入 `extract_features_temporal.py` 产出的 `.npz` 文件（推荐：特征只提取一次，可反复调 UMAP 参数出图）
2. **`--data` 模式**：直接传入 JSON 文件，内部自动调用时序特征提取（一步完成，支持缓存和 batch）

### 使用方法

#### 两步式（推荐，提取与可视化分离）

```bash
# 第一步：提取时序特征（耗时，可缓存）
python toolkits/extract_features_temporal.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --window-size 10 --window-stride 5 \
    --cache-dir .feature_cache/ \
    --output features_temporal.npz

# 第二步：可视化（快速，可反复调参）
python toolkits/visualize_temporal_umap.py \
    --features features_temporal.npz \
    --n-neighbors 20 --min-dist 0.15 \
    --output trajectory_evolution.png
```

#### 一步式（自动提取 + 可视化）

```bash
python toolkits/visualize_temporal_umap.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --window-size 10 --window-stride 5 \
    --cache-dir .feature_cache/ \
    --batch-size 512 \
    --output trajectory_evolution.png
```

#### 调整视角和样式

```bash
# 改变 3D 视角
python toolkits/visualize_temporal_umap.py \
    --features features_temporal.npz \
    --elev 30 --azim -45 \
    --output trajectory_evolution_angle2.png

# 调整点大小和线条透明度
python toolkits/visualize_temporal_umap.py \
    --features features_temporal.npz \
    --point-size 30 --line-alpha 0.4 \
    --figsize 16,12 --dpi 300 \
    --output high_res_plot.png
```

#### 按 policy 自动分组

```bash
# 单个 JSON 内包含多个 policy 的轨迹
python toolkits/visualize_temporal_umap.py \
    --data combined.json \
    --group-by policy \
    --cache-dir .feature_cache/ \
    --output trajectory_evolution.png
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--features` | （二选一） | 预提取的时序 `.npz` 特征文件 |
| `--data` | （二选一） | TrajectoryDataset JSON 文件 |
| `--labels` | 文件名 | 各文件的显示标签（`--data` 模式） |
| `--group-by` | `file` | 分组策略（`--data` 模式） |
| `--window-size` | `10` | 滑动窗口帧数（`--data` 模式） |
| `--window-stride` | `5` | 滑动窗口步长（`--data` 模式） |
| `--device` | 自动 | 推理设备（`--data` 模式） |
| `--num-gpus` | `8` | 多 GPU 并行数（`--data` 模式） |
| `--batch-size` | `512` | CLIP 前向传播 batch 大小（`--data` 模式） |
| `--cache-dir` | 无 | 特征缓存目录（`--data` 模式，与其他脚本共享） |
| `--max-trajectories` | 无 | 每组最大轨迹数（`--data` 模式） |
| `--n-neighbors` | `15` | UMAP n_neighbors 参数 |
| `--min-dist` | `0.1` | UMAP min_dist 参数 |
| `--output` | `trajectory_evolution.png` | 输出图片路径 |
| `--figsize` | `14,10` | 图片尺寸（宽,高，英寸） |
| `--dpi` | `200` | 输出分辨率 |
| `--elev` | `20` | 3D 视角仰角（度） |
| `--azim` | `-60` | 3D 视角方位角（度） |
| `--point-size` | `20` | 散点大小 |
| `--line-alpha` | `0.25` | 连接曲线透明度 |

---

## 典型工作流

### 流程 1：快速浏览数据集

```bash
python toolkits/visualize_preference_data.py --data logs/trajectories.json
```

### 流程 2：导出统计图表（无 GUI）

```bash
python toolkits/visualize_preference_data.py --data data.json --save-only --output output_vis/pref
python toolkits/visualize_quality.py --data data.json --save-only --output output_vis/quality
```

### 流程 3：2D 状态空间分析

```bash
# 两步式（推荐）
python toolkits/extract_features.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --cache-dir .feature_cache/ \
    --output features.npz

python toolkits/visualize_umap.py \
    --features features.npz \
    --n-neighbors 20 --min-dist 0.15 \
    --output state_distribution.png

# 一步式
python toolkits/visualize_umap.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --output state_distribution.png
```

### 流程 4：3D 时序轨迹演化分析（平行平面图）

```bash
# 两步式（推荐）
python toolkits/extract_features_temporal.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --cache-dir .feature_cache/ \
    --output features_temporal.npz

python toolkits/visualize_temporal_umap.py \
    --features features_temporal.npz \
    --output trajectory_evolution.png

# 一步式
python toolkits/visualize_temporal_umap.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --cache-dir .feature_cache/ \
    --output trajectory_evolution.png
```

### 流程 5：同时生成 2D 分布图 + 3D 时序图

```bash
# 提取特征（两种脚本共享缓存！）
python toolkits/extract_features.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --cache-dir .feature_cache/ \
    --output features.npz

python toolkits/extract_features_temporal.py \
    --data baseline.json ours.json \
    --labels "Baseline" "Ours" \
    --cache-dir .feature_cache/ \
    --output features_temporal.npz

# 可视化
python toolkits/visualize_umap.py --features features.npz --output state_distribution.png
python toolkits/visualize_temporal_umap.py --features features_temporal.npz --output trajectory_evolution.png
```

> **提示**：`extract_features.py` 和 `extract_features_temporal.py` 使用相同的缓存 key（基于视频路径哈希 + window_size + stride），因此**同参数下缓存完全互通**——先用任一脚本提取过的视频特征，另一脚本可以直接复用。

---

## 常见问题

### Q: 视频路径找不到？
脚本会自动尝试容器路径到宿主机路径的映射（`/workspace/RLinf/` → `~/`）。如果仍然找不到，检查 JSON 中 `video_path` 字段是否正确。

### Q: CLIP 模型下载慢？
默认使用 `hf-mirror.com` 镜像。如果仍有问题：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: Gradio 无法从远程访问？
```bash
ssh -L 7860:localhost:7860 user@server
# 浏览器打开 http://localhost:7860
```

### Q: UMAP 太慢？
- 使用 `--max-trajectories` 限制每组轨迹数
- 先提取并缓存特征（`--cache-dir`），再单独运行可视化脚本调参

### Q: 3D 时序图平面太多 / 太密？
平面数量 = `(视频帧数 - window_size) / stride + 1`。减少平面：
- 增大 `--window-stride`（如 5 → 10）
- 用 `--max-trajectories` 减少轨迹数，避免曲线过密

### Q: 3D 时序图视角不理想？
用 `--elev`（仰角）和 `--azim`（方位角）调整：
- 默认：`--elev 20 --azim -60`
- 正侧面：`--elev 15 --azim -45`
- 俯视：`--elev 60 --azim -60`

### Q: 不同轨迹视频长度不一致怎么办？
`extract_features_temporal.py` 自动处理：较短轨迹用 NaN 填充到 max_clips 长度，`valid_mask` 标记有效时间步。可视化时无效点自动跳过，不影响曲线绘制。
