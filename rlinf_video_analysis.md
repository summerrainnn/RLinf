# RLinf 项目视频导出功能调查报告

## 执行总结

RLinf 项目在 eval 代码中的视频导出功能由 `RecordVideo` wrapper 实现，它：
1. **从 observation 中提取图像**（而不是调用 render 函数）
2. **支持多环境batch并通过 tile_images 组织成6x6网格**（当有36个环境时）
3. **使用 imageio + 后台线程异步保存 MP4**
4. **通过 chunk_step 可以处理超过原始step数的多个observation帧**

当前**偏好数据收集**仅保存均匀采样的 **N 个关键帧**（默认8个），不生成视频文件。

---

## 1. 视频导出核心实现

### 1.1 RecordVideo Wrapper
**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/envs/wrappers/record_video.py`

这是一个 Gym wrapper，管理视频录制的生命周期：

**初始化**（第43-76行）：
```python
def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
    self.render_images: list[np.ndarray] = []  # 缓冲区存储所有帧
    self.video_cnt = 0
    self._num_envs = getattr(env, "num_envs", 1)
    self._executor = ThreadPoolExecutor(max_workers=1)  # 后台线程
    self._save_futures: list[Future] = []  # 异步保存任务队列
```

**核心方法**：

| 方法 | 功能 |
|-----|------|
| `_extract_frame_batches()` | 从 observation dict 提取图像，支持多种格式（3D/4D/5D张量） |
| `_append_frame()` | 将多个环境的图像拼接成一个tiled frame，覆盖info |
| `add_new_frames()` | 提取 obs 中的所有帧并加入缓冲区 |
| `reset()` / `step()` / `chunk_step()` | 环境交互时自动录制 |
| `flush_video()` | 将缓冲区中的所有帧保存为 MP4（异步） |
| `wait_for_saves()` | 等待所有后台保存完成 |

**关键数据流**：
```
环境交互（reset/step/chunk_step）
    ↓
add_new_frames(obs, rewards, terminations)
    ↓
_extract_frame_batches(obs)  ← 从 obs["main_images"] 等提取
    ↓
_append_frame(images, ...)  ← tile_images() 拼接多环境 + 叠加info
    ↓
self.render_images.append(frame)  ← 缓冲
    ↓
flush_video()  ← imageio.get_writer() 后台保存为 MP4
```

### 1.2 帧提取逻辑

**支持的 observation 格式**（第86-130行）：
```python
def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
    # 处理 dict: obs["main_images"], obs["images"], obs["rgb"], ...
    # 处理 list/tuple: [obs1_dict, obs2_dict, ...]
    # 处理 Tensor: 3D (H,W,C) / 4D (B,H,W,C) / 5D (B,T,H,W,C)
```

**关键字段优先级**（第88行）：
```python
for key in ("main_images", "images", "rgb", "full_image", "main_image"):
    if key in obs and obs[key] is not None:
        return obs[key]
```

**5D 张量处理**（chunk_step 来自的多步观测，第153-165行）：
```python
if img.ndim == 5:  # Shape: (B, T, C, H, W) or (B, T, H, W, C)
    # 转换到 (B, T, H, W, C)
    frames = []
    for t in range(img.shape[1]):  # ← T 时间步
        images = []
        for i in range(img.shape[0]):  # ← B 环境
            images.append(img[i, t])  # 单个环境的单个时间步
        frames.append(images)  # 返回 T 个列表，每个列表有 B 个图像
    return frames
```

这就是**帧数超过 env.step 数量的秘密**：chunk_step 中一次 step 调用返回的观测中包含了多个时间步的观测！

### 1.3 多环境网格布局

**Tiling 逻辑**（第246-251行）：
```python
if len(images) > 1:
    nrows = int(np.sqrt(len(images)))  # 计算网格行数
    full_image = tile_images(images, nrows=nrows)
    self.render_images.append(full_image)
else:
    self.render_images.append(images[0])  # 单环境直接保存
```

**6x6 网格示例**：
- 36 个环境 → `nrows = int(sqrt(36)) = 6` → 6×6 网格
- 25 个环境 → `nrows = int(sqrt(25)) = 5` → 5×5 网格
- 9 个环境 → `nrows = int(sqrt(9)) = 3` → 3×3 网格

### 1.4 Tile Images 实现

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/envs/utils.py` 第121-186行

核心算法：
1. **按高度排序**：最高的图像放在左边（节省空间）
2. **贪心布局**：按从左到右的顺序，尽量填满每一列
3. **填充**：零填充到最大高度和总宽度

```python
def tile_images(images, nrows: int = 1) -> Union[np.ndarray, torch.Tensor]:
    # 排序后按列堆砌，返回统一大小的网格图像
    # 支持非均匀大小的输入（只有当 nrows==1 时）
```

### 1.5 视频保存（异步）

**保存方法**（第352-363行）：
```python
def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
    video_writer = imageio.get_writer(mp4_path, fps=self._fps)
    for img in frames:
        video_writer.append_data(img)  # ffmpeg 编码
    video_writer.close()
```

**关键特性**：
- **异步保存**：后台线程不阻塞主训练循环
- **FPS 分辨率**：从 `video_cfg.fps` 或环境元数据推导（默认30）
- **压缩**：imageio 使用 ffmpeg 自动压缩，文件很小（几MB）
- **流式写入**：逐帧 append，不在内存中缓冲整个视频

---

## 2. 当前偏好数据收集实现

### 2.1 环境端：PreferenceEnvWorker

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/workers/env/preference_env_worker.py`

**方法**: `collect_preference_epoch()` 第48-158行

**功能**：运行一个完整的评估 epoch，跟踪每个环境的累积奖励和成功标志。

```python
def collect_preference_epoch(
    self, input_channel: Channel, output_channel: Channel
) -> dict[str, Any]:
    """返回: {"rewards": [N_envs], "successes": [N_envs], "lengths": [N_envs]}"""
    
    # 初始化每个环境的统计
    cum_rewards = np.zeros(n_envs_total, dtype=np.float32)
    successes = np.zeros(n_envs_total, dtype=bool)
    lengths = np.zeros(n_envs_total, dtype=np.int32)
    
    # 运行完整的 n_eval_chunk_steps
    for eval_step in range(self.n_eval_chunk_steps):
        for stage_id in range(self.stage_num):
            # chunk_step() 返回观测列表
            obs_list, chunk_rewards, ..., infos_list = (
                self.eval_env_list[stage_id].chunk_step(chunk_actions)
            )
            # 累积奖励和长度
            cum_rewards[global_i] += float(ep_rewards[local_i])
            lengths[global_i] += self.cfg.actor.model.num_action_chunks
            
            # 捕获成功标志（从 infos 或 terminations）
            success_tensor = self._extract_success(infos, done_flags)
```

**关键点**：
- **不生成视频**：只积累数值指标
- **完整轨迹**：总是运行满 `n_eval_chunk_steps` 来收集完整的 episode
- **obs_list**：chunk_step 返回的观测列表（每步一个 obs），传给 rollout worker

### 2.2 Rollout 端：PreferenceCollectionWorker

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/workers/rollout/hf/preference_collection_worker.py`

**方法**: `collect_episodes_epoch()` 第120-196行

**功能**：运行推理 rollout，并为每个环境采样 N 个均匀分布的关键帧。

```python
async def collect_episodes_epoch(
    self, input_channel: Channel, output_channel: Channel
) -> list[dict[str, Any]]:
    """返回: list[{"keyframes": [N], "wrist_keyframes": [N], "task_description": str}]"""
    
    # 预计算关键帧步数索引 (0 到 n_steps-1，共 n_keyframes 个)
    keyframe_step_indices: set[int] = set(
        np.linspace(0, n_steps - 1, self._n_keyframes).round().astype(int).tolist()
    )
    # 例：n_steps=100, n_keyframes=8 → {0, 14, 28, 43, 57, 71, 86, 99}
    
    # 运行完整的 n_eval_chunk_steps
    for step in range(n_steps):
        for _stage_id in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel, mode="eval")
            obs = env_output["obs"]  # ← 从环境接收观测
            actions, _ = self.predict(obs, mode="eval")  # ← 推理
            
            n = self.eval_batch_size
            for local_idx in range(n):
                global_idx = env_offset + local_idx
                
                # 第 0 步时捕获任务描述
                if step == 0:
                    task_desc_arr[global_idx] = _extract_task_description(obs, local_idx)
                
                # 在关键帧步数时捕获图像
                if step in keyframe_step_indices:
                    keyframes_arr[global_idx].append(
                        _extract_main_image(obs, local_idx)  # ← main_images
                    )
                    wrist_keyframes_arr[global_idx].append(
                        _extract_wrist_image(obs, local_idx)  # ← wrist_images
                    )
```

**关键点**：
- **仅采样关键帧**：不保存所有帧，只保存 ~8 个均匀分布的帧
- **从 main_images 提取**：`obs.get("main_images")[env_idx]`
- **支持 wrist 相机**：可选的`obs["wrist_images"]`
- **uint8 转换**：`_to_uint8_numpy()` 将张量转为 uint8 numpy（紧凑存储）

### 2.3 数据组装：EpisodeRecord

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/data/preference_data.py`

```python
@dataclass
class EpisodeRecord:
    task_description: str          # 任务指令
    keyframes: list                # N个 [H,W,C] uint8 numpy 数组
    cumulative_reward: float       # 累积奖励（来自env）
    success: bool                  # 成功标志（来自env）
    episode_length: int            # episode 步数（来自env）
    wrist_keyframes: Optional[list] = None  # 可选的腕部相机关键帧
```

**组装函数**（`build_episode_records()` 第199-239行）：
```python
def build_episode_records(
    raw_episodes: list[dict],  # 来自 PreferenceCollectionWorker
    rewards: np.ndarray,       # 来自 PreferenceEnvWorker
    successes: np.ndarray,     # 来自 PreferenceEnvWorker
    lengths: np.ndarray,       # 来自 PreferenceEnvWorker
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for ep, rew, suc, ln in zip(raw_episodes, rewards, successes, lengths):
        records.append(EpisodeRecord(
            task_description=ep["task_description"],
            keyframes=ep["keyframes"],              # 8 个主相机帧
            cumulative_reward=float(rew),
            success=bool(suc),
            episode_length=int(ln),
            wrist_keyframes=ep["wrist_keyframes"],  # 8 个腕部相机帧（可能为 None）
        ))
    return records
```

### 2.4 配对逻辑

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/data/preference_data.py` 第95-138行

```python
def create_preference_pairs(
    episodes: list[EpisodeRecord],
    min_reward_diff: float = 0.0,
) -> list[PreferencePair]:
    """
    组内配对策略：
    1. 按 success 分成两组（成功/失败）
    2. 每组内随机打乱并两两配对 (0,1), (2,3), ...
    3. 较高奖励 → chosen，较低奖励 → rejected
    4. 过滤：奖励差小于 min_reward_diff 的对被丢弃
    """
    successes = [ep for ep in episodes if ep.success]
    failures = [ep for ep in episodes if not ep.success]
    
    def _pair_within_group(group: list[EpisodeRecord]) -> list[PreferencePair]:
        rng.shuffle(group)
        pairs: list[PreferencePair] = []
        for i in range(0, len(group) - 1, 2):  # ← 两两配对，跳过最后一个奇数
            a, b = group[i], group[i + 1]
            if a.cumulative_reward >= b.cumulative_reward:
                chosen, rejected = a, b
            else:
                chosen, rejected = b, a
            diff = chosen.cumulative_reward - rejected.cumulative_reward
            if diff >= min_reward_diff:
                pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
        return pairs
    
    return _pair_within_group(successes) + _pair_within_group(failures)
```

---

## 3. 观测中的图像来源

### 3.1 ManiSkill 环境

**文件**: `/home/tianyi/align-VLA/RLinf/rlinf/envs/maniskill/maniskill_env.py`

**观测结构** （`_wrap_obs()` 第142-193行）：

```python
def _wrap_obs(self, raw_obs, infos=None):
    # wrap_obs_mode = "default" (常用)
    
    sensor_data = raw_obs["sensor_data"]
    # 优先使用 3rd_view_camera，否则用第一个可用相机
    camera_key = "3rd_view_camera" if "3rd_view_camera" in sensor_data else next(iter(sensor_data))
    
    obs_image = sensor_data[camera_key]["rgb"].to(torch.uint8)  # [B, H, W, C]
    proprioception = self.env.unwrapped.agent.robot.get_qpos()   # [B, qpos_dim]
    
    return {
        "main_images": obs_image,        # ← uint8 [B, H, W, C]
        "states": proprioception,        # ← float32 [B, qpos_dim]
        "task_descriptions": self.instruction,  # ← [B] strings
    }
```

**多视图模式**（wrap_obs_mode="simple"，第148-173行）：
```python
main_images = sensor_data["base_camera"]["rgb"]  # [B, H, W, C]
sorted_images = OrderedDict(sorted(sensor_data.items()))
sorted_images.pop("base_camera")
extra_view_images = torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
# extra_view_images: [B, num_extra_views, H, W, C]

return {
    "main_images": main_images,
    "extra_view_images": extra_view_images,  # 4D tensor，可被 RecordVideo 拆分
    "states": state,
}
```

**chunk_step 中的多步观测**（第323-371行）：
```python
def chunk_step(self, chunk_actions):
    # chunk_actions: [num_envs, chunk_steps, action_dim]
    chunk_size = chunk_actions.shape[1]
    obs_list = []
    
    for i in range(chunk_size):  # ← 每个 action 对应一个观测
        actions = chunk_actions[:, i]
        extracted_obs, step_reward, terminations, truncations, infos = self.step(actions)
        obs_list.append(extracted_obs)
        # ...
    
    return (
        obs_list,  # ← 列表长度 = chunk_steps，可能 >> 1
        chunk_rewards,
        chunk_terminations,
        chunk_truncations,
        infos_list,
    )
```

**例子**：
- chunk_size = 4（一次chunk有4个action）
- obs_list = [obs_t, obs_t+1, obs_t+2, obs_t+3]
- 每个obs["main_images"] 形状为 [B, H, W, C]
- RecordVideo 看到 5D 张量（B, T=4, H, W, C）并逐帧保存
- 结果：4 个帧（因为 chunk_size=4），而不是 1 个

---

## 4. 为什么帧数能超过 env.step 数

### 示例场景

**配置**：
- `max_steps_per_rollout_epoch: 80`  （80个原始 step）
- `num_action_chunks: 4`              （chunk_step 中的子 step 数）
- `n_eval_chunk_steps = 80 / 4 = 20` （调用 chunk_step 的次数）
- 每个 chunk_step 返回 obs_list，长度 = 4

**RecordVideo 视角**：
- 调用 1：reset() → add_new_frames(obs) → 1 帧
- 调用 2-21：chunk_step() 返回 obs_list[4] → add_new_frames(obs_list) → 4×20 = 80 帧
- **总计**：1 + 80 = **81 帧**，对应 80 个原始 step

**为什么超过 80**：
RecordVideo 在 reset 时也记录了初始观测，所以多了 1 帧。

### 多视图情况

如果环境返回 5D 张量 `extra_view_images: [B, num_views, H, W, C]`：
- 一个 chunk_step 返回 obs_list[4]
- 每个 obs 中 extra_view_images 形状 [B, 3, H, W, C]（例如3个额外视图）
- RecordVideo 将其视为时间维度，实际提取：4 × 3 = **12 帧**
- 20 个 chunk_step × 12 帧 = 240 帧（远超 80 原始 step）

---

## 5. 为什么视频文件很小

### 压缩策略

1. **H.264 编码**（ffmpeg 默认）：
   - 帧间压缩（相邻帧差异小）
   - 运动补偿
   - 典型比率：480p RGB → 1MB per 10 秒（30fps）

2. **分辨率**：
   - ManiSkill 默认 640×480 或 512×512
   - 6x6 网格后约 3840×2880（8K 级，但全黑区域可高度压缩）

3. **关键帧率**：
   - imageio ffmpeg 默认 keyint=250
   - 大多数帧只存储运动向量和残差

### 文件大小估计

| 配置 | 帧数 | 分辨率 | FPS | 文件大小 |
|-----|------|--------|-----|----------|
| 单环境，100帧 | 100 | 512×512 | 30 | ~2-3 MB |
| 6x6 网格（36env），100帧 | 100 | 3840×2880 | 30 | ~5-8 MB |
| 多视图，240帧 | 240 | 512×512 | 30 | ~5-8 MB |

---

## 6. 关键代码总结

### 视频导出关键路径
```
env_worker.evaluate() 或 env_worker.interact()
  ├─ env.reset() → RecordVideo.reset()
  │   └─ add_new_frames(obs) → render_images += [frame]
  │
  └─ env.chunk_step(actions) → RecordVideo.chunk_step()
      └─ add_new_frames(obs_list) → render_images += [frame_1, frame_2, ...]
          └─ _extract_frame_batches(obs_list)
              ├─ [时间维度处理] (5D → 4个 4D 列表)
              └─ [环境维度处理] (batch → per-env)
                  └─ _append_frame(images) → tile_images() + info overlay
                      └─ render_images.append(tiled_frame)

env_worker.finish_rollout()
  └─ env.flush_video()
      └─ _submit_save(frames, mp4_path)
          └─ [ThreadPoolExecutor] _save_video(frames)
              └─ imageio.get_writer() → append_data() → close()
                  └─ ffmpeg 编码到 MP4
```

### 偏好数据收集关键路径
```
preference_collection_runner.run()
  └─ for epoch in range(num_collection_epochs):
      └─ _run_one_epoch()
          ├─ env.collect_preference_epoch()
          │   └─ 返回 {"rewards", "successes", "lengths"}
          │
          └─ rollout.collect_episodes_epoch()
              └─ 返回 [{"keyframes", "wrist_keyframes", "task_description"}]
          
          └─ build_episode_records()
              └─ [EpisodeRecord] 组装 (env结果 + rollout keyframes)

  └─ create_preference_pairs()
      └─ 组内配对 (success vs success, failure vs failure)
      └─ 按奖励排序 (chosen=高，rejected=低)
      
  └─ save_preference_pairs(pairs, output_path)
      └─ pickle.dump() 到文件
```

---

## 7. 现状与差异分析

### RecordVideo (Eval Video Export)
✅ **实现完整**：
- 支持任意数量的环境（自动 sqrt 网格布局）
- 支持多步观测（5D 张量展开为多帧）
- 支持多视图（列表/元组观测）
- 异步后台保存（不阻塞训练）
- 信息叠加（奖励、终止标志）
- ffmpeg 压缩（紧凑格式）

### PreferenceCollectionWorker (Keyframe Sampling)
⚠️ **设计差异**：
- 仅采样 **N 个关键帧**（均匀分布），而不是全部帧
- 不调用 `RecordVideo`（没有视频输出）
- 直接从 obs 提取 uint8 numpy，存储为 pickle
- 目的：**紧凑存储**（几十 KB per episode）而不是视频（几 MB per episode）

### 核心区别
| 特性 | RecordVideo | PreferenceCollectionWorker |
|-----|-----------|------------------------|
| 输出格式 | MP4 视频 | Pickle (EpisodeRecord) |
| 帧数 | 全部帧（100+） | 采样帧（8 个） |
| 文件大小 | 数 MB | 数十 KB |
| 用途 | 可视化/debug | 奖励模型训练数据 |
| 图像来源 | obs["main_images"] | obs["main_images"]/["wrist_images"] |

---

## 8. 关键文件索引

| 文件 | 模块 | 功能 |
|-----|------|------|
| `rlinf/envs/wrappers/record_video.py` | RecordVideo | 视频录制 wrapper |
| `rlinf/envs/utils.py` | tile_images, put_info_on_image | 图像处理工具 |
| `rlinf/workers/env/preference_env_worker.py` | PreferenceEnvWorker | 偏好数据env端 |
| `rlinf/workers/rollout/hf/preference_collection_worker.py` | PreferenceCollectionWorker | 偏好数据rollout端 |
| `rlinf/data/preference_data.py` | EpisodeRecord, PreferencePair, create_preference_pairs | 数据结构和配对逻辑 |
| `rlinf/runners/preference_collection_runner.py` | PreferenceCollectionRunner | 偏好数据收集入口 |
| `rlinf/runners/embodied_eval_runner.py` | EmbodiedEvalRunner | 标准 eval runner |
| `rlinf/envs/maniskill/maniskill_env.py` | ManiskillEnv | 观测生成和 chunk_step |

