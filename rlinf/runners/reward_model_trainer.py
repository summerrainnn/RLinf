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

"""Standalone reward model trainer with DDP multi-GPU support.

Loads a preference dataset (``List[PreferencePair]``) produced by
:class:`~rlinf.runners.preference_collection_runner.PreferenceCollectionRunner`,
trains a :class:`~rlinf.models.reward_model.VLMRewardModel` with the
Bradley-Terry contrastive loss, and saves checkpoints.

Multi-GPU training uses PyTorch DistributedDataParallel (DDP).  Each GPU
holds a **full replica** of the model and processes a different subset of
the data.  Launch with ``torchrun --nproc_per_node=<N>``.

Config keys (read from ``cfg.reward_model``):
    * ``model_path``        – pretrained VLM path.
    * ``data_path``         – path to the ``.pkl`` preference dataset.
    * ``output_dir``        – where to save checkpoints.
    * ``batch_size``        – per-GPU training batch size.
    * ``num_epochs``        – number of passes over the dataset.
    * ``lr``                – learning rate.
    * ``value_head_lr``     – optional separate LR for the value head.
    * ``freeze_backbone``   – bool, freeze backbone and only train value head.
    * ``use_lora``          – bool, attach LoRA adapters to backbone.
    * ``lora_rank``         – int, LoRA rank.
    * ``lora_alpha``        – int, LoRA alpha.
    * ``save_every_n_steps``– checkpoint save frequency.
    * ``hidden_dim``        – value-head hidden size.
    * ``torch_dtype``       – "bf16" | "fp16" | "fp32".
    * ``device``            – "cuda" | "cpu" | "auto".
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.preference_data import PreferencePair, get_keyframes, load_preference_pairs
from rlinf.models.reward_model.vlm_reward_model import bradley_terry_loss, build_reward_model
from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

logger = get_logger()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PreferenceDataset(Dataset):
    """PyTorch Dataset wrapping a list of ``PreferencePair`` objects.

    Supports both video-based and legacy keyframe-based episodes.
    When an episode has ``video_path``, frames are loaded on demand.
    """

    def __init__(self, pairs: list[PreferencePair], n_keyframes: int = 8):
        self.pairs = pairs
        self._n_keyframes = n_keyframes

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "chosen_task_desc": pair.chosen.task_description,
            "chosen_keyframes": get_keyframes(pair.chosen, self._n_keyframes),
            "rejected_task_desc": pair.rejected.task_description,
            "rejected_keyframes": get_keyframes(pair.rejected, self._n_keyframes),
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Collate a list of preference items — keyframes stay as lists of numpy arrays."""
    return {
        "chosen_task_desc": [item["chosen_task_desc"] for item in batch],
        "chosen_keyframes": [item["chosen_keyframes"] for item in batch],
        "rejected_task_desc": [item["rejected_task_desc"] for item in batch],
        "rejected_keyframes": [item["rejected_keyframes"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RewardModelTrainer:
    """Trains a VLMRewardModel on a preference dataset.

    Supports single-GPU and multi-GPU (DDP) training.  When launched via
    ``torchrun --nproc_per_node=N``, each GPU holds a full model replica and
    processes a different data subset.  DDP is auto-detected from the
    ``LOCAL_RANK`` environment variable set by ``torchrun``.

    Args:
        cfg: Full Hydra config.  Reward-model-specific settings are read from
            ``cfg.reward_model``.
    """

    def __init__(self, cfg: "DictConfig"):
        self.cfg = cfg
        rm_cfg = cfg.reward_model

        self.data_path: str = rm_cfg.data_path
        self.output_dir: str = rm_cfg.output_dir
        self.batch_size: int = int(rm_cfg.get("batch_size", 4))
        self.num_epochs: int = int(rm_cfg.get("num_epochs", 10))
        self.lr: float = float(rm_cfg.get("lr", 1e-5))
        self.value_head_lr: float = float(rm_cfg.get("value_head_lr", 1e-4))
        self.save_every_n_steps: int = int(rm_cfg.get("save_every_n_steps", 100))
        self.grad_accum_steps: int = int(rm_cfg.get("gradient_accumulation_steps", 1))
        self.n_keyframes: int = int(rm_cfg.get("n_keyframes", 8))

        dtype_str = rm_cfg.get("torch_dtype", "bf16")
        self.torch_dtype = _parse_dtype(dtype_str)

        # ----- DDP setup -----
        self.distributed = "LOCAL_RANK" in os.environ
        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            # Set device BEFORE any CUDA operations to avoid allocating on GPU 0
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group("nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main = self.local_rank == 0
        else:
            self.local_rank = 0
            self.world_size = 1
            device_str = rm_cfg.get("device", "auto")
            self.device = _resolve_device(device_str)
            self.is_main = True

        # Build model — loaded on CPU first, then moved to the assigned device
        use_dummy = bool(rm_cfg.get("use_dummy", False))
        self.model = build_reward_model(
            use_dummy=use_dummy,
            model_path=rm_cfg.get("model_path", ""),
            hidden_dim=int(rm_cfg.get("hidden_dim", 512)),
            freeze_backbone=bool(rm_cfg.get("freeze_backbone", False)),
            torch_dtype=self.torch_dtype,
            use_lora=bool(rm_cfg.get("use_lora", False)),
            lora_rank=int(rm_cfg.get("lora_rank", 16)),
            lora_alpha=int(rm_cfg.get("lora_alpha", 32)),
        )
        if not use_dummy:
            # Move model to this process's GPU
            self.model = self.model.to(self.device)
            # Enable gradient checkpointing to reduce activation memory
            if hasattr(self.model, "backbone"):
                self.model.backbone.gradient_checkpointing_enable()
                if self.is_main:
                    logger.info("Enabled gradient checkpointing on backbone.")

            # Wrap with DDP for multi-GPU training
            if self.distributed:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )
                if self.is_main:
                    logger.info(
                        f"DDP enabled: {self.world_size} GPUs, "
                        f"per-GPU batch_size={self.batch_size}."
                    )

    @property
    def _raw_model(self):
        """Access the unwrapped model (skipping DDP wrapper)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Load dataset, train, and save the final checkpoint."""
        if self.is_main:
            logger.info(f"Loading preference dataset from {self.data_path}")
        pairs = load_preference_pairs(self.data_path)
        if self.is_main:
            logger.info(f"Loaded {len(pairs)} preference pairs.")

        if len(pairs) == 0:
            raise ValueError("Preference dataset is empty — cannot train.")

        random.shuffle(pairs)
        dataset = PreferenceDataset(pairs, n_keyframes=self.n_keyframes)

        # Use DistributedSampler for DDP — each GPU gets a different subset
        sampler = (
            DistributedSampler(dataset, num_replicas=self.world_size, rank=self.local_rank)
            if self.distributed
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=_collate_fn,
            drop_last=True,
        )

        # Separate LR for value head vs backbone
        raw = self._raw_model
        backbone_params = list(raw.backbone.parameters())
        vh_params = list(raw.value_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr},
                {"params": vh_params, "lr": self.value_head_lr},
            ]
        )

        os.makedirs(self.output_dir, exist_ok=True)
        global_step = 0
        best_loss = float("inf")

        if self.is_main:
            logger.info(
                f"Starting training: {self.num_epochs} epochs, "
                f"per-GPU batch_size={self.batch_size}, "
                f"world_size={self.world_size}, "
                f"grad_accum={self.grad_accum_steps} "
                f"(effective={self.batch_size * self.world_size * self.grad_accum_steps})."
            )

        for epoch in range(self.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_losses = []
            self.model.train()

            for micro_step, batch in enumerate(dataloader):
                # Forward chosen
                scores_chosen = self.model(
                    task_descriptions=batch["chosen_task_desc"],
                    keyframes_list=batch["chosen_keyframes"],
                )
                # Forward rejected
                scores_rejected = self.model(
                    task_descriptions=batch["rejected_task_desc"],
                    keyframes_list=batch["rejected_keyframes"],
                )

                loss = bradley_terry_loss(scores_chosen, scores_rejected)
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (micro_step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Record un-scaled loss for logging
                loss_val = float(loss.detach().cpu()) * self.grad_accum_steps
                epoch_losses.append(loss_val)

                if (
                    self.is_main
                    and global_step > 0
                    and global_step % max(1, self.save_every_n_steps) == 0
                    and (micro_step + 1) % self.grad_accum_steps == 0
                ):
                    ckpt_dir = os.path.join(
                        self.output_dir, f"checkpoint_step_{global_step}"
                    )
                    self._raw_model.save_pretrained(ckpt_dir)
                    logger.info(
                        f"Step {global_step}: loss={loss_val:.4f}. "
                        f"Saved checkpoint to {ckpt_dir}"
                    )

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if self.is_main:
                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}: avg_loss={avg_loss:.4f}"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = os.path.join(self.output_dir, "best")
                    self._raw_model.save_pretrained(best_dir)
                    logger.info(f"New best loss {best_loss:.4f}. Saved to {best_dir}")

        # Save final checkpoint (rank 0 only)
        if self.is_main:
            final_dir = os.path.join(self.output_dir, "final")
            self._raw_model.save_pretrained(final_dir)
            logger.info(f"Training complete. Final model saved to {final_dir}")

        if self.distributed:
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        logger.warning(f"Unknown torch_dtype '{dtype_str}', defaulting to bfloat16.")
        return torch.bfloat16
    return mapping[dtype_str]


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
