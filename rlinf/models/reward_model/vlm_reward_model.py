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

"""VLM-based reward model for robot trajectory evaluation.

Architecture:

* **Backbone** – any HuggingFace multimodal LLM (default: Qwen2.5-VL).
  Visual tokens come from N evenly-spaced keyframes of the episode; text
  tokens encode the task description and scoring rubric.
* **Value head** – a two-layer MLP that maps the pooled backbone hidden state
  to a scalar reward score.

The model is trained with a Bradley-Terry contrastive loss on preference pairs
(see :mod:`rlinf.runners.reward_model_trainer`).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rlinf.utils.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Value head (shared with existing policy models)
# ---------------------------------------------------------------------------


class ScalarValueHead(nn.Module):
    """Two-layer MLP → scalar."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, D].  Returns: [B] scalar scores."""
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main reward model
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are an expert robot manipulation evaluator. "
    "You will be shown a sequence of keyframes from a robot episode together "
    "with the task instruction. "
    "Evaluate the episode quality according to the following criteria:\n"
    "1. Task completion: Did the robot successfully accomplish the stated goal?\n"
    "2. Execution efficiency: Did the robot complete the task with minimal "
    "unnecessary motion or wasted steps?\n"
    "3. Motion smoothness: Were the robot's movements fluid and controlled, "
    "avoiding abrupt jerks or collisions?\n"
    "4. Object handling: Did the robot interact with objects safely and "
    "precisely, without dropping or knocking them over?\n"
    "Your hidden representation will be used to produce a scalar quality score "
    "for this episode via a learned value head."
)


class VLMRewardModel(nn.Module):
    """VLM backbone + scalar value head for trajectory-level reward prediction.

    The model takes a **task description** and a **list of N keyframe images**
    (evenly spaced across the episode) and outputs a scalar reward score.
    During preference learning the score is compared between two trajectories
    using the Bradley-Terry loss.

    Supported VLM backbones (HuggingFace):
        * ``Qwen/Qwen2.5-VL-*`` (default)
        * Any HuggingFace AutoModel with ``past_key_values`` and hidden states.

    Args:
        model_path: Path to the pretrained VLM (local directory or HF hub id).
        hidden_dim: Hidden size of the value-head MLP.
        freeze_backbone: If ``True``, freeze all VLM parameters and only train
            the value head.  Set to ``False`` to fine-tune the entire model
            (typically with LoRA or full fine-tuning).
        torch_dtype: Dtype for VLM weights (``None`` → bfloat16).
        use_lora: Attach LoRA adapters to the VLM (requires ``peft``).
        lora_rank: LoRA rank (only used when ``use_lora=True``).
        lora_alpha: LoRA alpha (only used when ``use_lora=True``).
    """

    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        logger.info(f"Loading VLM reward model backbone from: {model_path}")
        self._load_backbone(model_path, torch_dtype)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        if use_lora:
            self._attach_lora(lora_rank, lora_alpha)

        # Determine backbone hidden size
        backbone_hidden = self._get_hidden_size()
        self.value_head = ScalarValueHead(backbone_hidden, hidden_dim)

    # ------------------------------------------------------------------
    # Backbone loading helpers
    # ------------------------------------------------------------------

    def _load_backbone(self, model_path: str, dtype: torch.dtype) -> None:
        """Load the VLM backbone and processor."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        try:
            self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                output_hidden_states=True,
                device_map="cpu",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            self._backbone_type = "qwen2vl"
            logger.info("Loaded Qwen2VL backbone.")
            return
        except Exception:
            pass

        # Generic fallback: try AutoModelForCausalLM with vision tower
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                output_hidden_states=True,
                trust_remote_code=True,
                device_map="cpu",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._backbone_type = "generic"
            logger.info("Loaded generic VLM backbone.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load VLM backbone from {model_path}: {e}"
            ) from e

    def _get_hidden_size(self) -> int:
        """Return the last-layer hidden size of the backbone."""
        cfg = self.backbone.config
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError("Cannot determine backbone hidden size from config.")

    def _attach_lora(self, rank: int, alpha: int) -> None:
        """Attach LoRA adapters to the backbone (requires ``peft``)."""
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        task_descriptions: list[str],
        keyframes_list: list[list[np.ndarray]],
    ) -> dict[str, torch.Tensor]:
        """Build tokenized model inputs from raw numpy keyframe sequences and text.

        Each item in the batch is presented as a sequence of labeled images
        (Frame 1/N … Frame N/N) followed by the task description, allowing
        the backbone to reason about temporal progress across the episode.

        Args:
            task_descriptions: List of B task description strings.
            keyframes_list: List of B episodes, each a list of N uint8 arrays
                [H, W, C] representing evenly-spaced episode keyframes.

        Returns:
            Tokenized inputs ready for ``self.backbone``.
        """
        messages_batch = []
        images_batch = []

        for task_desc, frames in zip(task_descriptions, keyframes_list):
            n_frames = len(frames)
            content: list[dict] = [
                {
                    "type": "text",
                    "text": f"Task: {task_desc}\n\nEpisode keyframes ({n_frames} frames):",
                }
            ]
            pil_frames = []
            for frame_idx, frame in enumerate(frames):
                if frame is None:
                    continue
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
                content.append(
                    {"type": "text", "text": f"\nFrame {frame_idx + 1}/{n_frames}:"}
                )
                content.append({"type": "image", "image": pil_frame})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            messages_batch.append(messages)
            images_batch.extend(pil_frames)

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages_batch
        ]

        inputs = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        task_descriptions: list[str],
        keyframes_list: list[list[np.ndarray]],
    ) -> torch.Tensor:
        """Compute reward scores for a batch of trajectory keyframe sequences.

        Args:
            task_descriptions: B task description strings.
            keyframes_list: B episodes, each a list of N uint8 numpy arrays
                [H, W, C] representing evenly-spaced episode keyframes.

        Returns:
            Scalar reward scores, shape [B], float32.
        """
        device = next(self.backbone.parameters()).device
        inputs = self._prepare_inputs(task_descriptions, keyframes_list)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs, output_hidden_states=True)

        # Pool the last hidden state over the sequence dimension
        hidden = outputs.hidden_states[-1]  # [B, T, D]
        # Mean-pool over non-padding positions
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)  # [B, D]

        pooled = pooled.to(torch.float32)
        scores = self.value_head(pooled)  # [B]
        return scores

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str) -> None:
        """Save value head weights and backbone (or LoRA adapters) to *save_dir*."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.value_head.state_dict(), os.path.join(save_dir, "value_head.pt"))
        try:
            self.backbone.save_pretrained(save_dir)
            self.processor.save_pretrained(save_dir)
            logger.info(f"Saved backbone + value head to {save_dir}")
        except Exception as e:
            logger.warning(f"Could not save backbone via save_pretrained: {e}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        checkpoint_dir: str,
        **kwargs,
    ) -> "VLMRewardModel":
        """Load a trained reward model from *checkpoint_dir*.

        For LoRA checkpoints (identified by ``adapter_config.json``), the base
        backbone is loaded from *model_path* and the adapter weights are merged
        from *checkpoint_dir*.  For full-weight checkpoints the backbone is
        loaded directly from *checkpoint_dir*.

        Args:
            model_path: Original pretrained VLM path (used to reload backbone).
            checkpoint_dir: Directory saved by :meth:`save_pretrained`.
            **kwargs: Passed to ``__init__``.

        Returns:
            Loaded :class:`VLMRewardModel`.
        """
        adapter_cfg = os.path.join(checkpoint_dir, "adapter_config.json")
        is_lora_ckpt = os.path.exists(adapter_cfg)

        if is_lora_ckpt:
            # Load the base backbone first, then apply saved LoRA adapter.
            # Do NOT pass use_lora=True to __init__ — we load the saved
            # adapter directly instead of creating a fresh one.
            init_kwargs = {k: v for k, v in kwargs.items() if k != "use_lora"}
            model = cls(model_path=model_path, use_lora=False, **init_kwargs)

            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, checkpoint_dir)
            logger.info(f"Loaded LoRA adapter from {checkpoint_dir}")
        else:
            model = cls(model_path=checkpoint_dir, **kwargs)

        vh_path = os.path.join(checkpoint_dir, "value_head.pt")
        if os.path.exists(vh_path):
            model.value_head.load_state_dict(torch.load(vh_path, map_location="cpu"))
            logger.info(f"Loaded value head from {vh_path}")
        return model


# ---------------------------------------------------------------------------
# Dummy reward model (for debugging)
# ---------------------------------------------------------------------------


class DummyRewardModel(nn.Module):
    """Reward model stub that always outputs zero — useful for debugging pipelines.

    Accepts the same ``forward`` signature as :class:`VLMRewardModel` so it
    can be used as a drop-in replacement without any model weights or GPU.

    To enable, set ``reward_model.use_dummy: true`` (trainer) or
    ``reward_model_eval.use_dummy: true`` (eval runner) in your YAML config.
    """

    def forward(
        self,
        task_descriptions: list[str],
        keyframes_list: list[list[np.ndarray]],
    ) -> torch.Tensor:
        """Return a zero score for every item in the batch.

        Args:
            task_descriptions: B task description strings (unused).
            keyframes_list: B keyframe sequences (unused).

        Returns:
            Zero tensor of shape [B], float32.
        """
        return torch.zeros(len(task_descriptions), dtype=torch.float32)

    def save_pretrained(self, save_dir: str) -> None:  # noqa: D102
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"DummyRewardModel: no weights to save (dir={save_dir}).")

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "DummyRewardModel":  # noqa: D102
        return cls()


def build_reward_model(
    use_dummy: bool = False,
    model_path: str = "",
    checkpoint_dir: str = "",
    hidden_dim: int = 512,
    freeze_backbone: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> Union["VLMRewardModel", "DummyRewardModel"]:
    """Factory that returns either a real or dummy reward model.

    Args:
        use_dummy: When ``True`` return a :class:`DummyRewardModel` (no GPU
            or model weights required).  All other args are ignored.
        model_path: Pretrained VLM path (only used for real model).
        checkpoint_dir: Checkpoint directory to load from (real model only).
            Pass an empty string to skip checkpoint loading and return a
            freshly initialised model.
        **remaining kwargs**: Forwarded to :class:`VLMRewardModel.__init__`.

    Returns:
        A reward model with the same ``forward(task_descriptions, keyframes_list)``
        interface.
    """
    if use_dummy:
        logger.info("Using DummyRewardModel (all scores will be 0).")
        return DummyRewardModel()

    if checkpoint_dir:
        return VLMRewardModel.from_pretrained(
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            torch_dtype=torch_dtype,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    return VLMRewardModel(
        model_path=model_path,
        hidden_dim=hidden_dim,
        freeze_backbone=freeze_backbone,
        torch_dtype=torch_dtype,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def bradley_terry_loss(scores_chosen: torch.Tensor, scores_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry contrastive loss for preference learning.

    Minimises :math:`-\\log \\sigma(r_w - r_l)` where ``r_w`` is the reward
    of the chosen (preferred) trajectory and ``r_l`` is the rejected one.

    Args:
        scores_chosen:   [B] reward scores for the preferred trajectories.
        scores_rejected: [B] reward scores for the dis-preferred trajectories.

    Returns:
        Scalar loss (mean over the batch).
    """
    return -F.logsigmoid(scores_chosen - scores_rejected).mean()


def margin_bradley_terry_loss(
    scores_chosen: torch.Tensor,
    scores_rejected: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry loss with a score-dependent margin.

    Minimises :math:`-\\log \\sigma(r_w - r_l - m)` where *m* is typically the
    difference between the annotation scores of the chosen and rejected
    trajectories.

    Args:
        scores_chosen:   [B] reward scores for preferred trajectories.
        scores_rejected: [B] reward scores for dis-preferred trajectories.
        margin:          [B] score margin (e.g. ``label_chosen - label_rejected``).

    Returns:
        Scalar loss (mean over the batch).
    """
    return -F.logsigmoid(scores_chosen - scores_rejected - margin).mean()


def listwise_plackett_luce_loss(
    scores: torch.Tensor,
    rankings: torch.Tensor,
) -> torch.Tensor:
    """Plackett-Luce listwise ranking loss.

    For each group of *G* items, the loss decomposes the ranking probability as
    a product of successive softmax choices:

    .. math::

        L = -\\sum_{i=0}^{G-2}
              \\bigl(s_{\\sigma(i)} - \\log \\sum_{j=i}^{G-1} e^{s_{\\sigma(j)}}\\bigr)

    Args:
        scores:   [B, G] — RM scores for B groups, each with G trajectories.
        rankings: [B, G] — ranking per item (0 = best, G-1 = worst).

    Returns:
        Scalar loss (mean over batches and positions).
    """
    B, G = scores.shape
    total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    for b in range(B):
        order = rankings[b].argsort()  # rank → trajectory index
        ordered = scores[b][order]
        for i in range(G - 1):
            total_loss -= ordered[i] - torch.logsumexp(ordered[i:], dim=0)
    return total_loss / (B * max(G - 1, 1))


def score_regression_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
) -> torch.Tensor:
    """MSE regression loss for direct score prediction.

    Args:
        predicted_scores: [B] predicted RM scores.
        target_scores:    [B] ground-truth annotation scores.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(predicted_scores, target_scores)
