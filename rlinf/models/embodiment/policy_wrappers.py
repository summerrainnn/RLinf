"""Policy wrappers for generating diverse-quality trajectory data.

Three wrappers that degrade a base policy's behavior in different ways:
- ActionNoisePolicy: Gaussian noise on output actions.
- PerceptionNoisePolicy: Gaussian noise on input images.
- ActionDelayPolicy: Uses stale observations to simulate latency.

All wrappers expose the same ``predict_action_batch`` interface as
``BasePolicy`` so they can be used as drop-in replacements.
"""

from typing import Optional

import torch


class ActionNoisePolicy:
    """Add Gaussian noise to the model's output actions.

    Args:
        base_policy: the underlying policy.
        noise_std: standard deviation in the normalized action space ([-1, 1]
            for position/rotation, [0, 1] for gripper).
    """

    def __init__(self, base_policy, noise_std: float):
        self.base_policy = base_policy
        self.noise_std = noise_std

    def predict_action_batch(self, **kwargs):
        actions, result = self.base_policy.predict_action_batch(**kwargs)
        # actions: [B, num_action_chunks, action_dim]
        noise = torch.randn_like(actions) * self.noise_std
        actions = actions + noise
        # Position + rotation dims (first 6): clip to [-1, 1]
        actions[:, :, :6] = actions[:, :, :6].clamp(-1.0, 1.0)
        # Gripper dim(s) (7th onward): clip to [0, 1]
        if actions.shape[-1] > 6:
            actions[:, :, 6:] = actions[:, :, 6:].clamp(0.0, 1.0)
        return actions, result

    def __getattr__(self, name):
        return getattr(self.base_policy, name)


class PerceptionNoisePolicy:
    """Add Gaussian noise to input images (pixel space, uint8).

    Args:
        base_policy: the underlying policy.
        noise_std: pixel-space standard deviation on [0, 255].
    """

    def __init__(self, base_policy, noise_std: float):
        self.base_policy = base_policy
        self.noise_std = noise_std

    def predict_action_batch(self, env_obs=None, **kwargs):
        if env_obs is not None:
            env_obs = self._add_noise_to_obs(env_obs)
        return self.base_policy.predict_action_batch(env_obs=env_obs, **kwargs)

    def _add_noise_to_obs(self, env_obs: dict) -> dict:
        obs = dict(env_obs)
        for key in ("main_images", "wrist_images", "extra_view_images"):
            if key in obs and obs[key] is not None:
                img = obs[key].float()
                noise = torch.randn_like(img) * self.noise_std
                obs[key] = (img + noise).clamp(0, 255).to(torch.uint8)
        return obs

    def get_noised_obs(self, env_obs: dict) -> dict:
        """Return a noised copy of the observation (for video recording)."""
        return self._add_noise_to_obs(env_obs)

    def __getattr__(self, name):
        return getattr(self.base_policy, name)


class ActionDelayPolicy:
    """Simulate action latency by feeding stale observations to the policy.

    Args:
        base_policy: the underlying policy.
        delay_steps: number of chunk-steps to delay (1 step = one call to
            predict_action_batch).
    """

    def __init__(self, base_policy, delay_steps: int):
        self.base_policy = base_policy
        self.delay_steps = delay_steps
        self._obs_buffer: list[dict] = []
        self._step_count = 0

    def reset_buffer(self) -> None:
        """Call at the start of each episode."""
        self._obs_buffer.clear()
        self._step_count = 0

    def predict_action_batch(self, env_obs=None, **kwargs):
        if env_obs is not None:
            self._obs_buffer.append(self._clone_images(env_obs))
            self._step_count += 1
            delayed_idx = max(0, self._step_count - 1 - self.delay_steps)
            delayed_obs = self._merge_delayed(env_obs, self._obs_buffer[delayed_idx])
            return self.base_policy.predict_action_batch(
                env_obs=delayed_obs, **kwargs
            )
        return self.base_policy.predict_action_batch(env_obs=env_obs, **kwargs)

    def get_delayed_obs(self, env_obs: dict) -> dict:
        """Return the delayed observation (for video recording)."""
        delayed_idx = max(0, self._step_count - 1 - self.delay_steps)
        if self._obs_buffer:
            return self._merge_delayed(env_obs, self._obs_buffer[delayed_idx])
        return env_obs

    def _clone_images(self, obs: dict) -> dict:
        cloned = {}
        for key in ("main_images", "wrist_images", "extra_view_images"):
            if key in obs and obs[key] is not None:
                cloned[key] = obs[key].clone()
        return cloned

    def _merge_delayed(self, current_obs: dict, delayed_images: dict) -> dict:
        merged = dict(current_obs)
        for key in ("main_images", "wrist_images", "extra_view_images"):
            if key in delayed_images:
                merged[key] = delayed_images[key]
        return merged

    def __getattr__(self, name):
        return getattr(self.base_policy, name)


def create_policy_wrapper(
    base_policy,
    wrapper_type: Optional[str],
    wrapper_params: Optional[dict] = None,
):
    """Factory function to create a policy wrapper.

    Args:
        base_policy: the underlying policy model.
        wrapper_type: one of "action_noise", "perception_noise", "action_delay",
            or None for no wrapper.
        wrapper_params: kwargs passed to the wrapper constructor.

    Returns:
        Wrapped or original policy.
    """
    if wrapper_type is None:
        return base_policy
    params = wrapper_params or {}
    if wrapper_type == "action_noise":
        return ActionNoisePolicy(base_policy, **params)
    if wrapper_type == "perception_noise":
        return PerceptionNoisePolicy(base_policy, **params)
    if wrapper_type == "action_delay":
        return ActionDelayPolicy(base_policy, **params)
    raise ValueError(f"Unknown wrapper type: {wrapper_type}")
