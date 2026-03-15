"""
Minimal standalone test for ManiskillEnv initialization and reset.

Bypasses Ray, Hydra, and model loading entirely.
Run with:
    python test_env.py

This catches all env-side bugs (get_language_instruction, camera key errors,
control_mode assertion errors, etc.) in a few seconds.
"""

from omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

cfg = OmegaConf.create(
    {
        "seed": 42,
        "auto_reset": True,
        "use_rel_reward": False,
        "ignore_terminations": False,
        "use_full_state": False,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "wrap_obs_mode": "default",
        "reward_mode": "only_success",
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
        },
        "init_params": {
            "id": "RotateSingleObjectInHandLevel0-v1",
            "obs_mode": "rgb+state",
            "control_mode": None,
            "sim_backend": "gpu",
            "sim_config": {
                "sim_freq": 100,
                "control_freq": 20,
            },
        },
    }
)

NUM_ENVS = 1  # keep small for fast iteration

print("Creating ManiskillEnv...")
env = ManiskillEnv(
    cfg=cfg,
    num_envs=NUM_ENVS,
    seed_offset=0,
    total_num_processes=1,
    worker_info=None,
    record_metrics=False,
)
print("  env created OK")

print("Calling reset()...")
obs, info = env.reset()
print("  reset OK")
print("  obs keys:", list(obs.keys()))
if "task_descriptions" in obs:
    print("  task_descriptions:", obs["task_descriptions"])
if "main_images" in obs:
    print("  main_images shape:", obs["main_images"].shape)
if "states" in obs:
    print("  states shape:", obs["states"].shape)

print("\nAll checks passed.")
