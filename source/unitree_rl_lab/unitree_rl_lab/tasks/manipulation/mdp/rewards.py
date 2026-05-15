"""Reward functions for valve-turn task.

Only task-specific terms here. Generic terms (action_rate_l2, joint_vel_l2,
joint_acc_l2, joint_pos_limits) come from isaaclab.envs.mdp directly.

Stage 1: pressure_error (dense). g(θ) firmware-locked; no DR.
CONTEXT.md §RL spec, §g(θ).
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_wheel_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel angle θ (rad) for every env. Shape: (num_envs,)."""
    valve: Articulation = env.scene["valve_rig"]
    return valve.data.joint_pos[:, 0]


# ---------------------------------------------------------------------------
# Task-specific reward terms
# ---------------------------------------------------------------------------

def pressure_error(
    env: ManagerBasedRLEnv,
    p_des: float,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
    p_span: float,
) -> torch.Tensor:
    """Dense reward: −|p_now − p_des| / p_span ∈ [−1, 0].

    Args:
        p_des:      target pressure (PSI). Fixed in Stage 1.
        pressure_a: g(θ) slope (PSI/rad). Firmware-locked.
        pressure_b: g(θ) intercept (PSI). Firmware-locked.
        p_min / p_max: clamp bounds (PSI).
        p_span:     p_max − p_min (PSI).
    """
    theta = _get_wheel_angle(env)
    p_now = torch.clamp(pressure_a * theta + pressure_b, p_min, p_max)
    return -torch.abs(p_now - p_des) / p_span
