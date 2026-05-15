"""Reward functions for valve-turn task.

Only task-specific terms here. Generic terms come from isaaclab.envs.mdp directly.

Stage 1: pressure_error (dense) + wheel_vel_toward_target.
g(θ) firmware-locked; no DR. CONTEXT.md §RL spec, §g(θ).
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_THETA_MAX_FALLBACK: float = 50.27  # rad — nan_to_num guard


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_wheel_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel angle θ (rad) for every env. Shape: (num_envs,)."""
    valve: Articulation = env.scene["valve_rig"]
    theta = valve.data.joint_pos[:, 0]
    return torch.nan_to_num(theta, nan=0.0, posinf=_THETA_MAX_FALLBACK, neginf=0.0)


def _get_wheel_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel angular velocity ω (rad/s). Shape: (num_envs,)."""
    valve: Articulation = env.scene["valve_rig"]
    omega = valve.data.joint_vel[:, 0]
    return torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)


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


def wheel_vel_toward_target(
    env: ManagerBasedRLEnv,
    p_des: float,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
) -> torch.Tensor:
    """Reward wheel angular velocity in direction that reduces pressure error.

    CCW (positive ω) raises p_now → rewarded when p_now < p_des.
    CW (negative ω) lowers p_now → rewarded when p_now > p_des.
    Only correct-direction motion rewarded; wrong direction clipped to 0.
    Range: [0, ∞).
    """
    theta = _get_wheel_angle(env)
    p_now = torch.clamp(pressure_a * theta + pressure_b, p_min, p_max)
    error_sign = torch.sign(p_des - p_now)   # +1 need CCW, -1 need CW
    omega = _get_wheel_vel(env)
    # Clamp to [0, 5] rad/s — prevents policy exploiting unbounded wheel spin
    return torch.clamp(omega * error_sign, min=0.0, max=5.0)


def arm_joint_motion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Small bonus for arm joint velocity magnitude.

    Breaks zero-motion local minimum when arms not yet contacting wheel.
    Stage 1 only — remove once contact + wheel-turn established.
    """
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # Clamp to [0, 1] — prevents policy exploiting unbounded norm to spin joints
    return torch.clamp(torch.nan_to_num(torch.norm(vel, dim=-1), nan=0.0), max=1.0)
