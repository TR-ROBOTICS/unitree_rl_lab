"""Task-specific observation functions for valve-turn task.

All pressure terms derived from g(θ) — firmware-locked, no DR.
CONTEXT.md §g(θ): p_now = a·θ + b, clamp [15, 200] PSI.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_THETA_MAX_FALLBACK: float = 50.27  # rad — nan_to_num guard


def valve_pressure_now(
    env: ManagerBasedRLEnv,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
    p_span: float,
) -> torch.Tensor:
    """Current pressure p_now, normalized to [0, 1] via p_span.

    Computed from valve angle θ via g(θ) = a·θ + b (firmware-locked).
    Clamped to [p_min, p_max] before normalization — matches the clamped
    signal fed back to the real ESP32 control loop.

    Shape: (num_envs, 1).
    """
    valve: Articulation = env.scene["valve_rig"]
    theta = valve.data.joint_pos[:, 0]
    theta = torch.nan_to_num(theta, nan=0.0, posinf=_THETA_MAX_FALLBACK, neginf=0.0)
    p_now = torch.clamp(pressure_a * theta + pressure_b, p_min, p_max)
    return (p_now / p_span).unsqueeze(-1)


def valve_pos_robot_frame(
    env: ManagerBasedRLEnv,
    valve_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Valve root position relative to robot root position, in world frame.

    Used by reach policy to give the policy a valve-relative spatial signal
    (valve position is randomized per episode; robot base is fixed).

    Shape: (num_envs, 3).
    """
    valve: Articulation = env.scene[valve_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    valve_pos = valve.data.root_pos_w   # (num_envs, 3)
    robot_pos = robot.data.root_pos_w   # (num_envs, 3)
    rel = valve_pos - robot_pos
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)


def last_arm_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous policy step's action (14d arm Δ-targets).

    Provides temporal context — policy can infer direction of last motion.
    Sourced from action_manager.action (current env step's action buffer,
    equivalent to the previous step's output at the next observation query).

    nan_to_num guard prevents NaN propagation on first step (action not yet set).
    Shape: (num_envs, 14).
    """
    action = env.action_manager.action
    return torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)


def valve_pressure_des(
    env: ManagerBasedRLEnv,
    p_des: float,
    p_span: float,
) -> torch.Tensor:
    """Target pressure p_des, normalized to [0, 1] via p_span.

    Stage 1: constant (fixed p_des=100 PSI). Stage 3: will vary per episode
    from command manager — override this term or source from env buffer.
    Including it now lets the Stage-1 policy learn to condition on the target
    without architecture changes at Stage 3 (obs dim stays 30).

    Shape: (num_envs, 1).
    """
    p_des_norm = p_des / p_span
    return torch.full(
        (env.num_envs, 1),
        p_des_norm,
        dtype=torch.float32,
        device=env.device,
    )


def valve_pressure_des_random(
    env: ManagerBasedRLEnv,
    p_span: float,
) -> torch.Tensor:
    """Per-env p_des from env.p_des_buf, normalized to [0, 1].

    Stage 2+: reads the buffer written by reset_p_des_random each episode.
    Shape: (num_envs, 1).
    """
    p_des = getattr(env, "p_des_buf", None)
    if p_des is None:
        # Called at init time for shape probe — buffer not yet set. Return zeros.
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return (p_des / p_span).unsqueeze(-1).clamp(0.0, 1.0)
