"""Termination terms for valve-turn task.

joint_vel_runaway — catastrophic-only joint-velocity guard with a post-reset
grace window.

Why this replaces IsaacLab `joint_vel_out_of_limit` / `..._manual_limit`:

Runs 7-9 showed every strict joint-vel termination kills Stage-1 training while
value/surrogate loss stays ~1e-3 (no real numeric blow-up — the dt fix in Run 6
permanently solved KE explosion). Causes:
  * `joint_vel_out_of_limit` ties the threshold to per-joint
    `velocity_limit_sim` → conflates actuator saturation with abort.
  * `reset_finger_grip` snaps fingers open→curl in one reset; the grip-close
    impulse reacts through the wrist → arm joint >10 rad/s in step 1-2 → instant
    termination at episode_length≈2. Reset artifact, not real dynamics (the real
    robot starts from a settled grasp, never a snap).

This term:
  1. No strict limit — `max_velocity` set to a value physically impossible for a
     G1 arm joint (≈50 rad/s); fires only on true runaway / NaN-ish divergence.
  2. Real-robot reference: unitree_sdk2 g1/common/terminations.hpp
     ::joint_vel_out_of_limit (flat 10.0, any motor → FSM Passive) is NOT wired
     into unitree_rl_lab deploy; real vel enforcement = motor firmware torque
     saturation. So a high catastrophic-only guard is faithful, not lax.
  3. Grace window: suppressed for the first `grace_steps` policy steps of every
     episode so the reset-grip impulse + contact settling cannot trip it.

CONTEXT.md §Episode end: Success / Timeout / contact-loss only — no joint-vel
termination in spec. This is a numerical safety net, nothing more.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import pressure


def pressure_success_hold(
    env: ManagerBasedRLEnv,
    hold_steps: int,
) -> torch.Tensor:
    """Terminate (success) when |p_now − p_des| < EPS_SIM for hold_steps consecutive steps.

    PRD user story 20: K=10 vision-frame increments × 5 policy steps = 50 steps = 1 s.
    CONTEXT.md §Success: ε_sim ≈ 1% of span (~1.85 PSI).

    p_now from the firmware-locked g(θ) model; p_des per-env from env.p_des_buf;
    tolerance EPS_SIM from the pressure model. Counter is per-env and resets when
    the condition is not met or on episode start. DoneTerm must use time_out=False
    — success terminal; RSL-RL bootstraps with 0 (not critic value) for non-timeout
    endings.

    Args:
        hold_steps: consecutive steps required inside tolerance (50 = 1 s @50 Hz).

    Returns:
        Bool [num_envs]: True where hold counter has reached hold_steps.
    """
    p_des = pressure.p_des(env)
    if p_des is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    within_tol = torch.abs(pressure.p_now(env) - p_des) < pressure.EPS_SIM  # (num_envs,) bool

    # Retrieve or initialise counter
    counter = getattr(env, "_success_hold_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._success_hold_counter = counter

    # Reset counter at episode start (episode_length_buf == 1 on first post-reset step)
    just_reset = env.episode_length_buf <= 1
    counter = torch.where(just_reset, torch.zeros_like(counter), counter)

    # Increment where within tolerance, reset where outside
    counter = torch.where(within_tol, counter + 1, torch.zeros_like(counter))
    env._success_hold_counter = counter

    return counter >= hold_steps


def reach_handoff_condition(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    valve_hub_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
    depth_threshold: float,
    inplane_threshold: float,
) -> torch.Tensor:
    """Terminate reach episode when both hands satisfy handoff position criteria.

    Per ADR 0004: handoff fires when BOTH hands satisfy:
        |Δx| < depth_threshold       (perpendicular to wheel face, world X assumed)
        √(Δy² + Δz²) < inplane_threshold  (within wheel plane)

    Δ is hand_pos − target_pos in world frame.
    Targets are computed as hub_COM_pos_w + per-hand FK offset.

    Args:
        depth_threshold:   REACH_DEPTH_THRESHOLD = 0.01 m.
        inplane_threshold: REACH_INPLANE_THRESHOLD = 0.03 m.

    Returns:
        Bool (num_envs,): True where handoff condition is met.
    """
    valve: Articulation = env.scene[valve_hub_cfg.name]
    robot_L: Articulation = env.scene[left_hand_cfg.name]
    robot_R: Articulation = env.scene[right_hand_cfg.name]

    hub_pos = valve.data.body_pos_w[:, valve_hub_cfg.body_ids[0], :]  # (N, 3)

    off_L = torch.tensor(left_target_offset, device=env.device, dtype=torch.float32)
    off_R = torch.tensor(right_target_offset, device=env.device, dtype=torch.float32)
    target_L = hub_pos + off_L
    target_R = hub_pos + off_R

    left_pos  = robot_L.data.body_pos_w[:, left_hand_cfg.body_ids[0], :]
    right_pos = robot_R.data.body_pos_w[:, right_hand_cfg.body_ids[0], :]

    def _hand_ok(hand_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = hand_pos - target                                 # (N, 3)
        depth_ok   = torch.abs(delta[:, 0]) < depth_threshold    # |Δx|
        inplane_ok = (delta[:, 1]**2 + delta[:, 2]**2).sqrt() < inplane_threshold
        return depth_ok & inplane_ok

    return _hand_ok(left_pos, target_L) & _hand_ok(right_pos, target_R)


def joint_vel_runaway(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    grace_steps: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate only on physically-impossible joint velocity, after a grace window.

    Args:
        max_velocity: catastrophic threshold (rad/s). Set well above any feasible
            G1 arm velocity (~50) — catches true runaway / divergence only.
        grace_steps: policy steps after reset during which this guard is
            suppressed (lets the reset-grip impulse + contact settle).
        asset_cfg: target articulation + joint subset (resolved by the manager).

    Returns:
        Bool [num_envs]: True where any selected joint exceeds ``max_velocity``
        AND the episode is past ``grace_steps``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    over = torch.any(torch.abs(vel) > max_velocity, dim=1)
    armed = env.episode_length_buf >= grace_steps
    return over & armed
