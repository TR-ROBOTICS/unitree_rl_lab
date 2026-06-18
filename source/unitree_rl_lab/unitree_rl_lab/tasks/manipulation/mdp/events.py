"""Event terms for valve-turn task.

reset_joints_to_fixed_pose — force a joint subset to an explicit pose at reset.

Why this exists: Inspire-hand finger joints carry an authored USD angular drive
(target=0, k=20). IsaacLab's `hands` ImplicitActuatorCfg does not override that
drive target, and `init_state.joint_pos` curl values do not propagate to the
finger default position-target. Result: fingers reset open (~0) → no grip → no
hand↔rim coupling → wheel never turns (Stage 1 dead).

This event writes finger joint state AND position-target every reset, after
`reset_scene_to_default`, so the k=200 hands actuator then holds the grip.
CONTEXT.md §Curriculum Stage 1: "hands pre-placed on wheel".

reset_valve_base_position_random — uniform DR on valve root position per episode.
ADR 0004: must be active in both reach and turn policies with identical ranges.
g(θ) is NOT randomized — firmware-locked.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def reset_valve_base_position_random(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float],
    half_range_xyz: tuple[float, float, float],
) -> None:
    """Randomise valve root position uniformly within ±half_range_xyz around base_pos.

    Orientation (base_quat, wxyz) is fixed — no rotation DR.
    All velocities zeroed. Used as valve position DR in reach and turn envs.

    ADR 0004: same half_range_xyz must be used for both reach and turn policies.
    g(θ) is not randomized — firmware-locked.

    Args:
        asset_cfg:       SceneEntityCfg targeting "valve_rig".
        base_pos:        Centre of DR distribution (x, y, z) in world frame (m).
        base_quat:       Fixed orientation (w, x, y, z). Mirrored from init_state.
        half_range_xyz:  Per-axis half-extent of uniform distribution (m).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = asset._ALL_INDICES

    n = len(env_ids)
    device = asset.device

    # Sample per-axis offsets ∈ [−half, +half]
    offsets = torch.zeros(n, 3, device=device)
    for i, half in enumerate(half_range_xyz):
        offsets[:, i] = (torch.rand(n, device=device) * 2.0 - 1.0) * half

    base_pos_t = torch.tensor(base_pos, device=device, dtype=torch.float32)
    env_origins = env.scene.env_origins[env_ids]  # (n, 3) world offset per env
    pos = env_origins + base_pos_t.unsqueeze(0) + offsets  # (n, 3)

    base_quat_t = torch.tensor(base_quat, device=device, dtype=torch.float32)
    quat = base_quat_t.unsqueeze(0).expand(n, -1)  # (n, 4) wxyz

    # root_state: [pos(3), quat(4), linvel(3), angvel(3)] = 13 columns
    root_state = torch.zeros(n, 13, device=device, dtype=torch.float32)
    root_state[:, :3] = pos
    root_state[:, 3:7] = quat   # velocities stay 0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_data_to_sim()


def reset_valve_to_random_angle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    angle_min: float,
    angle_max: float,
) -> None:
    """Reset valve RevoluteJoint to a uniform-random angle in [angle_min, angle_max].

    Stage 2+: random θ_init covers [θ_min, θ_max] so policy learns to turn in
    both CW (decrease θ → decrease p) and CCW (increase θ → increase p) directions.
    valve_rig has exactly one joint (RevoluteJoint at index 0).

    Args:
        asset_cfg:  SceneEntityCfg targeting "valve_rig".
        angle_min:  lower bound (rad). Typically _THETA_MIN = 9.42 rad.
        angle_max:  upper bound (rad). Typically _THETA_MAX = 50.27 rad.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = asset._ALL_INDICES

    n = len(env_ids)
    angles = torch.rand(n, device=asset.device) * (angle_max - angle_min) + angle_min
    # valve_rig has one joint — index 0
    joint_ids = torch.tensor([0], device=asset.device, dtype=torch.long)
    pos = angles.unsqueeze(-1)          # (n, 1)
    vel = torch.zeros_like(pos)

    asset.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_data_to_sim()


def reset_joints_to_fixed_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_pose: dict[str, float],
) -> None:
    """Force a joint subset to an explicit pose (state + position-target).

    Args:
        asset_cfg:  target articulation (joint_names ignored — keys in
                    ``joint_pose`` drive the selection).
        joint_pose: ``{joint_name_regex: value_rad}``. Each key full-matched
                    against articulation joint names; all matches set to value.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = asset._ALL_INDICES

    names = asset.joint_names
    idx: list[int] = []
    val: list[float] = []
    for pat, v in joint_pose.items():
        for i, n in enumerate(names):
            if re.fullmatch(pat, n):
                idx.append(i)
                val.append(float(v))

    if not idx:
        return

    joint_ids = torch.tensor(idx, device=asset.device, dtype=torch.long)
    pos = torch.tensor(val, device=asset.device, dtype=torch.float32)
    pos = pos.unsqueeze(0).expand(len(env_ids), -1).contiguous()
    vel = torch.zeros_like(pos)

    # write physics state
    asset.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)
    # hold there: actuator (k=1000) drives toward this target until changed
    asset.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_data_to_sim()


def reset_p_des_random(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    p_min: float,
    p_max: float,
) -> None:
    """Sample a fresh p_des per env from Uniform[p_min, p_max] and store in env.p_des_buf.

    Stage 2+: policy must learn to converge to arbitrary target pressures.
    All obs/reward/termination *_random variants read env.p_des_buf instead of a
    scalar constant.

    Args:
        p_min: lower bound (PSI). Typically 15.0.
        p_max: upper bound (PSI). Typically 200.0.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Initialise buffer on first call
    if not hasattr(env, "p_des_buf") or env.p_des_buf.shape[0] != env.num_envs:
        env.p_des_buf = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    env.p_des_buf[env_ids] = (
        torch.rand(len(env_ids), device=env.device) * (p_max - p_min) + p_min
    )


def reset_arm_staged(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    dataset_path: str,
    fallback_pose: dict[str, float],
) -> None:
    """Arm init that switches from pre-grip to dataset when auto-curriculum Stage ≥ 3.

    Checks ``env._autocurr_use_dataset`` (set by ``turn_auto_curriculum_stage_easy``).
      False (default) → always use fallback_pose (pre-grip). Mirrors v0/v1/v2 init.
      True            → use dataset (same logic as reset_arm_from_dataset). Mirrors v3+.

    Curriculum fires BEFORE events in _reset_idx(), so the flag set by the curriculum
    function takes effect in the same reset cycle — no one-episode lag.

    Args:
        asset_cfg:     SceneEntityCfg for robot with arm joint_names pattern.
        dataset_path:  Absolute path to .npy dataset file.
        fallback_pose: ``{joint_name_regex: value_rad}`` pre-grip arm pose.
    """
    if getattr(env, "_autocurr_use_dataset", False):
        reset_arm_from_dataset(env, env_ids, asset_cfg, dataset_path, fallback_pose)
    else:
        reset_joints_to_fixed_pose(env, env_ids, asset_cfg, fallback_pose)


def reset_arm_from_dataset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    dataset_path: str,
    fallback_pose: dict[str, float],
) -> None:
    """Reset arm joints to a pose sampled from a reach-terminal-state dataset.

    Lazy-loads ``dataset_path`` on first call and caches on ``env._arm_dataset``.
    Falls back to ``fallback_pose`` (pre-grip arm pose dict) if the file does not
    exist yet — v3 training can start before the dataset is collected.

    Dataset format: numpy array of shape (N, num_arm_joints), float32, rad.
    Joint order must match the resolved ``asset_cfg.joint_ids`` order.

    Intended usage: EventTerm running AFTER ``reset_scene_to_default`` so that
    the arm overrides the USD init_state pose, same pattern as ``reset_finger_grip``.

    Args:
        asset_cfg:     SceneEntityCfg for robot with arm joint_names pattern.
                       joint_ids resolved by manager before call.
        dataset_path:  Absolute path to .npy dataset file.
        fallback_pose: ``{joint_name_regex: value_rad}`` — exact same format as
                       ``reset_joints_to_fixed_pose``. Used when dataset absent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = asset._ALL_INDICES

    n = len(env_ids)
    joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
    num_joints = len(joint_ids)

    # Lazy load + cache
    dataset: torch.Tensor | None = getattr(env, "_arm_dataset", None)
    if dataset is None:
        p = pathlib.Path(dataset_path)
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            dataset = torch.tensor(arr, dtype=torch.float32, device=env.device)
            env._arm_dataset = dataset

    if dataset is not None and dataset.shape[0] > 0 and dataset.shape[1] == num_joints:
        # Sample uniformly from collected reach terminal states
        row_ids = torch.randint(0, dataset.shape[0], (n,), device=env.device)
        pos = dataset[row_ids]                                  # (n, num_joints)
    else:
        # Fallback: pre-grip pose (same as ValveSceneCfg init_state for arm joints)
        jnames = [asset.joint_names[int(jid)] for jid in asset_cfg.joint_ids]
        vals: list[float] = []
        for name in jnames:
            matched = False
            for pat, v in fallback_pose.items():
                if re.fullmatch(pat, name):
                    vals.append(float(v))
                    matched = True
                    break
            if not matched:
                vals.append(0.0)
                raise RuntimeError(f"Joint '{name}' not found in dataset or fallback_pose; defaulting to 0.0")

        pos_t = torch.tensor(vals, dtype=torch.float32, device=env.device)
        pos = pos_t.unsqueeze(0).expand(n, -1).contiguous()

    vel = torch.zeros_like(pos)
    asset.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_data_to_sim()


def reset_vision_zoh_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    vision_hz_range: tuple[float, float],
    latency_steps: int,
) -> None:
    """Reset per-env ZOH vision state at episode start.

    Samples a per-env vision update interval (in policy steps) from
    ``vision_hz_range`` and reinitialises the latency pipe with the current
    ground-truth p_now.  Must run AFTER ``reset_valve_angle`` so that the seeded
    pressure value reflects the new θ_init.

    Triggered as a ``mode="reset"`` EventTerm when VisionDRCfg.enabled=True.
    When disabled, this event is absent — all existing paths are byte-identical.

    Args:
        vision_hz_range:  (hz_lo, hz_hi) — per-env vision update rate sampled
                          uniformly (Hz).  Policy runs at 50 Hz; interval =
                          round(50 / sampled_hz), clamped to [1, 50].
                          Use (hz_lo, hz_lo) for a fixed rate.
        latency_steps:    Latency pipe depth (policy steps).  Must match the value
                          passed to valve_pressure_now_zoh obs term.
    """
    from . import pressure as _pressure
    from .observations import _vzoh_lazy_init

    n = env.num_envs
    dev = env.device
    pipe_depth = max(1, latency_steps + 1)

    if env_ids is None:
        env_ids = torch.arange(n, device=dev)

    # Ensure buffers exist (may be first call before any lazy init in obs)
    if not hasattr(env, "_vzoh_held") or env._vzoh_held.shape[0] != n:
        _vzoh_lazy_init(env, n, dev, pipe_depth)

    # Sample per-env vision Hz then convert to policy-step intervals
    hz_lo, hz_hi = vision_hz_range
    if hz_lo == hz_hi:
        hz = torch.full((len(env_ids),), hz_lo, dtype=torch.float32, device=dev)
    else:
        hz = torch.rand(len(env_ids), device=dev) * (hz_hi - hz_lo) + hz_lo

    # Policy runs at 50 Hz; clamp interval to [1, 50]
    _POLICY_HZ: float = 50.0
    intervals = torch.clamp(torch.round(_POLICY_HZ / hz).to(torch.int32), 1, 50)
    env._vzoh_interval[env_ids] = intervals

    # Seed pipe with fresh p_now for the new episode (after valve angle reset)
    p_now = _pressure.p_now(env)  # (num_envs,)
    env._vzoh_pipe[env_ids] = p_now[env_ids].unsqueeze(-1).expand(len(env_ids), pipe_depth).contiguous()

    # Reset counter so the first ZOH update fires after one full interval
    env._vzoh_counter[env_ids] = 0


def reset_arm_pregrasp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pregrasp_pose: dict[str, float],
    enabled: bool = True,
) -> None:
    """Reset arm joints to the canonical pre-grasp pose near the valve rim.

    v7 feature: when ``enabled=True`` (default), overrides whatever arm pose the
    USD init_state or ``reset_scene_to_default`` would produce and writes the
    tuned PREGRASP_ARM_POSE to both joint state and position-target. This shrinks
    the exploration problem — policy starts with hands already in a ready-to-grip
    configuration at the valve rim.

    When ``enabled=False`` the function is a no-op; the USD init_state pose is
    retained (same behaviour as v0–v4 non-curriculum arm reset). Toggle via the
    EventTerm params dict to ablate the pre-grasp init hypothesis.

    Pre-grasp joint targets (tuned IsaacSim 2026-05-15, wheel VERTICAL, spins world-X):
        shoulder_pitch: −0.7610 rad  (both sides, symmetric)
        shoulder_roll:  ±0.1937 rad  (outward, left positive / right negative)
        shoulder_yaw:   ∓0.1239/+0.1257 rad  (slight external rotation)
        elbow:          +0.4869 / +0.5236 rad  (flexed)
        wrist_roll:     +0.3787 / −0.4712 rad  (pronation, asymmetric)
        wrist_pitch/yaw: 0.0 rad  (neutral)

    The canonical PREGRASP_ARM_POSE dict lives in valve/presets.py. EventTerm
    params should pass it via ``pregrasp_pose=PREGRASP_ARM_POSE``.

    Args:
        asset_cfg:     SceneEntityCfg targeting robot, arm joint_names pattern.
        pregrasp_pose: ``{joint_name_regex: value_rad}`` — exact same format as
                       ``reset_joints_to_fixed_pose``.  Use PREGRASP_ARM_POSE from
                       valve/presets.py.
        enabled:       Master toggle.  False = no-op (USD init_state kept).
                       Set to False in the EventTerm params dict to ablate.
    """
    if not enabled:
        return
    reset_joints_to_fixed_pose(env, env_ids, asset_cfg, pregrasp_pose)


def reset_arm_mixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    dataset_path: str,
    fallback_pose: dict[str, float],
) -> None:
    """Curriculum-mixed arm init: per-env Bernoulli draw between pre-grip and dataset.

    Reads the canonical mixing fraction ``env._curr_dataset_pct`` (float 0.0→1.0),
    which both ``turn_smooth_curriculum_v5`` (step-function) and
    ``turn_pd_curriculum_v6`` (PD) keep synced from their internal stage state.
    Each env in env_ids independently draws Bernoulli(dataset_pct):
      True  → sample from reach dataset (same as reset_arm_from_dataset)
      False → use fallback_pose (pre-grip pose)

    When dataset_pct=0.0: all pre-grip — identical to v2 init.
    When dataset_pct=1.0: all dataset — identical to v3+ init.
    """
    dataset_pct: float = getattr(env, "_curr_dataset_pct", 0.0)

    if env_ids is None:
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids = asset._ALL_INDICES

    if dataset_pct <= 0.0:
        reset_joints_to_fixed_pose(env, env_ids, asset_cfg, fallback_pose)
        return
    if dataset_pct >= 1.0:
        reset_arm_from_dataset(env, env_ids, asset_cfg, dataset_path, fallback_pose)
        return

    # Split env_ids by Bernoulli draw
    mask = torch.rand(len(env_ids), device=env.device) < dataset_pct
    ids_dataset = env_ids[mask]
    ids_fallback = env_ids[~mask]

    if len(ids_dataset) > 0:
        reset_arm_from_dataset(env, ids_dataset, asset_cfg, dataset_path, fallback_pose)
    if len(ids_fallback) > 0:
        reset_joints_to_fixed_pose(env, ids_fallback, asset_cfg, fallback_pose)
