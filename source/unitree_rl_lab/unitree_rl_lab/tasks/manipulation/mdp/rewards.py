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
from isaaclab.utils.math import quat_apply

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


def pressure_progress(
    env: ManagerBasedRLEnv,
    p_des: float,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
    p_span: float,
) -> torch.Tensor:
    """Potential-based progress on θ-distance to θ_des. Scale: ·a/p_span.

    Primary Stage-1 signal. Uses ONLY θ (firmware-locked, sign-safe) — NO
    joint_vel (valve spawn rot vs joint axis makes joint_vel sign-inverted).

    On θ-distance, NOT clamped p_now: in the pressure clamp zone (θ<θ_min →
    p_now=p_min) |p_err| is constant w.r.t. θ → the p-based potential is FLAT
    (zero gradient) exactly where the episode starts (θ_min=9.42) → no signal
    to discover θ must rise (Run 13 → pressure_progress=0.0000, frozen).
    θ-distance |θ_des−θ| is monotone everywhere → gradient in the clamp zone
    too, pulling θ from θ_min → θ_des=(p_des−b)/a.

    Telescoping: Σ r_t = (|θ_des−θ_0|−|θ_des−θ_T|)·a/p_span — net progress,
    path-independent. Closed wobble sums to 0 → jitter-proof (signed, no clamp).
    Reset-safe: 0 on the first step of each episode.
    """
    theta = _get_wheel_angle(env)
    theta_des = (p_des - pressure_b) / pressure_a
    # PSI-equivalent θ error, but UNCLAMPED so gradient survives the p floor.
    err = torch.abs(theta_des - theta) * pressure_a

    prev = getattr(env, "_valve_prev_abs_err", None)
    if prev is None or prev.shape != err.shape:
        prev = err.clone()
        env._valve_prev_abs_err = prev

    fresh = env.episode_length_buf <= 1  # just reset → no progress yet
    progress = torch.where(
        fresh, torch.zeros_like(err), (prev - err) / p_span
    )
    env._valve_prev_abs_err = err.detach().clone()
    return torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)


def pressure_error_random(
    env: ManagerBasedRLEnv,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
    p_span: float,
) -> torch.Tensor:
    """pressure_error using per-env p_des from env.p_des_buf."""
    p_des = getattr(env, "p_des_buf", None)
    if p_des is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    theta = _get_wheel_angle(env)
    p_now = torch.clamp(pressure_a * theta + pressure_b, p_min, p_max)
    return -torch.abs(p_now - p_des) / p_span


def pressure_progress_random(
    env: ManagerBasedRLEnv,
    pressure_a: float,
    pressure_b: float,
    p_min: float,
    p_max: float,
    p_span: float,
) -> torch.Tensor:
    """pressure_progress using per-env p_des from env.p_des_buf.

    Same θ-distance potential as pressure_progress — telescoping, jitter-proof,
    reset-safe. Each env tracks its own θ_des = (p_des_buf - b) / a.
    """
    p_des = getattr(env, "p_des_buf", None)
    if p_des is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    theta = _get_wheel_angle(env)
    theta_des = (p_des - pressure_b) / pressure_a
    err = torch.abs(theta_des - theta) * pressure_a

    prev = getattr(env, "_valve_prev_abs_err", None)
    if prev is None or prev.shape != err.shape:
        prev = err.clone()
        env._valve_prev_abs_err = prev

    fresh = env.episode_length_buf <= 1
    progress = torch.where(fresh, torch.zeros_like(err), (prev - err) / p_span)
    env._valve_prev_abs_err = err.detach().clone()
    return torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)


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
    # Clamp SYMMETRIC [-1.5, 1.5] rad/s (Run 11). Was [0, 1.5]: one-sided clamp
    # rectified contact jitter — backward half-cycle clamped to 0, not penalised
    # → vibration farmed reward with zero net rotation (wheel_vel 0.76 while
    # pressure_error stayed flat -0.46, Run 10). Symmetric clamp penalises
    # backward motion equally → Σ(ω·sign·dt) = sign·Δθ = net displacement
    # (telescoping; jitter cancels exactly). Identical to a potential-based
    # progress reward, minus one term.
    return torch.clamp(omega * error_sign, min=-1.5, max=1.5)


def rim_distance_reward(
    env: ManagerBasedRLEnv,
    valve_hub_cfg: SceneEntityCfg,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    wheel_radius: float,
    wheel_normal_world: tuple[float, float, float],
    sigma: float,
    mode: str = "max",
    plane_offset: float = 0.0,
) -> torch.Tensor:
    """Dense reward: exp(−aggregate_dist_to_rim / σ) ∈ (0, 1].

    Replaces contact-loss termination (Stage 2+). Penalises slap-brake by
    requiring continuous hand proximity to the rim — a hand far from the rim
    at any step gets low reward regardless of pressure state.

    Per-hand min over body_ids (≥1 bodies per hand), then aggregate L+R via
    `mode`, then apply exp(−·/σ). Bimanual: aggregation forces both hands
    to engage instead of single-hand exploit (Run 22 bug — `mode="min"` over
    both hands' bodies let policy satisfy reward with one hand).

    Closed-form distance from point p to circle C (centre c, radius r, plane
    normal n̂ — see docs/distance_point_circunference.md):
        d_plane  = (p − c) · n̂
        d_radial = ‖p_proj − c‖   where p_proj = p − d_plane·n̂
        dist     = √(d_plane² + (d_radial − r)²)

    Args:
        valve_hub_cfg:      SceneEntityCfg("valve_rig", body_names=["mesh_50_AL_250_B7_8_A_stl"]).
                            Handwheel spinning body (body1 of RevoluteJoint). Symmetric →
                            COM on rotation axis → body_pos_w = wheel centre.
                            Confirmed Script Editor 2026-05-20.
        left_hand_cfg:      SceneEntityCfg("robot", body_names=[...left bodies...]).
        right_hand_cfg:     SceneEntityCfg("robot", body_names=[...right bodies...]).
        wheel_radius:       Handwheel rim radius (m). Confirmed 0.10 m via vertex
                            scan (Script Editor 2026-05-21): max XY radius of mesh
                            verts from hub COM = 0.1000 m. Prior value 0.20 was 2×
                            too big → policy converged to phantom circle 10cm beyond
                            real rim → no contact → Runs 20-23 stuck in static-grip
                            local min outside the actual wheel.
        wheel_normal_world: Unit normal to wheel plane in world frame.
                            Computed from RevoluteJoint local axis Z transformed by
                            hub world rotation → world axis ≈ (0, 0, -1). Wheel
                            rotates in XY plane (horizontal). Confirmed 2026-05-21.
        sigma:              Distance shaping bandwidth (m). At dist=sigma, reward
                            = e^{-1} ≈ 0.37.
        mode:               L/R aggregation of per-hand min distances:
                              "min":  min(d_L, d_R) — closest hand wins (single-hand exploit).
                              "max":  max(d_L, d_R) — worst hand caps reward (both must engage). [default]
                              "sum":  d_L + d_R     — harsh, σ effectively halved.
                              "mean": 0.5·(d_L+d_R) — balanced.
        plane_offset:       Signed offset (m) from hub body_pos_w along wheel normal
                            to wheel mid-plane. The hub rigid-body COM does not in
                            general coincide with the rim mid-plane; vertex scan
                            (Script Editor 2026-05-21) shows wheel verts z ∈
                            [hub_z+0.0018, hub_z+0.0418], mid ≈ hub_z+0.022. Set
                            plane_offset=0.022 to align reward target with real rim.

    Returns:
        (num_envs,) ∈ (0, 1].
    """
    valve: Articulation = env.scene[valve_hub_cfg.name]
    robot_L: Articulation = env.scene[left_hand_cfg.name]
    robot_R: Articulation = env.scene[right_hand_cfg.name]

    # Hub COM in world — (num_envs, 3)
    c_com = valve.data.body_pos_w[:, valve_hub_cfg.body_ids[0], :]

    # Wheel plane unit normal — constant world-frame vector
    n_hat = torch.tensor(wheel_normal_world, device=env.device, dtype=torch.float32)
    n_hat = n_hat / (n_hat.norm() + 1e-8)  # safety normalise

    # Rim-plane centre = hub COM offset along normal by plane_offset.
    # Corrects for asymmetric mesh where COM ≠ rim mid-plane.
    c = c_com + plane_offset * n_hat  # broadcast (N,3) + (3,) → (N,3)
    c_exp = c.unsqueeze(1)  # (num_envs, 1, 3)

    def _per_hand_min(robot: Articulation, body_ids) -> torch.Tensor:
        """Per-hand min distance to rim over specified body_ids."""
        p = robot.data.body_pos_w[:, body_ids, :]              # (N, B, 3)
        diff    = p - c_exp                                     # (N, B, 3)
        d_plane = (diff * n_hat).sum(-1)                        # (N, B)
        p_proj  = p - d_plane.unsqueeze(-1) * n_hat             # (N, B, 3)
        d_rad   = (p_proj - c_exp).norm(dim=-1)                 # (N, B)
        dist    = (d_plane**2 + (d_rad - wheel_radius)**2).sqrt()  # (N, B)
        return dist.min(dim=1).values                           # (N,)

    d_L = _per_hand_min(robot_L, left_hand_cfg.body_ids)
    d_R = _per_hand_min(robot_R, right_hand_cfg.body_ids)

    if mode == "min":
        d = torch.minimum(d_L, d_R)
    elif mode == "max":
        d = torch.maximum(d_L, d_R)
    elif mode == "sum":
        d = d_L + d_R
    elif mode == "mean":
        d = 0.5 * (d_L + d_R)
    else:
        raise ValueError(f"rim_distance_reward: unknown mode={mode!r} (expected min/max/sum/mean)")

    d = torch.nan_to_num(d, nan=1.0, posinf=1.0)
    return torch.exp(-d / sigma)


def reach_progress_reward(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    valve_hub_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
    distance_scale: float = 0.20,
) -> torch.Tensor:
    """Potential-based reach progress: r_t = (d_{t-1} − d_t) / distance_scale.

    Telescopes to total distance closed over episode — path-independent.
    Standing still → r=0 → advantage=0 → no zero-motion local min.
    Moving toward target → r>0. Moving away → r<0. Jitter cancels.

    Same design as pressure_progress in the turn task.
    Uses max(d_L, d_R) — forces both hands to close in.
    """
    valve: Articulation = env.scene[valve_hub_cfg.name]
    robot_L: Articulation = env.scene[left_hand_cfg.name]
    robot_R: Articulation = env.scene[right_hand_cfg.name]

    hub_pos = valve.data.body_pos_w[:, valve_hub_cfg.body_ids[0], :]
    off_L = torch.tensor(left_target_offset, device=env.device, dtype=torch.float32)
    off_R = torch.tensor(right_target_offset, device=env.device, dtype=torch.float32)
    target_L = hub_pos + off_L
    target_R = hub_pos + off_R

    left_pos  = robot_L.data.body_pos_w[:, left_hand_cfg.body_ids[0], :]
    right_pos = robot_R.data.body_pos_w[:, right_hand_cfg.body_ids[0], :]

    d_L = (left_pos  - target_L).norm(dim=-1)
    d_R = (right_pos - target_R).norm(dim=-1)
    d_curr = d_L + d_R  # sum: both hands get gradient simultaneously (max hid the closer hand)

    prev = getattr(env, "_reach_prev_d", None)
    if prev is None or prev.shape != d_curr.shape:
        prev = d_curr.clone()
        env._reach_prev_d = prev

    fresh = env.episode_length_buf <= 1
    progress = torch.where(fresh, torch.zeros_like(d_curr), (prev - d_curr) / distance_scale)
    env._reach_prev_d = d_curr.detach().clone()
    return torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)


def reach_approach_reward(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    valve_hub_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
    sigma: float,
    mode: str = "max",
) -> torch.Tensor:
    """Dense reach reward: exp(−d_to_target / σ) ∈ (0, 1].

    Unlike rim_distance_reward (which finds closest point on the full ring),
    this rewards proximity to SPECIFIC target points on the rim — left and right
    spokes at 90°/270° — so the policy learns to place hands at the grip
    positions rather than at the front of the ring (which is the closest-ring-
    point from the zero arm pose).

    mode="max": worst-hand caps reward → forces both hands to engage.
    """
    valve: Articulation = env.scene[valve_hub_cfg.name]
    robot_L: Articulation = env.scene[left_hand_cfg.name]
    robot_R: Articulation = env.scene[right_hand_cfg.name]

    hub_pos = valve.data.body_pos_w[:, valve_hub_cfg.body_ids[0], :]  # (N, 3)
    off_L = torch.tensor(left_target_offset, device=env.device, dtype=torch.float32)
    off_R = torch.tensor(right_target_offset, device=env.device, dtype=torch.float32)
    target_L = hub_pos + off_L   # (N, 3)
    target_R = hub_pos + off_R

    left_pos  = robot_L.data.body_pos_w[:, left_hand_cfg.body_ids[0], :]
    right_pos = robot_R.data.body_pos_w[:, right_hand_cfg.body_ids[0], :]

    d_L = (left_pos  - target_L).norm(dim=-1)  # (N,)
    d_R = (right_pos - target_R).norm(dim=-1)

    if mode == "mean":
        d = 0.5 * (d_L + d_R)
    else:  # "max" — default, forces both hands
        d = torch.maximum(d_L, d_R)

    d = torch.nan_to_num(d, nan=1.0, posinf=1.0)
    return torch.exp(-d / sigma)


def reach_hand_distance(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    valve_hub_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Dense reach reward: negative sum of L/R hand distances to FK target points.

    Targets are fixed offsets from the valve hub COM in world frame.
    Valve position is randomized per episode — targets follow the hub.

    Returns:
        (num_envs,) ∈ (−∞, 0]. 0 when both hands are exactly at target.
    """
    valve: Articulation = env.scene[valve_hub_cfg.name]
    robot_L: Articulation = env.scene[left_hand_cfg.name]
    robot_R: Articulation = env.scene[right_hand_cfg.name]

    hub_pos = valve.data.body_pos_w[:, valve_hub_cfg.body_ids[0], :]  # (N, 3)

    off_L = torch.tensor(left_target_offset, device=env.device, dtype=torch.float32)
    off_R = torch.tensor(right_target_offset, device=env.device, dtype=torch.float32)
    target_L = hub_pos + off_L   # (N, 3)
    target_R = hub_pos + off_R

    left_pos  = robot_L.data.body_pos_w[:, left_hand_cfg.body_ids[0], :]   # (N, 3)
    right_pos = robot_R.data.body_pos_w[:, right_hand_cfg.body_ids[0], :]  # (N, 3)

    d_left  = (left_pos  - target_L).norm(dim=-1)  # (N,)
    d_right = (right_pos - target_R).norm(dim=-1)

    return -(d_left + d_right)


def reach_handoff_bonus(
    env: ManagerBasedRLEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    valve_hub_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
    depth_threshold: float,
    inplane_threshold: float,
    bonus: float = 5.0,
) -> torch.Tensor:
    """Sparse bonus when handoff condition is met (both hands at target within thresholds).

    Mirrors reach_handoff_condition logic.  Returns `bonus` where condition True, 0 elsewhere.

    Args:
        depth_threshold:   |Δx| threshold (m) — perpendicular to wheel face.
        inplane_threshold: √(Δy²+Δz²) threshold (m) — within wheel plane.
        bonus:             Reward magnitude when condition satisfied.

    Returns:
        (num_envs,) scalar — `bonus` or 0.
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
        delta = hand_pos - target                                # (N, 3)
        depth_ok   = torch.abs(delta[:, 0]) < depth_threshold   # |Δx|
        inplane_ok = (delta[:, 1]**2 + delta[:, 2]**2).sqrt() < inplane_threshold
        return depth_ok & inplane_ok

    condition = _hand_ok(left_pos, target_L) & _hand_ok(right_pos, target_R)
    return torch.where(condition, torch.full_like(condition, bonus, dtype=torch.float32),
                       torch.zeros(env.num_envs, device=env.device, dtype=torch.float32))


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


# ---------------------------------------------------------------------------
# v5 contact rewards
# ---------------------------------------------------------------------------

def bilateral_contact(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "left_palm_sensor",
    right_sensor_name: str = "right_palm_sensor",
    f_max: float = 50.0,
    palm_normal_left: tuple[float, float, float] = (0.0, 0.0, 1.0),
    palm_normal_right: tuple[float, float, float] = (0.0, 0.0, -1.0),
    left_body_name: str = "left_hand_base_link",
    right_body_name: str = "right_hand_base_link",
    robot_name: str = "robot",
    valve_name: str = "valve_rig",
    use_palm_filter: bool = False,
) -> torch.Tensor:
    """Multiplicative bilateral contact reward ∈ [0, 1].

    Returns clamp(F_L/f_max) * clamp(F_R/f_max) optionally weighted by palm alignment.

    When use_palm_filter=True, each hand's contribution is multiplied by
    relu(palm_normal_world · hand_to_valve_unit) so that only palm-side contact
    is rewarded. Palm normals are specified in hand_base_link local frame;
    from the Inspire hand MJCF: left thumb at +Z → palm_normal_left=(0,0,1),
    right thumb at -Z → palm_normal_right=(0,0,-1). Verify sign in play mode.

    Args:
        palm_normal_left:  Palm outward normal in left_hand_base_link local frame.
        palm_normal_right: Palm outward normal in right_hand_base_link local frame.
        use_palm_filter:   Enable palm-normal contact filter. Default False.
    """
    left_sensor  = env.scene[left_sensor_name]
    right_sensor = env.scene[right_sensor_name]

    f_l = torch.norm(torch.nan_to_num(left_sensor.data.net_forces_w[:, 0, :],  nan=0.0), dim=-1)
    f_r = torch.norm(torch.nan_to_num(right_sensor.data.net_forces_w[:, 0, :], nan=0.0), dim=-1)

    f_l_norm = torch.clamp(f_l / f_max, 0.0, 1.0)
    f_r_norm = torch.clamp(f_r / f_max, 0.0, 1.0)

    if use_palm_filter:
        robot  = env.scene[robot_name]
        valve  = env.scene[valve_name]
        device = env.device

        # Cache body indices once
        if not hasattr(env, "_palm_body_idx_l"):
            names = robot.data.body_names
            env._palm_body_idx_l = names.index(left_body_name)
            env._palm_body_idx_r = names.index(right_body_name)

        valve_pos = valve.data.root_pos_w  # (N, 3)

        def _palm_align(body_idx: int, n_local: tuple[float, float, float]) -> torch.Tensor:
            hand_pos  = robot.data.body_pos_w[:, body_idx, :]   # (N, 3)
            hand_quat = robot.data.body_quat_w[:, body_idx, :]  # (N, 4) wxyz
            n_t = torch.tensor(n_local, device=device, dtype=torch.float32).expand(hand_pos.shape[0], 3)
            n_world = quat_apply(hand_quat, n_t)                # (N, 3)
            to_valve = valve_pos - hand_pos
            to_valve = torch.nn.functional.normalize(to_valve, dim=-1)
            return torch.relu((n_world * to_valve).sum(dim=-1))  # (N,) ∈ [0, 1]

        align_l = _palm_align(env._palm_body_idx_l, palm_normal_left)
        align_r = _palm_align(env._palm_body_idx_r, palm_normal_right)
        f_l_norm = f_l_norm * align_l
        f_r_norm = f_r_norm * align_r

    return f_l_norm * f_r_norm


def contact_force_jerk(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "left_palm_sensor",
    right_sensor_name: str = "right_palm_sensor",
) -> torch.Tensor:
    """Penalty for abrupt contact force changes (slapping).

    Returns −((||F_L_t|| − ||F_L_{t-1}||)² + (||F_R_t|| − ||F_R_{t-1}||)²).

    Uses net_forces_w_history (history_length=2 required on ContactSensorCfg):
      history[:, 0, 0, :] = current step
      history[:, 1, 0, :] = previous step

    Joint-space smoothness (action_rate_l2) cannot detect contact impact events;
    this directly penalises the force spike that occurs at impact.
    """
    left_sensor = env.scene[left_sensor_name]
    right_sensor = env.scene[right_sensor_name]

    # net_forces_w_history: (num_envs, T, 1, 3) — T=2, 1 body
    def _jerk(sensor) -> torch.Tensor:
        hist = torch.nan_to_num(sensor.data.net_forces_w_history, nan=0.0)
        f_now = torch.norm(hist[:, 0, 0, :], dim=-1)
        f_prev = torch.norm(hist[:, 1, 0, :], dim=-1)
        return (f_now - f_prev) ** 2

    return _jerk(left_sensor) + _jerk(right_sensor)
