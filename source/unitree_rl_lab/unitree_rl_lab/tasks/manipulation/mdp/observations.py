"""Task-specific observation functions for valve-turn task.

All pressure terms derived from g(θ) — firmware-locked, no DR.
CONTEXT.md §g(θ): p_now = a·θ + b, clamp [15, 200] PSI.

Vision ZOH model (sparse-feedback DR, E3 compatibility sweep):
  valve_pressure_now_zoh — replaces valve_pressure_now when VisionDRCfg.enabled=True.
  Implements ZOH at vision Hz + additive Gaussian noise + latency + optional
  quantization per env.  Default OFF — opt-in via TurnSpec.vision_dr.

  Per-env state tensors live on env (lazy-initialized, re-initialized each reset by
  reset_vision_zoh_state event):
    env._vzoh_held     (num_envs,)  — PSI value currently held (ZOH output, pre-latency)
    env._vzoh_pipe     (num_envs, L) — latency pipe; index 0 = oldest (delivered) reading
    env._vzoh_counter  (num_envs,)  — steps since last vision update (int32)
    env._vzoh_interval (num_envs,)  — update interval in policy steps (int32, per-env DR)
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import pressure


def valve_pressure_now(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current pressure p_now, normalized to [0, 1] via P_SPAN.

    From the firmware-locked g(θ) model (clamped to [P_MIN, P_MAX]) — matches
    the clamped signal fed back to the real ESP32 control loop.

    Shape: (num_envs, 1).
    """
    return pressure.normalize(pressure.p_now(env)).unsqueeze(-1)


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


def valve_pressure_des(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target pressure p_des, normalized to [0, 1] via P_SPAN.

    Per-env: reads env.p_des_buf, written by reset_p_des_random each episode (a
    constant for fixed-target configs, a sampled value for random-target ones).
    Returns zeros before the first reset (init-time shape probe).

    Shape: (num_envs, 1).
    """
    p_des = pressure.p_des(env)
    if p_des is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return pressure.normalize(p_des).unsqueeze(-1).clamp(0.0, 1.0)


def valve_pressure_now_zoh(
    env: ManagerBasedRLEnv,
    noise_psi: float = 0.0,
    latency_steps: int = 0,
    quant_psi: float = 0.0,
) -> torch.Tensor:
    """Sparse-feedback vision model for p_now obs (E3 compatibility sweep).

    Implements ZOH at per-env vision Hz + additive Gaussian noise + latency pipe
    + optional quantization.  Enabled only when this function is wired as the obs
    term (via VisionDRCfg.enabled=True in TurnSpec).  Default obs term is the
    unmodified valve_pressure_now — callers with VisionDRCfg disabled never reach
    this function.

    State tensors are lazy-initialized here and re-set at episode reset by
    ``reset_vision_zoh_state`` event.  The update interval per env (in policy steps)
    is stored in ``env._vzoh_interval`` and set by ``reset_vision_zoh_state``.

    ZOH update cadence:
        env._vzoh_counter[i] counts policy steps since the last fresh read for env i.
        When counter[i] >= interval[i], a fresh read is taken (p_now + Gaussian
        noise + optional quantization), pushed into the latency pipe, and counter
        reset to 0.  The delivered value is the oldest entry in the pipe
        (``_vzoh_pipe[:, 0]``), giving a ZOH lag of latency_steps policy steps.

    Args:
        noise_psi:     σ of additive Gaussian noise on each fresh read (PSI).
                       0.0 = no noise.
        latency_steps: number of policy steps the reading lags behind the fresh
                       sample.  0 = no latency (pipe depth 1, immediate delivery).
                       Must match the value passed to reset_vision_zoh_state.
        quant_psi:     quantization step (PSI).  0.0 = no quantization.

    Shape: (num_envs, 1).
    """
    n = env.num_envs
    dev = env.device
    pipe_depth = max(1, latency_steps + 1)

    # --- Lazy init (first call before any reset, e.g. obs-space shape probe) ---
    if not hasattr(env, "_vzoh_held") or env._vzoh_held.shape[0] != n:
        _vzoh_lazy_init(env, n, dev, pipe_depth)

    # --- Tick step counter ---
    env._vzoh_counter += 1  # (num_envs,) int32

    # --- Envs whose ZOH interval has elapsed → take a fresh reading ---
    update_mask = env._vzoh_counter >= env._vzoh_interval  # (num_envs,) bool

    if update_mask.any():
        # Ground-truth p_now from g(θ) — firmware-locked, no DR on g itself.
        p_fresh = pressure.p_now(env)  # (num_envs,) PSI

        # Additive Gaussian noise on fresh read
        if noise_psi > 0.0:
            p_fresh = p_fresh + torch.randn(n, device=dev, dtype=torch.float32) * noise_psi
            p_fresh = torch.clamp(p_fresh, pressure.P_MIN, pressure.P_MAX)

        # Optional quantization
        if quant_psi > 0.0:
            p_fresh = torch.round(p_fresh / quant_psi) * quant_psi
            p_fresh = torch.clamp(p_fresh, pressure.P_MIN, pressure.P_MAX)

        # Shift latency pipe: drop oldest ([:, 0]), push fresh to end ([:, -1])
        if pipe_depth > 1:
            env._vzoh_pipe[:, :-1] = env._vzoh_pipe[:, 1:].clone()
        # Write fresh value for updating envs only
        env._vzoh_pipe[update_mask, -1] = p_fresh[update_mask]

        # Reset counter for updated envs
        env._vzoh_counter[update_mask] = 0

    # Delivered value = oldest entry in pipe (latency_steps behind fresh read)
    delivered = env._vzoh_pipe[:, 0]  # (num_envs,) PSI

    return pressure.normalize(delivered).unsqueeze(-1).clamp(0.0, 1.0)


def _vzoh_lazy_init(env: ManagerBasedRLEnv, n: int, dev: str, pipe_depth: int) -> None:
    """Initialize ZOH state tensors.  Called on first obs query and at episode reset."""
    # Seed the pipe with current ground-truth p_now so the first policy step is
    # not a spurious zero even before any reset has fired.
    p_seed = pressure.p_now(env)  # (num_envs,) — may return zeros at shape-probe time
    env._vzoh_held = p_seed.clone()
    env._vzoh_pipe = p_seed.unsqueeze(-1).expand(n, pipe_depth).clone().contiguous()
    env._vzoh_counter = torch.zeros(n, dtype=torch.int32, device=dev)
    # Default interval: 5 steps (= 10 Hz at 50 Hz policy) — will be overwritten by
    # reset_vision_zoh_state on first real reset.
    env._vzoh_interval = torch.full((n,), 5, dtype=torch.int32, device=dev)
