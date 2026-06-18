"""g(θ) pressure model — single source of the firmware-locked valve pressure
mapping and the per-env target-pressure buffer.

CONTEXT.md §g(θ): p_now = a·θ + b, clamp [15, 200] PSI. Firmware-locked, no DR;
the sim mirrors the ESP32 firmware exactly. Because these values are a global
invariant (never randomized, never overridden per-config), they live here once
instead of being threaded through every reward / observation / termination
signature and re-declared in every env-cfg term's params dict.

`base_cfg.py` re-exports the constants (``_G_THETA_A`` etc.) so configs that
reference them (curriculum anchors, configclass fields) keep working.

The target pressure ``p_des`` is per-env: ``reset_p_des_random`` writes it into
``env.p_des_buf`` each episode reset (a constant for fixed-target configs, a
sampled value for random-target configs). ``p_des(env)`` is the single read
path — there is no scalar/buffer fork.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

# --- Firmware-locked g(θ) constants — the single definition for the whole task ---
A: float = 4.527       # PSI/rad  — g(θ) slope
B: float = -27.66      # PSI      — g(θ) intercept
P_MIN: float = 15.0    # PSI
P_MAX: float = 200.0   # PSI
P_SPAN: float = 185.0  # PSI      — P_MAX − P_MIN
THETA_MIN: float = 9.42    # rad  (1.5 rev)
THETA_MAX: float = 50.27   # rad  (8 rev)
EPS_SIM: float = 1.85      # PSI  (~1% of span) — convergence tolerance

_THETA_MAX_FALLBACK: float = THETA_MAX  # nan_to_num posinf guard


def wheel_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel angle θ (rad) for every env. Shape: (num_envs,)."""
    valve: Articulation = env.scene["valve_rig"]
    theta = valve.data.joint_pos[:, 0]
    return torch.nan_to_num(theta, nan=0.0, posinf=_THETA_MAX_FALLBACK, neginf=0.0)


def wheel_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel angular velocity ω (rad/s). Shape: (num_envs,)."""
    valve: Articulation = env.scene["valve_rig"]
    omega = valve.data.joint_vel[:, 0]
    return torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)


def p_now(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current pressure p_now (PSI), clamped to [P_MIN, P_MAX]. Shape: (num_envs,).

    Matches the clamped signal fed back to the real ESP32 control loop.
    """
    return torch.clamp(A * wheel_angle(env) + B, P_MIN, P_MAX)


def theta_des(p_des: torch.Tensor | float) -> torch.Tensor | float:
    """Inverse g(θ): wheel angle that yields ``p_des`` (unclamped).

    Accepts a float or a per-env tensor; returns the same form.
    """
    return (p_des - B) / A


def p_des(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Per-env target pressure (PSI) from ``env.p_des_buf``.

    Returns ``None`` before the first reset has populated the buffer (init-time
    shape probe) — callers return zeros in that case.
    """
    return getattr(env, "p_des_buf", None)


def normalize(p: torch.Tensor | float) -> torch.Tensor | float:
    """Normalize a pressure (PSI) to [0, 1] via P_SPAN."""
    return p / P_SPAN
