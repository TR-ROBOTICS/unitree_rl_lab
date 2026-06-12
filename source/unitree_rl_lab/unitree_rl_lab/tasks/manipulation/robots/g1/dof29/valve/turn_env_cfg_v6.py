"""Valve-turn task config — G1 29-DoF + Inspire hands, v6.

v6 = PD curriculum with decoupled θ/p expansion axes + dataset mixing.

Key changes vs v5:
  1. Sequential stage design (ADR 0008):
       Stage 0: θ_init expands only. p fixed at p_mid=107 PSI.
       Stage 1: p_des expands only. θ at full range.
       Stage 2: dataset arm-init mixing 0%→100%.
       Stage 3: terminal (fully open, 100% dataset).
  2. PD controller replaces step-function triggers:
       EMA_SR drives θ/p/mix boundaries continuously.
       δ<0 → automatic contraction (self-correcting on SR collapse).
       Kp=2.0, Kd=0.5 — matched to span time-constants in ADR.
  3. No bilateral_contact reward — v5 run log proves any positive weight
     creates touch-only local optimum regardless of scale.
  4. Train from scratch (fresh init) — CCD USDs must be active from iter 0.

See docs/adr/0008-v6-pd-curriculum-decoupled-axes.md for full design rationale.

g(θ) firmware-locked — no DR. Valve pos DR: disabled.
"""

from __future__ import annotations

import pathlib

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import _THETA_MIN, _THETA_MAX, _P_MIN, _P_MAX
from .turn_env_cfg_v2 import (
    EventCfgV2,
    ObservationsCfgV2,
    RewardsCfgV2,
    TerminationsCfgV2,
    ValveTurnEnvCfgV2,
)

# v6 curriculum constants — match ADR 0008
_THETA_MID: float = 29.75   # (107 − (−27.66)) / 4.527 ≈ midpoint of [9.42, 50.27]
_THETA_STEP: float = 2.04   # 5% of span (50.27 − 9.42 = 40.85 rad)
_P_MID: float = 107.0

_DATASET_PATH: str = str(
    pathlib.Path(__file__).parents[6] / "datasets" / "reach_arm_positions.npy"
)

_PREGRASP_ARM_POSE: dict[str, float] = {
    "left_shoulder_pitch_joint":  -0.7610,
    "right_shoulder_pitch_joint": -0.7610,
    "left_shoulder_roll_joint":    0.1937,
    "right_shoulder_roll_joint":  -0.1937,
    "left_shoulder_yaw_joint":    -0.1239,
    "right_shoulder_yaw_joint":    0.1257,
    "left_elbow_joint":            0.4869,
    "right_elbow_joint":           0.5236,
    "left_wrist_roll_joint":       0.3787,
    "right_wrist_roll_joint":     -0.4712,
    "left_wrist_pitch_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_wrist_yaw_joint":       0.0,
}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventCfgV6(EventCfgV2):
    """v2 events with:
    - narrow initial θ range [θ_min, θ_min+θ_step] — PD curriculum expands θ_hi
    - fixed initial p_des = p_mid=107 PSI — PD curriculum expands in Stage 1
    - Bernoulli dataset arm-init mixing — PD curriculum drives dataset_pct in Stage 2
    """

    reset_valve_angle = EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "angle_min": _THETA_MIN,                # 9.42 rad — v0-style start
            "angle_max": _THETA_MIN + _THETA_STEP,  # 11.46 rad — PD expands hi
        },
    )

    reset_p_des = EventTerm(
        func=mdp.reset_p_des_random,
        mode="reset",
        params={"p_min": _P_MID, "p_max": _P_MID},  # fixed 107 PSI — PD expands Stage 1
    )

    reset_arm_v6 = EventTerm(
        func=mdp.reset_arm_v6_mixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            ),
            "dataset_path": _DATASET_PATH,
            "fallback_pose": _PREGRASP_ARM_POSE,
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfgV6(RewardsCfgV2):
    """v2 pressure rewards + joint smoothness. No bilateral_contact (v5 proven harmful)."""

    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.0001,  # safe range per sim-isaaclab Don'ts
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class CurriculumCfgV6:
    """PD curriculum — sequential θ expansion, p expansion, dataset mixing."""

    stage = CurrTerm(
        func=mdp.turn_pd_curriculum_v6,
        params={
            "beta":             0.02,
            "sr_target":        0.85,
            "kp":               2.0,
            "kd":               0.5,
            "theta_scale":      1.0,
            "p_scale":          4.625,
            "mix_scale":        0.005,
            "confirm_iters":    20,
            "num_steps_per_env": 24,
            "theta_min":        _THETA_MIN,
            "theta_max":        _THETA_MAX,
            "theta_start_hi":   _THETA_MIN + _THETA_STEP,  # 11.46 rad
            "p_min":            _P_MIN,
            "p_max":            _P_MAX,
            "p_mid":            _P_MID,
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV6(ValveTurnEnvCfgV2):
    """v6: PD curriculum + decoupled θ/p axes + dataset mixing. Train from scratch."""

    observations: ObservationsCfgV2  = ObservationsCfgV2()
    rewards:      RewardsCfgV6       = RewardsCfgV6()
    terminations: TerminationsCfgV2  = TerminationsCfgV2()
    events:       EventCfgV6         = EventCfgV6()
    curriculum:   CurriculumCfgV6    = CurriculumCfgV6()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV6(ValveTurnEnvCfgV2):
    """Play config for v6 checkpoints.

    Inherits ValveTurnEnvCfgV2 directly — bypasses curriculum so play starts
    at full-range config immediately. Obs/action space identical to v6 training.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_p_des, apply_play_viewer
        apply_play_viewer(self)
        apply_play_p_des(self.events)
