"""Valve-turn task config — G1 29-DoF + Inspire hands, v5.

v5 = fresh training + smooth dual-axis curriculum + bimanual contact rewards + CCD fingers.

Key changes vs v4:
  1. CCD on all 24 finger phalanges (USDA) — eliminates thumb tunneling through spokes.
  2. bilateral_contact reward — multiplicative L×R palm force, prevents one-arm solutions.
  3. contact_force_jerk penalty — penalises abrupt contact force spikes (slapping).
  4. Smooth dual-axis curriculum (turn_smooth_curriculum_v5):
       Stage 0: θ and p_des expand simultaneously ±5% of span per 85% SR trigger
       Stage 1: dataset arm init mixing +10% per trigger
       Stage 2: fully open, 100% dataset init
  5. Train from scratch (fresh init) — CCD changes physics; prior checkpoints invalid.

See docs/adr/0007-v5-smooth-curriculum-bimanual-ccd.md for full design rationale.

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

# v5 curriculum constants — match ADR 0007
_THETA_MID: float = 29.75   # (107 − (−27.66)) / 4.527 ≈ midpoint of [9.42, 50.27]
_THETA_STEP: float = 2.04   # 5% of (50.27 − 9.42) = 40.85 rad
_P_MID: float = 107.0
_P_STEP: float = 9.25       # 5% of 185 PSI span

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
class EventCfgV5(EventCfgV2):
    """v2 events with:
    - narrow initial θ range [θ_mid±θ_step] (curriculum expands)
    - fixed initial p_des=107 (curriculum expands)
    - probabilistic dataset arm init (curriculum drives dataset_pct)
    """

    reset_valve_angle = EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "angle_min": _THETA_MIN,               # 9.42 rad — v0-style start, curriculum expands hi
            "angle_max": _THETA_MIN + _THETA_STEP, # 11.46 rad
        },
    )

    reset_p_des = EventTerm(
        func=mdp.reset_p_des_random,
        mode="reset",
        params={"p_min": 107.0, "p_max": 107.0},  # fixed 107 PSI — matches v0 start
    )

    reset_arm_v5 = EventTerm(
        func=mdp.reset_arm_v5_mixed,
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
class RewardsCfgV5(RewardsCfgV2):
    """v2 pressure rewards + joint smoothness + bilateral contact + force jerk."""

    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.0001,
    )

    bilateral_contact = RewTerm(
        func=mdp.bilateral_contact,
        weight=0.0,  # disabled — any positive weight creates touch-only local optimum (v5a/b/e/f/g all failed)
        params={
            "left_sensor_name":  "left_palm_sensor",
            "right_sensor_name": "right_palm_sensor",
            "f_max": 50.0,
        },
    )

    contact_force_jerk = RewTerm(
        func=mdp.contact_force_jerk,
        weight=0.0,
        params={
            "left_sensor_name":  "left_palm_sensor",
            "right_sensor_name": "right_palm_sensor",
        },
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class CurriculumCfgV5:
    """Smooth dual-axis curriculum — simultaneous θ + p_des expansion, then dataset mixing."""

    v5_stage = CurrTerm(
        func=mdp.turn_smooth_curriculum_v5,
        params={
            "success_threshold": 0.85,
            "window_iters":      10,
            "num_steps_per_env": 24,
            "theta_min":         _THETA_MIN,
            "theta_max":         _THETA_MAX,
            "theta_mid":         _THETA_MID,
            "theta_step":        _THETA_STEP,
            "p_min":             _P_MIN,
            "p_max":             _P_MAX,
            "p_mid":             _P_MID,
            "p_step":            _P_STEP,
            "dataset_step":      0.10,
            # v0-style start: θ fixed at θ_min, p_des=107 PSI — matches v0 which converged
            "theta_start_lo":    _THETA_MIN,
            "theta_start_hi":    _THETA_MIN + _THETA_STEP,
            "p_start":           107.0,
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV5(ValveTurnEnvCfgV2):
    """v5: smooth curriculum + bimanual contact rewards + CCD fingers. Train from scratch."""

    observations: ObservationsCfgV2  = ObservationsCfgV2()
    rewards:      RewardsCfgV5       = RewardsCfgV5()
    terminations: TerminationsCfgV2  = TerminationsCfgV2()
    events:       EventCfgV5         = EventCfgV5()
    curriculum:   CurriculumCfgV5    = CurriculumCfgV5()

    def __post_init__(self):
        super().__post_init__()


@configclass
class EventCfgV5Play(EventCfgV2):
    """Play events for v5: full θ/p range + 100% dataset arm init.

    Matches Stage-2 training distribution (dataset_pct=1.0) without running the
    curriculum. reset_arm_from_dataset uses the same dataset path and fallback pose
    as reset_arm_v5 (Stage 2 = 100% dataset), so eval init = training init.
    """

    reset_arm = EventTerm(
        func=mdp.reset_arm_from_dataset,
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


@configclass
class ValveTurnPlayEnvCfgV5(ValveTurnEnvCfgV2):
    """Play config for v5 checkpoints.

    Uses EventCfgV5Play: full θ/p range + 100% dataset arm init.
    Dataset init matches Stage-2 training distribution (model_1100 trained with
    dataset_pct=1.0). Without this, arm starts from USD init_state (pre-grip),
    which is OOD for the final 100 training iters.
    """

    events: EventCfgV5Play = EventCfgV5Play()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_p_des, apply_play_viewer
        apply_play_viewer(self)
        apply_play_p_des(self.events)
