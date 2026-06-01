"""Valve-turn task config — G1 29-DoF + Inspire hands, v4ae: auto-curriculum (easy).

v4ae = 4-stage auto-curriculum mirroring the human-guided v0→v1→v2→v3 chain.
Pre-grip arm init until Stage 3 (identical start to v0).

Curriculum stages:
  Stage 0 (start):  θ_init = θ_min (fixed), p_des = 50 PSI, pre-grip arm init  ← v0
  Stage 1:          θ_init ∈ [θ_min, θ_max] random, p_des = 50 PSI             ← v1
  Stage 2:          θ_init random, p_des ∈ [15, 200] PSI random                 ← v2
  Stage 3:          θ_init random, p_des random, dataset arm init               ← v3+v4

Advancement trigger: rolling SR ≥ 85% over window_iters training iterations.

Naming:
  v4a  — auto-curriculum, arm only, dataset arm init Stage 0 (hard)
  v4ae — auto-curriculum, arm only, pre-grip Stage 0 (easy, this file)
  v4aeh — auto-curriculum + hands (38 DoF)

Train from scratch — do NOT resume from any checkpoint.

g(θ) firmware-locked — no DR.
Valve pos DR: disabled.
"""

from __future__ import annotations

import pathlib

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import _THETA_MIN, _THETA_MAX, _P_MIN, _P_MAX
from .turn_env_cfg_v4 import ValveTurnEnvCfgV4
from .turn_env_cfg_v2 import EventCfgV2

_P_DES_INIT: float = 50.0

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


@configclass
class EventCfgV4AE(EventCfgV2):
    """v2 events + fixed θ_init + fixed p_des + staged arm init."""

    reset_valve_angle = EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "angle_min": _THETA_MIN,
            "angle_max": _THETA_MIN,
        },
    )

    reset_p_des = EventTerm(
        func=mdp.reset_p_des_random,
        mode="reset",
        params={"p_min": _P_DES_INIT, "p_max": _P_DES_INIT},
    )

    reset_arm_init = EventTerm(
        func=mdp.reset_arm_staged,
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
class CurriculumCfgV4AE:
    """4-stage auto-curriculum, iter-based dwell."""

    turn_auto_stage = CurrTerm(
        func=mdp.turn_auto_curriculum_stage_easy,
        params={
            "success_threshold": 0.85,
            "window_iters":      100,
            "num_steps_per_env": 24,
            "theta_max":         _THETA_MAX,
            "p_min_target":      _P_MIN,
            "p_max_target":      _P_MAX,
        },
    )


@configclass
class ValveTurnEnvCfgV4AE(ValveTurnEnvCfgV4):
    """v4ae: 4-stage auto-curriculum, pre-grip start, mirrors v0→v3 chain. Train from scratch."""

    events:     EventCfgV4AE      = EventCfgV4AE()
    curriculum: CurriculumCfgV4AE = CurriculumCfgV4AE()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV4AE(ValveTurnEnvCfgV4AE):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
