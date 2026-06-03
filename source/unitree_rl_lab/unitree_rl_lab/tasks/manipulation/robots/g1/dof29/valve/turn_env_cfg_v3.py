"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 2 v3.

v3 = v2 + arm init DR from reach-terminal-state dataset.

Changes vs v2:
  - Arm init: sampled from ``datasets/reach_arm_positions.npy`` instead of
    fixed pre-grip pose. EventTerm ``reset_arm_init`` runs after
    ``reset_scene_to_default``, overwriting arm joints with a random terminal
    state collected by the reach policy.
  - Fallback: if dataset file absent (before collection), arm stays at pre-grip
    pose (v2 behaviour). Training can start immediately.

Why: reach policy terminal states are closer to the wheel than pre-grip pose
(arms already moved ~14 cm toward valve). Bootstrapping turn with diverse
near-valve arm init → more informative gradient signal at episode start,
reduces reliance on a single pre-placed init that the real robot won't have.

Dataset format: (N, 14) float32 array, rad.  14 arm joints in the order
resolved by SceneEntityCfg("robot", joint_names=[".*_shoulder_.*",
".*_elbow_.*", ".*_wrist_.*"]).  Collected via
``scripts/rsl_rl/collect_reach_dataset.py``.

g(θ) firmware-locked — no DR.
Valve pos DR: disabled (same as v2).  Re-enable before sim2real.
"""

from __future__ import annotations

import pathlib

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import (
    _THETA_MIN,
    _THETA_MAX,
)
from .turn_env_cfg_v2 import (
    EventCfgV2,
    ObservationsCfgV2,
    RewardsCfgV2,
    TerminationsCfgV2,
    ValveTurnEnvCfgV2,
)

# ---------------------------------------------------------------------------
# Dataset path — same directory as assets/ (unitree_rl_lab package root)
# __file__ = .../dof29/valve/turn_env_cfg_v3.py
# parents[6] = unitree_rl_lab/ package root (see base_cfg.py comment)
# ---------------------------------------------------------------------------
_DATASET_PATH: str = str(
    pathlib.Path(__file__).parents[6] / "datasets" / "reach_arm_positions.npy"
)

# Pre-grip arm pose — fallback when dataset absent.
# Values from ValveSceneCfg.init_state.joint_pos (base_cfg.py).
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
# Events — v2 + arm init from dataset
# ---------------------------------------------------------------------------

@configclass
class EventCfgV3(EventCfgV2):
    """v2 events + arm joint init sampled from reach terminal-state dataset.

    Runs AFTER reset_all (declaration order = exec order), overwriting the arm
    joints that reset_scene_to_default set from ValveSceneCfg.init_state.
    Fingers remain at pre-grip (reset_finger_grip, inherited from v2).
    """

    reset_arm_init = EventTerm(
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


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV3(ValveTurnEnvCfgV2):
    """Stage 2 v3: random θ_init + random p_des + reach-bootstrapped arm init."""

    events: EventCfgV3 = EventCfgV3()

    def __post_init__(self):
        super().__post_init__()   # inherits DR disable + physics config from v2


@configclass
class ValveTurnPlayEnvCfgV3(ValveTurnEnvCfgV3):
    """Single-env play config for Stage 2 v3."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_p_des
        apply_play_p_des(self.events)
