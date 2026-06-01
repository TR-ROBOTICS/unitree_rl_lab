"""Valve-turn task config — G1 29-DoF + Inspire hands, v4aeh: auto-easy + active hands.

v4aeh = v4ae (4-stage auto-curriculum, pre-grip Stage 0) + Inspire hands in action space.

Purpose: experimental comparison — does active finger control improve turn performance
vs arm-only baseline (v4ae)?

Action space: arm(14, scale=0.1) + hands(24, scale=0.05) = 38 DoF total.
  use_default_offset=True on hands → action=0 maps to init_state grip curl values.
  reset_finger_grip event initialises fingers to curl at episode start; policy
  may tighten/release grip during episode.

Observation space:
  joint_pos_rel(14) + joint_vel_rel(14) + p_now(1) + p_des(1)
  + last_action(38, auto-expands from 14 because action_manager.action is 38d)
  + finger_joint_pos_rel(24)
  = 92 dims total (incompatible with v4ae/v4 checkpoints — train from scratch).

Curriculum: identical to v4ae (4-stage, iter-based dwell, window_iters=100).
Events:     identical to v4ae (staged arm init, fixed θ/p at Stage 0, reset_finger_grip).
Rewards:    identical to v4 (pressure_error, pressure_progress, action_rate_l2=-0.0001).
            action_rate_l2 now penalises all 38 joints — weight stays at -0.0001.

Train from scratch — do NOT resume from any checkpoint.

g(θ) firmware-locked — no DR.
Valve pos DR: disabled.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .base_cfg import ValveActionsCfg
from .turn_env_cfg_v4ae import (
    ValveTurnEnvCfgV4AE,
    EventCfgV4AE,
    CurriculumCfgV4AE,
)
from .turn_env_cfg_v2 import ObservationsCfgV2

# Finger joint patterns — match base_cfg "hands" actuator
_FINGER_JOINT_NAMES: list[str] = [
    ".*_index_proximal_joint",       ".*_index_intermediate_joint",
    ".*_middle_proximal_joint",      ".*_middle_intermediate_joint",
    ".*_pinky_proximal_joint",       ".*_pinky_intermediate_joint",
    ".*_ring_proximal_joint",        ".*_ring_intermediate_joint",
    ".*_thumb_proximal_yaw_joint",   ".*_thumb_proximal_pitch_joint",
    ".*_thumb_intermediate_joint",   ".*_thumb_distal_joint",
]


# ---------------------------------------------------------------------------
# Actions — arm (14) + hands (24) = 38 DoF
# ---------------------------------------------------------------------------

@configclass
class ValveActionsCfgHands(ValveActionsCfg):
    """Arm Δ-targets (14, scale=0.1) + hand Δ-targets (24, scale=0.05).

    use_default_offset=True: policy output 0 → init_state grip pose (curl values).
    """

    hands = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FINGER_JOINT_NAMES,
        scale=0.05,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations — v2 obs + finger joint positions (24d)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfgV4AEH(ObservationsCfgV2):
    @configclass
    class PolicyCfg(ObservationsCfgV2.PolicyCfg):
        """v2 policy obs + finger joint positions relative to init_state default."""

        finger_joint_pos_rel = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_FINGER_JOINT_NAMES,
                ),
            },
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Env config — v4ae base, override actions + observations
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV4AEH(ValveTurnEnvCfgV4AE):
    """v4aeh: v4ae curriculum + 38-DoF action space + finger obs (92d).

    Train from scratch — obs dim incompatible with v4/v4ae checkpoints.
    """

    actions:      ValveActionsCfgHands = ValveActionsCfgHands()
    observations: ObservationsCfgV4AEH = ObservationsCfgV4AEH()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV4AEH(ValveTurnEnvCfgV4AEH):
    """Single-env play config for v4aeh."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
