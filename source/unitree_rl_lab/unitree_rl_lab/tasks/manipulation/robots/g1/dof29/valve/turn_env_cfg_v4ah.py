"""Valve-turn task config — G1 29-DoF + Inspire hands, v4ah: auto-hard + active hands.

v4ah = v4a (3-stage auto-curriculum, dataset arm init Stage 0) + hands in action space.

Counterpart to v4aeh (which uses v4ae easy-start curriculum).
Harder Stage 0: dataset arm init from the start (vs pre-grip in v4aeh).

Action space: arm(14, scale=0.1) + hands(24, scale=0.05) = 38 DoF total.
Observations: joint_pos_rel(14) + joint_vel_rel(14) + p_now(1) + p_des(1)
              + last_action(38) + finger_joint_pos_rel(24) = 92 dims.

Curriculum stages:
  Stage 0: θ_init = θ_min (fixed), p_des = 50 PSI, dataset arm init
  Stage 1: θ_init random
  Stage 2: p_des random

Train from scratch — obs dim incompatible with v4a/v4 checkpoints.

g(θ) firmware-locked — no DR.
Valve pos DR: disabled.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .base_cfg import ValveActionsCfg
from .turn_env_cfg_v4a import ValveTurnEnvCfgV4A
from .turn_env_cfg_v2 import ObservationsCfgV2

_FINGER_JOINT_NAMES: list[str] = [
    ".*_index_proximal_joint",       ".*_index_intermediate_joint",
    ".*_middle_proximal_joint",      ".*_middle_intermediate_joint",
    ".*_pinky_proximal_joint",       ".*_pinky_intermediate_joint",
    ".*_ring_proximal_joint",        ".*_ring_intermediate_joint",
    ".*_thumb_proximal_yaw_joint",   ".*_thumb_proximal_pitch_joint",
    ".*_thumb_intermediate_joint",   ".*_thumb_distal_joint",
]


@configclass
class ValveActionsCfgHands(ValveActionsCfg):
    """Arm Δ-targets (14, scale=0.1) + hand Δ-targets (24, scale=0.05)."""

    hands = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FINGER_JOINT_NAMES,
        scale=0.05,
        use_default_offset=True,
    )


@configclass
class ObservationsCfgV4AH(ObservationsCfgV2):
    @configclass
    class PolicyCfg(ObservationsCfgV2.PolicyCfg):
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


@configclass
class ValveTurnEnvCfgV4AH(ValveTurnEnvCfgV4A):
    """v4ah: v4a curriculum (3-stage, dataset arm) + 38-DoF action + finger obs (92d).

    Train from scratch.
    """

    actions:      ValveActionsCfgHands = ValveActionsCfgHands()
    observations: ObservationsCfgV4AH  = ObservationsCfgV4AH()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV4AH(ValveTurnEnvCfgV4AH):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
