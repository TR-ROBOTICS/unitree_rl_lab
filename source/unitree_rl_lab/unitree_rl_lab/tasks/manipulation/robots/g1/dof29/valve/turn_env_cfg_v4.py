"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 2 v4.

v4 = v3 + smoothness penalty.

Changes vs v3:
  - action_rate_l2 weight: 0.0 → -0.005
    Penalises large action deltas each step. Addresses high-std swiping
    behaviour observed in v3 (mean_std rising 3.94→4.73, entropy rising).
    Weight chosen small enough to preserve 90%+ success while regularising
    toward smoother trajectories.

Resume from turn_v3_model_600 (iter 600 — lower std than iter 700 final).

g(θ) firmware-locked — no DR.
Valve pos DR: disabled (same as v3).
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .turn_env_cfg_v2 import RewardsCfgV2
from .turn_env_cfg_v3 import ValveTurnEnvCfgV3


@configclass
class RewardsCfgV4(RewardsCfgV2):
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0001)


@configclass
class ValveTurnEnvCfgV4(ValveTurnEnvCfgV3):
    """Stage 2 v4: v3 + smoothness penalty (action_rate_l2 = -0.005)."""

    rewards: RewardsCfgV4 = RewardsCfgV4()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV4(ValveTurnEnvCfgV4):
    """Single-env play config for Stage 2 v4."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
