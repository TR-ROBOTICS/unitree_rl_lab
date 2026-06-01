"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 2 v1.

v1 = v0 + random θ_init ∈ [θ_min, θ_max].
Everything else identical to v0 (pre-grip arm init, fixed p_des=50 PSI,
valve DR disabled, hold_steps=50).

Resume from v0 checkpoint (model_100) — policy already knows how to turn,
now must generalize to arbitrary start angles (both CW and CCW directions).

g(θ) firmware-locked — no DR.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import _THETA_MIN, _THETA_MAX
from .turn_env_cfg import EventCfg, ValveTurnEnvCfg


# ---------------------------------------------------------------------------
# Events — swap fixed θ_init for uniform random [θ_min, θ_max]
# ---------------------------------------------------------------------------

@configclass
class EventCfgV1(EventCfg):
    """v0 events + random θ_init."""

    reset_valve_angle = EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "angle_min": _THETA_MIN,
            "angle_max": _THETA_MAX,
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV1(ValveTurnEnvCfg):
    """Stage 2: random θ_init, all else v0."""

    events: EventCfgV1 = EventCfgV1()

    def __post_init__(self):
        super().__post_init__()  # inherits DR disable + physics config


@configclass
class ValveTurnPlayEnvCfgV1(ValveTurnEnvCfgV1):
    """Single-env play config for Stage 2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
