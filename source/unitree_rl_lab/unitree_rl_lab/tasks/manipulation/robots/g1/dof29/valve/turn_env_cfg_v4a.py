"""Valve-turn task config — G1 29-DoF + Inspire hands, v4a: auto-curriculum (hard).

v4a = v4 architecture + 3-stage automatic curriculum, dataset arm init from Stage 0.

Curriculum stages:
  Stage 0 (start):  θ_init = θ_min (fixed), p_des = 50 PSI (fixed), dataset arm init
  Stage 1:          θ_init ∈ [θ_min, θ_max] (random)                   ← v1 level
  Stage 2:          p_des  ∈ [15, 200] PSI  (random)                    ← v2 level

Advancement trigger: rolling SR ≥ 85% over window_iters training iterations.

Naming:
  v4a  — auto-curriculum, arm only, dataset arm init Stage 0 (hard)
  v4ae — auto-curriculum, arm only, pre-grip Stage 0 (easy)
  v4aeh — auto-curriculum + hands (38 DoF)

Train from scratch — do NOT resume from any checkpoint.

g(θ) firmware-locked — no DR.
Valve pos DR: disabled.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import _THETA_MIN, _THETA_MAX, _P_MIN, _P_MAX
from .turn_env_cfg_v3 import EventCfgV3
from .turn_env_cfg_v4 import ValveTurnEnvCfgV4

_P_DES_INIT: float = 107.0  # central target → θ_des≈29.7 mid-range → direction-balanced θ_init reset (see _P_DES_STAGE1)


@configclass
class EventCfgV4A(EventCfgV3):
    """v3 events + θ_init and p_des fixed at Stage 0."""

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


@configclass
class CurriculumCfgV4A:
    """3-stage auto-curriculum, iter-based dwell."""

    turn_auto_stage = CurrTerm(
        func=mdp.turn_auto_curriculum_stage,
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
class ValveTurnEnvCfgV4A(ValveTurnEnvCfgV4):
    """v4a: 3-stage auto-curriculum, dataset arm init from Stage 0. Train from scratch."""

    events:     EventCfgV4A      = EventCfgV4A()
    curriculum: CurriculumCfgV4A = CurriculumCfgV4A()

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV4A(ValveTurnEnvCfgV4):
    """Play config for v4a checkpoints.

    Inherits ValveTurnEnvCfgV4 (not V4A) — bypasses curriculum entirely so play
    starts at final-stage config (random θ, random p_des, dataset arm init) immediately.
    Obs/action space identical; v4a checkpoints load cleanly into this env.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_p_des, apply_play_viewer
        apply_play_viewer(self)
        apply_play_p_des(self.events)
