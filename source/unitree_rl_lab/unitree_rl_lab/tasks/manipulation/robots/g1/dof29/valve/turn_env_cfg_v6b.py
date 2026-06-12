"""Valve-turn task config — G1 29-DoF + Inspire hands, v6b.

v6b = v6 bimanual finetune endpoint (no curriculum).

Fine-tunes from turn_v6_model_1999.pt (Stage 3 terminal).  The PD curriculum
is dropped — environment starts at full-range Stage-3 distribution directly.
Only addition: rim_distance_reward(mode="max") to pull both arms to the valve
rim and break single-hand dominance.

Why bimanual_progress (multiplicative), not standalone rim_distance:
  - Standalone rim_distance (w=1.0, additive) collapsed in v6b run 1:
    policy parked both hands on rim (rim=0.88/step), stopped turning entirely
    (progress=0.003, success=11%, timeout=86%).  Same failure mode as
    bilateral_contact (ADR 0007) but geometric variant.
  - bimanual_progress = pressure_progress_random x rim_distance(mode="max"):
    zero unless turning AND both hands near rim.  No park-near-rim local
    optimum.  Two-hand turning gets ~2x reward vs one-hand.

No curriculum term — short finetune, no anneal needed.

See docs/adr/0008-v6-pd-curriculum-decoupled-axes.md for v6 design rationale.
"""

from __future__ import annotations

import pathlib

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import (
    _G_THETA_A,
    _G_THETA_B,
    _HUB_BODY_NAME,
    _LEFT_HAND_BODIES,
    _RIGHT_HAND_BODIES,
    _WHEEL_RADIUS,
    _WHEEL_NORMAL_WORLD,
    _WHEEL_PLANE_OFFSET,
    _THETA_MIN,
    _THETA_MAX,
    _P_MIN,
    _P_MAX,
    _P_SPAN,
)
from .turn_env_cfg_v6 import (
    EventCfgV6,
    RewardsCfgV6,
    ObservationsCfgV2,
    TerminationsCfgV2,
    ValveTurnEnvCfgV6,
    _DATASET_PATH,
    _PREGRASP_ARM_POSE,
)

# ---------------------------------------------------------------------------
# Events — v6 base with full-range overrides (Stage 3 equivalent)
# ---------------------------------------------------------------------------

@configclass
class EventCfgV6b(EventCfgV6):
    """v6 events with Stage-3 distribution: full θ range, full p range, 100% dataset.

    EventCfgV6 starts narrow (θ=[θ_min, θ_min+θ_step], p=107 fixed, dataset_pct=0)
    and relies on the PD curriculum to expand ranges.  v6b drops the curriculum,
    so we override the three affected terms to match Stage-3 directly.
    """

    # Full θ range — un-narrow the PD start window
    reset_valve_angle = EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "angle_min": _THETA_MIN,
            "angle_max": _THETA_MAX,
        },
    )

    # Full p range — PD Stage 1 was still p-expanding; skip to full
    reset_p_des = EventTerm(
        func=mdp.reset_p_des_random,
        mode="reset",
        params={"p_min": _P_MIN, "p_max": _P_MAX},
    )

    # 100% dataset arm init — replaces Bernoulli v6_mixed (which reads
    # env._v6curr_dataset_pct = 0.0 with no curriculum running)
    reset_arm_v6 = EventTerm(
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
# Rewards — v6 + bimanual_progress multiplicative term
# ---------------------------------------------------------------------------

@configclass
class RewardsCfgV6b(RewardsCfgV6):
    """v6 pressure rewards + smoothness + bimanual_progress multiplicative term.

    bimanual_progress = pressure_progress_random x rim_distance(mode="max").
    Multiplicative gate: zero unless valve is moving AND both hands near rim.
    Prevents park-near-rim local optimum (collapsed v6b run 1: standalone
    rim_distance w=1.0 -> success=11%, timeout=86%, progress~=0).
    weight=30.0 (same as pressure_progress_random) -> two-hand turning ~= 2x
    reward vs one-hand; no standalone hovering incentive.
    """

    bimanual_progress = RewTerm(
        func=mdp.bimanual_progress_reward,
        weight=30.0,
        params={
            "pressure_a":         _G_THETA_A,
            "pressure_b":         _G_THETA_B,
            "p_min":              _P_MIN,
            "p_max":              _P_MAX,
            "p_span":             _P_SPAN,
            "valve_hub_cfg":      SceneEntityCfg("valve_rig", body_names=[_HUB_BODY_NAME]),
            "left_hand_cfg":      SceneEntityCfg("robot", body_names=_LEFT_HAND_BODIES),
            "right_hand_cfg":     SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES),
            "wheel_radius":       _WHEEL_RADIUS,        # 0.10 m
            "wheel_normal_world": _WHEEL_NORMAL_WORLD,  # (0.0, 0.0, 1.0)
            "plane_offset":       _WHEEL_PLANE_OFFSET,  # 0.022 m
            "sigma":              0.15,                  # 15 cm bandwidth
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV6b(ValveTurnEnvCfgV6):
    """v6b: finetune endpoint — full range, rim_distance bimanual reward, no curriculum."""

    observations: ObservationsCfgV2 = ObservationsCfgV2()
    rewards:      RewardsCfgV6b     = RewardsCfgV6b()
    terminations: TerminationsCfgV2 = TerminationsCfgV2()
    events:       EventCfgV6b       = EventCfgV6b()
    curriculum:   None              = None  # no PD curriculum; base ValveTurnEnvCfg sets None

    def __post_init__(self):
        super().__post_init__()


@configclass
class ValveTurnPlayEnvCfgV6b(ValveTurnEnvCfgV6b):
    """Play config for v6b checkpoints.

    Full-range distribution; no curriculum.  Obs/action space identical to v6/v6b.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_p_des, apply_play_viewer
        apply_play_viewer(self)
        apply_play_p_des(self.events)
