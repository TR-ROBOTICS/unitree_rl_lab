"""Shared constants, scene, actions, and base events for valve task configs.

Imported by reach_env_cfg.py and turn_env_cfg.py.
g(θ) coefficients are firmware-locked — no DR.  CONTEXT.md §g(θ).
"""

from __future__ import annotations

import pathlib

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

# ---------------------------------------------------------------------------
# Constants — g(θ) firmware mapping (firmware-locked; no DR on these values)
# CONTEXT.md §g(θ): p_now = a·θ + b, clamp [15, 200] PSI
# ---------------------------------------------------------------------------
_G_THETA_A: float = 4.527    # PSI/rad
_G_THETA_B: float = -27.66   # PSI
_P_SPAN: float = 185.0        # PSI  (200 − 15)
_P_MIN: float = 15.0          # PSI
_P_MAX: float = 200.0         # PSI
_THETA_MIN: float = 9.42      # rad  (1.5 rev)
_THETA_MAX: float = 50.27     # rad  (8 rev)
_EPS_SIM: float = 1.85        # PSI  (~1% of span)

# ---------------------------------------------------------------------------
# Reach geometry — FK targets relative to wheel hub COM in world frame.
#
# Wheel is HORIZONTAL (normal = world-Z). Rim is a circle of radius 0.10 m
# in the XY plane at hub_z + plane_offset (0.022 m).
#
# Grip targets = left/right spokes (Y-axis positions):
#   Left  hand: (0,  +wheel_radius,  plane_offset) = (0, +0.10, +0.022)
#   Right hand: (0,  −wheel_radius,  plane_offset) = (0, −0.10, +0.022)
#
# X=0: hub COM is centered on the rim plane (no fore-aft offset).
# Y=±0.10: rim radius at 90°/270° around ring (left/right sides, not front).
# Z=+0.022: rim mid-plane height (plane_offset from hub COM vertex scan).
# ---------------------------------------------------------------------------
_REACH_LEFT_OFFSET: tuple[float, float, float] = (0.0, 0.10, 0.022)
_REACH_RIGHT_OFFSET: tuple[float, float, float] = (0.0, -0.10, 0.022)

# Handoff success thresholds (ADR 0004).
REACH_DEPTH_THRESHOLD: float = 0.01    # m  |Δx| — perpendicular to wheel face
REACH_INPLANE_THRESHOLD: float = 0.03  # m  √(Δy²+Δz²) — within wheel plane

# Hub body name — handwheel spinning body (body1 of RevoluteJoint).
# Confirmed Script Editor 2026-05-20.
_HUB_BODY_NAME: str = "mesh_50_AL_250_B7_8_A_stl"

# Wheel geometry — from v1 turn env (confirmed via Script Editor 2026-05-21).
_WHEEL_RADIUS: float = 0.10           # m — max XY-radial vertex distance from hub COM
_WHEEL_NORMAL_WORLD: tuple[float, float, float] = (0.0, 0.0, 1.0)  # world axis of revolution
_WHEEL_PLANE_OFFSET: float = 0.022   # m — hub COM to rim mid-plane along normal

# Hand body sets — same as v1 turn env (tip-led reach + palm fallback).
# Inspire topology: 4 fingers terminate at *_intermediate (no distal except thumb).
_LEFT_HAND_BODIES: list[str] = [
    "left_hand_base_link",
    "L_thumb_distal",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_ring_intermediate",
    "L_pinky_intermediate",
]
_RIGHT_HAND_BODIES: list[str] = [
    "right_hand_base_link",
    "R_thumb_distal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_ring_intermediate",
    "R_pinky_intermediate",
]

# Finger grip pose forced every reset — same as v0 (partial opposition, Grip A).
_FINGER_GRIP: dict[str, float] = {
    "L_index_proximal_joint": 0.6667,
    "L_index_intermediate_joint": 0.7278,
    "L_middle_proximal_joint": 0.7278,
    "L_middle_intermediate_joint": 1.1362,
    "L_pinky_proximal_joint": 0.6056,
    "L_pinky_intermediate_joint": 0.6266,
    "L_ring_proximal_joint": 0.6056,
    "L_ring_intermediate_joint": 1.0751,
    "L_thumb_proximal_yaw_joint": 0.6,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_thumb_intermediate_joint": 0.0,
    "L_thumb_distal_joint": 0.0,
    "R_index_proximal_joint": 0.5044,
    "R_index_intermediate_joint": 1.0542,
    "R_middle_proximal_joint": 0.6266,
    "R_middle_intermediate_joint": 1.0943,
    "R_pinky_proximal_joint": 0.7889,
    "R_pinky_intermediate_joint": 0.5044,
    "R_ring_proximal_joint": 0.9111,
    "R_ring_intermediate_joint": 0.6877,
    "R_thumb_proximal_yaw_joint": 0.6,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_thumb_intermediate_joint": 0.0,
    "R_thumb_distal_joint": 0.0,
}

# Asset paths — one level deeper than dof29/, so parents[6] not parents[5].
# __file__ = .../dof29/valve/base_cfg.py
# parents[0] = valve/, parents[1] = dof29/, ..., parents[6] = unitree_rl_lab/ (package root)
_ASSETS_DIR = pathlib.Path(__file__).parents[6] / "assets"
_VALVE_RIG_USD: str = str(_ASSETS_DIR / "valve_rig.usd")
_INSPIRE_USD: str = str(_ASSETS_DIR / "usd" / "g1_inspire_arm_collisions.usda")


# ---------------------------------------------------------------------------
# Scene — identical to ValveTurnSceneCfg in v0
# ---------------------------------------------------------------------------

@configclass
class ValveSceneCfg(InteractiveSceneCfg):
    """Scene: fixed-base G1 + valve rig."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_INSPIRE_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=0.5,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.793),
            joint_pos={
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                "left_shoulder_pitch_joint":   -0.7610,
                "right_shoulder_pitch_joint":  -0.7610,
                "left_shoulder_roll_joint":     0.1937,
                "right_shoulder_roll_joint":   -0.1937,
                "left_shoulder_yaw_joint":     -0.1239,
                "right_shoulder_yaw_joint":     0.1257,
                "left_elbow_joint":             0.4869,
                "right_elbow_joint":            0.5236,
                "left_wrist_roll_joint":        0.3787,
                "right_wrist_roll_joint":      -0.4712,
                "left_wrist_pitch_joint":       0.0,
                "right_wrist_pitch_joint":      0.0,
                "left_wrist_yaw_joint":         0.0,
                "right_wrist_yaw_joint":        0.0,
                "L_index_proximal_joint":       0.6667,
                "L_index_intermediate_joint":   0.7278,
                "L_middle_proximal_joint":      0.7278,
                "L_middle_intermediate_joint":  1.1362,
                "L_pinky_proximal_joint":       0.6056,
                "L_pinky_intermediate_joint":   0.6266,
                "L_ring_proximal_joint":        0.6056,
                "L_ring_intermediate_joint":    1.0751,
                "L_thumb_proximal_yaw_joint":   1.0524,
                "L_thumb_proximal_pitch_joint": 0.2496,
                "L_thumb_intermediate_joint":   0.3037,
                "L_thumb_distal_joint":         0.5149,
                "R_index_proximal_joint":       0.5044,
                "R_index_intermediate_joint":   1.0542,
                "R_middle_proximal_joint":      0.6266,
                "R_middle_intermediate_joint":  1.0943,
                "R_pinky_proximal_joint":       0.7889,
                "R_pinky_intermediate_joint":   0.5044,
                "R_ring_proximal_joint":        0.9111,
                "R_ring_intermediate_joint":    0.6877,
                "R_thumb_proximal_yaw_joint":   1.2990,
                "R_thumb_proximal_pitch_joint": 0.3089,
                "R_thumb_intermediate_joint":   0.5917,
                "R_thumb_distal_joint":         0.6440,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.90,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_knee_joint", ".*waist.*",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
                    ".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 35.0,
                    ".*waist_pitch_joint": 35.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0,
                    ".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0,
                    ".*waist_pitch_joint": 30.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0,
                    ".*waist.*": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0, ".*_knee_joint": 5.0,
                    ".*waist.*": 5.0,
                },
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=35.0,
                velocity_limit_sim=30.0,
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=37.0,
                stiffness=100.0,
                damping=2.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=37.0,
                stiffness=50.0,
                damping=2.0,
                armature=0.01,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*"],
                effort_limit_sim={
                    ".*_wrist_yaw_joint": 5.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                },
                velocity_limit_sim={
                    ".*_wrist_yaw_joint": 22.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                },
                stiffness=40.0,
                damping=2.0,
                armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_index_proximal_joint", ".*_index_intermediate_joint",
                    ".*_middle_proximal_joint", ".*_middle_intermediate_joint",
                    ".*_pinky_proximal_joint", ".*_pinky_intermediate_joint",
                    ".*_ring_proximal_joint", ".*_ring_intermediate_joint",
                    ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint",
                    ".*_thumb_intermediate_joint", ".*_thumb_distal_joint",
                ],
                effort_limit_sim=70.0,
                velocity_limit_sim=50.0,
                stiffness=400.0,
                damping=25.0,
                armature=0.0,
            ),
        },
    )

    valve_rig: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UsdFileCfg(usd_path=_VALVE_RIG_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.60, 0.0, 0.90),
            rot=(0.707, 0.0, 0.0, 0.707),
            joint_pos={"RevoluteJoint": _THETA_MIN},
        ),
        actuators={
            "wheel": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=50.0,
                stiffness=0.0,
                damping=1.0,
                armature=0.0,
            ),
        },
    )

    # Palm contact sensors — used for bilateral contact reward and force jerk penalty (v5+).
    # history_length=2: current step [t=0] + previous step [t=1], avoids storing prev force on env.
    # net_forces_w shape: (num_envs, 1, 3); norm gives scalar contact force per hand per env.
    # activate_contact_sensors=True on robot spawn (above) is required for these to receive data.
    left_palm_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_base_link",
        history_length=2,
        track_air_time=False,
    )

    right_palm_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link",
        history_length=2,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 1.0)),
    )


# ---------------------------------------------------------------------------
# Actions — 14 arm-joint Δ-targets, scale ±0.1 rad (action space spec)
# ---------------------------------------------------------------------------

@configclass
class ValveActionsCfg:
    arm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        scale=0.1,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Base events — shared by both reach and turn envs
# ---------------------------------------------------------------------------

@configclass
class ValveBaseEventCfg:
    """Events common to reach and turn: reset, valve DR, valve angle reset, finger grip."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # Valve position DR — uniform ±half_range per axis around base_pos.
    # ADR 0004: same DR range must be active in both policies (reach + turn).
    # g(θ) NOT randomized — firmware-locked.
    reset_valve_pos = EventTerm(
        func=mdp.reset_valve_base_position_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "base_pos": (0.60, 0.0, 0.90),
            "base_quat": (0.707, 0.0, 0.0, 0.707),   # wxyz; +90° Z
            "half_range_xyz": (0.05, 0.05, 0.05),
        },
    )

    # Explicit valve angle reset to θ_min after position DR.
    # Passive joint (stiffness=0) can drift; this guarantees θ = θ_min, ω = 0.
    reset_valve_angle = EventTerm(
        func=mdp.reset_joints_to_fixed_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve_rig"),
            "joint_pose": {"RevoluteJoint": _THETA_MIN},
        },
    )

    # Force finger curl grip — must run after reset_all (declaration order = exec order).
    # Inspire USD authored drive (tgt=0, k=20) wins over init_state; this overwrites it.
    reset_finger_grip = EventTerm(
        func=mdp.reset_joints_to_fixed_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pose": _FINGER_GRIP,
        },
    )
