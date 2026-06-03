"""Valve reach policy env config — G1 29-DoF + Inspire hands.

Phase 1 of the two-policy chain (ADR 0004): train an arm-reach policy whose
terminal states (both hands at hub rim targets) are collected by play.py to
bootstrap the turn policy's arm-init distribution.

Observation space: 45d
  joint_pos  (14d) — arm shoulder/elbow/wrist relative positions
  joint_vel  (14d) — arm relative velocities
  valve_pos  (3d)  — valve root position relative to robot root (world frame)
  last_action(14d) — previous step's arm action (temporal context)

Handoff condition (ADR 0004):
  |Δx| < 0.01 m  (depth, perpendicular to wheel face)
  √(Δy²+Δz²) < 0.03 m  (in-plane radius)
  both hands simultaneously.

g(θ) not used — reach policy does not observe pressure.
Valve position DR: ±5 cm XYZ, same as turn policy (ADR 0004 requirement).
Smoothness penalty active from start (no contact instability pre-grasp).
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import (
    ValveSceneCfg,
    ValveActionsCfg,
    ValveBaseEventCfg,
    _HUB_BODY_NAME,
    _LEFT_HAND_BODIES,
    _RIGHT_HAND_BODIES,
    _WHEEL_RADIUS,
    _WHEEL_NORMAL_WORLD,
    _WHEEL_PLANE_OFFSET,
    _REACH_LEFT_OFFSET,
    _REACH_RIGHT_OFFSET,
    REACH_DEPTH_THRESHOLD,
    REACH_INPLANE_THRESHOLD,
)

# Arm joints to zero at reach init — robot must LEARN to reach, not start pre-placed.
# Same list as v1 turn env (unitree_sdk2 g1_low_level_example.py Stage 1 = all zeros).
_G1_ARM_JOINTS = [
    "left_shoulder_pitch_joint",  "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "right_shoulder_yaw_joint",
    "left_elbow_joint",           "right_elbow_joint",
    "left_wrist_roll_joint",      "right_wrist_roll_joint",
    "left_wrist_pitch_joint",     "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",       "right_wrist_yaw_joint",
]


# ---------------------------------------------------------------------------
# Observations — 45d: 14 pos + 14 vel + 3 valve_pos + 14 last_action
# ---------------------------------------------------------------------------

@configclass
class ReachObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            )},
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            )},
            clip=(-20.0, 20.0),
        )
        # Valve position relative to robot root (world frame).
        # Valve pos is DR'd per episode — policy must condition on it.
        valve_pos = ObsTerm(
            func=mdp.valve_pos_robot_frame,
            params={
                "valve_cfg": SceneEntityCfg("valve_rig"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            clip=(-2.0, 2.0),
        )
        # Previous step's action — temporal context for smooth trajectories.
        last_action = ObsTerm(
            func=mdp.last_arm_action,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — distance-based dense + sparse handoff bonus + smoothness
# ---------------------------------------------------------------------------

@configclass
class ReachRewardsCfg:
    # Primary driver: potential-based progress — r_t = (d_{t-1} − d_t) / scale.
    # Telescopes to total distance closed. Standing still → r=0 → no zero-motion min.
    # Same design as pressure_progress in the turn task (fixes asymptotic collapse).
    # distance_scale=0.20: 1cm step → 0.05/step, 5cm step → 0.25/step.
    reach_progress = RewTerm(
        func=mdp.reach_progress_reward,
        weight=30.0,  # was 10.0 — tripled to break value_loss=0 plateau
        params={
            "valve_hub_cfg":       SceneEntityCfg("valve_rig", body_names=[_HUB_BODY_NAME]),
            "left_hand_cfg":       SceneEntityCfg("robot", body_names=_LEFT_HAND_BODIES),
            "right_hand_cfg":      SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES),
            "left_target_offset":  _REACH_LEFT_OFFSET,
            "right_target_offset": _REACH_RIGHT_OFFSET,
            "distance_scale":      0.10,  # was 0.20 — 2× per-step signal to combat value_loss=0
        },
    )

    # Sparse bonus when handoff condition met.
    handoff_bonus = RewTerm(
        func=mdp.reach_handoff_bonus,
        weight=1.0,
        params={
            "left_hand_cfg":  SceneEntityCfg("robot", body_names=["left_hand_base_link"]),
            "right_hand_cfg": SceneEntityCfg("robot", body_names=["right_hand_base_link"]),
            "valve_hub_cfg":  SceneEntityCfg("valve_rig", body_names=[_HUB_BODY_NAME]),
            "left_target_offset":  _REACH_LEFT_OFFSET,
            "right_target_offset": _REACH_RIGHT_OFFSET,
            "depth_threshold":    REACH_DEPTH_THRESHOLD,
            "inplane_threshold":  REACH_INPLANE_THRESHOLD,
            "bonus": 5.0,
        },
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=0.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class ReachTerminationsCfg:
    # Episode timeout — 15 s per CONTEXT.md §Timeouts (reach phase).
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Handoff success — terminates with time_out=False so RSL-RL bootstraps
    # terminal value with 0 (goal completion, not timeout).
    handoff = DoneTerm(
        func=mdp.reach_handoff_condition,
        time_out=False,
        params={
            "left_hand_cfg":  SceneEntityCfg("robot", body_names=["left_hand_base_link"]),
            "right_hand_cfg": SceneEntityCfg("robot", body_names=["right_hand_base_link"]),
            "valve_hub_cfg":  SceneEntityCfg("valve_rig", body_names=[_HUB_BODY_NAME]),
            "left_target_offset":  _REACH_LEFT_OFFSET,
            "right_target_offset": _REACH_RIGHT_OFFSET,
            "depth_threshold":    REACH_DEPTH_THRESHOLD,
            "inplane_threshold":  REACH_INPLANE_THRESHOLD,
        },
    )

    # Catastrophic-only joint-vel guard (same spec as turn env).
    # grace_steps=25 (0.5 s @50 Hz) suppresses reset snap on first steps.
    joint_vel_explosion = DoneTerm(
        func=mdp.joint_vel_runaway,
        params={
            "max_velocity": 50.0,
            "grace_steps": 25,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach policy env — 45d obs, distance reward, valve pos DR, handoff termination."""

    scene: ValveSceneCfg = ValveSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ReachObsCfg = ReachObsCfg()
    actions: ValveActionsCfg = ValveActionsCfg()
    rewards: ReachRewardsCfg = ReachRewardsCfg()
    terminations: ReachTerminationsCfg = ReachTerminationsCfg()
    events: ValveBaseEventCfg = ValveBaseEventCfg()
    curriculum = None
    commands = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.decimation = 4                          # 50 Hz policy
        self.sim.dt = 0.005                          # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0                 # reach timeout (CONTEXT.md §Timeouts)
        # Arms start at zero pose — policy must learn to reach from neutral.
        # ValveSceneCfg init_state has pre-grip arm pose (hands already near wheel);
        # reach training needs arms distant so the policy learns the full motion.
        for j in _G1_ARM_JOINTS:
            self.scene.robot.init_state.joint_pos[j] = 0.0

        # Disable valve pos DR for initial reach training — fixed valve = stable
        # optimization landscape, critic can learn non-constant V(s).
        # Re-enable (half_range 0.05) before sim2real.
        self.events.reset_valve_pos.params["half_range_xyz"] = (0.0, 0.0, 0.0)


@configclass
class ValveReachPlayEnvCfg(ValveReachEnvCfg):
    """Single-env play config for reach policy (dataset collection + inspection)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
