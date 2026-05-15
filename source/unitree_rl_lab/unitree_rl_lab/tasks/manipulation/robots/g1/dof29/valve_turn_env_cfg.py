"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 1.

RL spec ref: CONTEXT.md §RL spec
Robot USD:   /home/jescobars/unitree_model/G1/29dof_inspire/g1_29dof_with_inspire_rev_1_0.usd
             58 CollisionAPI prims — full arm + Inspire finger coverage.
             Verified 2026-05-15 via IsaacSim Script Editor.
"""

from __future__ import annotations

import pathlib

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.actuators import ImplicitActuatorCfg

import unitree_rl_lab.tasks.manipulation.mdp as mdp

# ---------------------------------------------------------------------------
# Constants — g(θ) firmware mapping (firmware-locked; no DR on these values)
# CONTEXT.md §g(θ): p_now = a·θ + b, clamp [15, 200] PSI
# ---------------------------------------------------------------------------
_G_THETA_A: float = 4.527   # PSI/rad
_G_THETA_B: float = -27.66  # PSI
_P_SPAN: float = 185.0       # PSI  (200 − 15)
_P_MIN: float = 15.0         # PSI
_P_MAX: float = 200.0        # PSI
_THETA_MIN: float = 9.42     # rad  (1.5 rev)
_THETA_MAX: float = 50.27    # rad  (8 rev)
_EPS_SIM: float = 1.85       # PSI  (~1% of span)

# Stage 1 fixed p_des — mid-stroke target
_P_DES_STAGE1: float = 100.0  # PSI

# Absolute path to valve_rig.usd — resolved from assets package, no env-var dependency
_ASSETS_DIR = pathlib.Path(__file__).parents[5] / "assets"
_VALVE_RIG_USD: str = str(_ASSETS_DIR / "valve_rig.usd")

# G1 + Inspire hands USD — full collision geometry (58 colliders, verified 2026-05-15)
# Source: unitreerobotics/unitree_sim_isaaclab_usds HuggingFace assets
_INSPIRE_USD: str = "/home/jescobars/unitree_model/G1/29dof_inspire/g1_29dof_with_inspire_rev_1_0.usd"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class ValveTurnSceneCfg(InteractiveSceneCfg):
    """Scene: fixed-base G1 + valve rig at (0.6, 0, 1.2)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # G1 + Inspire hands robot — base fixed in ValveTurnEnvCfg.__post_init__
    # USD verified: full CollisionAPI on all arm + finger links (58 colliders).
    # enabled_self_collisions=False — prevents solver instability with hand collision active.
    # Actuator gains from unitree_sim_isaaclab robots/unitree.py (manipulation-tuned).
    # Hands actuator: high stiffness (1000) freezes fingers at pre-grip open pose.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_INSPIRE_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,   # increased: contact stability
                solver_velocity_iteration_count=2,   # increased: contact stability
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.793),
            joint_pos={
                # Legs — neutral standing
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                # Arms — pre-grip pose tuned in IsaacSim 2026-05-15 with Inspire USD
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
                # Inspire fingers — pre-grip pose (partial curl, tuned 2026-05-15)
                "L_index_proximal_joint":      0.6667,
                "L_index_intermediate_joint":  0.7278,
                "L_middle_proximal_joint":     0.7278,
                "L_middle_intermediate_joint": 1.1362,
                "L_pinky_proximal_joint":      0.6056,
                "L_pinky_intermediate_joint":  0.6266,
                "L_ring_proximal_joint":       0.6056,
                "L_ring_intermediate_joint":   1.0751,
                "L_thumb_proximal_yaw_joint":  1.0524,
                "L_thumb_proximal_pitch_joint":0.2496,
                "L_thumb_intermediate_joint":  0.3037,
                "L_thumb_distal_joint":        0.5149,
                "R_index_proximal_joint":      0.5044,
                "R_index_intermediate_joint":  1.0542,
                "R_middle_proximal_joint":     0.6266,
                "R_middle_intermediate_joint": 1.0943,
                "R_pinky_proximal_joint":      0.7889,
                "R_pinky_intermediate_joint":  0.5044,
                "R_ring_proximal_joint":       0.9111,
                "R_ring_intermediate_joint":   0.6877,
                "R_thumb_proximal_yaw_joint":  1.2990,  # clamped: limit is [-0.10, 1.30]
                "R_thumb_proximal_pitch_joint":0.3089,
                "R_thumb_intermediate_joint":  0.5917,
                "R_thumb_distal_joint":        0.6440,
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
            # Inspire fingers — high stiffness freezes at init pose (pre-grip curl).
            # Not included in action space (arm only). Stage 2: add grasp curriculum.
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_index_proximal_joint", ".*_index_intermediate_joint",
                    ".*_middle_proximal_joint", ".*_middle_intermediate_joint",
                    ".*_pinky_proximal_joint", ".*_pinky_intermediate_joint",
                    ".*_ring_proximal_joint", ".*_ring_intermediate_joint",
                    ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint",
                    ".*_thumb_intermediate_joint", ".*_thumb_distal_joint",
                ],
                effort_limit_sim=100.0,
                velocity_limit_sim=50.0,
                stiffness=1000.0,
                damping=15.0,
                armature=0.0,
            ),
        },
    )

    # Valve rig — separate articulation; handwheel RevoluteJoint is passive.
    # Stem axis along world X; +90° around Z aligns face toward robot.
    valve_rig: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UsdFileCfg(usd_path=_VALVE_RIG_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.90),
            rot=(0.707, 0.0, 0.0, 0.707),   # +90° around Z: stem → world X, face toward robot
            joint_pos={"RevoluteJoint": _THETA_MIN},
        ),
        actuators={},  # passive joint
    )

    # Stage 1: contact sensors omitted.
    # handwheel is an articulation link → lives in PhysX articulation view, not rigid body view.
    # ContactSensorCfg filter_prim_paths_expr queries rigid body view → always 0 matches.
    # Stage 2 fix: detect contact via net contact forces on wrist links directly (no filter).

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 1.0)),
    )


# ---------------------------------------------------------------------------
# Observations — arm joint pos + vel (28d)
# Full 45-d obs (+ pressure terms) added in Stage 2+.
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"
            ])},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"
            ])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Actions — 14 arm-joint Δ-targets, scale ±0.1 rad
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    arm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        scale=0.1,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Rewards — Stage 1: pressure error (dense) + smoothness + jerk
# Contact-loss / success bonus added in Stage 2+.
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # Dense: −|p_now − p_des| / p_span ∈ [−1, 0]
    pressure_error = RewTerm(
        func=mdp.pressure_error,
        weight=1.0,
        params={
            "p_des": _P_DES_STAGE1,
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_span": _P_SPAN,
        },
    )

    # Smoothness penalties disabled Stage 1 — cause NaN overflow when physics explodes
    # on contact. Re-enable Stage 3+ once wheel-turn behaviour is stable.
    # action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)
    # joint_vel   = RewTerm(func=base_mdp.joint_vel_l2,   weight=-0.001, ...)
    # joint_acc   = RewTerm(func=base_mdp.joint_acc_l2,   weight=-2.5e-7, ...)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfg(ManagerBasedRLEnvCfg):
    """G1-29DoF valve-turn env — Stage 1.

    g(θ) coefficients are explicit configclass fields (firmware-locked; no DR).
    """

    # -- g(θ) firmware coefficients ------------------------------------------
    pressure_a: float = _G_THETA_A  # PSI/rad
    pressure_b: float = _G_THETA_B  # PSI

    # -- pressure range -------------------------------------------------------
    p_span: float = _P_SPAN
    p_min: float = _P_MIN
    p_max: float = _P_MAX
    eps_sim: float = _EPS_SIM

    # -- operating envelope ---------------------------------------------------
    theta_min: float = _THETA_MIN
    theta_max: float = _THETA_MAX

    # -- Stage 1: fixed p_des -------------------------------------------------
    p_des_range: tuple[float, float] = (_P_DES_STAGE1, _P_DES_STAGE1)

    # -- success hold counter -------------------------------------------------
    hold_steps_required: int = 50   # K=10 × 5 steps/frame

    # -- contact-loss timeout -------------------------------------------------
    contact_loss_steps_limit: int = 100  # 2 s × 50 Hz

    # -- managers -------------------------------------------------------------
    scene: ValveTurnSceneCfg = ValveTurnSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    curriculum = None
    commands = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.sim.dt = 0.02           # 50 Hz policy rate
        self.sim.render_interval = 4
        self.decimation = 4          # physics at 200 Hz
        self.episode_length_s = 30.0

        # fix_root_link set in ArticulationRootPropertiesCfg directly (robot ArticulationCfg).


@configclass
class ValveTurnPlayEnvCfg(ValveTurnEnvCfg):
    """Single-env play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
