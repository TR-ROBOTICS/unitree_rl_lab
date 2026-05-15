"""Valve-turn task config — G1 29-DoF, Stage 1.

RL spec ref: CONTEXT.md §RL spec
PRD ref:     docs/prd/IsaacLab-task.md
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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
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

    # G1 robot — fix_root_link applied in ValveTurnEnvCfg.__post_init__
    # Pre-grip pose: arms raised forward toward valve, elbows bent.
    # Shoulder pitch negative = forward raise in G1 convention.
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.793),
            joint_pos={
                # Legs — neutral standing
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                # Arms — pre-grip pose from IsaacSim manual IK (degrees→rad)
                ".*_shoulder_pitch_joint": -0.955,  # -54.7°
                "left_shoulder_roll_joint":  0.047, #  +2.7° (mirrored)
                "right_shoulder_roll_joint": -0.047,#  -2.7°
                ".*_elbow_joint":             1.251, #  71.7°
                "left_wrist_roll_joint":      0.583, #  +33.4° (mirrored)
                "right_wrist_roll_joint":    -0.583, #  -33.4°
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Valve rig — separate articulation; handwheel RevoluteJoint is passive.
    # Stem axis along world X; +90° around Z aligns face toward robot.
    valve_rig: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UsdFileCfg(usd_path=_VALVE_RIG_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.55),
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

    # Action rate penalty (from base_mdp — uses last_action buffer)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)

    # Joint velocity penalty — arm joints only
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        )},
    )

    # Joint acceleration penalty — arm joints only
    joint_acc = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        )},
    )


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
        self.scene.num_envs = 2048
        self.sim.dt = 0.02           # 50 Hz policy rate
        self.sim.render_interval = 4
        self.decimation = 4          # physics at 200 Hz
        self.episode_length_s = 30.0

        # Weld G1 base to world
        self.scene.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ValveTurnPlayEnvCfg(ValveTurnEnvCfg):
    """Single-env play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
