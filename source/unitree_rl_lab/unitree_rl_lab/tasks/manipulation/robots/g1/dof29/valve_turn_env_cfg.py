"""Valve-turn task config — G1 29-DoF, Stage 1 skeleton.

RL spec ref: CONTEXT.md §RL spec
PRD ref:     docs/prd/IsaacLab-task.md
"""

from __future__ import annotations

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path

import isaaclab.envs.mdp as base_mdp
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

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

# Absolute path to valve_rig.usd — resolved from assets package, no env-var dependency
_ASSETS_DIR = pathlib.Path(__file__).parents[5] / "assets"
_VALVE_RIG_USD: str = str(_ASSETS_DIR / "valve_rig.usd")


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class ValveTurnSceneCfg(InteractiveSceneCfg):
    """Scene: fixed-base G1 + valve rig at (0.6, 0, 1.2)."""

    # Ground plane — simple flat surface; no terrain generator needed
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # G1 robot — base welded to world via fix_base=True on the ArticulationRootProperties
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={},  # filled by pre-grasp event at reset
        ),
        actuators=ROBOT_CFG.actuators,
    )

    # Valve rig — separate articulation; handwheel RevoluteJoint is passive.
    # Stem axis along world X; pose per PRD §User Story 6.
    valve_rig: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UsdFileCfg(usd_path=_VALVE_RIG_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 1.2),
            joint_pos={"RevoluteJoint": _THETA_MIN},
        ),
    )

    # Wrist contact sensors — both wrists, filter on handwheel prim
    # Exact prim name confirmed after USD inspection (placeholder: .*wrist_yaw.*)
    wrist_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*wrist_yaw.*",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Valve/handwheel"],
    )

    # Undesired-contact sensor — all robot bodies
    body_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # Sky light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 1.0)),
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfg(ManagerBasedRLEnvCfg):
    """G1-29DoF valve-turn env — Stage 1 skeleton.

    g(θ) coefficients are explicit configclass fields (PRD §g(θ) coefficient placement).
    Change only here when firmware mapping updates.
    """

    # -- g(θ) firmware coefficients (no DR; firmware-locked) ---------------
    pressure_a: float = _G_THETA_A  # PSI/rad
    pressure_b: float = _G_THETA_B  # PSI

    # -- pressure range ------------------------------------------------------
    p_span: float = _P_SPAN         # PSI
    p_min: float = _P_MIN           # PSI
    p_max: float = _P_MAX           # PSI
    eps_sim: float = _EPS_SIM       # PSI  convergence tolerance

    # -- operating envelope --------------------------------------------------
    theta_min: float = _THETA_MIN   # rad
    theta_max: float = _THETA_MAX   # rad

    # -- target pressure range (Stage 1 = degenerate [100,100]) -------------
    p_des_range: tuple[float, float] = (100.0, 100.0)

    # -- success hold counter ------------------------------------------------
    # K=10 vision frames × 5 policy steps/frame = 50 policy steps (CONTEXT.md §Success criterion)
    hold_steps_required: int = 50

    # -- contact-loss timeout ------------------------------------------------
    # 2 s × 50 Hz = 100 policy steps (PRD §Terminations)
    contact_loss_steps_limit: int = 100

    # -- scene ---------------------------------------------------------------
    scene: ValveTurnSceneCfg = ValveTurnSceneCfg(num_envs=4096, env_spacing=2.5)

    # -- sim -----------------------------------------------------------------
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.02           # 50 Hz policy rate
        self.sim.render_interval = 4
        self.decimation = 4          # control at 50 Hz, physics at 200 Hz
        self.episode_length_s = 30.0
