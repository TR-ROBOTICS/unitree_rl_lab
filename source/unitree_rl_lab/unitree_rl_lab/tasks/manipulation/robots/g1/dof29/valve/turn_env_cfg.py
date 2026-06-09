"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 1 (migrated).

Migrated from dof29/valve_turn_env_cfg.py into the valve/ subpackage.
The shim at dof29/valve_turn_env_cfg.py re-exports ValveTurnEnvCfg and
ValveTurnPlayEnvCfg from here, preserving existing gym registration paths.

RL spec ref: CONTEXT.md §RL spec
g(θ) firmware-locked — no DR.
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
    _G_THETA_A,
    _G_THETA_B,
    _P_SPAN,
    _P_MIN,
    _P_MAX,
    _THETA_MIN,
    _THETA_MAX,
    _EPS_SIM,
)

# Stage 1 fixed p_des. Central target (θ_des at midpoint of [θ_min, θ_max]) so v1's
# random-θ_init reset is direction-balanced: ~50/50 CW(decrease θ) vs CCW(increase θ),
# equal mean |Δθ| ≈ 10 rad. (Was 50 PSI → θ_des≈17.2 near θ_min → 81% CW / 19% CCW,
# imprinting a CW bias that degraded CCW quality downstream; see docs/model_eval_qual.md.)
_P_DES_STAGE1: float = 107.0  # PSI  (θ_des ≈ 29.7 rad ≈ midpoint of [9.42, 50.27])


# ---------------------------------------------------------------------------
# Observations — 30d (same as original v0; schema must NOT change per sim-isaaclab.md)
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
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"
            ])},
            clip=(-20.0, 20.0),
        )
        p_now_normalized = ObsTerm(
            func=mdp.valve_pressure_now,
            params={
                "pressure_a": _G_THETA_A,
                "pressure_b": _G_THETA_B,
                "p_min": _P_MIN,
                "p_max": _P_MAX,
                "p_span": _P_SPAN,
            },
            clip=(0.0, 1.0),
        )
        p_des_normalized = ObsTerm(
            func=mdp.valve_pressure_des,
            params={
                "p_des": _P_DES_STAGE1,
                "p_span": _P_SPAN,
            },
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — Stage 1 (identical to original v0)
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    pressure_error = RewTerm(
        func=mdp.pressure_error,
        weight=0.2,
        params={
            "p_des": _P_DES_STAGE1,
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_span": _P_SPAN,
        },
    )

    pressure_progress = RewTerm(
        func=mdp.pressure_progress,
        weight=30.0,
        params={
            "p_des": _P_DES_STAGE1,
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_span": _P_SPAN,
        },
    )
    # Smoothness penalties DISABLED Stage 1 — see sim-isaaclab.md §Run 5.


# ---------------------------------------------------------------------------
# Terminations — identical to original v0
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    pressure_success = DoneTerm(
        func=mdp.pressure_success_hold,
        time_out=False,
        params={
            "p_des": _P_DES_STAGE1,
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "eps_psi": _EPS_SIM,
            "hold_steps": 50,
        },
    )

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
# Events — re-use ValveBaseEventCfg (includes valve pos DR + angle reset + grip)
# Alias EventCfg so v1 can import it by that name (matching original v0 export).
# ---------------------------------------------------------------------------

@configclass
class EventCfg(ValveBaseEventCfg):
    """Turn-env events: inherits base (reset_all, valve DR, angle, grip).

    Alias name EventCfg preserves the import in valve_turn_env_cfg_v1.py shim.
    """
    pass


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfg(ManagerBasedRLEnvCfg):
    """G1-29DoF valve-turn env — Stage 1.

    g(θ) coefficients are explicit configclass fields (firmware-locked; no DR).
    """

    pressure_a: float = _G_THETA_A
    pressure_b: float = _G_THETA_B
    p_span: float = _P_SPAN
    p_min: float = _P_MIN
    p_max: float = _P_MAX
    eps_sim: float = _EPS_SIM
    theta_min: float = _THETA_MIN
    theta_max: float = _THETA_MAX
    p_des_range: tuple[float, float] = (_P_DES_STAGE1, _P_DES_STAGE1)
    hold_steps_required: int = 50
    contact_loss_steps_limit: int = 100

    scene: ValveSceneCfg = ValveSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ValveActionsCfg = ValveActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    curriculum = None
    commands = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.sim.physx.gpu_max_rigid_patch_count = 786432
        # Disable valve pos DR for bootstrapping — fixed valve = stable critic.
        # Re-enable (half_range 0.05) before sim2real.
        self.events.reset_valve_pos.params["half_range_xyz"] = (0.0, 0.0, 0.0)


@configclass
class ValveTurnPlayEnvCfg(ValveTurnEnvCfg):
    """Single-env play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        from .play_overrides import apply_play_viewer
        apply_play_viewer(self)
