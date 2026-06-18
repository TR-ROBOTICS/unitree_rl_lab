"""Parameterized valve-turn env config + flat preset catalog (ADR 0009 item 5).

Replaces the v0→v6b subclass ladder (one file + subclass + shim per experiment)
with ONE parameterized config (`TurnEnvCfg`) and a data catalog (`TURN_PRESETS`).
A turn experiment is now a `TurnSpec` row, not a new module.

The interface is the set of axes that actually vary across experiments:
  - theta_init / p_des reset ranges
  - arm-init mode        (pregrip | dataset | staged | mixed)
  - reward profile       (smoothness, contact, bimanual flags)
  - curriculum           (None | auto | auto_easy | smooth_v5 | pd_v6 | pd_v7)
  - action/obs space     (arm-only 14-DoF | arm+hands 38-DoF)

Everything stable (scene, actuators, g(θ) pressure terms, terminations, sim
config) lives once. Gym IDs are unchanged — registered against the per-preset
factory callables at the bottom of this module.

Behavior parity vs the old ladder is by construction (same numbers, same event
exec-order: reset_all first; finger-grip + arm-init after). NOT import-verified
locally (IsaacLab needs the Sim app); requires `play.py` parity on every task —
see docs/adr/0009.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import (
    ValveActionsCfg,
    ValveBaseEventCfg,
    ValveSceneCfg,
    _G_THETA_A,
    _G_THETA_B,
    _P_SPAN,
    _P_MIN,
    _P_MAX,
    _EPS_SIM,
    _THETA_MIN,
    _THETA_MAX,
    _HUB_BODY_NAME,
    _LEFT_HAND_BODIES,
    _RIGHT_HAND_BODIES,
    _WHEEL_RADIUS,
    _WHEEL_NORMAL_WORLD,
    _WHEEL_PLANE_OFFSET,
    _INSPIRE_USD_PALMCAPS,
    _INSPIRE_USD_NOCAPS,
)
import pathlib

# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
_P_MID: float = 107.0           # central p_des → θ_des≈29.7 mid-range (direction-balanced)
_THETA_STEP: float = 2.04       # 5% of θ-span (40.85 rad) — curriculum start window
_P_STEP: float = 9.25           # 5% of p-span (185 PSI)
_THETA_MID: float = 29.75       # θ at p_des=107

_ARM_JOINTS: list[str] = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
_FINGER_JOINT_NAMES: list[str] = [
    ".*_index_proximal_joint",     ".*_index_intermediate_joint",
    ".*_middle_proximal_joint",    ".*_middle_intermediate_joint",
    ".*_pinky_proximal_joint",     ".*_pinky_intermediate_joint",
    ".*_ring_proximal_joint",      ".*_ring_intermediate_joint",
    ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint",
    ".*_thumb_intermediate_joint", ".*_thumb_distal_joint",
]

# Single source — was copy-pasted across v3/v4ae/v5/v6.
DATASET_PATH: str = str(
    pathlib.Path(__file__).parents[6] / "datasets" / "reach_arm_positions.npy"
)
PREGRASP_ARM_POSE: dict[str, float] = {
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
}


# ---------------------------------------------------------------------------
# Actions — arm-only (shared) and arm+hands (38-DoF)
# ---------------------------------------------------------------------------

@configclass
class ValveActionsCfgHands(ValveActionsCfg):
    """Arm Δ-targets (14, scale=0.1) + hand Δ-targets (24, scale=0.05)."""

    hands = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FINGER_JOINT_NAMES,
        scale=0.05,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations — base (arm) and +finger (hands). Pressure terms read p_des_buf.
# ---------------------------------------------------------------------------

@configclass
class _PolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=base_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        clip=(-10.0, 10.0),
    )
    joint_vel = ObsTerm(
        func=base_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        clip=(-20.0, 20.0),
    )
    # Default: fresh ground-truth p_now every step.
    # Replaced by valve_pressure_now_zoh when VisionDRCfg.enabled=True
    # (see _apply_vision_dr — the ObsTerm is swapped in-place after construction).
    p_now_normalized = ObsTerm(func=mdp.valve_pressure_now, clip=(0.0, 1.0))
    p_des_normalized = ObsTerm(func=mdp.valve_pressure_des, clip=(0.0, 1.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class _PolicyObsHandsCfg(_PolicyObsCfg):
    finger_joint_pos_rel = ObsTerm(
        func=base_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_FINGER_JOINT_NAMES)},
    )


@configclass
class TurnObsCfg:
    policy: _PolicyObsCfg = _PolicyObsCfg()


@configclass
class TurnObsHandsCfg:
    policy: _PolicyObsHandsCfg = _PolicyObsHandsCfg()


# ---------------------------------------------------------------------------
# Rewards — pressure terms always on; optional terms None until set by builder
# ---------------------------------------------------------------------------

@configclass
class TurnRewardsCfg:
    pressure_error = RewTerm(func=mdp.pressure_error, weight=0.2)
    pressure_progress = RewTerm(func=mdp.pressure_progress, weight=30.0)
    action_rate: RewTerm | None = None
    bilateral_contact: RewTerm | None = None
    contact_force_jerk: RewTerm | None = None
    bimanual_progress: RewTerm | None = None
    single_hand_turning_penalty: RewTerm | None = None


def _bimanual_geom_params() -> dict:
    return {
        "valve_hub_cfg":      SceneEntityCfg("valve_rig", body_names=[_HUB_BODY_NAME]),
        "left_hand_cfg":      SceneEntityCfg("robot", body_names=_LEFT_HAND_BODIES),
        "right_hand_cfg":     SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES),
        "wheel_radius":       _WHEEL_RADIUS,
        "wheel_normal_world": _WHEEL_NORMAL_WORLD,
        "plane_offset":       _WHEEL_PLANE_OFFSET,
        "sigma":              0.15,
    }


def build_rewards(spec: "TurnSpec") -> TurnRewardsCfg:
    r = TurnRewardsCfg()
    if spec.smoothness:
        r.action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0001)
    if spec.contact_rewards:
        r.bilateral_contact = RewTerm(
            func=mdp.bilateral_contact,
            weight=0.0,  # any positive weight → touch-only optimum (ADR 0007 run log)
            params={"left_sensor_name": "left_palm_sensor",
                    "right_sensor_name": "right_palm_sensor", "f_max": 50.0},
        )
        r.contact_force_jerk = RewTerm(
            func=mdp.contact_force_jerk,
            weight=0.0,
            params={"left_sensor_name": "left_palm_sensor",
                    "right_sensor_name": "right_palm_sensor"},
        )
    if spec.bimanual_rewards:
        r.bimanual_progress = RewTerm(
            func=mdp.bimanual_progress_reward, weight=30.0, params=_bimanual_geom_params()
        )
        r.single_hand_turning_penalty = RewTerm(
            func=mdp.single_hand_turning_penalty, weight=-15.0, params=_bimanual_geom_params()
        )
    return r


# ---------------------------------------------------------------------------
# Terminations — identical for every turn version
# ---------------------------------------------------------------------------

@configclass
class TurnTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    pressure_success = DoneTerm(
        func=mdp.pressure_success_hold, time_out=False, params={"hold_steps": 50}
    )
    joint_vel_explosion = DoneTerm(
        func=mdp.joint_vel_runaway,
        params={"max_velocity": 50.0, "grace_steps": 25,
                "asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
    )


# ---------------------------------------------------------------------------
# Events — base (reset_all, valve DR, valve angle, finger grip) + p_des + arm.
# Field order = exec order: reset_all first; finger-grip then reset_arm after.
# ---------------------------------------------------------------------------

@configclass
class TurnEventCfg(ValveBaseEventCfg):
    reset_p_des: EventTerm = EventTerm(
        func=mdp.reset_p_des_random, mode="reset", params={"p_min": _P_MID, "p_max": _P_MID}
    )
    reset_arm: EventTerm | None = None


def _valve_angle_term(theta: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.reset_valve_to_random_angle,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("valve_rig"),
                "angle_min": theta[0], "angle_max": theta[1]},
    )


def _p_des_term(p_des: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.reset_p_des_random, mode="reset",
        params={"p_min": p_des[0], "p_max": p_des[1]},
    )


def _arm_term(mode: str) -> EventTerm | None:
    if mode == "pregrip":
        return None
    func = {
        "dataset": mdp.reset_arm_from_dataset,
        "staged":  mdp.reset_arm_staged,
        "mixed":   mdp.reset_arm_mixed,
    }[mode]
    return EventTerm(
        func=func, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS),
                "dataset_path": DATASET_PATH, "fallback_pose": PREGRASP_ARM_POSE},
    )


def _pregrasp_arm_term(enabled: bool) -> EventTerm:
    """EventTerm that writes PREGRASP_ARM_POSE to arm joints every reset.

    When enabled=True (v7 default): explicitly resets arm to the canonical
    pre-grasp pose near the valve rim, overriding whatever USD init_state
    or reset_scene_to_default would produce.  This is the v7 Feature 1.

    When enabled=False (ablation): the EventTerm is still present but is
    a no-op (reset_arm_pregrasp checks enabled param before writing state).
    This allows ablation without changing the event graph structure.

    Exec order: runs AFTER reset_finger_grip (field declared after it in
    TurnEventCfg).  reset_scene_to_default runs first (always field 0).
    """
    return EventTerm(
        func=mdp.reset_arm_pregrasp,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]),
            "pregrasp_pose": PREGRASP_ARM_POSE,
            "enabled": enabled,
        },
    )


def _build_events(
    theta: tuple[float, float],
    p_des: tuple[float, float],
    arm: str,
    pregrasp_init: bool = False,
) -> TurnEventCfg:
    ev = TurnEventCfg()
    ev.reset_valve_angle = _valve_angle_term(theta)   # override base fixed-pose term
    ev.reset_p_des = _p_des_term(p_des)
    ev.reset_arm = _arm_term(arm)
    # Valve pos DR disabled for bootstrapping (was ValveTurnEnvCfg.__post_init__).
    ev.reset_valve_pos.params["half_range_xyz"] = (0.0, 0.0, 0.0)
    # v7 pre-grasp init: always add the term (enabled flag controls no-op).
    # When pregrasp_init=False the EventTerm is absent (preserves v0-v6 behaviour).
    if pregrasp_init:
        ev.reset_arm_pregrasp = _pregrasp_arm_term(enabled=True)
    return ev


# ---------------------------------------------------------------------------
# Curriculum profiles
# ---------------------------------------------------------------------------

@configclass
class _CurrAuto:
    turn_auto_stage = CurrTerm(
        func=mdp.turn_auto_curriculum_stage,
        params={"success_threshold": 0.85, "window_iters": 100, "num_steps_per_env": 24,
                "theta_max": _THETA_MAX, "p_min_target": _P_MIN, "p_max_target": _P_MAX},
    )


@configclass
class _CurrAutoEasy:
    turn_auto_stage = CurrTerm(
        func=mdp.turn_auto_curriculum_stage_easy,
        params={"success_threshold": 0.85, "window_iters": 100, "num_steps_per_env": 24,
                "theta_max": _THETA_MAX, "p_min_target": _P_MIN, "p_max_target": _P_MAX},
    )


@configclass
class _CurrSmoothV5:
    v5_stage = CurrTerm(
        func=mdp.turn_smooth_curriculum_v5,
        params={"success_threshold": 0.85, "window_iters": 10, "num_steps_per_env": 24,
                "theta_min": _THETA_MIN, "theta_max": _THETA_MAX, "theta_mid": _THETA_MID,
                "theta_step": _THETA_STEP, "p_min": _P_MIN, "p_max": _P_MAX, "p_mid": _P_MID,
                "p_step": _P_STEP, "dataset_step": 0.10,
                "theta_start_lo": _THETA_MIN, "theta_start_hi": _THETA_MIN + _THETA_STEP,
                "p_start": 107.0},
    )


@configclass
class _CurrPDV6:
    stage = CurrTerm(
        func=mdp.turn_pd_curriculum_v6,
        params={"beta": 0.02, "sr_target": 0.85, "kp": 2.0, "kd": 0.5,
                "theta_scale": 1.0, "p_scale": 4.625, "mix_scale": 0.005,
                "confirm_iters": 20, "num_steps_per_env": 24,
                "theta_min": _THETA_MIN, "theta_max": _THETA_MAX,
                "theta_start_hi": _THETA_MIN + _THETA_STEP,
                "p_min": _P_MIN, "p_max": _P_MAX, "p_mid": _P_MID},
    )


@configclass
class _CurrPDV7:
    """v7 independent-axis PD curriculum (ADR 0012).

    Two fully independent EMA+PD controllers — one per axis (θ and p).
    No dataset-mixing stage: v7 keeps pre-grasp arm init throughout.

    Stage 0: θ expansion only (p fixed at p_mid=107 PSI).
    Stage 1: p expansion only (θ at full range).
    Stage 2: terminal — both axes fully open, pre-grasp init throughout.

    Per-axis knobs exposed separately so θ and p can be tuned independently.
    All logging keys use v7_ prefix to avoid TensorBoard collision with v5/v6.
    """

    stage = CurrTerm(
        func=mdp.turn_pd_curriculum_v7,
        params={
            # θ-axis PD knobs
            "kp_theta": 2.0,
            "kd_theta": 0.5,
            "beta_theta": 0.02,
            "theta_scale": 1.0,
            # p-axis PD knobs
            "kp_p": 2.0,
            "kd_p": 0.5,
            "beta_p": 0.02,
            "p_scale": 4.625,
            # shared
            "sr_target": 0.85,
            "confirm_iters": 20,
            "num_steps_per_env": 24,
            # envelope
            "theta_min": _THETA_MIN,
            "theta_max": _THETA_MAX,
            "theta_start_hi": _THETA_MIN + _THETA_STEP,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_mid": _P_MID,
        },
    )


_CURRICULA = {
    "auto":      _CurrAuto,
    "auto_easy": _CurrAutoEasy,
    "smooth_v5": _CurrSmoothV5,
    "pd_v6":     _CurrPDV6,
    "pd_v7":     _CurrPDV7,
}


# ---------------------------------------------------------------------------
# Parameterized env config
# ---------------------------------------------------------------------------

@configclass
class TurnEnvCfg(ManagerBasedRLEnvCfg):
    """G1-29DoF valve-turn env. Built per preset by `build_env_cfg`.

    The g(θ) fields below are read by scripts/rsl_rl/probe_reset_direction.py;
    kept verbatim (firmware-locked, no DR — single source is mdp/pressure.py).
    """

    pressure_a: float = _G_THETA_A
    pressure_b: float = _G_THETA_B
    p_span: float = _P_SPAN
    p_min: float = _P_MIN
    p_max: float = _P_MAX
    eps_sim: float = _EPS_SIM
    theta_min: float = _THETA_MIN
    theta_max: float = _THETA_MAX
    p_des_range: tuple[float, float] = (_P_MID, _P_MID)
    hold_steps_required: int = 50
    contact_loss_steps_limit: int = 100

    scene: ValveSceneCfg = ValveSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TurnObsCfg = TurnObsCfg()
    actions: ValveActionsCfg = ValveActionsCfg()
    rewards: TurnRewardsCfg = TurnRewardsCfg()
    terminations: TurnTerminationsCfg = TurnTerminationsCfg()
    events: TurnEventCfg = TurnEventCfg()

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


# ---------------------------------------------------------------------------
# Preset catalog — one row per turn experiment (replaces the subclass ladder)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaySpec:
    """Play-mode distribution + overlays for a preset's checkpoints."""
    events: str = "train"          # "train" (use training θ/p/arm) | "terminal" (full θ/p, dataset arm)
    drop_curriculum: bool = False  # bypass curriculum at play
    apply_viewer: bool = True      # close-up viewer camera
    apply_p_des: bool = True        # honor VALVE_P_DES env override


@dataclass(frozen=True)
class VisionDRCfg:
    """Sparse-feedback vision model for p_now obs (E3 compatibility sweep).

    Models vision delivering p_now at low Hz instead of fresh every 50 Hz policy
    step.  Implements: ZOH at vision Hz + additive Gaussian noise + latency pipe
    + optional quantization.

    Default OFF (enabled=False) — when disabled, obs[28] is exactly
    ``valve_pressure_now`` (ground-truth p_now from g(θ)), and no extra event or
    buffer is created.  Existing runs and the R4 probe are byte-identical.

    Knobs:
        enabled:         Master switch.  False = standard valve_pressure_now.
        vision_hz:       (hz_lo, hz_hi) — per-env vision update rate sampled
                         uniformly.  Use (hz, hz) for a fixed rate.
                         E.g. (10.0, 10.0) for 10 Hz (policy at 50 Hz → interval=5).
        noise_psi:       σ of additive Gaussian noise on each fresh read (PSI).
                         0.0 = no noise.
        latency_steps:   ZOH lag in policy steps (delivered value lags fresh read
                         by this many steps).  0 = no latency.
        quant_psi:       Quantization step (PSI).  0.0 = off.

    Example — 10 Hz, 2 PSI noise, 2-step lag, no quantization:
        vision_dr=VisionDRCfg(enabled=True, vision_hz=(10.0, 10.0),
                              noise_psi=2.0, latency_steps=2)
    """
    enabled: bool = False
    vision_hz: tuple[float, float] = (10.0, 10.0)
    noise_psi: float = 0.0
    latency_steps: int = 0
    quant_psi: float = 0.0


@dataclass(frozen=True)
class TurnSpec:
    theta_init: tuple[float, float]
    p_des: tuple[float, float]
    arm_init: str                    # pregrip | dataset | staged | mixed
    smoothness: bool = False         # action_rate_l2 = -0.0001
    contact_rewards: bool = False    # bilateral_contact(0.0) + contact_force_jerk(0.0)
    bimanual_rewards: bool = False   # bimanual_progress(30.0) + single_hand_turning_penalty(-15.0)
    curriculum: str | None = None    # auto | auto_easy | smooth_v5 | pd_v6 | pd_v7
    hands: bool = False              # 38-DoF action + finger obs
    pregrasp_init: bool = False      # v7: explicit pre-grasp arm reset every episode
    inspire_usd: str | None = None   # override robot USD (capsule-ablation variants); None = base full-caps
    vision_dr: VisionDRCfg = field(default_factory=VisionDRCfg)  # default OFF
    play: PlaySpec = field(default_factory=PlaySpec)


_FIX_T = (_THETA_MIN, _THETA_MIN)
_FULL_T = (_THETA_MIN, _THETA_MAX)
_NARROW_T = (_THETA_MIN, _THETA_MIN + _THETA_STEP)
_P107 = (_P_MID, _P_MID)
_FULL_P = (_P_MIN, _P_MAX)
_TERMINAL_PLAY = PlaySpec(events="terminal", drop_curriculum=True)

TURN_PRESETS: dict[str, TurnSpec] = {
    # id suffix : spec
    "v0":    TurnSpec(_FIX_T,    _P107,   "pregrip", play=PlaySpec(apply_p_des=False)),
    "v1":    TurnSpec(_FULL_T,   _P107,   "pregrip", play=PlaySpec(apply_p_des=False)),
    "v2":    TurnSpec(_FULL_T,   _FULL_P, "pregrip"),
    "v3":    TurnSpec(_FULL_T,   _FULL_P, "dataset"),
    "v4":    TurnSpec(_FULL_T,   _FULL_P, "dataset", smoothness=True),
    "v4a":   TurnSpec(_FIX_T,    _P107,   "dataset", smoothness=True, curriculum="auto",
                      play=_TERMINAL_PLAY),
    "v4ae":  TurnSpec(_FIX_T,    _P107,   "staged",  smoothness=True, curriculum="auto_easy",
                      play=_TERMINAL_PLAY),
    "v4ah":  TurnSpec(_FIX_T,    _P107,   "dataset", smoothness=True, curriculum="auto",
                      hands=True, play=PlaySpec(apply_viewer=False, apply_p_des=False)),
    "v4aeh": TurnSpec(_FIX_T,    _P107,   "staged",  smoothness=True, curriculum="auto_easy",
                      hands=True, play=PlaySpec(apply_viewer=False, apply_p_des=False)),
    "v5":    TurnSpec(_NARROW_T, _P107,   "mixed",   smoothness=True, contact_rewards=True,
                      curriculum="smooth_v5", play=_TERMINAL_PLAY),
    "v6":    TurnSpec(_NARROW_T, _P107,   "mixed",   smoothness=True, curriculum="pd_v6",
                      play=_TERMINAL_PLAY),
    "v6b":   TurnSpec(_FULL_T,   _FULL_P, "dataset", smoothness=True, bimanual_rewards=True,
                      play=_TERMINAL_PLAY),
    # v7: Option-X retrain — pre-grasp arm init (fixed, no dataset mixing) +
    #     independent per-axis PD curriculum (θ axis then p axis, separate EMA+PD).
    #     Arms start at PREGRASP_ARM_POSE every episode; this is the main hypothesis:
    #     fixed bilateral pre-grip reduces exploration burden vs reach-dataset init.
    #     Toggle pregrasp_init=False to ablate (reverts to USD init_state arm pose).
    #     Ref: CONTEXT.md mission "retrain from fixed bilateral pre-grip pose".
    "v7":    TurnSpec(_NARROW_T, _P107,   "pregrip", smoothness=True, curriculum="pd_v7",
                      pregrasp_init=True, play=_TERMINAL_PLAY),
    # Arm-capsule ablation (sim2sim-diff isolation). Identical to v7 except robot USD:
    #   v7_palmcaps — palm sphere only (drop wrist/elbow/shoulder capsules)
    #   v7_nocaps   — no arm colliders at all
    # Both keep finger CCD + enabledSelfCollisions=True. Control = existing v7 (full caps).
    # Train seed 42, matched budget (~3K) for paired comparison vs v7a@2700.
    "v7_palmcaps": TurnSpec(_NARROW_T, _P107, "pregrip", smoothness=True, curriculum="pd_v7",
                            pregrasp_init=True, inspire_usd=_INSPIRE_USD_PALMCAPS,
                            play=_TERMINAL_PLAY),
    "v7_nocaps":   TurnSpec(_NARROW_T, _P107, "pregrip", smoothness=True, curriculum="pd_v7",
                            pregrasp_init=True, inspire_usd=_INSPIRE_USD_NOCAPS,
                            play=_TERMINAL_PLAY),
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _apply_vision_dr(cfg: TurnEnvCfg, spec: TurnSpec) -> None:
    """Wire VisionDRCfg into obs + events when enabled; no-op when disabled.

    When enabled:
      - Replaces the p_now_normalized ObsTerm in the policy obs group with
        valve_pressure_now_zoh, passing noise_psi / latency_steps / quant_psi.
      - Appends a reset_vision_zoh_state EventTerm (mode="reset") that fires AFTER
        reset_valve_angle so the seed p_now reflects the new θ_init.

    When disabled (default): zero changes — obs and events are untouched.
    Existing runs and the R4 probe are byte-identical.
    """
    vdr = spec.vision_dr
    if not vdr.enabled:
        return  # default OFF — nothing to do

    # Swap p_now_normalized to ZOH variant with knob params
    zoh_term = ObsTerm(
        func=mdp.valve_pressure_now_zoh,
        params={
            "noise_psi":     vdr.noise_psi,
            "latency_steps": vdr.latency_steps,
            "quant_psi":     vdr.quant_psi,
        },
        clip=(0.0, 1.0),
    )
    cfg.observations.policy.p_now_normalized = zoh_term

    # Append reset event — must execute after reset_valve_angle so seed p_now is valid.
    # EventTermCfg field name = "reset_vision_zoh" (does not clash with existing fields).
    cfg.events.reset_vision_zoh = EventTerm(
        func=mdp.reset_vision_zoh_state,
        mode="reset",
        params={
            "vision_hz_range": vdr.vision_hz,
            "latency_steps":   vdr.latency_steps,
        },
    )


def _apply_common(cfg: TurnEnvCfg, spec: TurnSpec) -> None:
    cfg.observations = TurnObsHandsCfg() if spec.hands else TurnObsCfg()
    cfg.actions = ValveActionsCfgHands() if spec.hands else ValveActionsCfg()
    cfg.rewards = build_rewards(spec)
    cfg.terminations = TurnTerminationsCfg()


def build_env_cfg(name: str) -> TurnEnvCfg:
    """Training config for a preset id (e.g. "v7")."""
    spec = TURN_PRESETS[name]
    cfg = TurnEnvCfg()
    _apply_common(cfg, spec)
    if spec.inspire_usd is not None:
        cfg.scene.robot.spawn.usd_path = spec.inspire_usd
    cfg.events = _build_events(spec.theta_init, spec.p_des, spec.arm_init,
                               pregrasp_init=spec.pregrasp_init)
    cfg.curriculum = _CURRICULA[spec.curriculum]() if spec.curriculum else None
    # Vision DR wiring (no-op when spec.vision_dr.enabled=False — default).
    _apply_vision_dr(cfg, spec)
    return cfg


def build_play_cfg(name: str) -> TurnEnvCfg:
    """Single-env play config for a preset id."""
    spec = TURN_PRESETS[name]
    cfg = TurnEnvCfg()
    _apply_common(cfg, spec)
    if spec.inspire_usd is not None:
        cfg.scene.robot.spawn.usd_path = spec.inspire_usd

    if spec.play.events == "terminal":
        # Stage-terminal distribution: full θ/p range, 100% dataset arm init.
        # For v7 (pregrasp_init=True), keep pre-grasp arm init at play time too —
        # this mirrors train distribution (no dataset mixing in v7).
        cfg.events = _build_events(_FULL_T, _FULL_P,
                                   "pregrip" if spec.pregrasp_init else "dataset",
                                   pregrasp_init=spec.pregrasp_init)
    else:
        cfg.events = _build_events(spec.theta_init, spec.p_des, spec.arm_init,
                                   pregrasp_init=spec.pregrasp_init)

    cfg.curriculum = None if spec.play.drop_curriculum else (
        _CURRICULA[spec.curriculum]() if spec.curriculum else None
    )

    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    from .play_overrides import apply_play_p_des, apply_play_viewer
    if spec.play.apply_viewer:
        apply_play_viewer(cfg)
    if spec.play.apply_p_des:
        apply_play_p_des(cfg.events)
    # Vision DR wiring (no-op when spec.vision_dr.enabled=False — default).
    _apply_vision_dr(cfg, spec)
    return cfg


# Per-preset zero-arg factory callables — gym entry_point strings resolve to these.
# (e.g. "...presets:turn_v6" / "...presets:turn_v6_play")
for _name in TURN_PRESETS:
    globals()[f"turn_{_name}"] = partial(build_env_cfg, _name)
    globals()[f"turn_{_name}_play"] = partial(build_play_cfg, _name)
