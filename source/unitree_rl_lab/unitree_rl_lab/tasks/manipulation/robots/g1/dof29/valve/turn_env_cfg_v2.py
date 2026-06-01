"""Valve-turn task config — G1 29-DoF + Inspire hands, Stage 2 v2.

v2 = v1 + random p_des per episode.
  - θ_init: uniform [θ_min, θ_max]       (inherited from v1)
  - p_des:  uniform [p_min, p_max] PSI   (new — from env.p_des_buf)
  - Arm init: pre-grip pose              (inherited)
  - Valve DR: disabled                   (inherited)
  - hold_steps: 50                       (inherited)

Resume from v1 model_300 — policy knows bidirectional turning,
now must generalise to arbitrary pressure targets.

g(θ) firmware-locked — no DR.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.manipulation.mdp as mdp

from .base_cfg import (
    _G_THETA_A, _G_THETA_B,
    _P_SPAN, _P_MIN, _P_MAX,
    _THETA_MIN, _THETA_MAX,
    _EPS_SIM,
)
from .turn_env_cfg_v1 import EventCfgV1, ValveTurnEnvCfgV1
from .turn_env_cfg import RewardsCfg, ObservationsCfg, TerminationsCfg


# ---------------------------------------------------------------------------
# Events — v1 + reset p_des each episode
# ---------------------------------------------------------------------------

@configclass
class EventCfgV2(EventCfgV1):
    """v1 events + random p_des sampling."""

    reset_p_des = EventTerm(
        func=mdp.reset_p_des_random,
        mode="reset",
        params={
            "p_min": _P_MIN,
            "p_max": _P_MAX,
        },
    )


# ---------------------------------------------------------------------------
# Observations — swap constant p_des for per-env buffer read
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfgV2(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        p_des_normalized = ObsTerm(
            func=mdp.valve_pressure_des_random,
            params={"p_span": _P_SPAN},
            clip=(0.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — swap scalar p_des for per-env buffer variants
# ---------------------------------------------------------------------------

@configclass
class RewardsCfgV2(RewardsCfg):
    pressure_error = RewTerm(
        func=mdp.pressure_error_random,
        weight=0.2,
        params={
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_span": _P_SPAN,
        },
    )

    pressure_progress = RewTerm(
        func=mdp.pressure_progress_random,
        weight=30.0,
        params={
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "p_span": _P_SPAN,
        },
    )


# ---------------------------------------------------------------------------
# Terminations — swap scalar p_des for per-env buffer variant
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfgV2(TerminationsCfg):
    pressure_success = DoneTerm(
        func=mdp.pressure_success_hold_random,
        time_out=False,
        params={
            "pressure_a": _G_THETA_A,
            "pressure_b": _G_THETA_B,
            "p_min": _P_MIN,
            "p_max": _P_MAX,
            "eps_psi": _EPS_SIM,
            "hold_steps": 50,
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class ValveTurnEnvCfgV2(ValveTurnEnvCfgV1):
    """Stage 2 v2: random θ_init + random p_des, all else v1."""

    observations: ObservationsCfgV2 = ObservationsCfgV2()
    rewards:      RewardsCfgV2      = RewardsCfgV2()
    terminations: TerminationsCfgV2 = TerminationsCfgV2()
    events:       EventCfgV2        = EventCfgV2()

    def __post_init__(self):
        super().__post_init__()  # inherits DR disable + physics config


@configclass
class ValveTurnPlayEnvCfgV2(ValveTurnEnvCfgV2):
    """Single-env play config for Stage 2 v2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
