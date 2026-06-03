"""Shared play-mode overrides for valve-turn environments.

Usage — set VALVE_P_DES before running play.py to fix p_des for every episode reset:

    VALVE_P_DES=50  python scripts/rsl_rl/play.py --task Unitree-G1-29dof-ValveTurn-v4 ...
    VALVE_P_DES=107 python scripts/rsl_rl/play.py --task Unitree-G1-29dof-ValveTurn-v4 ...
    VALVE_P_DES=170 python scripts/rsl_rl/play.py --task Unitree-G1-29dof-ValveTurn-v4 ...

If VALVE_P_DES is not set: p_des remains random (as trained). Only affects v2+ envs
(v0/v1 have p_des baked into obs/reward params — not controllable via event override).

Applies to: ValveTurnPlayEnvCfgV2, V3, V4, V4A, V4AE.
"""

import os


def apply_play_p_des(events_cfg) -> None:
    """Override reset_p_des event params from VALVE_P_DES env var if set.

    Args:
        events_cfg: the env's events config object (self.events in __post_init__).
    """
    val = os.environ.get("VALVE_P_DES")
    if val is None:
        return
    p = float(val)
    events_cfg.reset_p_des.params["p_min"] = p
    events_cfg.reset_p_des.params["p_max"] = p
